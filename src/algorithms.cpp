#include "algorithms.h"
#include <algorithm>
#include <cmath>
#include <set>

Vec3 lbs_vertex(const Vec3& pos, const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    double x = 0.0, y = 0.0, z = 0.0;
    for (const auto& kv : weights) {
        const RigidTransform& C = Cs[kv.first];
        Vec3 p = apply_transform(C, pos);
        x += p.x * kv.second;
        y += p.y * kv.second;
        z += p.z * kv.second;
    }
    return { x, y, z };
}

namespace {

    inline std::vector<std::pair<unsigned int, float>>
        mapToSortedVec(const WeightMap& weights) {
        std::vector<std::pair<unsigned int, float>> v;
        v.reserve(weights.size());
        for (const auto& kv : weights) {
            if (kv.second > 1e-12) v.push_back({ kv.first, (float)kv.second });
        }
        std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.second > b.second; });
        return v;
    }

} // anonymous namespace

// ================================================================
// v2 接口
// ================================================================

Quat qlerp_v2(const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs) {
    if (weights.empty()) return { 0, 0, 0, 1 };
    Quat ref_q = Cs[weights[0].first].q;
    Quat q_sum = { 0.0, 0.0, 0.0, 0.0 };
    for (const auto& kv : weights) {
        Quat q = Cs[kv.first].q;
        if (q_dot(q, ref_q) < 0.0) q = q_scale(q, -1.0);
        q_sum = q_add(q_sum, q_scale(q, (double)kv.second));
    }
    return q_normalize(q_sum);
}

Vec3 cor_vertex_v2(const Vec3& pos, const Vec3& cor,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs) {
    if (weights.empty()) return pos;
    Quat q_p = qlerp_v2(weights, Cs);
    Vec3 cor_runtime = { 0.0, 0.0, 0.0 };
    for (const auto& kv : weights) {
        Vec3 t = apply_transform(Cs[kv.first], cor);
        cor_runtime = v3_add(cor_runtime, v3_scale(t, (double)kv.second));
    }
    Vec3 cor_rotated = q_rotate_v3(q_p, cor);
    Vec3 t_p = v3_sub(cor_runtime, cor_rotated);
    return v3_add(q_rotate_v3(q_p, pos), t_p);
}

Vec3 dlb_vertex_v2(const Vec3& pos,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs) {
    if (weights.empty()) return pos;
    Quat ref_qr = Cs[weights[0].first].q;
    Quat q_r_sum = { 0.0, 0.0, 0.0, 0.0 };
    Quat q_d_sum = { 0.0, 0.0, 0.0, 0.0 };
    for (const auto& kv : weights) {
        DualQuat dq = dual_quat_from_C(Cs[kv.first]);
        Quat q_r = dq.real, q_d = dq.dual;
        if (q_dot(q_r, ref_qr) < 0.0) {
            q_r = q_scale(q_r, -1.0);
            q_d = q_scale(q_d, -1.0);
        }
        double w = (double)kv.second;
        q_r_sum = q_add(q_r_sum, q_scale(q_r, w));
        q_d_sum = q_add(q_d_sum, q_scale(q_d, w));
    }
    double n = q_norm(q_r_sum);
    if (n < 1e-8) return pos;
    return dlb_transform(q_scale(q_r_sum, 1.0 / n),
        q_scale(q_d_sum, 1.0 / n), pos);
}

// ================================================================
// DQS 补偿 v2 —— 对称化 f + 关节中心 r（你之前说"对称了"的版本）
//   · bisector = -normalize(dir1 + dir2) + hierarchy 反转（Mark 版）
//   · f(w) = f(min(w, 1-w))  对称化
//   · r = 顶点到关节中心距离
//   · k3 = 1 - W3/W2 + clamp
// ================================================================
Vec3 dqs_compensate_vertex_v2(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    const std::vector<int>& depths,
    double strength) {

    if (weights.size() < 2) return dqs_pos;

    unsigned int I1 = weights[0].first;  double W1 = (double)weights[0].second;
    unsigned int I2 = weights[1].first;  double W2 = (double)weights[1].second;
    double W3 = (weights.size() > 2) ? (double)weights[2].second : 0.0;

    if (W1 + W2 < 1e-6) return dqs_pos;

    // 1. RQ = RQ1 · RQ2⁻¹
    Quat q1 = Cs[I1].q;
    Quat q2 = Cs[I2].q;
    Quat q_rel = q_mul(q1, q_inverse(q2));

    // 2. V_axis
    Vec3 axis = { q_rel.x, q_rel.y, q_rel.z };
    double ax_len = v3_len(axis);

    // 3. V_bisector = -normalise(dir1 + dir2)
    Vec3 bisector = v3_add(bone_dirs[I1], bone_dirs[I2]);
    if (v3_len(bisector) < 1e-6) return dqs_pos;
    bisector = v3_norm(bisector);
    bisector = v3_scale(bisector, -1.0);

    // 4. hierarchy 反转
    int d1 = (I1 < depths.size()) ? depths[I1] : 0;
    int d2 = (I2 < depths.size()) ? depths[I2] : 0;
    if (d2 > d1) bisector = v3_scale(bisector, -1.0);

    // 5. V_offset = V_bisector - V_axis · (V_axis · V_bisector)
    Vec3 V_offset = bisector;
    if (ax_len > 1e-9) {
        Vec3 axis_n = v3_scale(axis, 1.0 / ax_len);
        double d = v3_dot(axis_n, bisector);
        V_offset = v3_sub(bisector, v3_scale(axis_n, d));
    }

    // 6. w 和对称化 f
    double w = W2 / (W1 + W2);
    double w_sym = std::min(w, 1.0 - w);
    double w_sym2 = w_sym * w_sym;
    double f = 2.2 * w_sym - 8.1 * w_sym2 + 7.4 * w_sym2 * w_sym;

    // 7. geo = min(1, 2·√(1 - RQ.W))
    double cos_half = std::abs(q_rel.w);
    if (cos_half > 1.0) cos_half = 1.0;
    double geo = 2.0 * std::sqrt(std::max(0.0, 1.0 - cos_half));
    if (geo > 1.0) geo = 1.0;

    // 8. r = 顶点到关节中心距离
    //    关节中心 = jp[更深的骨骼]（弯曲的 pivot）
    double r = 0.0;
    if (I1 < joint_positions.size() && I2 < joint_positions.size()) {
        unsigned int deeper = (d1 > d2) ? I1 : I2;
        Vec3 jc = joint_positions[deeper];
        r = v3_len(v3_sub(rest_pos, jc));
    }

    // 9. k3 = 1 - W3 / W2 + clamp
    double k3 = 1.0;
    if (W3 > 1e-6 && W2 > 1e-6) {
        k3 = 1.0 - W3 / W2;
        if (k3 < 0.0) k3 = 0.0;
        if (k3 > 1.0) k3 = 1.0;
    }

    // 10. l = f · r · geo · (W1+W2) · k3 · strength
    double l = f * r * geo * (W1 + W2) * k3 * strength;

    // 11. V = V_orig + V_offset · l
    return v3_add(dqs_pos, v3_scale(V_offset, l));
}

// ================================================================
// v1 接口
// ================================================================

Quat qlerp(const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    return qlerp_v2(mapToSortedVec(weights), Cs);
}

Vec3 cor_vertex(const Vec3& pos, const Vec3& cor,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    return cor_vertex_v2(pos, cor, mapToSortedVec(weights), Cs);
}

Vec3 dlb_vertex(const Vec3& pos, const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    return dlb_vertex_v2(pos, mapToSortedVec(weights), Cs);
}

Vec3 dqs_compensate_vertex(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    const std::vector<int>& depths,
    double strength) {
    return dqs_compensate_vertex_v2(
        rest_pos, dqs_pos, mapToSortedVec(weights), Cs,
        bone_dirs, joint_positions, depths, strength);
}