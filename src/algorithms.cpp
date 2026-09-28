#include "algorithms.h"
#include <algorithm>
#include <cmath>

// ================================================================
// LBS
// ================================================================
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

// ================================================================
// QLERP
// ================================================================
Quat qlerp(const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    unsigned int ref_idx = 0;
    double max_w = -1.0;
    for (const auto& kv : weights) {
        if (kv.second > max_w) { max_w = kv.second; ref_idx = kv.first; }
    }
    Quat ref_q = Cs[ref_idx].q;

    Quat q_sum = { 0.0, 0.0, 0.0, 0.0 };
    for (const auto& kv : weights) {
        Quat q = Cs[kv.first].q;
        if (q_dot(q, ref_q) < 0.0) q = q_scale(q, -1.0);
        q_sum = q_add(q_sum, q_scale(q, kv.second));
    }
    return q_normalize(q_sum);
}

// ================================================================
// DLB
// ================================================================
Vec3 dlb_vertex(const Vec3& pos, const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    struct Item { Quat q_r, q_d; double w; };
    std::vector<Item> items;

    for (const auto& kv : weights) {
        DualQuat dq = dual_quat_from_C(Cs[kv.first]);
        items.push_back({ dq.real, dq.dual, kv.second });
    }

    Quat ref_qr = items[0].q_r;
    double max_w = items[0].w;
    for (const auto& it : items) {
        if (it.w > max_w) { max_w = it.w; ref_qr = it.q_r; }
    }

    Quat q_r_sum = { 0.0, 0.0, 0.0, 0.0 };
    Quat q_d_sum = { 0.0, 0.0, 0.0, 0.0 };

    for (auto& it : items) {
        Quat q_r = it.q_r, q_d = it.q_d;
        if (q_dot(q_r, ref_qr) < 0.0) {
            q_r = q_scale(q_r, -1.0);
            q_d = q_scale(q_d, -1.0);
        }
        q_r_sum = q_add(q_r_sum, q_scale(q_r, it.w));
        q_d_sum = q_add(q_d_sum, q_scale(q_d, it.w));
    }

    double n = q_norm(q_r_sum);
    if (n < 1e-8) return pos;

    return dlb_transform(q_scale(q_r_sum, 1.0 / n),
        q_scale(q_d_sum, 1.0 / n), pos);
}

// ================================================================
// CoR
// ================================================================
Vec3 cor_vertex(const Vec3& pos, const Vec3& cor,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs) {
    Quat q_p = qlerp(weights, Cs);

    Vec3 cor_runtime = { 0.0, 0.0, 0.0 };
    for (const auto& kv : weights) {
        Vec3 t = apply_transform(Cs[kv.first], cor);
        cor_runtime = v3_add(cor_runtime, v3_scale(t, kv.second));
    }

    Vec3 cor_rotated = q_rotate_v3(q_p, cor);
    Vec3 t_p = v3_sub(cor_runtime, cor_rotated);
    return v3_add(q_rotate_v3(q_p, pos), t_p);
}

// ================================================================
// DQS 补偿  ★ 用 dqs_pos 到关节中心的方向动态决定 bisector 朝向
// ================================================================
Vec3 dqs_compensate_vertex(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    double strength) {

    std::vector<std::pair<unsigned int, double>> sorted(
        weights.begin(), weights.end());
    std::sort(sorted.begin(), sorted.end(),
        [](const auto& a, const auto& b) { return a.second > b.second; });

    if (sorted.size() < 2) return dqs_pos;

    unsigned int b1 = sorted[0].first;
    double w1 = sorted[0].second;
    unsigned int b2 = sorted[1].first;
    double w2 = sorted[1].second;
    double w3 = (sorted.size() > 2) ? sorted[2].second : 0.0;

    double w_sum = w1 + w2;
    if (w_sum < 1e-6) return dqs_pos;

    double w2_norm = w2 / w_sum;
    double f = 2.2 * w2_norm - 8.1 * w2_norm * w2_norm + 7.4 * w2_norm * w2_norm * w2_norm;

    // 基础 bisector：两骨骼当前姿势 Y 轴之和
    Vec3 bisector = v3_add(bone_dirs[b1], bone_dirs[b2]);
    if (v3_len(bisector) < 1e-6) return dqs_pos;
    bisector = v3_norm(bisector);

    // 扭转修正
    Quat q1 = Cs[b1].q, q2 = Cs[b2].q;
    Quat q_rel = q_mul(q_inverse(q1), q2);
    Vec3 axis = { q_rel.x, q_rel.y, q_rel.z };
    if (v3_len(axis) > 1e-6) {
        axis = v3_norm(axis);
        double d = v3_dot(axis, bisector);
        bisector = v3_sub(bisector, v3_scale(axis, d));
        if (v3_len(bisector) > 1e-6) bisector = v3_norm(bisector);
    }

    // ★ 核心修正：让 bisector 指向关节中心
    //   关节中心 = 层次更深的骨骼（索引大的）位置
    unsigned int deeper = (b1 > b2) ? b1 : b2;
    Vec3 joint_center = joint_positions[deeper];

    Vec3 to_center = v3_sub(joint_center, dqs_pos);
    if (v3_len(to_center) > 1e-6) {
        to_center = v3_norm(to_center);
        if (v3_dot(bisector, to_center) < 0.0) {
            bisector = v3_scale(bisector, -1.0);
        }
    }

    double q_rel_w = std::abs(q_rel.w);
    if (q_rel_w > 1.0) q_rel_w = 1.0;
    double alpha = 2.0 * std::acos(q_rel_w);

    double min_dist = 1e18;
    for (size_t i = 0; i + 1 < joint_positions.size(); ++i) {
        double d = point_to_segment_distance(rest_pos,
            joint_positions[i],
            joint_positions[i + 1]);
        if (d < min_dist) min_dist = d;
    }

    double cos_term = std::cos(alpha / 2.0);
    double geo = 2.0 * std::sqrt(std::max(0.0, 1.0 - cos_term));
    if (geo > 1.0) geo = 1.0;
    double l = f * min_dist * geo;

    double k3 = (w3 < 1e-6 || w2 < 1e-6) ? 1.0 : 1.0 - w3 / (w2 + w3);

    Vec3 offset = v3_scale(bisector, l * k3 * strength);
    return v3_add(dqs_pos, offset);
}