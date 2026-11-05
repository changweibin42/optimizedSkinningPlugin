#pragma once
#include <cmath>
#include <algorithm>

// ================================================================
// 纯数学库 — 无 Maya 依赖，可直接复用到 GPU 内核
// 约定：
//   Vec3 = {x, y, z}
//   Quat = {x, y, z, w}
//   矩阵运算用 (q, t) 对表示刚性变换
// ================================================================

namespace mathcore {

// ---------- Vec3 ----------
struct Vec3 {
    double x, y, z;
};

inline Vec3 v3_add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}
inline Vec3 v3_sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}
inline Vec3 v3_scale(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}
inline double v3_dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}
inline Vec3 v3_cross(const Vec3& a, const Vec3& b) {
    return { a.y * b.z - a.z * b.y,
            a.z * b.x - a.x * b.z,
            a.x * b.y - a.y * b.x };
}
inline double v3_len(const Vec3& a) {
    return std::sqrt(v3_dot(a, a));
}
inline Vec3 v3_norm(const Vec3& a) {
    double l = v3_len(a);
    if (l < 1e-12) return {0.0, 0.0, 0.0};
    return {a.x / l, a.y / l, a.z / l};
}

// ---------- Quat ----------
struct Quat {
    double x, y, z, w;
};

inline Quat q_mul(const Quat& q1, const Quat& q2) {
    return {
        q1.w*q2.x + q1.x*q2.w + q1.y*q2.z - q1.z*q2.y,
        q1.w*q2.y - q1.x*q2.z + q1.y*q2.w + q1.z*q2.x,
        q1.w*q2.z + q1.x*q2.y - q1.y*q2.x + q1.z*q2.w,
        q1.w*q2.w - q1.x*q2.x - q1.y*q2.y - q1.z*q2.z
    };
}
inline Quat q_conjugate(const Quat& q) {
    return {-q.x, -q.y, -q.z, q.w};
}
inline Quat q_inverse(const Quat& q) {
    return q_conjugate(q);   // 单位四元数
}
inline double q_norm(const Quat& q) {
    return std::sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
}
inline Quat q_normalize(const Quat& q) {
    double n = q_norm(q);
    if (n < 1e-12) return {0.0, 0.0, 0.0, 1.0};
    return {q.x/n, q.y/n, q.z/n, q.w/n};
}
inline Quat q_scale(const Quat& q, double s) {
    return {q.x*s, q.y*s, q.z*s, q.w*s};
}
inline Quat q_add(const Quat& q1, const Quat& q2) {
    return {q1.x+q2.x, q1.y+q2.y, q1.z+q2.z, q1.w+q2.w};
}
inline double q_dot(const Quat& q1, const Quat& q2) {
    return q1.x*q2.x + q1.y*q2.y + q1.z*q2.z + q1.w*q2.w;
}

inline Vec3 q_rotate_v3(const Quat& q, const Vec3& v) {
    Quat v_q = {v.x, v.y, v.z, 0.0};
    Quat r = q_mul(q_mul(q, v_q), q_conjugate(q));
    return {r.x, r.y, r.z};
}

inline Quat q_axis_angle(const Vec3& axis, double angle) {
    double len = std::sqrt(axis.x * axis.x + axis.y * axis.y + axis.z * axis.z);
    if (len < 1e-12 || std::abs(angle) < 1e-9)
        return { 0.0, 0.0, 0.0, 1.0 };
    double s = std::sin(angle * 0.5) / len;
    return { axis.x * s, axis.y * s, axis.z * s, std::cos(angle * 0.5) };
}

inline Quat q_between(const Vec3& a, const Vec3& b) {
    Vec3 an = v3_norm(a);
    Vec3 bn = v3_norm(b);
    double d = v3_dot(an, bn);
    if (d > 1.0 - 1e-9) return { 0.0, 0.0, 0.0, 1.0 };
    if (d < -1.0 + 1e-9) {
        Vec3 ax = v3_cross(an, { 1.0, 0.0, 0.0 });
        if (v3_len(ax) < 1e-6) ax = v3_cross(an, { 0.0, 1.0, 0.0 });
        return q_axis_angle(ax, 3.14159265358979323846);
    }
    // 显式 clamp，避免 std::min/max 依赖
    if (d < -1.0) d = -1.0;
    if (d > 1.0) d = 1.0;
    return q_axis_angle(v3_cross(an, bn), std::acos(d));
}


// ---------- 刚性变换 (q, t) ----------
struct RigidTransform {
    Quat q;
    Vec3 t;
};

inline Vec3 apply_transform(const RigidTransform& rt, const Vec3& v) {
    return v3_add(q_rotate_v3(rt.q, v), rt.t);
}

// ---------- 核心算法函数 ----------

// C = P · R⁻¹
inline RigidTransform compute_C(const Quat& pose_q, const Vec3& pose_t,
                                 const Quat& rest_q, const Vec3& rest_t) {
    Quat c_q = q_mul(pose_q, q_inverse(rest_q));
    Vec3 rotated = q_rotate_v3(c_q, rest_t);
    Vec3 c_t = {pose_t.x - rotated.x,
                pose_t.y - rotated.y,
                pose_t.z - rotated.z};
    return {c_q, c_t};
}

// 双四元数：(q_real, q_dual)
struct DualQuat {
    Quat real;
    Quat dual;
};

inline DualQuat dual_quat_from_C(const RigidTransform& C) {
    Quat t_q = {C.t.x, C.t.y, C.t.z, 0.0};
    Quat q_d = q_scale(q_mul(t_q, C.q), 0.5);
    return {C.q, q_d};
}

// p' = q_r · p · q_r⁻¹ + 2 · (q_d · q_r⁻¹).xyz
inline Vec3 dlb_transform(const Quat& q_r, const Quat& q_d, const Vec3& p) {
    Vec3 p_rot = q_rotate_v3(q_r, p);
    Quat trans_q = q_mul(q_d, q_conjugate(q_r));
    Vec3 trans = {2*trans_q.x, 2*trans_q.y, 2*trans_q.z};
    return v3_add(p_rot, trans);
}

// ---------- 点线距离 ----------
inline double point_to_segment_distance(const Vec3& p,
                                         const Vec3& a, const Vec3& b) {
    Vec3 ab = v3_sub(b, a);
    double ab_len_sq = v3_dot(ab, ab);
    if (ab_len_sq < 1e-12) return v3_len(v3_sub(p, a));
    double t = v3_dot(v3_sub(p, a), ab) / ab_len_sq;
    if (t < 0.0) t = 0.0;
    else if (t > 1.0) t = 1.0;
    Vec3 closest = v3_add(a, v3_scale(ab, t));
    return v3_len(v3_sub(p, closest));
}

} // namespace mathcore