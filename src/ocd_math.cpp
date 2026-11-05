#include "ocd_math.h"
#include <cmath>
#include <algorithm>

namespace ocd_math {

using namespace mathcore;

int effective_degree(int n_ctrl, int degree) {
    if (degree < 1) return 0;
    if (n_ctrl <= 1) return 0;
    // 控制点数必须严格大于阶数，否则 De Boor 取 d[j]=ctrl[k-degree+j] 会越界
    if (n_ctrl < degree + 1) return n_ctrl - 1;
    return degree;
}

std::vector<double> make_clamped_knots(int n_ctrl, int degree) {
    if (n_ctrl <= 0) return {};
    // 先按控制点数降阶，否则退化曲线会构造出长度与 n_ctrl+degree+1 不符的 knot 向量
    degree = effective_degree(n_ctrl, degree);
    if (degree < 1) {
        // 只有一个控制点：常量曲线，仅两个端节点
        return { 0.0, 1.0 };
    }
    // clamped B-spline：knot 总数 = n_ctrl + degree + 1，两端各重复 degree+1 个
    int n_knots = n_ctrl + degree + 1;
    int n_inner = n_ctrl - degree - 1;   // 内节点数量，可为 0
    std::vector<double> knots;
    knots.reserve(n_knots);
    for (int i = 0; i < degree + 1; ++i) knots.push_back(0.0);
    for (int i = 1; i <= n_inner; ++i)
        knots.push_back(double(i) / double(n_inner + 1));
    for (int i = 0; i < degree + 1; ++i) knots.push_back(1.0);
    return knots;
}

std::vector<double> make_clamped_knots_uneven(int n_ctrl, int degree,
    const std::vector<double>& inner) {
    if (n_ctrl <= 0) return {};
    degree = effective_degree(n_ctrl, degree);
    if (degree < 1) return { 0.0, 1.0 };

    const int n_inner = n_ctrl - degree - 1;
    // 无内节点或调用方给的点数不够时，均匀节点与 clamped 结构等价，直接复用
    if (n_inner <= 0 || (int)inner.size() < n_inner)
        return make_clamped_knots(n_ctrl, degree);

    std::vector<double> knots;
    knots.reserve(n_ctrl + degree + 1);
    for (int i = 0; i < degree + 1; ++i) knots.push_back(0.0);

    // 强制严格递增并夹到 (0,1)：重复或倒序内节点会让 De Boor 出现零除与 span 查找越界
    double prev = 0.0;
    for (int i = 0; i < n_inner; ++i) {
        double v = inner[i];
        if (v <= prev + 1e-9) v = prev + 1e-3;
        if (v >= 1.0 - 1e-6) v = 1.0 - 1e-6;
        knots.push_back(v);
        prev = v;
    }

    for (int i = 0; i < degree + 1; ++i) knots.push_back(1.0);

    // 兜底校验：上面「递增 + 夹取」在极端输入下仍可能把末位内节点挤到 >=1，
    // 任何非法结果一律退回均匀节点，保证下游 De Boor 求值安全
    if ((int)knots.size() != n_ctrl + degree + 1)
        return make_clamped_knots(n_ctrl, degree);
    for (size_t i = 1; i < knots.size(); ++i)
        if (knots[i] < knots[i - 1]) return make_clamped_knots(n_ctrl, degree);

    return knots;
}

Vec3 bspline_eval(const std::vector<Vec3>& ctrl,
                  const std::vector<double>& knots,
                  int degree, double t) {
    if (ctrl.empty()) return { 0.0, 0.0, 0.0 };
    // 控制点数不足以支撑该阶数时直接返回末点，避免越界读 ctrl
    if ((int)ctrl.size() < degree + 1) return ctrl.back();

    int n = (int)ctrl.size() - 1;   // 最高合法 span 编号
    // knot 向量长度必须满足 n + degree + 2，否则后续 knots[i+...] 会越界
    if ((int)knots.size() < n + degree + 2) return ctrl.back();

    // span 查找：二分定位最大的 k 使 knots[k] <= t，且 degree <= k <= n
    int k = degree;
    if (t >= 1.0 - 1e-15) {
        k = n;
    } else if (t > knots[degree]) {
        int lo = degree, hi = n;
        while (lo < hi) {
            int mid = (lo + hi + 1) / 2;
            if (knots[mid] <= t) lo = mid; else hi = mid - 1;
        }
        k = lo;
    }

    std::vector<Vec3> d(degree + 1);
    for (int j = 0; j <= degree; ++j) d[j] = ctrl[k - degree + j];

    for (int r = 1; r <= degree; ++r) {
        for (int j = degree; j >= r; --j) {
            int i = k - degree + j;
            double den = knots[i + degree - r + 1] - knots[i];
            double a = (std::abs(den) > 1e-12) ? (t - knots[i]) / den : 0.0;
            d[j] = v3_add(v3_scale(d[j-1], 1.0 - a), v3_scale(d[j], a));
        }
    }
    return d[degree];
}

Vec3 bspline_deriv(const std::vector<Vec3>& ctrl,
                   const std::vector<double>& knots,
                   int degree, double t) {
    int n = (int)ctrl.size() - 1;
    if (n < 1) return {0.0, 0.0, 0.0};
    std::vector<Vec3> dp;
    dp.reserve(n);
    for (int i = 0; i < n; ++i) {
        double den = knots[i + degree + 1] - knots[i + 1];
        if (std::abs(den) > 1e-12)
            dp.push_back(v3_scale(v3_sub(ctrl[i+1], ctrl[i]),
                                  double(degree) / den));
        else
            dp.push_back({0.0, 0.0, 0.0});
    }
    if (degree == 1) return dp.empty() ? Vec3{0,0,0} : dp[0];
    std::vector<double> sub_knots(knots.begin() + 1, knots.end() - 1);
    return bspline_eval(dp, sub_knots, degree - 1, t);
}

void build_frame(const Vec3& T_in, Vec3& Tn, Vec3& N, Vec3& B,
                 const Vec3& ref_in) {
    Vec3 T = v3_norm(T_in);
    if (v3_len(T) < 1e-12) {
        Tn = {0.0, 1.0, 0.0}; N = {1.0, 0.0, 0.0}; B = {0.0, 0.0, 1.0};
        return;
    }
    Vec3 ref = ref_in;
    if (std::abs(v3_dot(T, ref)) > 0.9) {
        ref = {0.0, 1.0, 0.0};
        if (std::abs(v3_dot(T, ref)) > 0.9) ref = {1.0, 0.0, 0.0};
    }
    Vec3 n = v3_cross(T, ref);
    if (v3_len(n) < 1e-12) n = {1.0, 0.0, 0.0};
    N = v3_norm(n);
    B = v3_cross(T, N);
    Tn = T;
}

ArclenLUT build_arclen_lut(const std::vector<Vec3>& ctrl,
                           const std::vector<double>& knots,
                           int degree, int n) {
    // 采样点少于 2 时除零（t = i/(n-1)），直接抬到 2
    if (n < 2) n = 2;
    ArclenLUT lut;
    lut.ts.resize(n);
    lut.s.resize(n);
    std::vector<Vec3> pts(n);
    for (int i = 0; i < n; ++i) {
        double t = double(i) / double(n - 1);
        lut.ts[i] = t;
        pts[i] = bspline_eval(ctrl, knots, degree, t);
    }
    lut.s[0] = 0.0;
    for (int i = 1; i < n; ++i)
        lut.s[i] = lut.s[i-1] + v3_len(v3_sub(pts[i], pts[i-1]));
    lut.s_total = lut.s[n-1];
    return lut;
}

double arclen_at_t(const ArclenLUT& lut, double t) {
    int n = (int)lut.ts.size();
    if (t <= lut.ts[0]) return lut.s[0];
    if (t >= lut.ts[n-1]) return lut.s[n-1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (lut.ts[m] < t) lo = m; else hi = m;
    }
    double seg = lut.ts[hi] - lut.ts[lo];
    if (seg < 1e-12) return lut.s[lo];
    double f = (t - lut.ts[lo]) / seg;
    return lut.s[lo] + f * (lut.s[hi] - lut.s[lo]);
}

double t_at_arclen(const ArclenLUT& lut, double target_s) {
    int n = (int)lut.s.size();
    if (target_s <= 0.0) return lut.ts[0];
    if (target_s >= lut.s_total) return lut.ts[n-1];
    int lo = 0, hi = n - 1;
    while (hi - lo > 1) {
        int m = (lo + hi) / 2;
        if (lut.s[m] < target_s) lo = m; else hi = m;
    }
    double seg = lut.s[hi] - lut.s[lo];
    if (seg < 1e-12) return lut.ts[lo];
    double f = (target_s - lut.s[lo]) / seg;
    return lut.ts[lo] + f * (lut.ts[hi] - lut.ts[lo]);
}

double refine_closest_t(const std::vector<Vec3>& ctrl,
                        const std::vector<double>& knots,
                        int degree, const Vec3& p, double t0,
                        int n_iter) {
    double t = t0;
    for (int it = 0; it < n_iter; ++it) {
        Vec3 C = bspline_eval(ctrl, knots, degree, t);
        Vec3 dC = bspline_deriv(ctrl, knots, degree, t);
        Vec3 diff = v3_sub(C, p);
        double f = v3_dot(diff, dC);

        double eps = 1e-4;
        double t1 = std::max(0.0, t - eps);
        double t2 = std::min(1.0, t + eps);
        Vec3 dC1 = bspline_deriv(ctrl, knots, degree, t1);
        Vec3 dC2 = bspline_deriv(ctrl, knots, degree, t2);
        Vec3 d2C = v3_scale(v3_sub(dC2, dC1), 1.0 / std::max(t2 - t1, 1e-9));
        double fp = v3_dot(dC, dC) + v3_dot(diff, d2C);

        if (std::abs(fp) < 1e-14) break;
        double tn = t - f / fp;
        if (tn < 0.0) tn = 0.0;
        if (tn > 1.0) tn = 1.0;
        if (std::abs(tn - t) < 1e-14) break;
        t = tn;
    }
    return t;
}

// ================================================================
// Mat3：3x3 线性变换（行主序）
// ================================================================
Mat3 Mat3::identity() {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = 0.0;
    r.m[0] = r.m[4] = r.m[8] = 1.0;
    return r;
}

Vec3 Mat3::apply(const Vec3& v) const {
    return {
        m[0] * v.x + m[1] * v.y + m[2] * v.z,
        m[3] * v.x + m[4] * v.y + m[5] * v.z,
        m[6] * v.x + m[7] * v.y + m[8] * v.z
    };
}

Mat3 Mat3::operator*(const Mat3& rhs) const {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            r.m[i * 3 + j] = m[i * 3 + 0] * rhs.m[0 * 3 + j]
                           + m[i * 3 + 1] * rhs.m[1 * 3 + j]
                           + m[i * 3 + 2] * rhs.m[2 * 3 + j];
        }
    }
    return r;
}

Mat3 Mat3::transposed() const {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j)
            r.m[i * 3 + j] = m[j * 3 + i];
    return r;
}

double Mat3::det() const {
    return m[0] * (m[4] * m[8] - m[5] * m[7])
         - m[1] * (m[3] * m[8] - m[5] * m[6])
         + m[2] * (m[3] * m[7] - m[4] * m[6]);
}

bool Mat3::inverse(Mat3& out) const {
    const double d = det();
    // 接近奇异时不输出，由调用方回退单位阵；静默返回垃圾逆矩阵会让控制点被甩飞
    if (std::abs(d) < 1e-12) return false;
    const double inv = 1.0 / d;
    Mat3 r;
    r.m[0] = (m[4] * m[8] - m[5] * m[7]) * inv;
    r.m[1] = (m[2] * m[7] - m[1] * m[8]) * inv;
    r.m[2] = (m[1] * m[5] - m[2] * m[4]) * inv;
    r.m[3] = (m[5] * m[6] - m[3] * m[8]) * inv;
    r.m[4] = (m[0] * m[8] - m[2] * m[6]) * inv;
    r.m[5] = (m[2] * m[3] - m[0] * m[5]) * inv;
    r.m[6] = (m[3] * m[7] - m[4] * m[6]) * inv;
    r.m[7] = (m[1] * m[6] - m[0] * m[7]) * inv;
    r.m[8] = (m[0] * m[4] - m[1] * m[3]) * inv;
    out = r;
    return true;
}

Mat3 polar_rotation(const Mat3& A) {
    // Newton 迭代 R ← (R + R⁻ᵀ)/2：从 A 出发二次收敛到 A 的正交极因子。
    // 本用途下 A = Σ w_b·R_b 已是接近刚性的混合矩阵，通常 4~6 次即收敛到 1e-12。
    Mat3 R = A;
    for (int it = 0; it < 8; ++it) {
        Mat3 Rti;
        // R⁻ᵀ = (Rᵀ)⁻¹；奇异说明输入退化，直接退回单位阵由调用方兜底
        if (!R.transposed().inverse(Rti)) return Mat3::identity();
        double diff = 0.0;
        for (int i = 0; i < 9; ++i) {
            const double v = 0.5 * (R.m[i] + Rti.m[i]);
            const double d = v - R.m[i];
            diff += d * d;
            R.m[i] = v;
        }
        if (diff < 1e-24) break;
    }
    // 退化保护：纯旋转的行列式应为 +1，浮点异常或收敛到反射时回退单位阵
    if (!(R.det() > 1e-6)) return Mat3::identity();
    return R;
}

// ================================================================
// 四元数辅助（不依赖 mathcore 是否提供 conj/normalize）
// ================================================================
namespace {

const double kOcdPi = 3.14159265358979323846;

Quat q_identity_local() { return { 0.0, 0.0, 0.0, 1.0 }; }

Quat q_conjugate_local(const Quat& q) {
    return { -q.x, -q.y, -q.z, q.w };
}

Quat q_normalized_local(const Quat& q) {
    const double len = std::sqrt(q.x * q.x + q.y * q.y + q.z * q.z + q.w * q.w);
    if (len < 1e-12) return q_identity_local();
    const double inv = 1.0 / len;
    return { q.x * inv, q.y * inv, q.z * inv, q.w * inv };
}

}  // namespace

Mat3 mat3_from_quat(const Quat& q_in) {
    const Quat q = q_normalized_local(q_in);
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    const double xx = x * x, yy = y * y, zz = z * z;
    const double xy = x * y, xz = x * z, yz = y * z;
    const double wx = w * x, wy = w * y, wz = w * z;

    Mat3 r;
    r.m[0] = 1.0 - 2.0 * (yy + zz);
    r.m[1] = 2.0 * (xy - wz);
    r.m[2] = 2.0 * (xz + wy);
    r.m[3] = 2.0 * (xy + wz);
    r.m[4] = 1.0 - 2.0 * (xx + zz);
    r.m[5] = 2.0 * (yz - wx);
    r.m[6] = 2.0 * (xz - wy);
    r.m[7] = 2.0 * (yz + wx);
    r.m[8] = 1.0 - 2.0 * (xx + yy);
    return r;
}

Quat mat3_to_quat(const Mat3& m) {
    // Shepperd's method：按 trace 分支选主分量，避免除以小数放大误差
    const double trace = m.m[0] + m.m[4] + m.m[8];
    if (trace > 0.0) {
        const double s = std::sqrt(trace + 1.0) * 2.0;
        const double inv = 1.0 / s;
        return {
            (m.m[7] - m.m[5]) * inv,
            (m.m[2] - m.m[6]) * inv,
            (m.m[3] - m.m[1]) * inv,
            0.25 * s
        };
    }
    else if (m.m[0] > m.m[4] && m.m[0] > m.m[8]) {
        const double s = std::sqrt(1.0 + m.m[0] - m.m[4] - m.m[8]) * 2.0;
        const double inv = 1.0 / s;
        return {
            0.25 * s,
            (m.m[1] + m.m[3]) * inv,
            (m.m[2] + m.m[6]) * inv,
            (m.m[7] - m.m[5]) * inv
        };
    }
    else if (m.m[4] > m.m[8]) {
        const double s = std::sqrt(1.0 + m.m[4] - m.m[0] - m.m[8]) * 2.0;
        const double inv = 1.0 / s;
        return {
            (m.m[1] + m.m[3]) * inv,
            0.25 * s,
            (m.m[5] + m.m[7]) * inv,
            (m.m[2] - m.m[6]) * inv
        };
    }
    else {
        const double s = std::sqrt(1.0 + m.m[8] - m.m[0] - m.m[4]) * 2.0;
        const double inv = 1.0 / s;
        return {
            (m.m[2] + m.m[6]) * inv,
            (m.m[5] + m.m[7]) * inv,
            0.25 * s,
            (m.m[3] - m.m[1]) * inv
        };
    }
}

Quat quat_axis_angle(const Vec3& axis_unit, double angle) {
    Vec3 a = v3_norm(axis_unit);
    if (v3_len(a) < 1e-12) return q_identity_local();
    const double half = 0.5 * angle;
    const double s = std::sin(half);
    return { a.x * s, a.y * s, a.z * s, std::cos(half) };
}

Mat3 mat3_axis_angle(const Vec3& axis_unit, double angle) {
    return mat3_from_quat(quat_axis_angle(axis_unit, angle));
}

Quat quat_shortest_arc(const Vec3& from_unit, const Vec3& to_unit) {
    const Vec3 f = v3_norm(from_unit);
    const Vec3 t = v3_norm(to_unit);
    if (v3_len(f) < 1e-12 || v3_len(t) < 1e-12) return q_identity_local();

    const double d = v3_dot(f, t);
    if (d > 1.0 - 1e-9) return q_identity_local();
    if (d < -1.0 + 1e-9) {
        // 共线反向：最短弧不唯一，任取一个与 f 垂直的轴转 pi
        Vec3 axis = v3_cross(f, { 0.0, 1.0, 0.0 });
        if (v3_len(axis) < 1e-9) axis = v3_cross(f, { 1.0, 0.0, 0.0 });
        return quat_axis_angle(v3_norm(axis), kOcdPi);
    }

    const Vec3 axis = v3_norm(v3_cross(f, t));
    const double angle = std::acos(std::max(-1.0, std::min(1.0, d)));
    return quat_axis_angle(axis, angle);
}

void decompose_swing_twist(const Quat& R_in, const Vec3& axis_bind_in,
                           Quat& R_swing, Quat& R_twist,
                           Vec3& axis_cur_unit, double& twist_angle) {
    R_swing = q_identity_local();
    R_twist = q_identity_local();
    twist_angle = 0.0;

    const Vec3 axis_bind = v3_norm(axis_bind_in);
    if (v3_len(axis_bind) < 1e-12) {
        axis_cur_unit = { 0.0, 1.0, 0.0 };
        return;
    }

    const Quat R = q_normalized_local(R_in);

    // 当前轴：R 作用在 bind 轴上
    axis_cur_unit = v3_norm(q_rotate_v3(R, axis_bind));
    if (v3_len(axis_cur_unit) < 1e-12) axis_cur_unit = axis_bind;

    // swing 只负责把 bind 轴转到当前轴。它的几何效应已经由 b_j 的平移传递，
    // 所以 M_j 中不能再完整施加 R，否则 swing 被重复旋转（历史故障：关节整体偏转）
    R_swing = quat_shortest_arc(axis_bind, axis_cur_unit);

    // twist = 余下的绕当前轴的旋转
    R_twist = q_normalized_local(mathcore::q_mul(R, q_conjugate_local(R_swing)));

    const Vec3 v = { R_twist.x, R_twist.y, R_twist.z };
    const double sin_half = v3_dot(v, axis_cur_unit);

    // atan2 同时给出符号：sin 分量投影到当前轴，w 为 cos(theta/2)
    double angle = 2.0 * std::atan2(sin_half, R_twist.w);
    while (angle > kOcdPi) angle -= 2.0 * kOcdPi;
    while (angle <= -kOcdPi) angle += 2.0 * kOcdPi;
    twist_angle = angle;
}

Mat3 mat3_axis_scale(const Vec3& axis_unit, double s) {
    const Vec3 a = v3_norm(axis_unit);
    if (v3_len(a) < 1e-12) return Mat3::identity();

    const double d = s - 1.0;
    Mat3 r = Mat3::identity();
    r.m[0] += d * a.x * a.x; r.m[1] += d * a.x * a.y; r.m[2] += d * a.x * a.z;
    r.m[3] += d * a.y * a.x; r.m[4] += d * a.y * a.y; r.m[5] += d * a.y * a.z;
    r.m[6] += d * a.z * a.x; r.m[7] += d * a.z * a.y; r.m[8] += d * a.z * a.z;
    return r;
}

Mat3 mat3_anisotropic_scale(double sx, double sy, double sz) {
    Mat3 r = Mat3::identity();
    r.m[0] = sx;
    r.m[4] = sy;
    r.m[8] = sz;
    return r;
}

Mat3 build_control_M(const Quat& R,
                     const Vec3& L0, const Vec3& L1, const Vec3& L2,
                     const Vec3& axis_bind_unit,
                     double twist_weight, double scale_weight, int scale_mode) {
    // M_j = (S_j · W_j) · R_j
    // v 是 bind 世界空间下的偏移方向量，必须先由相对旋转 R 带到当前姿势，
    // 再在当前局部标架内施加缩放/twist。
    // 不能用共轭形式 R(SW)R^T：S=W=I 时它会退化为 I，v 不随骨骼旋转，
    // 弯折时内侧顶点被压扁（实测故障）。
    // b_j 是控制点位置（已含旋转），v 是方向量，两者乘 R 不会重复。
    const Mat3 Rm = mat3_from_quat(R);
    Mat3 local = Mat3::identity();   // 局部标架下的 S_j · W_j

    const Vec3 a_bind = v3_norm(axis_bind_unit);
    // 当前轴：R 作用在 bind 轴上；bind 时 R = I，退化为 a_bind 本身
    const Vec3 axis_cur = (v3_len(a_bind) > 1e-12)
        ? v3_norm(Rm.apply(a_bind))
        : Vec3{ 0.0, 1.0, 0.0 };

    // ---- W_j：沿当前轴的纯 twist 残差 ----
    if (twist_weight > 1e-9 && v3_len(a_bind) > 1e-12) {
        Quat R_swing, R_twist;
        Vec3 twist_axis = axis_cur;
        double twist_angle = 0.0;
        decompose_swing_twist(R, a_bind, R_swing, R_twist,
                              twist_axis, twist_angle);
        if (std::abs(twist_angle) > 1e-9) {
            local = mat3_axis_angle(twist_axis, twist_weight * twist_angle) * local;
        }
    }

    // ---- S_j：轴向 / 各向异性缩放 ----
    if (scale_weight > 1e-9 && scale_mode != 0) {
        const bool hasLinear = v3_len(L0) > 1e-12 ||
                               v3_len(L1) > 1e-12 ||
                               v3_len(L2) > 1e-12;

        if (hasLinear && v3_len(a_bind) > 1e-12) {
            if (scale_mode == 2) {
                // 各向异性：三个列向量的模长作为主轴缩放（主轴对齐，共轭后仍成立）
                const Mat3 S = mat3_anisotropic_scale(
                    std::pow(std::max(v3_len(L0), 1e-9), scale_weight),
                    std::pow(std::max(v3_len(L1), 1e-9), scale_weight),
                    std::pow(std::max(v3_len(L2), 1e-9), scale_weight));
                local = S * local;
            }
            else {
                // 仅轴向：s_axis 是线性部分沿 bind 轴的长度，bind 姿势下恒为 1
                const Vec3 La = v3_add(
                    v3_add(v3_scale(L0, a_bind.x), v3_scale(L1, a_bind.y)),
                    v3_scale(L2, a_bind.z));
                const double s_axis = v3_len(La);
                if (s_axis > 1e-9) {
                    // 幂次而非线性插值：weight=0 时 s^0=1 严格回到无缩放
                    local = mat3_axis_scale(axis_cur,
                        std::pow(s_axis, scale_weight)) * local;
                }
            }
        }
    }

    return local * Rm;
}

bool solve4x4(double A[4][4], const double b[4], double x[4]) {
    double m[4][5];
    for (int i = 0; i < 4; ++i) {
        for (int j = 0; j < 4; ++j) m[i][j] = A[i][j];
        m[i][4] = b[i];
    }

    for (int col = 0; col < 4; ++col) {
        // 部分主元：避免小主元放大舍入误差
        int pivot = col;
        for (int r = col + 1; r < 4; ++r)
            if (std::abs(m[r][col]) > std::abs(m[pivot][col])) pivot = r;
        if (std::abs(m[pivot][col]) < 1e-12) return false;

        if (pivot != col)
            for (int j = col; j < 5; ++j) std::swap(m[col][j], m[pivot][j]);

        const double inv = 1.0 / m[col][col];
        for (int j = col; j < 5; ++j) m[col][j] *= inv;

        // Gauss-Jordan：消去其余各行，直接得到解
        for (int r = 0; r < 4; ++r) {
            if (r == col) continue;
            const double f = m[r][col];
            if (std::abs(f) < 1e-18) continue;
            for (int j = col; j < 5; ++j) m[r][j] -= f * m[col][j];
        }
    }

    for (int i = 0; i < 4; ++i) x[i] = m[i][4];
    return true;
}

double tau_from_t(const ArclenLUT& lut, double t) {
    if (lut.ts.size() < 2 || lut.s.size() < 2) return t;
    if (lut.s_total < 1e-12) return t;
    return arclen_at_t(lut, t) / lut.s_total;
}

double t_from_tau(const ArclenLUT& lut, double tau) {
    if (lut.ts.size() < 2 || lut.s.size() < 2) return tau;
    if (lut.s_total < 1e-12) return tau;
    return t_at_arclen(lut, tau * lut.s_total);
}

// ================================================================
// 体积补偿：专利 US 8,400,455 B2 的 space-unfolding 参数
// ================================================================
double space_unfold_tau(const Vec3& P, const Vec3& O,
                        const Vec3& Q, const Vec3& R) {
    const Vec3 p = v3_sub(P, O);
    const Vec3 q = v3_sub(Q, O);
    const Vec3 r = v3_sub(R, O);

    const double lp = v3_len(p);
    const double lq = v3_len(q);
    const double lr = v3_len(r);

    // 关节重合或顶点落在关节上时角度无定义，返回 1 表示「无额外弯曲补偿」
    if (lp < 1e-9 || lq < 1e-9 || lr < 1e-9) return 1.0;

    const Vec3 pn = v3_scale(p, 1.0 / lp);
    const Vec3 qn = v3_scale(q, 1.0 / lq);
    const Vec3 rn = v3_scale(r, 1.0 / lr);

    // Eq.4：∠ROQ
    const double theta =
        std::acos(std::max(-1.0, std::min(1.0, v3_dot(rn, qn))));
    // Eq.5：∠POQ
    const double alpha_min =
        std::acos(std::max(-1.0, std::min(1.0, v3_dot(pn, qn))));

    // 完全伸直时无需展开，直接返回 1，避免 alpha 作为除数趋于 0
    if (alpha_min < 1e-9) return 1.0;

    // Eq.6：用叉积同侧性取夹角，保证 R 与 P 在 OQ 的同一侧展开
    const double side = v3_dot(v3_cross(pn, qn), v3_cross(rn, qn));
    const double alpha = (side >= 0.0) ? alpha_min : (2.0 * kOcdPi - alpha_min);
    if (alpha < 1e-9) return 1.0;

    // Eq.8：把 ∠ROQ 按比例展开到与 ∠POQ 一致
    const double theta_flat = theta * (kOcdPi / alpha);
    // Eq.9：R 在展开后沿 OQ 的有符号投影距离
    const double epsilon = lr * std::cos(theta_flat);

    // Eq.10：按两段骨骼长度加权的归一化位置参数
    const double a = lp;
    const double b = lq;
    return (a + a * std::min(0.0, epsilon) + b * std::max(0.0, epsilon)) / (a + b);
}

std::vector<double> inverse_distance_sq_weights(const Vec3& ref,
                                                const std::vector<Vec3>& cvs) {
    std::vector<double> w(cvs.size(), 0.0);
    if (cvs.empty()) return w;

    double sum = 0.0;
    size_t zero_idx = cvs.size();   // 与 ref 重合的控制点下标

    for (size_t i = 0; i < cvs.size(); ++i) {
        const Vec3 d = v3_sub(ref, cvs[i]);
        const double d2 = v3_dot(d, d);
        if (d2 < 1e-18) { zero_idx = i; break; }
        // 逆距离平方：越靠近关节的控制点被 Tau 推拉得越多
        w[i] = 1.0 / d2;
        sum += w[i];
    }

    // 控制点正好落在关节上时该点独占权重，避免 1/0 变成 inf 污染整条曲线
    if (zero_idx < cvs.size()) {
        std::fill(w.begin(), w.end(), 0.0);
        w[zero_idx] = 1.0;
        return w;
    }

    if (sum > 1e-30) {
        for (double& x : w) x /= sum;
    }
    return w;
}

}  // namespace ocd_math
