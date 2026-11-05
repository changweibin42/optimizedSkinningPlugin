#pragma once
#include "math_core.h"
#include <vector>

// ================================================================
// OCD 纯数学：De Boor、曲线标架、弧长 LUT、最近点搜索
// 无 Maya 依赖，可复用到 GPU 内核
// ================================================================
namespace ocd_math {

using mathcore::Vec3;
using mathcore::Quat;

/// <summary>
/// 计算给定控制点数量下实际可用的曲线阶数。
/// 三次 B-spline 至少需要 4 个控制点，控制点不足时自动降阶，
/// 避免构造出非法 knot 向量或 De Boor 越界访问。
/// </summary>
/// <param name="n_ctrl">控制点数量</param>
/// <param name="degree">期望阶数</param>
/// <returns>实际可用阶数；控制点少于 2 个时返回 0</returns>
int effective_degree(int n_ctrl, int degree);

// Clamped 均匀节点向量（内部会按 effective_degree 自动降阶）
std::vector<double> make_clamped_knots(int n_ctrl, int degree);

/// <summary>
/// 用给定内节点构造 clamped 节点向量。
/// 内节点取自控制点的骨骼参数 cvs_u，使 knot 分布跟随骨骼参数而非均匀分布，
/// 从而让曲线参数 τ 与沿链骨骼参数近似线性对应。
/// 关键：内节点序列通过对称取法（见 finalize_curve）保证关于 τ=0.5 镜像，
///      使 B-spline 基函数 N_j(τ) 关于 τ=0.5 对称，两侧变形严格对称。
/// </summary>
/// <param name="n_ctrl">控制点数量，必须大于 0</param>
/// <param name="degree">期望阶数，内部会先按 effective_degree 降阶</param>
/// <param name="inner">期望的内节点序列，应递增且落在 (0,1) 开区间；
///                     长度不足 n_ctrl-degree-1 时回退为均匀节点</param>
/// <returns>长度恰为 n_ctrl+degree+1 的节点向量；
///          内节点倒序、越界或数量不足时静默回退为 make_clamped_knots 的结果</returns>
std::vector<double> make_clamped_knots_uneven(int n_ctrl, int degree,
    const std::vector<double>& inner);

// 三次 B-spline 求值（De Boor）
Vec3 bspline_eval(const std::vector<Vec3>& ctrl,
                  const std::vector<double>& knots,
                  int degree, double t);

// 一阶导数（降阶 B-spline 求值）
Vec3 bspline_deriv(const std::vector<Vec3>& ctrl,
                   const std::vector<double>& knots,
                   int degree, double t);

// 曲线局部标架 (T, N, B)，用固定参考轴
void build_frame(const Vec3& T_in, Vec3& Tn, Vec3& N, Vec3& B,
                 const Vec3& ref);

// 弧长 LUT
struct ArclenLUT {
    std::vector<double> ts;
    std::vector<double> s;
    double s_total = 0.0;
};

ArclenLUT build_arclen_lut(const std::vector<Vec3>& ctrl,
                           const std::vector<double>& knots,
                           int degree, int n = 512);

double arclen_at_t(const ArclenLUT& lut, double t);
double t_at_arclen(const ArclenLUT& lut, double target_s);

// 牛顿迭代找最近点参数 t
double refine_closest_t(const std::vector<Vec3>& ctrl,
                        const std::vector<double>& knots,
                        int degree, const Vec3& p, double t0,
                        int n_iter = 20);

// ================================================================
// M_j = S_j · W_j · R_j 所需的线性代数原语
// 说明：mathcore::RigidTransform 不含缩放，无法表达 S_j，
//       故这里用独立的 3×3 矩阵表达「相对 bind 的线性变换」。
// ================================================================

/**
 * @brief 3×3 线性变换（行主序，m[0..2] 为第一行）。
 * 职责：表达相对 bind 的局部变换 M_j = S_j·W_j·R_j；
 *      与 RigidTransform 的区别是可包含非均匀缩放。
 */
struct Mat3 {
    double m[9];

    /// <summary>单位阵</summary>
    static Mat3 identity();
    /// <summary>作用到向量：返回 M · v</summary>
    Vec3 apply(const Vec3& v) const;
    /// <summary>矩阵乘：返回 this * rhs</summary>
    Mat3 operator*(const Mat3& rhs) const;
    /// <summary>转置</summary>
    Mat3 transposed() const;
    /// <summary>行列式</summary>
    double det() const;
    /// <summary>
    /// 求逆矩阵。
    /// </summary>
    /// <param name="out">输出逆矩阵；奇异时不写入</param>
    /// <returns>false 表示奇异，调用方须回退为单位阵</returns>
    bool inverse(Mat3& out) const;
};

/// <summary>
/// 3×3 线性变换的极分解旋转部分：R = A·(AᵀA)^{-1/2}。
/// 用 Newton 迭代 R ← (R + R⁻ᵀ)/2 求解，把任意非奇异线性变换投影为最近的正交旋转。
/// 用途：关节处先对多骨骼旋转做加权平均得到线性混合矩阵，再用本函数恢复刚性旋转，
///       使等效旋转角严格等于权重的线性混合角，不会因基函数或控制点位置差异而衰减。
/// </summary>
/// <param name="A">待分解的线性变换，通常是 Σ w_b·R_b</param>
/// <returns>最接近 A 的纯旋转；A 奇异、迭代失败或行列式退化时返回单位阵</returns>
Mat3 polar_rotation(const Mat3& A);

/// <summary>由四元数构造旋转矩阵，内部先归一化，零四元数退回单位阵</summary>
Mat3 mat3_from_quat(const Quat& q);

/// <summary>
/// 从旋转矩阵反推四元数（Shepperd's method）。
/// 输入必须是纯旋转矩阵（行列式≈1）；数值退化时返回单位四元数。
/// 用途：polar_rotation 返回的是 Mat3，Twist 主动控制需要在四元数空间分解后再重组。
/// </summary>
Quat mat3_to_quat(const Mat3& m);

/// <summary>由单位轴与弧度构造旋转矩阵，角度不做归一化以支持连续 twist</summary>
Mat3 mat3_axis_angle(const Vec3& axis_unit, double angle);

/// <summary>由单位轴与弧度构造四元数，不依赖 mathcore 是否提供同名函数</summary>
Quat quat_axis_angle(const Vec3& axis_unit, double angle);

/// <summary>
/// 最短弧旋转：把单位向量 from 旋到单位向量 to。
/// 同向返回单位四元数；反向时绕任意垂直轴旋转 π。
/// </summary>
Quat quat_shortest_arc(const Vec3& from_unit, const Vec3& to_unit);

/// <summary>
/// swing-twist 分解（Wires 式父关节分解）。
/// </summary>
/// <param name="R">相对 bind 的完整旋转</param>
/// <param name="axis_bind_unit">骨骼 bind 单位轴</param>
/// <param name="R_swing">输出：把 axis_bind 转到当前轴的最小旋转</param>
/// <param name="R_twist">输出：余下的绕当前轴的纯 twist</param>
/// <param name="axis_cur_unit">输出：当前单位轴（= R · axis_bind）</param>
/// <param name="twist_angle">输出：带符号 twist 角，∈ (-π, π]</param>
void decompose_swing_twist(const Quat& R, const Vec3& axis_bind_unit,
                           Quat& R_swing, Quat& R_twist,
                           Vec3& axis_cur_unit, double& twist_angle);

/// <summary>轴向对称缩放：I + (s−1)·(axis ⊗ axis)</summary>
Mat3 mat3_axis_scale(const Vec3& axis_unit, double s);

/// <summary>各向异性缩放 diag(sx, sy, sz)</summary>
Mat3 mat3_anisotropic_scale(double sx, double sy, double sz);

/// <summary>
/// 构造 M_j = S_j · W_j · R_j。
/// </summary>
/// <param name="R">相对 bind 旋转</param>
/// <param name="L0">骨骼线性部分的第 0 列向量</param>
/// <param name="L1">骨骼线性部分的第 1 列向量</param>
/// <param name="L2">骨骼线性部分的第 2 列向量</param>
/// <param name="axis_bind_unit">控制点绑定骨骼的 bind 单位轴</param>
/// <param name="twist_weight">[0,1]，0 时 W = I</param>
/// <param name="scale_weight">[0,1]，0 时 S = I</param>
/// <param name="scale_mode">0=关闭 1=仅轴向 2=各向异性</param>
/// <returns>M_j；R 为单位且两个权重为 0 时严格返回单位阵</returns>
Mat3 build_control_M(const Quat& R,
                     const Vec3& L0, const Vec3& L1, const Vec3& L2,
                     const Vec3& axis_bind_unit,
                     double twist_weight, double scale_weight, int scale_mode);

/// <summary>
/// 带部分主元的高斯消元求解 4×4 线性系统 A·x = b。
/// </summary>
/// <param name="A">4×4 系数矩阵（行主序，会被原地修改）</param>
/// <param name="b">右端项</param>
/// <param name="x">输出解</param>
/// <returns>false 表示主元过小（奇异），调用方须回退 A=I 路径</returns>
bool solve4x4(double A[4][4], const double b[4], double x[4]);

/// <summary>spline 参数 t → 分数弧长 τ ∈ [0,1]</summary>
double tau_from_t(const ArclenLUT& lut, double t);

/// <summary>分数弧长 τ → spline 参数 t ∈ [0,1]</summary>
double t_from_tau(const ArclenLUT& lut, double tau);

// ================================================================
// 体积补偿：专利 US 8,400,455 B2 的 space-unfolding 参数
// ================================================================

/// <summary>
/// 计算专利 Eq.1-10 的 space-unfolding 参数 tau。
/// 几何含义：把弯曲姿势在关节处「摊平」，得到顶点沿两段骨骼的归一化位置；
/// tau 与 bind 时的差值即为该顶点需要的沿轴补偿量。
/// </summary>
/// <param name="P">父关节世界位置（肩）</param>
/// <param name="O">中间关节世界位置（肘，即当前弯曲关节）</param>
/// <param name="Q">子关节世界位置（腕）</param>
/// <param name="R">待求值顶点的世界位置</param>
/// <returns>tau；两点重合或完全伸直时返回 1（表示无额外弯曲补偿）</returns>
double space_unfold_tau(const Vec3& P, const Vec3& O, const Vec3& Q, const Vec3& R);

/// <summary>
/// 逆距离平方权重：w_i = (1/|ref - cv_i|²) 归一化。
/// 用于决定每个控制点被 Tau 推拉的程度——越靠近关节，推拉量越大。
/// </summary>
/// <param name="ref">参考点，通常是弯曲关节 O</param>
/// <param name="cvs">控制点位置</param>
/// <returns>与 cvs 等长的归一化权重；某个 cv 与 ref 重合时该点独占权重 1</returns>
std::vector<double> inverse_distance_sq_weights(const Vec3& ref, const std::vector<Vec3>& cvs);

}  // namespace ocd_math