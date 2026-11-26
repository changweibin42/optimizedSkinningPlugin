#pragma once
#include "math_core.h"
#include <array>
#include <vector>

// ================================================================
// OCD 铰链几何：scripts/ocd_elbow_reference.py 的 C++ 移植
//
// 参考实现把肢体建模成"半径 = 1、沿轴向 ±HALF"的圆柱，采用
// 归一化单位。插件里顶点在任意网格上，故统一按肢体半径 R 归一：
//   station_u = station / R        （轴向坐标，单位 R）
//   (y, z)    = 径向偏移 / R       （归一化截面坐标，[-1, 1]）
// 输出再乘回 R。R 取整条肢体截面最大半径，与参考实现的 radius 一致。
// ================================================================
namespace ocd_hinge {

using mathcore::Vec3;
using mathcore::Quat;

inline constexpr double halfSpan = 6.0;    // 参考实现的 HALF
inline constexpr int halfRings = 9;        // 参考实现的 HALF_RINGS
inline constexpr double ringStep = halfSpan / halfRings;
inline constexpr double reach = 5.1;       // 参考实现的 REACH

// 参考实现的 OUTER / INNER 偏移表（单位：肢体半径）
extern const double outerOffsets[halfRings + 1];
extern const double innerOffsets[halfRings + 1];
extern const double innerClearOffsets[halfRings + 1];

struct Pose {
    double bend = 0.0;       // 关节弯曲角，度，带符号，|bend| <= 150
    double twist = 0.0;      // 轴向扭转，度
    double direction = 0.0;  // 弯曲平面方向，度
    bool avoidOverlap = false;  // 旧版内侧避让
    // 11 个补偿通道，单位见参考实现的 VIDEO 表；全零表示不做次级编辑。
    std::array<double, 11> values{};
};

struct Rail {
    double x = 0.0;
    double y = 0.0;
};

struct Section {
    double ox = 0.0;  // 外轨 x
    double oy = 0.0;  // 外轨 y
    double rx = 0.0;  // 半宽
    double ry = 0.0;  // 半高
};

/// 一次姿态求值所需的常量，避免每个顶点重复三角函数。
class Elbow {
public:
    explicit Elbow(const Pose& pose);

    Rail rail(double station, const double offsets[halfRings + 1], int pole) const;
    Section section(double station) const;

    /// station：轴向坐标（单位：肢体半径，参考实现里就是 s）；
    /// y：归一化截面径向坐标（[-1,1]）；z：归一化截面侧向坐标。
    Vec3 point(double station, double y, double z) const;

    double bendRadians() const { return bend_; }
    double halfRadians() const { return h_; }
    double cosHalf() const { return ch_; }
    double sinHalf() const { return sh_; }

    /// 顶点 twist 沿链的衰减：参考实现的 skin_weight(station) 乘 twist。
    double twistSkin(double station) const;

    /// 参考实现的 shoulder(distance, peak, end)：中心与刚性端为零。
    double shoulder(double distance, double peak, double end) const;

private:
    double cp_ = 1.0, sp_ = 0.0;   // cos/sin(plane)
    double cb_ = 1.0, sb_ = 0.0;   // cos/sin(beta)
    double h_ = 0.0, ch_ = 1.0, sh_ = 0.0;
    double bend_ = 0.0;
    double fraction_ = 0.0;
    double tipY_ = 1.0, tipX_ = 0.0;
    double innerReach_ = 3.15;
    double tangentX_ = 0.0;
    double innerClearance_ = 1.0;
    double twistRadians_ = 0.0;
    bool avoidOverlap_ = false;
    bool edited_ = false;
    double response_ = 0.0;
    std::array<double, 11> values_{};
    static constexpr double inTravelGain_ = 0.37674;
    static constexpr double inTravelPeak_ = 0.30413;
    static constexpr double inTravelEnd_ = 4.9902;
};

// ================================================================
// 局部结构 CoR：hinge_section / hinge_displacement / structural_centers
// ================================================================

struct HingeParams {
    double bend = 0.0;       // 度，带符号
    double twist = 0.0;      // 度
    double direction = 0.0;  // 度
    double radius = 1.0;     // 肢体半径
    // 11 个补偿通道（Center Slide In/Out、Slide In/Out Prev/Next、
    // Ortho In/Out Prev/Next、Bulge），单位与参考实现一致。
    std::array<double, 11> values{};
};

/// hinge_section：给定轴向坐标与厚度坐标，返回弯曲平面内的 (x, h)。
struct HingePoint {
    double x = 0.0;
    double h = 0.0;
};

HingePoint hingeSection(double station, double depth, double radius,
    double weightParent, double weightChild, double weightHelper,
    double angle, double support);

/// 由绑定顶点与骨骼权重求 support（参考实现里的全局量）。
/// stations/depths 为归一化前轴向/厚度坐标；mixed 表示该顶点受多骨骼影响。
double hingeSupport(const std::vector<double>& stations,
    const std::vector<double>& distances,
    const std::vector<char>& mixed, double radius, double angle);

/// hinge_displacement：返回每个顶点的位移（目标 - 当前位姿）。
struct HingeWeight {
    double parent = 1.0;
    double child = 0.0;
    double helper = 0.0;
};

std::vector<Vec3> hingeDisplacement(
    const std::vector<double>& stations,
    const std::vector<double>& depths,
    const std::vector<double>& laterals,
    const std::vector<Vec3>& current,
    const std::vector<HingeWeight>& weights,
    const HingeParams& params);

} // namespace ocd_hinge
