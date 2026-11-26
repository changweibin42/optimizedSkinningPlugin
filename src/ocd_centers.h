#pragma once
#include "math_core.h"
#include "ocd_math.h"

#include <array>
#include <vector>

// ================================================================
// 局部结构 CoR 中心求解：scripts/ocd_elbow_reference.py 的
// structural_centers / center_response / outer_participation 的 C++ 移植。
//
// 目标：在 bind 空间求每个顶点的 CoR 中心偏移量 delta，使变形后的
// 顶点位置尽量贴合 guide（OCD 铰链位移）所提供的目标形状，同时
//   · 保持 bind 网格边向量（局部结构）不被拉断
//   · 用方向性边长权重避免压扁环向
// 求解用 PCG，约束是逐顶点的 3×3 块对角预条件。
// ================================================================
namespace ocd_centers {

using ocd_math::Mat3;
using mathcore::Quat;
using mathcore::Vec3;

/// 单条骨骼对某个顶点的影响：权重 + 相对 bind 的旋转。
struct Influence {
    unsigned int bone = 0;
    double weight = 0.0;
};

/// bind 网格边：索引、rest 向量、刚度（面积 / 边长²）。
struct Edges {
    std::vector<unsigned int> a;
    std::vector<unsigned int> b;
    std::vector<Vec3> rest;
    std::vector<double> stiffness;
};

/// 由四边形/三角形面构造边表，权重为「相邻面积 / 边长²」。
Edges buildEdges(const std::vector<Vec3>& points,
    const std::vector<std::vector<unsigned int>>& faces);

/// 凸侧参与度：弯曲外侧顶点为 1，内侧为 0，连续过渡。
/// rest 为 bind 位置，weights 为三骨骼权重（parent/child/helper）。
std::vector<double> outerParticipation(const std::vector<Vec3>& rest,
    const std::vector<std::array<double, 3>>& weights,
    double bendDegrees, double twistDegrees, double directionDegrees);

struct Settings {
    std::vector<Quat> rotations;   // 每骨骼相对 bind 的旋转（索引与 weights 列对应）
    double centerLimit = 16.0;     // 中心偏移上限（世界单位），<=0 表示不限
    int maxIterations = 100;
};

/// 求解逐顶点中心偏移量。
/// weights[i] 为第 i 个顶点在 rotations 各列上的权重（未归一化时内部归一化）。
std::vector<Vec3> structuralCenters(const Edges& edges,
    const std::vector<Vec3>& original,
    const std::vector<std::array<double, 3>>& weights,
    const Settings& settings,
    const std::vector<Vec3>& guide,
    const std::vector<double>& outer);

} // namespace ocd_centers
