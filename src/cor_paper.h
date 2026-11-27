#pragma once

#include <array>
#include <functional>
#include <utility>
#include <vector>

#include "math_core.h"

namespace cor_paper {

using Weights = std::vector<std::pair<unsigned int, double>>;
using Triangle = std::array<unsigned int, 3>;

// 曲线关联度：给定查询顶点索引与该采样点的曲线归属信息，返回关联度 g∈[0,1]。
// 空（默认）表示 g≡1，即原始 CoR 行为。
struct CurveSample {
    mathcore::Vec3 position{0.0, 0.0, 0.0};
    // 采样点对各曲线的归属权重与弧长，与 CurveBinding 的顺序一致。
    const std::vector<double>* membership = nullptr;
    const std::vector<double>* arc = nullptr;
};
using CurveGate = std::function<double(unsigned int, const CurveSample&)>;

// 每个顶点的曲线归属与弧长（与 CorGate 曲线数等长）。
struct CurveVertex {
    std::vector<double> membership;
    std::vector<double> arc;
};

// Weight lists are normalized and ordered by increasing influence index.
//
// curveGate 为空时，逐位复现原实现（含按权重分组共享中心的优化）。
// 非空时中心依赖查询顶点位置，改为逐顶点积分；curveVertices 提供每个
// 顶点的曲线归属，参与三角形细分的样本会沿父顶点插值得到。
std::vector<mathcore::Vec3> computeCenters(
    const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<Weights>& weights,
    const std::vector<Triangle>& triangles,
    double sigma = 0.1,
    const CurveGate& curveGate = CurveGate(),
    const std::vector<CurveVertex>* curveVertices = nullptr);

// The center stays in bind space. The returned transform acts in the
// coordinate system of bindToPose and separates rotation from translation.
mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center,
    const Weights& weights,
    const std::vector<mathcore::RigidTransform>& bindToPose);

mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center,
    const Weights& rotationWeights, const Weights& translationWeights,
    const std::vector<mathcore::RigidTransform>& bindToPose);

} // namespace cor_paper
