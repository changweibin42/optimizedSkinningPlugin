#include "cor_paper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <optional>
#include <unordered_map>

namespace cor_paper {
namespace {

using namespace mathcore;
constexpr double subdivisionDistance = 0.1;

// The three weight lists, and their interpolants, use the same list of
// influence IDs within one input triangle. Densifying locally makes its
// recursive midpoint and error calculations independent of the rig size.
struct SampleVertex {
    Vec3 point;
    std::vector<double> weights;
    // 曲线归属与弧长（开启曲线引导时才有内容）。长度 = 曲线数。
    std::vector<double> curveMembership;
    std::vector<double> curveArc;
};

double weightDistanceSquared(const SampleVertex& a, const SampleVertex& b) {
    double squared = 0.0;
    for (size_t i = 0; i < a.weights.size(); ++i) {
        const double difference = a.weights[i] - b.weights[i];
        squared += difference * difference;
    }
    return squared;
}

struct PairSample {
    double firstWeight, secondWeight;
    double mass;
    Vec3 moment;
    // 采样点位置。曲线因子依赖「查询顶点 -> 采样点」的距离，所以必须
    // 在这里保留位置；仅仅保留 moment（等于 pos*mass）在乘上因子后不再够用。
    Vec3 position;
    // 采样点对每条曲线的归属权重（与 CurveBinding::membership 等长）。
    // 开启曲线引导时按它加权计算采样点的曲线参数。
    std::vector<double> curveMembership;
    // 采样点在各曲线上的弧长（与 membership 等长）。
    std::vector<double> curveArc;
};

using PairKey = std::uint64_t;
using PairSamples = std::unordered_map<PairKey, std::vector<PairSample>>;

PairKey pairKey(unsigned int first, unsigned int second) {
    return (static_cast<PairKey>(first) << 32) | second;
}

void accumulateTriangle(const std::array<SampleVertex, 3>& triangle,
    const std::vector<unsigned int>& influences, PairSamples& samples) {
    const auto& a = triangle[0];
    const auto& b = triangle[1];
    const auto& c = triangle[2];
    const double area = 0.5 * v3_len(v3_cross(v3_sub(b.point, a.point), v3_sub(c.point, a.point)));
    if (area == 0.0) return;
    const Vec3 centroid = v3_scale(v3_add(v3_add(a.point, b.point), c.point), 1.0 / 3.0);

    std::vector<double> means(influences.size());
    for (size_t i = 0; i < means.size(); ++i)
        means[i] = (a.weights[i] + b.weights[i] + c.weights[i]) / 3.0;

    // 曲线归属与弧长：沿三个父顶点线性插值。membership 插值后保持和为 1，
    // arc 只在各自 membership 非零处插值，避免把两条链的弧长混成一个。
    std::vector<double> curveMembership, curveArc;
    if (!a.curveMembership.empty()) {
        curveMembership.assign(a.curveMembership.size(), 0.0);
        curveArc.assign(a.curveMembership.size(), 0.0);
        for (size_t k = 0; k < curveMembership.size(); ++k) {
            curveMembership[k] = (a.curveMembership[k] + b.curveMembership[k] +
                c.curveMembership[k]) / 3.0;
            // 弧长按归属权重加权，权重为 0 的曲线不参与。
            double wsum = a.curveMembership[k] + b.curveMembership[k] +
                c.curveMembership[k];
            if (wsum > 1e-12) {
                curveArc[k] = (a.curveMembership[k] * a.curveArc[k] +
                    b.curveMembership[k] * b.curveArc[k] +
                    c.curveMembership[k] * c.curveArc[k]) / wsum;
            }
        }
    }

    for (size_t i = 0; i < means.size(); ++i) {
        if (means[i] == 0.0) continue;
        for (size_t j = i + 1; j < means.size(); ++j) {
            if (means[j] == 0.0) continue;
            const double mass = area * means[i] * means[j];
            samples[pairKey(influences[i], influences[j])].push_back({
                means[i], means[j], mass, v3_scale(centroid, mass), centroid,
                curveMembership, curveArc
            });
        }
    }
}

void subdivide(const std::array<SampleVertex, 3>& triangle,
    const std::vector<unsigned int>& influences, PairSamples& samples) {
    const double distances[] = {
        weightDistanceSquared(triangle[0], triangle[1]),
        weightDistanceSquared(triangle[1], triangle[2]),
        weightDistanceSquared(triangle[2], triangle[0])
    };
    const int first = static_cast<int>(std::max_element(distances, distances + 3) - distances);
    if (distances[first] <= subdivisionDistance * subdivisionDistance) {
        accumulateTriangle(triangle, influences, samples);
        return;
    }

    const int second = (first + 1) % 3;
    SampleVertex midpoint{v3_scale(v3_add(triangle[first].point, triangle[second].point), 0.5),
        triangle[first].weights};
    for (size_t i = 0; i < midpoint.weights.size(); ++i)
        midpoint.weights[i] = (midpoint.weights[i] + triangle[second].weights[i]) * 0.5;

    // 曲线归属与弧长也取中点平均值（与权重同样的插值规则）。
    if (!triangle[first].curveMembership.empty()) {
        const size_t n = triangle[first].curveMembership.size();
        midpoint.curveMembership.assign(n, 0.0);
        midpoint.curveArc.assign(n, 0.0);
        for (size_t k = 0; k < n; ++k) {
            midpoint.curveMembership[k] =
                (triangle[first].curveMembership[k] +
                 triangle[second].curveMembership[k]) * 0.5;
            const double wsum = triangle[first].curveMembership[k] +
                triangle[second].curveMembership[k];
            if (wsum > 1e-12) {
                midpoint.curveArc[k] =
                    (triangle[first].curveMembership[k] * triangle[first].curveArc[k] +
                     triangle[second].curveMembership[k] * triangle[second].curveArc[k]) /
                    wsum;
            }
        }
    }

    auto half = triangle;
    half[second] = midpoint;
    subdivide(half, influences, samples);
    half[second] = triangle[second];
    half[first] = std::move(midpoint);
    subdivide(half, influences, samples);
}

// curveGate: 可选的曲线关联度。第一个参数是查询顶点索引，第二个是采样点
// 位置，返回该样本对该查询顶点的关联度 g（∈[0,1]）。为空表示 g≡1。
using cor_paper::CurveGate;

std::optional<Vec3> integrate(const Weights& vertex, const PairSamples& samples,
    double sigma, unsigned int query, const CurveGate& curveGate) {
    Vec3 moment{0.0, 0.0, 0.0};
    double mass = 0.0;
    for (size_t i = 0; i < vertex.size(); ++i) {
        const auto [first, a] = vertex[i];
        for (size_t j = i + 1; j < vertex.size(); ++j) {
            const auto [second, b] = vertex[j];
            const auto found = samples.find(pairKey(first, second));
            if (found == samples.end()) continue;

            // Eq. (1) counts both pair orders. Their common factor of two
            // cancels in the quotient in Eq. (4).
            const double contribution = a * b;
            for (const PairSample& sample : found->second) {
                const double d = (a * sample.secondWeight - b * sample.firstWeight) / sigma;
                double similarity = contribution * std::exp(-d * d);
                if (curveGate) {
                    const CurveSample view{sample.position, &sample.curveMembership,
                        &sample.curveArc};
                    similarity *= curveGate(query, view);
                }
                if (similarity == 0.0) continue;
                mass += similarity * sample.mass;
                moment = v3_add(moment, v3_scale(sample.moment, similarity));
            }
        }
    }
    if (mass <= 0.0 || !std::isfinite(mass)) return std::nullopt;
    const Vec3 center{moment.x / mass, moment.y / mass, moment.z / mass};
    if (!std::isfinite(center.x) || !std::isfinite(center.y) || !std::isfinite(center.z))
        return std::nullopt;
    return center;
}

} // namespace

std::vector<mathcore::Vec3> computeCenters(const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<Weights>& weights, const std::vector<Triangle>& triangles, double sigma,
    const CurveGate& curveGate, const std::vector<CurveVertex>* curveVertices) {
    const bool useCurves = curveGate && curveVertices &&
        curveVertices->size() == restPoints.size();
    PairSamples samples;
    for (const auto& triangle : triangles) {
        std::vector<unsigned int> influences;
        for (unsigned int vertex : triangle)
            for (const auto& weight : weights[vertex])
                influences.push_back(weight.first);
        std::sort(influences.begin(), influences.end());
        influences.erase(std::unique(influences.begin(), influences.end()), influences.end());
        if (influences.size() < 2) continue;

        std::array<SampleVertex, 3> vertices;
        for (size_t i = 0; i < vertices.size(); ++i) {
            const unsigned int vertex = triangle[i];
            vertices[i] = {restPoints[vertex], std::vector<double>(influences.size()), {}, {}};
            for (const auto& [influence, weight] : weights[vertex]) {
                const size_t index = std::lower_bound(influences.begin(), influences.end(),
                    influence) - influences.begin();
                vertices[i].weights[index] = weight;
            }
            if (useCurves) {
                vertices[i].curveMembership = (*curveVertices)[vertex].membership;
                vertices[i].curveArc = (*curveVertices)[vertex].arc;
            }
        }
        subdivide(vertices, influences, samples);
    }

    // 开启曲线因子时，中心依赖查询顶点自身的位置，同权重组不能再共享结果，
    // 必须逐顶点积分。关闭时保持原来的分组路径，确保与既有行为逐位一致。
    std::vector<size_t> queries;
    for (size_t i = 0; i < weights.size(); ++i)
        if (weights[i].size() >= 2) queries.push_back(i);

    std::vector<Vec3> centers = restPoints;
    if (curveGate) {
        const auto count = static_cast<std::ptrdiff_t>(queries.size());
#pragma omp parallel for schedule(dynamic, 8)
        for (std::ptrdiff_t q = 0; q < count; ++q) {
            const size_t i = queries[q];
            const auto center = integrate(weights[i], samples, sigma,
                static_cast<unsigned int>(i), curveGate);
            if (center) centers[i] = *center;
        }
        return centers;
    }

    std::sort(queries.begin(), queries.end(),
        [&](size_t a, size_t b) { return weights[a] < weights[b]; });
    std::vector<size_t> groups;
    for (size_t i = 0; i < queries.size(); ++i)
        if (i == 0 || weights[queries[i]] != weights[queries[i - 1]]) groups.push_back(i);
    groups.push_back(queries.size());

    const PairSamples& querySamples = samples;
    const auto numGroups = static_cast<std::ptrdiff_t>(groups.size() - 1);
#pragma omp parallel for schedule(dynamic, 8)
    for (std::ptrdiff_t group = 0; group < numGroups; ++group) {
        const size_t begin = groups[group];
        const auto center = integrate(weights[queries[begin]], querySamples, sigma,
            static_cast<unsigned int>(queries[begin]), CurveGate());
        if (!center) continue;
        for (size_t i = begin; i < groups[group + 1]; ++i)
            centers[queries[i]] = *center;
    }
    return centers;
}

mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center, const Weights& weights,
    const std::vector<mathcore::RigidTransform>& bindToPose) {
    return vertexTransform(center, weights, weights, bindToPose);
}

mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center,
    const Weights& rotationWeights, const Weights& translationWeights,
    const std::vector<mathcore::RigidTransform>& bindToPose) {
    if (rotationWeights.empty() || translationWeights.empty())
        return {{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};

    Quat sum{0.0, 0.0, 0.0, 0.0};
    for (const auto& [influence, weight] : rotationWeights) {
        const auto& q = bindToPose[influence].q;
        // Resolve antipodality against the running quaternion sum.
        const double signedWeight = q_dot(sum, q) < 0.0 ? -weight : weight;
        sum = q_add(sum, q_scale(q, signedWeight));
    }
    Vec3 posedCenter{0.0, 0.0, 0.0};
    for (const auto& [influence, weight] : translationWeights)
        posedCenter = v3_add(posedCenter, v3_scale(apply_transform(bindToPose[influence], center), weight));
    const Quat rotation = q_normalize(sum);
    return {rotation, v3_sub(posedCenter, q_rotate_v3(rotation, center))};
}

} // namespace cor_paper
