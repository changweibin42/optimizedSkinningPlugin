#include "cor_paper.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
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

    for (size_t i = 0; i < means.size(); ++i) {
        if (means[i] == 0.0) continue;
        for (size_t j = i + 1; j < means.size(); ++j) {
            if (means[j] == 0.0) continue;
            const double mass = area * means[i] * means[j];
            samples[pairKey(influences[i], influences[j])].push_back({
                means[i], means[j], mass, v3_scale(centroid, mass)
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

    auto half = triangle;
    half[second] = midpoint;
    subdivide(half, influences, samples);
    half[second] = triangle[second];
    half[first] = std::move(midpoint);
    subdivide(half, influences, samples);
}

std::optional<Vec3> integrate(const Weights& vertex, const PairSamples& samples, double sigma) {
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
                const double similarity = contribution * std::exp(-d * d);
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
    const std::vector<Weights>& weights, const std::vector<Triangle>& triangles, double sigma) {
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
            vertices[i] = {restPoints[vertex], std::vector<double>(influences.size())};
            for (const auto& [influence, weight] : weights[vertex]) {
                const size_t index = std::lower_bound(influences.begin(), influences.end(),
                    influence) - influences.begin();
                vertices[i].weights[index] = weight;
            }
        }
        subdivide(vertices, influences, samples);
    }

    // Equal weights have the same integral, including across disjoint shells.
    // A vertex without a valid integral still falls back to its own position.
    std::vector<size_t> queries;
    for (size_t i = 0; i < weights.size(); ++i)
        if (weights[i].size() >= 2) queries.push_back(i);
    std::sort(queries.begin(), queries.end(),
        [&](size_t a, size_t b) { return weights[a] < weights[b]; });
    std::vector<size_t> groups;
    for (size_t i = 0; i < queries.size(); ++i)
        if (i == 0 || weights[queries[i]] != weights[queries[i - 1]]) groups.push_back(i);
    groups.push_back(queries.size());

    std::vector<Vec3> centers = restPoints;
    const PairSamples& querySamples = samples;
    const auto numGroups = static_cast<std::ptrdiff_t>(groups.size() - 1);
#pragma omp parallel for schedule(dynamic, 8)
    for (std::ptrdiff_t group = 0; group < numGroups; ++group) {
        const size_t begin = groups[group];
        const auto center = integrate(weights[queries[begin]], querySamples, sigma);
        if (!center) continue;
        for (size_t i = begin; i < groups[group + 1]; ++i)
            centers[queries[i]] = *center;
    }
    return centers;
}

mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center, const Weights& weights,
    const std::vector<mathcore::RigidTransform>& bindToPose) {
    if (weights.empty()) return {{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};

    Quat sum{0.0, 0.0, 0.0, 0.0};
    Vec3 posedCenter{0.0, 0.0, 0.0};
    for (const auto& [influence, weight] : weights) {
        const auto& transform = bindToPose[influence];
        // Algorithm 1, line 5: apply the sign correction to the running
        // quaternion sum, in influence order.
        const double signedWeight = q_dot(sum, transform.q) < 0.0 ? -weight : weight;
        sum = q_add(sum, q_scale(transform.q, signedWeight));
        posedCenter = v3_add(posedCenter, v3_scale(apply_transform(transform, center), weight));
    }
    const Quat rotation = q_normalize(sum);
    return {rotation, v3_sub(posedCenter, q_rotate_v3(rotation, center))};
}

} // namespace cor_paper
