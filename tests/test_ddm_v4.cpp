#include "ddm_v4.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace mathcore;
using namespace ddm_v4;

namespace {

void check(bool condition, const char* label) {
    if (!condition) {
        std::cerr << label << '\n';
        std::exit(1);
    }
}

void near(double actual, double expected, const char* label) {
    check(std::isfinite(actual) && std::abs(actual - expected) < 1e-10, label);
}

double boneWeight(const cor_paper::Weights& weights, unsigned int bone) {
    for (const auto& [id, weight] : weights)
        if (id == bone) return weight;
    return 0.0;
}

} // namespace

int main() {
    const std::vector<Vec3> points{{0, 0, 0}, {2, 0, 0}, {4, 0, 0},
        {50, 0, 0}, {52, 0, 0}, {54, 0, 0}};
    const std::vector<cor_paper::Weights> weights{{{0, 1}}, {{1, 1}}, {{1, 1}},
        {}, {{1, 1}}, {{1, 1}}};
    const std::vector<Edge> edges{{0, 1}, {1, 2}, {0, 1}, {2, 3}, {3, 4}, {4, 5}};
    BindData data;
    check(precompute(points, weights, 2, edges, {1, 1, 0, 0.75}, data), "bind failed");

    // Exact inverse of I+L_normalized for the three-node path.
    near(data.centers[0].x, 1.0, "smoothed endpoints");
    near(data.centers[1].x, 2.0, "smoothed middle");
    near(data.centers[2].x, 3.0, "smoothed endpoints");
    near(data.centers[3].x, 50.0, "unweighted point stays fixed");
    near(boneWeight(data.rotationWeights[0], 0), 7.0 / 12, "first rotation weight");
    near(boneWeight(data.rotationWeights[1], 0), 1.0 / 6, "middle rotation weight");
    near(boneWeight(data.rotationWeights[2], 0), 1.0 / 12, "last rotation weight");
    near(boneWeight(data.translationWeights[2], 0), 1.0 / 48, "independent translation weight");
    near(boneWeight(data.rotationWeights[4], 0), 0.0, "no diffusion across unweighted vertices");
    check(data.rotationWeights[3].empty() && data.translationWeights[3].empty(),
        "unweighted vertex acquired weights");

    const Quat quarterTurn = q_axis_angle({0, 0, 1}, std::acos(-1.0) / 2.0);
    const std::vector<RigidTransform> rest(2, {{0, 0, 0, 1}, {0, 0, 0}});
    const std::vector<RigidTransform> pose{{{0, 0, 0, 1}, {0, 0, 0}}, {quarterTurn, {1, 2, 0}}};
    const auto& psi = data.rotationWeights[2];
    const auto& omega = data.translationWeights[2];
    const Vec3 center = data.centers[2];
    const auto atRest = cor_paper::vertexTransform(center, psi, omega, rest);
    near(v3_len(v3_sub(apply_transform(atRest, points[2]), points[2])), 0.0, "bind pose drift");
    const auto split = cor_paper::vertexTransform(center, psi, omega, pose);
    const auto combined = cor_paper::vertexTransform(center, psi, pose);
    near(split.q.w, combined.q.w, "translation weights affect rotation");
    check(v3_len(v3_sub(split.t, combined.t)) > 0.01, "translation did not split");
    const auto identical = cor_paper::vertexTransform(center, psi, psi, pose);
    near(v3_len(v3_sub(combined.t, identical.t)), 0.0, "CoR equivalence translation");
    near(q_dot(combined.q, identical.q), 1.0, "CoR equivalence rotation");

    check(precompute(points, weights, 2, edges, {2, 1, 1, 0.9}, data), "equal diffusion bind failed");
    check(data.translationWeights.empty(), "equal controls should reuse rotation weights");
    check(precompute(points, weights, 2, edges, {0, 20, 1, 0.9}, data), "zero iterations bind failed");
    check(data.rotationWeights == weights, "zero-iteration weights changed");
    near(data.centers[0].x, points[0].x, "zero-iteration center changed");
    std::cout << "DDM v4 regression checks passed\n";
}
