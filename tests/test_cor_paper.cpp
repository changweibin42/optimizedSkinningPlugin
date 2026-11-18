#include "cor_paper.h"

#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace mathcore;
using namespace cor_paper;

namespace {

void expectNear(double actual, double expected, const char* description) {
    if (!std::isfinite(actual) || std::abs(actual - expected) > 1e-11) {
        std::cerr << description << ": expected " << expected << ", got " << actual << '\n';
        std::exit(1);
    }
}

void expectPoint(const Vec3& actual, const Vec3& expected, const char* description) {
    expectNear(actual.x, expected.x, description);
    expectNear(actual.y, expected.y, description);
    expectNear(actual.z, expected.z, description);
}

void integrateDisconnectedTriangles() {
    const std::vector<Vec3> points{{0, 0, 0}, {2, 0, 0}, {0, 2, 0},
        {500, 0, 0}, {501, 0, 0}, {500, 1, 0}, {8, 9, 10}, {-1, 0, 0}};
    const Weights paired{{3, 0.5}, {7, 0.5}};
    const std::vector<Weights> weights{paired, paired, paired, paired, paired, paired, {{3, 1}}, {}};
    const auto centers = computeCenters(points, weights, {{0, 1, 2}, {3, 4, 5}});
    const Vec3 expected = v3_scale(v3_add(v3_scale({2.0 / 3, 2.0 / 3, 0}, 2),
        v3_scale({500 + 1.0 / 3, 1.0 / 3, 0}, 0.5)), 1.0 / 2.5);
    for (size_t i = 0; i < 6; ++i)
        expectPoint(centers[i], expected, "area integral across shells");
    expectPoint(centers[6], points[6], "single influence center");
    expectPoint(centers[7], points[7], "zero influence center");
}

void integrateOneBisectedEdge() {
    const std::vector<Vec3> points{{0, 0, 0}, {2, 0, 0}, {0, 2, 0}};
    const std::vector<Weights> weights{{{0, 0.46}, {1, 0.54}},
        {{0, 0.54}, {1, 0.46}}, {{0, 0.5}, {1, 0.5}}};

    // Only edge AB exceeds 0.1. The two halves have equal area.
    const double sigma = 0.03;
    const double leftMean = (0.46 + 0.5 + 0.5) / 3;
    const double rightMean = (0.54 + 0.5 + 0.5) / 3;
    const auto similarity = [&](double x) {
        const double d = (0.46 - x) / sigma;
        return x * (1 - x) * std::exp(-d * d);
    };
    const double left = similarity(leftMean);
    const double right = similarity(rightMean);
    const auto centers = computeCenters(points, weights, {{0, 1, 2}}, sigma);
    expectPoint(centers[0], {(left / 3 + right) / (left + right), 2.0 / 3, 0},
        "weighted quadrature after bisection");
    expectNear(centers[1].x + centers[0].x, 4.0 / 3, "symmetric bisection");
}

void separateRotationAndTranslation() {
    const Vec3 center{3, 4, 5};
    const double sine = std::sin(1.4);
    const double cosine = std::cos(1.4);
    const std::vector<RigidTransform> transforms{{{0, 0, 0, 1}, {1, 2, 3}},
        {{sine, 0, 0, cosine}, {0, 2, -1}}, {{-sine, 0, 0, cosine}, {-2, 4, 0}}};
    const Weights weights{{0, 0.1}, {1, 0.5}, {2, 0.4}};
    const auto result = vertexTransform(center, weights, transforms);
    const Quat expected = q_normalize({0.9 * sine, 0, 0, 0.1 + 0.1 * cosine});
    expectNear(result.q.x, expected.x, "running quaternion sign");
    expectNear(result.q.w, expected.w, "running quaternion normalization");

    Vec3 posedCenter{0, 0, 0};
    for (const auto& [bone, weight] : weights)
        posedCenter = v3_add(posedCenter, v3_scale(apply_transform(transforms[bone], center), weight));
    expectPoint(apply_transform(result, center), posedCenter, "translation maps center to LBS center");

    const Vec3 vertex{10, 20, 30};
    expectPoint(apply_transform(vertexTransform(center, {{1, 1}}, transforms), vertex),
        apply_transform(transforms[1], vertex), "single influence runtime");
    expectPoint(apply_transform(vertexTransform(center, {}, transforms), vertex), vertex,
        "zero influence runtime");
}

} // namespace

int main() {
    integrateDisconnectedTriangles();
    integrateOneBisectedEdge();
    separateRotationAndTranslation();
    std::cout << "CoR paper regression checks passed\n";
}
