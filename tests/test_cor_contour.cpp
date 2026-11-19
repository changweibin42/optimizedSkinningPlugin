#include "cor_contour.h"
#include "cor_paper.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>

using namespace mathcore;
using namespace cor_contour;

namespace {

constexpr double pi = 3.14159265358979323846;
const Quat identity{0, 0, 0, 1};
const Vec3 pivot{0, 0, 0};
const Vec3 parent{-2, 0, 0};

void require(bool condition, const char* description) {
    if (!condition) {
        std::cerr << description << '\n';
        std::exit(1);
    }
}

void checkBend() {
    const Vec3 inside{0, 1, 0}, outside{0, -1, 0};
    const auto child = q_axis_angle({0, 0, 1}, pi / 2);
    const auto frame = buildFrame(parent, pivot, pivot, identity, child);
    // The middle ring of a straight, symmetrically weighted cylinder has its center on the axis.
    const auto cor = cor_paper::vertexTransform(pivot, {{0, 0.5}, {1, 0.5}},
        {{identity, pivot}, {child, pivot}});
    const auto innerBase = apply_transform(cor, inside);
    const auto outerBase = apply_transform(cor, outside);
    const auto inner = correction(frame, inside, innerBase, cor.q, 1, 1, 1);
    const auto outer = correction(frame, outside, outerBase, cor.q, 1, 1, 1);
    require(inner.x < -0.2 && inner.y > 0.2 && inner.z == 0, "inner direction at 90 degrees");
    require(outer.x > 0.05 && outer.y < -0.05 && outer.z == 0, "outer direction at 90 degrees");
    require(v3_len(correction(frame, inside, innerBase, cor.q, 1, 0, 1)) < 1e-12, "disable inner side");
    require(v3_len(correction(frame, outside, outerBase, cor.q, 1, 1, 0)) < 1e-12, "disable outer side");
    require(v3_len(correction(frame, inside, innerBase, cor.q, 0, 1, 1)) == 0, "rigid weights");
    require(v3_len(correction(frame, {-3, 1, 0}, innerBase, cor.q, 1, 1, 1)) == 0, "past axial support");
    const auto halfAngle = q_axis_angle({0, 0, 1}, pi / 4);
    const auto halfway = buildFrame(parent, pivot, pivot, identity, halfAngle);
    const auto quarter = q_axis_angle({0, 0, 1}, pi / 8);
    const auto halfDelta = correction(halfway, inside, q_rotate_v3(quarter, inside), quarter, 1, 1, 1);
    require(v3_len(halfDelta) > 0 && v3_len(halfDelta) < v3_len(inner), "45 degree transition");

    const auto reverseChild = q_axis_angle({0, 0, 1}, -pi / 2);
    const auto reverseBlend = q_axis_angle({0, 0, 1}, -pi / 4);
    const auto reverseFrame = buildFrame(parent, pivot, pivot, identity, reverseChild);
    const auto reversed = correction(reverseFrame, outside, q_rotate_v3(reverseBlend, outside),
        reverseBlend, 1, 1, 1);
    require(std::abs(reversed.x - inner.x) < 1e-10 && std::abs(reversed.y + inner.y) < 1e-10,
        "reversing the bend swaps sides");

    std::cout << "90 degree midpoint contour displacement, rest radius = 1: inner="
        << v3_len(inner) << " outer=" << v3_len(outer) << '\n';
}

// Test the full CoR + contour path on a triangulated tube. A midpoint-only
// check misses the folded rings seen next to the inner corner in Maya.
void checkTube(double transitionWidth, double angle) {
    constexpr int rings = 61;
    constexpr int sides = 24;
    constexpr double spacing = 0.1;
    std::vector<Vec3> rest;
    std::vector<cor_paper::Weights> weights;
    std::vector<cor_paper::Triangle> triangles;
    for (int ring = 0; ring < rings; ++ring) {
        const double x = (ring - rings / 2) * spacing;
        const double t = std::clamp(0.5 + 0.5 * x / transitionWidth, 0.0, 1.0);
        const double w = t * t * (3.0 - 2.0 * t);
        for (int side = 0; side < sides; ++side) {
            const double phi = 2.0 * pi * side / sides;
            rest.push_back({x, std::cos(phi), std::sin(phi)});
            if (w == 0.0) weights.push_back({{0, 1.0}});
            else if (w == 1.0) weights.push_back({{1, 1.0}});
            else weights.push_back({{0, 1.0 - w}, {1, w}});
        }
        if (ring == 0) continue;
        for (int side = 0; side < sides; ++side) {
            const unsigned int a = (ring - 1) * sides + side;
            const unsigned int b = (ring - 1) * sides + (side + 1) % sides;
            triangles.push_back({a, b, a + sides});
            triangles.push_back({b, b + sides, a + sides});
        }
    }

    const auto centers = cor_paper::computeCenters(rest, weights, triangles);
    const auto child = q_axis_angle({0, 0, 1}, angle);
    const auto frame = buildFrame(parent, pivot, pivot, identity, child);
    const std::vector<RigidTransform> bones{{identity, pivot}, {child, pivot}};
    std::vector<Vec3> posed;
    for (size_t i = 0; i < rest.size(); ++i) {
        const auto cor = cor_paper::vertexTransform(centers[i], weights[i], bones);
        const auto base = apply_transform(cor, rest[i]);
        posed.push_back(v3_add(base, correction(frame, rest[i], base, cor.q, 1, 1, 1)));
    }

    if (angle == pi / 2) {
        const auto& outsideCorner = posed[(rings / 2) * sides + sides / 2];
        require(outsideCorner.x > 0.89 && outsideCorner.y < -0.89,
            "outer corner must be fuller while keeping rounded clearance");
        for (int ring = 0; ring < rings; ++ring) {
            const double x = (ring - rings / 2) * spacing;
            if (std::abs(x) > 1.3) continue;
            for (int side = 0; side < sides; ++side) {
                const auto& point = posed[ring * sides + side];
                require(point.x <= 1.0 + 1e-10 && point.y >= -1.0 - 1e-10,
                    "outer curve must stay within both straight side bounds");
            }
        }
    }

    double minimum = spacing;
    for (int ring = 0; ring < rings - 1; ++ring) {
        const double along = (ring + 0.5 - rings / 2) * spacing;
        const double halfSpan = 1.3 * std::tan(angle * 0.5);
        const double phase = angle * std::clamp(0.5 + 0.5 * along / halfSpan, 0.0, 1.0);
        const Vec3 tangent{std::cos(phase), std::sin(phase), 0};
        for (int side = 0; side < sides; ++side) {
            const int at = ring * sides + side;
            minimum = std::min(minimum, v3_dot(v3_sub(posed[at + sides], posed[at]), tangent));
        }
    }
    std::cout << "Tube angle=" << angle * 180.0 / pi << " weight half-width="
        << transitionWidth << " minimum axial spacing=" << minimum << '\n';
    require(minimum > 0.0, "adjacent rings must advance along the bend");
    for (const auto& triangle : triangles) {
        const auto a = triangle[0], b = triangle[1], c = triangle[2];
        const auto restCenter = v3_scale(v3_add(rest[a], v3_add(rest[b], rest[c])), 1.0 / 3.0);
        const double halfSpan = 1.3 * std::tan(angle * 0.5);
        const double phase = angle * std::clamp(0.5 + 0.5 * restCenter.x / halfSpan, 0.0, 1.0);
        const Vec3 outward{-std::sin(phase) * restCenter.y, std::cos(phase) * restCenter.y, restCenter.z};
        const auto normal = v3_cross(v3_sub(posed[b], posed[a]), v3_sub(posed[c], posed[a]));
        require(v3_dot(normal, outward) > 0.0, "tube triangle must keep outward orientation");
    }
}

void checkNoBend() {
    const auto twist = q_axis_angle({1, 0, 0}, pi / 2);
    const auto rigid = q_axis_angle({0, 0, 1}, pi / 3);
    require(buildFrame(parent, pivot, pivot, identity, identity).activation == 0, "bind pose");
    require(buildFrame(parent, pivot, pivot, identity, twist).activation == 0, "pure twist");
    require(buildFrame(parent, pivot, {1, 2, 3}, rigid, rigid).activation == 0, "common rigid motion");
}

} // namespace

int main() {
    checkBend();
    checkNoBend();
    for (const double width : {0.75, 1.5, 2.5}) {
        checkTube(width, pi / 4);
        checkTube(width, pi / 2);
    }
    std::cout << "CoR contour tests passed\n";
}
