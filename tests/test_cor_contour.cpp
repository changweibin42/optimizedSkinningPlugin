#include "cor_contour.h"
#include "cor_paper.h"

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
    require(inner.x < -0.15 && inner.y > 0.15 && inner.z == 0, "inner direction at 90 degrees");
    require(outer.x > 0.15 && outer.y < -0.15 && outer.z == 0, "outer direction at 90 degrees");
    require(v3_len(correction(frame, inside, innerBase, cor.q, 1, 0, 1)) == 0, "disable inner side");
    require(v3_len(correction(frame, outside, outerBase, cor.q, 1, 1, 0)) == 0, "disable outer side");
    require(v3_len(correction(frame, inside, innerBase, cor.q, 0, 1, 1)) == 0, "rigid weights");
    require(v3_len(correction(frame, {-3, 1, 0}, innerBase, cor.q, 1, 1, 1)) == 0, "past axial support");
    require(v3_len(correction(frame, {0, 0, 1}, {0, 0, 1}, cor.q, 1, 1, 1)) == 0, "side seam");
    require(v3_len(correction(frame, inside, {-2, 2, 0}, cor.q, 1, 1, 1)) == 0, "full inner contour");
    require(v3_len(correction(frame, outside, {2, -2, 0}, cor.q, 1, 1, 1)) == 0, "full outer contour");

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
    std::cout << "CoR contour tests passed\n";
}
