#include "mos_math.h"

#include <cstdlib>
#include <iostream>

using namespace mos_math;

void expectNear(const Vec3& actual, const Vec3& expected, const char* reason) {
    if (!(v3_len(v3_sub(actual, expected)) < 1e-9)) {
        std::cerr << reason << '\n';
        std::exit(1);
    }
}

int main() {
    const std::vector<Vec3> points{{-2, 0, 0}, {-1, 0.2, 0}, {-0.4, 0, 0},
        {0.4, 0, 0}, {1, -0.2, 0}, {2, 0, 0}};
    const auto knots = ocd_math::make_clamped_knots(6, 3);
    const auto z = Vec3{0, 0, 1};
    std::vector<ocd_math::Mat3> frames;
    for (int i = 0; i < 6; ++i)
        frames.push_back(ocd_math::mat3_from_quat(q_axis_angle(z, 0.1 * i)));
    const auto global = q_axis_angle({1, 0, 0}, 0.7);
    const auto rotate = ocd_math::mat3_from_quat(global);
    const Vec3 shift{2, 4, 8};
    for (int i = 0; i <= 100; ++i) {
        const double t = i / 100.0;
        const auto basis = sampleBasis(knots, 3, t);
        Vec3 center{};
        for (int k = 0; k < basis.count; ++k)
            center = v3_add(center, v3_scale(points[basis.first + k], basis.weights[k]));
        expectNear(center, ocd_math::bspline_eval(points, knots, 3, t), "cubic basis agrees with de Boor");

        const Vec3 vertex{2.0 * t - 1.0, 0.7, 0.9};
        Vec3 offset;
        if (!bindOffset(vertex, points, frames, basis, offset)) return 1;
        Vec3 rest{}, deformed{};
        for (int k = 0; k < basis.count; ++k) {
            const int j = basis.first + k;
            rest = v3_add(rest, v3_scale(sampleControl(points[j], frames[j], offset, {}, {}), basis.weights[k]));
            const auto posedPoint = v3_add(rotate.apply(points[j]), shift);
            deformed = v3_add(deformed, v3_scale(sampleControl(posedPoint, rotate * frames[j],
                offset, {}, {}), basis.weights[k]));
        }
        expectNear(rest, vertex, "Eq. 9 must reconstruct bind vertices");
        expectNear(deformed, v3_add(rotate.apply(vertex), shift), "Eq. 5 must carry rigid poses");
    }

    const auto identity = ocd_math::Mat3::identity();
    BendControl bend{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}, 0.5, 2.0, true};
    Secondary settings;
    settings.bulgeInner = 0.2;
    expectNear(sampleControl({}, identity, {0, 2, 0}, bend, settings),
        {0, 2.2, 0}, "bulge scales the radial sample");
    expectNear(sampleControl({}, identity, {0, -2, 0}, bend, settings),
        {0, -2, 0}, "inner bulge must not reach the outer meridian");
    settings = {};
    settings.slideInner = 0.2;
    expectNear(sampleControl({}, identity, {0, 2, 0}, bend, settings),
        {0.2, 2, 0}, "incoming inner slide is axial and toward the hinge");
    bend.incoming = false;
    expectNear(sampleControl({}, identity, {0, 2, 0}, bend, settings),
        {-0.2, 2, 0}, "outgoing inner slide is axial and toward the hinge");
    settings = {};
    settings.slideOuter = 0.2;
    expectNear(sampleControl({}, identity, {0, -2, 0}, bend, settings),
        {0.2, -2, 0}, "outgoing outer slide goes away from the hinge");
    settings = {};
    settings.bulgeOuter = 0.2;
    expectNear(sampleControl({}, identity, {0, -2, 0}, bend, settings),
        {0, -2.2, 0}, "outer bulge scales the outer sample");
    expectNear(sampleControl({}, identity, {0, 3, 0}, bend, settings),
        {0, 3, 0}, "outer bulge must not reach the inner meridian");
    settings = {};
    settings.slideInner = 0.2;
    bend.response = 0.0;
    expectNear(sampleControl({}, identity, {0, 2, 0}, bend, settings),
        {0, 2, 0}, "secondary operations vanish at rest");
    std::cout << "MOS bind, pose, bulge and slide checks passed\n";
}
