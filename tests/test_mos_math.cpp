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
    const Quat fixed{0, 0, 0, 1};
    const double pi = 3.14159265358979323846;
    const auto rightAngle = q_axis_angle(z, pi / 2.0);
    const auto guide = cor_contour::buildFrame({-3, 0, 0}, {}, {}, fixed, rightAngle);
    const auto inside = sampleTwoBoneGuide(guide, {0, 1, 0}, fixed, rightAngle, 3, {}, {}, {});
    const auto outside = sampleTwoBoneGuide(guide, {0, -1, 0}, fixed, rightAngle, 3, {}, {}, {});
    if (inside.coverage != 1.0 || outside.coverage != 1.0 ||
        inside.position.x < -1.05 || inside.position.y > 1.05 ||
        outside.position.x < 0.89 || outside.position.y > -0.89) {
        std::cerr << "Guide must preserve the inner fillet and outer corner\n";
        return 1;
    }
    for (const double sign : {-1.0, 1.0}) {
        const auto previous = sampleTwoBoneGuide(guide, {sign * 2, 1, 0},
            fixed, rightAngle, 3, {}, {}, {});
        const auto opposite = sampleTwoBoneGuide(guide, {sign * 2, -1, 0},
            fixed, rightAngle, 3, {}, {}, {});
        const auto expectedDiameter = sign < 0.0 ? Vec3{0, 2, 0} : Vec3{-2, 0, 0};
        expectNear(v3_sub(previous.position, opposite.position), expectedDiameter,
            "straight guide rings must share their axial coordinate");
    }
    auto outward = Secondary{};
    outward.bulgeOuter = 0.2;
    const auto fuller = sampleTwoBoneGuide(guide, {0, -1, 0},
        fixed, rightAngle, 3, outward, outward, {});
    if (fuller.position.x <= outside.position.x || fuller.position.y >= outside.position.y) return 1;
    expectNear(sampleTwoBoneGuide(guide, {0, 1, 0},
        fixed, rightAngle, 3, outward, outward, {}).position, inside.position,
        "guide outer bulge must not move the inner corner");
    auto axial = Secondary{};
    axial.slideInner = 0.2;
    const auto sliding = sampleTwoBoneGuide(guide, {-1.5, 1, 0},
        fixed, rightAngle, 3, axial, {}, {});
    const auto nominal = sampleTwoBoneGuide(guide, {-1.5, 1, 0},
        fixed, rightAngle, 3, {}, {}, {});
    if (sliding.position.x <= nominal.position.x || std::abs(sliding.position.y - nominal.position.y) > 1e-10)
        return 1;
    for (const double degrees : {45.0, 90.0, 120.0}) {
        const auto bent = q_axis_angle(z, pi * degrees / 180.0);
        const auto frame = cor_contour::buildFrame({-3, 0, 0}, {}, {}, fixed, bent);
        for (const double radial : {-1.0, 1.0}) {
            auto last = sampleTwoBoneGuide(frame, {-3, radial, 0}, fixed, bent, 3, {}, {}, {}).position;
            for (int ring = 1; ring <= 60; ++ring) {
                const Vec3 restPoint{-3.0 + 0.1 * ring, radial, 0};
                const auto point = sampleTwoBoneGuide(frame, restPoint, fixed, bent, 3, {}, {}, {});
                const auto surface = cor_contour::sampleSurface(frame, restPoint, fixed, true);
                if (v3_dot(v3_sub(point.position, last), surface.tangent) <= 0.0) {
                    std::cerr << "Guide rings must advance in order\n";
                    return 1;
                }
                last = point.position;
            }
        }
    }
    std::cout << "MOS bind, pose, bulge and slide checks passed\n";
}
