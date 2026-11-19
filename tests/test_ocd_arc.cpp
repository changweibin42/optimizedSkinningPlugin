#include "ocd_arc.h"

#include <cstdlib>
#include <iostream>

using namespace ocd_arc;

namespace {

constexpr double pi = 3.14159265358979323846;
const Vec3 xAxis{1, 0, 0};
const Quat identity{0, 0, 0, 1};

void require(bool condition, const char* description) {
    if (!condition) {
        std::cerr << description << '\n';
        std::exit(1);
    }
}

Vec3 evaluate(const Pose& pose, const Bind& bind,
    const mos_math::Secondary& before = {}, const mos_math::Secondary& after = {}) {
    Vec3 result{};
    require(sample(pose, bind, before, after, {}, result), "expected a valid offset curve");
    return result;
}

void checkBend(double degrees) {
    const double angle = degrees * pi / 180.0;
    const auto rotation = q_axis_angle({0, 0, 1}, angle);
    const auto pose = buildPose({}, xAxis, identity, rotation, 4.0, 4.0);
    for (const double offset : {-1.0, 0.0, 1.0}) {
        const auto first = evaluate(pose, {0.0, {0, offset, 0}});
        const auto last = evaluate(pose, {1.0, {0, offset, 0}});
        require(v3_len(v3_sub(first, {-4, offset, 0})) < 1e-10, "incoming endpoint");
        require(v3_len(v3_sub(last, q_rotate_v3(rotation, {4, offset, 0}))) < 1e-10,
            "outgoing endpoint");
        const double radius = std::max(-offset, 0.14 * std::abs(offset));
        const double straight = 4.0 - (offset + radius) * std::tan(angle * 0.5);
        const double total = 2.0 * straight + radius * angle;
        const double join = straight / total;
        const double step = 1e-6;
        const auto atJoin = evaluate(pose, {join, {0, offset, 0}});
        const auto left = evaluate(pose, {join - step, {0, offset, 0}});
        const auto right = evaluate(pose, {join + step, {0, offset, 0}});
        if (radius > 0.0)
            require(v3_dot(v3_norm(v3_sub(atJoin, left)), v3_norm(v3_sub(right, atJoin))) >
                1.0 - 1e-7, "line and arc must meet tangentially");
        auto previous = first;
        for (int i = 1; i <= 2048; ++i) {
            const double f = static_cast<double>(i) / 2048.0;
            const auto current = evaluate(pose, {f, {0, offset, 0}});
            const double arcPhase = radius > 0.0
                ? std::clamp((f * total - straight) / radius, 0.0, angle)
                : (f < join ? 0.0 : angle);
            const auto tangent = q_rotate_v3(q_axis_angle({0, 0, 1}, arcPhase), xAxis);
            const auto delta = v3_sub(current, previous);
            if (radius > 0.0 || f != join)
                require(v3_dot(delta, tangent) > 0.0, "vertex samples must advance along the curve");
            if (radius > 0.0)
                require(v3_len(delta) > 0.999 * total / 2048.0, "arc-length spacing must be uniform");
            previous = current;
        }
    }
}

void checkLocalBend(double degrees) {
    const double angle = degrees * pi / 180.0;
    const auto rotation = q_axis_angle({0, 0, 1}, angle);
    const auto pose = buildPose({}, xAxis, identity, rotation, 5.0, 5.0);
    const auto region = localRegion(pose, 1.0);
    mos_math::Secondary edit;
    edit.bulgeOuter = 0.2;
    edit.slideInner = 0.1;
    for (const double side : {-1.0, 1.0}) {
        for (const double radial : {-1.0, 1.0}) {
            const auto cap = bindVertex({5 * side, radial, 0}, {}, xAxis, 5, 5);
            require(sampleLocal(pose, region, cap, edit, edit, {}).coverage == 0.0,
                "end rings must be fully controlled by LBS");
            const auto join = bindVertex({region.anchorPose.before * side, radial, 0}, {}, xAxis, 5, 5);
            const auto base = sampleLocal(pose, region, join, {}, {}, {});
            const auto modified = sampleLocal(pose, region, join, edit, edit, {});
            const auto expected = q_rotate_v3(side < 0.0 ? identity : rotation,
                {region.anchorPose.before * side, radial, 0});
            require(base.coverage > 0.999 && v3_len(v3_sub(base.position, expected)) < 1e-10,
                "straight anchors must remain in the original cross-section plane");
            require(v3_len(v3_sub(base.position, modified.position)) < 1e-10,
                "bulge and slide must vanish at the anchors");
        }
    }

    constexpr int sides = 24;
    constexpr int rings = 201;
    std::vector<Vec3> posed;
    for (int ring = 0; ring < rings; ++ring) {
        const double x = -5.0 + 0.05 * ring;
        const double t = std::clamp(0.5 + x / 5.0, 0.0, 1.0);
        const double w = t * t * (3.0 - 2.0 * t);
        for (int k = 0; k < sides; ++k) {
            const double phi = k * 2.0 * pi / sides;
            const Vec3 p{x, std::cos(phi), std::sin(phi)};
            const auto bind = bindVertex(p, {}, xAxis, 5, 5);
            const auto sample = sampleLocal(pose, region, bind, {}, {}, {});
            const auto lbs = v3_add(v3_scale(p, 1.0 - w), v3_scale(q_rotate_v3(rotation, p), w));
            const auto result = v3_add(v3_scale(lbs, 1.0 - sample.coverage),
                v3_scale(sample.position, sample.coverage));
            if (std::abs(x) >= region.endBefore)
                require(v3_len(v3_sub(result, lbs)) == 0.0, "outside support must match LBS exactly");
            posed.push_back(result);
        }
    }
    for (int ring = 0; ring < rings - 1; ++ring) {
        const double x = -5.0 + 0.05 * (ring + 0.5);
        const double phase = angle * std::clamp(0.5 + x / (2.0 * region.anchorPose.before), 0.0, 1.0);
        for (int k = 0; k < sides; ++k) {
            const int at = ring * sides + k;
            const int next = ring * sides + (k + 1) % sides;
            const double phi = (k + 0.5) * 2.0 * pi / sides;
            const auto outward = q_rotate_v3(q_axis_angle({0, 0, 1}, phase),
                {0, std::cos(phi), std::sin(phi)});
            const auto normal = v3_cross(v3_sub(posed[next], posed[at]), v3_sub(posed[at + sides], posed[at]));
            require(v3_dot(normal, outward) > 0.0, "the local-to-LBS tube must keep outward winding");
        }
    }
}

void checkWeightedHandoff() {
    const auto rotation = q_axis_angle({0, 0, 1}, pi / 2);
    const auto pose = buildPose({}, xAxis, identity, rotation, 5, 5);
    const auto region = localRegion(pose, 1.0);
    for (const double side : {-1.0, 1.0}) {
        auto bind = bindVertex({side * 4.0, 1.0, 0}, {}, xAxis, 5, 5);
        const auto rigid = q_rotate_v3(side < 0.0 ? identity : rotation, {side * 4.0, 1.0, 0});
        require(sampleLocal(pose, region, bind, {}, {}, {}).coverage == 0.0,
            "beyond the geometric region a pure LBS limb remains untouched");
        bind.oppositeWeight = 0.012;
        const auto retained = sampleLocal(pose, region, bind, {}, {}, {});
        require(retained.coverage == 1.0 && v3_len(v3_sub(retained.position, rigid)) < 1e-10,
            "retain the straight branch through the remaining mixed-weight band");
        bind.fraction = side < 0.0 ? 0.0 : 1.0;
        require(sampleLocal(pose, region, bind, {}, {}, {}).coverage == 0.0,
            "the end ring stays with LBS even if painted with mixed weights");
    }
}

void checkDeepRelief() {
    const auto rotation = q_axis_angle({0, 0, 1}, 140.0 * pi / 180.0);
    const auto pose = buildPose({}, xAxis, identity, rotation, 4.0, 4.0);
    const auto region = localRegion(pose, 1.0);
    const auto inside = bindVertex({0, 1, 0}, {}, xAxis, 4.0, 4.0);
    const auto outside = bindVertex({0, -1, 0}, {}, xAxis, 4.0, 4.0);
    const auto inner = sampleLocal(pose, region, inside, {}, {}, {});
    require(inner.coverage > 0.999 && inner.position.x > -2.1 && inner.position.x < -1.4,
        "the inner bend relief brings the groove closer to the hinge");
    const auto outer = sampleLocal(pose, region, outside, {}, {}, {});
    require(outer.coverage > 0.999 && std::abs(v3_len(outer.position) - 0.84) < 1e-10,
        "deep bend should have less outer extension");
    const auto rightAngle = buildPose({}, xAxis, identity, q_axis_angle({0, 0, 1}, pi / 2), 4, 4);
    const auto rightRegion = localRegion(rightAngle, 1.0);
    mos_math::Secondary edit;
    edit.bulgeOuter = 0.2;
    const double deepEdit = v3_len(v3_sub(
        sampleLocal(pose, region, outside, edit, edit, {}).position, outer.position));
    const double rightEdit = v3_len(v3_sub(
        sampleLocal(rightAngle, rightRegion, outside, edit, edit, {}).position,
        sampleLocal(rightAngle, rightRegion, outside, {}, {}, {}).position));
    require(deepEdit > 0.0 && deepEdit < rightEdit, "secondary bulge must be weaker at 140 degrees");
    checkLocalBend(140.0);
}

void checkTwist() {
    const auto twist = q_axis_angle(xAxis, 120.0 * pi / 180.0);
    const auto pose = buildPose({}, xAxis, identity, twist, 4, 4);
    const auto region = localRegion(pose, 1.0);
    for (int ring = 0; ring <= 20; ++ring) {
        const double x = region.anchorPose.before * (0.1 * ring - 1.0);
        const double mix = 0.5 + 0.5 * x / region.anchorPose.before;
        const Vec3 rest{x, 1, 0};
        const auto guided = sampleLocal(pose, region, bindVertex(rest, {}, xAxis, 4, 4), {}, {}, {});
        const auto expected = q_rotate_v3(q_axis_angle(xAxis, mix * 120.0 * pi / 180.0), rest);
        require(guided.coverage > 0.999 && v3_len(v3_sub(guided.position, expected)) < 1e-10,
            "pure twist must interpolate angle without shrinking the section");
    }

    const auto blend = blendWithLbs(pose, {1, 0.8, 0}, {2, 0, 1}, 0.5);
    require(std::abs(blend.x - 1.5) < 1e-10 &&
        std::abs(v3_len({0, blend.y, blend.z}) - 0.9) < 1e-10,
        "hybrid twist must not add chord shrinkage");
    require(v3_len(v3_sub(blendWithLbs(pose, {1, 0.8, 0}, {}, 0.0), {1, 0.8, 0})) == 0.0,
        "outside support the actual LBS output is preserved");

    const auto bend = q_axis_angle({0, 0, 1}, pi / 2);
    const auto bent = buildPose({}, xAxis, identity, q_mul(bend, twist), 4, 4);
    require(std::abs(bent.angle - pi / 2) < 1e-10 &&
        std::abs(bent.twistAngle - 120.0 * pi / 180.0) < 1e-10,
        "a large bend must preserve the independent twist angle");
}

} // namespace

int main() {
    const auto bind = bindVertex({0.8, 0.5, -0.3}, {}, xAxis, 3.0, 4.0);
    const auto straight = buildPose({}, xAxis, identity, identity, 3.0, 4.0);
    require(v3_len(v3_sub(evaluate(straight, bind), {0.8, 0.5, -0.3})) < 1e-12,
        "straight bind must recover the input vertex");
    for (const auto angle : {45.0, 90.0, 120.0}) {
        checkBend(angle);
        checkLocalBend(angle);
    }

    checkWeightedHandoff();
    checkDeepRelief();
    checkTwist();

    const auto pose = buildPose({}, xAxis, identity, q_axis_angle({0, 0, 1}, pi / 2), 4.0, 4.0);
    const auto inside = bindVertex({}, {}, xAxis, 4.0, 4.0);
    const Bind inner{inside.fraction, {0, 1, 0}}, outer{inside.fraction, {0, -1, 0}};
    const auto p = evaluate(pose, inner);
    const auto q = evaluate(pose, outer);
    require(p.x < -1.0 && p.x > -1.06 && p.y > 1.0 && p.y < 1.06, "inner fillet");
    require(std::abs(q.x - std::sqrt(0.5)) < 1e-10 && std::abs(q.y + std::sqrt(0.5)) < 1e-10,
        "outer branch uses the minimum arc tangent at the bone ends");
    mos_math::Secondary bulge;
    bulge.bulgeOuter = 0.2;
    require(v3_len(v3_sub(evaluate(pose, inner, bulge, bulge), p)) < 1e-10, "outer-only bulge");
    require(evaluate(pose, outer, bulge, bulge).x > q.x, "outer bulge expands the offset curve");
    mos_math::Secondary slide;
    slide.slideInner = 0.1;
    const auto input = bindVertex({-2, 1, 0}, {}, xAxis, 4.0, 4.0);
    const auto base = evaluate(pose, input);
    const auto moved = evaluate(pose, input, slide);
    require(moved.x > base.x && std::abs(moved.y - base.y) < 1e-10,
        "incoming slide moves along the offset curve");
    std::cout << "OCD arc bind, spacing, 45/90/120/140 degree bends, bulge and slide checks passed\n";
}
