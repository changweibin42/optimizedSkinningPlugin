#include "mos_curve_sampling.h"

#include <cstdlib>
#include <iostream>

using namespace mos_curve_sampling;
namespace {

constexpr double pi = 3.14159265358979323846;
const Vec3 zAxis{0, 0, 1};
const mos_math::Secondary zero{};

void require(bool result, const char* message) {
    if (!result) {
        std::cerr << message << '\n';
        std::exit(1);
    }
}

void close(const Vec3& actual, const Vec3& expected, const char* message, double tolerance = 1e-8) {
    require(v3_len(v3_sub(actual, expected)) < tolerance, message);
}

Guide elbow() {
    Guide guide;
    guide.joints = {{{-4, 0, 0}}, {{0, 0, 0}}};
    guide.terminalLength = 4;
    require(prepare(guide, 128), "prepare elbow");
    return guide;
}

Attachment at(const Guide& guide, const Vec3& point) {
    Attachment result;
    require(bind(guide, point, result), "bind vertex");
    return result;
}

Pose elbowPose(const Guide& guide, double degrees, bool secondary = true) {
    auto joints = guide.joints;
    joints[1].rotation = q_axis_angle(zAxis, degrees * pi / 180.0);
    return makePose(guide, joints, {}, {}, 1.0, secondary);
}

void bindPoseAndEquation() {
    const auto guide = elbow();
    const auto rest = makePose(guide, guide.joints, {}, {}, 1.0, true);
    const auto rigidQ = q_axis_angle(v3_norm({1, 2, 3}), 0.7);
    const Vec3 translation{4, -2, 8};
    auto movedJoints = guide.joints;
    for (auto& joint : movedJoints) {
        joint.position = v3_add(q_rotate_v3(rigidQ, joint.position), translation);
        joint.rotation = q_mul(rigidQ, joint.rotation);
    }
    const auto rigid = makePose(guide, movedJoints, {}, {}, 0.0, true);
    mos_math::Secondary edits;
    edits.bulgeInner = 0.2;
    edits.slideOuter = 0.3;
    for (const double x : {-4.5, -2.3, 0.0, 2.3, 4.5}) {
        const Vec3 point{x, 1.0, -0.5};
        const auto attachment = at(guide, point);
        close(evaluate(guide, rest, attachment, edits, edits), point, "bind identity with controls enabled");
        close(evaluate(guide, rigid, attachment, edits, edits),
            v3_add(q_rotate_v3(rigidQ, point), translation), "rigid motion and no spurious bend");
        for (const double degrees : {45.0, 90.0, 140.0}) {
            const auto pose = elbowPose(guide, degrees, false);
            std::vector<Vec3> offsetCurve;
            for (size_t i = 0; i < pose.centers.size(); ++i)
                offsetCurve.push_back(v3_add(pose.centers[i], pose.frames[i].apply(attachment.offset)));
            const auto expected = ocd_math::bspline_eval(offsetCurve, guide.knots, 3, attachment.parameter);
            close(evaluate(guide, pose, attachment, zero, zero), expected,
                "Eq. 5 equals explicit deformed offset-curve evaluation");
        }
    }
}

void secondaryDirections() {
    const auto guide = elbow();
    // This station's four active controls all belong to the incoming limb.
    const auto inside = at(guide, {-2.7, 1.0, 0});
    const auto side = at(guide, {-2.7, 0, 1.0});
    const auto tail = at(guide, {-4.1, 1.0, 0});
    mos_math::Secondary slide, ortho, bulge, center;
    slide.slideInner = 0.5;
    ortho.orthoInner = 0.5;
    bulge.bulgeSide = 0.5;
    center.centerInner = 0.5;
    const auto pose90 = elbowPose(guide, 90);
    const auto pose140 = elbowPose(guide, 140);
    const auto displacement = [&](const Pose& pose, const Attachment& vertex,
        const mos_math::Secondary& prev, const mos_math::Secondary& next) {
        return v3_sub(evaluate(guide, pose, vertex, prev, next),
            evaluate(guide, pose, vertex, zero, zero));
    };
    const auto s = displacement(pose90, inside, slide, zero);
    const auto o = displacement(pose90, inside, ortho, zero);
    const auto b = displacement(pose90, side, bulge, bulge);
    require(s.x > 0.05 && std::abs(s.y) < 1e-9, "incoming slide follows twist axis");
    require(o.y > 0.05 && std::abs(o.x) < 1e-9, "slide ortho extends radially across twist axis");
    require(b.z > 0.01 && std::abs(b.x) < 1e-9, "side bulge extends sideways");
    close(displacement(pose90, inside, zero, slide), {}, "next does not move incoming support");
    close(displacement(pose140, inside, slide, zero), v3_scale(s, 140.0 / 90.0),
        "secondary response continues beyond 90 degrees");
    const auto centerVertex = at(guide, {-0.2, 1, 0});
    require(v3_len(displacement(pose90, centerVertex, center, center)) > 0.1, "center slide acts near the bend");
    close(displacement(pose90, tail, slide, zero), {}, "slide reaches zero at the guide endpoint");

    auto joints = guide.joints;
    joints[1].rotation = q_axis_angle(zAxis, 90 * pi / 180.0);
    const auto manual = makePose(guide, joints, {}, {{}, {0, 0, 1}}, 1.0, true);
    require(v3_len(displacement(manual, side, slide, zero)) > 0.05,
        "joint-space manual in-vector moves selected side into inner sector");
    close(displacement(manual, inside, slide, zero), {}, "manual in-vector replaces automatic sector");
}

void multipleDrivers() {
    Guide guide;
    guide.joints = {{{0, 0, 0}}, {{4, 0, 0}}, {{8, 0, 0}}};
    require(prepare(guide, 128), "prepare two-driver guide");
    const auto nearFirst = at(guide, {-0.1, 1, 0});
    auto joints = guide.joints;
    joints[1].rotation = q_axis_angle(zAxis, pi / 2);
    joints[2].rotation = q_axis_angle(zAxis, 3 * pi / 4);
    const auto pose = makePose(guide, joints, {}, {}, 1.0, true);
    mos_math::Secondary edit;
    edit.bulgeInner = edit.bulgeOuter = 0.4;
    edit.slideInner = 0.3;
    edit.slideOuter = -0.3;
    const auto base = evaluate(guide, pose, nearFirst, zero, zero);
    const auto correction = [&](std::vector<Driver> drivers) {
        guide.drivers[0] = std::move(drivers);
        return v3_sub(evaluate(guide, pose, nearFirst, edit, zero), base);
    };
    const auto first = correction({{1, 1, 1}});
    const auto second = correction({{2, 1, 1}});
    require(first.x > second.x && second.x > 0.1, "each joint contributes with its own bend");
    const auto combined = correction({{1, 0.9, 0.6}, {2, 0.1, 0.4}});
    close(combined, {0.9 * first.x + 0.1 * second.x, 0.6 * first.y + 0.4 * second.y, 0},
        "slide and bulge driver weights accumulate independently");

    guide.controls[0].edits[0] = edit;
    close(v3_sub(evaluate(guide, pose, nearFirst, zero, zero), base), combined,
        "authored control values work with node-wide values disabled");
}

void closedSeam() {
    Guide guide;
    guide.closed = true;
    guide.joints = {{{0, 0, 0}}, {{3, 0, 0}}, {{3, 3, 0.4}}, {{0, 3, 0.8}}};
    require(prepare(guide, 256), "prepare closed guide");
    close(guide.samples.front(), guide.samples.back(), "periodic knot seam");
    auto joints = guide.joints;
    joints.back().position.z += 0.7;
    joints.back().rotation = q_axis_angle({0, 1, 0}, 0.5);
    const auto posed = makePose(guide, joints, {}, {}, 0.4, false);
    const Vec3 offset{0.2, 0.15, 0.22};
    const Attachment a{mos_math::sampleBasis(guide.knots, 3, 0), offset};
    const Attachment b{mos_math::sampleBasis(guide.knots, 3, 1), offset};
    close(evaluate(guide, posed, a, zero, zero), evaluate(guide, posed, b, zero, zero),
        "ghost-padded frames close the offset-curve seam in a nonplanar pose");
}

} // namespace

int main() {
    bindPoseAndEquation();
    secondaryDirections();
    multipleDrivers();
    closedSeam();
    std::cout << "MOS curve sampling: Eq. 5/9, secondary directions, 140-degree response, "
        "multiple drivers and closed seam passed\n";
}
