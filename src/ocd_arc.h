#pragma once

#include "mos_math.h"

namespace ocd_arc {

using namespace mathcore;

struct Bind {
    double fraction = 0.0;
    Vec3 radial{};
};

inline Bind bindVertex(const Vec3& point, const Vec3& pivot, const Vec3& axis,
    double before, double after) {
    const auto relative = v3_sub(point, pivot);
    const double axial = v3_dot(relative, axis);
    return {(axial + before) / (before + after),
        v3_sub(relative, v3_scale(axis, axial))};
}

struct Pose {
    Vec3 pivot{};
    Vec3 bindAxis{};
    Vec3 incoming{};
    Vec3 outgoing{};
    Vec3 normal{};
    Vec3 inward{};
    Quat parent{};
    Quat child{};
    double angle = 0.0;
    double before = 0.0;
    double after = 0.0;
};

inline Pose buildPose(const Vec3& pivot, const Vec3& axis, const Quat& parent,
    const Quat& child, double before, double after) {
    Pose pose{pivot, axis, q_rotate_v3(parent, axis), q_rotate_v3(child, axis),
        {}, {}, parent, child, 0.0, before, after};
    const auto cross = v3_cross(pose.incoming, pose.outgoing);
    const double sine = v3_len(cross);
    pose.angle = std::atan2(sine, v3_dot(pose.incoming, pose.outgoing));
    if (sine > 1e-8) {
        pose.normal = v3_scale(cross, 1.0 / sine);
        pose.inward = v3_cross(pose.normal, pose.incoming);
    }
    return pose;
}

inline double redistributeLength(double fraction, double bindLength, double posedLength) {
    const double t = std::clamp(fraction, 0.0, 1.0);
    const double ramp = std::min(0.20, 0.45 * posedLength / bindLength);
    if (ramp <= 1e-8) return t * posedLength;
    double shift = (t - 0.5 * ramp) / (1.0 - ramp);
    if (t < ramp) shift = 0.5 * t * t / (ramp * (1.0 - ramp));
    if (t > 1.0 - ramp) {
        const double remaining = 1.0 - t;
        shift = 1.0 - 0.5 * remaining * remaining / (ramp * (1.0 - ramp));
    }
    return fraction * bindLength + shift * (posedLength - bindLength);
}

// Straight bind specialization of the Sony segment–arc–segment example.
// Each vertex's offset has its own two tangent lines and its own arc length.
inline bool sample(const Pose& pose, const Bind& bind, const mos_math::Secondary& before,
    const mos_math::Secondary& after, const Vec3& manualIn, Vec3& result, bool anchored = false) {
    constexpr double pi = 3.14159265358979323846;
    if (pose.before <= 0.0 || pose.after <= 0.0 || pose.angle >= pi - 1e-5) return false;
    const double axial = bind.fraction * (pose.before + pose.after) - pose.before;
    const double twistMix = 0.5 + 0.5 * std::clamp(axial /
        (axial < 0.0 ? pose.before : pose.after), -1.0, 1.0);
    const double sign = q_dot(pose.parent, pose.child) < 0.0 ? -1.0 : 1.0;
    const auto section = q_normalize(q_add(q_scale(pose.parent, 1.0 - twistMix),
        q_scale(pose.child, sign * twistMix)));
    const auto sectionAxis = q_rotate_v3(section, pose.bindAxis);
    auto radial = q_rotate_v3(q_between(sectionAxis, pose.incoming),
        q_rotate_v3(section, bind.radial));
    if (pose.angle < 1e-8) {
        result = v3_add(pose.pivot, v3_add(v3_scale(pose.incoming, axial), radial));
        return true;
    }

    // MOS controls edit offsets and the distance to sample before the arc
    // is evaluated. A slide follows the resulting curve through the fillet.
    const double span = axial < 0.0 ? pose.before : pose.after;
    const double normalized = std::min(1.0, std::abs(axial) / span);
    const double fade = (1.0 - normalized) * (1.0 - normalized) * (1.0 + 2.0 * normalized);
    auto inward = pose.inward;
    auto normal = pose.normal;
    const auto manual = q_rotate_v3(pose.parent, manualIn);
    const auto projected = v3_norm(v3_sub(manual, v3_scale(pose.incoming, v3_dot(manual, pose.incoming))));
    if (v3_len(projected) > 0.0) {
        inward = projected;
        normal = v3_cross(pose.incoming, inward);
    }
    const auto identity = ocd_math::Mat3::identity();
    mos_math::BendControl control{pose.incoming, inward, normal, fade * pose.angle / pi,
        std::min(pose.before, pose.after), true};
    const auto editBefore = mos_math::sampleControl({}, identity, radial, control, before);
    control.incoming = false;
    const auto editAfter = mos_math::sampleControl({}, identity, radial, control, after);
    const auto edited = v3_add(v3_scale(editBefore, 1.0 - twistMix), v3_scale(editAfter, twistMix));
    const double slide = v3_dot(edited, pose.incoming);
    radial = v3_sub(edited, v3_scale(pose.incoming, slide));

    const double offset = v3_dot(radial, pose.inward);
    const double height = v3_dot(radial, pose.normal);
    const double halfAngleTan = std::tan(0.5 * pose.angle);
    // Outside, radius -offset turns at the bone ends. Inside, leave a
    // small fillet to avoid a zero-length arc at the miter.
    const double radius = std::max(-offset, 0.14 * v3_len(bind.radial));
    const double removed = (offset + radius) * halfAngleTan;
    if (removed >= std::min(pose.before, pose.after)) return false;

    const auto first = v3_add(pose.pivot, v3_add(radial, v3_scale(pose.incoming, -pose.before)));
    const double firstLength = pose.before - removed;
    const double arcLength = radius * pose.angle;
    const double lastLength = pose.after - removed;
    const double posedLength = firstLength + arcLength + lastLength;
    const double distance = anchored
        ? redistributeLength(bind.fraction, pose.before + pose.after, posedLength) + slide
        : std::clamp(bind.fraction * posedLength + slide, 0.0, posedLength);
    if (distance <= firstLength) {
        result = v3_add(first, v3_scale(pose.incoming, distance));
    }
    else if (distance < firstLength + arcLength) {
        const double phase = (distance - firstLength) / radius;
        const auto tangentPoint = v3_add(first, v3_scale(pose.incoming, firstLength));
        result = v3_add(tangentPoint, v3_add(v3_scale(pose.incoming, radius * std::sin(phase)),
            v3_scale(pose.inward, radius * (1.0 - std::cos(phase)))));
    }
    else {
        const auto outwardNormal = v3_cross(pose.normal, pose.outgoing);
        result = v3_add(pose.pivot, v3_add(v3_scale(pose.normal, height),
            v3_add(v3_scale(outwardNormal, offset),
                v3_scale(pose.outgoing, removed + distance - firstLength - arcLength))));
    }
    return true;
}

struct LocalRegion {
    Pose anchorPose{};
    double endBefore = 0.0;
    double endAfter = 0.0;
};

inline LocalRegion localRegion(const Pose& pose, double radius) {
    if (radius <= 1e-8) return {};
    LocalRegion region{pose};
    constexpr double maxBend = 2.35; // Bound the support near a fully folded hinge.
    const double tangentClearance = 1.45 * std::tan(0.5 * std::min(pose.angle, maxBend));
    const double wanted = radius * std::max(2.35, tangentClearance);
    region.anchorPose.before = std::min(wanted, 0.80 * pose.before);
    region.anchorPose.after = std::min(wanted, 0.80 * pose.after);
    region.endBefore = std::min(0.96 * pose.before, region.anchorPose.before + 1.25 * radius);
    region.endAfter = std::min(0.96 * pose.after, region.anchorPose.after + 1.25 * radius);
    return region;
}

inline mos_math::GuideSample sampleLocal(const Pose& pose, const LocalRegion& region,
    const Bind& bind, const mos_math::Secondary& before, const mos_math::Secondary& after,
    const Vec3& manualIn) {
    const double axial = bind.fraction * (pose.before + pose.after) - pose.before;
    const double start = axial < 0.0 ? region.anchorPose.before : region.anchorPose.after;
    const double end = axial < 0.0 ? region.endBefore : region.endAfter;
    const double distance = std::abs(axial);
    if (distance >= end) return {};
    const double t = std::clamp((distance - start) / (end - start), 0.0, 1.0);
    const double coverage = (1.0 - t) * (1.0 - t) * (1.0 - t) * (1.0 + 3.0 * t + 6.0 * t * t);
    Bind local = bind;
    local.fraction = (axial + region.anchorPose.before) /
        (region.anchorPose.before + region.anchorPose.after);
    Vec3 result{};
    if (!sample(region.anchorPose, local, before, after, manualIn, result, true)) return {};
    return {result, coverage};
}

} // namespace ocd_arc
