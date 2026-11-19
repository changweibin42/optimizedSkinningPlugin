#pragma once

#include "mos_math.h"

namespace ocd_arc {

using namespace mathcore;

struct Bind {
    double fraction = 0.0;
    Vec3 radial{};
    double oppositeWeight = 0.0;
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
    Vec3 pivotDelta{};
    double twistAngle = 0.0;
};

inline Pose buildPose(const Vec3& pivot, const Vec3& axis, const Quat& parent,
    const Quat& child, double before, double after, const Vec3& pivotDelta = {}) {
    Pose pose{pivot, axis, q_rotate_v3(parent, axis), q_rotate_v3(child, axis),
        {}, {}, parent, child, 0.0, before, after, pivotDelta};
    const auto swing = q_between(pose.incoming, pose.outgoing);
    const auto childOnIncoming = q_mul(q_inverse(swing), child);
    auto relativeTwist = q_mul(childOnIncoming, q_inverse(parent));
    if (relativeTwist.w < 0.0) relativeTwist = q_scale(relativeTwist, -1.0);
    pose.twistAngle = 2.0 * std::atan2(v3_dot(
        {relativeTwist.x, relativeTwist.y, relativeTwist.z}, pose.incoming), relativeTwist.w);
    const auto cross = v3_cross(pose.incoming, pose.outgoing);
    const double sine = v3_len(cross);
    pose.angle = std::atan2(sine, v3_dot(pose.incoming, pose.outgoing));
    if (sine > 1e-8) {
        pose.normal = v3_scale(cross, 1.0 / sine);
        pose.inward = v3_cross(pose.normal, pose.incoming);
    }
    return pose;
}

inline double smoothUnit(double value) {
    const double t = std::clamp(value, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double deepFalloff(double angle) {
    constexpr double degrees = 3.14159265358979323846 / 180.0;
    return smoothUnit((angle - 90.0 * degrees) / (50.0 * degrees));
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
    // Transport swing along the curve. Twist rotates the cross section
    // independently, without averaging the two swung joint frames.
    auto radial = q_rotate_v3(q_axis_angle(pose.incoming, pose.twistAngle * twistMix),
        q_rotate_v3(pose.parent, bind.radial));
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
    const double deep = anchored ? deepFalloff(pose.angle) : 0.0;
    const double response = fade * pose.angle / pi * (1.0 - 0.55 * deep);
    mos_math::BendControl control{pose.incoming, inward, normal, response,
        std::min(pose.before, pose.after), true};
    const auto editBefore = mos_math::sampleControl({}, identity, radial, control, before);
    control.incoming = false;
    const auto editAfter = mos_math::sampleControl({}, identity, radial, control, after);
    const auto edited = v3_add(v3_scale(editBefore, 1.0 - twistMix), v3_scale(editAfter, twistMix));
    const double slide = v3_dot(edited, pose.incoming);
    radial = v3_sub(edited, v3_scale(pose.incoming, slide));

    const double originalOffset = v3_dot(radial, pose.inward);
    double offset = originalOffset;
    const double height = v3_dot(radial, pose.normal);
    const double halfAngleTan = std::tan(0.5 * pose.angle);
    // Outside, radius -offset turns at the bone ends. Inside, leave a
    // small fillet to avoid a zero-length arc at the miter.
    const double bindRadius = v3_len(bind.radial);
    const double fillet = 0.14 * bindRadius;
    if (anchored && offset > 0.0 && bindRadius > 1e-8) {
        constexpr double degrees = pi / 180.0;
        const double innerRelief = smoothUnit((pose.angle - 110.0 * degrees) / (30.0 * degrees));
        const double facing = smoothUnit(offset / (0.5 * bindRadius));
        const double narrowed = offset * (1.0 - 0.42 * innerRelief * facing);
        const double fit = 0.92 * std::min(pose.before, pose.after) / halfAngleTan - fillet;
        offset = std::min(narrowed, std::max(0.0, fit));
    }
    else if (offset < 0.0 && bindRadius > 1e-8) {
        const double facing = smoothUnit(-offset / (0.5 * bindRadius));
        offset *= 1.0 - 0.16 * deep * facing;
    }
    radial = v3_add(radial, v3_scale(pose.inward, offset - originalOffset));
    const double radius = std::max(-offset, fillet);
    const double relief = originalOffset - offset;
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
        const double restore = smoothUnit(1.0 - distance / firstLength);
        result = v3_add(first, v3_add(v3_scale(pose.incoming, distance),
            v3_scale(pose.inward, relief * restore)));
    }
    else if (distance < firstLength + arcLength) {
        const double phase = (distance - firstLength) / radius;
        const auto tangentPoint = v3_add(first, v3_scale(pose.incoming, firstLength));
        result = v3_add(tangentPoint, v3_add(v3_scale(pose.incoming, radius * std::sin(phase)),
            v3_scale(pose.inward, radius * (1.0 - std::cos(phase)))));
    }
    else {
        const auto outwardNormal = v3_cross(pose.normal, pose.outgoing);
        const double restore = smoothUnit((distance - firstLength - arcLength) / lastLength);
        result = v3_add(pose.pivot, v3_add(v3_scale(pose.normal, height),
            v3_add(v3_scale(outwardNormal, offset + relief * restore),
                v3_scale(pose.outgoing, removed + distance - firstLength - arcLength))));
    }
    return true;
}

struct LocalRegion {
    Pose anchorPose{};
    double endBefore = 0.0;
    double endAfter = 0.0;
    double radius = 0.0;
};

inline LocalRegion localRegion(const Pose& pose, double radius) {
    if (radius <= 1e-8) return {};
    LocalRegion region{pose};
    region.radius = radius;
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
    const double span = axial < 0.0 ? pose.before : pose.after;
    if (distance >= span || end <= start) return {};
    const double t = std::clamp((distance - start) / (end - start), 0.0, 1.0);
    const double geometric = (1.0 - t) * (1.0 - t) * (1.0 - t) * (1.0 + 3.0 * t + 6.0 * t * t);
    // Compare LBS's cross-sectional displacement against the rigid limb.
    // The same opposite weight pulls more on a ring farther from the hinge.
    const auto axis = axial < 0.0 ? pose.incoming : pose.outgoing;
    const auto fromPivot = v3_add(v3_scale(pose.bindAxis, axial), bind.radial);
    const auto difference = v3_add(pose.pivotDelta, v3_sub(q_rotate_v3(pose.child, fromPivot),
        q_rotate_v3(pose.parent, fromPivot)));
    const auto normalError = v3_sub(difference, v3_scale(axis, v3_dot(difference, axis)));
    const double error = bind.oppositeWeight * v3_len(normalError);
    const double skin = smoothUnit(error / (0.025 * region.radius));
    const double cap = 1.0 - smoothUnit((distance / span - 0.93) / 0.07);
    const double coverage = geometric + (1.0 - geometric) * skin * cap;
    if (coverage <= 0.0) return {};
    Bind local = bind;
    local.fraction = (axial + region.anchorPose.before) /
        (region.anchorPose.before + region.anchorPose.after);
    Vec3 result{};
    if (!sample(region.anchorPose, local, before, after, manualIn, result, true)) return {};
    return {result, coverage};
}

inline Vec3 blendWithLbs(const Pose& pose, const Vec3& lbs, const Vec3& curve, double mix) {
    if (mix <= 0.0) return lbs;
    if (mix >= 1.0) return curve;
    const auto linear = v3_add(v3_scale(lbs, 1.0 - mix), v3_scale(curve, mix));
    constexpr double degrees = 3.14159265358979323846 / 180.0;
    const double twistShare = 1.0 - smoothUnit(pose.angle / (15.0 * degrees));
    if (twistShare <= 0.0 || std::abs(pose.twistAngle) < 1e-8) return linear;

    const auto axis = pose.incoming;
    const auto perpendicular = [&](const Vec3& point) {
        const auto fromPivot = v3_sub(point, pose.pivot);
        return v3_sub(fromPivot, v3_scale(axis, v3_dot(fromPivot, axis)));
    };
    const auto lbsRadial = perpendicular(lbs);
    const auto arcRadial = perpendicular(curve);
    const double lbsRadius = v3_len(lbsRadial), arcRadius = v3_len(arcRadial);
    if (lbsRadius < 1e-8 || arcRadius < 1e-8) return linear;

    // Interpolating radial vectors makes the hybrid path cut through the
    // section. Follow the short arc between their phases instead.
    const double angle = std::atan2(v3_dot(axis, v3_cross(lbsRadial, arcRadial)),
        v3_dot(lbsRadial, arcRadial));
    const double radius = (1.0 - mix) * lbsRadius + mix * arcRadius;
    const auto radial = v3_scale(q_rotate_v3(q_axis_angle(axis, mix * angle), lbsRadial),
        radius / lbsRadius);
    return v3_add(linear, v3_scale(v3_sub(radial, perpendicular(linear)), twistShare));
}

} // namespace ocd_arc
