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

// Smooth the onset of a geometric upper limit without exceeding it.
inline double fitLimit(double value, double limit, double width) {
    if (width <= 1e-12) return std::min(value, limit);
    const double excess = std::max(0.0, width - std::abs(value - limit));
    return std::min(value, limit) - excess * excess / (4.0 * width);
}

inline double deepFalloff(double angle) {
    constexpr double degrees = 3.14159265358979323846 / 180.0;
    return smoothUnit((angle - 90.0 * degrees) / (50.0 * degrees));
}

inline double redistributeLength(double fraction, double bindLength, double posedLength, double deep) {
    const double t = std::clamp(fraction, 0.0, 1.0);
    const double ramp = std::min(0.20 - 0.16 * deep, 0.45 * posedLength / bindLength);
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
    const double response = fade * (anchored ? std::min(pose.angle / pi, 0.5) : pose.angle / pi);
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
    double fillet = 0.14 * bindRadius;
    constexpr double degrees = pi / 180.0;
    const double deepFit = anchored ? smoothUnit((pose.angle - 120.0 * degrees) / (35.0 * degrees)) : 0.0;
    // Reserve tangent span to restore the section gradually. Fit the fillet
    // across the whole section so near-neutral offsets can also turn.
    const double tangentBudget = (0.92 - 0.26 * deepFit) *
        std::min(pose.before, pose.after) / halfAngleTan;
    // Fit the inner roundover before reducing the limb-normal offset.
    // Its tangent points stay on the two posed limb directions.
    const double remainingForFillet = tangentBudget - std::max(0.0, offset);
    // Keep a small finite turn when the inner branches approach each other.
    // The regular fillet stays in place wherever it still fits.
    const double sideBlend = bindRadius > 1e-8 ? smoothUnit(std::abs(height) / (0.35 * bindRadius)) : 0.0;
    const double minimumFillet = std::min((0.015 + 0.06 * sideBlend) * bindRadius, 0.5 * tangentBudget);
    const double fitWidth = 0.10 * bindRadius * deep;
    const double fittedFillet = std::max(minimumFillet, fitLimit(remainingForFillet, fillet, fitWidth));
    fillet += (fittedFillet - fillet) * deep;
    if (anchored && offset > 0.0 && bindRadius > 1e-8) {
        // Ease out of the early depth increase before the short tangent
        // span forces the innermost offset to turn back.
        constexpr double smoothingStart = 112.0 * degrees;
        if (pose.angle > smoothingStart) {
            const double halfCos = std::cos(0.5 * pose.angle);
            const double extraAngle = pose.angle - smoothingStart;
            const double referenceFillet = 0.14 * bindRadius;
            const double referenceCos = std::cos(0.5 * smoothingStart);
            const double referenceOffset = std::min(offset, std::max(0.0,
                0.92 * std::min(pose.before, pose.after) / std::tan(0.5 * smoothingStart) - referenceFillet));
            const double referenceDepth = (referenceOffset + referenceFillet) / referenceCos - referenceFillet;
            const double depthRate = 0.5 * (referenceDepth + referenceFillet) * std::tan(0.5 * smoothingStart);
            const double depthReserve = 0.22 * bindRadius;
            const double depth = referenceDepth - depthReserve * std::expm1(-depthRate * extraAngle / depthReserve);
            const double roundedOffset = std::max(0.0, (depth + fillet) * halfCos - fillet);
            const double facing = smoothUnit((offset / bindRadius - 0.65) / 0.35);
            offset -= std::max(0.0, offset - roundedOffset) * facing;
        }
        const double offsetLimit = std::max(0.0, tangentBudget - fillet);
        offset = std::max(0.0, fitLimit(offset, offsetLimit, fitWidth));
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
        ? redistributeLength(bind.fraction, pose.before + pose.after, posedLength, deep) + slide
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
    Pose coveragePose{};
    double endBefore = 0.0;
    double endAfter = 0.0;
    double radius = 0.0;
    double foldBefore = 0.0;
    double foldAfter = 0.0;
};

inline LocalRegion localRegion(const Pose& pose, double radius) {
    if (radius <= 1e-8) return {};
    LocalRegion region{pose, pose};
    region.radius = radius;
    // Use the right-angle tangent anchors for both the curve solve and
    // the blend. A deeper miter must fit between the same anchors.
    const double wanted = 2.35 * radius;
    region.anchorPose.before = std::min(wanted, 0.80 * pose.before);
    region.anchorPose.after = std::min(wanted, 0.80 * pose.after);
    // Geometry uses the live angle. Coverage uses the same bind footprint
    // and LBS pull as the right-angle pose after the bend passes 90 degrees.
    constexpr double rightAngle = 1.57079632679489661923;
    if (pose.angle > rightAngle && v3_len(pose.normal) > 0.0) {
        const auto unwind = q_axis_angle(pose.normal, rightAngle - pose.angle);
        region.coveragePose = buildPose(pose.pivot, pose.bindAxis, pose.parent,
            q_mul(unwind, pose.child), pose.before, pose.after, pose.pivotDelta);
    }
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
    const auto& coveragePose = region.coveragePose;
    const auto axis = axial < 0.0 ? coveragePose.incoming : coveragePose.outgoing;
    const auto fromPivot = v3_add(v3_scale(pose.bindAxis, axial), bind.radial);
    const auto difference = v3_add(coveragePose.pivotDelta,
        v3_sub(q_rotate_v3(coveragePose.child, fromPivot), q_rotate_v3(coveragePose.parent, fromPivot)));
    const auto normalError = v3_sub(difference, v3_scale(axis, v3_dot(difference, axis)));
    const double error = bind.oppositeWeight * v3_len(normalError);
    // Let the residual LBS pull determine the handoff on each limb. Only
    // reserve a wider endpoint taper when mixed weights still pull there.
    const double skin = smoothUnit(error / (0.004 * region.radius));
    const double capWidth = std::min(0.07 * span, 0.04 * region.radius + 2.0 * error);
    const double cap = smoothUnit((span - distance) / capWidth);
    const double coverage = geometric + (1.0 - geometric) * skin * cap;
    if (coverage <= 0.0) return {};
    Bind local = bind;
    local.fraction = (axial + region.anchorPose.before) /
        (region.anchorPose.before + region.anchorPose.after);
    Vec3 result{};
    if (!sample(region.anchorPose, local, before, after, manualIn, result, true)) return {};
    return {result, coverage};
}

inline double radialReferenceAngle(double angle) {
    constexpr double rightAngle = 1.57079632679489661923;
    constexpr double knee = rightAngle * 5.0 / 6.0;
    const double t = std::clamp((angle - knee) / (rightAngle - knee), 0.0, 1.0);
    return angle <= knee ? angle : knee + (rightAngle - knee) * (t + t * t * (1.0 - t));
}

inline Vec3 narrowFoldCorrection(const Pose& pose, const LocalRegion& region, const Bind& bind,
    const Vec3& lbs, const Vec3& correction, double axial, double radius, double childWeight) {
    constexpr double degrees = 3.14159265358979323846 / 180.0;
    const double bend = smoothUnit((pose.angle - 95.0 * degrees) / (45.0 * degrees));
    const double span = axial < 0.0 ? region.foldBefore : region.foldAfter;
    const double shorter = std::min(region.foldBefore, region.foldAfter);
    const double mixed = smoothUnit(std::min(childWeight, 1.0 - childWeight) / 0.04);
    if (bend == 0.0 || mixed == 0.0 || shorter <= 1e-8 || std::abs(axial) >= span) return correction;

    const auto parentRadial = q_rotate_v3(pose.parent, bind.radial);
    const auto halfTwist = q_axis_angle(pose.incoming, 0.5 * pose.twistAngle);
    const auto middleRadial = q_rotate_v3(halfTwist, parentRadial);
    const double innerOffset = v3_dot(middleRadial, pose.inward);
    const double facing = smoothUnit((innerOffset / radius - 0.15) / 0.7);
    if (facing == 0.0) return correction;

    // Lay out both inner walls as long, shallow rails. A short quadratic
    // across their common corner keeps both rail tangents continuous.
    const auto parentPivot = v3_sub(pose.pivot, pose.pivotDelta);
    const auto midPivot = v3_sub(pose.pivot, v3_scale(pose.pivotDelta, 0.5));
    const auto left = v3_add(parentPivot, v3_sub(parentRadial, v3_scale(pose.incoming, region.foldBefore)));
    const auto right = v3_add(pose.pivot, v3_add(q_rotate_v3(pose.child, bind.radial),
        v3_scale(pose.outgoing, region.foldAfter)));
    const auto bisector = v3_norm(v3_sub(pose.outgoing, pose.incoming));
    const double shoulderDepth = std::min(v3_dot(v3_sub(left, midPivot), bisector),
        v3_dot(v3_sub(right, midPivot), bisector));
    if (shoulderDepth <= 1e-8) return correction;
    const double depth = std::min(1.8 * innerOffset, 0.8 * shoulderDepth);
    const auto tip = v3_add(midPivot, v3_add(v3_scale(bisector, depth),
        v3_scale(pose.normal, v3_dot(middleRadial, pose.normal))));
    const auto leftSlope = v3_scale(v3_sub(left, tip), 1.0 / region.foldBefore);
    const auto rightSlope = v3_scale(v3_sub(right, tip), 1.0 / region.foldAfter);
    const double lateral = std::abs(v3_dot(middleRadial, pose.normal)) / radius;
    const double offCenter = smoothUnit(lateral / 0.35);
    const double roundWidth = std::min((0.10 + 0.33 * offCenter) * radius, 0.18 * shorter);
    Vec3 rail;
    if (std::abs(axial) < roundWidth) {
        const double u = 0.5 + 0.5 * axial / roundWidth;
        rail = v3_add(tip, v3_add(v3_scale(leftSlope, roundWidth * (1.0 - u) * (1.0 - u)),
            v3_scale(rightSlope, roundWidth * u * u)));
    } else {
        rail = v3_add(tip, v3_scale(axial < 0.0 ? leftSlope : rightSlope, std::abs(axial)));
    }
    const double shoulder = 1.0 - smoothUnit(std::abs(axial) / span);
    const double blend = bend * facing * shoulder * mixed;
    return v3_add(v3_scale(correction, 1.0 - blend), v3_scale(v3_sub(rail, lbs), blend));
}

inline Vec3 radialCorrection(const Pose& pose, const LocalRegion& region, const Bind& bind, const Vec3& lbs,
    double referenceAngle, const Quat& referenceChild, const mos_math::Secondary* before,
    const mos_math::Secondary* after, const Vec3& manualIn) {
    const double axial = bind.fraction * (pose.before + pose.after) - pose.before;
    const double radius = v3_len(bind.radial);
    if (radius <= 1e-9) return {};

    const double childWeight = std::clamp(axial < 0.0 ? bind.oppositeWeight :
        1.0 - bind.oppositeWeight, 0.0, 1.0);
    // Follow the minority skin weight along both limbs. Round only the
    // small interval around equal weights so the ridge has no slope break.
    constexpr double corner = 0.15;
    const double imbalance = std::abs(2.0 * childWeight - 1.0);
    const double weightFalloff = imbalance < corner
        ? 1.0 - imbalance * imbalance / (corner * (2.0 - corner))
        : (1.0 - imbalance) / (1.0 - 0.5 * corner);
    const auto parentRadial = q_rotate_v3(pose.parent, bind.radial);
    const auto twist = q_axis_angle(pose.incoming, pose.twistAngle * childWeight);
    const auto twisted = q_rotate_v3(twist, parentRadial);
    const auto referenceSwing = q_axis_angle(pose.normal, 0.5 * referenceAngle);
    const auto referenceTwist = q_axis_angle(pose.incoming, 0.5 * pose.twistAngle);
    const auto referenceDirection = v3_scale(
        q_rotate_v3(referenceSwing, q_rotate_v3(referenceTwist, parentRadial)), 1.0 / radius);
    const auto blendedReference = v3_scale(v3_add(parentRadial, q_rotate_v3(referenceChild, bind.radial)), 0.5);
    const double thicknessLoss = std::max(0.0, radius - v3_dot(blendedReference, referenceDirection));

    // Keep the LBS station along the limb. Transport only the direction
    // of the cross-section correction as the actual pose continues to turn.
    const auto swing = q_axis_angle(pose.normal, pose.angle * childWeight);
    const auto radial = q_rotate_v3(swing, twisted);
    Vec3 correction = v3_scale(radial, weightFalloff * thicknessLoss / radius);
    const double deep = deepFalloff(pose.angle);
    if (deep > 0.0) {
        const double facing = smoothUnit(v3_dot(twisted, pose.inward) / (0.5 * radius));
        if (facing > 0.0) {
            // Near the opposite limb, ease the swelling along the bend
            // bisector while retaining the component that closes the fold.
            const auto parentRay = v3_scale(pose.incoming, -1.0);
            const auto parent = v3_sub(lbs, v3_sub(pose.pivot, pose.pivotDelta));
            const auto child = v3_sub(lbs, pose.pivot);
            const auto rayDistance = [](const Vec3& point, const Vec3& ray) {
                return v3_len(v3_sub(point, v3_scale(ray, std::max(0.0, v3_dot(point, ray)))));
            };
            const double contactRadius = std::min(radius, region.radius);
            if (contactRadius > 1e-9) {
                const double distance = (1.0 - childWeight) * rayDistance(child, pose.outgoing) +
                    childWeight * rayDistance(parent, parentRay);
                const double proximity = smoothUnit((1.15 - distance / contactRadius) / 0.35);
                const auto bisector = v3_norm(v3_sub(pose.outgoing, pose.incoming));
                const auto swelling = v3_scale(bisector, std::max(0.0, v3_dot(correction, bisector)));
                correction = v3_sub(correction, v3_scale(swelling, deep * facing * proximity));
            }
        }
    }
    correction = narrowFoldCorrection(pose, region, bind, lbs, correction, axial, radius, childWeight);
    // Authored offsets peak at the joint and fade across the painted blend
    // region. The flat interior of the base correction is not used here.
    const double beforeSpan = region.foldBefore > 0.0 ? region.foldBefore : region.endBefore;
    const double afterSpan = region.foldAfter > 0.0 ? region.foldAfter : region.endAfter;
    const double span = axial < 0.0 ? beforeSpan : afterSpan;
    const double crossfade = std::min(0.35 * region.radius, 0.25 * std::min(beforeSpan, afterSpan));
    if (before && after && pose.angle > 1e-8 && crossfade > 1e-8 && std::abs(axial) < span) {
        const double axialFalloff = 1.0 - smoothUnit(std::abs(axial) / span);
        const double blendSupport = smoothUnit(std::min(childWeight, 1.0 - childWeight) / 0.04);
        const double nextMix = smoothUnit(0.5 + 0.5 * axial / crossfade);
        const auto axis = q_rotate_v3(swing, pose.incoming);
        auto inward = q_rotate_v3(swing, pose.inward);
        const auto frame = q_mul(swing, q_mul(twist, pose.parent));
        const auto manual = q_rotate_v3(frame, manualIn);
        const auto projected = v3_norm(v3_sub(manual, v3_scale(axis, v3_dot(manual, axis))));
        if (v3_len(projected) > 0.0) inward = projected;
        const auto identity = ocd_math::Mat3::identity();
        mos_math::BendControl control{axis, inward, v3_cross(axis, inward),
            referenceAngle / 3.14159265358979323846,
            std::min(region.anchorPose.before, region.anchorPose.after), true};
        const auto incoming = mos_math::sampleControl({}, identity, radial, control, *before);
        control.incoming = false;
        const auto outgoing = mos_math::sampleControl({}, identity, radial, control, *after);
        const auto edit = v3_sub(v3_add(v3_scale(incoming, 1.0 - nextMix),
            v3_scale(outgoing, nextMix)), radial);
        correction = v3_add(correction, v3_scale(edit, axialFalloff * blendSupport));
    }
    return correction;
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
