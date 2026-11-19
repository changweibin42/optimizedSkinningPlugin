#include "cor_contour.h"

#include <algorithm>
#include <cmath>

namespace cor_contour {
namespace {

double smoothStep(double start, double end, double value) {
    const double t = std::clamp((value - start) / (end - start), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

} // namespace

using namespace mathcore;

BendFrame buildFrame(const Vec3& bindParent, const Vec3& bindPivot, const Vec3& posePivot,
    const Quat& parentRotation, const Quat& childRotation) {
    BendFrame frame;
    frame.bindPivot = bindPivot;
    frame.bindAxis = v3_norm(v3_sub(bindPivot, bindParent));
    frame.posePivot = posePivot;
    frame.incoming = q_rotate_v3(parentRotation, frame.bindAxis);
    frame.outgoing = q_rotate_v3(childRotation, frame.bindAxis);
    const auto normal = v3_cross(frame.incoming, frame.outgoing);
    const double sine = v3_len(normal);
    if (sine < 1e-8) return frame;

    constexpr double degrees = 3.14159265358979323846 / 180.0;
    frame.angle = std::atan2(sine, v3_dot(frame.incoming, frame.outgoing));
    frame.activation = smoothStep(0.0, 60.0 * degrees, frame.angle) *
        (1.0 - smoothStep(100.0 * degrees, 145.0 * degrees, frame.angle));
    frame.bendNormal = v3_scale(normal, 1.0 / sine);
    frame.incomingNormal = v3_cross(frame.bendNormal, frame.incoming);
    return frame;
}

Vec3 correction(const BendFrame& frame, const Vec3& rest, const Vec3& posed,
    const Quat& vertexRotation, double overlap, double innerStrength, double outerStrength) {
    if (frame.activation == 0.0 || overlap <= 0.0) return {};

    const auto offset = v3_sub(rest, frame.bindPivot);
    const double along = v3_dot(offset, frame.bindAxis);
    const auto radial = v3_sub(offset, v3_scale(frame.bindAxis, along));
    const double radius = v3_len(radial);
    if (radius < 1e-8 || std::abs(along) >= 2.75 * radius) return {};

    // The sweep radius exceeds the rest radius to keep inside rings ordered.
    const double sweepRadius = 1.3 * radius;
    const double halfSpan = sweepRadius * std::tan(frame.angle * 0.5);
    const double t = std::clamp(0.5 + 0.5 * along / halfSpan, 0.0, 1.0);
    const double phase = t * frame.angle;
    const double sine = std::sin(phase);
    const double cosine = std::cos(phase);
    const auto tangent = v3_add(v3_scale(frame.incoming, cosine), v3_scale(frame.incomingNormal, sine));
    const auto inward = v3_cross(frame.bendNormal, tangent);
    auto center = v3_add(frame.posePivot,
        v3_add(v3_scale(frame.incoming, sweepRadius * sine - halfSpan),
            v3_scale(frame.incomingNormal, sweepRadius * (1.0 - cosine))));
    if (along < -halfSpan)
        center = v3_add(center, v3_scale(frame.incoming, along + halfSpan));
    if (along > halfSpan)
        center = v3_add(center, v3_scale(frame.outgoing, along - halfSpan));

    // Carry CoR's twist across while aligning the section with the sweep tangent.
    const auto posedAxis = q_rotate_v3(vertexRotation, frame.bindAxis);
    const auto align = q_between(posedAxis, tangent);
    const auto section = q_rotate_v3(align, q_rotate_v3(vertexRotation, radial));
    const double across = v3_dot(section, inward) / radius;
    const double side = smoothStep(-1.0, 1.0, across);
    const double strength = std::clamp(outerStrength * (1.0 - side) + innerStrength * side, 0.0, 1.0);

    // Bound the outer ring by the offsets of the two straight sides. A
    // smooth maximum rounds their intersection without pushing either side out.
    constexpr double cornerBlend = 0.12;
    const double outgoingCosine = std::cos(frame.angle - phase);
    const double difference = std::abs(cosine - outgoingCosine);
    const double blend = std::max(0.0, cornerBlend - difference);
    const double limitingCosine = std::min(1.0,
        std::max(cosine, outgoingCosine) + blend * blend / (4.0 * cornerBlend));
    const double outer = smoothStep(0.0, 1.0, -across);
    const double depth = std::max(0.0, -across * radius);
    const double expansion = outer * (sweepRadius + depth) * (1.0 / limitingCosine - 1.0);
    const auto target = v3_add(center, v3_sub(section, v3_scale(inward, expansion)));
    const double axialMask = 1.0 - smoothStep(1.35, 2.75, std::abs(along) / radius);
    const double weightMask = smoothStep(0.0, 0.12, overlap);
    return v3_scale(v3_sub(target, posed), frame.activation * strength * axialMask * weightMask);
}

} // namespace cor_contour
