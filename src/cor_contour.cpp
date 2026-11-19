#include "cor_contour.h"

#include <algorithm>
#include <cmath>

namespace cor_contour {
namespace {

double smoothStep(double start, double end, double value) {
    const double t = std::clamp((value - start) / (end - start), 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

// Recover the extra spacing near the joint across the full support. The
// cubic window gives a minimum slope of 1 - gain / 4 and joins the rest
// spacing with matching slope and curvature at the end of the region.
double redistributeRings(double along, double span) {
    constexpr double gain = 1.5;
    const double remaining = std::max(0.0, 1.0 - std::abs(along) / span);
    return along * (1.0 + gain * remaining * remaining * remaining);
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
        (1.0 - smoothStep(130.0 * degrees, 162.0 * degrees, frame.angle));
    frame.bendNormal = v3_scale(normal, 1.0 / sine);
    frame.incomingNormal = v3_cross(frame.bendNormal, frame.incoming);
    return frame;
}

Vec3 correction(const BendFrame& frame, const Vec3& rest, const Vec3& posed,
    const Quat& vertexRotation, double overlap, double innerStrength, double outerStrength, bool isolatedBend) {
    if (frame.activation == 0.0 || overlap <= 0.0) return {};

    const auto offset = v3_sub(rest, frame.bindPivot);
    const double along = v3_dot(offset, frame.bindAxis);
    const auto radial = v3_sub(offset, v3_scale(frame.bindAxis, along));
    const double radius = v3_len(radial);
    if (radius < 1e-8) return {};
    const double distance = std::abs(along) / radius;
    // On a two-joint mesh, keep the sweep tangent through the near straight
    // sections before handing back to CoR. Use local support for shared joints.
    const double fullSupport = isolatedBend ? 3.25 : 1.35;
    const double endSupport = isolatedBend ? 4.5 : 2.75;
    if (distance >= endSupport) return {};

    // The sweep radius exceeds the rest radius to keep inside rings ordered.
    const double sweepRadius = sweepRadiusRatio * radius;
    const double halfSpan = sweepRadius * std::tan(frame.angle * 0.5);
    const auto posedAxis = q_rotate_v3(vertexRotation, frame.bindAxis);
    const auto posedSection = q_rotate_v3(vertexRotation, radial);
    // Use one axial parameter for the full cross section. Different
    // inside/outside parameters shear rings even on the straight branches.
    const double spreadSpan = std::min(6.0 * halfSpan, endSupport * radius);
    const double sample = redistributeRings(along, spreadSpan);
    const double t = std::clamp(0.5 + 0.5 * sample / halfSpan, 0.0, 1.0);
    const double phase = t * frame.angle;
    const double sine = std::sin(phase);
    const double cosine = std::cos(phase);
    const auto tangent = v3_add(v3_scale(frame.incoming, cosine), v3_scale(frame.incomingNormal, sine));
    const auto inward = v3_cross(frame.bendNormal, tangent);
    auto center = v3_add(frame.posePivot,
        v3_add(v3_scale(frame.incoming, sweepRadius * sine - halfSpan),
            v3_scale(frame.incomingNormal, sweepRadius * (1.0 - cosine))));
    if (sample < -halfSpan)
        center = v3_add(center, v3_scale(frame.incoming, sample + halfSpan));
    if (sample > halfSpan)
        center = v3_add(center, v3_scale(frame.outgoing, sample - halfSpan));

    // Carry CoR's twist across while aligning the section with the sweep tangent.
    const auto align = q_between(posedAxis, tangent);
    const auto section = q_rotate_v3(align, posedSection);
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
    const double fade = std::clamp((distance - fullSupport) / (endSupport - fullSupport), 0.0, 1.0);
    const double axialMask = (1.0 - fade) * (1.0 - fade) * (1.0 - fade) *
        (1.0 + 3.0 * fade + 6.0 * fade * fade);
    const double weightMask = smoothStep(0.0, 0.12, overlap);
    return v3_scale(v3_sub(target, posed), frame.activation * strength * axialMask * weightMask);
}

} // namespace cor_contour
