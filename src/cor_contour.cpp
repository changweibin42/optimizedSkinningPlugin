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
    const auto incoming = q_rotate_v3(parentRotation, frame.bindAxis);
    const auto outgoing = q_rotate_v3(childRotation, frame.bindAxis);
    const auto bendNormal = v3_cross(incoming, outgoing);
    const double sine = v3_len(bendNormal);
    if (sine < 1e-8) return frame;

    constexpr double degrees = 3.14159265358979323846 / 180.0;
    const double angle = std::atan2(sine, v3_dot(incoming, outgoing));
    frame.activation = smoothStep(0.0, 90.0 * degrees, angle) *
        (1.0 - smoothStep(150.0 * degrees, 170.0 * degrees, angle));
    const auto planeNormal = v3_scale(bendNormal, 1.0 / sine);
    frame.incomingNormal = v3_cross(planeNormal, incoming);
    frame.outgoingNormal = v3_cross(planeNormal, outgoing);
    frame.inward = v3_norm(v3_add(frame.incomingNormal, frame.outgoingNormal));
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

    const auto axis = q_rotate_v3(vertexRotation, frame.bindAxis);
    const auto inward = v3_norm(v3_sub(frame.inward, v3_scale(axis, v3_dot(frame.inward, axis))));
    const double target = v3_dot(q_rotate_v3(vertexRotation, radial), inward);
    const double sideMask = smoothStep(0.15, 0.65, std::abs(target) / radius);
    const bool inner = target > 0.0;
    const double strength = inner ? innerStrength : outerStrength;
    if (sideMask == 0.0 || strength <= 0.0) return {};

    // Smooth the intersection of the two offset planes to round the corner.
    const auto fromPivot = v3_sub(posed, frame.posePivot);
    const double incomingDistance = v3_dot(fromPivot, frame.incomingNormal);
    const double outgoingDistance = v3_dot(fromPivot, frame.outgoingNormal);
    const double width = radius * (inner ? 0.15 : 0.40);
    const double blend = std::clamp(0.5 + 0.5 * (outgoingDistance - incomingDistance) / width, 0.0, 1.0);
    const double distance = blend * incomingDistance + (1.0 - blend) * outgoingDistance -
        width * blend * (1.0 - blend);
    const double deficit = target - distance;
    if ((inner && deficit <= 0.0) || (!inner && deficit >= 0.0)) return {};

    const auto gradient = v3_add(v3_scale(frame.incomingNormal, blend),
        v3_scale(frame.outgoingNormal, 1.0 - blend));
    const double gradientLength = v3_len(gradient);
    const double axialMask = 1.0 - smoothStep(0.35, 2.75, std::abs(along) / radius);
    const double amount = std::clamp(deficit / gradientLength, -0.6 * radius, 0.6 * radius);
    return v3_scale(gradient, amount / gradientLength *
        frame.activation * overlap * sideMask * axialMask * strength);
}

} // namespace cor_contour
