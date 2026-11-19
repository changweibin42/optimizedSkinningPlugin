#pragma once

#include "math_core.h"

namespace cor_contour {

inline constexpr double sweepRadiusRatio = 1.14;

struct BendFrame {
    mathcore::Vec3 bindPivot{};
    mathcore::Vec3 bindAxis{};
    mathcore::Vec3 posePivot{};
    mathcore::Vec3 incoming{};
    mathcore::Vec3 outgoing{};
    mathcore::Vec3 incomingNormal{};
    mathcore::Vec3 bendNormal{};
    double angle = 0.0;
    double activation = 0.0;
};

BendFrame buildFrame(const mathcore::Vec3& bindParent, const mathcore::Vec3& bindPivot,
    const mathcore::Vec3& posePivot, const mathcore::Quat& parentRotation,
    const mathcore::Quat& childRotation);

struct SurfaceSample {
    mathcore::Vec3 center{};
    mathcore::Vec3 radial{};
    mathcore::Vec3 tangent{};
    mathcore::Vec3 inward{};
    double across = 0.0;
    double influence = 0.0;
    double phase = 0.0;
};

SurfaceSample sampleSurface(const BendFrame& frame, const mathcore::Vec3& rest,
    const mathcore::Quat& sectionRotation, bool isolatedBend);

mathcore::Vec3 correction(const BendFrame& frame, const mathcore::Vec3& rest,
    const mathcore::Vec3& posed, const mathcore::Quat& vertexRotation,
    double overlap, double innerStrength, double outerStrength,
    bool isolatedBend = false);

} // namespace cor_contour
