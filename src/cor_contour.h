#pragma once

#include "math_core.h"

namespace cor_contour {

struct BendFrame {
    mathcore::Vec3 bindPivot{};
    mathcore::Vec3 bindAxis{};
    mathcore::Vec3 posePivot{};
    mathcore::Vec3 incomingNormal{};
    mathcore::Vec3 outgoingNormal{};
    mathcore::Vec3 inward{};
    double activation = 0.0;
};

BendFrame buildFrame(const mathcore::Vec3& bindParent, const mathcore::Vec3& bindPivot,
    const mathcore::Vec3& posePivot, const mathcore::Quat& parentRotation,
    const mathcore::Quat& childRotation);

mathcore::Vec3 correction(const BendFrame& frame, const mathcore::Vec3& rest,
    const mathcore::Vec3& posed, const mathcore::Quat& vertexRotation,
    double overlap, double innerStrength, double outerStrength);

} // namespace cor_contour
