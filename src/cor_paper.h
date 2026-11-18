#pragma once

#include <array>
#include <utility>
#include <vector>

#include "math_core.h"

namespace cor_paper {

using Weights = std::vector<std::pair<unsigned int, double>>;
using Triangle = std::array<unsigned int, 3>;

// Weight lists are normalized and ordered by increasing influence index.
std::vector<mathcore::Vec3> computeCenters(
    const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<Weights>& weights,
    const std::vector<Triangle>& triangles,
    double sigma = 0.1);

// The center stays in bind space. The returned transform acts in the
// coordinate system of bindToPose and separates rotation from translation.
mathcore::RigidTransform vertexTransform(const mathcore::Vec3& center,
    const Weights& weights,
    const std::vector<mathcore::RigidTransform>& bindToPose);

} // namespace cor_paper
