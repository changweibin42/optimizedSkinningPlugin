#pragma once

#include <cstddef>

#include "cor_paper.h"

namespace ddm_v4 {

using Edge = std::array<unsigned int, 2>;

struct Settings {
    int iterations = 20;
    double lambda = 20.0;
    double kappa = 1.0;
    double alpha = 0.9;

    bool operator==(const Settings& other) const;
};

struct BindData {
    std::vector<mathcore::Vec3> centers;
    std::vector<cor_paper::Weights> rotationWeights;
    std::vector<cor_paper::Weights> translationWeights;
};

// Implicit diffusion on the undirected mesh edge graph. Unweighted
// vertices retain Maya's passthrough behavior and do not participate.
bool precompute(const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<cor_paper::Weights>& inputWeights, size_t influenceCount,
    const std::vector<Edge>& edges, const Settings& settings, BindData& output);

} // namespace ddm_v4
