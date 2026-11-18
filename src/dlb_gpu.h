#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <maya/MMatrix.h>
#include <maya/MObject.h>
#include <maya/MPointArray.h>

#include "math_core.h"

class DlbGpu {
public:
    using VertexWeights = std::vector<std::vector<std::pair<unsigned int, float>>>;

    DlbGpu();
    ~DlbGpu();

    bool deform(MPointArray& points, const VertexWeights& weights,
        unsigned long long weightVersion,
        const std::vector<mathcore::RigidTransform>& transforms,
        const MMatrix& worldMatrix, const MMatrix& worldInverse,
        double envelope, const MObject& node);

private:
    struct State;
    std::unique_ptr<State> state_;
    std::optional<std::string> reportedStatus_;
};
