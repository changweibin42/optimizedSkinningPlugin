#pragma once

#include <memory>
#include <optional>
#include <string>
#include <utility>
#include <vector>

#include <maya/MMatrix.h>
#include <maya/MObject.h>
#include <maya/MPointArray.h>

#include "cor_paper.h"

class SkinningGpu {
public:
    using VertexWeights = std::vector<std::vector<std::pair<unsigned int, float>>>;

    SkinningGpu();
    ~SkinningGpu();

    bool deformDLB(MPointArray& points, const VertexWeights& weights,
        unsigned long long weightVersion,
        const std::vector<mathcore::RigidTransform>& transforms,
        const MMatrix& worldMatrix, const MMatrix& worldInverse,
        double envelope, const MObject& node);

    struct CoRTiming {
        double kernelMs = -1.0;
        double openCLMs = -1.0;
    };

    bool deformCoR(MPointArray& points, const std::vector<cor_paper::Weights>& weights,
        const std::vector<mathcore::Vec3>& centers, unsigned long long bindGeneration,
        const std::vector<mathcore::RigidTransform>& transforms, const MMatrix& worldMatrix,
        double envelope, const MObject& node, CoRTiming* timing);

private:
    struct State;
    std::unique_ptr<State> corState_;
    std::optional<std::string> corStatus_;
    std::unique_ptr<State> state_;
    std::optional<std::string> reportedStatus_;
};
