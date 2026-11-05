#include "skinClusterData.h"

const MString SkinClusterData::typeName("skinClusterData");
MTypeId       SkinClusterData::id(0x00142A10);

SkinClusterData::SkinClusterData() {}
SkinClusterData::~SkinClusterData() {}

MTypeId SkinClusterData::typeId() const { return SkinClusterData::id; }
MString SkinClusterData::name()   const { return SkinClusterData::typeName; }

void SkinClusterData::copy(const MPxData& other) {
    const SkinClusterData& o = static_cast<const SkinClusterData&>(other);
    algorithmType = o.algorithmType;
    dqsStrength = o.dqsStrength;
    useGPU = o.useGPU;
    weightVersion = o.weightVersion;
    restList = o.restList;
    poseList = o.poseList;
    vertices = o.vertices;
    cors = o.cors;
    boneDirs = o.boneDirs;
    jointPositions = o.jointPositions;
    depths = o.depths;
}

void* SkinClusterData::creator() {
    return new SkinClusterData();
}