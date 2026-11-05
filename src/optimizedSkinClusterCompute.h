#pragma once

#include <maya/MPxDeformerNode.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>
#include <maya/MString.h>
#include <maya/MDataBlock.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>

#include <vector>
#include "math_core.h"
#include "skinClusterData.h"

class OptimizedSkinClusterCompute : public MPxDeformerNode {
public:
    OptimizedSkinClusterCompute();
    virtual ~OptimizedSkinClusterCompute();

    static void* creator();
    static MStatus initialize();

    virtual MStatus deform(MDataBlock& block, MItGeometry& iter,
        const MMatrix& mat, unsigned int multiIndex) override;

    virtual MStatus setDependentsDirty(const MPlug& plug,
        MPlugArray& plugArray) override;

    static MTypeId id;
    static MString typeName;

    static MObject aInData;
    static MObject aInUseGPU;

private:
    MStatus deformCoR(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int, SkinClusterData*);
    MStatus deformDLB(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int, SkinClusterData*);
    MStatus deformDQSComp(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int, SkinClusterData*);

    bool readBoneTransforms(SkinClusterData* sd,
        std::vector<mathcore::RigidTransform>& out_rest,
        std::vector<mathcore::RigidTransform>& out_pose,
        std::vector<mathcore::Quat>& out_pose_qs);

    double readEnvelopeFromNode();
    mathcore::Vec3 applyEnvelope(const mathcore::Vec3& orig,
        const mathcore::Vec3& deformed,
        double envelope);
};