#pragma once

#include <maya/MPxSkinCluster.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>
#include <maya/MString.h>
#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MPxGeometryFilter.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>

#include <vector>
#include "math_core.h"

class OptimizedSkinCluster : public MPxSkinCluster {
public:
    OptimizedSkinCluster();
    virtual ~OptimizedSkinCluster();

    static void* creator();
    static MStatus initialize();

    virtual void postConstructor() override;

    virtual MStatus deform(
        MDataBlock& block,
        MItGeometry& iter,
        const MMatrix& mat,
        unsigned int multiIndex
    ) override;

    virtual MStatus setDependentsDirty(
        const MPlug& plug,
        MPlugArray& plugArray
    ) override;

    static MTypeId id;
    static MString typeName;

    static MObject aAlgorithmType;
    static MObject aDQSCompStrength;

private:
    MStatus deformCoR(MDataBlock& block, MItGeometry& iter,
        const MMatrix& mat, unsigned int multiIndex);
    MStatus deformDLB(MDataBlock& block, MItGeometry& iter,
        const MMatrix& mat, unsigned int multiIndex);
    MStatus deformDQSComp(MDataBlock& block, MItGeometry& iter,
        const MMatrix& mat, unsigned int multiIndex);

    bool readBoneTransforms(MDataBlock& block,
        std::vector<mathcore::RigidTransform>& out_rest,
        std::vector<mathcore::RigidTransform>& out_pose,
        std::vector<mathcore::Quat>& out_pose_qs);

    // CoR 缓存
    struct CoRCache {
        bool valid = false;
        unsigned int numVertices = 0;
        std::vector<mathcore::Vec3> cors;
        bool loggedOnce = false;
    };
    CoRCache m_corCache;

    // 自动预计算 CoR
    bool computeCoRCache(MDataBlock& block, unsigned int numVertices);

    // 从节点属性读 envelope（MPlug 直读，避免 MDataBlock 缓存问题）
    double readEnvelopeFromNode();

    // envelope 线性混合
    mathcore::Vec3 applyEnvelope(const mathcore::Vec3& orig,
        const mathcore::Vec3& deformed,
        double envelope);
};