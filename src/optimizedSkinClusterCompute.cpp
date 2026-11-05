#include "optimizedSkinClusterCompute.h"
#include "algorithms.h"
#include "node_ids.h"
#include "plugin_debug.h"

#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MFnPluginData.h>
#include <maya/MGlobal.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MQuaternion.h>
#include <maya/MTransformationMatrix.h>

#include <vector>
#include <cmath>

MTypeId OptimizedSkinClusterCompute::id(OPTIMIZED_SKIN_COMPUTE_NODE_ID);
MString OptimizedSkinClusterCompute::typeName("optimizedSkinClusterCompute");

MObject OptimizedSkinClusterCompute::aInData;
MObject OptimizedSkinClusterCompute::aInUseGPU;

OptimizedSkinClusterCompute::OptimizedSkinClusterCompute() {}
OptimizedSkinClusterCompute::~OptimizedSkinClusterCompute() {}
void* OptimizedSkinClusterCompute::creator() { return new OptimizedSkinClusterCompute(); }

MStatus OptimizedSkinClusterCompute::initialize() {
    MStatus status;
    MFnNumericAttribute nAttr;
    MFnTypedAttribute   tAttr;

    aInData = tAttr.create("inData", "ind",
        SkinClusterData::id,
        MObject::kNullObj,
        &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    tAttr.setStorable(false);
    tAttr.setConnectable(true);
    tAttr.setWritable(true);
    tAttr.setReadable(true);
    tAttr.setKeyable(false);
    addAttribute(aInData);

    aInUseGPU = nAttr.create("inUseGPU", "inugpu",
        MFnNumericData::kBoolean, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setStorable(true);
    nAttr.setConnectable(true);
    nAttr.setKeyable(false);
    addAttribute(aInUseGPU);

    attributeAffects(aInData, MPxDeformerNode::outputGeom);
    attributeAffects(aInUseGPU, MPxDeformerNode::outputGeom);

    return MS::kSuccess;
}

MStatus OptimizedSkinClusterCompute::setDependentsDirty(
    const MPlug& /*plug*/, MPlugArray& /*plugArray*/) {
    return MS::kSuccess;
}

mathcore::Vec3 OptimizedSkinClusterCompute::applyEnvelope(
    const mathcore::Vec3& orig, const mathcore::Vec3& deformed, double env) {
    if (env >= 1.0 - 1e-6) return deformed;
    if (env <= 1e-6) return orig;
    return { env * deformed.x + (1.0 - env) * orig.x,
             env * deformed.y + (1.0 - env) * orig.y,
             env * deformed.z + (1.0 - env) * orig.z };
}

double OptimizedSkinClusterCompute::readEnvelopeFromNode() {
    MPlug p(thisMObject(), MPxDeformerNode::envelope);
    return p.isNull() ? 1.0 : p.asDouble();
}

bool OptimizedSkinClusterCompute::readBoneTransforms(
    SkinClusterData* sd,
    std::vector<mathcore::RigidTransform>& out_rest,
    std::vector<mathcore::RigidTransform>& out_pose,
    std::vector<mathcore::Quat>& out_pose_qs) {

    if (!sd) return false;
    unsigned int nB = (unsigned int)sd->restList.size();
    if (nB == 0 || sd->poseList.size() != nB) return false;

    out_rest.resize(nB);
    out_pose.resize(nB);
    out_pose_qs.resize(nB);
    for (unsigned int i = 0; i < nB; ++i) {
        const RigidData& r = sd->restList[i];
        const RigidData& p = sd->poseList[i];
        out_rest[i].q = { r.qx, r.qy, r.qz, r.qw };
        out_rest[i].t = { r.tx, r.ty, r.tz };
        out_pose[i].q = { p.qx, p.qy, p.qz, p.qw };
        out_pose[i].t = { p.tx, p.ty, p.tz };
        out_pose_qs[i] = out_pose[i].q;
    }
    return true;
}

static WeightMap readVertexWeightsB(SkinClusterData* sd, unsigned int vtxIdx) {
    WeightMap w;
    if (!sd) return w;
    if (vtxIdx >= sd->vertices.size()) return w;
    const SkinVertex& sv = sd->vertices[vtxIdx];
    for (int k = 0; k < sv.boneCount; ++k) {
        w[sv.boneIds[k]] = sv.weights[k];
    }
    return w;
}

MStatus OptimizedSkinClusterCompute::deform(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex) {

    MStatus status;

    MDataHandle h = block.inputValue(aInData, &status);
    if (!status) return MS::kSuccess;
    MObject obj = h.data();
    if (obj.isNull()) return MS::kSuccess;

    MFnPluginData fnD(obj, &status);
    if (!status) return MS::kSuccess;
    SkinClusterData* sd = dynamic_cast<SkinClusterData*>(fnD.data(&status));
    if (!status || !sd) return MS::kSuccess;

    switch (sd->algorithmType) {
    case 0: return deformCoR(block, iter, worldMat, multiIndex, sd);
    case 1: return deformDLB(block, iter, worldMat, multiIndex, sd);
    case 2: return deformDQSComp(block, iter, worldMat, multiIndex, sd);
    default: return MS::kFailure;
    }
}

MStatus OptimizedSkinClusterCompute::deformCoR(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/,
    SkinClusterData* sd) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(sd, rest, Cs, poseQ)) {
        return MS::kSuccess;
    }

    unsigned int nV = iter.count();

    if (sd->cors.size() < nV) {
        if (optSkinDebugLog()) {
            MGlobal::displayWarning(
                "[optimizedSkinClusterCompute] No CoR from A, fallback DLB");
        }
        return deformDLB(block, iter, worldMat, 0, sd);
    }

    MMatrix wInv = worldMat.inverse();
    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint ptW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };
        WeightMap weights = readVertexWeightsB(sd, iter.index());
        if (weights.empty()) continue;

        unsigned int vi = iter.index();
        mathcore::Vec3 corLocal = { sd->cors[vi].x,
                                    sd->cors[vi].y,
                                    sd->cors[vi].z };
        MPoint cw(corLocal.x, corLocal.y, corLocal.z);
        cw *= worldMat;
        mathcore::Vec3 cor = { cw.x, cw.y, cw.z };

        mathcore::Vec3 def = cor_vertex(origPos, cor, weights, Cs);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, env);
        MPoint out(bl.x, bl.y, bl.z); out *= wInv;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}

MStatus OptimizedSkinClusterCompute::deformDLB(
    MDataBlock& /*block*/, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/,
    SkinClusterData* sd) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(sd, rest, Cs, poseQ)) return MS::kSuccess;

    MMatrix wInv = worldMat.inverse();
    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint ptW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };
        WeightMap weights = readVertexWeightsB(sd, iter.index());
        if (weights.empty()) continue;
        mathcore::Vec3 def = dlb_vertex(origPos, weights, Cs);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, env);
        MPoint out(bl.x, bl.y, bl.z); out *= wInv;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}

// ★ DQS 补偿 —— 恢复上一版调用（不传 poseQ）
MStatus OptimizedSkinClusterCompute::deformDQSComp(
    MDataBlock& /*block*/, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/,
    SkinClusterData* sd) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(sd, rest, Cs, poseQ)) return MS::kSuccess;

    double strength = sd->dqsStrength;
    MMatrix wInv = worldMat.inverse();

    unsigned int n = (unsigned int)rest.size();
    std::vector<mathcore::Vec3> jp(n, { 0,0,0 });
    std::vector<mathcore::Vec3> bd(n, { 0,1,0 });
    for (unsigned int i = 0; i < n; ++i) {
        mathcore::Quat rq = mathcore::q_inverse(rest[i].q);
        mathcore::Vec3 rot = mathcore::q_rotate_v3(rq, rest[i].t);
        jp[i] = { -rot.x, -rot.y, -rot.z };

        mathcore::Vec3 restDir = { 0.0, 1.0, 0.0 };
        if (i < sd->boneDirs.size()) {
            restDir = { sd->boneDirs[i].x, sd->boneDirs[i].y, sd->boneDirs[i].z };
        }
        bd[i] = mathcore::q_rotate_v3(Cs[i].q, restDir);
    }

    const std::vector<int>& depths = sd->depths;

    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint ptW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };
        WeightMap weights = readVertexWeightsB(sd, iter.index());
        if (weights.empty()) continue;
        mathcore::Vec3 dqs = dlb_vertex(origPos, weights, Cs);
        // ★ 上一版：不传 poseQ
        mathcore::Vec3 def = dqs_compensate_vertex(
            origPos, dqs, weights, Cs, bd, jp, depths, strength);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, env);
        MPoint out(bl.x, bl.y, bl.z); out *= wInv;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}