#include "optimizedSkinCluster.h"
#include "algorithms.h"
#include "node_ids.h"

#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MGlobal.h>
#include <maya/MItGeometry.h>
#include <maya/MDataBlock.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MPoint.h>
#include <maya/MPointArray.h>
#include <maya/MVector.h>
#include <maya/MMatrix.h>
#include <maya/MQuaternion.h>
#include <maya/MTransformationMatrix.h>

#include <unordered_map>
#include <vector>
#include <algorithm>
#include <cstdio>
#include <cmath>

MTypeId OptimizedSkinCluster::id(OPTIMIZED_SKIN_CLUSTER_NODE_ID);
MString OptimizedSkinCluster::typeName("optimizedSkinCluster");

MObject OptimizedSkinCluster::aAlgorithmType;
MObject OptimizedSkinCluster::aDQSCompStrength;

OptimizedSkinCluster::OptimizedSkinCluster() {}
OptimizedSkinCluster::~OptimizedSkinCluster() {}

void* OptimizedSkinCluster::creator() {
    return new OptimizedSkinCluster();
}

// ================================================================
// postConstructor — 每个节点实例的 envelope 默认值设为 1.0
// ================================================================
void OptimizedSkinCluster::postConstructor() {
    MPxSkinCluster::postConstructor();

    MStatus status;
    MFnDependencyNode fnNode(thisMObject(), &status);
    if (status == MS::kSuccess) {
        MPlug envPlug = fnNode.findPlug("envelope", true, &status);
        if (status == MS::kSuccess && !envPlug.isNull()) {
            envPlug.setDouble(1.0);
        }
    }
}

// ================================================================
// initialize
// ================================================================
MStatus OptimizedSkinCluster::initialize() {
    MStatus status;
    MFnEnumAttribute    eAttr;
    MFnNumericAttribute nAttr;

    aAlgorithmType = eAttr.create("algorithmType", "algo", 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Centers of Rotation", 0);
    eAttr.addField("Dual Quaternion Linear Blending", 1);
    eAttr.addField("DQS Compensation", 2);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    addAttribute(aAlgorithmType);

    aDQSCompStrength = nAttr.create("dqsCompStrength", "dqscs",
        MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    addAttribute(aDQSCompStrength);

    attributeAffects(aAlgorithmType, MPxSkinCluster::outputGeom);
    attributeAffects(aDQSCompStrength, MPxSkinCluster::outputGeom);

    return MS::kSuccess;
}

// ================================================================
// applyEnvelope
// ================================================================
mathcore::Vec3 OptimizedSkinCluster::applyEnvelope(
    const mathcore::Vec3& orig,
    const mathcore::Vec3& deformed,
    double envelope) {
    if (envelope >= 1.0 - 1e-6) return deformed;
    if (envelope <= 1e-6) return orig;
    return {
        envelope * deformed.x + (1.0 - envelope) * orig.x,
        envelope * deformed.y + (1.0 - envelope) * orig.y,
        envelope * deformed.z + (1.0 - envelope) * orig.z
    };
}

// ================================================================
// readEnvelopeFromNode — 从 MPlug 直读 envelope
// ================================================================
double OptimizedSkinCluster::readEnvelopeFromNode() {
    MPlug envPlug(thisMObject(), MPxGeometryFilter::envelope);
    if (envPlug.isNull()) return 1.0;
    return envPlug.asDouble();
}

// ================================================================
// setDependentsDirty — 权重变化时标记 CoR 缓存失效
// ================================================================
MStatus OptimizedSkinCluster::setDependentsDirty(
    const MPlug& plug,
    MPlugArray& /*plugArray*/) {

    MObject attr = plug.attribute();
    if (attr == MPxSkinCluster::weightList ||
        attr == MPxSkinCluster::weights) {
        m_corCache.valid = false;
    }
    return MS::kSuccess;
}

// ================================================================
// deform 主入口
// ================================================================
MStatus OptimizedSkinCluster::deform(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex) {

    MStatus status;
    MDataHandle algoHandle = block.inputValue(aAlgorithmType, &status);
    if (!status) return status;
    int algo = algoHandle.asShort();

    switch (algo) {
    case 0: return deformCoR(block, iter, worldMat, multiIndex);
    case 1: return deformDLB(block, iter, worldMat, multiIndex);
    case 2: return deformDQSComp(block, iter, worldMat, multiIndex);
    default: return MS::kFailure;
    }
}

// ================================================================
// readBoneTransforms
// ================================================================
bool OptimizedSkinCluster::readBoneTransforms(
    MDataBlock& block,
    std::vector<mathcore::RigidTransform>& out_rest,
    std::vector<mathcore::RigidTransform>& out_pose,
    std::vector<mathcore::Quat>& out_pose_qs) {

    MStatus status;

    MArrayDataHandle transformsHandle =
        block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return false;
    unsigned int nb_joints = transformsHandle.elementCount();
    if (nb_joints == 0) return false;

    MArrayDataHandle bindHandle =
        block.inputArrayValue(MPxSkinCluster::bindPreMatrix, &status);
    if (!status) return false;

    unsigned int max_idx = 0;
    transformsHandle.jumpToArrayElement(0);
    for (unsigned int i = 0; i < nb_joints; ++i) {
        unsigned int logical = transformsHandle.elementIndex();
        if (logical > max_idx) max_idx = logical;
        transformsHandle.next();
    }
    transformsHandle.jumpToArrayElement(0);

    std::vector<mathcore::Quat> pose_qs(max_idx + 1);
    std::vector<mathcore::Vec3> pose_ts(max_idx + 1);
    std::vector<mathcore::Quat> rest_qs(max_idx + 1);
    std::vector<mathcore::Vec3> rest_ts(max_idx + 1);

    for (unsigned int i = 0; i < nb_joints; ++i) {
        unsigned int logical = transformsHandle.elementIndex();
        MMatrix m = MFnMatrixData(transformsHandle.inputValue().data()).matrix();
        MTransformationMatrix tm(m);
        MQuaternion q = tm.rotation();
        mathcore::Vec3 t = { m[3][0], m[3][1], m[3][2] };
        pose_qs[logical] = { q.x, q.y, q.z, q.w };
        pose_ts[logical] = t;
        transformsHandle.next();
    }

    unsigned int bind_count = bindHandle.elementCount();
    if (bind_count > 0) {
        bindHandle.jumpToArrayElement(0);
        for (unsigned int i = 0; i < bind_count; ++i) {
            unsigned int logical = bindHandle.elementIndex();
            if (logical > max_idx) { bindHandle.next(); continue; }
            MMatrix m = MFnMatrixData(bindHandle.inputValue().data()).matrix();
            MTransformationMatrix tm(m);
            MQuaternion q = tm.rotation();
            mathcore::Vec3 t = { m[3][0], m[3][1], m[3][2] };
            rest_qs[logical] = { q.x, q.y, q.z, q.w };
            rest_ts[logical] = t;
            bindHandle.next();
        }
    }

    out_rest.resize(max_idx + 1);
    out_pose.resize(max_idx + 1);
    out_pose_qs.resize(max_idx + 1);
    for (unsigned int i = 0; i <= max_idx; ++i) {
        out_rest[i] = { rest_qs[i], rest_ts[i] };
        mathcore::Quat c_q = mathcore::q_mul(pose_qs[i], rest_qs[i]);
        mathcore::Vec3 rot_bt = mathcore::q_rotate_v3(pose_qs[i], rest_ts[i]);
        mathcore::Vec3 c_t = {
            pose_ts[i].x + rot_bt.x,
            pose_ts[i].y + rot_bt.y,
            pose_ts[i].z + rot_bt.z
        };
        out_pose[i] = { c_q, c_t };
        out_pose_qs[i] = pose_qs[i];
    }

    return true;
}

// ================================================================
// readVertexWeights
// ================================================================
static WeightMap readVertexWeights(MDataBlock& block, unsigned int vertexIdx) {
    WeightMap weights;
    MStatus status;

    MArrayDataHandle weightListHandle =
        block.inputArrayValue(MPxSkinCluster::weightList, &status);
    if (!status) return weights;
    if (weightListHandle.jumpToArrayElement(vertexIdx) != MS::kSuccess)
        return weights;

    MDataHandle wHandle = weightListHandle.inputValue(&status);
    if (!status) return weights;

    MArrayDataHandle weightsHandle = wHandle.child(MPxSkinCluster::weights);
    unsigned int n = weightsHandle.elementCount();
    if (n == 0) return weights;
    weightsHandle.jumpToArrayElement(0);
    for (unsigned int i = 0; i < n; ++i) {
        unsigned int logical = weightsHandle.elementIndex();
        double w = weightsHandle.inputValue().asDouble();
        weights[logical] = w;
        weightsHandle.next();
    }
    return weights;
}

// ================================================================
// computeCoRCache — 从权重分组自动预计算 CoR
// ================================================================
bool OptimizedSkinCluster::computeCoRCache(
    MDataBlock& block, unsigned int numVertices) {

    std::vector<WeightMap> allWeights(numVertices);
    for (unsigned int i = 0; i < numVertices; ++i) {
        allWeights[i] = readVertexWeights(block, i);
    }

    MStatus status;
    MArrayDataHandle inputArrayHandle =
        block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return false;
    inputArrayHandle.jumpToArrayElement(0);
    MDataHandle inputDataHandle = inputArrayHandle.inputValue(&status);
    if (!status) return false;
    MDataHandle geomHandle = inputDataHandle.child(MPxSkinCluster::inputGeom);
    if (!status) return false;
    MObject meshObj = geomHandle.asMesh();
    if (meshObj.isNull()) return false;

    MFnMesh meshFn(meshObj);
    MPointArray points;
    meshFn.getPoints(points, MSpace::kObject);
    if (points.length() < numVertices) return false;

    const double weight_epsilon = 0.05;

    auto similar = [&](const WeightMap& w1, const WeightMap& w2) -> bool {
        std::vector<unsigned int> keys;
        for (const auto& kv : w1) keys.push_back(kv.first);
        for (const auto& kv : w2) {
            if (std::find(keys.begin(), keys.end(), kv.first) == keys.end())
                keys.push_back(kv.first);
        }
        for (unsigned int k : keys) {
            double v1 = 0.0, v2 = 0.0;
            auto it1 = w1.find(k); if (it1 != w1.end()) v1 = it1->second;
            auto it2 = w2.find(k); if (it2 != w2.end()) v2 = it2->second;
            if (std::abs(v1 - v2) > weight_epsilon) return false;
        }
        return true;
        };

    struct Group {
        WeightMap refWeight;
        std::vector<unsigned int> verts;
    };
    std::vector<Group> groups;

    for (unsigned int i = 0; i < numVertices; ++i) {
        bool found = false;
        for (auto& g : groups) {
            if (similar(allWeights[i], g.refWeight)) {
                g.verts.push_back(i);
                found = true;
                break;
            }
        }
        if (!found) {
            Group g;
            g.refWeight = allWeights[i];
            g.verts.push_back(i);
            groups.push_back(g);
        }
    }

    m_corCache.cors.resize(numVertices);
    for (const auto& g : groups) {
        double cx = 0.0, cy = 0.0, cz = 0.0;
        for (unsigned int v : g.verts) {
            cx += points[v].x;
            cy += points[v].y;
            cz += points[v].z;
        }
        double n = static_cast<double>(g.verts.size());
        cx /= n; cy /= n; cz /= n;
        for (unsigned int v : g.verts) {
            m_corCache.cors[v] = { cx, cy, cz };
        }
    }

    m_corCache.numVertices = numVertices;
    m_corCache.valid = true;

    // 只在第一次分组成功时打印一次
    if (!m_corCache.loggedOnce && groups.size() > 1) {
        m_corCache.loggedOnce = true;
        char buf[256];
        snprintf(buf, sizeof(buf),
            "[optimizedSkinCluster] CoR cache: %d groups, %d verts",
            static_cast<int>(groups.size()),
            static_cast<int>(numVertices));
        MGlobal::displayInfo(MString(buf));
    }

    return true;
}

// ================================================================
// deformCoR
// ================================================================
MStatus OptimizedSkinCluster::deformCoR(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double envelope = readEnvelopeFromNode();

    std::vector<mathcore::RigidTransform> rest_transforms, Cs;
    std::vector<mathcore::Quat> pose_qs;
    if (!readBoneTransforms(block, rest_transforms, Cs, pose_qs))
        return MS::kSuccess;

    unsigned int numVertices = iter.count();

    if (!m_corCache.valid || m_corCache.numVertices != numVertices) {
        if (!computeCoRCache(block, numVertices)) {
            MGlobal::displayWarning(
                "[optimizedSkinCluster] CoR cache failed, fallback to DLB");
            return deformDLB(block, iter, worldMat, 0);
        }
    }

    MMatrix worldMatInverse = worldMat.inverse();

    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint origPtW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { origPtW.x, origPtW.y, origPtW.z };

        WeightMap weights = readVertexWeights(block, iter.index());
        if (weights.empty()) continue;

        unsigned int vtxIdx = iter.index();
        mathcore::Vec3 cor = origPos;
        if (vtxIdx < m_corCache.cors.size()) {
            mathcore::Vec3 corLocal = m_corCache.cors[vtxIdx];
            MPoint corMP(corLocal.x, corLocal.y, corLocal.z);
            MPoint corWorld = corMP * worldMat;
            cor = { corWorld.x, corWorld.y, corWorld.z };
        }

        mathcore::Vec3 deformed = cor_vertex(origPos, cor, weights, Cs);
        mathcore::Vec3 blended = applyEnvelope(origPos, deformed, envelope);

        MPoint out(blended.x, blended.y, blended.z);
        out *= worldMatInverse;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}

// ================================================================
// deformDLB
// ================================================================
MStatus OptimizedSkinCluster::deformDLB(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double envelope = readEnvelopeFromNode();

    std::vector<mathcore::RigidTransform> rest_transforms, Cs;
    std::vector<mathcore::Quat> pose_qs;
    if (!readBoneTransforms(block, rest_transforms, Cs, pose_qs))
        return MS::kSuccess;

    MMatrix worldMatInverse = worldMat.inverse();

    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint origPtW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { origPtW.x, origPtW.y, origPtW.z };

        WeightMap weights = readVertexWeights(block, iter.index());
        if (weights.empty()) continue;

        mathcore::Vec3 deformed = dlb_vertex(origPos, weights, Cs);
        mathcore::Vec3 blended = applyEnvelope(origPos, deformed, envelope);

        MPoint out(blended.x, blended.y, blended.z);
        out *= worldMatInverse;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}

// ================================================================
// deformDQSComp
// ================================================================
MStatus OptimizedSkinCluster::deformDQSComp(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double envelope = readEnvelopeFromNode();

    std::vector<mathcore::RigidTransform> rest_transforms, Cs;
    std::vector<mathcore::Quat> pose_qs;
    if (!readBoneTransforms(block, rest_transforms, Cs, pose_qs))
        return MS::kSuccess;

    double strength = block.inputValue(aDQSCompStrength).asDouble();
    MMatrix worldMatInverse = worldMat.inverse();

    unsigned int n = rest_transforms.size();
    std::vector<mathcore::Vec3> joint_positions(n, { 0, 0, 0 });
    std::vector<mathcore::Vec3> bone_dirs(n, { 0, 1, 0 });

    for (unsigned int i = 0; i < n; ++i) {
        mathcore::Quat r_inv_q = rest_transforms[i].q;
        mathcore::Vec3 r_inv_t = rest_transforms[i].t;
        mathcore::Quat r_q = mathcore::q_inverse(r_inv_q);
        mathcore::Vec3 rot = mathcore::q_rotate_v3(r_q, r_inv_t);
        joint_positions[i] = { -rot.x, -rot.y, -rot.z };
        bone_dirs[i] = mathcore::q_rotate_v3(pose_qs[i], { 0.0, 1.0, 0.0 });
    }

    for (iter.reset(); !iter.isDone(); iter.next()) {
        MPoint origPtW = iter.position() * worldMat;
        mathcore::Vec3 origPos = { origPtW.x, origPtW.y, origPtW.z };

        WeightMap weights = readVertexWeights(block, iter.index());
        if (weights.empty()) continue;

        mathcore::Vec3 dqs_p = dlb_vertex(origPos, weights, Cs);
        mathcore::Vec3 deformed = dqs_compensate_vertex(
            origPos, dqs_p, weights, Cs,
            bone_dirs, joint_positions, strength);

        mathcore::Vec3 blended = applyEnvelope(origPos, deformed, envelope);

        MPoint out(blended.x, blended.y, blended.z);
        out *= worldMatInverse;
        iter.setPosition(out);
    }
    return MS::kSuccess;
}