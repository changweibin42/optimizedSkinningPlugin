#include "optimizedSkinCluster.h"
#include "algorithms.h"
#include "node_ids.h"
#include "plugin_debug.h"

#include <maya/MFnEnumAttribute.h>
#include <maya/MFnNumericAttribute.h>
#include <maya/MFnUnitAttribute.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnMesh.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MFnDagNode.h>
#include <maya/MDagPath.h>
#include <maya/MFnTypedAttribute.h>
#include <maya/MGlobal.h>
#include <maya/MDataHandle.h>
#include <maya/MArrayDataHandle.h>
#include <maya/MPoint.h>
#include <maya/MIntArray.h>
#include <maya/MQuaternion.h>
#include <maya/MTransformationMatrix.h>

#include <unordered_map>
#include <vector>
#include <utility>
#include <functional>
#include <algorithm>
#include <cmath>
#include <climits>
#include <chrono>
#include <cstdio>
#include <string>

using optimized_skin_internal::SparseW;
using optimized_skin_internal::TriInfo;
using optimized_skin_internal::SubTri;
using optimized_skin_internal::weightL2DistSq;
using optimized_skin_internal::paper_similarity;
using optimized_skin_internal::midSparseW;
using optimized_skin_internal::avg3SparseW;
using optimized_skin_internal::sparseEqual;
using optimized_skin_internal::buildSingleTriInfo;
using optimized_skin_internal::subdivideSingleTriangle;
using optimized_skin_internal::computeSingleVertexCoR;

MTypeId OptimizedSkinCluster::id(OPTIMIZED_SKIN_CLUSTER_NODE_ID);
MString OptimizedSkinCluster::typeName("optimizedSkinCluster");

MObject OptimizedSkinCluster::aAlgorithmType;
MObject OptimizedSkinCluster::aDQSCompStrength;
MObject OptimizedSkinCluster::aForceFullCoR;
MObject OptimizedSkinCluster::aCorTriLayerExpand;
MObject OptimizedSkinCluster::aCorVertLayerExpand;
MObject OptimizedSkinCluster::aCorIncBudgetDiv;
MObject OptimizedSkinCluster::aDisplayCoR;

MObject OptimizedSkinCluster::aMosInfluenceRadius;
MObject OptimizedSkinCluster::aMosCurveMode;
MObject OptimizedSkinCluster::aMosEndLock;
MObject OptimizedSkinCluster::aMosCurveSamples;
MObject OptimizedSkinCluster::aMosDebugLog;

MObject OptimizedSkinCluster::aMosInBulgeWeight;
MObject OptimizedSkinCluster::aMosOutBulgeWeight;
MObject OptimizedSkinCluster::aMosSideBulgeWeight; 
MObject OptimizedSkinCluster::aMosCenterSlideInWeight;
MObject OptimizedSkinCluster::aMosCenterSlideOutWeight;

MObject OptimizedSkinCluster::aMosSlideInWeightPrev;
MObject OptimizedSkinCluster::aMosSlideInWeightNext;
MObject OptimizedSkinCluster::aMosSlideOutWeightPrev;
MObject OptimizedSkinCluster::aMosSlideOutWeightNext;
MObject OptimizedSkinCluster::aMosSlideOrthoInWeightPrev;
MObject OptimizedSkinCluster::aMosSlideOrthoInWeightNext;
MObject OptimizedSkinCluster::aMosSlideOrthoOutWeightPrev;
MObject OptimizedSkinCluster::aMosSlideOrthoOutWeightNext;

MObject OptimizedSkinCluster::aMosUseManualInVector;
MObject OptimizedSkinCluster::aMosManualInVector;

MObject OptimizedSkinCluster::aMosTwistWeight;

MObject OptimizedSkinCluster::aMosScaleWeight;
MObject OptimizedSkinCluster::aMosScaleMode;

OptimizedSkinCluster::OptimizedSkinCluster() {}
OptimizedSkinCluster::~OptimizedSkinCluster() {}
void* OptimizedSkinCluster::creator() { return new OptimizedSkinCluster(); }

// ================================================================
// 单顶点权重读取
// ================================================================
static WeightMap readVertexWeightsFromHandle(
    MArrayDataHandle& wlH, unsigned int vtxIdx) {

    WeightMap w;
    if (!wlH.jumpToArrayElement(vtxIdx)) return w;
    unsigned int got = wlH.elementIndex();
    if (got == static_cast<unsigned int>(-1) || got != vtxIdx) return w;

    MStatus status;
    MDataHandle wH = wlH.inputValue(&status);
    if (!status) return w;

    MArrayDataHandle wiH = wH.child(MPxSkinCluster::weights);
    unsigned int n = wiH.elementCount();
    if (n == 0) return w;
    if (n > 1024) n = 1024;

    for (unsigned int i = 0; i < n; ++i) {
        if (!wiH.jumpToArrayElement(i)) continue;
        unsigned int idx = wiH.elementIndex();
        if (idx == static_cast<unsigned int>(-1) || idx > 1024) continue;
        MStatus s;
        MDataHandle h = wiH.inputValue(&s);
        if (!s) continue;
        double v = h.asDouble();
        if (v > 1e-12) w[idx] = v;
    }

    if (!w.empty()) {
        double sum = 0.0;
        for (const auto& kv : w) sum += kv.second;
        if (sum > 1e-12) for (auto& kv : w) kv.second /= sum;
    }
    return w;
}

// ================================================================
// postConstructor
// ================================================================
void OptimizedSkinCluster::postConstructor() {
    MPxSkinCluster::postConstructor();
    MStatus status;
    MFnDependencyNode fn(thisMObject(), &status);
    if (status) {
        MPlug env = fn.findPlug("envelope", true, &status);
        if (status && !env.isNull()) {
            env.setDouble(1.0);
            m_envelopePlug = env;
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
    eAttr.addField("MOS", 3);
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

    aForceFullCoR = nAttr.create("forceFullCoR", "ffcor",
        MFnNumericData::kBoolean, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(false);
    nAttr.setConnectable(false);
    nAttr.setChannelBox(false);
    addAttribute(aForceFullCoR);
    attributeAffects(aForceFullCoR, MPxSkinCluster::outputGeom);

    aCorTriLayerExpand = nAttr.create("corTriLayerExpand", "ctle",
        MFnNumericData::kInt, 1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0);
    nAttr.setMax(4);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Tri Layers");
    addAttribute(aCorTriLayerExpand);
    attributeAffects(aCorTriLayerExpand, MPxSkinCluster::outputGeom);

    aCorVertLayerExpand = nAttr.create("corVertLayerExpand", "cvle",
        MFnNumericData::kInt, -1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-1);
    nAttr.setMax(5);
    nAttr.setSoftMin(-1);
    nAttr.setSoftMax(3);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Vert Layers");
    addAttribute(aCorVertLayerExpand);
    attributeAffects(aCorVertLayerExpand, MPxSkinCluster::outputGeom);

    aCorIncBudgetDiv = nAttr.create("corIncBudgetDiv", "cibd",
        MFnNumericData::kInt, 10, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(1);
    nAttr.setMax(100);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Inc Budget");
    addAttribute(aCorIncBudgetDiv);
    attributeAffects(aCorIncBudgetDiv, MPxSkinCluster::outputGeom);

    aDisplayCoR = nAttr.create("displayCoR", "dcr",
        MFnNumericData::kBoolean, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setChannelBox(true);
    nAttr.setNiceNameOverride("Show CoR");
    addAttribute(aDisplayCoR);

    // ================================================================
    // MOS 可调参数
    // ================================================================
    aMosInfluenceRadius = nAttr.create("mosInfluenceRadius", "mosR",
        MFnNumericData::kDouble, 0.6, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setSoftMax(5.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Weight");
    addAttribute(aMosInfluenceRadius);
    attributeAffects(aMosInfluenceRadius, MPxSkinCluster::outputGeom);

    aMosCurveMode = eAttr.create("mosCurveMode", "mosCm", 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Single Long Curve", 0);
    eAttr.addField("Per-Joint Curves", 1);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    eAttr.setNiceNameOverride("MOS Curve Mode");
    addAttribute(aMosCurveMode);
    attributeAffects(aMosCurveMode, MPxSkinCluster::outputGeom);

    aMosEndLock = nAttr.create("mosEndLock", "mosEl",
        MFnNumericData::kBoolean, 1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS End Lock");
    addAttribute(aMosEndLock);
    attributeAffects(aMosEndLock, MPxSkinCluster::outputGeom);

    aMosCurveSamples = nAttr.create("mosCurveSamples", "mosCs",
        MFnNumericData::kInt, 1024, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(64);
    nAttr.setMax(8192);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Samples");
    addAttribute(aMosCurveSamples);
    attributeAffects(aMosCurveSamples, MPxSkinCluster::outputGeom);

    aMosDebugLog = nAttr.create("mosDebugLog", "mosDl",
        MFnNumericData::kBoolean, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Debug");
    addAttribute(aMosDebugLog);

    // ---- Bulge 组 ----
    aMosInBulgeWeight = nAttr.create("mosInBulgeWeight", "mosIbw",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS In Bulge");
    addAttribute(aMosInBulgeWeight);
    attributeAffects(aMosInBulgeWeight, MPxSkinCluster::outputGeom);

    aMosOutBulgeWeight = nAttr.create("mosOutBulgeWeight", "mosObw",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Out Bulge");
    addAttribute(aMosOutBulgeWeight);
    attributeAffects(aMosOutBulgeWeight, MPxSkinCluster::outputGeom);

    aMosSideBulgeWeight = nAttr.create("mosSideBulgeWeight", "mosSbw",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Side Bulge");
    addAttribute(aMosSideBulgeWeight);
    attributeAffects(aMosSideBulgeWeight, MPxSkinCluster::outputGeom);

    // ---- Slide 组 ----
    aMosSlideInWeightPrev = nAttr.create("mosSlideInWeightPrev", "mosSlIp",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide In Prev");
    addAttribute(aMosSlideInWeightPrev);
    attributeAffects(aMosSlideInWeightPrev, MPxSkinCluster::outputGeom);

    aMosSlideInWeightNext = nAttr.create("mosSlideInWeightNext", "mosSlIn",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide In Next");
    addAttribute(aMosSlideInWeightNext);
    attributeAffects(aMosSlideInWeightNext, MPxSkinCluster::outputGeom);

    aMosSlideOutWeightPrev = nAttr.create("mosSlideOutWeightPrev", "mosSlOp",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide Out Prev");
    addAttribute(aMosSlideOutWeightPrev);
    attributeAffects(aMosSlideOutWeightPrev, MPxSkinCluster::outputGeom);

    aMosSlideOutWeightNext = nAttr.create("mosSlideOutWeightNext", "mosSlOn",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide Out Next");
    addAttribute(aMosSlideOutWeightNext);
    attributeAffects(aMosSlideOutWeightNext, MPxSkinCluster::outputGeom);

    aMosSlideOrthoInWeightPrev = nAttr.create("mosSlideOrthoInWeightPrev", "mosSoIp",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide O-In Prev");
    addAttribute(aMosSlideOrthoInWeightPrev);
    attributeAffects(aMosSlideOrthoInWeightPrev, MPxSkinCluster::outputGeom);

    aMosSlideOrthoInWeightNext = nAttr.create("mosSlideOrthoInWeightNext", "mosSoIn",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide O-In Next");
    addAttribute(aMosSlideOrthoInWeightNext);
    attributeAffects(aMosSlideOrthoInWeightNext, MPxSkinCluster::outputGeom);

    aMosSlideOrthoOutWeightPrev = nAttr.create("mosSlideOrthoOutWeightPrev", "mosSoOp",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide O-Out Prev");
    addAttribute(aMosSlideOrthoOutWeightPrev);
    attributeAffects(aMosSlideOrthoOutWeightPrev, MPxSkinCluster::outputGeom);

    aMosSlideOrthoOutWeightNext = nAttr.create("mosSlideOrthoOutWeightNext", "mosSoOn",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-10.0);
    nAttr.setMax(10.0);
    nAttr.setSoftMin(-2.0);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Slide O-Out Next");
    addAttribute(aMosSlideOrthoOutWeightNext);
    attributeAffects(aMosSlideOrthoOutWeightNext, MPxSkinCluster::outputGeom);

    // ---- 手动 in vector ----
    aMosUseManualInVector = nAttr.create("mosUseManualInVector", "mosUmiv",
        MFnNumericData::kBoolean, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Manual In");
    addAttribute(aMosUseManualInVector);
    attributeAffects(aMosUseManualInVector, MPxSkinCluster::outputGeom);

    aMosManualInVector = nAttr.create("mosManualInVector", "mosMiv",
        MFnNumericData::k3Double, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setDefault(0.0, 1.0, 0.0);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS In Vector");
    addAttribute(aMosManualInVector);
    attributeAffects(aMosManualInVector, MPxSkinCluster::outputGeom);

    // ---- Twist（当前不生效，保留供 ghost transform 使用）----
    aMosTwistWeight = nAttr.create("mosTwistWeight", "mosTw",
        MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Twist (inactive)");
    addAttribute(aMosTwistWeight);
    attributeAffects(aMosTwistWeight, MPxSkinCluster::outputGeom);

    // ---- Scale ----
    aMosScaleWeight = nAttr.create("mosScaleWeight", "mosScw",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Scale");
    addAttribute(aMosScaleWeight);
    attributeAffects(aMosScaleWeight, MPxSkinCluster::outputGeom);

    aMosScaleMode = eAttr.create("mosScaleMode", "mosScm", 1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Off", 0);
    eAttr.addField("Axis Symmetric", 1);
    eAttr.addField("Anisotropic", 2);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    eAttr.setNiceNameOverride("MOS Scale Mode");
    addAttribute(aMosScaleMode);
    attributeAffects(aMosScaleMode, MPxSkinCluster::outputGeom);

    // ---- Center Slide（当前死代码，下轮清理时一并删除）----
    aMosCenterSlideInWeight = nAttr.create("mosCenterSlideInWeight", "mosCsi",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Center Slide In");
    addAttribute(aMosCenterSlideInWeight);
    attributeAffects(aMosCenterSlideInWeight, MPxSkinCluster::outputGeom);

    aMosCenterSlideOutWeight = nAttr.create("mosCenterSlideOutWeight", "mosCso",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Center Slide Out");
    addAttribute(aMosCenterSlideOutWeight);
    attributeAffects(aMosCenterSlideOutWeight, MPxSkinCluster::outputGeom);

    return MS::kSuccess;
}

// ================================================================
// setDependentsDirty
// ================================================================
MStatus OptimizedSkinCluster::setDependentsDirty(
    const MPlug& plug, MPlugArray& /*plugArray*/) {

    MObject attr = plug.attribute();

    if (attr == aDisplayCoR) {
        MPlug displayPlug(thisMObject(), aDisplayCoR);
        bool displayOn = displayPlug.asBool();
        if (displayOn) {
            MFnDependencyNode fn(thisMObject());
            MString nodeName = fn.name();
            MString mel;
            mel.format(
                "scriptJob -runOnce true -idleEvent "
                "\"if (`objExists ^1s` && `getAttr ^1s.displayCoR` && "
                "!`objExists ^1s_corViewer`) {"
                "  optimizedSkinClusterCoR -create -node ^1s;"
                "}\"",
                nodeName, nodeName, nodeName, nodeName);
            MGlobal::executeCommand(mel, false, false);
        }
        return MS::kSuccess;
    }

    if (attr == aMosInfluenceRadius || attr == aMosCurveMode ||
        attr == aMosEndLock || attr == aMosCurveSamples ||
        attr == aMosInBulgeWeight ||
        attr == aMosOutBulgeWeight ||
        attr == aMosSideBulgeWeight ||
        attr == aMosCenterSlideInWeight ||
        attr == aMosCenterSlideOutWeight ||
        attr == aMosSlideInWeightPrev ||
        attr == aMosSlideInWeightNext ||
        attr == aMosSlideOutWeightPrev ||
        attr == aMosSlideOutWeightNext ||
        attr == aMosSlideOrthoInWeightPrev ||
        attr == aMosSlideOrthoInWeightNext ||
        attr == aMosSlideOrthoOutWeightPrev ||
        attr == aMosSlideOrthoOutWeightNext ||
        attr == aMosUseManualInVector ||
        attr == aMosManualInVector ||
        attr == aMosTwistWeight ||
        attr == aMosScaleWeight ||
        attr == aMosScaleMode) {
        m_mosCache.valid = false;
        if (optSkinDebugLog()) {
            MFnDependencyNode fn(thisMObject());
            MGlobal::displayInfo(MString("[MOS] cache invalidated by ") + fn.name());
        }
        return MS::kSuccess;
    }

    if (attr == MPxSkinCluster::weightList) {
        m_needFullWeightScan = true;
        m_vtxCache.valid = false;
        m_mosCache.valid = false;
        ++m_currentWeightVersion;
        return MS::kSuccess;
    }

    if (attr == MPxSkinCluster::weights) {
        MPlug p = plug;
        int safety = 0;
        while (!p.isNull() &&
            p.attribute() != MPxSkinCluster::weightList &&
            safety++ < 4) p = p.parent();
        if (!p.isNull() && p.attribute() == MPxSkinCluster::weightList) {
            int idx = p.logicalIndex();
            if (idx >= 0) {
                markDirtyVert((unsigned int)idx);
                m_hasPlugDirty = true;
                m_vtxCache.valid = false;
                m_mosCache.valid = false;
                ++m_currentWeightVersion;
                return MS::kSuccess;
            }
        }
        m_needFullWeightScan = true;
        m_vtxCache.valid = false;
        m_mosCache.valid = false;
        ++m_currentWeightVersion;
        return MS::kSuccess;
    }
    return MS::kSuccess;
}

void OptimizedSkinCluster::markDirtyVert(unsigned int vi) {
    if (m_dirtyVertFlags.size() <= vi) m_dirtyVertFlags.resize(vi + 1, 0);
    m_dirtyVertFlags[vi] = 1;
}

void OptimizedSkinCluster::clearDirtyFlags() {
    if (!m_dirtyVertFlags.empty()) {
        std::fill(m_dirtyVertFlags.begin(), m_dirtyVertFlags.end(), 0);
        if (m_dirtyVertFlags.size() > 100000) {
            m_dirtyVertFlags.clear();
            m_dirtyVertFlags.shrink_to_fit();
        }
    }
    m_hasPlugDirty = false;
    m_needFullWeightScan = false;
}

int OptimizedSkinCluster::computeVertLayers(
    unsigned int dirtyCount, unsigned int numVertices,
    int vertLayerOverride, int budgetDiv) {

    if (vertLayerOverride >= 0) return vertLayerOverride;
    if (budgetDiv < 1) budgetDiv = 1;
    unsigned int budget = numVertices / (unsigned int)budgetDiv;
    if (budget < 500) budget = 500;
    if (budget > 3000) budget = 3000;
    if (dirtyCount * 30u <= budget) return 2;
    if (dirtyCount * 10u <= budget) return 1;
    return 0;
}

mathcore::Vec3 OptimizedSkinCluster::applyEnvelope(
    const mathcore::Vec3& orig, const mathcore::Vec3& deformed, double env) {
    if (env >= 1.0 - 1e-6) return deformed;
    if (env <= 1e-6) return orig;
    return { env * deformed.x + (1.0 - env) * orig.x,
             env * deformed.y + (1.0 - env) * orig.y,
             env * deformed.z + (1.0 - env) * orig.z };
}

double OptimizedSkinCluster::readEnvelopeFromNode() {
    if (m_envelopePlug.isNull()) {
        MPlug p(thisMObject(), MPxGeometryFilter::envelope);
        return p.isNull() ? 1.0 : p.asDouble();
    }
    return m_envelopePlug.asDouble();
}

// ================================================================
// readBoneTransforms
// ================================================================
bool OptimizedSkinCluster::readBoneTransforms(
    MDataBlock& block,
    std::vector<mathcore::RigidTransform>& out_rest,
    std::vector<mathcore::RigidTransform>& out_pose,
    std::vector<mathcore::Quat>& out_pose_qs) {

    const unsigned int MAX_BONES = 1024;
    MStatus status;
    MArrayDataHandle matH = block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return false;
    unsigned int nMat = matH.elementCount();
    if (nMat == 0) return false;
    if (nMat > MAX_BONES) nMat = MAX_BONES;

    MArrayDataHandle bpmH = block.inputArrayValue(MPxSkinCluster::bindPreMatrix, &status);
    if (!status) return false;
    unsigned int nBpm = bpmH.elementCount();
    if (nBpm > MAX_BONES) nBpm = MAX_BONES;

    std::vector<unsigned int> idxList;
    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        unsigned int idx = matH.elementIndex();
        if (idx == static_cast<unsigned int>(-1)) continue;
        if (idx >= MAX_BONES) continue;
        idxList.push_back(idx);
    }
    if (idxList.empty()) return false;

    unsigned int maxIdx = 0;
    for (unsigned int v : idxList) if (v > maxIdx) maxIdx = v;

    std::vector<mathcore::Quat> poseQ(maxIdx + 1);
    std::vector<mathcore::Vec3> poseT(maxIdx + 1);
    std::vector<mathcore::Quat> restQ(maxIdx + 1);
    std::vector<mathcore::Vec3> restT(maxIdx + 1);
    for (unsigned int i = 0; i <= maxIdx; ++i) {
        poseQ[i] = { 0, 0, 0, 1 };
        restQ[i] = { 0, 0, 0, 1 };
    }

    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        unsigned int idx = matH.elementIndex();
        if (idx > maxIdx) continue;
        MStatus s;
        MDataHandle h = matH.inputValue(&s);
        if (!s || h.data().isNull()) continue;
        if (!h.data().hasFn(MFn::kMatrixData)) continue;
        MMatrix m = MFnMatrixData(h.data()).matrix();
        MTransformationMatrix tm(m);
        MQuaternion q = tm.rotation();
        poseQ[idx] = { q.x, q.y, q.z, q.w };
        poseT[idx] = { m[3][0], m[3][1], m[3][2] };
    }

    for (unsigned int i = 0; i < nBpm; ++i) {
        if (!bpmH.jumpToArrayElement(i)) continue;
        unsigned int idx = bpmH.elementIndex();
        if (idx > maxIdx) continue;
        MStatus s;
        MDataHandle h = bpmH.inputValue(&s);
        if (!s || h.data().isNull()) continue;
        if (!h.data().hasFn(MFn::kMatrixData)) continue;
        MMatrix m = MFnMatrixData(h.data()).matrix();
        MTransformationMatrix tm(m);
        MQuaternion q = tm.rotation();
        restQ[idx] = { q.x, q.y, q.z, q.w };
        restT[idx] = { m[3][0], m[3][1], m[3][2] };
    }

    out_rest.resize(maxIdx + 1);
    out_pose.resize(maxIdx + 1);
    out_pose_qs.resize(maxIdx + 1);
    for (unsigned int i = 0; i <= maxIdx; ++i) {
        out_rest[i] = { restQ[i], restT[i] };
        mathcore::Quat c_q = mathcore::q_mul(poseQ[i], restQ[i]);
        mathcore::Vec3 rot = mathcore::q_rotate_v3(poseQ[i], restT[i]);
        mathcore::Vec3 c_t = { poseT[i].x + rot.x,
                               poseT[i].y + rot.y,
                               poseT[i].z + rot.z };
        out_pose[i] = { c_q, c_t };
        out_pose_qs[i] = poseQ[i];
    }
    return true;
}

// ============================================================
// 读取骨骼完整 3×3 线性部分（含缩放/shear）
// readBoneTransforms 返回的 RigidTransform 只保留旋转与平移，丢失缩放；
// Scale 主动控制需要真实缩放，故单独读取 matrix × bindPreMatrix 的 4×4 并提取列。
// ============================================================
bool OptimizedSkinCluster::readBoneLinearParts(
    MDataBlock& block, const MMatrix& /*worldMat*/,
    std::vector<mathcore::Vec3>& col0,
    std::vector<mathcore::Vec3>& col1,
    std::vector<mathcore::Vec3>& col2) {

    MStatus status;
    MArrayDataHandle matH = block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return false;
    const unsigned int nMat = matH.elementCount();
    if (nMat == 0) return false;

    std::vector<unsigned int> idxList;
    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        const unsigned int idx = matH.elementIndex();
        if (idx == static_cast<unsigned int>(-1)) continue;
        idxList.push_back(idx);
    }
    if (idxList.empty()) return false;

    unsigned int maxIdx = 0;
    for (unsigned int v : idxList) if (v > maxIdx) maxIdx = v;

    col0.assign(maxIdx + 1, { 1.0, 0.0, 0.0 });
    col1.assign(maxIdx + 1, { 0.0, 1.0, 0.0 });
    col2.assign(maxIdx + 1, { 0.0, 0.0, 1.0 });

    // bindPreMatrix 存的是 bind 逆矩阵，取逆后才是 bind 世界矩阵
    std::vector<MMatrix> restMats(maxIdx + 1);
    std::vector<bool> hasRest(maxIdx + 1, false);
    MArrayDataHandle bpmH = block.inputArrayValue(MPxSkinCluster::bindPreMatrix, &status);
    if (status) {
        const unsigned int nBpm = bpmH.elementCount();
        for (unsigned int i = 0; i < nBpm; ++i) {
            if (!bpmH.jumpToArrayElement(i)) continue;
            const unsigned int idx = bpmH.elementIndex();
            if (idx > maxIdx) continue;
            MStatus s;
            MDataHandle h = bpmH.inputValue(&s);
            if (!s || h.data().isNull()) continue;
            if (!h.data().hasFn(MFn::kMatrixData)) continue;
            restMats[idx] = MFnMatrixData(h.data()).matrix().inverse();
            hasRest[idx] = true;
        }
    }

    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        const unsigned int idx = matH.elementIndex();
        if (idx > maxIdx) continue;
        MStatus s;
        MDataHandle h = matH.inputValue(&s);
        if (!s || h.data().isNull()) continue;
        if (!h.data().hasFn(MFn::kMatrixData)) continue;

        const MMatrix pose = MFnMatrixData(h.data()).matrix();
        // Cs = pose × bindPreMatrix⁻¹（Maya 行向量约定下 p·Cs = p·pose·rest）
        const MMatrix cs = hasRest[idx] ? (pose * restMats[idx]) : pose;

        // 行主序 mat[i][j] → 列向量 col_j 的第 i 个分量
        col0[idx] = { cs[0][0], cs[1][0], cs[2][0] };
        col1[idx] = { cs[0][1], cs[1][1], cs[2][1] };
        col2[idx] = { cs[0][2], cs[1][2], cs[2][2] };
    }
    return true;
}

// ================================================================
// rebuildVtxCache
// ================================================================
void OptimizedSkinCluster::rebuildVtxCache(
    MDataBlock& block, unsigned int numVertices) {

    if (m_vtxCache.valid &&
        m_vtxCache.numVertices == numVertices &&
        m_vtxCache.weightVersion == m_currentWeightVersion) return;

    m_vtxCache.vertices.clear();
    m_vtxCache.vertices.resize(numVertices);

    MStatus status;
    MArrayDataHandle wlH = block.inputArrayValue(
        MPxSkinCluster::weightList, &status);
    if (!status) {
        m_vtxCache.valid = true;
        m_vtxCache.numVertices = numVertices;
        m_vtxCache.weightVersion = m_currentWeightVersion;
        return;
    }

    for (unsigned int i = 0; i < numVertices; ++i) {
        WeightMap w = readVertexWeightsFromHandle(wlH, i);
        std::vector<std::pair<unsigned int, float>> v;
        v.reserve(w.size());
        for (const auto& kv : w) {
            if (kv.second > 1e-9) v.push_back({ kv.first, (float)kv.second });
        }
        std::sort(v.begin(), v.end(),
            [](const std::pair<unsigned int, float>& a,
                const std::pair<unsigned int, float>& b) {
                    return a.second > b.second;
            });
        m_vtxCache.vertices[i] = std::move(v);
    }

    m_vtxCache.valid = true;
    m_vtxCache.numVertices = numVertices;
    m_vtxCache.weightVersion = m_currentWeightVersion;
}

// ================================================================
// collectBoneInfo
// ================================================================
void OptimizedSkinCluster::collectBoneInfo(MDataBlock& block) {
    MStatus status;
    MArrayDataHandle matH = block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return;
    unsigned int nMat = matH.elementCount();
    if (nMat == 0) return;

    unsigned long long sig = 1469598103934665603ULL;
    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        unsigned int idx = matH.elementIndex();
        if (idx == static_cast<unsigned int>(-1)) continue;
        sig ^= (unsigned long long)idx;
        sig *= 1099511628211ULL;
    }
    sig ^= (unsigned long long)nMat;
    sig *= 1099511628211ULL;

    unsigned int nBpm = 0;
    {
        MArrayDataHandle bpmH = block.inputArrayValue(
            MPxSkinCluster::bindPreMatrix, &status);
        if (status) {
            nBpm = bpmH.elementCount();
            for (unsigned int i = 0; i < nBpm; ++i) {
                if (!bpmH.jumpToArrayElement(i)) continue;
                unsigned int idx = bpmH.elementIndex();
                if (idx == static_cast<unsigned int>(-1)) continue;
                sig ^= (unsigned long long)idx;
                sig *= 1099511628211ULL;
                MStatus s;
                MDataHandle h = bpmH.inputValue(&s);
                if (s && !h.data().isNull() && h.data().hasFn(MFn::kMatrixData)) {
                    MMatrix m = MFnMatrixData(h.data()).matrix();
                    long long q0 = (long long)(m[3][0] * 1e4);
                    long long q1 = (long long)(m[3][1] * 1e4);
                    long long q2 = (long long)(m[3][2] * 1e4);
                    sig ^= (unsigned long long)q0; sig *= 1099511628211ULL;
                    sig ^= (unsigned long long)q1; sig *= 1099511628211ULL;
                    sig ^= (unsigned long long)q2; sig *= 1099511628211ULL;
                }
            }
        }
    }
    sig ^= (unsigned long long)nBpm;
    sig *= 1099511628211ULL;

    if (m_boneInfo.valid && m_boneInfo.topologySignature == sig) return;

    MPlug matPlug(thisMObject(), MPxSkinCluster::matrix);
    std::vector<MObject> joints;
    std::vector<unsigned int> logicalIdx;

    matH.jumpToArrayElement(0);
    for (unsigned int i = 0; i < nMat; ++i) {
        unsigned int idx = matH.elementIndex();
        MPlug elem = matPlug.elementByLogicalIndex(idx);
        MPlugArray conns;
        elem.connectedTo(conns, true, false);
        if (conns.length() > 0) {
            MObject jnt = conns[0].node();
            if (jnt.hasFn(MFn::kJoint)) {
                joints.push_back(jnt);
                logicalIdx.push_back(idx);
            }
        }
        matH.next();
    }

    unsigned int nB = (unsigned int)joints.size();
    if (nB == 0) { m_boneInfo.valid = false; return; }

    unsigned int maxIdx = 0;
    for (unsigned int i = 0; i < nB; ++i)
        if (logicalIdx[i] > maxIdx) maxIdx = logicalIdx[i];

    m_boneInfo.depths.assign(maxIdx + 1, 0);
    m_boneInfo.boneDirRest.assign(maxIdx + 1, { 0.0, 1.0, 0.0 });
    m_boneInfo.numBones = maxIdx + 1;

    {
        std::vector<mathcore::Vec3> jp(maxIdx + 1, { 0.0, 0.0, 0.0 });
        MArrayDataHandle bpmH2 = block.inputArrayValue(
            MPxSkinCluster::bindPreMatrix, &status);
        if (status) {
            unsigned int nBpm2 = bpmH2.elementCount();
            for (unsigned int i = 0; i < nBpm2; ++i) {
                if (!bpmH2.jumpToArrayElement(i)) continue;
                unsigned int idx = bpmH2.elementIndex();
                if (idx > maxIdx) continue;
                MStatus s;
                MDataHandle h = bpmH2.inputValue(&s);
                if (!s || h.data().isNull()) continue;
                if (!h.data().hasFn(MFn::kMatrixData)) continue;
                MMatrix m_inv = MFnMatrixData(h.data()).matrix();
                MMatrix m = m_inv.inverse();
                jp[idx] = { m[3][0], m[3][1], m[3][2] };
            }
        }
        m_boneInfo.jointPosRest = std::move(jp);
    }

    for (unsigned int i = 0; i < nB; ++i) {
        unsigned int d = 0;
        MObject curJnt = joints[i];
        while (d < 512) {
            MFnDagNode dag(curJnt);
            MObject parent = dag.parent(0);
            if (parent.isNull() || !parent.hasFn(MFn::kJoint)) break;
            bool found = false;
            for (unsigned int j = 0; j < nB; ++j) {
                if (joints[j] == parent) { curJnt = parent; found = true; break; }
            }
            if (!found) break;
            ++d;
        }
        m_boneInfo.depths[logicalIdx[i]] = (int)d;
    }

    for (unsigned int i = 0; i < nB; ++i) {
        unsigned int idx = logicalIdx[i];
        MFnDagNode dag(joints[i]);
        mathcore::Vec3 bonePos = m_boneInfo.jointPosRest[idx];
        unsigned int nChild = dag.childCount();
        bool found = false;
        for (unsigned int c = 0; c < nChild; ++c) {
            MObject child = dag.child(c);
            if (!child.hasFn(MFn::kJoint)) continue;
            int childListIdx = -1;
            for (unsigned int k = 0; k < nB; ++k) {
                if (joints[k] == child) { childListIdx = (int)k; break; }
            }
            if (childListIdx < 0) continue;
            unsigned int childIdx = logicalIdx[childListIdx];
            mathcore::Vec3 childPos = m_boneInfo.jointPosRest[childIdx];
            mathcore::Vec3 dir = { childPos.x - bonePos.x,
                                   childPos.y - bonePos.y,
                                   childPos.z - bonePos.z };
            double len = mathcore::v3_len(dir);
            if (len > 1e-9) {
                m_boneInfo.boneDirRest[idx] = mathcore::v3_scale(dir, 1.0 / len);
                found = true; break;
            }
        }
        if (!found) {
            MObject parentObj = dag.parent(0);
            if (!parentObj.isNull() && parentObj.hasFn(MFn::kJoint)) {
                int parentListIdx = -1;
                for (unsigned int k = 0; k < nB; ++k) {
                    if (joints[k] == parentObj) { parentListIdx = (int)k; break; }
                }
                if (parentListIdx >= 0) {
                    unsigned int parentIdx = logicalIdx[parentListIdx];
                    mathcore::Vec3 parentPos = m_boneInfo.jointPosRest[parentIdx];
                    mathcore::Vec3 dir = { bonePos.x - parentPos.x,
                                           bonePos.y - parentPos.y,
                                           bonePos.z - parentPos.z };
                    double len = mathcore::v3_len(dir);
                    if (len > 1e-9) {
                        m_boneInfo.boneDirRest[idx] = mathcore::v3_scale(dir, 1.0 / len);
                        found = true;
                    }
                }
            }
        }
        if (!found) m_boneInfo.boneDirRest[idx] = { 0.0, 1.0, 0.0 };
    }

    m_boneInfo.topologySignature = sig;
    m_boneInfo.valid = true;

    if (optSkinDebugLog()) {
        char buf[256];
        snprintf(buf, sizeof(buf),
            "[optimizedSkinCluster] BoneInfo rebuilt: %u joints, sig=%llu",
            nB, (unsigned long long)sig);
        MGlobal::displayInfo(MString(buf));
    }
}

// ================================================================
// computeCoRCache
// ================================================================
bool OptimizedSkinCluster::computeCoRCache(
    MDataBlock& block, unsigned int numVertices) {

    auto tStart = std::chrono::high_resolution_clock::now();

    m_corCache.allW.resize(numVertices);
    {
        MStatus wStatus;
        MArrayDataHandle wlH = block.inputArrayValue(
            MPxSkinCluster::weightList, &wStatus);
        if (!wStatus) return false;
        for (unsigned int i = 0; i < numVertices; ++i) {
            WeightMap w = readVertexWeightsFromHandle(wlH, i);
            m_corCache.allW[i].buildFrom(w);
        }
    }

    MStatus status;
    MArrayDataHandle inArr = block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return false;
    inArr.jumpToArrayElement(0);
    MDataHandle inData = inArr.inputValue(&status);
    if (!status) return false;
    MObject meshObj = inData.child(MPxSkinCluster::inputGeom).asMesh();
    if (meshObj.isNull()) return false;

    MFnMesh meshFn(meshObj);
    MPointArray points;
    meshFn.getPoints(points, MSpace::kObject);
    if (points.length() < numVertices) return false;

    MIntArray triCounts, triVerts;
    meshFn.getTriangles(triCounts, triVerts);
    unsigned int numTris = triVerts.length() / 3;
    if (numTris == 0) return false;

    m_corCache.numTris = numTris;
    m_corCache.triVertCache.resize(numTris);
    for (unsigned int t = 0; t < numTris; ++t) {
        m_corCache.triVertCache[t] = {
            (unsigned int)triVerts[t * 3 + 0],
            (unsigned int)triVerts[t * 3 + 1],
            (unsigned int)triVerts[t * 3 + 2]
        };
    }

    m_corCache.tris.resize(numTris);
    for (unsigned int t = 0; t < numTris; ++t) {
        m_corCache.tris[t] = buildSingleTriInfo(
            points, m_corCache.triVertCache[t].data(), m_corCache.allW);
    }

    m_corCache.triSubdivs.assign(numTris, {});
    for (unsigned int t = 0; t < numTris; ++t) {
        m_corCache.triSubdivs[t] = subdivideSingleTriangle(
            points, m_corCache.triVertCache[t].data(), m_corCache.allW);
    }

    m_corCache.triAdj.assign(numTris, {});
    {
        std::unordered_map<unsigned long long, unsigned int> edge_to_tri;
        edge_to_tri.reserve(numTris * 3);
        auto edgeKey = [](int a, int b) -> unsigned long long {
            if (a > b) { int tmp = a; a = b; b = tmp; }
            return ((unsigned long long)(unsigned int)a << 32) | (unsigned int)b;
            };
        for (unsigned int t = 0; t < numTris; ++t) {
            const auto& tv = m_corCache.triVertCache[t];
            for (int k = 0; k < 3; ++k) {
                unsigned long long key = edgeKey(tv[k], tv[(k + 1) % 3]);
                auto it = edge_to_tri.find(key);
                if (it == edge_to_tri.end()) edge_to_tri[key] = t;
                else {
                    unsigned int other = it->second;
                    m_corCache.triAdj[t].push_back(other);
                    m_corCache.triAdj[other].push_back(t);
                }
            }
        }
    }

    m_corCache.vertToTris.assign(numVertices, {});
    for (unsigned int t = 0; t < numTris; ++t) {
        for (unsigned int v : m_corCache.triVertCache[t]) {
            if (v < numVertices) m_corCache.vertToTris[v].push_back(t);
        }
    }

    unsigned int K = (unsigned int)std::sqrt((double)numVertices);
    if (K < 8) K = 8;
    if (K > 512) K = 512;
    if (K > numVertices) K = numVertices;
    m_corCache.K = K;

    m_corCache.centers = optimized_skin_internal::farthestPointClustering(
        m_corCache.allW, K);

    m_corCache.clusterMembers.assign(K, {});
    for (unsigned int i = 0; i < numVertices; ++i) {
        unsigned int bestK = 0;
        double bestD = weightL2DistSq(m_corCache.allW[i],
            m_corCache.allW[m_corCache.centers[0]]);
        for (unsigned int k = 1; k < K; ++k) {
            double d = weightL2DistSq(m_corCache.allW[i],
                m_corCache.allW[m_corCache.centers[k]]);
            if (d < bestD) { bestD = d; bestK = k; }
        }
        m_corCache.clusterMembers[bestK].push_back(i);
    }

    m_corCache.vertClusterDists.assign(numVertices, {});
    const unsigned int N_NEAREST = 3;
    for (unsigned int i = 0; i < numVertices; ++i) {
        std::vector<std::pair<double, unsigned int>> cd(K);
        for (unsigned int k = 0; k < K; ++k) {
            cd[k] = { weightL2DistSq(m_corCache.allW[i],
                     m_corCache.allW[m_corCache.centers[k]]), k };
        }
        std::partial_sort(cd.begin(),
            cd.begin() + std::min<size_t>(N_NEAREST, K), cd.end());
        m_corCache.vertClusterDists[i] = std::move(cd);
    }

    m_corCache.cors.resize(numVertices);
#pragma omp parallel for schedule(dynamic, 64)
    for (int ii = 0; ii < (int)numVertices; ++ii) {
        unsigned int i = (unsigned int)ii;
        m_corCache.cors[i] = computeSingleVertexCoR(
            m_corCache.allW[i], points, i,
            m_corCache.triSubdivs, m_corCache.triAdj, m_corCache.vertToTris,
            K, m_corCache.clusterMembers, m_corCache.vertClusterDists,
            m_corCache.allW, numTris);
    }

    for (unsigned int i = 0; i < numVertices; ++i) {
        mathcore::Vec3 p = { points[i].x, points[i].y, points[i].z };
        mathcore::Vec3 c = m_corCache.cors[i];
        if (std::isnan(c.x) || std::isnan(c.y) || std::isnan(c.z) ||
            std::isinf(c.x) || std::isinf(c.y) || std::isinf(c.z)) {
            m_corCache.cors[i] = p; continue;
        }
        mathcore::Vec3 d = mathcore::v3_sub(c, p);
        if (mathcore::v3_len(d) > 100.0) m_corCache.cors[i] = p;
    }

    m_corCache.numVertices = numVertices;
    m_corCache.valid = true;
    clearDirtyFlags();

    auto tEnd = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    if (optSkinDebugLog()) {
        char buf[320];
        snprintf(buf, sizeof(buf),
            "[optimizedSkinCluster] CoR full rebuild: %u verts, %u tris, K=%u  [%.1fms]",
            numVertices, numTris, K, ms);
        MGlobal::displayInfo(MString(buf));
    }
    return true;
}

// ================================================================
// updateCoRCacheIncremental
// ================================================================
void OptimizedSkinCluster::updateCoRCacheIncremental(
    MDataBlock& block, const std::vector<unsigned int>& dirtyList,
    int triLayers, int vertLayers) {

    if (!m_corCache.valid || m_corCache.numVertices == 0) return;
    if (dirtyList.empty()) return;

    unsigned int numVertices = m_corCache.numVertices;
    unsigned int numTris = m_corCache.numTris;

    MStatus status;
    MArrayDataHandle inArr = block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return;
    inArr.jumpToArrayElement(0);
    MDataHandle inData = inArr.inputValue(&status);
    if (!status) return;
    MObject meshObj = inData.child(MPxSkinCluster::inputGeom).asMesh();
    if (meshObj.isNull()) return;

    MFnMesh meshFn(meshObj);
    MPointArray points;
    meshFn.getPoints(points, MSpace::kObject);
    if (points.length() < numVertices) return;

    std::vector<unsigned int> validDirty;
    validDirty.reserve(dirtyList.size());
    {
        MStatus wStatus;
        MArrayDataHandle wlH = block.inputArrayValue(
            MPxSkinCluster::weightList, &wStatus);
        if (!wStatus) return;
        for (unsigned int v : dirtyList) {
            if (v >= numVertices) continue;
            WeightMap w = readVertexWeightsFromHandle(wlH, v);
            SparseW sw; sw.buildFrom(w);
            if (!sparseEqual(sw, m_corCache.allW[v])) {
                m_corCache.allW[v] = std::move(sw);
                validDirty.push_back(v);
            }
        }
    }
    if (validDirty.empty()) return;

    for (unsigned int v : validDirty) {
        unsigned int K = m_corCache.K;
        std::vector<std::pair<double, unsigned int>> cd(K);
        for (unsigned int k = 0; k < K; ++k) {
            cd[k] = { weightL2DistSq(m_corCache.allW[v],
                     m_corCache.allW[m_corCache.centers[k]]), k };
        }
        std::partial_sort(cd.begin(),
            cd.begin() + std::min<size_t>(3, K), cd.end());
        m_corCache.vertClusterDists[v] = std::move(cd);
    }

    std::vector<unsigned char> triAffected(numTris, 0);
    std::vector<unsigned int> tAffected;
    for (unsigned int v : validDirty) {
        for (unsigned int t : m_corCache.vertToTris[v]) {
            if (!triAffected[t]) { triAffected[t] = 1; tAffected.push_back(t); }
        }
    }
    for (int layer = 0; layer < triLayers; ++layer) {
        std::vector<unsigned int> newTris;
        for (unsigned int t : tAffected) {
            for (unsigned int nb : m_corCache.triAdj[t]) {
                if (!triAffected[nb]) { triAffected[nb] = 1; newTris.push_back(nb); }
            }
        }
        tAffected.insert(tAffected.end(), newTris.begin(), newTris.end());
    }

    for (unsigned int t : tAffected) {
        m_corCache.tris[t] = buildSingleTriInfo(
            points, m_corCache.triVertCache[t].data(), m_corCache.allW);
        m_corCache.triSubdivs[t] = subdivideSingleTriangle(
            points, m_corCache.triVertCache[t].data(), m_corCache.allW);
    }

    std::vector<unsigned char> vertAffected(numVertices, 0);
    std::vector<unsigned int> vAffected;
    auto addV = [&](unsigned int v) {
        if (v < numVertices && !vertAffected[v]) {
            vertAffected[v] = 1; vAffected.push_back(v);
        }
        };
    for (unsigned int v : validDirty) addV(v);
    for (unsigned int t : tAffected)
        for (unsigned int v : m_corCache.triVertCache[t]) addV(v);
    for (int layer = 0; layer < vertLayers; ++layer) {
        std::vector<unsigned int> newVerts;
        size_t baseSize = vAffected.size();
        for (size_t k = 0; k < baseSize; ++k) {
            unsigned int v = vAffected[k];
            for (unsigned int t : m_corCache.vertToTris[v]) {
                for (unsigned int v2 : m_corCache.triVertCache[t]) {
                    if (!vertAffected[v2] && v2 < numVertices) {
                        vertAffected[v2] = 1; newVerts.push_back(v2);
                    }
                }
            }
        }
        vAffected.insert(vAffected.end(), newVerts.begin(), newVerts.end());
    }

#pragma omp parallel for schedule(dynamic, 8)
    for (int ii = 0; ii < (int)vAffected.size(); ++ii) {
        unsigned int v = vAffected[ii];
        m_corCache.cors[v] = computeSingleVertexCoR(
            m_corCache.allW[v], points, v,
            m_corCache.triSubdivs, m_corCache.triAdj, m_corCache.vertToTris,
            m_corCache.K, m_corCache.clusterMembers, m_corCache.vertClusterDists,
            m_corCache.allW, numTris);
    }
}

// ================================================================
// deform 主入口
// ================================================================
MStatus OptimizedSkinCluster::deform(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex) {

    MStatus status;
    int algo = block.inputValue(aAlgorithmType, &status).asShort();
    if (!status) return status;

    unsigned int nV = iter.count();
    collectBoneInfo(block);
    rebuildVtxCache(block, nV);

    switch (algo) {
    case 0: return deformCoR(block, iter, worldMat, multiIndex);
    case 1: return deformDLB(block, iter, worldMat, multiIndex);
    case 2: return deformDQSComp(block, iter, worldMat, multiIndex);
    case 3: return deformMOS(block, iter, worldMat, multiIndex);
    default: return MS::kFailure;
    }
}

// ================================================================
// deformCoR
// ================================================================
MStatus OptimizedSkinCluster::deformCoR(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    unsigned int nV = iter.count();

    MStatus s2;
    bool forceFull = block.inputValue(aForceFullCoR, &s2).asBool();
    if (s2 && forceFull) m_corCache.valid = false;

    MStatus s3;
    int triLayers = block.inputValue(aCorTriLayerExpand, &s3).asInt();
    if (!s3) triLayers = 1;
    int vertLayerOverride = block.inputValue(aCorVertLayerExpand, &s3).asInt();
    if (!s3) vertLayerOverride = -1;
    int budgetDiv = block.inputValue(aCorIncBudgetDiv, &s3).asInt();
    if (!s3) budgetDiv = 10;

    if (!m_corCache.valid || m_corCache.numVertices != nV) {
        if (!computeCoRCache(block, nV))
            return deformDLB(block, iter, worldMat, 0);
    }
    else if (m_hasPlugDirty || m_needFullWeightScan) {
        std::vector<unsigned int> dirtyList;
        if (m_hasPlugDirty) {
            for (unsigned int i = 0;
                i < (unsigned int)m_dirtyVertFlags.size() && i < nV; ++i)
                if (m_dirtyVertFlags[i]) dirtyList.push_back(i);
        }
        if (m_needFullWeightScan) {
            MStatus wStatus;
            MArrayDataHandle wlH = block.inputArrayValue(
                MPxSkinCluster::weightList, &wStatus);
            if (wStatus) {
                for (unsigned int i = 0; i < nV; ++i) {
                    WeightMap curW = readVertexWeightsFromHandle(wlH, i);
                    SparseW sw; sw.buildFrom(curW);
                    if (!sparseEqual(sw, m_corCache.allW[i])) dirtyList.push_back(i);
                }
            }
        }
        if (!dirtyList.empty()) {
            std::sort(dirtyList.begin(), dirtyList.end());
            dirtyList.erase(std::unique(dirtyList.begin(), dirtyList.end()),
                dirtyList.end());
            if (dirtyList.size() > (size_t)nV / 4) {
                computeCoRCache(block, nV);
            }
            else {
                int vertLayers = computeVertLayers(
                    (unsigned int)dirtyList.size(), nV,
                    vertLayerOverride, budgetDiv);
                updateCoRCacheIncremental(block, dirtyList, triLayers, vertLayers);
            }
        }
        clearDirtyFlags();
    }

    MPointArray points;
    iter.allPositions(points);
    MMatrix wInv = worldMat.inverse();
    const double envFinal = env;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)nV; ++i) {
        const auto& vw = m_vtxCache.vertices[i];
        if (vw.empty()) continue;
        MPoint ptW = points[i] * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };

        mathcore::Vec3 cor = origPos;
        if ((unsigned)i < m_corCache.cors.size()) {
            mathcore::Vec3 cl = m_corCache.cors[i];
            MPoint cw(cl.x, cl.y, cl.z); cw *= worldMat;
            cor = { cw.x, cw.y, cw.z };
        }

        mathcore::Vec3 def = cor_vertex_v2(origPos, cor, vw, Cs);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, envFinal);
        MPoint out(bl.x, bl.y, bl.z);
        out *= wInv;
        points[i] = out;
    }
    iter.setAllPositions(points);
    return MS::kSuccess;
}

// ================================================================
// deformDLB
// ================================================================
MStatus OptimizedSkinCluster::deformDLB(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    unsigned int nV = iter.count();
    MPointArray points;
    iter.allPositions(points);
    MMatrix wInv = worldMat.inverse();
    const double envFinal = env;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)nV; ++i) {
        const auto& vw = m_vtxCache.vertices[i];
        if (vw.empty()) continue;
        MPoint ptW = points[i] * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };
        mathcore::Vec3 def = dlb_vertex_v2(origPos, vw, Cs);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, envFinal);
        MPoint out(bl.x, bl.y, bl.z);
        out *= wInv;
        points[i] = out;
    }
    iter.setAllPositions(points);
    return MS::kSuccess;
}

// ================================================================
// deformDQSComp
// ================================================================
MStatus OptimizedSkinCluster::deformDQSComp(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    double strength = block.inputValue(aDQSCompStrength).asDouble();
    MMatrix wInv = worldMat.inverse();

    unsigned int n = (unsigned int)rest.size();
    std::vector<mathcore::Vec3> bd(n, { 0.0, 1.0, 0.0 });
    for (unsigned int i = 0; i < n; ++i) {
        mathcore::Vec3 dirRest = { 0.0, 1.0, 0.0 };
        if (i < m_boneInfo.boneDirRest.size()) dirRest = m_boneInfo.boneDirRest[i];
        bd[i] = mathcore::q_rotate_v3(Cs[i].q, dirRest);
    }

    const std::vector<mathcore::Vec3>& jpCached = m_boneInfo.jointPosRest;
    std::vector<mathcore::Vec3> jpFallback;
    const std::vector<mathcore::Vec3>* jpPtr = &jpCached;
    if (jpCached.size() < n) {
        jpFallback.assign(n, { 0.0, 0.0, 0.0 });
        for (unsigned int i = 0; i < n; ++i) {
            mathcore::Quat rq_inv = mathcore::q_inverse(rest[i].q);
            mathcore::Vec3 pos = mathcore::q_rotate_v3(rq_inv, rest[i].t);
            jpFallback[i] = { -pos.x, -pos.y, -pos.z };
        }
        jpPtr = &jpFallback;
    }
    const std::vector<mathcore::Vec3>& jp = *jpPtr;
    const std::vector<int>& depths = m_boneInfo.depths;

    unsigned int nV = iter.count();
    MPointArray points;
    iter.allPositions(points);
    const double envFinal = env;
    const double strengthFinal = strength;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)nV; ++i) {
        const auto& vw = m_vtxCache.vertices[i];
        if (vw.empty()) continue;
        MPoint ptW = points[i] * worldMat;
        mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };
        mathcore::Vec3 dqs = dlb_vertex_v2(origPos, vw, Cs);
        mathcore::Vec3 def = dqs_compensate_vertex_v2(
            origPos, dqs, vw, Cs, bd, jp, depths, strengthFinal);
        mathcore::Vec3 bl = applyEnvelope(origPos, def, envFinal);
        MPoint out(bl.x, bl.y, bl.z);
        out *= wInv;
        points[i] = out;
    }
    iter.setAllPositions(points);
    return MS::kSuccess;
}

OptimizedSkinCluster::MosParams OptimizedSkinCluster::readMosParams(MDataBlock& block) {
    MosParams p;
    MStatus st;

    if (!aMosInfluenceRadius.isNull()) {
        MDataHandle h = block.inputValue(aMosInfluenceRadius, &st);
        if (st) p.influenceRadius = h.asDouble();
    }
    if (!aMosCurveMode.isNull()) {
        MDataHandle h = block.inputValue(aMosCurveMode, &st);
        if (st) p.curveMode = h.asShort();
    }
    if (!aMosEndLock.isNull()) {
        MDataHandle h = block.inputValue(aMosEndLock, &st);
        if (st) p.endLock = h.asBool();
    }
    if (!aMosCurveSamples.isNull()) {
        MDataHandle h = block.inputValue(aMosCurveSamples, &st);
        if (st) p.curveSamples = h.asInt();
    }
    if (!aMosDebugLog.isNull()) {
        MDataHandle h = block.inputValue(aMosDebugLog, &st);
        if (st) p.debugLog = h.asBool();
    }
    if (!aMosInBulgeWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosInBulgeWeight, &st);
        if (st) p.inBulgeWeight = h.asDouble();
    }
    if (!aMosOutBulgeWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosOutBulgeWeight, &st);
        if (st) p.outBulgeWeight = h.asDouble();
    }
    if (!aMosSideBulgeWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosSideBulgeWeight, &st);
        if (st) p.sideBulgeWeight = h.asDouble();
    }
    // ---- 方案B: 8个slide变体 ----
    if (!aMosSlideInWeightPrev.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideInWeightPrev, &st);
        if (st) p.slideInWeight_prev = h.asDouble();
    }
    if (!aMosSlideInWeightNext.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideInWeightNext, &st);
        if (st) p.slideInWeight_next = h.asDouble();
    }
    if (!aMosSlideOutWeightPrev.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOutWeightPrev, &st);
        if (st) p.slideOutWeight_prev = h.asDouble();
    }
    if (!aMosSlideOutWeightNext.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOutWeightNext, &st);
        if (st) p.slideOutWeight_next = h.asDouble();
    }
    if (!aMosSlideOrthoInWeightPrev.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOrthoInWeightPrev, &st);
        if (st) p.slideOrthoInWeight_prev = h.asDouble();
    }
    if (!aMosSlideOrthoInWeightNext.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOrthoInWeightNext, &st);
        if (st) p.slideOrthoInWeight_next = h.asDouble();
    }
    if (!aMosSlideOrthoOutWeightPrev.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOrthoOutWeightPrev, &st);
        if (st) p.slideOrthoOutWeight_prev = h.asDouble();
    }
    if (!aMosSlideOrthoOutWeightNext.isNull()) {
        MDataHandle h = block.inputValue(aMosSlideOrthoOutWeightNext, &st);
        if (st) p.slideOrthoOutWeight_next = h.asDouble();
    }
    // ---- 手动 in vector ----
    if (!aMosUseManualInVector.isNull()) {
        MDataHandle h = block.inputValue(aMosUseManualInVector, &st);
        if (st) p.useManualInVector = h.asBool();
    }
    if (!aMosManualInVector.isNull()) {
        MDataHandle h = block.inputValue(aMosManualInVector, &st);
        if (st) {
            // asDouble3() 返回 double*（3 元素内部数组），不是 double3 结构
            double* v = h.asDouble3();
            if (v) {
                p.manualInVector = { v[0], v[1], v[2] };
            }
        }
    }
    if (!aMosTwistWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosTwistWeight, &st);
        if (st) p.twistWeight = h.asDouble();
    }
    if (!aMosScaleWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosScaleWeight, &st);
        if (st) p.scaleWeight = h.asDouble();
    }
    if (!aMosScaleMode.isNull()) {
        MDataHandle h = block.inputValue(aMosScaleMode, &st);
        if (st) p.scaleMode = h.asShort();
    }
    if (!aMosCenterSlideInWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosCenterSlideInWeight, &st);
        if (st) p.centerSlideInWeight = h.asDouble();
    }
    if (!aMosCenterSlideOutWeight.isNull()) {
        MDataHandle h = block.inputValue(aMosCenterSlideOutWeight, &st);
        if (st) p.centerSlideOutWeight = h.asDouble();
    }
    return p;
}

bool OptimizedSkinCluster::MosParams::nearlyEqual(const MosParams& rhs) const {
    auto deq = [](double a, double b) { return std::abs(a - b) <= 1e-9; };
    return deq(influenceRadius, rhs.influenceRadius) &&
        curveMode == rhs.curveMode &&
        endLock == rhs.endLock &&
        curveSamples == rhs.curveSamples &&
        deq(inBulgeWeight, rhs.inBulgeWeight) &&
        deq(outBulgeWeight, rhs.outBulgeWeight) &&
        deq(sideBulgeWeight, rhs.sideBulgeWeight) &&
        deq(slideInWeight_prev, rhs.slideInWeight_prev) &&
        deq(slideInWeight_next, rhs.slideInWeight_next) &&
        deq(slideOutWeight_prev, rhs.slideOutWeight_prev) &&
        deq(slideOutWeight_next, rhs.slideOutWeight_next) &&
        deq(slideOrthoInWeight_prev, rhs.slideOrthoInWeight_prev) &&
        deq(slideOrthoInWeight_next, rhs.slideOrthoInWeight_next) &&
        deq(slideOrthoOutWeight_prev, rhs.slideOrthoOutWeight_prev) &&
        deq(slideOrthoOutWeight_next, rhs.slideOrthoOutWeight_next) &&
        useManualInVector == rhs.useManualInVector &&
        deq(manualInVector.x, rhs.manualInVector.x) &&
        deq(manualInVector.y, rhs.manualInVector.y) &&
        deq(manualInVector.z, rhs.manualInVector.z) &&
        deq(twistWeight, rhs.twistWeight) &&
        deq(scaleWeight, rhs.scaleWeight) &&
        scaleMode == rhs.scaleMode &&
        deq(centerSlideInWeight, rhs.centerSlideInWeight) &&
        deq(centerSlideOutWeight, rhs.centerSlideOutWeight) &&
        debugLog == rhs.debugLog;
}

bool OptimizedSkinCluster::computeMosCache(
    MDataBlock& block, unsigned int numVertices,
    const MosParams& params) {

    auto tStart = std::chrono::high_resolution_clock::now();

    m_mosCache.params = params;

    if (!m_boneInfo.valid) collectBoneInfo(block);
    if (!m_boneInfo.valid || m_boneInfo.jointPosRest.empty()) {
        if (optSkinDebugLog())
            MGlobal::displayWarning("[MOS] bone info 未建立");
        return false;
    }

    MStatus status;
    MArrayDataHandle inArr = block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return false;
    inArr.jumpToArrayElement(0);
    MDataHandle inData = inArr.inputValue(&status);
    if (!status) return false;
    MObject meshObj = inData.child(MPxSkinCluster::inputGeom).asMesh();
    if (meshObj.isNull()) return false;

    MFnMesh meshFn(meshObj);
    MPointArray points;
    meshFn.getPoints(points, MSpace::kWorld);
    if (points.length() < (int)numVertices) return false;

    // ---- 收集 skinCluster matrix[] 连接的 joint ----
    MPlug matPlug(thisMObject(), MPxSkinCluster::matrix);
    std::vector<MObject> joints;
    std::vector<unsigned int> logicalIdx;

    MArrayDataHandle matH = block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return false;
    unsigned int nMat = matH.elementCount();
    for (unsigned int i = 0; i < nMat; ++i) {
        if (!matH.jumpToArrayElement(i)) continue;
        unsigned int idx = matH.elementIndex();
        MPlug elem = matPlug.elementByLogicalIndex(idx);
        MPlugArray conns;
        elem.connectedTo(conns, true, false);
        if (conns.length() > 0) {
            MObject jnt = conns[0].node();
            if (jnt.hasFn(MFn::kJoint)) {
                joints.push_back(jnt);
                logicalIdx.push_back(idx);
            }
        }
    }
    if (joints.empty()) return false;

    std::unordered_map<std::string, size_t> name_to_idx;
    for (size_t i = 0; i < joints.size(); ++i) {
        MFnDagNode dag(joints[i]);
        name_to_idx[std::string(dag.fullPathName().asChar())] = i;
    }

    // ---- 建立 parent → child 映射 ----
    std::vector<int> child_of(joints.size(), -1);
    for (size_t i = 0; i < joints.size(); ++i) {
        std::vector<MObject> frontier;
        MFnDagNode dag(joints[i]);
        unsigned int nChild = dag.childCount();
        for (unsigned int c = 0; c < nChild; ++c) {
            MObject ch = dag.child(c);
            if (ch.hasFn(MFn::kJoint)) frontier.push_back(ch);
        }
        int found = -1, visited = 0;
        while (!frontier.empty() && visited < 20 && found < 0) {
            MObject cur = frontier.front();
            frontier.erase(frontier.begin());
            visited++;
            MFnDagNode cdag(cur);
            auto it = name_to_idx.find(std::string(cdag.fullPathName().asChar()));
            if (it != name_to_idx.end()) { found = (int)it->second; break; }
            unsigned int nc = cdag.childCount();
            for (unsigned int cc = 0; cc < nc; ++cc) {
                MObject c2 = cdag.child(cc);
                if (c2.hasFn(MFn::kJoint)) frontier.push_back(c2);
            }
        }
        child_of[i] = found;
    }

    // ---- 找 root，构建 chain ----
    std::vector<bool> is_child(joints.size(), false);
    for (size_t i = 0; i < joints.size(); ++i)
        if (child_of[i] >= 0) is_child[child_of[i]] = true;
    int root_i = -1;
    for (size_t i = 0; i < joints.size(); ++i)
        if (!is_child[i]) { root_i = (int)i; break; }
    if (root_i < 0) return false;

    std::vector<unsigned int> chain;
    int cur = root_i;
    std::vector<bool> seen(joints.size(), false);
    while (cur >= 0 && !seen[cur]) {
        seen[cur] = true;
        chain.push_back(logicalIdx[cur]);
        cur = child_of[cur];
    }
    if (chain.size() < 2) return false;

    const auto& bind_jp = m_boneInfo.jointPosRest;
    for (unsigned int b : chain) {
        if (b >= bind_jp.size()) {
            if (optSkinDebugLog())
                MGlobal::displayWarning("[MOS] chain 逻辑索引越界，回退 DLB");
            return false;
        }
    }

    const int n_bones = (int)chain.size();
    std::vector<mathcore::Vec3> bone_dir(n_bones);
    std::vector<double> bone_len(n_bones);
    for (int i = 0; i + 1 < n_bones; ++i) {
        unsigned int b0 = chain[i], b1 = chain[i + 1];
        mathcore::Vec3 d = mathcore::v3_sub(bind_jp[b1], bind_jp[b0]);
        bone_len[i] = mathcore::v3_len(d);
        bone_dir[i] = (bone_len[i] > 1e-9)
            ? mathcore::v3_scale(d, 1.0 / bone_len[i])
            : mathcore::Vec3{ 0, 1, 0 };
    }
    if (n_bones >= 2) {
        bone_dir[n_bones - 1] = bone_dir[n_bones - 2];
        bone_len[n_bones - 1] = bone_len[n_bones - 2];
    }

    m_mosCache.curves.clear();

    // 追加一个控制点：同时记录位置、绑定骨骼与沿链归一化参数 u。
    // u 是后续 F_j 软权重混合与内节点布置的唯一依据，必须在构建时落定。
    auto push_cv = [](MosCurve& c, const mathcore::Vec3& p,
        unsigned int bone, double u) {
            c.cvs_rest.push_back(p);
            c.cvs_bone.push_back(bone);
            c.cvs_u.push_back(u);
        };

    auto finalize_curve = [&](MosCurve& c) -> bool {
        const int n_cv_local = (int)c.cvs_rest.size();
        if (n_cv_local < 1) return false;
        // cvs_u 必须与 cvs_rest 等长且单调非减：后续要当作 knot 内节点使用，
        // 长度或顺序不对会直接破坏 De Boor 的区间查找。
        if ((int)c.cvs_u.size() != n_cv_local) return false;
        for (int i = 1; i < n_cv_local; ++i)
            if (c.cvs_u[i] < c.cvs_u[i - 1] - 1e-12) return false;
        c.chain_bones = chain;
        c.degree = ocd_math::effective_degree(n_cv_local, 3);
        if (c.degree < 1) return false;

        // ★ 内节点取对称位置：j = i + (n_cv - n_inner)/2。
        //   令 inner[k] 与 inner[n_inner-1-k] 关于 0.5 镜像，
        //   从而保证 B-spline 基函数关于 τ=0.5 对称。
        //
        //   若用 j = i + degree（原代码），内部节点整体右移：
        //     7点 per-joint → inner = [0.5, 0.6, 0.8]（不对称）
        //     8点 single     → inner = [0.444, 0.556, 0.694, 0.861]（不对称）
        //   B-spline 不对称 → curve_mos 在 τ=0.5 两侧形状不同 → 变形左右略偏。
        //   改为对称取法后：
        //     7点 per-joint → inner = [0.4, 0.5, 0.6]（对称）
        //     8点 single     → inner = [0.306, 0.444, 0.556, 0.694]（对称）
        std::vector<double> inner;
        {
            const int n_inner = n_cv_local - c.degree - 1;
            const double u0 = c.cvs_u.front();
            const double u1 = c.cvs_u.back();
            const double u_span = u1 - u0;
            if (n_inner > 0 && u_span > 1e-9) {
                inner.reserve(n_inner);
                const int offset = (n_cv_local - n_inner) / 2;
                for (int i = 0; i < n_inner; ++i) {
                    int j = i + offset;
                    if (j < 0) j = 0;
                    if (j >= n_cv_local) j = n_cv_local - 1;
                    inner.push_back((c.cvs_u[j] - u0) / u_span);
                }
            }
        }
        c.knots = ocd_math::make_clamped_knots_uneven(n_cv_local, c.degree, inner);
        if ((int)c.knots.size() != n_cv_local + c.degree + 1) return false;
        return true;
        };

    // ---- 模式 A：单条长曲线 ----
    if (params.curveMode == 0) {
        MosCurve c;
        // 曲线跨 n_bones-1 段骨骼，第 i 段占参数区间 [i, i+1]/(n_bones-1)，
        // 保证多骨骼拼接后 u 仍严格递增且在 [0,1] 内。
        const double u_den = (n_bones > 1) ? double(n_bones - 1) : 1.0;
        for (int i = 0; i + 1 < n_bones; ++i) {
            const unsigned int B_i = chain[i];
            const mathcore::Vec3 start = bind_jp[B_i];
            const mathcore::Vec3 dir = bone_dir[i];
            const double L = bone_len[i];

            push_cv(c, mathcore::v3_add(start,
                mathcore::v3_scale(dir, L * 0.10)), B_i, (i + 0.10) / u_den);
            push_cv(c, mathcore::v3_add(start,
                mathcore::v3_scale(dir, L * 0.35)), B_i, (i + 0.35) / u_den);
            push_cv(c, mathcore::v3_add(start,
                mathcore::v3_scale(dir, L * 0.65)), B_i, (i + 0.65) / u_den);
            push_cv(c, mathcore::v3_add(start,
                mathcore::v3_scale(dir, L * 0.90)), B_i, (i + 0.90) / u_den);
        }

        // 末端锁定：锁最后一个控制点到父骨骼
        // 2 骨骼 → 8 个控制点，条件从 >=6 放宽到 >=4 就够
        if (params.endLock && c.cvs_rest.size() >= 4) {
            c.cvs_bone.back() = chain[n_bones - 2];
        }
        // single_curve 模式：弯曲关节在 deform 时动态选
        c.bend_joint_idx = -1;


        if (!finalize_curve(c)) return false;
        m_mosCache.curves.push_back(std::move(c));
    }
    else {
        // ---- 模式 B：每关节一条曲线（per-joint，专利 US9786083B2 语义）----
        //   曲线跨越关节，cv 分布在关节两侧骨骼上。
        for (int j = 1; j < n_bones; ++j) {
            const unsigned int B_parent = chain[j - 1];
            const unsigned int B_child = chain[j];
            const mathcore::Vec3 joint = bind_jp[B_child];
            const mathcore::Vec3 dir_parent = bone_dir[j - 1];
            const mathcore::Vec3 dir_child = bone_dir[j];
            const double L_parent = bone_len[j - 1];
            const double L_child = bone_len[j];

            MosCurve c;
            // 7 点对称布局：parent 侧 3 点、关节 1 点、child 侧 3 点
            // u 关于 0.5 镜像：0.25 ↔ 0.75, 0.35 ↔ 0.65, 0.45 ↔ 0.55, 0.5 自对称
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.5)), B_parent, 0.25);
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.3)), B_parent, 0.35);
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.1)), B_parent, 0.45);
            // 关节 1 点（u = 0.5，父子对称唯一锚点）
            push_cv(c, joint, B_child, 0.5);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.1)), B_child, 0.55);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.3)), B_child, 0.65);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.5)), B_child, 0.75);

            if (params.endLock && j + 1 == n_bones && c.cvs_rest.size() >= 4) {
                c.cvs_bone.back() = B_parent;
            }

            c.joint_anchor_bind = joint;
            if (!finalize_curve(c)) return false;

            c.bend_joint_idx = j;
            m_mosCache.curves.push_back(std::move(c));
        }
    }

    m_mosCache.n_curves = (unsigned int)m_mosCache.curves.size();
    if (m_mosCache.curves.empty()) return false;

    // ---- 平均骨骼长度 ----
    {
        double total = 0.0;
        int cnt = 0;
        for (size_t i = 0; i + 1 < chain.size(); ++i) {
            unsigned int b0 = chain[i], b1 = chain[i + 1];
            if (b0 < bind_jp.size() && b1 < bind_jp.size()) {
                total += mathcore::v3_len(
                    mathcore::v3_sub(bind_jp[b1], bind_jp[b0]));
                ++cnt;
            }
        }
        m_mosCache.avg_bone_length = (cnt > 0) ? (total / cnt) : 1.0;
        if (m_mosCache.avg_bone_length < 1e-6)
            m_mosCache.avg_bone_length = 1.0;
    }

    // ---- 预采样 ----
    const int N_SAMPLE = std::max(64, params.curveSamples);
    struct CurveSamples {
        std::vector<mathcore::Vec3> samples;
        std::vector<double> ts;
    };
    std::vector<CurveSamples> all_samples(m_mosCache.curves.size());
    for (size_t c = 0; c < m_mosCache.curves.size(); ++c) {
        const auto& cv = m_mosCache.curves[c];
        auto& s = all_samples[c];
        s.samples.resize(N_SAMPLE);
        s.ts.resize(N_SAMPLE);
        for (int i = 0; i < N_SAMPLE; ++i) {
            double t = double(i) / double(N_SAMPLE - 1);
            s.ts[i] = t;
            s.samples[i] = ocd_math::bspline_eval(
                cv.cvs_rest, cv.knots, cv.degree, t);
        }
    }

    // ---- 逐顶点：解 τ / O_i / w_c / vertex_joint ----
    m_mosCache.verts.assign(numVertices, {});

#pragma omp parallel for schedule(static)
    for (int vi = 0; vi < (int)numVertices; ++vi) {
        const mathcore::Vec3 pv = {
            (double)points[vi].x, (double)points[vi].y, (double)points[vi].z
        };

        // vertex_joint：该顶点在 chain 上最近的 joint
        int vj = -1;
        {
            double best_d2 = 1e30;
            for (int i = 0; i < n_bones; ++i) {
                mathcore::Vec3 diff = mathcore::v3_sub(pv, bind_jp[chain[i]]);
                double d2 = mathcore::v3_dot(diff, diff);
                if (d2 < best_d2) { best_d2 = d2; vj = i; }
            }
        }

        MosVertexBind vb;
        vb.per_curve.resize(m_mosCache.curves.size());

        for (size_t c = 0; c < m_mosCache.curves.size(); ++c) {
            const auto& cv = m_mosCache.curves[c];
            const auto& s = all_samples[c];

            // 粗采样 → 牛顿细化
            int best_i = 0;
            double best_d = 1e30;
            for (int i = 0; i < N_SAMPLE; ++i) {
                mathcore::Vec3 diff = mathcore::v3_sub(pv, s.samples[i]);
                double dd = mathcore::v3_dot(diff, diff);
                if (dd < best_d) { best_d = dd; best_i = i; }
            }
            double t = ocd_math::refine_closest_t(
                cv.cvs_rest, cv.knots, cv.degree, pv, s.ts[best_i]);
            if (t < 0.0) t = 0.0;
            if (t > 1.0) t = 1.0;

            const mathcore::Vec3 C_tau = ocd_math::bspline_eval(
                cv.cvs_rest, cv.knots, cv.degree, t);

            // ★ EQ.9 在 bind 时（F_ij = I）的精确特例
            const mathcore::Vec3 O_i = mathcore::v3_sub(pv, C_tau);

            // 衰减中心：single_curve / per-joint 都用顶点在曲线上的最近点 C_tau
            mathcore::Vec3 w_center = C_tau;
            if (params.curveMode != 0 && cv.cvs_rest.size() >= 3) {
                w_center = cv.cvs_rest[cv.cvs_rest.size() / 2];
            }
            const double d_curve = mathcore::v3_len(
                mathcore::v3_sub(pv, w_center));

            const double L_bone = m_mosCache.avg_bone_length;
            double w_c = 0.0;

            if (params.curveMode == 0) {
                // single_curve：1L 高斯（曲线覆盖整条链，保持原行为）
                const double x_wc = d_curve / std::max(1e-9, L_bone);
                w_c = std::exp(-x_wc * x_wc);
            }
            else {
                // ★ per_joint：w_c 直接等于线性权重混合度
                //   修复 W 形：去掉几何衰减（w_geom）和 gamma 抬升（w_mix_gamma），
                //   让"限定范围"完全由权重混合度控制。
                //
                //   衰减曲线（线性）：
                //     权重 0.5/0.5 → w_mix_raw = 1.0 → 100%（中线峰，不变）
                //     权重 0.4/0.6 → w_mix_raw = 0.8 → 80%（用户要求的"8成"）
                //     权重 0.3/0.7 → w_mix_raw = 0.6 → 60%
                //     权重 0.1/0.9 → w_mix_raw = 0.2 → 20%
                //     权重 1/0   → w_mix_raw = 0.0 → 0%（范围严格结束）
                //
                //   这个曲线与 LBS 的 candy-wrapper 收缩量成正比——
                //   MOS 的物理定位就是"补偿权重混合区的收缩"，收缩量 ∝ 权重混合度。
                //   这也避免了 w_mix × spatial_w 双重衰减导致 W 形的问题。
                const int bj_local = cv.bend_joint_idx;
                if (bj_local >= 1 && bj_local < n_bones &&
                    vi < (int)m_vtxCache.vertices.size() &&
                    !m_vtxCache.vertices[vi].empty()) {
                    const unsigned int parent_bone = chain[bj_local - 1];
                    const unsigned int child_bone = chain[bj_local];
                    double w_parent = 0.0;
                    double w_child = 0.0;
                    for (const auto& kv : m_vtxCache.vertices[vi]) {
                        if (kv.first == parent_bone) w_parent = (double)kv.second;
                        else if (kv.first == child_bone) w_child = (double)kv.second;
                    }
                    w_c = 2.0 * std::min(w_parent, w_child);
                }
            }

            MosPerCurveBind pc;
            pc.tau = t;
            pc.O = O_i;
            pc.w_c = w_c;
            pc.vertex_joint = vj;
            vb.per_curve[c] = pc;
        }

        m_mosCache.verts[vi] = std::move(vb);
    }

    m_mosCache.numVertices = numVertices;
    m_mosCache.valid = true;

    auto tEnd = std::chrono::high_resolution_clock::now();
    double ms = std::chrono::duration<double, std::milli>(tEnd - tStart).count();

    if (optSkinDebugLog() || params.debugLog) {
        size_t active = 0;
        for (const auto& vb : m_mosCache.verts) {
            for (const auto& pcb : vb.per_curve) {
                if (pcb.w_c > 0.0) { ++active; break; }
            }
        }
        const double ratio = m_mosCache.verts.empty()
            ? 0.0 : (double)active / (double)m_mosCache.verts.size();

        char buf[320];
        snprintf(buf, sizeof(buf),
            "[MOS] bind: %u verts, %u curves, w_c>0=%.1f%%, mode=%d, avgLen=%.3f  [%.1fms]",
            numVertices, m_mosCache.n_curves, ratio * 100.0,
            params.curveMode, m_mosCache.avg_bone_length, ms);
        MGlobal::displayInfo(MString(buf));
        // τ 直方图：看参与 MOS 的顶点参数分布
        int bins[10] = { 0 };
        int total = 0;
        for (const auto& vb : m_mosCache.verts) {
            for (const auto& pcb : vb.per_curve) {
                if (pcb.w_c > 0.0) {
                    int b = (int)(pcb.tau * 10.0);
                    if (b < 0) b = 0;
                    if (b > 9) b = 9;
                    bins[b]++;
                    total++;
                }
            }
        }
        char hbuf[256];
        snprintf(hbuf, sizeof(hbuf),
            "[MOS] tau histogram (w_c>0, total=%d): "
            "[0-.1]=%d [.1-.2]=%d [.2-.3]=%d [.3-.4]=%d [.4-.5]=%d "
            "[.5-.6]=%d [.6-.7]=%d [.7-.8]=%d [.8-.9]=%d [.9-1]=%d",
            total, bins[0], bins[1], bins[2], bins[3], bins[4],
            bins[5], bins[6], bins[7], bins[8], bins[9]);
        MGlobal::displayInfo(MString(hbuf));
    }
    return true;
}

MStatus OptimizedSkinCluster::deformMOS(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    const unsigned int nV = iter.count();

    MosParams curParams = readMosParams(block);
    if (!m_mosCache.valid || m_mosCache.numVertices != nV ||
        !m_mosCache.params.nearlyEqual(curParams)) {
        if (!computeMosCache(block, nV, curParams))
            return deformDLB(block, iter, worldMat, 0);
    }

    const double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    const int n_curves = (int)m_mosCache.curves.size();
    if (n_curves == 0) return deformDLB(block, iter, worldMat, 0);

    const MosParams& P = m_mosCache.params;

    // ============================================================
    // 每条曲线的弯曲关节（per-joint 静态指定；single_curve 动态选最大旋转关节）
    // ============================================================
    std::vector<int> curve_bend_idx(n_curves, -1);
    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        if (cv.bend_joint_idx >= 0) {
            curve_bend_idx[c] = cv.bend_joint_idx;
        }
        else {
            const int nB = (int)cv.chain_bones.size();
            int best_j = -1;
            double best_angle = 0.0;
            for (int j = 1; j < nB; ++j) {
                const unsigned int bp = cv.chain_bones[j - 1];
                const unsigned int bc = cv.chain_bones[j];
                if (bp >= Cs.size() || bc >= Cs.size()) continue;
                mathcore::Quat q_rel = mathcore::q_mul(
                    Cs[bc].q, mathcore::q_inverse(Cs[bp].q));
                double cos_half = std::abs(q_rel.w);
                if (cos_half > 1.0) cos_half = 1.0;
                double angle = 2.0 * std::acos(cos_half);
                if (angle > best_angle) { best_angle = angle; best_j = j; }
            }
            curve_bend_idx[c] = best_j;
        }
    }

    // ============================================================
    // 为每条曲线预计算：
    //   curve_tau_bend[c]   : 关节在曲线上的参数位置
    //   curve_bend_angle[c] : 弯曲角（swing 分离后，纯 twist 不触发 bulge）
    // ============================================================
    std::vector<double> curve_tau_bend(n_curves, 0.5);
    std::vector<double> curve_bend_angle(n_curves, 0.0);

    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        const int bj = curve_bend_idx[c];
        if (bj < 1 || bj >= (int)cv.chain_bones.size()) continue;

        const unsigned int bp = cv.chain_bones[bj - 1];
        const unsigned int bc = cv.chain_bones[bj];

        // single_curve 和 per-joint 都用关节在曲线上的最近参数作为 tau_bend
        if (bc < m_boneInfo.jointPosRest.size()) {
            const mathcore::Vec3 joint_bind = m_boneInfo.jointPosRest[bc];
            double t_best = 0.5;
            double d_best = 1e30;
            const int N_SAMPLE_T = 64;
            for (int i = 0; i <= N_SAMPLE_T; ++i) {
                const double t = (double)i / N_SAMPLE_T;
                const mathcore::Vec3 p = ocd_math::bspline_eval(
                    cv.cvs_rest, cv.knots, cv.degree, t);
                const double d = mathcore::v3_len(
                    mathcore::v3_sub(p, joint_bind));
                if (d < d_best) { d_best = d; t_best = t; }
            }
            curve_tau_bend[c] = t_best;
        }
        else {
            curve_tau_bend[c] = 0.5;
        }

        // 弯曲角（swing 分离）
        // 把相对旋转 q_rel 分解为 swing（弯曲）+ twist（绕轴），
        // bulge 只由 swing 分量驱动，纯 twist 不触发 bulge。
        if (bp < Cs.size() && bc < Cs.size()) {
            mathcore::Quat q_rel = mathcore::q_mul(
                Cs[bc].q, mathcore::q_inverse(Cs[bp].q));

            mathcore::Vec3 axis_bind = mathcore::Vec3{ 0.0, 1.0, 0.0 };
            if (bc < m_boneInfo.boneDirRest.size()) {
                axis_bind = m_boneInfo.boneDirRest[bc];
            }

            mathcore::Quat R_swing, R_twist;
            mathcore::Vec3 axis_cur;
            double twist_angle = 0.0;
            ocd_math::decompose_swing_twist(
                q_rel, axis_bind, R_swing, R_twist, axis_cur, twist_angle);

            double cos_half = std::abs(R_swing.w);
            if (cos_half > 1.0) cos_half = 1.0;
            const double angle = 2.0 * std::acos(cos_half);
            curve_bend_angle[c] =
                std::min(angle / 3.14159265358979323846, 1.0);
        }
    }

    if (optSkinDebugLog() || P.debugLog) {
        char buf[256];
        for (int c = 0; c < n_curves; ++c) {
            snprintf(buf, sizeof(buf),
                "[MOS] curve[%d]: bend_idx=%d tau_bend=%.4f angle=%.3f",
                c, curve_bend_idx[c],
                curve_tau_bend[c], curve_bend_angle[c]);
            MGlobal::displayInfo(MString(buf));
        }
    }

    // ============================================================
    // 每个控制点对弯曲关节的从属权重 w_j
    //   w_j = exp(-(dist(cv_bind_j, joint_bind) / (0.5 * L_bone))²)
    // 用于 centerSlide 控制点级修正
    // ============================================================
    std::vector<std::vector<double>> all_cv_bend_weight(n_curves);
    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        const int n_cv = (int)cv.cvs_rest.size();
        all_cv_bend_weight[c].assign(n_cv, 0.0);

        const int bj = curve_bend_idx[c];
        if (bj < 1 || bj >= (int)cv.chain_bones.size()) continue;

        const unsigned int bc = cv.chain_bones[bj];
        if (bc >= m_boneInfo.jointPosRest.size()) continue;

        const mathcore::Vec3 joint_bind = m_boneInfo.jointPosRest[bc];
        const double half = 0.5 * m_mosCache.avg_bone_length;
        const double inv_half = 1.0 / std::max(1e-9, half);

        for (int j = 0; j < n_cv; ++j) {
            const double d = mathcore::v3_len(
                mathcore::v3_sub(cv.cvs_rest[j], joint_bind));
            const double x = d * inv_half;
            all_cv_bend_weight[c][j] = std::exp(-x * x);
        }
    }

    // ============================================================
    // 预计算控制点输入 p_ij / F_ij（专利 US9786083B2 EQ.5 单骨骼语义）
    //
    //   v_i = Σ_j (p_ij + F_ij·O_i) · w_ij
    //     · p_ij = 控制点 j 在**单一骨骼** cvs_bone[j] 变换下的位置
    //     · F_ij = **单一骨骼** cvs_bone[j] 的旋转
    //     · w_ij = B-spline 基函数（关节处的 C2 平滑由基函数提供）
    //
    //   之前用相邻骨骼软权重混合 p_ij（等价于 LBS 位置），
    //   会让曲线本身跟随 LBS 在关节处收缩，
    //   导致中线区域向骨骼折线内侧塌陷——这是中线偏移的根因之一。
    //   单骨骼语义下，父骨骼控制点不动、子骨骼控制点跟随旋转，
    //   B-spline 平滑折角，曲线经过关节而不收缩。
    // ============================================================
    std::vector<std::vector<mathcore::Vec3>> all_p_j(n_curves);
    std::vector<std::vector<ocd_math::Mat3>> all_F_j(n_curves);

    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        const int n_cv = (int)cv.cvs_rest.size();
        const int nB = (int)cv.chain_bones.size();
        all_p_j[c].resize(n_cv);
        all_F_j[c].resize(n_cv);

        // u_j → chain 段索引 t_global 的仿射映射：
        //   per-joint（bend_joint_idx >= 1）：曲线只跨一个关节，t = (j-1) + 2u
        //   single_curve（bend_joint_idx = -1）：曲线跨整条链，t = u·(nB-1)
        double seg_begin = 0.0;
        double seg_scale = (nB > 1) ? double(nB - 1) : 1.0;
        if (cv.bend_joint_idx >= 1 && cv.bend_joint_idx < nB) {
            seg_begin = double(cv.bend_joint_idx - 1);
            seg_scale = 2.0;
        }

        for (int j = 0; j < n_cv; ++j) {
            // ---- p_j：单骨骼硬分配（保留位置不因 LBS 收缩）----
            const unsigned int bone = cv.cvs_bone[j];
            if (bone < (unsigned int)Cs.size()) {
                all_p_j[c][j] = mathcore::apply_transform(Cs[bone], cv.cvs_rest[j]);
            }
            else {
                all_p_j[c][j] = cv.cvs_rest[j];
            }

            // ---- F_j：按 cvs_u 的软权重混合 + polar_rotation ----
            // 让每个控制点的旋转等于"该位置 t_global 处的骨骼混合旋转"：
            //   t_global = k + 0.5 处 = 完整 k 骨骼旋转
            //   t_global = k 处（关节）= k-1/k 各 0.5 → 45°（mid 弯 90° 时）
            // 这样 B-spline 混合后 F_cur(τ) 是 τ 的连续渐变，
            // 中线 τ=0.5 严格 45°，两侧 0°→45°→90° 平滑过渡。
            // 硬分配时 F_4 = 90° 会把 F_cur(0.5) 拉到 52.5°，这是中线跑偏的根因。
            const double u = (j < (int)cv.cvs_u.size()) ? cv.cvs_u[j] : 0.0;
            const double t_global = seg_begin + u * seg_scale;

            ocd_math::Mat3 R_acc;
            for (int ii = 0; ii < 9; ++ii) R_acc.m[ii] = 0.0;
            double w_sum = 0.0;
            for (int k = 0; k < nB; ++k) {
                const double w = 1.0 - std::abs(t_global - (double(k) + 0.5));
                if (w <= 1e-9) continue;
                const unsigned int b = cv.chain_bones[k];
                if (b >= (unsigned int)Cs.size()) continue;
                const ocd_math::Mat3 R_b = ocd_math::mat3_from_quat(Cs[b].q);
                for (int ii = 0; ii < 9; ++ii) R_acc.m[ii] += R_b.m[ii] * w;
                w_sum += w;
            }

            if (w_sum > 1e-9) {
                const double inv = 1.0 / w_sum;
                for (int ii = 0; ii < 9; ++ii) R_acc.m[ii] *= inv;
                all_F_j[c][j] = ocd_math::polar_rotation(R_acc);
            }
            else if (bone < (unsigned int)Cs.size()) {
                // 兜底：曲线链与控制链骨骼索引不一致时退回单骨骼硬分配
                all_F_j[c][j] = ocd_math::mat3_from_quat(Cs[bone].q);
            }
            else {
                all_F_j[c][j] = ocd_math::Mat3::identity();
            }
        }
    }

    // ============================================================
    // ★ Scale 主动控制（专利 US9418465B2）
    //   骨骼缩放（squash/stretch）通过 matrix 的 3×3 线性部分传递到皮肤。
    //   实现：从 matrix × bindPreMatrix 提取列向量 L0/L1/L2，
    //        按 bind 骨骼轴投影得到缩放标量 s_axis，
    //        在 all_F_j 上叠加 mat3_axis_scale(axis_cur, s_axis^scaleWeight)，
    //        使顶点从控制点向外膨胀/收缩，跟随骨骼缩放。
    //   scaleWeight=0 → s^0=1 → S=I，严格零回归。
    // ============================================================
    if (P.scaleWeight > 1e-9 && P.scaleMode != 0) {
        std::vector<mathcore::Vec3> L0, L1, L2;
        const bool haveLinear = readBoneLinearParts(block, worldMat, L0, L1, L2);
        if (haveLinear) {
            for (int c = 0; c < n_curves; ++c) {
                const auto& cv = m_mosCache.curves[c];
                const int n_cv = (int)cv.cvs_rest.size();

                for (int j = 0; j < n_cv; ++j) {
                    const unsigned int bone = cv.cvs_bone[j];
                    if (bone >= (unsigned int)L0.size()) continue;

                    if (P.scaleMode == 1) {
                        // ---- 轴向对称缩放 ----
                        // bind 骨骼轴（控制点所在骨骼的 bind 方向）
                        mathcore::Vec3 a_bind = { 0.0, 1.0, 0.0 };
                        if (bone < m_boneInfo.boneDirRest.size()) {
                            a_bind = m_boneInfo.boneDirRest[bone];
                        }

                        // s_axis = |L · a_bind|（线性部分沿骨骼轴的长度）
                        const mathcore::Vec3 La = mathcore::v3_add(
                            mathcore::v3_add(
                                mathcore::v3_scale(L0[bone], a_bind.x),
                                mathcore::v3_scale(L1[bone], a_bind.y)),
                            mathcore::v3_scale(L2[bone], a_bind.z));
                        const double s_axis = mathcore::v3_len(La);
                        if (s_axis < 1e-9) continue;

                        // 当前轴（all_F_j 作用在 bind 轴上的方向）
                        const mathcore::Vec3 axis_cur = all_F_j[c][j].apply(a_bind);
                        if (mathcore::v3_len(axis_cur) < 1e-9) continue;

                        // 缩放幂次：s^scaleWeight（0 → 1，1 → s_axis）
                        const double s_pow = std::pow(s_axis, P.scaleWeight);
                        const ocd_math::Mat3 S =
                            ocd_math::mat3_axis_scale(axis_cur, s_pow);
                        all_F_j[c][j] = S * all_F_j[c][j];
                    }
                    else if (P.scaleMode == 2) {
                        // ---- 各向异性缩放 ----
                        // 用三个列向量的模长作为三个主轴的缩放
                        const double sx = mathcore::v3_len(L0[bone]);
                        const double sy = mathcore::v3_len(L1[bone]);
                        const double sz = mathcore::v3_len(L2[bone]);
                        const ocd_math::Mat3 S = ocd_math::mat3_anisotropic_scale(
                            std::pow(std::max(sx, 1e-9), P.scaleWeight),
                            std::pow(std::max(sy, 1e-9), P.scaleWeight),
                            std::pow(std::max(sz, 1e-9), P.scaleWeight));
                        all_F_j[c][j] = S * all_F_j[c][j];
                    }
                }
            }
        }
    }

    MPointArray points;
    iter.allPositions(points);
    const MMatrix wInv = worldMat.inverse();

    const bool hasInBulge = (std::abs(P.inBulgeWeight) > 1e-9);
    const bool hasOutBulge = (std::abs(P.outBulgeWeight) > 1e-9);
    const bool hasSideBulge = (std::abs(P.sideBulgeWeight) > 1e-9);
    const bool hasCenterSlide =
        (std::abs(P.centerSlideInWeight) > 1e-9 ||
            std::abs(P.centerSlideOutWeight) > 1e-9);
    const bool hasInSlide = (std::abs(P.slideInWeight_prev) > 1e-9 ||
        std::abs(P.slideInWeight_next) > 1e-9);
    const bool hasOutSlide = (std::abs(P.slideOutWeight_prev) > 1e-9 ||
        std::abs(P.slideOutWeight_next) > 1e-9);
    const bool hasInOrthoSlide = (std::abs(P.slideOrthoInWeight_prev) > 1e-9 ||
        std::abs(P.slideOrthoInWeight_next) > 1e-9);
    const bool hasOutOrthoSlide = (std::abs(P.slideOrthoOutWeight_prev) > 1e-9 ||
        std::abs(P.slideOrthoOutWeight_next) > 1e-9);
    // ★ 总开关：任一 slide 属性非零时进入 Slide 段
    // （Slide 段内部会按 hasInSlide / hasOutSlide / ... 细分判断，这里只做快速跳过）
    const bool hasAnySlide = hasInSlide || hasOutSlide ||
        hasInOrthoSlide || hasOutOrthoSlide ||
        hasCenterSlide;


    // 调试统计
    double g_cvdisp_sum = 0.0;
    double g_cvdisp_max = 0.0;
    int    g_cvdisp_count = 0;
    double g_bind_max_diff = 0.0;
    int    g_bind_max_idx = -1;

    // Bind 姿势检测（所有 Cs.q ≈ 单位四元数）
    bool is_bind_pose = true;
    for (size_t i = 0; i < Cs.size(); ++i) {
        if (std::abs(Cs[i].q.w) < 0.9999 ||
            std::abs(Cs[i].q.x) > 1e-3 ||
            std::abs(Cs[i].q.y) > 1e-3 ||
            std::abs(Cs[i].q.z) > 1e-3) {
            is_bind_pose = false;
            break;
        }
    }



#pragma omp parallel
    {
        std::vector<mathcore::Vec3> offset_cvs;
        double l_cvdisp_sum = 0.0;
        double l_cvdisp_max = 0.0;
        int    l_cvdisp_count = 0;

#pragma omp for schedule(static)
        for (int vi = 0; vi < (int)nV; ++vi) {
            const auto& vb = m_mosCache.verts[vi];

            MPoint ptW = points[vi] * worldMat;
            const mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };

            // LBS 兜底
            mathcore::Vec3 lbs_result = origPos;
            if (vi < (int)m_vtxCache.vertices.size() &&
                !m_vtxCache.vertices[vi].empty()) {
                mathcore::Vec3 acc = { 0.0, 0.0, 0.0 };
                double w_tot = 0.0;
                for (const auto& kv : m_vtxCache.vertices[vi]) {
                    const unsigned int b = kv.first;
                    const double w = (double)kv.second;
                    if (b >= (unsigned int)Cs.size()) continue;
                    acc = mathcore::v3_add(acc,
                        mathcore::v3_scale(
                            mathcore::apply_transform(Cs[b], origPos), w));
                    w_tot += w;
                }
                if (w_tot > 1e-9)
                    lbs_result = mathcore::v3_scale(acc, 1.0 / w_tot);
            }

            mathcore::Vec3 def = lbs_result;
            if (!vb.per_curve.empty()) {
                mathcore::Vec3 blended = { 0.0, 0.0, 0.0 };
                double w_total = 0.0;

                for (int c = 0; c < n_curves; ++c) {
                    const auto& cv = m_mosCache.curves[c];
                    const auto& pc = vb.per_curve[c];
                    if (pc.w_c <= 0.0) continue;

                    const int n_cv_local = (int)cv.cvs_rest.size();
                    offset_cvs.resize(n_cv_local);

                    // ★ 过渡区体积补偿：w_c ∈ (0,1) 时放大控制点偏移量
                    //   · w_c=0（LBS 边界）：w_edge=0，expand_scale=1，不放大
                    //   · w_c=1（中线 0.5/0.5）：w_edge=0，expand_scale=1，中线 45° 保持
                    //   · w_c=0.5（过渡区核心）：w_edge=1，expand_scale=1+0.3·angle
                    //   · bind 时 curve_bend_angle=0 → expand_scale=1 → bind 精确性不变
                    //
                    //   ★ 关键：只放大「弯曲平面内」分量，侧向分量保持 1 倍。
                    //   侧向补偿完全由 mosSideBulgeWeight 控制的 Bulge 段独立提供，
                    //   expand_scale 不再引入额外侧向膨胀。
                    double expand_scale = 1.0;
                    if (pc.w_c > 1e-4 && pc.w_c < 1.0 - 1e-4) {
                        const double w_edge = 4.0 * pc.w_c * (1.0 - pc.w_c);
                        expand_scale = 1.0 + 0.3 * w_edge * curve_bend_angle[c];
                    }

                    // 计算当前曲线弯曲平面的法向（侧向方向），用于分离平面内/侧向分量
                    //   需要 chain 中至少有 3 根骨骼（prev / bend / next）才能算出法向
                    mathcore::Vec3 side_normal_exp = { 0.0, 0.0, 0.0 };
                    bool has_side_normal_exp = false;
                    if (expand_scale > 1.0) {
                        const int bj_sn = curve_bend_idx[c];
                        const int nB_sn = (int)cv.chain_bones.size();
                        if (bj_sn >= 1 && bj_sn + 1 < nB_sn) {
                            const unsigned int bp_sn = cv.chain_bones[bj_sn - 1];
                            const unsigned int bc_sn = cv.chain_bones[bj_sn];
                            const unsigned int bn_sn = cv.chain_bones[bj_sn + 1];
                            if (bp_sn < rest.size() && bc_sn < rest.size() && bn_sn < rest.size() &&
                                bp_sn < Cs.size() && bc_sn < Cs.size() && bn_sn < Cs.size()) {
                                const mathcore::Vec3 bp_cur_sn = mathcore::apply_transform(Cs[bp_sn], rest[bp_sn].t);
                                const mathcore::Vec3 bc_cur_sn = mathcore::apply_transform(Cs[bc_sn], rest[bc_sn].t);
                                const mathcore::Vec3 bn_cur_sn = mathcore::apply_transform(Cs[bn_sn], rest[bn_sn].t);
                                const mathcore::Vec3 ax_prev_sn = mathcore::v3_sub(bc_cur_sn, bp_cur_sn);
                                const mathcore::Vec3 ax_next_sn = mathcore::v3_sub(bn_cur_sn, bc_cur_sn);
                                mathcore::Vec3 sn = mathcore::v3_cross(ax_prev_sn, ax_next_sn);
                                const double sn_len = mathcore::v3_len(sn);
                                if (sn_len > 1e-6) {
                                    side_normal_exp = mathcore::v3_scale(sn, 1.0 / sn_len);
                                    has_side_normal_exp = true;
                                }
                            }
                        }
                    }

                    for (int j = 0; j < n_cv_local; ++j) {
                        const mathcore::Vec3 p_j = all_p_j[c][j];
                        mathcore::Vec3 F_O = all_F_j[c][j].apply(pc.O);

                        if (expand_scale > 1.0) {
                            if (has_side_normal_exp) {
                                // 分解：侧向分量保持 1 倍，平面内分量放大 expand_scale 倍
                                const double side_comp = mathcore::v3_dot(F_O, side_normal_exp);
                                const mathcore::Vec3 side_part = mathcore::v3_scale(side_normal_exp, side_comp);
                                const mathcore::Vec3 plane_part = mathcore::v3_sub(F_O, side_part);
                                F_O = mathcore::v3_add(
                                    mathcore::v3_scale(plane_part, expand_scale),
                                    side_part);
                            }
                            else {
                                // 退化（2 骨骼链等无法算侧向）：各向同性放大
                                F_O = mathcore::v3_scale(F_O, expand_scale);
                            }
                        }

                        offset_cvs[j] = mathcore::v3_add(p_j, F_O);
                    }

                    double t = pc.tau;
                    if (t < 0.0) t = 0.0;
                    if (t > 1.0 - 1e-12) t = 1.0 - 1e-12;

                    const mathcore::Vec3 curve_mos = ocd_math::bspline_eval(
                        offset_cvs, cv.knots, cv.degree, t);


                   
                    // ============================================================
                    // ★ 顶点级 Bulge + Slide（专利 US9418465B2）
                    //   Bulge：out（+radial）/ in（-radial）/ side（cross(radial, bone_axis)）
                    //   Slide：8 方向（prev/next × in/out × 轴向/正交），
                    //          轴向用当前关节位置差，与顶点权重无关
                    //   三向 bulge 和 8 向 slide 共享：
                    //     spatial_w（τ 局部化）、q_mix（顶点权重混合旋转）、
                    //     radial_cur（bind 径向旋转到当前姿势）
                    //   任意一项非零即进入，不再嵌套在 bulge 内部
                    // ============================================================
                    mathcore::Vec3 curve_mos_final = curve_mos;
                    if ((hasInBulge || hasOutBulge || hasSideBulge || hasAnySlide) &&
                        curve_bend_angle[c] > 1e-6 &&
                        curve_bend_idx[c] >= 1 &&
                        curve_bend_idx[c] < (int)cv.chain_bones.size() &&
                        vi < (int)m_vtxCache.vertices.size() &&
                        !m_vtxCache.vertices[vi].empty()) {

                        // ★ 去掉 spatial_w 几何衰减：范围完全交给 w_c（权重混合度）
                        //   保留一个大括号作用域，避免内部变量名冲突。
                            {
                            // ---- 共享：顶点权重混合旋转 ----
                            mathcore::Quat q_mix = { 0.0, 0.0, 0.0, 0.0 };
                            double wsum = 0.0;
                            for (const auto& kv : m_vtxCache.vertices[vi]) {
                                const unsigned int b = kv.first;
                                const double w = (double)kv.second;
                                if (b >= (unsigned int)Cs.size() || w <= 1e-9) continue;
                                q_mix.x += Cs[b].q.x * w;
                                q_mix.y += Cs[b].q.y * w;
                                q_mix.z += Cs[b].q.z * w;
                                q_mix.w += Cs[b].q.w * w;
                                wsum += w;
                            }

                            if (wsum > 1e-9) {
                                q_mix = mathcore::q_normalize(q_mix);

                                // ---- 共享：当前关节位置与骨骼轴向 ----
                                const int bj = curve_bend_idx[c];
                                const int nB = (int)cv.chain_bones.size();
                                const unsigned int bp_idx = cv.chain_bones[bj - 1];
                                const unsigned int bc_idx = cv.chain_bones[bj];
                                const unsigned int bn_idx = (bj + 1 < nB)
                                    ? cv.chain_bones[bj + 1] : bc_idx;

                                mathcore::Vec3 bp_cur = { 0.0, 0.0, 0.0 };
                                mathcore::Vec3 bc_cur = { 0.0, 0.0, 0.0 };
                                mathcore::Vec3 bn_cur = { 0.0, 0.0, 0.0 };
                                if (bp_idx < rest.size() && bp_idx < Cs.size())
                                    bp_cur = mathcore::apply_transform(
                                        Cs[bp_idx], rest[bp_idx].t);
                                if (bc_idx < rest.size() && bc_idx < Cs.size())
                                    bc_cur = mathcore::apply_transform(
                                        Cs[bc_idx], rest[bc_idx].t);
                                if (bn_idx < rest.size() && bn_idx < Cs.size() &&
                                    bn_idx != bc_idx)
                                    bn_cur = mathcore::apply_transform(
                                        Cs[bn_idx], rest[bn_idx].t);
                                else
                                    bn_cur = bc_cur;

                                mathcore::Vec3 bone_axis_prev_cur =
                                    mathcore::v3_sub(bc_cur, bp_cur);
                                const double len_prev =
                                    mathcore::v3_len(bone_axis_prev_cur);
                                if (len_prev > 1e-9)
                                    bone_axis_prev_cur = mathcore::v3_scale(
                                        bone_axis_prev_cur, 1.0 / len_prev);
                                else
                                    bone_axis_prev_cur = mathcore::Vec3{ 0.0, 1.0, 0.0 };

                                mathcore::Vec3 bone_axis_next_cur =
                                    mathcore::v3_sub(bn_cur, bc_cur);
                                const double len_next =
                                    mathcore::v3_len(bone_axis_next_cur);
                                if (len_next > 1e-9)
                                    bone_axis_next_cur = mathcore::v3_scale(
                                        bone_axis_next_cur, 1.0 / len_next);
                                else
                                    bone_axis_next_cur = mathcore::Vec3{ 0.0, 1.0, 0.0 };

                                // in vector：内角平分线方向
                                mathcore::Vec3 in_dir_cur = { 0.0, 0.0, 0.0 };
                                {
                                    mathcore::Vec3 b = mathcore::v3_sub(
                                        bone_axis_next_cur, bone_axis_prev_cur);
                                    if (mathcore::v3_len(b) > 1e-6)
                                        in_dir_cur = mathcore::v3_norm(b);
                                }

                                mathcore::Vec3 radial_cur;
                                {
                                    bool used_manual = false;
                                    if (P.useManualInVector) {
                                        const mathcore::Vec3 mv = P.manualInVector;
                                        const double mv_len = mathcore::v3_len(mv);
                                        if (mv_len > 1e-9) {
                                            const mathcore::Vec3 mv_n =
                                                mathcore::v3_scale(mv, 1.0 / mv_len);
                                            const mathcore::Vec3 mv_cur =
                                                mathcore::q_rotate_v3(q_mix, mv_n);
                                            const double o_len = mathcore::v3_len(pc.O);
                                            radial_cur = mathcore::v3_scale(mv_cur, o_len);
                                            used_manual = true;
                                        }
                                    }
                                    if (!used_manual) {
                                        radial_cur = mathcore::q_rotate_v3(q_mix, pc.O);
                                    }

                                    // ★ 修复方向偏差：将 radial_cur 投影到"垂直骨骼轴"平面
                                    //   根因：q_mix 混合旋转对混合权重顶点会混入骨骼轴分量。
                                    //   mid 侧内侧顶点的 bind 径向 (-r,0,0) 经 q_mix 后变成
                                    //   (-0.3r, 0.7r, 0)——70% 是 +Y 骨骼轴分量，不再是"纯径向"。
                                    //   沿这个方向推 bulge 会朝骨骼轴倾斜 → 视觉上"挤压"而非"发散"，
                                    //   不同位置顶点偏移量不同 → 叠加 spatial_w 后形成 W 形。
                                    //
                                    //   修复：减去骨骼轴分量，保留垂直骨骼轴的"纯径向"。
                                    //   骨骼轴用"顶点权重混合的当前骨骼轴向"（bone_axis_mix_cur），
                                    //   与顶点的位置/权重一致，避免引入新的不连续性。
                                    {
                                        const double radial_len_orig =
                                            mathcore::v3_len(radial_cur);
                                        if (radial_len_orig > 1e-9) {
                                            // 计算顶点权重混合的当前骨骼轴向
                                            mathcore::Vec3 bone_axis_mix_cur =
                                            { 0.0, 0.0, 0.0 };
                                            double axis_w = 0.0;
                                            for (const auto& kv : m_vtxCache.vertices[vi]) {
                                                const unsigned int b = kv.first;
                                                const double w = (double)kv.second;
                                                if (b >= (unsigned int)Cs.size()) continue;
                                                if (b >= m_boneInfo.boneDirRest.size())
                                                    continue;
                                                if (w <= 1e-9) continue;
                                                const mathcore::Vec3 ba =
                                                    mathcore::q_rotate_v3(
                                                        Cs[b].q,
                                                        m_boneInfo.boneDirRest[b]);
                                                bone_axis_mix_cur = mathcore::v3_add(
                                                    bone_axis_mix_cur,
                                                    mathcore::v3_scale(ba, w));
                                                axis_w += w;
                                            }
                                            if (axis_w > 1e-9) {
                                                bone_axis_mix_cur = mathcore::v3_norm(
                                                    mathcore::v3_scale(
                                                        bone_axis_mix_cur,
                                                        1.0 / axis_w));

                                                // 投影：radial - (radial · axis) * axis
                                                const double axial_comp =
                                                    mathcore::v3_dot(
                                                        radial_cur,
                                                        bone_axis_mix_cur);
                                                const mathcore::Vec3 radial_perp =
                                                    mathcore::v3_sub(
                                                        radial_cur,
                                                        mathcore::v3_scale(
                                                            bone_axis_mix_cur,
                                                            axial_comp));
                                                const double perp_len =
                                                    mathcore::v3_len(radial_perp);

                                                if (perp_len > 1e-6) {
                                                    // 保持原长度（|pc.O|），只校正方向
                                                    radial_cur = mathcore::v3_scale(
                                                        radial_perp,
                                                        radial_len_orig / perp_len);
                                                }
                                                // perp_len 太小（径向几乎平行骨骼轴）时
                                                // 保留原 radial_cur——此时 bulge 本身很弱，
                                                // 方向偏差视觉影响小
                                            }
                                        }
                                    }
                                }

                                const double in_len = mathcore::v3_len(in_dir_cur);
                                const double pc_o_len = mathcore::v3_len(pc.O);
                                const double radial_len = mathcore::v3_len(radial_cur);

                                // ============================================================
                                // ★ 修复 W 形：base 引入 w_mix_gamma 作为主驱动
                                //   bulge 的物理目的是补偿 LBS 在权重混合区的 candy-wrapper
                                //   收缩，所以补偿量应与"权重混合度"同步衰减，而不是
                                //   与几何位置（spatial_w）同步。两者独立衰减会叠加成 W 形。
                                //
                                //   新公式：base = curve_bend_angle × w_mix_gamma × sqrt(spatial_w)
                                //     · w_mix_gamma：主驱动，与 LBS 收缩同步
                                //     · sqrt(spatial_w)：辅助限域，开方让几何衰减变缓
                                //   端点不变：中线 w_mix=1×spatial_w=1 → base 不变；
                                //           边界 w_mix=0 → base=0，范围不溢出。
                                // ============================================================
                                double w_mix_gamma_base = 0.0;
                                {
                                    const int bj_b = curve_bend_idx[c];
                                    const int nB_b = (int)cv.chain_bones.size();
                                    if (bj_b >= 1 && bj_b < nB_b &&
                                        vi < (int)m_vtxCache.vertices.size() &&
                                        !m_vtxCache.vertices[vi].empty()) {
                                        const unsigned int pb =
                                            cv.chain_bones[bj_b - 1];
                                        const unsigned int cb =
                                            cv.chain_bones[bj_b];
                                        double wp = 0.0, wcv = 0.0;
                                        for (const auto& kv : m_vtxCache.vertices[vi]) {
                                            if (kv.first == pb) wp = (double)kv.second;
                                            else if (kv.first == cb) wcv = (double)kv.second;
                                        }
                                        const double w_mix_raw =
                                            2.0 * std::min(wp, wcv);
                                        // gamma power 与 computeMosCache 的 w_mix 保持
                                        // 一致（2.0），保证 base 与 w_c 的权重曲线同源
                                        w_mix_gamma_base =
                                            1.0 - std::pow(1.0 - w_mix_raw, 2.0);
                                    }
                                }

                                const double base = curve_bend_angle[c];

                                // ==================== Bulge 段 ====================
                                // 方向分离：
                                //   in / out：沿 in_dir_cur（弯曲平面内，前后方向），
                                //             纯前后推动，不含左右分量
                                //   side    ：沿 side_normal（弯曲平面法向，左右方向），
                                //             中线衰减，左右对称向外（正值）/向内（负值）
                                // 两个方向正交，互补不重叠。
                                if (hasInBulge || hasOutBulge || hasSideBulge) {
                                    mathcore::Vec3 total_bulge = { 0.0, 0.0, 0.0 };

                                    if (in_len > 1e-9 && pc_o_len > 1e-9) {
                                        // ---- 分区权重：内侧/外侧 ----
                                        // cos_side = dot(pc.O, in_dir_cur) / |pc.O|
                                        //   cos_side > +0.5 → 内侧扇区（w_in 主导）
                                        //   cos_side < -0.5 → 外侧扇区（w_out 主导）
                                        //   交界处 cos_side ∈ [-0.5, +0.5] → smoothstep 平滑过渡
                                        const double cos_side =
                                            mathcore::v3_dot(pc.O, in_dir_cur) / pc_o_len;

                                        double t_norm = (cos_side + 0.5) / 1.0;
                                        if (t_norm < 0.0) t_norm = 0.0;
                                        if (t_norm > 1.0) t_norm = 1.0;
                                        const double smooth_w =
                                            t_norm * t_norm * (3.0 - 2.0 * t_norm);

                                        const double w_out = smooth_w;
                                        const double w_in = 1.0 - smooth_w;

                                        // ---- in / out：沿 in_dir_cur（前后方向）----
                                        // ★ in 沿 -in_dir（朝内角），out 沿 +in_dir（朝外角）
                                        if (hasInBulge && w_in > 1e-9) {
                                            total_bulge = mathcore::v3_add(
                                                total_bulge,
                                                mathcore::v3_scale(in_dir_cur,
                                                    -P.inBulgeWeight * base * w_in));
                                        }
                                        if (hasOutBulge && w_out > 1e-9) {
                                            total_bulge = mathcore::v3_add(
                                                total_bulge,
                                                mathcore::v3_scale(in_dir_cur,
                                                    P.outBulgeWeight * base * w_out));
                                        }

                                        // ---- side：沿 side_normal（左右方向）----
                                        // side_normal = normalize(cross(prev_axis, next_axis))
                                        // cos_side_axis = dot(radial_cur, side_normal) / |radial_cur|
                                        //   → 判断顶点在左（+n 侧）还是右（-n 侧）
                                        // w_side = smoothstep(|cos_side_axis|)：中线处归零，侧面满权重
                                        // side_dir_out = sign(cos_side_axis) * side_normal
                                        //   → 每个顶点朝自己的左右外侧
                                        // 正值左右同时向外，负值左右同时向内
                                        if (hasSideBulge && radial_len > 1e-9) {
                                            mathcore::Vec3 side_normal = mathcore::v3_cross(
                                                bone_axis_prev_cur, bone_axis_next_cur);
                                            const double sn_len =
                                                mathcore::v3_len(side_normal);
                                            if (sn_len > 1e-6) {
                                                side_normal = mathcore::v3_scale(
                                                    side_normal, 1.0 / sn_len);

                                                const double cos_side_axis =
                                                    mathcore::v3_dot(radial_cur,
                                                        side_normal) / radial_len;

                                                const double abs_cos = std::abs(cos_side_axis);
                                                const double w_side =
                                                    abs_cos * abs_cos * (3.0 - 2.0 * abs_cos);

                                                if (w_side > 1e-9 && abs_cos > 1e-9) {
                                                    const mathcore::Vec3 side_dir_out =
                                                        mathcore::v3_scale(side_normal,
                                                            (cos_side_axis > 0.0) ? 1.0 : -1.0);
                                                    total_bulge = mathcore::v3_add(
                                                        total_bulge,
                                                        mathcore::v3_scale(side_dir_out,
                                                            P.sideBulgeWeight * base * w_side));
                                                }
                                            }
                                        }
                                    }

                                    curve_mos_final = mathcore::v3_add(
                                        curve_mos_final, total_bulge);
                                }

                                // ==================== Slide 段 ====================
                                if (hasAnySlide) {
                                    mathcore::Vec3 total_slide = { 0.0, 0.0, 0.0 };

                                    if (in_len > 1e-9 && pc_o_len > 1e-9 &&
                                        radial_len > 1e-9) {

                                        // 分区：内/外侧（与 Bulge 一致）
                                        const double cos_side_slide =
                                            mathcore::v3_dot(pc.O, in_dir_cur) / pc_o_len;
                                        double t_side_slide = (cos_side_slide + 0.5) / 1.0;
                                        if (t_side_slide < 0.0) t_side_slide = 0.0;
                                        if (t_side_slide > 1.0) t_side_slide = 1.0;
                                        const double smooth_side_slide =
                                            t_side_slide * t_side_slide * (3.0 - 2.0 * t_side_slide);
                                        const double w_in_slide = 1.0 - smooth_side_slide;
                                        const double w_out_slide = smooth_side_slide;

                                        // 分段：上/下段（tau 与 tau_bend 距离）
                                        const double tau_dist_slide =
                                            pc.tau - curve_tau_bend[c];
                                        double t_prev = -tau_dist_slide / 0.25;
                                        if (t_prev < 0.0) t_prev = 0.0;
                                        if (t_prev > 1.0) t_prev = 1.0;
                                        const double w_prev_slide =
                                            t_prev * t_prev * (3.0 - 2.0 * t_prev);
                                        double t_next = tau_dist_slide / 0.25;
                                        if (t_next < 0.0) t_next = 0.0;
                                        if (t_next > 1.0) t_next = 1.0;
                                        const double w_next_slide =
                                            t_next * t_next * (3.0 - 2.0 * t_next);

                                        const mathcore::Vec3 radial_dir_slide =
                                            mathcore::v3_scale(radial_cur, 1.0 / radial_len);

                                        // 1. 内侧轴向滑动（朝关节）
                                        if (w_in_slide > 1e-9) {
                                            if (std::abs(P.slideInWeight_prev) > 1e-9 &&
                                                w_prev_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(bone_axis_prev_cur,
                                                        P.slideInWeight_prev * base *
                                                        w_in_slide * w_prev_slide));
                                            }
                                            if (std::abs(P.slideInWeight_next) > 1e-9 &&
                                                w_next_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(bone_axis_next_cur,
                                                        -P.slideInWeight_next * base *
                                                        w_in_slide * w_next_slide));
                                            }
                                        }

                                        // 2. 外侧轴向滑动（远离关节）
                                        if (w_out_slide > 1e-9) {
                                            if (std::abs(P.slideOutWeight_prev) > 1e-9 &&
                                                w_prev_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(bone_axis_prev_cur,
                                                        -P.slideOutWeight_prev * base *
                                                        w_out_slide * w_prev_slide));
                                            }
                                            if (std::abs(P.slideOutWeight_next) > 1e-9 &&
                                                w_next_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(bone_axis_next_cur,
                                                        P.slideOutWeight_next * base *
                                                        w_out_slide * w_next_slide));
                                            }
                                        }

                                        // 3. 内侧径向鼓起（沿 -radial_dir）
                                        if (w_in_slide > 1e-9) {
                                            if (std::abs(P.slideOrthoInWeight_prev) > 1e-9 &&
                                                w_prev_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(radial_dir_slide,
                                                        -P.slideOrthoInWeight_prev * base *
                                                        w_in_slide * w_prev_slide));
                                            }
                                            if (std::abs(P.slideOrthoInWeight_next) > 1e-9 &&
                                                w_next_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(radial_dir_slide,
                                                        -P.slideOrthoInWeight_next * base *
                                                        w_in_slide * w_next_slide));
                                            }
                                        }

                                        // 4. 外侧径向鼓起（沿 +radial_dir）
                                        if (w_out_slide > 1e-9) {
                                            if (std::abs(P.slideOrthoOutWeight_prev) > 1e-9 &&
                                                w_prev_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(radial_dir_slide,
                                                        P.slideOrthoOutWeight_prev * base *
                                                        w_out_slide * w_prev_slide));
                                            }
                                            if (std::abs(P.slideOrthoOutWeight_next) > 1e-9 &&
                                                w_next_slide > 1e-9) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(radial_dir_slide,
                                                        P.slideOrthoOutWeight_next * base *
                                                        w_out_slide * w_next_slide));
                                            }
                                        }

                                        // 5. CenterSlide
                                        if (hasCenterSlide) {
                                            const mathcore::Vec3 rel_v =
                                                mathcore::v3_sub(origPos, bc_cur);
                                            const double side_v =
                                                mathcore::v3_dot(rel_v, in_dir_cur);

                                            if (side_v > 0.0) {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(in_dir_cur,
                                                        P.centerSlideInWeight * base));
                                            }
                                            else {
                                                total_slide = mathcore::v3_add(
                                                    total_slide,
                                                    mathcore::v3_scale(in_dir_cur,
                                                        -P.centerSlideOutWeight * base));
                                            }
                                        }
                                    }

                                    curve_mos_final = mathcore::v3_add(
                                        curve_mos_final, total_slide);
                                }
                            }
                        }
                    }



                    blended = mathcore::v3_add(blended,
                        mathcore::v3_scale(curve_mos_final, pc.w_c));
                    w_total += pc.w_c;
                }

                if (w_total > 1e-6) {
                    const mathcore::Vec3 mos =
                        mathcore::v3_scale(blended, 1.0 / w_total);
                    double w = w_total * P.influenceRadius;
                    if (w > 1.0) w = 1.0;

                    // ★ 修复 W：w 用 gamma 曲线抬升过渡区
                    //
                    //   根因：视觉补偿量 = w × (mos - lbs)，是两个衰减的乘积：
                    //     · w 的衰减：2*min(w_parent,w_child)，线性
                    //     · mos - lbs 的衰减：曲线变形集中在关节附近，钟形
                    //   两者相乘 → 中线外补偿量急剧下降 → 视觉 W。
                    //
                    //   修复：让 w 的衰减变缓（sqrt / pow），补偿 mos-lbs 的
                    //   自然衰减。中线和边界严格不变（pow(1,p)=1, pow(0,p)=0），
                    //   范围严格不变。
                    //
                    //   power 选择：
                    //     1.0（关闭）：原始线性，W 明显
                    //     0.7        ：轻度抬升
                    //     0.5（推荐）：中度抬升，视觉 W 基本消失
                    //     0.4        ：较激进，两侧几乎与中线持平
                    //     0.33       ：非常激进，可能在边界处出现轻微台阶
                    const double w_curve_power = 0.5;
                    w = std::pow(w, w_curve_power);

                    def = mathcore::v3_add(
                        mathcore::v3_scale(lbs_result, 1.0 - w),
                        mathcore::v3_scale(mos, w));
                }
            }

            // Bind 精确性检查
            if (is_bind_pose) {
                const double dx = def.x - origPos.x;
                const double dy = def.y - origPos.y;
                const double dz = def.z - origPos.z;
                const double d = std::sqrt(dx * dx + dy * dy + dz * dz);
#pragma omp critical(mosbindcheck)
                {
                    if (d > g_bind_max_diff) {
                        g_bind_max_diff = d;
                        g_bind_max_idx = vi;
                    }
                }
            }

            const mathcore::Vec3 bl = applyEnvelope(origPos, def, env);
            MPoint out(bl.x, bl.y, bl.z);
            out *= wInv;
            points[vi] = out;
        }

#pragma omp critical
        {
            g_cvdisp_sum += l_cvdisp_sum;
            g_cvdisp_count += l_cvdisp_count;
            if (l_cvdisp_max > g_cvdisp_max) g_cvdisp_max = l_cvdisp_max;
        }
    }

    iter.setAllPositions(points);

    // 只有弯曲角变化时才打印，避免每帧刷屏
    {
        static std::vector<double> s_last_angles;
        static unsigned int s_frame_count = 0;
        bool changed = (s_last_angles.size() != (size_t)n_curves);
        if (!changed) {
            for (int c = 0; c < n_curves; ++c) {
                if (std::abs(s_last_angles[c] - curve_bend_angle[c]) > 0.005) {
                    changed = true;
                    break;
                }
            }
        }
        // 或者每 60 帧强制打印一次，避免长时间静默
        const bool periodic = (s_frame_count++ % 60 == 0);
        if ((changed || periodic) && (optSkinDebugLog() || P.debugLog)) {
            char buf[256];
            for (int c = 0; c < n_curves; ++c) {
                snprintf(buf, sizeof(buf),
                    "[MOS] curve[%d]: bend_idx=%d tau_bend=%.4f angle=%.3f",
                    c, curve_bend_idx[c],
                    curve_tau_bend[c], curve_bend_angle[c]);
                MGlobal::displayInfo(MString(buf));
            }
            s_last_angles.resize(n_curves);
            for (int c = 0; c < n_curves; ++c) s_last_angles[c] = curve_bend_angle[c];
        }
    }

    if (is_bind_pose && (optSkinDebugLog() || P.debugLog)) {
        char buf[320];
        const char* verdict = (g_bind_max_diff < 1e-9) ? "PASS"
            : (g_bind_max_diff < 1e-6) ? "WARN"
            : "FAIL";
        snprintf(buf, sizeof(buf),
            "[MOS] bind-exact: %s  maxDiff=%.3e at vtx %d  (threshold 1e-9)",
            verdict, g_bind_max_diff, g_bind_max_idx);
        MGlobal::displayInfo(MString(buf));
    }

    {
        static double s_last_max_angle = -1.0;
        static unsigned int s_frame_count2 = 0;
        double max_angle = 0.0;
        for (int c = 0; c < n_curves; ++c)
            if (curve_bend_angle[c] > max_angle) max_angle = curve_bend_angle[c];
        const bool changed = (std::abs(s_last_max_angle - max_angle) > 0.005);
        const bool periodic = (s_frame_count2++ % 120 == 0);

        if ((changed || periodic) && (optSkinDebugLog() || P.debugLog)) {
            char buf[320];
            snprintf(buf, sizeof(buf),
                "[MOS] deform: nV=%u curves=%d mode=%d  cvDisp: count=%d avg=%.5f max=%.5f (Bw=%.2f, IBw=%.2f, SBw=%.2f, IR=%.2f)",
                nV, n_curves, P.curveMode,
                g_cvdisp_count, g_cvdisp_sum / std::max(1, g_cvdisp_count), g_cvdisp_max,
                P.inBulgeWeight, P.outBulgeWeight, P.sideBulgeWeight,
                P.influenceRadius);
            MGlobal::displayInfo(MString(buf));

            if (is_bind_pose) {
                const char* verdict = (g_bind_max_diff < 1e-9) ? "PASS"
                    : (g_bind_max_diff < 1e-6) ? "WARN"
                    : "FAIL";
                snprintf(buf, sizeof(buf),
                    "[MOS] bind-exact: %s  maxDiff=%.3e at vtx %d  (threshold 1e-9)",
                    verdict, g_bind_max_diff, g_bind_max_idx);
                MGlobal::displayInfo(MString(buf));
            }

            s_last_max_angle = max_angle;
        }
    }

    return MS::kSuccess;
}

/// ================================================================
// MOS 控制点级次级操作（专利 US9418465B2）
// 关键原则：
//   1. bulge / slide 都修改控制点 p_j，不修改 B-spline 输出
//   2. 用 cv_bend_weight[j] 把有效修改限制在关节附近的控制点，
//      远端控制点 w_j ≈ 0，不参与修改，root 段完全不受影响
//   3. B-spline 局部支撑 + w_j 形成"双重权重"，修改经求值后进一步平滑
//   4. @deprecated 控制点在骨骼中心线上 side≈0，centerSlideIn/Out 永远不生效，
//      已改为 deformMOS 内顶点级实现。函数保留但不再被调用，未来可整体删除。
// ================================================================

mathcore::Vec3 OptimizedSkinCluster::applyMosSecondaryOps(
    const MosCurve& cv, int j, int bend_idx,
    const std::vector<double>& cv_bend_weight,
    const MosPerCurveBind& /*pc*/,
    const std::vector<mathcore::RigidTransform>& rest,
    const std::vector<mathcore::RigidTransform>& Cs,
    double avg_bone_length,
    const MosParams& P,
    const mathcore::Vec3& p_j) const {

    if (bend_idx < 1 || bend_idx >= (int)cv.chain_bones.size())
        return p_j;

    const int bj = bend_idx;
    const int bj_prev = bj - 1;
    const int bj_next = (bj + 1 < (int)cv.chain_bones.size()) ? bj + 1 : bj;

    const unsigned int bp = cv.chain_bones[bj_prev];
    const unsigned int bc = cv.chain_bones[bj];
    const unsigned int bn = cv.chain_bones[bj_next];

    if (bp >= rest.size() || bc >= rest.size() || bn >= rest.size() ||
        bp >= Cs.size() || bc >= Cs.size() || bn >= Cs.size())
        return p_j;

    const mathcore::Vec3 jp = mathcore::apply_transform(Cs[bp], rest[bp].t);
    const mathcore::Vec3 jc = mathcore::apply_transform(Cs[bc], rest[bc].t);
    const mathcore::Vec3 jn = mathcore::apply_transform(Cs[bn], rest[bn].t);

    const mathcore::Vec3 d_prev = mathcore::v3_sub(jc, jp);
    const mathcore::Vec3 d_next = mathcore::v3_sub(jn, jc);
    if (mathcore::v3_len(d_prev) < 1e-9 || mathcore::v3_len(d_next) < 1e-9)
        return p_j;

    const mathcore::Vec3 dn_prev = mathcore::v3_norm(d_prev);
    const mathcore::Vec3 dn_next = mathcore::v3_norm(d_next);

    // in_dir = -normalize(dir_prev + dir_next)，指向弯折内侧
    mathcore::Vec3 in_dir = { 0.0, 0.0, 0.0 };
    {
        mathcore::Vec3 b = mathcore::v3_sub(dn_next, dn_prev);
        if (mathcore::v3_len(b) > 1e-6)
            in_dir = mathcore::v3_norm(b);
    }
    if (mathcore::v3_len(in_dir) < 1e-9)
        return p_j;

    // bend = 两段骨骼方向夹角 / π，∈ [0,1]；不含 twist
    double cos_bend = mathcore::v3_dot(dn_prev, dn_next);
    if (cos_bend > 1.0) cos_bend = 1.0;
    if (cos_bend < -1.0) cos_bend = -1.0;
    const double bend = std::acos(cos_bend) / 3.14159265358979323846;

    const double L_bone = avg_bone_length;

    // ★ 该控制点对弯曲关节的从属权重（0~1）
    //   远端控制点 w_j ≈ 0，直接跳过
    const double w_j = (j >= 0 && j < (int)cv_bend_weight.size())
        ? cv_bend_weight[j] : 0.0;
    if (w_j < 1e-6) return p_j;

    mathcore::Vec3 corr = { 0.0, 0.0, 0.0 };

    // ---- centerSlide：沿内侧/外侧方向偏移控制点 ----
    // 保留原有 side 判断：控制点在中心线上 side ≈ 0，
    // centerSlide 对中心线控制点不生效（rig 手动编辑控制点后才触发），
    // 符合专利"rig 可额外调整"的语义。
    const mathcore::Vec3 rel = mathcore::v3_sub(p_j, jc);
    const double side = mathcore::v3_dot(rel, in_dir);

    const double R = P.influenceRadius * L_bone;
    double t_in = 0.0, t_out = 0.0;
    if (R > 1e-9) {
        if (side > 0.0) {
            double x = side / R; if (x > 1.0) x = 1.0;
            t_in = x * x * (3.0 - 2.0 * x);
        }
        else {
            double x = -side / R; if (x > 1.0) x = 1.0;
            t_out = x * x * (3.0 - 2.0 * x);
        }
    }

    if (std::abs(P.centerSlideInWeight) > 1e-9 && t_in > 0.0) {
        const double amount = P.centerSlideInWeight * L_bone * bend * w_j * t_in;
        corr = mathcore::v3_add(corr,
            mathcore::v3_scale(in_dir, amount));
    }
    if (std::abs(P.centerSlideOutWeight) > 1e-9 && t_out > 0.0) {
        const double amount = P.centerSlideOutWeight * L_bone * bend * w_j * t_out;
        corr = mathcore::v3_add(corr,
            mathcore::v3_scale(in_dir, -amount));
    }

    return mathcore::v3_add(p_j, corr);
}

// ================================================================
// getCoRData
// ================================================================
bool OptimizedSkinCluster::getCoRData(
    std::vector<mathcore::Vec3>& out) const {
    if (!m_corCache.valid) return false;
    out = m_corCache.cors;
    return true;
}