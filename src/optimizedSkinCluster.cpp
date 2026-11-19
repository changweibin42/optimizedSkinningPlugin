#include "optimizedSkinCluster.h"
#include "algorithms.h"
#include "cor_contour.h"
#include "dlb_gpu.h"
#include "node_ids.h"
#include "plugin_debug.h"

#include <maya/MEvaluationNode.h>
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
#include <cstring>
#include <string>
#include <set>

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
MObject OptimizedSkinCluster::aDlbUseOpenCL;
MObject OptimizedSkinCluster::aDQSCompStrength;
MObject OptimizedSkinCluster::aForceFullCoR;
MObject OptimizedSkinCluster::aCorTriLayerExpand;
MObject OptimizedSkinCluster::aCorVertLayerExpand;
MObject OptimizedSkinCluster::aCorMosSecondaryWeight;
MObject OptimizedSkinCluster::aCorCompStrength;   // 新增
MObject OptimizedSkinCluster::aCorCompFalloff;    // 新增
MObject OptimizedSkinCluster::aCorCompStrengthIn;   // 新增
MObject OptimizedSkinCluster::aCorCompStrengthOut;  // 新增
MObject OptimizedSkinCluster::aCorSigma;
MObject OptimizedSkinCluster::aCorDebugLog;
MObject OptimizedSkinCluster::aCorContourIn;
MObject OptimizedSkinCluster::aCorContourOut;
MObject OptimizedSkinCluster::aDdmIterations;
MObject OptimizedSkinCluster::aDdmLambda;
MObject OptimizedSkinCluster::aDdmKappa;
MObject OptimizedSkinCluster::aDdmAlpha;
MObject OptimizedSkinCluster::aDdmDebugLog;
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

// [DIAG] 诊断开关属性
MObject OptimizedSkinCluster::aDiagFlags;

OptimizedSkinCluster::OptimizedSkinCluster() : m_dlbGpu(std::make_unique<DlbGpu>()) {}
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
    eAttr.addField("CoR + MOS Secondary", 4);
    eAttr.addField("CoR + Compensation", 5);
    eAttr.addField("DDM v4 (2019)", 6);
    eAttr.addField("CoR + Contour", 7);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    addAttribute(aAlgorithmType);

    aDlbUseOpenCL = nAttr.create("dlbUseOpenCL", "dlbcl", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("DLB Use OpenCL");
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aDlbUseOpenCL));
    CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aDlbUseOpenCL, MPxSkinCluster::outputGeom));

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

    aCorMosSecondaryWeight = nAttr.create("corMosSecondaryWeight", "cmsc",
        MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);    
    nAttr.setMax(2.0);      
    nAttr.setSoftMin(-1.0);    
    nAttr.setSoftMax(1.0);     
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR+MOS Secondary Weight");
    addAttribute(aCorMosSecondaryWeight);
    attributeAffects(aCorMosSecondaryWeight, MPxSkinCluster::outputGeom);

    aCorCompStrength = nAttr.create("corCompStrength", "ccs",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-1.0);
    nAttr.setMax(3.0);
    nAttr.setSoftMin(0.0);
    nAttr.setSoftMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Compensation Strength");
    addAttribute(aCorCompStrength);
    attributeAffects(aCorCompStrength, MPxSkinCluster::outputGeom);

    aCorCompFalloff = nAttr.create("corCompFalloff", "ccf",
        MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.25);
    nAttr.setMax(4.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Compensation Falloff");
    addAttribute(aCorCompFalloff);
    attributeAffects(aCorCompFalloff, MPxSkinCluster::outputGeom);

    aCorCompStrengthIn = nAttr.create("corCompStrengthIn", "ccsi",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);
    nAttr.setMax(3.0);
    nAttr.setSoftMin(-1.0);
    nAttr.setSoftMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Comp Strength (Inner)");
    addAttribute(aCorCompStrengthIn);
    attributeAffects(aCorCompStrengthIn, MPxSkinCluster::outputGeom);

    aCorCompStrengthOut = nAttr.create("corCompStrengthOut", "ccso",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(-2.0);
    nAttr.setMax(3.0);
    nAttr.setSoftMin(-1.0);
    nAttr.setSoftMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Comp Strength (Outer)");
    addAttribute(aCorCompStrengthOut);
    attributeAffects(aCorCompStrengthOut, MPxSkinCluster::outputGeom);

    aCorSigma = nAttr.create("corSigma", "csm",
        MFnNumericData::kDouble, 0.1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.02);
    nAttr.setMax(0.5);
    nAttr.setSoftMin(0.05);
    nAttr.setSoftMax(0.2);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Sigma (similarity width)");
    addAttribute(aCorSigma);
    attributeAffects(aCorSigma, MPxSkinCluster::outputGeom);

    aCorDebugLog = nAttr.create("corDebugLog", "crdl", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setStorable(false);
    nAttr.setKeyable(false);
    nAttr.setNiceNameOverride("CoR Debug Log");
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aCorDebugLog));
    CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aCorDebugLog, MPxSkinCluster::outputGeom));

    const auto addContourControl = [&](MObject& attr, const char* name, const char* shortName,
        const char* label) -> MStatus {
        MStatus result;
        attr = nAttr.create(name, shortName, MFnNumericData::kDouble, 1.0, &result);
        if (!result) return result;
        nAttr.setMin(0.0);
        nAttr.setMax(2.0);
        nAttr.setSoftMax(1.0);
        nAttr.setKeyable(true);
        nAttr.setStorable(true);
        nAttr.setNiceNameOverride(label);
        result = addAttribute(attr);
        if (!result) return result;
        return attributeAffects(attr, MPxSkinCluster::outputGeom);
    };
    CHECK_MSTATUS_AND_RETURN_IT(addContourControl(aCorContourIn, "corContourIn", "ccin", "CoR Contour Inner"));
    CHECK_MSTATUS_AND_RETURN_IT(addContourControl(aCorContourOut, "corContourOut", "ccout", "CoR Contour Outer"));

    const ddm_v4::Settings ddmDefaults;
    const auto addDdmControl = [&](MObject& attr, const char* name, const char* shortName,
        MFnNumericData::Type type, double value, double maximum) -> MStatus {
        MStatus result;
        attr = nAttr.create(name, shortName, type, value, &result);
        if (!result) return result;
        nAttr.setMin(0.0);
        nAttr.setMax(maximum);
        nAttr.setKeyable(true);
        nAttr.setStorable(true);
        result = addAttribute(attr);
        return result ? attributeAffects(attr, MPxSkinCluster::outputGeom) : result;
    };
    CHECK_MSTATUS_AND_RETURN_IT(addDdmControl(aDdmIterations, "ddmIterations", "ddmi",
        MFnNumericData::kInt, ddmDefaults.iterations, 100.0));
    CHECK_MSTATUS_AND_RETURN_IT(addDdmControl(aDdmLambda, "ddmLambda", "ddml",
        MFnNumericData::kDouble, ddmDefaults.lambda, 100.0));
    CHECK_MSTATUS_AND_RETURN_IT(addDdmControl(aDdmKappa, "ddmKappa", "ddmk",
        MFnNumericData::kDouble, ddmDefaults.kappa, 100.0));
    CHECK_MSTATUS_AND_RETURN_IT(addDdmControl(aDdmAlpha, "ddmAlpha", "ddma",
        MFnNumericData::kDouble, ddmDefaults.alpha, 1.0));

    aDdmDebugLog = nAttr.create("ddmDebugLog", "ddmdl", MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(false);
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aDdmDebugLog));
    CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aDdmDebugLog, MPxSkinCluster::outputGeom));

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
        MFnNumericData::kDouble, 1.0, &status);
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
    attributeAffects(aMosDebugLog, MPxSkinCluster::outputGeom);

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

    // [DIAG] 诊断开关：bit0=FJ_HARD, bit1=W_LINEAR, bit2=WC_UNITY
    // 用 Maya 属性而不是环境变量，允许在同一 Maya session 里切换，不用重启
    aDiagFlags = nAttr.create("diagFlags", "dfg",
        MFnNumericData::kInt, 0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0);
    nAttr.setMax(7);
    nAttr.setStorable(true);
    nAttr.setKeyable(false);
    nAttr.setConnectable(false);
    nAttr.setChannelBox(false);
    nAttr.setNiceNameOverride("Diag Flags (0=none,1=FJ,2=W,4=WC)");
    addAttribute(aDiagFlags);
    attributeAffects(aDiagFlags, MPxSkinCluster::outputGeom);

    return MS::kSuccess;
}

// ================================================================
// setDependentsDirty
// ================================================================
MStatus OptimizedSkinCluster::setDependentsDirty(
    const MPlug& plug, MPlugArray& /*plugArray*/) {

    MObject attr = plug.attribute();

    if (attr == MPxSkinCluster::input || attr == MPxSkinCluster::inputGeom) {
        m_paperCoR.valid = false;
        m_ddmV4.valid = false;
        if (m_mosCache.direct) m_mosCache.valid = false;
    }

    if (attr == MPxSkinCluster::bindPreMatrix && m_mosCache.direct) {
        m_mosCache.valid = false;
        m_boneInfo.valid = false;
    }

    if (attr == aDiagFlags) {
        // diagFlags 影响 bind 时的 w_c 计算与 deform 时的 F 计算，
        // 直接失效 MOS cache 让下次 deform 全量重建
        m_mosCache.valid = false;
        return MS::kSuccess;
    }

    if (attr == aCorSigma) {
        // σ 影响 CoR 中心的权重相似性核，必须强制全量重算 CoR cache
        m_corCache.valid = false;
        m_paperCoR.valid = false;
        return MS::kSuccess;
    }

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
        attr == aMosScaleMode ||
        attr == aCorMosSecondaryWeight) 
    {
        if (m_mosCache.direct && attr != aMosCurveMode && attr != aMosCurveSamples)
            return MS::kSuccess;
        m_mosCache.valid = false;

        if (optSkinDebugLog()) {
            MFnDependencyNode fn(thisMObject());
            MGlobal::displayInfo(MString("[MOS] cache invalidated by ") + fn.name());
        }
        if (attr == aCorCompStrength || attr == aCorCompFalloff) {
            // 补偿增强不需要重算 CoR cache（CoR 结果与 Strength 无关）
            // 但因为 outputGeom 已 attributeAffects，Maya 会重跑 deform
            return MS::kSuccess;
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

MStatus OptimizedSkinCluster::preEvaluation(const MDGContext& context,
    const MEvaluationNode& evaluationNode) {
    if (evaluationNode.dirtyPlugExists(MPxSkinCluster::weightList) ||
        evaluationNode.dirtyPlugExists(MPxSkinCluster::weights)) {
        m_vtxCache.valid = false;
        if (m_mosCache.direct) m_mosCache.valid = false;
        ++m_currentWeightVersion;
    }
    if (evaluationNode.dirtyPlugExists(MPxSkinCluster::inputGeom)) {
        m_paperCoR.valid = false;
        m_ddmV4.valid = false;
        if (m_mosCache.direct) m_mosCache.valid = false;
    }
    return MPxSkinCluster::preEvaluation(context, evaluationNode);
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

    // [新增] 缓存每个 joint 的"唯一" child（用于 deform 时算当前 pose 的内角方向）
    // 多 child 或无 child 时保持 -1，deform 时 in/out 分区会回退到 0.5/0.5
    m_boneInfo.boneSoleChild.assign(maxIdx + 1, -1);
    for (unsigned int i = 0; i < nB; ++i) {
        const unsigned int idx = logicalIdx[i];
        MFnDagNode dag(joints[i]);
        MObject soleChild;
        int childCount = 0;
        unsigned int nChild = dag.childCount();
        for (unsigned int c = 0; c < nChild; ++c) {
            MObject obj = dag.child(c);
            if (!obj.hasFn(MFn::kJoint)) continue;
            for (unsigned int j = 0; j < nB; ++j) {
                if (joints[j] == obj) {
                    if (childCount == 0) soleChild = obj;
                    ++childCount;
                    break;
                }
            }
            if (childCount > 1) break;
        }
        if (childCount == 1) {
            for (unsigned int j = 0; j < nB; ++j) {
                if (joints[j] == soleChild) {
                    m_boneInfo.boneSoleChild[idx] = (int)logicalIdx[j];
                    break;
                }
            }
        }
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

    // 读 σ 属性（必须放在 omp 循环外，MDataBlock 不是线程安全的）
    double sigmaVal = 0.1;
    {
        MStatus sigmaSt;
        MDataHandle h = block.inputValue(aCorSigma, &sigmaSt);
        if (sigmaSt) sigmaVal = h.asDouble();
    }

    m_corCache.cors.resize(numVertices);
#pragma omp parallel for schedule(dynamic, 64)
    for (int ii = 0; ii < (int)numVertices; ++ii) {
        unsigned int i = (unsigned int)ii;
        m_corCache.cors[i] = computeSingleVertexCoR(
            m_corCache.allW[i], points, i,
            m_corCache.triSubdivs, m_corCache.triAdj, m_corCache.vertToTris,
            K, m_corCache.clusterMembers, m_corCache.vertClusterDists,
            m_corCache.allW, numTris, sigmaVal);   // ★ 传 σ
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

    // 读 σ（与 cache 构建时一致；若已改过 σ，m_corCache.valid 已被 setDependentsDirty 置 false，走不到这里）
    double sigmaVal = 0.1;
    {
        MStatus sigmaSt;
        MDataHandle h = block.inputValue(aCorSigma, &sigmaSt);
        if (sigmaSt) sigmaVal = h.asDouble();
    }

#pragma omp parallel for schedule(dynamic, 8)
    for (int ii = 0; ii < (int)vAffected.size(); ++ii) {
        unsigned int v = vAffected[ii];
        m_corCache.cors[v] = computeSingleVertexCoR(
            m_corCache.allW[v], points, v,
            m_corCache.triSubdivs, m_corCache.triAdj, m_corCache.vertToTris,
            m_corCache.K, m_corCache.clusterMembers, m_corCache.vertClusterDists,
            m_corCache.allW, numTris, sigmaVal);   // ★ 传 σ
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

    if (algo == 0) return deformCoR(block, iter, worldMat, multiIndex);
    if (algo == 6) return deformCoR(block, iter, worldMat, multiIndex, CenteredMode::DdmV4);
    if (algo == 7) return deformCoR(block, iter, worldMat, multiIndex, CenteredMode::Contour);

    unsigned int nV = iter.count();
    collectBoneInfo(block);
    rebuildVtxCache(block, nV);

    switch (algo) {
    case 1: return deformDLB(block, iter, worldMat, multiIndex,
        block.inputValue(aDlbUseOpenCL).asBool());
    case 2: return deformDQSComp(block, iter, worldMat, multiIndex);
    case 3: return deformMOS(block, iter, worldMat, multiIndex);
    case 4: return deformCoRMosSecondary(block, iter, worldMat, multiIndex);
    case 5: return deformCoRComp(block, iter, worldMat, multiIndex);
    default: return MS::kFailure;
    }
}

// ================================================================
// deformCoR
// ================================================================
MStatus OptimizedSkinCluster::rebuildCenteredSkinning(
    MDataBlock& block, unsigned int geometryIndex, double sigma, const ddm_v4::Settings* settings,
    CenteredSkinningCache& cache, const char*& stage) {
    MStatus status;
    stage = "input geometry";
    auto geometry = block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return status;
    status = geometry.jumpToElement(geometryIndex);
    if (!status) return status;
    auto input = geometry.inputValue(&status);
    if (!status) return status;
    const auto meshObject = input.child(MPxSkinCluster::inputGeom).asMesh();
    MFnMesh mesh(meshObject, &status);
    if (!status) return status;

    stage = "rest mesh points";
    MPointArray positions;
    status = mesh.getPoints(positions, MSpace::kObject);
    if (!status) return status;
    std::vector<mathcore::Vec3> restPoints;
    restPoints.reserve(positions.length());
    for (unsigned int i = 0; i < positions.length(); ++i) {
        const auto& p = positions[i];
        restPoints.push_back({p.x, p.y, p.z});
    }

    stage = "vertex weights";
    auto weightList = block.inputArrayValue(MPxSkinCluster::weightList, &status);
    if (!status) return status;

    CenteredSkinningCache next;
    next.weights.resize(restPoints.size());
    std::set<unsigned int> influenceIds;
    const unsigned int rows = weightList.elementCount(&status);
    if (!status) return status;
    for (unsigned int row = 0; row < rows; ++row) {
        status = weightList.jumpToArrayElement(row);
        if (!status) return status;
        const unsigned int vertexId = weightList.elementIndex(&status);
        if (!status) return status;
        if (vertexId >= next.weights.size()) continue;
        auto element = weightList.inputValue(&status);
        if (!status) return status;
        MArrayDataHandle influences(element.child(MPxSkinCluster::weights));
        auto& vertex = next.weights[vertexId];
        const unsigned int count = influences.elementCount();
        for (unsigned int j = 0; j < count; ++j) {
            status = influences.jumpToArrayElement(j);
            if (!status) return status;
            const unsigned int id = influences.elementIndex(&status);
            if (!status) return status;
            auto value = influences.inputValue(&status);
            if (!status) return status;
            const double weight = value.asDouble();
            if (weight > 0.0 && std::isfinite(weight))
                vertex.emplace_back(id, weight);
        }
        std::sort(vertex.begin(), vertex.end());
        double total = 0.0;
        for (const auto& weight : vertex) total += weight.second;
        if (!std::isfinite(total)) return MS::kFailure;
        for (auto& weight : vertex) {
            weight.second /= total;
            influenceIds.insert(weight.first);
        }
    }

    next.influenceIds.assign(influenceIds.begin(), influenceIds.end());
    for (auto& vertex : next.weights) {
        for (auto& weight : vertex) {
            weight.first = static_cast<unsigned int>(std::lower_bound(next.influenceIds.begin(),
                next.influenceIds.end(), weight.first) - next.influenceIds.begin());
        }
    }

    if (settings) {
        stage = "rest mesh edges";
        const int edgeCount = mesh.numEdges(&status);
        if (!status) return status;
        std::vector<ddm_v4::Edge> edges;
        edges.reserve(edgeCount);
        for (int edge = 0; edge < edgeCount; ++edge) {
            int2 vertices;
            status = mesh.getEdgeVertices(edge, vertices);
            if (!status) return status;
            edges.push_back({static_cast<unsigned int>(vertices[0]),
                static_cast<unsigned int>(vertices[1])});
        }

        stage = "DDM implicit bind diffusion";
        ddm_v4::BindData data;
        if (!ddm_v4::precompute(restPoints, next.weights, next.influenceIds.size(), edges,
            *settings, data)) return MS::kFailure;
        next.centers = std::move(data.centers);
        next.weights = std::move(data.rotationWeights);
        next.translationWeights = std::move(data.translationWeights);
        next.ddmSettings = *settings;
    }
    else {
        stage = "rest mesh triangles";
        MIntArray triangleCounts, indices;
        status = mesh.getTriangles(triangleCounts, indices);
        if (!status) return status;
        std::vector<cor_paper::Triangle> triangles;
        triangles.reserve(indices.length() / 3);
        for (unsigned int i = 0; i + 2 < indices.length(); i += 3) {
            cor_paper::Triangle triangle = {static_cast<unsigned int>(indices[i]),
                static_cast<unsigned int>(indices[i + 1]), static_cast<unsigned int>(indices[i + 2])};
            if (triangle[0] >= restPoints.size() || triangle[1] >= restPoints.size() ||
                triangle[2] >= restPoints.size()) return MS::kFailure;
            triangles.push_back(triangle);
        }
        next.centers = cor_paper::computeCenters(restPoints, next.weights, triangles, sigma);
    }
    next.geometryIndex = geometryIndex;
    next.weightVersion = m_currentWeightVersion;
    next.sigma = sigma;
    next.valid = true;
    cache = std::move(next);
    return MS::kSuccess;
}

MStatus OptimizedSkinCluster::deformCoR(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex, CenteredMode mode) {
    const bool useDdm = mode == CenteredMode::DdmV4;
    const bool useContour = mode == CenteredMode::Contour;
    const bool debug = optSkinDebugLog() ||
        block.inputValue(useDdm ? aDdmDebugLog : aCorDebugLog).asBool();
    const MString prefix = debug
        ? MString(useDdm ? "[DDM v4] " : useContour ? "[CoR Contour] " : "[CoR] ") + MFnDependencyNode(thisMObject()).name() + ": "
        : MString();
    const auto reportFailure = [&](MStatus status, const char* stage) {
        if (!status && debug) MGlobal::displayWarning(prefix + stage + " - " + status.errorString());
        return status;
    };

    const double envelope = readEnvelopeFromNode();
    MStatus status;
    double innerStrength = 0.0, outerStrength = 0.0;
    if (useContour) {
        innerStrength = block.inputValue(aCorContourIn, &status).asDouble();
        if (!status) return reportFailure(status, "read corContourIn");
        outerStrength = block.inputValue(aCorContourOut, &status).asDouble();
        if (!status) return reportFailure(status, "read corContourOut");
    }

    double sigma = 0.0;
    bool forceFull = false;
    ddm_v4::Settings ddmSettings;
    if (useDdm) {
        ddmSettings.iterations = block.inputValue(aDdmIterations, &status).asInt();
        if (!status) return reportFailure(status, "read ddmIterations");
        ddmSettings.lambda = block.inputValue(aDdmLambda, &status).asDouble();
        if (!status) return reportFailure(status, "read ddmLambda");
        ddmSettings.kappa = block.inputValue(aDdmKappa, &status).asDouble();
        if (!status) return reportFailure(status, "read ddmKappa");
        ddmSettings.alpha = block.inputValue(aDdmAlpha, &status).asDouble();
        if (!status) return reportFailure(status, "read ddmAlpha");
    }
    else {
        sigma = block.inputValue(aCorSigma, &status).asDouble();
        if (!status) return reportFailure(status, "read corSigma");
        forceFull = block.inputValue(aForceFullCoR, &status).asBool();
        if (!status) return reportFailure(status, "read forceFullCoR");
    }

    auto& cache = useDdm ? m_ddmV4 : m_paperCoR;
    if (forceFull || !cache.valid || cache.geometryIndex != multiIndex ||
        cache.weightVersion != m_currentWeightVersion ||
        (useDdm ? !(cache.ddmSettings == ddmSettings) : cache.sigma != sigma)) {
        cache.valid = false;
        const char* stage = "bind cache";
        const auto start = std::chrono::steady_clock::now();
        status = rebuildCenteredSkinning(block, multiIndex, sigma, useDdm ? &ddmSettings : nullptr,
            cache, stage);
        if (!status) return reportFailure(status, stage);
        if (debug) {
            const auto weighted = std::count_if(cache.weights.begin(), cache.weights.end(),
                [](const auto& weights) { return !weights.empty(); });
            const double bindMs = std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - start).count();
            char message[200];
            snprintf(message, sizeof(message),
                "bind geometry=%u vertices=%zu weighted=%zu influences=%zu bindMs=%.3f",
                multiIndex, cache.centers.size(), static_cast<size_t>(weighted),
                cache.influenceIds.size(), bindMs);
            const size_t used = std::strlen(message);
            if (useDdm)
                snprintf(message + used, sizeof(message) - used, " p=%d lambda=%.3g kappa=%.3g alpha=%.3g",
                    ddmSettings.iterations, ddmSettings.lambda, ddmSettings.kappa, ddmSettings.alpha);
            else
                snprintf(message + used, sizeof(message) - used, " sigma=%.4f", sigma);
            MGlobal::displayInfo(prefix + message);
        }
    }

    if (envelope <= 1e-6 || cache.influenceIds.empty()) {
        if (debug) MGlobal::displayWarning(prefix + (cache.influenceIds.empty()
            ? "no positive skin weights in the input geometry" : "envelope is zero"));
        return MS::kSuccess;
    }

    const auto loadMatrices = [&](const MObject& attribute, std::vector<MMatrix>& matrices,
        std::vector<bool>& present) -> MStatus {
        MStatus readStatus;
        auto values = block.inputArrayValue(attribute, &readStatus);
        if (!readStatus) return readStatus;
        const unsigned int count = values.elementCount(&readStatus);
        if (!readStatus) return readStatus;
        for (unsigned int entry = 0; entry < count; ++entry) {
            readStatus = values.jumpToArrayElement(entry);
            if (!readStatus) return readStatus;
            const unsigned int id = values.elementIndex(&readStatus);
            if (!readStatus) return readStatus;

            const auto found = std::lower_bound(cache.influenceIds.begin(), cache.influenceIds.end(), id);
            if (found == cache.influenceIds.end() || *found != id) continue;
            auto value = values.inputValue(&readStatus);
            if (!readStatus) return readStatus;
            // SkinCluster stores matrix objects. Read them through the same
            // Maya function set used by the DevKit skinCluster example.
            MFnMatrixData data(value.data(), &readStatus);
            if (!readStatus) return readStatus;
            const size_t slot = found - cache.influenceIds.begin();
            matrices[slot] = data.matrix(&readStatus);
            if (!readStatus) return readStatus;
            present[slot] = true;
        }
        return readStatus;
    };

    const size_t influenceCount = cache.influenceIds.size();
    std::vector<MMatrix> poses(influenceCount), bindInverses(influenceCount);
    std::vector<bool> hasPose(influenceCount), hasBind(influenceCount);
    status = loadMatrices(MPxSkinCluster::matrix, poses, hasPose);
    if (!status) return reportFailure(status, "read matrix");
    status = loadMatrices(MPxSkinCluster::bindPreMatrix, bindInverses, hasBind);
    if (!status) return reportFailure(status, "read bindPreMatrix");

    const mathcore::RigidTransform identity{{0.0, 0.0, 0.0, 1.0}, {0.0, 0.0, 0.0}};
    std::vector<mathcore::RigidTransform> bindToPose(influenceCount, identity);
    size_t posedCount = 0;
    for (size_t i = 0; i < influenceCount; ++i) {
        if (!hasPose[i]) continue;

        // Maya uses row vectors, so inverse bind is applied before pose.
        const MMatrix relative = bindInverses[i] * poses[i];
        const MQuaternion q = MTransformationMatrix(relative).rotation();
        bindToPose[i] = {mathcore::q_normalize({q.x, q.y, q.z, q.w}),
            {relative[3][0], relative[3][1], relative[3][2]}};
        if (debug && !relative.isEquivalent(MMatrix::identity, 1e-8)) ++posedCount;
    }

    struct ContourPair {
        size_t parent = 0;
        cor_contour::BendFrame frame;
    };
    std::vector<ContourPair> contourPairs;
    size_t pairCount = 0, activeBends = 0;
    const cor_contour::BendFrame* isolatedBend = nullptr;
    if (innerStrength > 0.0 || outerStrength > 0.0) {
        const MPlug matrixPlug(thisMObject(), MPxSkinCluster::matrix);
        std::vector<MObject> joints(influenceCount);
        std::vector<mathcore::Vec3> bindPivots(influenceCount);
        for (unsigned int entry = 0; entry < matrixPlug.numConnectedElements(); ++entry) {
            const auto element = matrixPlug.connectionByPhysicalIndex(entry);
            const unsigned int id = element.logicalIndex();
            const auto found = std::lower_bound(cache.influenceIds.begin(), cache.influenceIds.end(), id);
            if (found == cache.influenceIds.end() || *found != id) continue;
            const size_t slot = found - cache.influenceIds.begin();
            if (!hasBind[slot] || !hasPose[slot] || bindInverses[slot].isSingular()) continue;
            MPlugArray sources;
            if (!element.connectedTo(sources, true, false) || sources.length() != 1) continue;
            const auto joint = sources[0].node();
            if (!joint.hasFn(MFn::kJoint)) continue;
            joints[slot] = joint;
            const auto bind = bindInverses[slot].inverse();
            bindPivots[slot] = {bind[3][0], bind[3][1], bind[3][2]};
        }

        contourPairs.resize(influenceCount);
        for (size_t child = 0; child < influenceCount; ++child) {
            if (joints[child].isNull()) continue;
            MFnDagNode ancestor(joints[child]);
            while (ancestor.parentCount() > 0) {
                const auto parentJoint = ancestor.parent(0);
                const auto found = std::find(joints.begin(), joints.end(), parentJoint);
                if (found != joints.end()) {
                    const size_t parent = found - joints.begin();
                    const auto& pose = poses[child];
                    auto frame = cor_contour::buildFrame(bindPivots[parent], bindPivots[child],
                        {pose[3][0], pose[3][1], pose[3][2]}, bindToPose[parent].q, bindToPose[child].q);
                    contourPairs[child] = {parent, frame};
                    ++pairCount;
                    if (frame.activation > 0.0) ++activeBends;
                    break;
                }
                if (!parentJoint.hasFn(MFn::kDagNode) || !ancestor.setObject(parentJoint)) break;
            }
        }
        // A two-influence tube can have overlapping rigid sections even where
        // weights are already 100% on one side. Let the axial taper reach those rings.
        if (influenceCount == 2 && activeBends == 1) {
            for (const auto& pair : contourPairs)
                if (pair.frame.activation > 0.0) isolatedBend = &pair.frame;
        }
    }

    MPointArray points;
    status = iter.allPositions(points);
    if (!status) return reportFailure(status, "read output geometry points");
    bool dense = points.length() == cache.centers.size();
    unsigned int sequentialIndex = 0;
    for (; dense && !iter.isDone(); iter.next())
        dense = iter.index() == static_cast<int>(sequentialIndex++);
    dense = dense && sequentialIndex == points.length();
    status = iter.reset();
    if (!status) return reportFailure(status, "reset geometry iterator");

    const MMatrix worldInverse = worldMat.inverse();
    const auto deformPoint = [&](const MPoint& p, unsigned int vertex) {
        if (vertex >= cache.weights.size() || cache.weights[vertex].empty()) return p;
        const MPoint restWorld = p * worldMat;
        const auto& center = cache.centers[vertex];
        const MPoint centerWorld = MPoint(center.x, center.y, center.z) * worldMat;
        const auto& rotationWeights = cache.weights[vertex];
        const auto& translationWeights = cache.translationWeights.empty()
            ? rotationWeights : cache.translationWeights[vertex];
        const auto transform = cor_paper::vertexTransform(
            {centerWorld.x, centerWorld.y, centerWorld.z}, rotationWeights, translationWeights, bindToPose);
        const mathcore::Vec3 original{restWorld.x, restWorld.y, restWorld.z};
        const auto base = mathcore::apply_transform(transform, original);
        auto posed = base;
        if (isolatedBend) {
            posed = mathcore::v3_add(posed, cor_contour::correction(*isolatedBend, original, base,
                transform.q, 1.0, innerStrength, outerStrength, true));
        }
        else if (activeBends > 0) {
            for (const auto& child : rotationWeights) {
                const auto& pair = contourPairs[child.first];
                if (pair.frame.activation == 0.0) continue;
                const auto parent = std::lower_bound(rotationWeights.begin(), rotationWeights.end(), pair.parent,
                    [](const auto& weight, size_t index) { return weight.first < index; });
                if (parent == rotationWeights.end() || parent->first != pair.parent) continue;
                const double overlap = 4.0 * parent->second * child.second;
                posed = mathcore::v3_add(posed, cor_contour::correction(pair.frame, original, base,
                    transform.q, overlap, innerStrength, outerStrength));
            }
        }
        const auto result = applyEnvelope(original, posed, envelope);
        return MPoint(result.x, result.y, result.z) * worldInverse;
    };

    double maxDeltaSquared = 0.0;
    unsigned int affected = points.length();
    const auto measure = [&](const MPoint& inputPoint, const MPoint& outputPoint) {
        const auto delta = outputPoint - inputPoint;
        maxDeltaSquared = std::max(maxDeltaSquared, delta * delta);
    };
    if (dense) {
        MPointArray original;
        if (debug) original = points;
#pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(points.length()); ++i)
            points[i] = deformPoint(points[i], static_cast<unsigned int>(i));
        if (debug) {
            for (unsigned int i = 0; i < points.length(); ++i) measure(original[i], points[i]);
        }
        status = iter.setAllPositions(points);
        if (!status) return reportFailure(status, "write output geometry points");
    }
    else {
        // Respect the iterator's vertex IDs for a deformer component subset.
        affected = 0;
        for (; !iter.isDone(); iter.next()) {
            const MPoint inputPoint = iter.position();
            const MPoint outputPoint = deformPoint(inputPoint, static_cast<unsigned int>(iter.index()));
            status = iter.setPosition(outputPoint);
            if (!status) return reportFailure(status, "write output component");
            if (debug) measure(inputPoint, outputPoint);
            ++affected;
        }
    }

    if (debug) {
        const auto poseCount = static_cast<size_t>(std::count(hasPose.begin(), hasPose.end(), true));
        const auto bindCount = static_cast<size_t>(std::count(hasBind.begin(), hasBind.end(), true));
        char message[230];
        snprintf(message, sizeof(message),
            "pose matrices=%zu/%zu bind matrices=%zu/%zu nonIdentity=%zu affected=%u envelope=%.3f maxDelta=%.8g",
            poseCount, influenceCount, bindCount, influenceCount, posedCount, affected,
            envelope, std::sqrt(maxDeltaSquared));
        if (useContour) {
            const auto used = std::strlen(message);
            snprintf(message + used, sizeof(message) - used,
                " jointPairs=%zu activeBends=%zu", pairCount, activeBends);
        }
        MGlobal::displayInfo(prefix + message);
    }
    return MS::kSuccess;
}

// ================================================================
// deformDLB
// ================================================================
MStatus OptimizedSkinCluster::deformDLB(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/, bool useOpenCL) {

    double env = readEnvelopeFromNode();
    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    unsigned int nV = iter.count();
    MPointArray points;
    iter.allPositions(points);
    MMatrix wInv = worldMat.inverse();
    const double envFinal = env;

    if (useOpenCL && m_dlbGpu->deform(points, m_vtxCache.vertices, m_vtxCache.weightVersion,
        Cs, worldMat, wInv, envFinal, thisMObject())) {
        return iter.setAllPositions(points);
    }

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
    if (!aCorMosSecondaryWeight.isNull()) {
        MDataHandle h = block.inputValue(aCorMosSecondaryWeight, &st);
        if (st) p.corMosSecondaryWeight = h.asDouble();
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
        deq(corMosSecondaryWeight, rhs.corMosSecondaryWeight) &&   // ★ 新增
        debugLog == rhs.debugLog;

}

bool OptimizedSkinCluster::computeMosCache(
    MDataBlock& block, unsigned int numVertices,
    const MosParams& params, const MMatrix* bindSpace, unsigned int geometryIndex) {
    const bool direct = bindSpace != nullptr;
    m_mosCache.valid = false;

    // [DIAG] 读诊断开关（wc_unity 影响 bind 时 w_c）
    int diagFlags = 0;
    {
        MStatus dfStatus;
        MDataHandle dfh = block.inputValue(aDiagFlags, &dfStatus);
        if (dfStatus) diagFlags = dfh.asInt();
    }

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
    status = direct ? inArr.jumpToElement(geometryIndex) : inArr.jumpToArrayElement(0);
    if (!status) return false;
    MDataHandle inData = inArr.inputValue(&status);
    if (!status) return false;
    MObject meshObj = inData.child(MPxSkinCluster::inputGeom).asMesh();
    if (meshObj.isNull()) return false;

    MFnMesh meshFn(meshObj);
    MPointArray points;
    status = meshFn.getPoints(points, direct ? MSpace::kObject : MSpace::kWorld);
    if (!status || points.length() < numVertices) return false;
    if (direct) {
        for (unsigned int i = 0; i < points.length(); ++i) points[i] *= *bindSpace;
    }

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

    std::vector<mathcore::RigidTransform> bindInverses, relative;
    std::vector<mathcore::Quat> poseRotations;
    if (direct && !readBoneTransforms(block, bindInverses, relative, poseRotations)) return false;
    if (direct && std::any_of(chain.begin(), chain.end(), [&](unsigned int index) {
        return index >= bindInverses.size();
    })) return false;

    m_mosCache.curves.clear();

    auto push_cv = [](MosCurve& c, const mathcore::Vec3& p,
        unsigned int bone, double u) {
            c.cvs_rest.push_back(p);
            c.cvs_bone.push_back(bone);
            c.cvs_u.push_back(u);
        };

    auto finalize_curve = [&](MosCurve& c) -> bool {
        const int n_cv_local = (int)c.cvs_rest.size();
        if (n_cv_local < 1) return false;
        if ((int)c.cvs_u.size() != n_cv_local) return false;
        for (int i = 1; i < n_cv_local; ++i)
            if (c.cvs_u[i] < c.cvs_u[i - 1] - 1e-12) return false;
        c.chain_bones = chain;
        if (direct) {
            for (const auto bone : c.cvs_bone) {
                c.bindFrames.push_back(ocd_math::mat3_from_quat(
                    mathcore::q_inverse(bindInverses[bone].q)));
            }
        }
        c.degree = ocd_math::effective_degree(n_cv_local, 3);
        if (c.degree < 1) return false;

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

    if (params.curveMode == 0) {
        MosCurve c;
        const int segmentCount = direct ? n_bones : n_bones - 1;
        const double u_den = segmentCount;
        for (int i = 0; i < segmentCount; ++i) {
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

        if (!direct && params.endLock && c.cvs_rest.size() >= 4) {
            c.cvs_bone.back() = chain[n_bones - 2];
        }
        c.bend_joint_idx = -1;

        if (!finalize_curve(c)) return false;
        m_mosCache.curves.push_back(std::move(c));
    }
    else {
        for (int j = 1; j < n_bones; ++j) {
            const unsigned int B_parent = chain[j - 1];
            const unsigned int B_child = chain[j];
            const mathcore::Vec3 joint = bind_jp[B_child];
            const mathcore::Vec3 dir_parent = bone_dir[j - 1];
            const mathcore::Vec3 dir_child = bone_dir[j];
            const double L_parent = bone_len[j - 1];
            const double L_child = bone_len[j];

            MosCurve c;
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.5)), B_parent, 0.25);
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.3)), B_parent, 0.35);
            push_cv(c, mathcore::v3_sub(joint,
                mathcore::v3_scale(dir_parent, L_parent * 0.1)), B_parent, 0.45);
            push_cv(c, joint, B_child, 0.5);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.1)), B_child, 0.55);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.3)), B_child, 0.65);
            push_cv(c, mathcore::v3_add(joint,
                mathcore::v3_scale(dir_child, L_child * 0.5)), B_child, 0.75);

            if (!direct && params.endLock && j + 1 == n_bones && c.cvs_rest.size() >= 4) {
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

    m_mosCache.verts.assign(numVertices, {});

#pragma omp parallel for schedule(static)
    for (int vi = 0; vi < (int)numVertices; ++vi) {
        const mathcore::Vec3 pv = {
            (double)points[vi].x, (double)points[vi].y, (double)points[vi].z
        };

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

            const mathcore::Vec3 O_i = mathcore::v3_sub(pv, C_tau);

            mathcore::Vec3 w_center = C_tau;
            if (params.curveMode != 0 && cv.cvs_rest.size() >= 3) {
                w_center = cv.cvs_rest[cv.cvs_rest.size() / 2];
            }
            const double d_curve = mathcore::v3_len(
                mathcore::v3_sub(pv, w_center));

            const double L_bone = m_mosCache.avg_bone_length;
            double w_c = 0.0;

            if (direct && params.curveMode == 0) {
                if (vi < static_cast<int>(m_vtxCache.vertices.size())) {
                    double total = 0.0;
                    for (const auto& weight : m_vtxCache.vertices[vi]) {
                        total += weight.second;
                        if (std::find(chain.begin(), chain.end(), weight.first) != chain.end())
                            w_c += weight.second;
                    }
                    if (total > 0.0) w_c /= total;
                }
            }
            else if (params.curveMode == 0) {
                const double x_wc = d_curve / std::max(1e-9, L_bone);
                w_c = std::exp(-x_wc * x_wc);
            }
            else {
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

                    // [DIAG-WC-UNITY] 忽略骨骼权重混合度
                    if (diagFlags & 4) {
                        if (w_c > 0.0) w_c = 1.0;
                    }
                }
            }

            MosPerCurveBind pc;
            pc.tau = t;
            pc.O = O_i;
            if (direct && w_c > 0.0) {
                pc.basis = mos_math::sampleBasis(cv.knots, cv.degree, t);
                if (!mos_math::bindOffset(pv, cv.cvs_rest, cv.bindFrames, pc.basis, pc.O))
                    w_c = 0.0;
            }
            pc.w_c = w_c;
            pc.vertex_joint = vj;
            vb.per_curve[c] = pc;
        }

        m_mosCache.verts[vi] = std::move(vb);
    }

    m_mosCache.numVertices = numVertices;
    m_mosCache.direct = direct;
    m_mosCache.geometryIndex = geometryIndex;
    if (direct) m_mosCache.bindSpace = *bindSpace;
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
    }
    return true;
}

MStatus OptimizedSkinCluster::deformMOS(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex) {
    using namespace mathcore;
    const unsigned int count = iter.count();
    const MosParams params = readMosParams(block);
    const bool debug = optSkinDebugLog() || params.debugLog;
    if (!m_mosCache.valid || !m_mosCache.direct || m_mosCache.geometryIndex != multiIndex ||
        !m_mosCache.bindSpace.isEquivalent(worldMat, 1e-12) || m_mosCache.numVertices != count ||
        m_mosCache.params.curveMode != params.curveMode ||
        m_mosCache.params.curveSamples != params.curveSamples) {
        if (!computeMosCache(block, count, params, &worldMat, multiIndex)) {
            if (debug) MGlobal::displayWarning("[MOS] No usable curve binding; using DLB.");
            return deformDLB(block, iter, worldMat, multiIndex);
        }
    }

    std::vector<RigidTransform> bindInverses, transforms;
    std::vector<Quat> poseRotations;
    if (!readBoneTransforms(block, bindInverses, transforms, poseRotations)) return MS::kFailure;

    struct PoseControl {
        Vec3 center;
        ocd_math::Mat3 frame;
        mos_math::BendControl bend;
    };
    std::vector<std::vector<PoseControl>> poseCurves(m_mosCache.curves.size());
    const auto& pivots = m_boneInfo.jointPosRest;
    const bool secondary = params.inBulgeWeight != 0.0 || params.outBulgeWeight != 0.0 ||
        params.sideBulgeWeight != 0.0 || params.slideInWeight_prev != 0.0 ||
        params.slideInWeight_next != 0.0 || params.slideOutWeight_prev != 0.0 ||
        params.slideOutWeight_next != 0.0 || params.slideOrthoInWeight_prev != 0.0 ||
        params.slideOrthoInWeight_next != 0.0 || params.slideOrthoOutWeight_prev != 0.0 ||
        params.slideOrthoOutWeight_next != 0.0 || params.centerSlideInWeight != 0.0 ||
        params.centerSlideOutWeight != 0.0;

    for (size_t c = 0; c < m_mosCache.curves.size(); ++c) {
        const auto& curve = m_mosCache.curves[c];
        auto& controls = poseCurves[c];
        controls.reserve(curve.cvs_rest.size());
        for (size_t j = 0; j < curve.cvs_rest.size(); ++j) {
            const auto bone = curve.cvs_bone[j];
            const auto& restPoint = curve.cvs_rest[j];
            PoseControl control{apply_transform(transforms[bone], restPoint),
                ocd_math::mat3_from_quat(transforms[bone].q) * curve.bindFrames[j], {}};
            if (secondary) {
                const auto& chain = curve.chain_bones;
                const size_t begin = curve.bend_joint_idx > 0 ? curve.bend_joint_idx : 1;
                const size_t end = curve.bend_joint_idx > 0 ? begin + 1 : chain.size();
                double bestDistance = 1.0;
                for (size_t k = begin; k < end; ++k) {
                    const auto parent = chain[k - 1];
                    const auto child = chain[k];
                    const bool incoming = bone == parent;
                    if (!incoming && bone != child) continue;
                    const auto segment = v3_sub(pivots[child], pivots[parent]);
                    const double length = v3_len(segment);
                    if (length < 1e-8) continue;
                    const double distance = v3_len(v3_sub(restPoint, pivots[child])) /
                        (0.85 * length);
                    if (distance >= bestDistance) continue;

                    const auto axis = v3_scale(segment, 1.0 / length);
                    const auto outgoingBind = k + 1 < chain.size()
                        ? v3_norm(v3_sub(pivots[chain[k + 1]], pivots[child])) : axis;
                    const auto incomingPose = q_rotate_v3(transforms[parent].q, axis);
                    const auto outgoingPose = q_rotate_v3(transforms[child].q, outgoingBind);
                    const auto normal = v3_cross(incomingPose, outgoingPose);
                    const double sine = v3_len(normal);
                    if (sine < 1e-8) continue;
                    const double bindAngle = std::acos(std::clamp(v3_dot(axis, outgoingBind), -1.0, 1.0));
                    const double angle = std::atan2(sine, v3_dot(incomingPose, outgoingPose));
                    const double bend = std::abs(angle - bindAngle) / 3.14159265358979323846;
                    const auto planeNormal = v3_scale(normal, 1.0 / sine);
                    const auto localAxis = incoming ? incomingPose : outgoingPose;
                    auto inward = v3_cross(planeNormal, localAxis);
                    auto localNormal = planeNormal;
                    if (params.useManualInVector) {
                        const auto manual = q_rotate_v3(transforms[bone].q, params.manualInVector);
                        const auto projected = v3_norm(v3_sub(manual, v3_scale(localAxis, v3_dot(manual, localAxis))));
                        if (v3_len(projected) > 0.0) {
                            inward = projected;
                            localNormal = v3_cross(localAxis, inward);
                        }
                    }
                    const double falloff = 1.0 - distance * distance * (3.0 - 2.0 * distance);
                    control.bend = {localAxis, inward, localNormal, bend * falloff, length, incoming};
                    bestDistance = distance;
                }
            }
            controls.push_back(control);
        }
    }

    if (params.scaleWeight > 1e-9 && params.scaleMode != 0) {
        std::vector<Vec3> x, y, z;
        if (readBoneLinearParts(block, worldMat, x, y, z)) {
            for (size_t c = 0; c < poseCurves.size(); ++c) {
                auto& controls = poseCurves[c];
                const auto& curve = m_mosCache.curves[c];
                for (size_t j = 0; j < controls.size(); ++j) {
                    const auto bone = curve.cvs_bone[j];
                    if (bone >= x.size()) continue;
                    auto& control = controls[j];
                    const auto scale = [&](double value) {
                        return std::pow(std::max(1e-9, value), params.scaleWeight);
                    };
                    if (params.scaleMode == 1) {
                        const auto axis = m_boneInfo.boneDirRest[bone];
                        const auto mapped = v3_add(v3_scale(x[bone], axis.x),
                            v3_add(v3_scale(y[bone], axis.y), v3_scale(z[bone], axis.z)));
                        control.frame = ocd_math::mat3_axis_scale(q_rotate_v3(transforms[bone].q, axis),
                            scale(v3_len(mapped))) * control.frame;
                    }
                    else {
                        control.frame = ocd_math::mat3_anisotropic_scale(scale(v3_len(x[bone])),
                            scale(v3_len(y[bone])), scale(v3_len(z[bone]))) * control.frame;
                    }
                }
            }
        }
    }

    const mos_math::Secondary incoming{params.inBulgeWeight, params.outBulgeWeight, params.sideBulgeWeight,
        params.slideInWeight_prev, params.slideOutWeight_prev, params.slideOrthoInWeight_prev,
        params.slideOrthoOutWeight_prev, params.centerSlideInWeight, params.centerSlideOutWeight};
    const mos_math::Secondary outgoing{params.inBulgeWeight, params.outBulgeWeight, params.sideBulgeWeight,
        params.slideInWeight_next, params.slideOutWeight_next, params.slideOrthoInWeight_next,
        params.slideOrthoOutWeight_next, params.centerSlideInWeight, params.centerSlideOutWeight};
    const double envelope = readEnvelopeFromNode();
    const auto worldInverse = worldMat.inverse();
    const auto deformPoint = [&](const MPoint& point, unsigned int vertex) {
        if (vertex >= m_mosCache.verts.size() || vertex >= m_vtxCache.vertices.size()) return point;
        const auto& weights = m_vtxCache.vertices[vertex];
        if (weights.empty()) return point;
        const MPoint world = point * worldMat;
        const Vec3 original{world.x, world.y, world.z};
        Vec3 sampled{};
        double coverage = 0.0;
        const auto& bind = m_mosCache.verts[vertex].per_curve;
        for (size_t c = 0; c < bind.size(); ++c) {
            const auto& attachment = bind[c];
            if (attachment.w_c <= 0.0) continue;
            Vec3 value{};
            for (int k = 0; k < attachment.basis.count; ++k) {
                const auto& control = poseCurves[c][attachment.basis.first + k];
                const auto sample = mos_math::sampleControl(control.center, control.frame, attachment.O,
                    control.bend, control.bend.incoming ? incoming : outgoing);
                value = v3_add(value, v3_scale(sample, attachment.basis.weights[k]));
            }
            sampled = v3_add(sampled, v3_scale(value, attachment.w_c));
            coverage += attachment.w_c;
        }
        const double mix = std::clamp(coverage * params.influenceRadius, 0.0, 1.0);
        Vec3 result = coverage > 0.0 ? v3_scale(sampled, mix / coverage) : Vec3{};
        if (mix < 1.0) {
            Vec3 fallback{};
            double total = 0.0;
            for (const auto& weight : weights) {
                if (weight.first >= transforms.size()) continue;
                fallback = v3_add(fallback, v3_scale(apply_transform(transforms[weight.first], original),
                    weight.second));
                total += weight.second;
            }
            result = v3_add(result, v3_scale(total > 0.0 ? v3_scale(fallback, 1.0 / total) : original,
                1.0 - mix));
        }
        result = applyEnvelope(original, result, envelope);
        return MPoint(result.x, result.y, result.z) * worldInverse;
    };

    MPointArray points;
    MStatus status = iter.allPositions(points);
    if (!status) return status;
    std::vector<unsigned int> vertexIndices;
    vertexIndices.reserve(points.length());
    for (; !iter.isDone(); iter.next()) vertexIndices.push_back(iter.index());
    if (vertexIndices.size() != points.length()) return MS::kFailure;
#pragma omp parallel for schedule(static)
    for (int row = 0; row < static_cast<int>(points.length()); ++row)
        points[row] = deformPoint(points[row], vertexIndices[row]);
    status = iter.reset();
    if (!status) return status;
    return iter.setAllPositions(points);
}

// ================================================================
// V2 模式 4 的两个 helper
// ================================================================

// 顶点级 Joint Influence Mask：M = 2 · min(w1, w2)
//
// w1 / w2 = 顶点最大的两个骨骼权重（m_vtxCache.vertices 已按降序排序）。
//   · 单骨骼（w2 = 0）          → M = 0（不在关节混合区，不参与 CoR/MOS）
//   · 0.7 / 0.3                 → M = 0.6
//   · 0.5 / 0.5                 → M = 1.0（中线，完整保留 CoR 外心）
//   · 3 骨骼 0.5 / 0.3 / 0.2    → M = 0.6（只取最大两个，方案 A 的既定近似）
//
// 语义：M 同时作为「CoR 混合权重」和「MOS 影响范围」，是模式 4 里唯一
// 逐顶点范围控制。influenceRadius 在模式 4 不参与（见主循环注释）。
static double computeVertexJointMask(
    const std::vector<std::pair<unsigned int, float>>& vw) {
    if (vw.size() < 2) return 0.0;
    const double w1 = (double)vw[0].second;
    const double w2 = (double)vw[1].second;
    return 2.0 * std::min(w1, w2);
}

// 局部 CoR：V_base = V_LBS + M · (V_CoR − V_LBS)
//
//   · M = 0 → 完全 LBS（单骨骼顶点）
//   · M = 1 → 完全 CoR（关节中线，CoR 外心效应完整保留）
//   · 中间  → 平滑混合
//
// 注意：
//   · 此公式仅用于模式 4；模式 0（deformCoR）保持全局纯 CoR 不变。
//   · 内部重算 LBS：cor_vertex_v2 用的是「混合旋转 + 加权平移」的 CoR 公式，
//     与 LBS 是不同算式，不能直接复用。
//   · bind 姿势下 V_CoR = V_LBS = origPos，差值 0，M 不影响 bind 精确性。
static mathcore::Vec3 computeLocalCorBase(
    const mathcore::Vec3& origPos,
    const mathcore::Vec3& cor,
    const std::vector<std::pair<unsigned int, float>>& vw,
    const std::vector<mathcore::RigidTransform>& Cs,
    double M) {
    // ---- V_LBS ----
    mathcore::Vec3 lbs = { 0.0, 0.0, 0.0 };
    double w_tot = 0.0;
    for (const auto& kv : vw) {
        const unsigned int b = kv.first;
        const double w = (double)kv.second;
        if (b >= (unsigned int)Cs.size()) continue;
        lbs = mathcore::v3_add(lbs,
            mathcore::v3_scale(mathcore::apply_transform(Cs[b], origPos), w));
        w_tot += w;
    }
    if (w_tot < 1e-9) return origPos;
    lbs = mathcore::v3_scale(lbs, 1.0 / w_tot);

    // ---- V_CoR ----
    const mathcore::Vec3 corFull = cor_vertex_v2(origPos, cor, vw, Cs);

    // ---- 混合 ----
    return mathcore::v3_add(lbs, mathcore::v3_scale(
        mathcore::v3_sub(corFull, lbs), M));
}

// ================================================================
// V1: CoR + MOS Secondary
//
//   V_final = V_CoR + α · W_MOS · ΔV_MOS
//
//   · V_CoR      : CoR 主变形（与模式 0 完全一致）
//   · ΔV_MOS     : 各曲线 Bulge/Slide 净增量，按 w_c 加权
//   · W_MOS      : w_total · influenceRadius，线性 clamp 到 1（V1 不做 gamma 抬升）
//   · α          : corMosSecondaryWeight ∈ [-2, 2]
//   · Envelope   : 只在最后统一执行一次
//
// 关键约束：
//   1) 不调用 deformCoR() / deformMOS()，避免二次 Envelope 与顶点写回
//   2) ΔV 不含 curve_mos 本身（基础曲线变形），也不含 expand_scale
//   3) α=0 严格退化为纯 CoR；α=0 时提前 return，不构建 MOS cache
//   4) 本函数内 CoR cache 维护逻辑与 deformCoR 内联，保持一致；未来抽成
//      ensureCoRCache() 时两处同改
// ================================================================
MStatus OptimizedSkinCluster::deformCoRMosSecondary(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    // [DIAG] 读诊断开关（模式 4 也需要 bit8）
    int diagFlags = 0;
    {
        MStatus dfStatus;
        MDataHandle dfh = block.inputValue(aDiagFlags, &dfStatus);
        if (dfStatus) diagFlags = dfh.asInt();
    }

    // ---- A. 参数 ----
    const double env = readEnvelopeFromNode();
    MosParams P = readMosParams(block);

    // α=0：完全退化为纯 CoR，跳过 MOS cache 构建
    if (std::abs(P.corMosSecondaryWeight) < 1e-9) {
        return deformCoR(block, iter, worldMat, 0);
    }

    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    const unsigned int nV = iter.count();

    // ---- B. CoR cache 维护（与 deformCoR 内联，保持一致） ----
    {
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
                return deformDLB(block, iter, worldMat, 0);   // 同 deformCoR 的 fallback
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
    }

    // ---- C. MOS cache 构建 ----
    if (!m_mosCache.valid || m_mosCache.direct || m_mosCache.numVertices != nV ||
        !m_mosCache.params.nearlyEqual(P)) {
        if (!computeMosCache(block, nV, P))
            return deformCoR(block, iter, worldMat, 0);   // MOS cache 失败退纯 CoR
    }
    const MosParams& Pc = m_mosCache.params;

    const int n_curves = (int)m_mosCache.curves.size();
    if (n_curves == 0) return deformCoR(block, iter, worldMat, 0);

    // ---- D. 每条曲线的弯曲关节索引（V1 从 deformMOS 逐字拷贝）----
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

    // ---- E. curve_tau_bend / curve_bend_angle（V1 从 deformMOS 拷贝）----
    std::vector<double> curve_tau_bend(n_curves, 0.5);
    std::vector<double> curve_bend_angle(n_curves, 0.0);
    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        const int bj = curve_bend_idx[c];
        if (bj < 1 || bj >= (int)cv.chain_bones.size()) continue;

        const unsigned int bp = cv.chain_bones[bj - 1];
        const unsigned int bc = cv.chain_bones[bj];

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

        if (bp < Cs.size() && bc < Cs.size()) {
            mathcore::Quat q_rel = mathcore::q_mul(
                Cs[bc].q, mathcore::q_inverse(Cs[bp].q));
            mathcore::Vec3 axis_bind = mathcore::Vec3{ 0.0, 1.0, 0.0 };
            if (bc < m_boneInfo.boneDirRest.size())
                axis_bind = m_boneInfo.boneDirRest[bc];

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

    // ---- F. all_p_j / all_F_j 预计算（V1 从 deformMOS 拷贝）----
    std::vector<std::vector<mathcore::Vec3>> all_p_j(n_curves);
    std::vector<std::vector<ocd_math::Mat3>> all_F_j(n_curves);

    for (int c = 0; c < n_curves; ++c) {
        const auto& cv = m_mosCache.curves[c];
        const int n_cv = (int)cv.cvs_rest.size();
        const int nB = (int)cv.chain_bones.size();
        all_p_j[c].resize(n_cv);
        all_F_j[c].resize(n_cv);

        double seg_begin = 0.0;
        double seg_scale = (nB > 1) ? double(nB - 1) : 1.0;
        if (cv.bend_joint_idx >= 1 && cv.bend_joint_idx < nB) {
            seg_begin = double(cv.bend_joint_idx - 1);
            seg_scale = 2.0;
        }

        for (int j = 0; j < n_cv; ++j) {
            const unsigned int bone = cv.cvs_bone[j];
            if (bone < (unsigned int)Cs.size())
                all_p_j[c][j] = mathcore::apply_transform(Cs[bone], cv.cvs_rest[j]);
            else
                all_p_j[c][j] = cv.cvs_rest[j];

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
                all_F_j[c][j] = ocd_math::mat3_from_quat(Cs[bone].q);
            }
            else {
                all_F_j[c][j] = ocd_math::Mat3::identity();
            }
        }
    }

    // Scale 叠加（V1 从 deformMOS 拷贝）
    if (Pc.scaleWeight > 1e-9 && Pc.scaleMode != 0) {
        std::vector<mathcore::Vec3> L0, L1, L2;
        const bool haveLinear = readBoneLinearParts(block, worldMat, L0, L1, L2);
        if (haveLinear) {
            for (int c = 0; c < n_curves; ++c) {
                const auto& cv = m_mosCache.curves[c];
                const int n_cv = (int)cv.cvs_rest.size();
                for (int j = 0; j < n_cv; ++j) {
                    const unsigned int bone = cv.cvs_bone[j];
                    if (bone >= (unsigned int)L0.size()) continue;

                    if (Pc.scaleMode == 1) {
                        mathcore::Vec3 a_bind = { 0.0, 1.0, 0.0 };
                        if (bone < m_boneInfo.boneDirRest.size())
                            a_bind = m_boneInfo.boneDirRest[bone];

                        const mathcore::Vec3 La = mathcore::v3_add(
                            mathcore::v3_add(
                                mathcore::v3_scale(L0[bone], a_bind.x),
                                mathcore::v3_scale(L1[bone], a_bind.y)),
                            mathcore::v3_scale(L2[bone], a_bind.z));
                        const double s_axis = mathcore::v3_len(La);
                        if (s_axis < 1e-9) continue;

                        const mathcore::Vec3 axis_cur = all_F_j[c][j].apply(a_bind);
                        if (mathcore::v3_len(axis_cur) < 1e-9) continue;

                        const double s_pow = std::pow(s_axis, Pc.scaleWeight);
                        const ocd_math::Mat3 S =
                            ocd_math::mat3_axis_scale(axis_cur, s_pow);
                        all_F_j[c][j] = S * all_F_j[c][j];
                    }
                    else if (Pc.scaleMode == 2) {
                        const double sx = mathcore::v3_len(L0[bone]);
                        const double sy = mathcore::v3_len(L1[bone]);
                        const double sz = mathcore::v3_len(L2[bone]);
                        const ocd_math::Mat3 S = ocd_math::mat3_anisotropic_scale(
                            std::pow(std::max(sx, 1e-9), Pc.scaleWeight),
                            std::pow(std::max(sy, 1e-9), Pc.scaleWeight),
                            std::pow(std::max(sz, 1e-9), Pc.scaleWeight));
                        all_F_j[c][j] = S * all_F_j[c][j];
                    }
                }
            }
        }
    }

    // ---- G. 主循环 ----
    MPointArray points;
    iter.allPositions(points);
    const MMatrix wInv = worldMat.inverse();

    const bool hasInBulge = (std::abs(Pc.inBulgeWeight) > 1e-9);
    const bool hasOutBulge = (std::abs(Pc.outBulgeWeight) > 1e-9);
    const bool hasSideBulge = (std::abs(Pc.sideBulgeWeight) > 1e-9);
    const bool hasCenterSlide =
        (std::abs(Pc.centerSlideInWeight) > 1e-9 ||
            std::abs(Pc.centerSlideOutWeight) > 1e-9);
    const bool hasInSlide = (std::abs(Pc.slideInWeight_prev) > 1e-9 ||
        std::abs(Pc.slideInWeight_next) > 1e-9);
    const bool hasOutSlide = (std::abs(Pc.slideOutWeight_prev) > 1e-9 ||
        std::abs(Pc.slideOutWeight_next) > 1e-9);
    const bool hasInOrthoSlide = (std::abs(Pc.slideOrthoInWeight_prev) > 1e-9 ||
        std::abs(Pc.slideOrthoInWeight_next) > 1e-9);
    const bool hasOutOrthoSlide = (std::abs(Pc.slideOrthoOutWeight_prev) > 1e-9 ||
        std::abs(Pc.slideOrthoOutWeight_next) > 1e-9);
    const bool hasAnySlide = hasInSlide || hasOutSlide ||
        hasInOrthoSlide || hasOutOrthoSlide || hasCenterSlide;
    const bool hasAnySecondary = hasInBulge || hasOutBulge || hasSideBulge || hasAnySlide;

    const double alpha = Pc.corMosSecondaryWeight;
    // [V2] influenceRadius 在模式 4 不参与：
    //   · 影响范围由 computeVertexJointMask(vw) 统一控制（CoR / MOS 共用）
    //   · 参数仅在模式 3（deformMOS）中保留原语义

#pragma omp parallel for schedule(static)
    for (int vi = 0; vi < (int)nV; ++vi) {
        const auto& vw = m_vtxCache.vertices[vi];
        if (vw.empty()) continue;

        MPoint ptW = points[vi] * worldMat;
        const mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };

        // 统一 Joint Influence Mask（CoR 与 MOS 共用同一个 M）
        const double M = computeVertexJointMask(vw);

        // ---------- 1. 局部 CoR 基础变形 ----------
        // V_base = LBS + M · (CoR − LBS)
        //   · M = 0（单骨骼）    → LBS
        //   · M = 1（关节中线）  → 纯 CoR
        // bind 姿势下 CoR = LBS = origPos，M 不影响精确性。
        mathcore::Vec3 cor = origPos;
        if ((unsigned)vi < m_corCache.cors.size()) {
            mathcore::Vec3 cl = m_corCache.cors[vi];
            MPoint cw(cl.x, cl.y, cl.z); cw *= worldMat;
            cor = { cw.x, cw.y, cw.z };
        }
        const mathcore::Vec3 corDef =
            computeLocalCorBase(origPos, cor, vw, Cs, M);

        // ---------- 2. MOS 次级增量 ΔV ----------
        mathcore::Vec3 delta_mos = { 0.0, 0.0, 0.0 };
        double w_total = 0.0;

        if (hasAnySecondary) {
            const auto& vb = m_mosCache.verts[vi];
            if (!vb.per_curve.empty()) {
                mathcore::Vec3 blended_delta = { 0.0, 0.0, 0.0 };

                for (int c = 0; c < n_curves; ++c) {
                    const auto& cv = m_mosCache.curves[c];
                    const auto& pc = vb.per_curve[c];
                    if (pc.w_c <= 0.0) continue;

                    const int n_cv_local = (int)cv.cvs_rest.size();
                    std::vector<mathcore::Vec3> offset_cvs(n_cv_local);

                    // ---- expand_scale：曲线基础变形，不算次级 ----
                    double expand_scale = 1.0;
                    // [DIAG bit8] 关掉 expand_scale：验证 W 是否消失
                    //   expand_scale 在 τ=0.25/0.75 最大（w_c=0.5），
                    //   与 bulge 在 τ=0.5 最大的方向相反，是 W 的头号嫌疑
                    if (!(diagFlags & 8) &&
                        pc.w_c > 1e-4 && pc.w_c < 1.0 - 1e-4) {
                        const double w_edge = 4.0 * pc.w_c * (1.0 - pc.w_c);
                        expand_scale = 1.0 + 0.3 * w_edge * curve_bend_angle[c];
                    }

                    // ---- 侧向法向（用于分离平面内/侧向分量）----
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
                                const double side_comp = mathcore::v3_dot(F_O, side_normal_exp);
                                const mathcore::Vec3 side_part = mathcore::v3_scale(side_normal_exp, side_comp);
                                const mathcore::Vec3 plane_part = mathcore::v3_sub(F_O, side_part);
                                F_O = mathcore::v3_add(
                                    mathcore::v3_scale(plane_part, expand_scale),
                                    side_part);
                            }
                            else {
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
                    mathcore::Vec3 curve_mos_final = curve_mos;

                    // ==================== Bulge + Slide（逐字从 deformMOS 拷贝） ====================
                    if (hasAnySecondary &&
                        curve_bend_angle[c] > 1e-6 &&
                        curve_bend_idx[c] >= 1 &&
                        curve_bend_idx[c] < (int)cv.chain_bones.size()) {

                        // ---- 顶点权重混合旋转 ----
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
                                bp_cur = mathcore::apply_transform(Cs[bp_idx], rest[bp_idx].t);
                            if (bc_idx < rest.size() && bc_idx < Cs.size())
                                bc_cur = mathcore::apply_transform(Cs[bc_idx], rest[bc_idx].t);
                            if (bn_idx < rest.size() && bn_idx < Cs.size() &&
                                bn_idx != bc_idx)
                                bn_cur = mathcore::apply_transform(Cs[bn_idx], rest[bn_idx].t);
                            else
                                bn_cur = bc_cur;

                            mathcore::Vec3 bone_axis_prev_cur =
                                mathcore::v3_sub(bc_cur, bp_cur);
                            const double len_prev = mathcore::v3_len(bone_axis_prev_cur);
                            if (len_prev > 1e-9)
                                bone_axis_prev_cur = mathcore::v3_scale(
                                    bone_axis_prev_cur, 1.0 / len_prev);
                            else
                                bone_axis_prev_cur = mathcore::Vec3{ 0.0, 1.0, 0.0 };

                            mathcore::Vec3 bone_axis_next_cur =
                                mathcore::v3_sub(bn_cur, bc_cur);
                            const double len_next = mathcore::v3_len(bone_axis_next_cur);
                            if (len_next > 1e-9)
                                bone_axis_next_cur = mathcore::v3_scale(
                                    bone_axis_next_cur, 1.0 / len_next);
                            else
                                bone_axis_next_cur = mathcore::Vec3{ 0.0, 1.0, 0.0 };

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
                                if (Pc.useManualInVector) {
                                    const mathcore::Vec3 mv = Pc.manualInVector;
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
                                if (!used_manual)
                                    radial_cur = mathcore::q_rotate_v3(q_mix, pc.O);

                                // 投影到垂直骨骼轴平面
                                {
                                    const double radial_len_orig =
                                        mathcore::v3_len(radial_cur);
                                    if (radial_len_orig > 1e-9) {
                                        mathcore::Vec3 bone_axis_mix_cur =
                                        { 0.0, 0.0, 0.0 };
                                        double axis_w = 0.0;
                                        for (const auto& kv : m_vtxCache.vertices[vi]) {
                                            const unsigned int b = kv.first;
                                            const double w = (double)kv.second;
                                            if (b >= (unsigned int)Cs.size()) continue;
                                            if (b >= m_boneInfo.boneDirRest.size()) continue;
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
                                                radial_cur = mathcore::v3_scale(
                                                    radial_perp,
                                                    radial_len_orig / perp_len);
                                            }
                                        }
                                    }
                                }
                            }

                            const double in_len = mathcore::v3_len(in_dir_cur);
                            const double pc_o_len = mathcore::v3_len(pc.O);
                            const double radial_len = mathcore::v3_len(radial_cur);
                            const double base = curve_bend_angle[c];

                            // ---- Bulge 段 ----
                            if (hasInBulge || hasOutBulge || hasSideBulge) {
                                mathcore::Vec3 total_bulge = { 0.0, 0.0, 0.0 };
                                if (in_len > 1e-9 && pc_o_len > 1e-9) {
                                    const double cos_side =
                                        mathcore::v3_dot(pc.O, in_dir_cur) / pc_o_len;
                                    double t_norm = (cos_side + 0.5) / 1.0;
                                    if (t_norm < 0.0) t_norm = 0.0;
                                    if (t_norm > 1.0) t_norm = 1.0;
                                    const double smooth_w =
                                        t_norm * t_norm * (3.0 - 2.0 * t_norm);
                                    const double w_out = smooth_w;
                                    const double w_in = 1.0 - smooth_w;

                                    if (hasInBulge && w_in > 1e-9) {
                                        total_bulge = mathcore::v3_add(
                                            total_bulge,
                                            mathcore::v3_scale(in_dir_cur,
                                                -Pc.inBulgeWeight * base * w_in));
                                    }
                                    if (hasOutBulge && w_out > 1e-9) {
                                        total_bulge = mathcore::v3_add(
                                            total_bulge,
                                            mathcore::v3_scale(in_dir_cur,
                                                Pc.outBulgeWeight * base * w_out));
                                    }

                                    if (hasSideBulge && radial_len > 1e-9) {
                                        mathcore::Vec3 side_normal = mathcore::v3_cross(
                                            bone_axis_prev_cur, bone_axis_next_cur);
                                        const double sn_len = mathcore::v3_len(side_normal);
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
                                                        Pc.sideBulgeWeight * base * w_side));
                                            }
                                        }
                                    }
                                }
                                curve_mos_final = mathcore::v3_add(
                                    curve_mos_final, total_bulge);
                            }

                            // ---- Slide 段 ----
                            if (hasAnySlide) {
                                mathcore::Vec3 total_slide = { 0.0, 0.0, 0.0 };
                                if (in_len > 1e-9 && pc_o_len > 1e-9 &&
                                    radial_len > 1e-9) {

                                    const double cos_side_slide =
                                        mathcore::v3_dot(pc.O, in_dir_cur) / pc_o_len;
                                    double t_side_slide = (cos_side_slide + 0.5) / 1.0;
                                    if (t_side_slide < 0.0) t_side_slide = 0.0;
                                    if (t_side_slide > 1.0) t_side_slide = 1.0;
                                    const double smooth_side_slide =
                                        t_side_slide * t_side_slide * (3.0 - 2.0 * t_side_slide);
                                    const double w_in_slide = 1.0 - smooth_side_slide;
                                    const double w_out_slide = smooth_side_slide;

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

                                    // 1. 内侧轴向
                                    if (w_in_slide > 1e-9) {
                                        if (std::abs(Pc.slideInWeight_prev) > 1e-9 &&
                                            w_prev_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(bone_axis_prev_cur,
                                                    Pc.slideInWeight_prev * base *
                                                    w_in_slide * w_prev_slide));
                                        }
                                        if (std::abs(Pc.slideInWeight_next) > 1e-9 &&
                                            w_next_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(bone_axis_next_cur,
                                                    -Pc.slideInWeight_next * base *
                                                    w_in_slide * w_next_slide));
                                        }
                                    }
                                    // 2. 外侧轴向
                                    if (w_out_slide > 1e-9) {
                                        if (std::abs(Pc.slideOutWeight_prev) > 1e-9 &&
                                            w_prev_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(bone_axis_prev_cur,
                                                    -Pc.slideOutWeight_prev * base *
                                                    w_out_slide * w_prev_slide));
                                        }
                                        if (std::abs(Pc.slideOutWeight_next) > 1e-9 &&
                                            w_next_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(bone_axis_next_cur,
                                                    Pc.slideOutWeight_next * base *
                                                    w_out_slide * w_next_slide));
                                        }
                                    }
                                    // 3. 内侧径向
                                    if (w_in_slide > 1e-9) {
                                        if (std::abs(Pc.slideOrthoInWeight_prev) > 1e-9 &&
                                            w_prev_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(radial_dir_slide,
                                                    -Pc.slideOrthoInWeight_prev * base *
                                                    w_in_slide * w_prev_slide));
                                        }
                                        if (std::abs(Pc.slideOrthoInWeight_next) > 1e-9 &&
                                            w_next_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(radial_dir_slide,
                                                    -Pc.slideOrthoInWeight_next * base *
                                                    w_in_slide * w_next_slide));
                                        }
                                    }
                                    // 4. 外侧径向
                                    if (w_out_slide > 1e-9) {
                                        if (std::abs(Pc.slideOrthoOutWeight_prev) > 1e-9 &&
                                            w_prev_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(radial_dir_slide,
                                                    Pc.slideOrthoOutWeight_prev * base *
                                                    w_out_slide * w_prev_slide));
                                        }
                                        if (std::abs(Pc.slideOrthoOutWeight_next) > 1e-9 &&
                                            w_next_slide > 1e-9) {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(radial_dir_slide,
                                                    Pc.slideOrthoOutWeight_next * base *
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
                                                    Pc.centerSlideInWeight * base));
                                        }
                                        else {
                                            total_slide = mathcore::v3_add(
                                                total_slide,
                                                mathcore::v3_scale(in_dir_cur,
                                                    -Pc.centerSlideOutWeight * base));
                                        }
                                    }
                                }
                                curve_mos_final = mathcore::v3_add(
                                    curve_mos_final, total_slide);
                            }
                        }
                    }

                    // ---- ΔV_c = curve_mos_final - curve_mos ----
                    const mathcore::Vec3 dV_c =
                        mathcore::v3_sub(curve_mos_final, curve_mos);
                    blended_delta = mathcore::v3_add(
                        blended_delta, mathcore::v3_scale(dV_c, pc.w_c));
                    w_total += pc.w_c;
                }

                if (w_total > 1e-6) {
                    delta_mos = mathcore::v3_scale(blended_delta, 1.0 / w_total);
                }
            }
        }

        // ---------- 3. 合成：V_final = V_localCor + α · M · ΔV_MOS ----------
        // 统一 Mask 语义：
        //   · M 同时限制 CoR 混合（见上一步）和 MOS 修正范围
        //   · 不再用 influenceRadius，不再做 pow(w, 0.5) 二次抬升
        //   · α 只调 MOS 补偿强度，范围由 M 决定
        const mathcore::Vec3 combined = mathcore::v3_add(
            corDef,
            mathcore::v3_scale(delta_mos, alpha * M));

        // ---------- 4. Envelope 唯一一次 ----------
        const mathcore::Vec3 bl = applyEnvelope(origPos, combined, env);
        MPoint out(bl.x, bl.y, bl.z);
        out *= wInv;
        points[vi] = out;
    }

    iter.setAllPositions(points);
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
// V1: CoR + Compensation
//
//   P_final = P_CoR + Strength · M^falloff · (P_CoR − P_LBS)
//           = P_LBS + (1 + Strength · M^falloff) · (P_CoR − P_LBS)
//
//   · M = 2·min(w1, w2)：顶点级 Joint Influence Mask（与模式 4 一致）
//   · Strength = 0：严格等于纯 CoR（回归测试用）
//   · Strength > 0：放大 CoR 补偿（更饱满）
//   · Strength < 0：削弱 CoR 补偿（趋近 LBS）
//   · bind 姿势下 ΔP = 0，M 无影响（bind 精确性不变）
// ================================================================
MStatus OptimizedSkinCluster::deformCoRComp(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int /*multiIndex*/) {

    const double env = readEnvelopeFromNode();
    MStatus st;
    const double strength = block.inputValue(aCorCompStrength, &st).asDouble();
    const double falloff = block.inputValue(aCorCompFalloff, &st).asDouble();
    const double strengthIn = block.inputValue(aCorCompStrengthIn, &st).asDouble();
    const double strengthOut = block.inputValue(aCorCompStrengthOut, &st).asDouble();

    std::vector<mathcore::RigidTransform> rest, Cs;
    std::vector<mathcore::Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    const unsigned int nV = iter.count();

    // ---- CoR cache 维护（与 deformCoR 内联，保持一致）----
    {
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
                if (dirtyList.size() > (size_t)nV / 4)
                    computeCoRCache(block, nV);
                else {
                    int vertLayers = computeVertLayers(
                        (unsigned int)dirtyList.size(), nV,
                        vertLayerOverride, budgetDiv);
                    updateCoRCacheIncremental(block, dirtyList, triLayers, vertLayers);
                }
            }
            clearDirtyFlags();
        }
    }

    MPointArray points;
    iter.allPositions(points);
    const MMatrix wInv = worldMat.inverse();
    const double envFinal = env;
    const double strengthFinal = strength;
    const double falloffFinal = (falloff < 1e-6) ? 1.0 : falloff;

#pragma omp parallel for schedule(static)
    for (int i = 0; i < (int)nV; ++i) {
        const auto& vw = m_vtxCache.vertices[i];
        if (vw.empty()) continue;

        MPoint ptW = points[i] * worldMat;
        const mathcore::Vec3 origPos = { ptW.x, ptW.y, ptW.z };

        // ---- 1. CoR ----
        mathcore::Vec3 cor = origPos;
        if ((unsigned)i < m_corCache.cors.size()) {
            mathcore::Vec3 cl = m_corCache.cors[i];
            MPoint cw(cl.x, cl.y, cl.z); cw *= worldMat;
            cor = { cw.x, cw.y, cw.z };
        }
        const mathcore::Vec3 corDef = cor_vertex_v2(origPos, cor, vw, Cs);

        // ---- 2. LBS（重算，cor_vertex_v2 内部不是 LBS）----
        mathcore::Vec3 lbs = { 0.0, 0.0, 0.0 };
        double wTot = 0.0;
        for (const auto& kv : vw) {
            const unsigned int b = kv.first;
            const double w = (double)kv.second;
            if (b >= (unsigned int)Cs.size()) continue;
            lbs = mathcore::v3_add(lbs,
                mathcore::v3_scale(mathcore::apply_transform(Cs[b], origPos), w));
            wTot += w;
        }
        if (wTot > 1e-9) lbs = mathcore::v3_scale(lbs, 1.0 / wTot);
        else lbs = origPos;

        // ---- 3. Mask：M = 2·min(w1, w2)，falloff 作为整体倍率 ----
        //
        // 原 pow(M, falloff) 是"形状控制"：M=1 处恒不变 → 中线成尖峰
        // 现改为整体倍率：falloff 也缩放中线，避免"中线单独凸起"
        //   falloff = 1.0：原始分布（中线 1，边缘 0）
        //   falloff = 2.0：整体倍率 2×（中线饱和到 1，边缘从 0 增强到 0~1）
        //   falloff = 0.5：整体减半（中线 0.5，边缘 0）
        //   falloff = 0.25：整体四分之一（中线 0.25）
        double M = 0.0;
        if (vw.size() >= 2) {
            const double w1 = (double)vw[0].second;
            const double w2 = (double)vw[1].second;
            M = 2.0 * std::min(w1, w2);
        }
        if (falloffFinal != 1.0) {
            M *= falloffFinal;
            if (M > 1.0) M = 1.0;
        }

        // ---- 3.5 内侧/外侧分区权重 ----
        // 关键：必须用【当前 pose】的关节位置算内角方向。
        // 上一版用 bind 骨骼方向（boneDirRest）是错的：
        //   bind 时骨骼伸直，parent_dir ≈ child_dir，child_dir - parent_dir ≈ 0
        //   → in_dir 长度过不了 1e-6，全部回退到 0.5/0.5 → 看起来像"整体加半"
        // 正确做法（MOS 逻辑）：
        //   prev_dir = normalize(bc_cur - bp_cur)  // parent 骨骼 当前方向
        //   next_dir = normalize(bn_cur - bc_cur)  // child  骨骼 当前方向
        //   in_dir   = normalize(next_dir - prev_dir)  // 内角方向
        //   用 LBS 后位置 lbs 相对关节 bc_cur 的偏移方向与 in_dir 点积判定
        double w_in = 0.5;
        double w_out = 0.5;
        mathcore::Vec3 in_dir_cur = { 0.0, 0.0, 0.0 };
        bool has_in_dir = false;
        if (vw.size() >= 2) {
            const unsigned int b1 = vw[0].first;
            const unsigned int b2 = vw[1].first;

            const int d1 = (b1 < m_boneInfo.depths.size())
                ? m_boneInfo.depths[b1] : 0;
            const int d2 = (b2 < m_boneInfo.depths.size())
                ? m_boneInfo.depths[b2] : 0;

            if (d1 != d2) {
                const unsigned int b_parent = (d1 < d2) ? b1 : b2;
                const unsigned int b_child = (d1 < d2) ? b2 : b1;

                const int bn_idx = (b_child < m_boneInfo.boneSoleChild.size())
                    ? m_boneInfo.boneSoleChild[b_child] : -1;

                if (bn_idx >= 0
                    && b_parent < (unsigned int)m_boneInfo.jointPosRest.size()
                    && b_child < (unsigned int)m_boneInfo.jointPosRest.size()
                    && (unsigned int)bn_idx < m_boneInfo.jointPosRest.size()
                    && b_parent < (unsigned int)Cs.size()
                    && b_child < (unsigned int)Cs.size()
                    && (unsigned int)bn_idx < (unsigned int)Cs.size()) {

                    const mathcore::Vec3 bp_cur = mathcore::apply_transform(
                        Cs[b_parent], m_boneInfo.jointPosRest[b_parent]);
                    const mathcore::Vec3 bc_cur = mathcore::apply_transform(
                        Cs[b_child], m_boneInfo.jointPosRest[b_child]);
                    const mathcore::Vec3 bn_cur = mathcore::apply_transform(
                        Cs[bn_idx], m_boneInfo.jointPosRest[bn_idx]);

                    const mathcore::Vec3 prev_dir = mathcore::v3_norm(
                        mathcore::v3_sub(bc_cur, bp_cur));
                    const mathcore::Vec3 next_dir = mathcore::v3_norm(
                        mathcore::v3_sub(bn_cur, bc_cur));
                    const mathcore::Vec3 in_dir_raw =
                        mathcore::v3_sub(next_dir, prev_dir);
                    const double in_len = mathcore::v3_len(in_dir_raw);

                    if (in_len > 1e-6) {
                        const mathcore::Vec3 in_dir =
                            mathcore::v3_scale(in_dir_raw, 1.0 / in_len);
                        in_dir_cur = in_dir;
                        has_in_dir = true;

                        // 用 LBS 后位置作为"顶点当前方位"参考
                        const mathcore::Vec3 rel = mathcore::v3_sub(lbs, bc_cur);
                        const double rel_len = mathcore::v3_len(rel);

                        if (rel_len > 1e-9) {
                            const double cos_side =
                                mathcore::v3_dot(rel, in_dir) / rel_len;
                            double t = (cos_side + 1.0) * 0.5;
                            if (t < 0.0) t = 0.0;
                            if (t > 1.0) t = 1.0;
                            w_in = t * t * (3.0 - 2.0 * t);
                            w_out = 1.0 - w_in;
                        }
                    }
                }
            }
        }

        // ---- 4. 合成 ----
        // 两层叠加，语义完全解耦：
        //
        //   [层 A] 基础补偿：strength 放大 CoR 补偿量
        //          combined = P_CoR + strength·M·(P_CoR − P_LBS)
        //          这是 V1 原有语义，strength=0 时完全关闭
        //
        //   [层 B] 方向位移：in/out 沿 in_dir 加固定方向位移
        //          内角：沿 -in_dir 推（in > 0 时朝外鼓）
        //          外角：沿 +in_dir 推（out > 0 时朝内收）
        //          w_in/w_out 严格分区，in 只影响内角、out 只影响外角
        //          幅度基准 = |P_CoR − P_LBS| · M
        //
        //   关键：层 B 不是"放大补偿向量"，而是"沿固定方向加独立位移"，
        //         所以 in 不会跨区影响外角轮廓，反之亦然。
        //         这是 MOS 里 in/out bulge 的做法。
        const mathcore::Vec3 delta = mathcore::v3_sub(corDef, lbs);
        const double delta_scale = mathcore::v3_len(delta);

        // 层 A：基础补偿
        mathcore::Vec3 combined = mathcore::v3_add(
            corDef,
            mathcore::v3_scale(delta, strengthFinal * M));

        // 层 B：内/外单侧方向位移
        if (has_in_dir && delta_scale > 1e-9) {
            const double amt = delta_scale * M;

            // 内角：沿 -in_dir 外推（strengthIn > 0 时内角外鼓）
            //   w_in ≈ 1（内角顶点）→ 完整幅度
            //   w_in ≈ 0（外角顶点）→ 无影响
            combined = mathcore::v3_add(
                combined,
                mathcore::v3_scale(in_dir_cur, -strengthIn * w_in * amt));

            // 外角：沿 +in_dir 内收（strengthOut > 0 时外角内收）
            //   w_out ≈ 1（外角顶点）→ 完整幅度
            //   w_out ≈ 0（内角顶点）→ 无影响
            combined = mathcore::v3_add(
                combined,
                mathcore::v3_scale(in_dir_cur, strengthOut * w_out * amt));
        }

        // ---- 5. Envelope（一次）----
        const mathcore::Vec3 bl = applyEnvelope(origPos, combined, envFinal);
        MPoint out(bl.x, bl.y, bl.z);
        out *= wInv;
        points[i] = out;
    }

    iter.setAllPositions(points);
    return MS::kSuccess;
}

// ================================================================
// getCoRData
// ================================================================
bool OptimizedSkinCluster::getCoRData(
    std::vector<mathcore::Vec3>& out) const {
    const short algorithm = MPlug(thisMObject(), aAlgorithmType).asShort();
    if (algorithm == 0 || algorithm == 6 || algorithm == 7) {
        const auto& cache = algorithm == 6 ? m_ddmV4 : m_paperCoR;
        if (!cache.valid || cache.weightVersion != m_currentWeightVersion) return false;
        out = cache.centers;
        return true;
    }
    if (!m_corCache.valid) return false;
    out = m_corCache.cors;
    return true;
}
