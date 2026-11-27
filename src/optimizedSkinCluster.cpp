#include "optimizedSkinCluster.h"
#include "algorithms.h"
#include "cor_curve.h"
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
#include <maya/MFnSkinCluster.h>
#include <maya/MDagPathArray.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MDoubleArray.h>
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
MObject OptimizedSkinCluster::aFitOffsets;
MObject OptimizedSkinCluster::aCorTriLayerExpand;
MObject OptimizedSkinCluster::aCorVertLayerExpand;
MObject OptimizedSkinCluster::aCorMosSecondaryWeight;
MObject OptimizedSkinCluster::aOcdTwistWeight;
MObject OptimizedSkinCluster::aOcdAvoidOverlap;
MObject OptimizedSkinCluster::aOcdCenterLimit;
MObject OptimizedSkinCluster::aOcdEnableCenterLimit;
MObject OptimizedSkinCluster::aOcdOuterGather;
MObject OptimizedSkinCluster::aOcdBendFlip;
MObject OptimizedSkinCluster::aOcdElbowGuide;
MObject OptimizedSkinCluster::aOcdCreaseFillet;
MObject OptimizedSkinCluster::aCorCurveGate;
MObject OptimizedSkinCluster::aCorCurveSigma;
MObject OptimizedSkinCluster::aCorCurveCrossOnly;
MObject OptimizedSkinCluster::aCorCurveAssign;
MObject OptimizedSkinCluster::aOcdCenterOffsets;
MObject OptimizedSkinCluster::aOcdHingeParent;
MObject OptimizedSkinCluster::aOcdHingeChild;
MObject OptimizedSkinCluster::aOcdHingeReport;
MObject OptimizedSkinCluster::aOcdGuideDump;
MObject OptimizedSkinCluster::aOcdDeltaDump;
MObject OptimizedSkinCluster::aOcdCenterLimitDump;
MObject OptimizedSkinCluster::aOcdCenterGateLow;
MObject OptimizedSkinCluster::aOcdCenterGateHigh;
MObject OptimizedSkinCluster::aOcdCenterSlideIn;
MObject OptimizedSkinCluster::aOcdCenterSlideOut;
MObject OptimizedSkinCluster::aOcdSlideInPrev;
MObject OptimizedSkinCluster::aOcdSlideInNext;
MObject OptimizedSkinCluster::aOcdSlideOutPrev;
MObject OptimizedSkinCluster::aOcdSlideOutNext;
MObject OptimizedSkinCluster::aOcdSlideOrthoInPrev;
MObject OptimizedSkinCluster::aOcdSlideOrthoInNext;
MObject OptimizedSkinCluster::aOcdSlideOrthoOutPrev;
MObject OptimizedSkinCluster::aOcdSlideOrthoOutNext;
MObject OptimizedSkinCluster::aOcdBulge;
MObject OptimizedSkinCluster::aCorSigma;
MObject OptimizedSkinCluster::aCorDebugLog;
MObject OptimizedSkinCluster::aDdmIterations;
MObject OptimizedSkinCluster::aDdmLambda;
MObject OptimizedSkinCluster::aDdmKappa;
MObject OptimizedSkinCluster::aDdmAlpha;
MObject OptimizedSkinCluster::aDdmDebugLog;
MObject OptimizedSkinCluster::aCorIncBudgetDiv;
MObject OptimizedSkinCluster::aDisplayCoR;

MObject OptimizedSkinCluster::aMosInfluenceRadius;
MObject OptimizedSkinCluster::aMosCurveMode;
MObject OptimizedSkinCluster::aMosShapeGuide;
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
    eAttr.addField("OCD", 3);
    eAttr.addField("MOS", 4);
    eAttr.addField("OCD + Corrected CoR", 5);
    eAttr.addField("DDM v4 (2019)", 6);
    eAttr.addField("CoR Fixed Center Fit", 8);
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

    // CoR Fixed Center Fit（alg 8）：偏移按绑定对象空间向量保存，
    // 由外部（例如参考实现的 NumPy 求解）写入。插件只做
    // center + offset 之后的一次 vertexTransform。
    {
        MFnTypedAttribute fitAttr;
        aFitOffsets = fitAttr.create("corFitOffsets", "cfo",
            MFnData::kDoubleArray, MObject::kNullObj, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        fitAttr.setStorable(true);
        fitAttr.setKeyable(false);
        fitAttr.setNiceNameOverride("CoR Fixed Center Fit Offsets (bind object space)");
        CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aFitOffsets));
        CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aFitOffsets, MPxSkinCluster::outputGeom));
    }

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

    // ---- OCD 铰链参数（OCD 与 OCD+Corrected CoR 共用） ----
    aOcdTwistWeight = nAttr.create("ocdTwistWeight", "octw",
        MFnNumericData::kDouble, 1.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Twist Weight");
    addAttribute(aOcdTwistWeight);
    attributeAffects(aOcdTwistWeight, MPxSkinCluster::outputGeom);

    aOcdAvoidOverlap = nAttr.create("ocdAvoidOverlap", "ocao",
        MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Legacy Inner Avoidance");
    addAttribute(aOcdAvoidOverlap);
    attributeAffects(aOcdAvoidOverlap, MPxSkinCluster::outputGeom);

    aOcdCenterLimit = nAttr.create("ocdCenterLimit", "occl",
        MFnNumericData::kDouble, 16.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.1);
    nAttr.setMax(256.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Corrected CoR Limit");
    addAttribute(aOcdCenterLimit);
    attributeAffects(aOcdCenterLimit, MPxSkinCluster::outputGeom);

    // 中心限幅开关：默认 true 保持旧场景行为。关闭后跳过 structuralCenters
    // 末尾的逐顶点长度截断，对齐参考实现的 center_limit=None。
    // 固定硬限额会把相邻顶点截断到不同程度，在低权重边界形成局部折角；
    // 关闭后可直接观察完整求解结果。
    aOcdEnableCenterLimit = nAttr.create("ocdEnableCenterLimit", "oclen",
        MFnNumericData::kBoolean, true, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Enable Corrected CoR Limit");
    addAttribute(aOcdEnableCenterLimit);
    attributeAffects(aOcdEnableCenterLimit, MPxSkinCluster::outputGeom);

    // 凸侧 gather 系数：参考实现的 0.85 在深弯时会把外轮廓拉瘪。
    // 0 表示外侧按刚体搬运，保留绑定网格原本饱满的外侧。
    aOcdOuterGather = nAttr.create("ocdOuterGather", "ocdog",
        MFnNumericData::kDouble, 0.85, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Outer Gather");
    addAttribute(aOcdOuterGather);
    attributeAffects(aOcdOuterGather, MPxSkinCluster::outputGeom);

    // 负弯时是否翻转弯面 180 度。参考实现翻转；关闭后 bind.y<0 的一侧
    // 在负弯时仍按外侧处理，用于对比外轮廓是否因此更饱满。
    aOcdBendFlip = nAttr.create("ocdBendFlip", "ocdbfl",
        MFnNumericData::kBoolean, true, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Flip Plane On Negative Bend");
    addAttribute(aOcdBendFlip);
    attributeAffects(aOcdBendFlip, MPxSkinCluster::outputGeom);

    // 算法 5 的 guide 来源：true 用 Elbow::point，false 用 hingeDisplacement。
    // 默认 false：Elbow::point 的目标在「未旋转的肢体局部系」，而 originalLocal
    // 已经是随子骨旋转后的 CoR 输出，两者不在同一坐标系，直接相减会得到
    // 量级 3~4 的 guide（实测正侧 guide.y 达 3.88），求解后顶点被推离目标。
    // 保留开关方便继续对齐，但默认走已验证一致的 hingeDisplacement。
    aOcdElbowGuide = nAttr.create("ocdElbowGuide", "ocdeg",
        MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Guide From Elbow Model");
    addAttribute(aOcdElbowGuide);
    attributeAffects(aOcdElbowGuide, MPxSkinCluster::outputGeom);

    // station=0 折角的圆角窗口（肢体半径倍数）。默认 0 = 关闭，保持参考实现
    // 的硬旋转切换。参考实现在这里有一个 15.5° 的折角：458 点低密度网格的
    // 环距 0.667 跨过了折角，看起来圆滑；16080 点环距 0.060 会如实采出来。
    // 试出来的平衡点是 0.15：高密度网格内轨三点转角 164.9° -> 49.5°，
    // 而 458 点造型只动 0.0389（不到半径的 4%）。
    aOcdCreaseFillet = nAttr.create("ocdCreaseFillet", "occf",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(3.0);
    nAttr.setSoftMax(0.3);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("OCD Crease Fillet (body radii)");
    addAttribute(aOcdCreaseFillet);
    attributeAffects(aOcdCreaseFillet, MPxSkinCluster::outputGeom);

    // ---- CoR 曲线引导（方向 A）----
    // 只改 CoR 基础中心积分时的邻域权重：
    //     s*_ij = s^W_ij . g_curve(i, sample)
    //     g = exp( -(d / radius)^2 / (2 sigma^2) ),  d = 采样点沿曲线的弧长 - 查询顶点
    // 关闭（默认 0）时 computeCenters 走原来的分组路径，逐位不变。
    // 首版用「骨骼轴上的归一化位置」当曲线参数：与绑定 axis 对齐，无需额外
    // 曲线数据即可先验证 A/B；后续接入真实 Influence Curve 时替换取参函数。
    aCorCurveGate = nAttr.create("corCurveGate", "ccg",
        MFnNumericData::kDouble, 0.0, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(0.0);
    nAttr.setMax(1.0);
    nAttr.setSoftMax(1.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Curve Gate (0=off, 1=on)");
    addAttribute(aCorCurveGate);
    attributeAffects(aCorCurveGate, MPxSkinCluster::outputGeom);

    aCorCurveSigma = nAttr.create("corCurveSigma", "ccs",
        MFnNumericData::kDouble, 0.1, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setMin(1e-4);
    nAttr.setMax(1.0e3);
    nAttr.setSoftMax(2.0);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Curve Gate Sigma (body radii)");
    addAttribute(aCorCurveSigma);
    attributeAffects(aCorCurveSigma, MPxSkinCluster::outputGeom);

    // B 组对照：同曲线因子恒为 1，只衰减跨曲线贡献。
    // 0 = C 组（归属 + 同曲线弧长高斯），1 = B 组（只保留归属）。
    aCorCurveCrossOnly = nAttr.create("corCurveCrossOnly", "ccco",
        MFnNumericData::kBoolean, false, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("CoR Curve Gate: cross-curve only");
    addAttribute(aCorCurveCrossOnly);
    attributeAffects(aCorCurveCrossOnly, MPxSkinCluster::outputGeom);

    // 手工曲线归属：长度必须为 0（自动）或 3 以外的顶点数，值为 curveId。
    // 用于「权重配方相同、归属不同」的场景 —— 骨骼权重无法区分两块表面时，
    // 只能靠这个显式指定。空数组 = 按骨骼链自动推导。
    MFnTypedAttribute assignAttr;
    aCorCurveAssign = assignAttr.create("corCurveAssign", "ccas",
        MFnData::kDoubleArray, MObject::kNullObj, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    assignAttr.setStorable(true);
    assignAttr.setNiceNameOverride("CoR Curve Assign (curveId per vertex)");
    addAttribute(aCorCurveAssign);
    attributeAffects(aCorCurveAssign, MPxSkinCluster::outputGeom);

    MFnTypedAttribute offsetAttr;
    aOcdCenterOffsets = offsetAttr.create("ocdCenterOffsets", "ocdcf",
        MFnData::kDoubleArray, MObject::kNullObj, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    offsetAttr.setStorable(true);
    offsetAttr.setKeyable(false);
    offsetAttr.setNiceNameOverride("OCD Corrected CoR Offsets (debug)");
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aOcdCenterOffsets));
    CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aOcdCenterOffsets, MPxSkinCluster::outputGeom));

    MObject* const hingeAttrs[2] = {&aOcdHingeParent, &aOcdHingeChild};
    const char* const hingeNames[2] = {"ocdHingeParent", "ocdHingeChild"};
    const char* const hingeShorts[2] = {"ochp", "ochc"};
    const char* const hingeLabels[2] = {"OCD Hinge Parent (override)",
                                        "OCD Hinge Child (override)"};
    for (int hinge = 0; hinge < 2; ++hinge) {
        MFnTypedAttribute hingeAttr;
        *hingeAttrs[hinge] = hingeAttr.create(hingeNames[hinge], hingeShorts[hinge],
            MFnData::kString, MObject::kNullObj, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        hingeAttr.setStorable(true);
        hingeAttr.setNiceNameOverride(hingeLabels[hinge]);
        CHECK_MSTATUS_AND_RETURN_IT(addAttribute(*hingeAttrs[hinge]));
        CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(*hingeAttrs[hinge],
            MPxSkinCluster::outputGeom));
    }

    {
        MFnTypedAttribute reportAttr;
        aOcdHingeReport = reportAttr.create("ocdHingeReport", "ochr",
            MFnData::kString, MObject::kNullObj, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        reportAttr.setStorable(false);
        reportAttr.setKeyable(false);
        reportAttr.setWritable(false);
        reportAttr.setNiceNameOverride("OCD Hinge Report (debug)");
        CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aOcdHingeReport));
    }

    // 诊断导出：guide / delta / 实际使用的 centerLimit。
    // 用普通可写 doubleArray（非 output），由 deform 里回写；
    // 不参与 attributeAffects，也不影响驱动。
    // 低权重门控（实验性）：两个可写 double，默认 0/0 = 关闭。
    {
        MObject* const gates[2] = {&aOcdCenterGateLow, &aOcdCenterGateHigh};
        const char* const gnames[2] = {"ocdCenterGateLow", "ocdCenterGateHigh"};
        const char* const gshorts[2] = {"ocgl", "ocgh"};
        const char* const glabels[2] = {"OCD Center Gate Low (w1)",
                                        "OCD Center Gate High (w1)"};
        for (int g = 0; g < 2; ++g) {
            MFnNumericAttribute gAttr;
            *gates[g] = gAttr.create(gnames[g], gshorts[g],
                MFnNumericData::kDouble, 0.0, &status);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            gAttr.setStorable(true);
            gAttr.setKeyable(true);
            gAttr.setNiceNameOverride(glabels[g]);
            CHECK_MSTATUS_AND_RETURN_IT(addAttribute(*gates[g]));
            CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(*gates[g],
                MPxSkinCluster::outputGeom));
        }
    }

    {
        MObject* const dumps[3] = {&aOcdGuideDump, &aOcdDeltaDump,
                                   &aOcdCenterLimitDump};
        const char* const names[3] = {"ocdGuideDump", "ocdDeltaDump",
                                      "ocdCenterLimitDump"};
        // 注意："occl" 已被 ocdCenterLimit 占用，不能重用短名。
        const char* const shorts[3] = {"ocgd", "ocdd", "occlu"};
        const char* const labels[3] = {"OCD Guide Dump (debug)",
                                       "OCD Delta Dump (debug)",
                                       "OCD Center Limit Used (debug)"};
        for (int dump = 0; dump < 3; ++dump) {
            MFnTypedAttribute dumpAttr;
            *dumps[dump] = dumpAttr.create(names[dump], shorts[dump],
                MFnData::kDoubleArray, MObject::kNullObj, &status);
            CHECK_MSTATUS_AND_RETURN_IT(status);
            // storable 必须为 true：false 时 Maya 不保留我们在 deform() 里
            // 用 setMObject 写进去的值，getAttr 会一直读到空数组。
            dumpAttr.setStorable(true);
            dumpAttr.setReadable(true);
            dumpAttr.setWritable(true);
            dumpAttr.setKeyable(false);
            dumpAttr.setNiceNameOverride(labels[dump]);
            CHECK_MSTATUS_AND_RETURN_IT(addAttribute(*dumps[dump]));
            CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(*dumps[dump],
                MPxSkinCluster::outputGeom));
        }
    }

    // ---- OCD 11 个补偿通道（对应参考实现的 Channel Box 滑块） ----
    // 单位与参考实现一致；0 表示不施加该次级编辑。
    const struct {
        MObject* attr;
        const char* name;
        const char* shortName;
        const char* label;
        double defaultValue;
        double minimum;
        double maximum;
    } ocdChannels[ocdChannelCount] = {
        {&aOcdCenterSlideIn, "ocdCenterSlideIn", "occi", "OCD Center Slide In", 0.0, -1.0, 1.0},
        {&aOcdCenterSlideOut, "ocdCenterSlideOut", "occo", "OCD Center Slide Out", 0.0, -1.0, 1.0},
        {&aOcdSlideInPrev, "ocdSlideInPrev", "ocsip", "OCD Slide In Prev", 0.0, -1.0, 1.0},
        {&aOcdSlideInNext, "ocdSlideInNext", "ocsin", "OCD Slide In Next", 0.0, -1.0, 1.0},
        {&aOcdSlideOutPrev, "ocdSlideOutPrev", "ocsop", "OCD Slide Out Prev", 0.0, -1.0, 1.0},
        {&aOcdSlideOutNext, "ocdSlideOutNext", "ocson", "OCD Slide Out Next", 0.0, -1.0, 1.0},
        {&aOcdSlideOrthoInPrev, "ocdSlideOrthoInPrev", "ocnip", "OCD Slide Ortho In Prev", 0.0, -1.0, 1.0},
        {&aOcdSlideOrthoInNext, "ocdSlideOrthoInNext", "ocnin", "OCD Slide Ortho In Next", 0.0, -1.0, 1.0},
        {&aOcdSlideOrthoOutPrev, "ocdSlideOrthoOutPrev", "ocnop", "OCD Slide Ortho Out Prev", 0.0, -1.0, 1.0},
        {&aOcdSlideOrthoOutNext, "ocdSlideOrthoOutNext", "ocnon", "OCD Slide Ortho Out Next", 0.0, -1.0, 1.0},
        {&aOcdBulge, "ocdBulge", "ocbl", "OCD Bulge", 0.0, -1.0, 1.0},
    };
    for (int index = 0; index < ocdChannelCount; ++index) {
        const auto& channel = ocdChannels[index];
        *channel.attr = nAttr.create(channel.name, channel.shortName,
            MFnNumericData::kDouble, channel.defaultValue, &status);
        CHECK_MSTATUS_AND_RETURN_IT(status);
        nAttr.setMin(channel.minimum);
        nAttr.setMax(channel.maximum);
        nAttr.setKeyable(true);
        nAttr.setStorable(true);
        nAttr.setNiceNameOverride(channel.label);
        addAttribute(*channel.attr);
        attributeAffects(*channel.attr, MPxSkinCluster::outputGeom);
    }

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

    aMosCurveMode = eAttr.create("mosCurveMode", "mosCm", MosParams::arcSegments, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    eAttr.addField("Single Long Curve", 0);
    eAttr.addField("Per-Joint Curves", 1);
    eAttr.addField("OCD Arc Segments", MosParams::arcSegments);
    eAttr.setKeyable(true);
    eAttr.setStorable(true);
    eAttr.setNiceNameOverride("MOS Curve Mode");
    addAttribute(aMosCurveMode);
    attributeAffects(aMosCurveMode, MPxSkinCluster::outputGeom);

    aMosShapeGuide = nAttr.create("mosShapeGuide", "mosSg", MFnNumericData::kBoolean, true, &status);
    CHECK_MSTATUS_AND_RETURN_IT(status);
    nAttr.setKeyable(false);
    nAttr.setChannelBox(true);
    nAttr.setStorable(true);
    nAttr.setNiceNameOverride("MOS Two-Bone Shape Guide");
    CHECK_MSTATUS_AND_RETURN_IT(addAttribute(aMosShapeGuide));
    CHECK_MSTATUS_AND_RETURN_IT(attributeAffects(aMosShapeGuide, MPxSkinCluster::outputGeom));

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

    m_boneInfo.boneParent.assign(maxIdx + 1, -1);
    m_boneInfo.boneParentSteps.assign(maxIdx + 1, 1);
    for (unsigned int i = 0; i < nB; ++i) {
        // 沿真实 DAG 一直向上走：中间可能夹着没有蒙皮的关节。
        // 之前遇到未蒙皮父骨就 break，使该子骨被记为 depth=0，
        // 导致「上臂 → 未蒙皮中间关节 → 前臂」里前臂直接失去候选资格。
        int skinnedAncestors = 0;
        int nearestSkinned = -1;
        int stepsToNearest = 0;
        int steps = 0;
        MObject curJnt = joints[i];
        while (steps < 512) {
            MFnDagNode dag(curJnt);
            MObject parent = dag.parent(0);
            if (parent.isNull() || !parent.hasFn(MFn::kJoint)) break;
            ++steps;
            int foundIndex = -1;
            for (unsigned int j = 0; j < nB; ++j) {
                if (joints[j] == parent) { foundIndex = static_cast<int>(logicalIdx[j]); break; }
            }
            if (foundIndex >= 0) {
                ++skinnedAncestors;
                if (nearestSkinned < 0) {
                    nearestSkinned = foundIndex;
                    stepsToNearest = steps;
                }
            }
            curJnt = parent;
        }
        m_boneInfo.depths[logicalIdx[i]] = skinnedAncestors;
        m_boneInfo.boneParent[logicalIdx[i]] = nearestSkinned;
        m_boneInfo.boneParentSteps[logicalIdx[i]] = stepsToNearest;
    }

    // 该骨骼是否还有子骨骼：用来区分「延续链条的子骨」与
    // 「共点的 helper」。同位置的 helper 没有子骨，不应被选为主铰链的 child。
    m_boneInfo.boneHasJointChild.assign(maxIdx + 1, 0);
    for (unsigned int i = 0; i < nB; ++i) {
        MFnDagNode dag(joints[i]);
        const unsigned int childCount = dag.childCount();
        for (unsigned int child = 0; child < childCount; ++child) {
            if (!dag.child(child).hasFn(MFn::kJoint)) continue;
            m_boneInfo.boneHasJointChild[logicalIdx[i]] = 1;
            break;
        }
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
    if (algo == 8) return deformCoR(block, iter, worldMat, multiIndex, CenteredMode::FixedFit);

    unsigned int nV = iter.count();
    collectBoneInfo(block);
    rebuildVtxCache(block, nV);

    switch (algo) {
    case 1: return deformDLB(block, iter, worldMat, multiIndex,
        block.inputValue(aDlbUseOpenCL).asBool());
    case 2: return deformDQSComp(block, iter, worldMat, multiIndex);
    case 3: return deformOCD(block, iter, worldMat, multiIndex, false);
    case 4: return deformMOS(block, iter, worldMat, multiIndex);
    case 5: return deformOCD(block, iter, worldMat, multiIndex, true);
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

    // 方向 A 用：按紧凑 influence 顺序取出绑定关节的位置，作为曲线折线。
    // 第一版把整条关节序列当成一条链（curveId = 0）。绑定期执行，
    // 因此关节的世界位置就是绑定位置。
    std::vector<mathcore::Vec3> jointPositions;
    std::vector<unsigned int> jointChain;
    if (next.influenceIds.size() >= 2) {
        MDagPathArray influencePaths;
        MFnSkinCluster skinFn(thisMObject(), &status);
        if (status && skinFn.influenceObjects(influencePaths, &status) && status) {
            jointPositions.reserve(influencePaths.length());
            for (unsigned int k = 0; k < influencePaths.length(); ++k) {
                const MMatrix world = influencePaths[k].inclusiveMatrix();
                jointPositions.push_back({world[3][0], world[3][1], world[3][2]});
                jointChain.push_back(k);
            }
        } else {
            status = MS::kSuccess;
            jointPositions.clear();
            jointChain.clear();
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

        // ---- 方向 A：曲线引导的邻域权重 ----
        // 开关关闭时不构造 gate，computeCenters 走原分组路径，逐位不变。
        const double curveGateOn = block.inputValue(aCorCurveGate, &status).asDouble();
        if (!status) return status;
        const double curveSigma = block.inputValue(aCorCurveSigma, &status).asDouble();
        if (!status) return status;
        const bool curveCrossOnly = block.inputValue(aCorCurveCrossOnly, &status).asBool();
        if (!status) return status;

        cor_paper::CurveGate gate;
        std::vector<cor_paper::CurveVertex> curveVertices;
        if (curveGateOn >= 0.5 && curveSigma > 0.0) {
            // 曲线归属的两种来源：
            //   1) 手工指定 corCurveAssign（长度=顶点数，值为 curveId）
            //      —— 用于「权重配方相同、归属不同」的场景，骨骼权重无法区分
            //         两块表面时必须靠这个。
            //   2) 自动：按骨骼链推导（整条关节序列视为一条曲线）。
            std::vector<int> manualAssign;
            {
                const MObject assignData = block.inputValue(aCorCurveAssign, &status).data();
                if (!status) return status;
                if (!assignData.isNull()) {
                    MFnDoubleArrayData arrayData(assignData, &status);
                    if (status) {
                        const MDoubleArray values = arrayData.array(&status);
                        if (status && values.length() == restPoints.size()) {
                            manualAssign.resize(values.length());
                            for (unsigned int i = 0; i < values.length(); ++i)
                                manualAssign[i] = static_cast<int>(values[i]);
                        }
                    }
                    status = MS::kSuccess;
                }
            }

            std::vector<cor_curve::Chain> chains;
            std::vector<int> chainSourceId;   // 每条链对应的原始 curveId
            if (!manualAssign.empty()) {
                int maxId = 0;
                for (int id : manualAssign) maxId = std::max(maxId, id);
                for (int id = 0; id <= maxId; ++id) {
                    std::vector<mathcore::Vec3> pts;
                    for (size_t v = 0; v < restPoints.size(); ++v) {
                        if (manualAssign[v] == id) pts.push_back(restPoints[v]);
                    }
                    if (pts.size() < 2) continue;
                    mathcore::Vec3 lo = pts.front(), hi = pts.front();
                    for (const auto& p : pts) {
                        lo.x = std::min(lo.x, p.x); hi.x = std::max(hi.x, p.x);
                        lo.y = std::min(lo.y, p.y); hi.y = std::max(hi.y, p.y);
                        lo.z = std::min(lo.z, p.z); hi.z = std::max(hi.z, p.z);
                    }
                    // 主方向取跨度最大的轴；折线两端点沿该轴。
                    const double dx = hi.x - lo.x, dy = hi.y - lo.y, dz = hi.z - lo.z;
                    mathcore::Vec3 a = lo, b = hi;
                    if (dx >= dy && dx >= dz) {
                        a.y = (lo.y + hi.y) * 0.5; a.z = (lo.z + hi.z) * 0.5;
                        b.y = a.y; b.z = a.z;
                    } else if (dy >= dz) {
                        a.x = (lo.x + hi.x) * 0.5; a.z = (lo.z + hi.z) * 0.5;
                        b.x = a.x; b.z = a.z;
                    } else {
                        a.x = (lo.x + hi.x) * 0.5; a.y = (lo.y + hi.y) * 0.5;
                        b.x = a.x; b.y = a.y;
                    }
                    cor_curve::Chain chain;
                    chain.points = {a, b};
                    cor_curve::buildChainGeometry(chain);
                    chains.push_back(std::move(chain));
                    chainSourceId.push_back(id);
                }
            } else {
                // 自动模式：整条关节序列一条曲线。
                cor_curve::Chain chain;
                chain.bones = jointChain;
                chain.points = jointPositions;
                if (chain.points.size() >= 2) {
                    cor_curve::buildChainGeometry(chain);
                    chains.push_back(std::move(chain));
                    chainSourceId.push_back(-1);
                }
            }

            if (!chains.empty()) {
                const auto bindCurves = cor_curve::bindVertices(chains, restPoints,
                    next.weights);
                const double scale = std::max(bindCurves.meanChainLength, 1e-12);
                const double invTwoSigma = 1.0 / (2.0 * curveSigma * curveSigma);
                const bool crossOnly = curveCrossOnly;

                struct QueryInfo {
                    std::vector<double> membership;
                    std::vector<double> arc;
                };
                std::vector<QueryInfo> queryInfo(restPoints.size());
                curveVertices.resize(restPoints.size());
                for (size_t i = 0; i < restPoints.size(); ++i) {
                    if (!manualAssign.empty()) {
                        // 手工模式：one-hot 归属，弧长用该顶点在自己链上的投影。
                        std::vector<double> mem(chains.size(), 0.0);
                        std::vector<double> arc(chains.size(), 0.0);
                        for (size_t c = 0; c < chains.size(); ++c) {
                            if (chainSourceId[c] != manualAssign[i]) continue;
                            mem[c] = 1.0;
                            arc[c] = cor_curve::detail::projectArc(
                                chains[c], restPoints[i]);
                        }
                        queryInfo[i].membership = mem;
                        queryInfo[i].arc = arc;
                        curveVertices[i].membership = mem;
                        curveVertices[i].arc = arc;
                    } else {
                        queryInfo[i].membership = bindCurves.vertices[i].membership;
                        queryInfo[i].arc = bindCurves.vertices[i].arcLength;
                        curveVertices[i].membership = bindCurves.vertices[i].membership;
                        curveVertices[i].arc = bindCurves.vertices[i].arcLength;
                    }
                }

                gate = [queryInfo, scale, invTwoSigma, crossOnly](
                           unsigned int query, const cor_paper::CurveSample& sample) -> double {
                    if (query >= queryInfo.size()) return 1.0;
                    if (!sample.membership || !sample.arc) return 1.0;
                    const auto& qm = queryInfo[query].membership;
                    const auto& qa = queryInfo[query].arc;
                    const auto& sm = *sample.membership;
                    const auto& sa = *sample.arc;
                    const size_t n = std::min(qm.size(), sm.size());
                    if (n == 0) return 1.0;
                    // 跨曲线归属：用两侧归属权重的重叠量作为同属程度。
                    double overlap = 0.0;
                    double total = 0.0;
                    for (size_t k = 0; k < n; ++k) {
                        overlap += std::min(qm[k], sm[k]);
                        total += qm[k];
                    }
                    // 归一化到 [0,1]：完全同属 -> 1。
                    const double sameCurve = total > 1e-12
                        ? std::max(0.0, std::min(1.0, overlap / total)) : 1.0;
                    double g = 1.0;
                    if (crossOnly) {
                        // B 组：同曲线 g=1，只衰减跨曲线贡献。
                        g = sameCurve;
                    } else {
                        // C 组：在 B 之上再叠加同曲线弧长高斯。
                        double arc = 0.0;
                        double weightSum = 0.0;
                        for (size_t k = 0; k < n; ++k) {
                            if (qm[k] <= 0.0) continue;
                            arc += qm[k] * (sa[k] - qa[k]);
                            weightSum += qm[k];
                        }
                        const double d = weightSum > 1e-12
                            ? std::fabs(arc / weightSum) / scale : 0.0;
                        g = sameCurve * std::exp(-(d * d) * invTwoSigma);
                    }
                    return g;
                };

                if (crossOnly) {
                    // B 组只看归属，不需要弧长参与，但仍要把归属传进去。
                }
            }
        }

        next.centers = cor_paper::computeCenters(restPoints, next.weights, triangles,
            sigma, gate, gate ? &curveVertices : nullptr);
        next.curveGate = curveGateOn >= 0.5 ? 1.0 : 0.0;
        next.curveSigma = curveSigma;
        next.curveCrossOnly = curveCrossOnly ? 1.0 : 0.0;
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
    const bool debug = optSkinDebugLog() ||
        block.inputValue(useDdm ? aDdmDebugLog : aCorDebugLog).asBool();
    const MString prefix = debug
        ? MString(useDdm ? "[DDM v4] " : "[CoR] ") + MFnDependencyNode(thisMObject()).name() + ": "
        : MString();
    const auto reportFailure = [&](MStatus status, const char* stage) {
        if (!status && debug) MGlobal::displayWarning(prefix + stage + " - " + status.errorString());
        return status;
    };

    const double envelope = readEnvelopeFromNode();
    MStatus status;

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
    // 曲线引导开关/尺度变化也必须使基础中心失效，否则 A/B 会读到上一组缓存。
    double curveGateNow = 0.0;
    double curveSigmaNow = 0.0;
    if (!useDdm) {
        curveGateNow = block.inputValue(aCorCurveGate, &status).asDouble();
        if (!status) return reportFailure(status, "read corCurveGate");
        curveSigmaNow = block.inputValue(aCorCurveSigma, &status).asDouble();
        if (!status) return reportFailure(status, "read corCurveSigma");
    }
    if (forceFull || !cache.valid || cache.geometryIndex != multiIndex ||
        cache.weightVersion != m_currentWeightVersion ||
        (useDdm ? !(cache.ddmSettings == ddmSettings)
                : (cache.sigma != sigma ||
                   cache.curveGate != curveGateNow ||
                   cache.curveSigma != curveSigmaNow ||
                   cache.curveCrossOnly !=
                       (block.inputValue(aCorCurveCrossOnly, &status).asBool()
                            ? 1.0 : 0.0)))) {
        if (!status) return reportFailure(status, "read corCurveCrossOnly");
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
    // 固定中心拟合的偏移按绑定对象空间保存，中心也在同一空间，直接相加。
    // 长度必须为 0（不启用）或 3*顶点数，否则报错而不是静默忽略，
    // 以免外部求解器写入的数组长度不对时得到看似正常的错误结果。
    std::vector<mathcore::Vec3> fittedOffsets;
    if (mode == CenteredMode::FixedFit) {
        const MObject data = block.inputValue(aFitOffsets, &status).data();
        if (!status) return reportFailure(status, "read corFitOffsets");
        if (!data.isNull()) {
            MFnDoubleArrayData arrayData(data, &status);
            if (!status) return reportFailure(status, "corFitOffsets data");
            const MDoubleArray values = arrayData.array(&status);
            if (!status) return reportFailure(status, "corFitOffsets array");
            if (values.length() != 0 &&
                values.length() != 3 * cache.centers.size()) {
                return reportFailure(MS::kInvalidParameter, "corFitOffsets length");
            }
            fittedOffsets.resize(values.length() / 3);
            for (size_t i = 0; i < fittedOffsets.size(); ++i)
                fittedOffsets[i] = {values[3 * i], values[3 * i + 1], values[3 * i + 2]};
        }
    }
    const auto deformPoint = [&](const MPoint& p, unsigned int vertex) {
        if (vertex >= cache.weights.size() || cache.weights[vertex].empty()) return p;
        const MPoint restWorld = p * worldMat;
        auto center = cache.centers[vertex];
        if (vertex < fittedOffsets.size())
            center = mathcore::v3_add(center, fittedOffsets[vertex]);
        const MPoint centerWorld = MPoint(center.x, center.y, center.z) * worldMat;
        const auto& rotationWeights = cache.weights[vertex];
        const auto& translationWeights = cache.translationWeights.empty()
            ? rotationWeights : cache.translationWeights[vertex];
        const auto transform = cor_paper::vertexTransform(
            {centerWorld.x, centerWorld.y, centerWorld.z}, rotationWeights, translationWeights, bindToPose);
        const mathcore::Vec3 original{restWorld.x, restWorld.y, restWorld.z};
        const auto posed = mathcore::apply_transform(transform, original);
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
    const bool shapeGuide = direct && block.inputValue(aMosShapeGuide).asBool();
    const bool arcRequested = direct && params.curveMode == MosParams::arcSegments;
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
    if (arcRequested) {
        std::set<unsigned int> weighted;
        for (const auto& weights : m_vtxCache.vertices)
            for (const auto& weight : weights)
                if (weight.second > 0.0f) weighted.insert(weight.first);
        if (weighted.size() == 2) {
            for (size_t i = logicalIdx.size(); i-- > 0;) {
                if (weighted.count(logicalIdx[i])) continue;
                logicalIdx.erase(logicalIdx.begin() + i);
                joints.erase(joints.begin() + i);
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

    m_mosCache.arcVertices.clear();
    m_mosCache.arcBefore = m_mosCache.arcAfter = m_mosCache.arcRadius = 0.0;
    if (arcRequested && joints.size() == 2 && n_bones == 2 && bone_len[0] > 1e-8) {
        const auto pivot = bind_jp[chain[1]];
        const auto axis = bone_dir[0];
        for (unsigned int i = 0; i < numVertices; ++i) {
            if (m_vtxCache.vertices[i].empty()) continue;
            const auto& point = points[i];
            const double axial = mathcore::v3_dot(
                mathcore::v3_sub({point.x, point.y, point.z}, pivot), axis);
            m_mosCache.arcBefore = std::max(m_mosCache.arcBefore, -axial);
            m_mosCache.arcAfter = std::max(m_mosCache.arcAfter, axial);
        }
        if (m_mosCache.arcBefore > 1e-8 && m_mosCache.arcAfter > 1e-8) {
            m_mosCache.arcVertices.resize(numVertices);
            const double radiusWindow = 0.5 * std::min(m_mosCache.arcBefore, m_mosCache.arcAfter);
            for (unsigned int i = 0; i < numVertices; ++i) {
                if (m_vtxCache.vertices[i].empty()) continue;
                const auto& point = points[i];
                m_mosCache.arcVertices[i] = ocd_arc::bindVertex({point.x, point.y, point.z},
                    pivot, axis, m_mosCache.arcBefore, m_mosCache.arcAfter);
                const double axial = m_mosCache.arcVertices[i].fraction *
                    (m_mosCache.arcBefore + m_mosCache.arcAfter) - m_mosCache.arcBefore;
                if (std::abs(axial) <= radiusWindow)
                    m_mosCache.arcRadius = std::max(m_mosCache.arcRadius,
                        mathcore::v3_len(m_mosCache.arcVertices[i].radial));
                const auto oppositeBone = axial < 0.0 ? chain[1] : chain[0];
                double opposite = 0.0, total = 0.0;
                for (const auto& weight : m_vtxCache.vertices[i]) {
                    total += weight.second;
                    if (weight.first == oppositeBone) opposite += weight.second;
                }
                if (total > 0.0) m_mosCache.arcVertices[i].oppositeWeight = opposite / total;
            }
        }
    }

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

    const bool sharedGuide = shapeGuide && n_bones == 2;
    if (params.curveMode == 0 || sharedGuide || arcRequested) {
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

            if (direct && (params.curveMode == 0 || sharedGuide || arcRequested)) {
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
    m_mosCache.shapeGuide = shapeGuide;
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
        if (arcRequested) {
            if (m_mosCache.arcVertices.empty())
                MGlobal::displayWarning("[OCD Arc] A connected two-bone bind with weighted vertices on both sides is required; using MOS cubic.");
            else
                MGlobal::displayInfo("[OCD Arc] Bound local two-bone offset curves with LBS outside the bend.");
        }
    }
    return true;
}

// ================================================================
// deformOCD  --  OCD 铰链算法（algorithmType = 3 / 5）
//
// 数学来源：scripts/ocd_elbow_reference.py 的 Elbow / hinge_section /
// hinge_displacement / structural_centers。移植在 ocd_hinge.cpp 与
// ocd_centers.cpp，逐点比对参考实现的误差 < 1e-11。
//
// 算法选择：
//   · 3 = OCD           : 曲线截面直接给出目标形状
//   · 5 = OCD + Corrected CoR : 把 OCD 位移当作 guide，用绑定网格边向量
//                         求解逐顶点 CoR 中心修正，再由 CoR 求值
//
// 骨骼链：沿 influence 的层级找出唯一一条「父子」对；超过一对时回退 DLB，
// 因为参考实现只定义了两段肢体的铰链。
// ================================================================
MStatus OptimizedSkinCluster::deformOCD(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex, bool correctCenters) {

    using namespace mathcore;
    const double envelope = readEnvelopeFromNode();
    const bool debug = optSkinDebugLog();
    MStatus debugStatus;
    const bool reportHinge = debug
        || block.inputValue(aCorDebugLog, &debugStatus).asBool();
    MStatus status;

    std::vector<RigidTransform> rest, Cs;
    std::vector<Quat> poseQ;
    if (!readBoneTransforms(block, rest, Cs, poseQ)) return MS::kSuccess;

    collectBoneInfo(block);
    const unsigned int nV = iter.count();
    rebuildVtxCache(block, nV);
    if (nV == 0) return MS::kSuccess;
    if (m_vtxCache.vertices.size() < nV) return MS::kSuccess;
    if (m_boneInfo.jointPosRest.size() < rest.size() || m_boneInfo.depths.size() < rest.size())
        return deformDLB(block, iter, worldMat, multiIndex);

    const auto& pivots = m_boneInfo.jointPosRest;
    const auto& depths = m_boneInfo.depths;

    // ---- 1. 识别主铰链与（可选的）辅助骨 ----
    // 参考实现的验证模型是 prev→elbow 两段肢体，另挂一根 helper 骨
    // 给过渡区保留次级影响。这里取「父子对里权重和最大」的一对作为主铰链，
    // 其余有权重的骨骼作为辅助骨。超过一根辅助骨时
    // structural_centers 的三列权重表达不了，回退 DLB。
    std::vector<double> boneWeight(depths.size(), 0.0);
    for (const auto& vertex : m_vtxCache.vertices)
        for (const auto& kv : vertex)
            if (kv.first < boneWeight.size()) boneWeight[kv.first] += kv.second;

    const auto& dagParents = m_boneInfo.boneParent;
    int parentBone = -1, childBone = -1;
    double bestPairScore = -1.0;
    for (size_t child = 0; child < depths.size() && child < rest.size(); ++child) {
        if (depths[child] == 0) continue;
        int best = -1;
        // 优先用真实 DAG 父骨；只有查不到时才退回「层数+距离」猜测。
        if (child < dagParents.size()) best = dagParents[child];
        if (best < 0 || static_cast<size_t>(best) >= rest.size()) {
            best = -1;
            double bestDistance = 1e300;
            for (size_t candidate = 0; candidate < depths.size() && candidate < rest.size(); ++candidate) {
                if (depths[candidate] != depths[child] - 1) continue;
                const double d = v3_len(v3_sub(pivots[child], pivots[candidate]));
                if (d < bestDistance) { bestDistance = d; best = static_cast<int>(candidate); }
            }
        }
        if (best < 0) continue;
        // 「延续链条的子骨」优先于「共点的 helper」。
        // 否则只要 helper 累计权重更大，就会把它选成主铰链，
        // 而求解器仍会输出平滑网格，只是跟错了弯曲角。
        const bool continuesChain = child < m_boneInfo.boneHasJointChild.size()
            && m_boneInfo.boneHasJointChild[child] != 0;
        const double pairScore = (continuesChain ? 1.0e6 : 0.0)
            + std::min(boneWeight[best], boneWeight[child]);
        if (pairScore > bestPairScore) {
            bestPairScore = pairScore;
            parentBone = best;
            childBone = static_cast<int>(child);
        }
    }
    if (parentBone < 0 || childBone < 0) {
        if (debug) MGlobal::displayWarning("[OCD] 未找到父子骨骼对，回退 DLB。");
        return deformDLB(block, iter, worldMat, multiIndex);
    }

    // 显式指定主铰链（空字符串 = 自动）。真实角色上如果自动选择有歧义，
    // 用 ocdHingeParent / ocdHingeChild 直接锁定，不靠启发式。
    {
        const auto resolveBoneByName = [&](const MString& name) -> int {
            if (name.length() == 0) return -1;
            const MPlug matrixPlug(thisMObject(), MPxSkinCluster::matrix);
            for (unsigned int entry = 0; entry < matrixPlug.numConnectedElements(); ++entry) {
                const auto element = matrixPlug.connectionByPhysicalIndex(entry);
                MPlugArray sources;
                if (!element.connectedTo(sources, true, false) || sources.length() != 1) continue;
                MFnDagNode node(sources[0].node());
                if (node.name() == name) return static_cast<int>(element.logicalIndex());
            }
            return -1;
        };
        MStatus nameStatus;
        const MString parentName = block.inputValue(aOcdHingeParent, &nameStatus).asString();
        const MString childName = nameStatus
            ? block.inputValue(aOcdHingeChild, &nameStatus).asString() : MString();
        if (nameStatus && parentName.length() && childName.length()) {
            const int overrideParent = resolveBoneByName(parentName);
            const int overrideChild = resolveBoneByName(childName);
            if (overrideParent >= 0 && overrideChild >= 0 && overrideParent != overrideChild) {
                parentBone = overrideParent;
                childBone = overrideChild;
                if (debug) MGlobal::displayInfo(MString("[OCD] hinge overridden by attributes: ")
                    + parentName + " -> " + childName);
            }
            else if (debug && optSkinDebugLog()) {
                MGlobal::displayWarning("[OCD] ocdHingeParent/ocdHingeChild 无法解析，回退自动选择。");
            }
        }
    }

    // 辅助骨：除主铰链两端外仍有显著权重的骨骼。
    std::vector<int> helperBones;
    for (size_t bone = 0; bone < depths.size() && bone < rest.size(); ++bone) {
        if (static_cast<int>(bone) == parentBone) continue;
        if (static_cast<int>(bone) == childBone) continue;
        if (boneWeight[bone] > 1e-6) helperBones.push_back(static_cast<int>(bone));
    }
    if (helperBones.size() > 1) {
        if (debug) MGlobal::displayWarning(
            "[OCD] 主铰链之外有超过一根参与权重的骨骼，回退 DLB。");
        return deformDLB(block, iter, worldMat, multiIndex);
    }
    const int helperBone = helperBones.empty() ? -1 : helperBones.front();

    if (reportHinge) {
        // 真实角色上一定要能看到它选了哪三根骨骼。
        const MPlug matrixPlug(thisMObject(), MPxSkinCluster::matrix);
        const auto jointName = [&](int bone) -> MString {
            if (bone < 0) return MString("<none>");
            for (unsigned int entry = 0; entry < matrixPlug.numConnectedElements(); ++entry) {
                const auto element = matrixPlug.connectionByPhysicalIndex(entry);
                if (static_cast<int>(element.logicalIndex()) != bone) continue;
                MPlugArray sources;
                if (element.connectedTo(sources, true, false) && sources.length() == 1)
                    return MFnDagNode(sources[0].node()).name();
            }
            return MString("<unresolved>");
        };
        MString info = MString("[OCD] hinge parent=") + jointName(parentBone) +
            " child=" + jointName(childBone) + " helper=" + jointName(helperBone);
        if (static_cast<size_t>(childBone) < m_boneInfo.boneParentSteps.size()) {
            const int steps = m_boneInfo.boneParentSteps[childBone];
            if (steps > 1) {
                char suffix[64];
                snprintf(suffix, sizeof(suffix), " (%d unskinned joints crossed)", steps - 1);
                info += suffix;
            }
        }
        MGlobal::displayInfo(info);
        // 同时写成属性，供测试/脚本直接读，不靠日志抓取。
        MStatus reportStatus;
        MPlug reportPlug(thisMObject(), aOcdHingeReport);
        reportPlug.setString(MString("parent=") + jointName(parentBone) +
            " child=" + jointName(childBone) + " helper=" + jointName(helperBone));
        (void)reportStatus;
    }

    // ---- 2. 由 bind 空间构造铰链坐标系 ----
    // 参考实现把肢体放在 x 轴、半径 1 的圆柱上；这里做同样的对齐。
    const Vec3 bindParentPivot = pivots[parentBone];
    const Vec3 bindChildPivot = pivots[childBone];
    const Vec3 bindAxisRaw = v3_sub(bindChildPivot, bindParentPivot);
    double segmentLength = v3_len(bindAxisRaw);
    if (segmentLength < 1e-8) {
        if (debug) MGlobal::displayWarning("[OCD] 骨骼长度为零，回退 DLB。");
        return deformDLB(block, iter, worldMat, multiIndex);
    }
    const Vec3 bindAxis = v3_scale(bindAxisRaw, 1.0 / segmentLength);

    // bind 坐标系基，与参考实现的规范空间一一对应：
    //   x = 骨骼轴（参考实现的 +X）
    //   y = 绑定弯曲平面内的径向（参考实现的 +Y）
    //   z = 弯曲平面法线 / 侧向（参考实现的 +Z）
    // 参考实现把 direction 当作「绕骨骼轴在绑定帧内旋转弯面」，由 Elbow
    // 内部处理，因此这里只固定绑定帧，不做任何姿态投影。
    //
    // 弯面的参考轴不能用世界 +Y：那样把整条 rig 旋转一下，
    // direction 就会变，弯面、内外侧参与度与纵向滑动都会跟着变。
    // 改用子骨骼的绑定世界旋转导出参考轴：它随 rig 一起旋转，
    // 所以解答对整体平移与旋转是不变的。
    // 绑定世界旋转 = (绑定逆旋转)⁻¹，rest[] 存的是逆向。
    // 铰链坐标系挂在入射骨（父骨）上：父骨再转动时局部坐标随之运动。
    const Quat parentBindRotation = [&] {
        if (parentBone >= 0 && static_cast<size_t>(parentBone) < rest.size())
            return q_inverse(rest[parentBone].q);
        return Quat{0.0, 0.0, 0.0, 1.0};
    }();
    const Vec3 referenceUp = q_rotate_v3(parentBindRotation, Vec3{0.0, 1.0, 0.0});
    Vec3 basisY = v3_sub(referenceUp, v3_scale(bindAxis, v3_dot(referenceUp, bindAxis)));
    if (v3_len(basisY) < 1e-6) {
        // 参考轴与骨骼轴平行时退回绑定 +Z。
        const Vec3 referenceSide = q_rotate_v3(parentBindRotation, Vec3{0.0, 0.0, 1.0});
        basisY = v3_sub(referenceSide, v3_scale(bindAxis, v3_dot(referenceSide, bindAxis)));
    }
    basisY = v3_norm(basisY);
    const Vec3 basisZ = v3_norm(v3_cross(bindAxis, basisY));

    // 局部基矩阵（列为 axis / basisY / basisZ）与它的四元数。
    // 把世界旋转共轭到局部空间：R_local = Mᵀ·R_world·M。
    ocd_math::Mat3 frameBasis;
    frameBasis.m[0] = bindAxis.x; frameBasis.m[1] = basisY.x; frameBasis.m[2] = basisZ.x;
    frameBasis.m[3] = bindAxis.y; frameBasis.m[4] = basisY.y; frameBasis.m[5] = basisZ.y;
    frameBasis.m[6] = bindAxis.z; frameBasis.m[7] = basisY.z; frameBasis.m[8] = basisZ.z;
    const Quat frameQuat = ocd_math::mat3_to_quat(frameBasis);
    const Quat frameQuatInverse = q_inverse(frameQuat);
    const auto toLocalRotation = [&](const Quat& worldRotation) {
        return q_mul(q_mul(frameQuatInverse, worldRotation), frameQuat);
    };

    // ---- 3. 每个顶点在绑定铰链坐标系下的 station / 径向 ----
    MPointArray points;
    iter.allPositions(points);
    if (points.length() < nV) return MS::kSuccess;
    const MMatrix worldInverse = worldMat.inverse();

    struct VertexFrame {
        double station = 0.0;   // 沿骨骼轴坐标（世界单位）
        Vec3 radial{0.0, 0.0, 0.0};   // 相对轴的径向偏移
        Vec3 rest{};            // bind 世界位置
        double radius = 1.0;
    };
    std::vector<VertexFrame> frames(nV);
    double maxRadius = 0.0;
    for (unsigned int i = 0; i < nV; ++i) {
        const MPoint world = points[i] * worldMat;
        const Vec3 restPoint{world.x, world.y, world.z};
        const Vec3 relative = v3_sub(restPoint, bindChildPivot);
        const double station = v3_dot(relative, bindAxis);
        const Vec3 radial = v3_sub(relative, v3_scale(bindAxis, station));
        frames[i] = {station, radial, restPoint, v3_len(radial)};
        if (frames[i].radius > maxRadius) maxRadius = frames[i].radius;
    }
    if (maxRadius < 1e-9) {
        if (debug) MGlobal::displayWarning("[OCD] 截面半径为零，回退 DLB。");
        return deformDLB(block, iter, worldMat, multiIndex);
    }

    // ---- 4. 姿态：全部在父骨相对帧里解析 ----
    // 只用子骨相对父骨的局部旋转，再把它共轭进铰链局部基。
    // 这样 bend / twist / direction 与父骨在绑定之后的旋转、平移无关。
    // （旧版把「已随父骨运动的世界轴」投影到「绑定基」上，父骨一转
    //   direction 就会变，弯面与纵向滑动也跟着变。）
    // 子骨相对父骨的「局部旋转」才是父骨不变量。
    // 直接用 Cs_child · Cs_parent⁻¹ 是错的：Cs 都是「绑定世界 → 当前世界」，
    // 父骨一转，两项都带上了同一个左乘项，并不互相抵消；只有绕同一轴旋转
    // 时两者恰好可交换，所以纯 Z 旋转的测试会假通过。
    // 正确做法： poseLocal = poseParent⁻¹ · poseChild，再除以绑定时的局部旋转。
    // Cs 是世界空间算子（pose · bind⁻¹），会带上整条 rig 的摆放旋转。
    // 因此要先把它共轭子骨自身的局部帧，得到与摆放无关的局部旋转；
    // 再用绑定时的父子局部旋转把它变成「相对父骨」，最后进铰链局部基。
    // 绑定时子骨在父骨局部帧下的旋转。
    const auto bindLocalRotation = [&](int bone) -> Quat {
        if (bone < 0 || static_cast<size_t>(bone) >= rest.size())
            return Quat{0.0, 0.0, 0.0, 1.0};
        return q_mul(q_inverse(parentBindRotation), q_inverse(rest[bone].q));
    };
    // 相对父骨的旋转，对「整条 rig 的摆放」与「绑定后的父骨动画」都不变。
    //
    // pose = bind · Cs，所以父子局部位姿是 Cs_p⁻¹ · Lbind · Cs_c，
    // 除以绑定局部 Lbind 得：
    //     relative = Cs_p⁻¹ · Lbind · Cs_c · Lbind⁻¹
    // 少了中间的 Lbind 就会把两种情形各弄坏一个。
    // Cs 是「pose · bind⁻¹」，所以骨骼的世界位姿旋转是 Cs · bind。
    // 弄反这个乘序会把整条 rig 的摆放旋转带进解析，表现为
    // 摆放不同 bend/direction 就不同。
    const auto bonePoseRotation = [&](int bone) -> Quat {
        return q_mul(Cs[bone].q, q_inverse(rest[bone].q));
    };
    const auto hingeRotation = [&](int bone) -> Quat {
        if (bone < 0 || static_cast<size_t>(bone) >= rest.size())
            return Quat{0.0, 0.0, 0.0, 1.0};
        const Quat bindLocal = bindLocalRotation(bone);
        // 子骨在父骨局部帧里的当前位姿。
        const Quat poseLocal = q_mul(q_inverse(bonePoseRotation(parentBone)),
                                     bonePoseRotation(bone));
        const Quat relative = q_mul(poseLocal, q_inverse(bindLocal));
        // 父骨局部帧 → 绑定世界 → 铰链局部基。
        const Quat inWorld = q_mul(q_mul(parentBindRotation, relative),
                                   q_inverse(parentBindRotation));
        return toLocalRotation(inWorld);
    };
    const Quat relativeLocal = hingeRotation(childBone);
    // 局部基里骨骼轴就是 +X，于是当前子骨轴与弯面法线都是常规量。
    const Vec3 axisLocal{1.0, 0.0, 0.0};
    const Vec3 currentAxisLocal = q_rotate_v3(relativeLocal, axisLocal);
    const Vec3 normalLocal = v3_cross(axisLocal, currentAxisLocal);
    const double bendSine = v3_len(normalLocal);
    double bendDegrees = 0.0;
    double directionDegrees = 0.0;
    if (bendSine > 1e-8) {
        bendDegrees = std::atan2(bendSine, v3_dot(axisLocal, currentAxisLocal)) *
            180.0 / 3.14159265358979323846;
        // 弯面方位角：以 basisZ 为 0，朝 -basisY 增大。
        const Vec3 unitNormal = v3_scale(normalLocal, 1.0 / bendSine);
        directionDegrees = std::atan2(-unitNormal.y, unitNormal.z) *
            180.0 / 3.14159265358979323846;
    }
    // 扭转：从相对旋转里剥离 swing，取绕骨骼轴的分量。
    const Quat swingLocal = q_between(axisLocal, currentAxisLocal);
    const Quat twistQuat = q_mul(q_inverse(swingLocal), relativeLocal);
    double twistDegrees = 2.0 * std::atan2(twistQuat.x, twistQuat.w) *
        180.0 / 3.14159265358979323846;
    if (!std::isfinite(twistDegrees)) twistDegrees = 0.0;

    const double twistWeight = block.inputValue(aOcdTwistWeight, &status).asDouble();
    if (!status) return status;
    twistDegrees *= std::clamp(twistWeight, 0.0, 1.0);

    // ---- 5. 每个顶点在链上的权重（parent / child / helper） ----
    // 第三列对应 helper 骨本身（不是「其余骨骼权重之和」），
    // 否则下游用它乘上真实 helper 旋转时会错。
    std::vector<std::array<double, 3>> chainWeights(nV);
    for (unsigned int i = 0; i < nV; ++i) {
        double wParent = 0.0, wChild = 0.0, wHelper = 0.0;
        for (const auto& kv : m_vtxCache.vertices[i]) {
            if (static_cast<int>(kv.first) == parentBone) wParent += kv.second;
            else if (static_cast<int>(kv.first) == childBone) wChild += kv.second;
            else if (static_cast<int>(kv.first) == helperBone) wHelper += kv.second;
        }
        const double sum = wParent + wChild + wHelper;
        if (sum > 1e-9) { wParent /= sum; wChild /= sum; wHelper /= sum; }
        chainWeights[i] = {wParent, wChild, wHelper};
    }

    // ---- 6. 统一到铰链局部坐标系（关节在原点、骨骼沿 +X、半径 1） ----
    //
    // 参考实现的 bind 网格就活在这个规范空间里，而 buildEdges /
    // outerParticipation / structural_centers 都以该空间为前提：
    //   · outerParticipation 用绝对 y/z 判断内侧/外侧
    //   · structural_centers 用 rest 边的 x 分量判断纵向拉伸
    // 之前直接把世界坐标送进求解器，模型一旦不平移对齐到原点、或肘部不是沿
    // 世界 X 轴，内外侧参与度与纵向滑动都会算错。这里把参与求解的量全部换算
    // 到同一套局部坐标，世界坐标只用于最后写回。
    const double radius = maxRadius;

    const auto toLocalPoint = [&](const Vec3& world) {
        const Vec3 offset = v3_sub(world, bindChildPivot);
        return Vec3{ v3_dot(offset, bindAxis) / radius,
                     v3_dot(offset, basisY) / radius,
                     v3_dot(offset, basisZ) / radius };
    };
    // 位移与中心偏移都是向量：换回世界时不能叠加关节平移。
    const auto localOffsetToWorld = [&](const Vec3& local) {
        return v3_add(v3_scale(bindAxis, local.x * radius),
            v3_add(v3_scale(basisY, local.y * radius),
                   v3_scale(basisZ, local.z * radius)));
    };
    const auto localPointToWorld = [&](const Vec3& local) {
        return v3_add(bindChildPivot, localOffsetToWorld(local));
    };

    // 父骨的当前位姿（bind → pose）。铰链局部坐标系挂在它上面，
    // 所以目标位置要经过它才能落到当前世界，而逆变换用来把
    // 已随父骨运动的点拉回绑定帧。
    const RigidTransform& parentPose = Cs[parentBone];
    const RigidTransform parentPoseInverse{
        q_inverse(parentPose.q),
        q_rotate_v3(q_inverse(parentPose.q), v3_scale(parentPose.t, -1.0))};
    const auto localToPosePoint = [&](const Vec3& local) {
        return apply_transform(parentPose, localPointToWorld(local));
    };
    const auto poseToLocalPoint = [&](const Vec3& world) {
        return toLocalPoint(apply_transform(parentPoseInverse, world));
    };
    // 对象空间向量 → 世界空间向量（Maya 行向量约定：p' = p * M）。
    const auto objectVectorToWorld = [&](const Vec3& local) {
        return Vec3{
            local.x * worldMat[0][0] + local.y * worldMat[1][0] + local.z * worldMat[2][0],
            local.x * worldMat[0][1] + local.y * worldMat[1][1] + local.z * worldMat[2][1],
            local.x * worldMat[0][2] + local.y * worldMat[1][2] + local.z * worldMat[2][2]};
    };

    // 绑定顶点在局部空间的位置：参考实现的 s / y / z。
    std::vector<Vec3> bindLocal(nV);
    std::vector<Vec3> bindWorld(nV);
    for (unsigned int i = 0; i < nV; ++i) {
        bindWorld[i] = frames[i].rest;
        bindLocal[i] = toLocalPoint(frames[i].rest);
    }

    // station=0 折角的圆角窗口（肢体半径倍数，0 = 关闭）。
    MStatus filletStatus;
    const double creaseFillet =
        block.inputValue(aOcdCreaseFillet, &filletStatus).asDouble();
    const double filletWindow = filletStatus ? std::max(0.0, creaseFillet) : 0.0;

    // ---- 7. 补偿通道与目标形状 ----
    ocd_hinge::HingeParams hingeParams;
    hingeParams.bend = bendDegrees;
    hingeParams.twist = twistDegrees;
    hingeParams.direction = directionDegrees;
    hingeParams.radius = 1.0;      // 局部坐标已按半径归一，这里必须是 1
    {
        MStatus gatherStatus;
        const double gather = block.inputValue(aOcdOuterGather, &gatherStatus).asDouble();
        if (gatherStatus) hingeParams.gatherScale = std::clamp(gather, 0.0, 2.0);
        MStatus flipStatus;
        const bool flip = block.inputValue(aOcdBendFlip, &flipStatus).asBool();
        if (flipStatus) hingeParams.bendFlip = flip;
    }
    hingeParams.creaseFillet = filletWindow;
    {
        // 11 个补偿通道按参考实现的 CHANNELS 顺序读取。
        MObject* const channelAttrs[ocdChannelCount] = {
            &aOcdCenterSlideIn, &aOcdCenterSlideOut,
            &aOcdSlideInPrev, &aOcdSlideInNext,
            &aOcdSlideOutPrev, &aOcdSlideOutNext,
            &aOcdSlideOrthoInPrev, &aOcdSlideOrthoInNext,
            &aOcdSlideOrthoOutPrev, &aOcdSlideOrthoOutNext,
            &aOcdBulge,
        };
        for (int channel = 0; channel < ocdChannelCount; ++channel) {
            MStatus channelStatus;
            const double value = block.inputValue(*channelAttrs[channel], &channelStatus).asDouble();
            hingeParams.values[channel] = channelStatus ? value : 0.0;
        }
    }
    const bool avoidOverlap = [&] {
        MStatus avoidStatus;
        const bool value = block.inputValue(aOcdAvoidOverlap, &avoidStatus).asBool();
        return avoidStatus ? value : false;
    }();

    // 算法 3：世界空间目标位置。算法 5：局部空间相对位移（guide）。
    std::vector<Vec3> guide(nV);

    if (!correctCenters) {
        // 算法 3 (OCD)：Elbow::point 是唯一会应用 11 个补偿通道的路径。
        ocd_hinge::Pose elbowPose;
        elbowPose.bend = bendDegrees;
        elbowPose.twist = twistDegrees;
        elbowPose.direction = directionDegrees;
        elbowPose.avoidOverlap = avoidOverlap;
        elbowPose.values = hingeParams.values;
        elbowPose.creaseFillet = filletWindow;
        const ocd_hinge::Elbow elbow(elbowPose);
        for (unsigned int i = 0; i < nV; ++i) {
            // 形状在局部帧里建好，再经父骨当前位姿放回世界；
            // 否则父骨转动后肘部会留在绑定位置不跟随。
            guide[i] = localToPosePoint(
                elbow.point(bindLocal[i].x, bindLocal[i].y, bindLocal[i].z));
        }
    }

    // ---- 8. 算法 5：original = 零偏移时当前姿态的 CoR 输出，guide = 目标位移 ----
    std::vector<Vec3> centers(nV, Vec3{0.0, 0.0, 0.0});
    if (correctCenters) {
        // 网格面。与 bindLocal 同源（对象空间），只用来构造边表与三角形。
        std::vector<std::vector<unsigned int>> faces;
        MObject meshObject;
        {
            MStatus meshStatus;
            auto geometry = block.inputArrayValue(MPxSkinCluster::input, &meshStatus);
            if (meshStatus && geometry.jumpToElement(multiIndex)) {
                auto input = geometry.inputValue(&meshStatus);
                if (meshStatus) meshObject = input.child(MPxSkinCluster::inputGeom).asMesh();
            }
        }
        // 基础 CoR 的积分三角形必须与算法 0 一致：都用 Maya 的
        // getTriangles()。之前算法 5 从 face[0] 扇形三角化，在四边面上它恰好
        // 选同一条对角线，但在 n-gon 上不保证，会让两个算法的中心不同。
        std::vector<cor_paper::Triangle> corTriangles;
        if (!meshObject.isNull()) {
            MFnMesh mesh(meshObject, &status);
            if (status) {
                const int polygons = mesh.numPolygons(&status);
                if (status && polygons > 0) {
                    faces.reserve(polygons);
                    for (int polygon = 0; polygon < polygons; ++polygon) {
                        MIntArray vertices;
                        mesh.getPolygonVertices(polygon, vertices);
                        std::vector<unsigned int> face;
                        face.reserve(vertices.length());
                        for (unsigned int k = 0; k < vertices.length(); ++k)
                            face.push_back(static_cast<unsigned int>(vertices[k]));
                        faces.push_back(std::move(face));
                    }
                }
                MIntArray triCounts, triVerts;
                mesh.getTriangles(triCounts, triVerts);
                {
                    const unsigned int triangleCount = triVerts.length() / 3;
                    corTriangles.reserve(triangleCount);
                    for (unsigned int triangle = 0; triangle < triangleCount; ++triangle) {
                        corTriangles.push_back({
                            static_cast<unsigned int>(triVerts[triangle * 3 + 0]),
                            static_cast<unsigned int>(triVerts[triangle * 3 + 1]),
                            static_cast<unsigned int>(triVerts[triangle * 3 + 2])});
                    }
                }
            }
        }

        if (faces.empty()) {
            // 没有拓扑就无法构造边表；退化为普通 CoR（等价于偏移为零）。
            if (debug) MGlobal::displayWarning(
                "[OCD] corrected CoR needs polygon topology; falling back to CoR.");
            return deformCoR(block, iter, worldMat, multiIndex);
        }

        // 每个顶点的 CoR 权重，以及 computeCenters 需要的三角形表。
        std::vector<cor_paper::Weights> rotationWeights(nV);
        for (unsigned int i = 0; i < nV; ++i) {
            for (const auto& kv : m_vtxCache.vertices[i])
                rotationWeights[i].push_back({kv.first, kv.second});
            std::sort(rotationWeights[i].begin(), rotationWeights[i].end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }
        // buildEdges 用原始多边形边（保留局部结构，不引入对角线）；
        // computeCenters 用与算法 0 相同的 getTriangles() 结果。
        const std::vector<cor_paper::Triangle>& triangles = corTriangles;
        if (triangles.empty()) {
            if (debug) MGlobal::displayWarning(
                "[OCD] getTriangles() 没有返回三角形，回退到 CoR。");
            return deformCoR(block, iter, worldMat, multiIndex);
        }

        double sigma = 0.1;
        {
            MStatus sigmaStatus;
            const double value = block.inputValue(aCorSigma, &sigmaStatus).asDouble();
            if (sigmaStatus && value > 0.0) sigma = value;
        }
        // 标准 CoR 中心（世界/对象空间），与 deformCoR 使用同一份定义。
        const auto baseCenters = cor_paper::computeCenters(
            bindWorld, rotationWeights, triangles, sigma);

        // original：把中心偏移归零，在当前骨骼姿态下求 CoR，再换到局部空间。
        std::vector<Vec3> originalLocal(nV);
        for (unsigned int i = 0; i < nV; ++i) {
            const Vec3 center = i < baseCenters.size() ? baseCenters[i] : frames[i].rest;
            const auto transform = cor_paper::vertexTransform(
                center, rotationWeights[i], rotationWeights[i], Cs);
            // CoR 输出已经随父骨运动；先撤掉父骨当前姿态，再换进绑定铰链局部空间。
            originalLocal[i] = poseToLocalPoint(apply_transform(transform, frames[i].rest));
        }

        // 求解用的边表与内外侧参与度：全部在局部空间。
        const auto edges = ocd_centers::buildEdges(bindLocal, faces);
        const auto outer = ocd_centers::outerParticipation(
            bindLocal, chainWeights, bendDegrees, twistDegrees, directionDegrees);

        // guide 是相对位移（hinge 目标 − 当前 CoR 输出），不是绝对位置。
        std::vector<double> stationsLocal(nV), depthsLocal(nV), lateralsLocal(nV);
        std::vector<ocd_hinge::HingeWeight> hingeWeights(nV);
        for (unsigned int i = 0; i < nV; ++i) {
            stationsLocal[i] = bindLocal[i].x;
            depthsLocal[i] = bindLocal[i].y;
            lateralsLocal[i] = bindLocal[i].z;
            // 参考实现的权重就是 (parent, child, helper)。
            hingeWeights[i] = {chainWeights[i][0], chainWeights[i][1], chainWeights[i][2]};
        }
        // guide 来源有两种：
        //   · hingeDisplacement：内外两条轮廓 + 厚度梯度插值
        //   · Elbow::point：与算法 3 相同的解析截面构造
        // 两者在高弯曲下的外轮廓差别很大：前者会把外侧压瘪（负弯时
        // bind.y<0 的一侧被 plane 翻转成「内侧」，厚度只由 inner 决定），
        // 后者保留铰链解析模型本来就有的凸出量。参考预览里被认可的形状
        // 来自 Elbow，所以默认走 Elbow。
        MStatus guideSourceStatus;
        const bool useElbowGuide =
            block.inputValue(aOcdElbowGuide, &guideSourceStatus).asBool();
        if (!guideSourceStatus || useElbowGuide) {
            ocd_hinge::Pose elbowPose;
            elbowPose.bend = bendDegrees;
            elbowPose.twist = twistDegrees;
            elbowPose.direction = directionDegrees;
            elbowPose.avoidOverlap = avoidOverlap;
            elbowPose.values = hingeParams.values;
            elbowPose.creaseFillet = filletWindow;
            const ocd_hinge::Elbow elbow(elbowPose);
            for (unsigned int i = 0; i < nV; ++i) {
                // Elbow 给出的是"当前姿态下的绝对位置（局部帧）"，
                // 减掉零偏移的 CoR 输出才是相对位移。
                const Vec3 target = elbow.point(bindLocal[i].x,
                    bindLocal[i].y, bindLocal[i].z);
                guide[i] = v3_sub(target, originalLocal[i]);
            }
        }
        else {
            guide = ocd_hinge::hingeDisplacement(stationsLocal, depthsLocal, lateralsLocal,
                originalLocal, hingeWeights, hingeParams);
        }

        ocd_centers::Settings settings;
        // 用「相对父骨」的旋转。绝对旋转会让中心修正随父骨方向变化，
        // 而它本应只取决于弯曲量。参考实现里 prev 不动，
        // 所以它的 [单位, 弯曲, helper] 与这里的 [单位, 相对弯曲, 相对 helper] 等价。
        settings.rotations = {Quat{0.0, 0.0, 0.0, 1.0},
                              relativeLocal,
                              hingeRotation(helperBone)};
        MStatus limitStatus;
        double centerLimit = block.inputValue(aOcdCenterLimit, &limitStatus).asDouble();
        if (!limitStatus || centerLimit <= 0.0) centerLimit = 16.0;
        // 开关关闭时跳过末尾的逐顶点截断（对齐参考实现的 center_limit=None）。
        // 注意必须在上面那个 16 回退之后判断，否则关闭路径会被回退值覆盖。
        MStatus enableLimitStatus;
        const bool enableCenterLimit =
            block.inputValue(aOcdEnableCenterLimit, &enableLimitStatus).asBool();
        // 属性按世界单位暴露；求解空间是半径归一，故换算成「肢体半径的倍数」。
        // centerLimit <= 0 在 structuralCenters 中表示不限幅。
        settings.centerLimit = (enableLimitStatus && !enableCenterLimit)
            ? 0.0
            : centerLimit / radius;

        ocd_centers::Diagnostics solveDiagnostics;
        settings.diagnostics = &solveDiagnostics;
        std::vector<Vec3> offsetsLocal = ocd_centers::structuralCenters(
            edges, originalLocal, chainWeights, settings, guide, outer);

        // ---- 诊断导出：guide / delta / 实际限额 ----
        // 只写属性，不影响正常驱动。
        {
            MDoubleArray guideOut, deltaOut, limitOut;
            guideOut.setLength(static_cast<unsigned int>(3 * nV));
            deltaOut.setLength(static_cast<unsigned int>(3 * nV));
            for (unsigned int i = 0; i < nV; ++i) {
                guideOut[3 * i + 0] = guide[i].x;
                guideOut[3 * i + 1] = guide[i].y;
                guideOut[3 * i + 2] = guide[i].z;
                deltaOut[3 * i + 0] = offsetsLocal[i].x;
                deltaOut[3 * i + 1] = offsetsLocal[i].y;
                deltaOut[3 * i + 2] = offsetsLocal[i].z;
            }
            limitOut.append(settings.centerLimit);
            limitOut.append(static_cast<double>(solveDiagnostics.iterations));
            limitOut.append(solveDiagnostics.converged ? 1.0 : 0.0);
            limitOut.append(solveDiagnostics.residualRatio);
            MPlug guidePlug(thisMObject(), aOcdGuideDump);
            guidePlug.setMObject(MFnDoubleArrayData().create(guideOut));
            MPlug deltaPlug(thisMObject(), aOcdDeltaDump);
            deltaPlug.setMObject(MFnDoubleArrayData().create(deltaOut));
            MPlug limitPlug(thisMObject(), aOcdCenterLimitDump);
            limitPlug.setMObject(MFnDoubleArrayData().create(limitOut));
        }

        // 可选的注入偏移（对象空间向量），用于与参考实现逐级对拍。
        std::vector<Vec3> injected;
        {
            MStatus offsetStatus;
            const MObject data = block.inputValue(aOcdCenterOffsets, &offsetStatus).data();
            if (offsetStatus && !data.isNull()) {
                MFnDoubleArrayData arrayData(data, &offsetStatus);
                if (offsetStatus) {
                    const MDoubleArray values = arrayData.array(&offsetStatus);
                    if (offsetStatus && values.length() == 3 * nV) {
                        injected.resize(nV);
                        for (unsigned int i = 0; i < nV; ++i)
                            injected[i] = {values[3 * i], values[3 * i + 1], values[3 * i + 2]};
                    }
                }
            }
        }

        // [实验] delta 门控：低权重区（w1 小）不应应用中心偏移。
        // 实测：注入全零 delta 后 x=-0.5 处圆度 0.096，
        // 而自动求解是 0.215（更差），解析模型 0.102。
        // 所以 delta 在 w1 偏低时会破坏截面圆度。
        MStatus gateStatus;
        double gateLow = block.inputValue(aOcdCenterGateLow, &gateStatus).asDouble();
        double gateHigh = gateStatus
            ? block.inputValue(aOcdCenterGateHigh, &gateStatus).asDouble() : 0.0;
        // high <= low 表示门控关闭（保持旧行为）。
        const bool applyGate = gateHigh > gateLow + 1e-9;
        for (unsigned int i = 0; i < nV; ++i) {
            const Vec3 base = i < baseCenters.size() ? baseCenters[i] : frames[i].rest;
            if (injected.empty()) {
                Vec3 offset = localOffsetToWorld(offsetsLocal[i]);
                if (applyGate) {
                    const double w1 = chainWeights[i][1];
                    const double gate = std::clamp((w1 - gateLow) / (gateHigh - gateLow), 0.0, 1.0);
                    offset = v3_scale(offset, gate);
                }
                centers[i] = v3_add(base, offset);
            }
            else {
                // 属性按对象空间声明，而 centres 是世界空间；
                // 必须先乘 worldMat 的线性部分，否则对象有旋转/缩放时含义不同。
                centers[i] = v3_add(base, objectVectorToWorld(injected[i]));
            }
        }
    }

    // ---- 9. 写出结果 ----
    if (correctCenters) {
        // OCD + Corrected CoR：用中心偏移驱动 cor_paper 求值
        std::vector<cor_paper::Weights> rotationWeights(nV);
        std::vector<cor_paper::Weights> translationWeights(nV);
        for (unsigned int i = 0; i < nV; ++i) {
            for (const auto& kv : m_vtxCache.vertices[i]) {
                rotationWeights[i].push_back({kv.first, kv.second});
                translationWeights[i].push_back({kv.first, kv.second});
            }
            std::sort(rotationWeights[i].begin(), rotationWeights[i].end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
            std::sort(translationWeights[i].begin(), translationWeights[i].end(),
                [](const auto& a, const auto& b) { return a.first < b.first; });
        }

        double maxDeltaSquared = 0.0;
#pragma omp parallel for schedule(static)
        for (int index = 0; index < static_cast<int>(nV); ++index) {
            const Vec3 restPoint = frames[index].rest;
            const Vec3 center = centers[index];
            const auto transform = cor_paper::vertexTransform(
                center, rotationWeights[index], translationWeights[index], Cs);
            const Vec3 base = apply_transform(transform, restPoint);
            // 参考实现在 CoR 中心已由拟合吸收 OCD 形状，这里直接输出
            const Vec3 result = applyEnvelope(restPoint, base, envelope);
            points[index] = MPoint(result.x, result.y, result.z) * worldInverse;
        }
        (void)maxDeltaSquared;
        iter.setAllPositions(points);
    }
    else {
        double maxDeltaSquared = 0.0;
#pragma omp parallel for schedule(static)
        for (int index = 0; index < static_cast<int>(nV); ++index) {
            const Vec3 restPoint = frames[index].rest;
            const Vec3 posed = guide[index];
            const Vec3 result = applyEnvelope(restPoint, posed, envelope);
            points[index] = MPoint(result.x, result.y, result.z) * worldInverse;
        }
        (void)maxDeltaSquared;
        iter.setAllPositions(points);
    }

    (void)multiIndex;
    return MS::kSuccess;
}

MStatus OptimizedSkinCluster::deformMOS(
    MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int multiIndex) {
    using namespace mathcore;
    const unsigned int count = iter.count();
    const MosParams params = readMosParams(block);
    const bool shapeGuide = block.inputValue(aMosShapeGuide).asBool();
    const bool debug = optSkinDebugLog() || params.debugLog;
    if (!m_mosCache.valid || !m_mosCache.direct || m_mosCache.shapeGuide != shapeGuide ||
        m_mosCache.geometryIndex != multiIndex ||
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
    cor_contour::BendFrame guideFrame;
    unsigned int guideParent = 0, guideChild = 0;
    double guideLength = 0.0;
    ocd_arc::Pose arcPose;
    const bool useArc = params.curveMode == MosParams::arcSegments && !m_mosCache.arcVertices.empty() &&
        params.scaleWeight <= 1e-9;
    if (useArc) {
        const auto& chain = m_mosCache.curves.front().chain_bones;
        const auto parent = chain[0], child = chain[1];
        const auto childPivot = apply_transform(transforms[child], pivots[child]);
        const auto parentPivot = apply_transform(transforms[parent], pivots[child]);
        arcPose = ocd_arc::buildPose(childPivot, v3_norm(v3_sub(pivots[child], pivots[parent])),
            transforms[parent].q, transforms[child].q, m_mosCache.arcBefore, m_mosCache.arcAfter,
            v3_sub(childPivot, parentPivot));
    }
    const auto arcRegion = ocd_arc::localRegion(arcPose, m_mosCache.arcRadius);
    if (params.curveMode != MosParams::arcSegments && shapeGuide && params.scaleWeight <= 1e-9 &&
        m_mosCache.curves.size() == 1) {
        const auto& chain = m_mosCache.curves.front().chain_bones;
        if (chain.size() == 2) {
            guideParent = chain[0];
            guideChild = chain[1];
            guideLength = v3_len(v3_sub(pivots[guideChild], pivots[guideParent]));
            guideFrame = cor_contour::buildFrame(pivots[guideParent], pivots[guideChild],
                apply_transform(transforms[guideChild], pivots[guideChild]),
                transforms[guideParent].q, transforms[guideChild].q);
        }
    }
    const Vec3 guideManualIn = params.useManualInVector ? params.manualInVector : Vec3{};
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
        const auto arc = useArc ? ocd_arc::sampleLocal(arcPose, arcRegion,
            m_mosCache.arcVertices[vertex], incoming, outgoing, guideManualIn) : mos_math::GuideSample{};
        const double arcMix = std::clamp(arc.coverage * params.influenceRadius, 0.0, 1.0);
        if (arcMix >= 1.0) {
            const auto result = applyEnvelope(original, arc.position, envelope);
            return MPoint(result.x, result.y, result.z) * worldInverse;
        }
        mos_math::GuideSample guide;
        if (guideFrame.activation > 0.0 && !bind.empty() && bind.front().w_c >= 1.0 - 1e-8) {
            guide = mos_math::sampleTwoBoneGuide(guideFrame, original,
                transforms[guideParent].q, transforms[guideChild].q, guideLength,
                incoming, outgoing, guideManualIn);
        }
        if (guide.coverage == 1.0 && params.influenceRadius >= 1.0) {
            const auto result = applyEnvelope(original, guide.position, envelope);
            return MPoint(result.x, result.y, result.z) * worldInverse;
        }
        for (size_t c = 0; !useArc && c < bind.size(); ++c) {
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
        if (guide.coverage > 0.0) {
            sampled = v3_add(v3_scale(sampled, 1.0 - guide.coverage),
                v3_scale(guide.position, coverage * guide.coverage));
        }
        const double mix = useArc ? arcMix : std::clamp(coverage * params.influenceRadius, 0.0, 1.0);
        Vec3 result = useArc ? v3_scale(arc.position, arcMix) :
            (coverage > 0.0 ? v3_scale(sampled, mix / coverage) : Vec3{});
        if (mix < 1.0) {
            Vec3 fallback{};
            double total = 0.0;
            for (const auto& weight : weights) {
                if (weight.first >= transforms.size()) continue;
                fallback = v3_add(fallback, v3_scale(apply_transform(transforms[weight.first], original),
                    weight.second));
                total += weight.second;
            }
            const auto lbs = total > 0.0 ? v3_scale(fallback, 1.0 / total) : original;
            result = useArc ? ocd_arc::blendWithLbs(arcPose, lbs, arc.position, mix) :
                v3_add(result, v3_scale(lbs, 1.0 - mix));
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

// getCoRData
// ================================================================
bool OptimizedSkinCluster::getCoRData(
    std::vector<mathcore::Vec3>& out) const {
    const short algorithm = MPlug(thisMObject(), aAlgorithmType).asShort();
    if (algorithm == 0 || algorithm == 6) {
        const auto& cache = algorithm == 6 ? m_ddmV4 : m_paperCoR;
        if (!cache.valid || cache.weightVersion != m_currentWeightVersion) return false;
        out = cache.centers;
        return true;
    }
    if (!m_corCache.valid) return false;
    out = m_corCache.cors;
    return true;
}
