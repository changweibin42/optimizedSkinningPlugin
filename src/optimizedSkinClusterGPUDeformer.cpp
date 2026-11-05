#include "optimizedSkinClusterGPUDeformer.h"
#include "optimizedSkinClusterCompute.h"
#include "skinClusterData.h"
#include "node_ids.h"
#include "plugin_debug.h"

#include <maya/MGlobal.h>
#include <maya/MDataHandle.h>
#include <maya/MFnPluginData.h>
#include <maya/MPxDeformerNode.h>
#include <maya/MIndexMapper.h>

#include <vector>
#include <cstring>

MString OptimizedSkinClusterGPUDeformer::s_pluginPath;

OptimizedSkinClusterGPUDeformer::OptimizedSkinClusterGPUDeformer()
    : MPxGPUStandardDeformer() {
}

OptimizedSkinClusterGPUDeformer::~OptimizedSkinClusterGPUDeformer() {
    terminate();
}

void OptimizedSkinClusterGPUDeformer::terminate() {
    m_kernelInfo.reset();
    m_boneIdsBuf.reset();
    m_boneWtsBuf.reset();
    m_boneCountBuf.reset();
    m_restQBuf.reset();
    m_restTBuf.reset();
    m_poseQBuf.reset();
    m_poseTBuf.reset();
    m_corBuf.reset();
    m_boneDirsBuf.reset();
    m_jointPosBuf.reset();
    m_context = nullptr;
    m_lastAlgo = -1;
    m_lastWeightVersion = -1;           // ★
    m_lastFullCount = 0;                // ★
    m_vtxBuffersCached = false;         // ★
    MPxGPUStandardDeformer::terminate();
}

bool OptimizedSkinClusterGPUDeformer::ensureContext() {
    if (m_context) return true;
    const MGPUDeformerBuffer& inPos = inputPositions();
    if (!inPos.isValid()) return false;
    cl_mem raw = inPos.buffer().get();
    if (!raw) return false;
    cl_int err = clGetMemObjectInfo(raw, CL_MEM_CONTEXT,
        sizeof(cl_context), &m_context, nullptr);
    return (err == CL_SUCCESS && m_context);
}

bool OptimizedSkinClusterGPUDeformer::uploadBuffer(
    MOpenCLBuffer& buf, size_t bytes, const void* data) {
    if (bytes == 0) return true;
    if (buf.bufferSize() < bytes) {
        buf.reset();
        cl_int err = CL_SUCCESS;
        buf = MOpenCLBuffer(m_context, CL_MEM_READ_WRITE, bytes, nullptr, &err);
        if (err != CL_SUCCESS) return false;
    }
    cl_int err = MOpenCLUtils::uploadToGPU(
        bytes, const_cast<void*>(data), buf, MOpenCLUtils::kBlocking);
    return err == CL_SUCCESS;
}

// ================================================================
// uploadSkinData（★ 加 needVtxUpload）
// ================================================================
bool OptimizedSkinClusterGPUDeformer::uploadSkinData(
    MDataBlock& block, unsigned int fullCount,
    int& outAlgo, float& outDqsStrength, int& outNumBones,
    bool needVtxUpload) {

    MStatus status;
    MDataHandle h = block.inputValue(OptimizedSkinClusterCompute::aInData, &status);
    if (!status) return false;
    MObject obj = h.data();
    if (obj.isNull()) return false;

    MFnPluginData fnD(obj, &status);
    if (!status) return false;
    SkinClusterData* sd = dynamic_cast<SkinClusterData*>(fnD.data(&status));
    if (!status || !sd) return false;

    outAlgo = sd->algorithmType;
    outDqsStrength = sd->dqsStrength;

    unsigned int nB = (unsigned int)sd->restList.size();
    outNumBones = (int)nB;

    // ---- ★ 顶点级：只在 needVtxUpload 时上传 ----
    if (needVtxUpload) {
        std::vector<int>   boneIds(fullCount * 4, 0);
        std::vector<float> boneWts(fullCount * 4, 0.0f);
        std::vector<int>   boneCounts(fullCount, 0);
        std::vector<float> cors(fullCount * 3, 0.0f);

        unsigned int nV = (unsigned int)sd->vertices.size();
        unsigned int limitV = (nV < fullCount) ? nV : fullCount;
        for (unsigned int i = 0; i < limitV; ++i) {
            const SkinVertex& sv = sd->vertices[i];
            int n = sv.boneCount;
            if (n > 4) n = 4;
            boneCounts[i] = n;
            for (int k = 0; k < n; ++k) {
                boneIds[i * 4 + k] = sv.boneIds[k];
                boneWts[i * 4 + k] = sv.weights[k];
            }
        }

        unsigned int nC = (unsigned int)sd->cors.size();
        unsigned int limitC = (nC < fullCount) ? nC : fullCount;
        for (unsigned int i = 0; i < limitC; ++i) {
            cors[i * 3 + 0] = sd->cors[i].x;
            cors[i * 3 + 1] = sd->cors[i].y;
            cors[i * 3 + 2] = sd->cors[i].z;
        }

        if (!uploadBuffer(m_boneIdsBuf, fullCount * 4 * sizeof(int), boneIds.data()))    return false;
        if (!uploadBuffer(m_boneWtsBuf, fullCount * 4 * sizeof(float), boneWts.data()))    return false;
        if (!uploadBuffer(m_boneCountBuf, fullCount * sizeof(int), boneCounts.data())) return false;

        if (outAlgo == 0) {
            if (!uploadBuffer(m_corBuf, fullCount * 3 * sizeof(float), cors.data())) return false;
        }

        m_vtxBuffersCached = true;
        m_lastFullCount = fullCount;
    }

    // ---- 骨骼级（每帧上传）----
    std::vector<float> restQ(nB * 4, 0.0f);
    std::vector<float> restT(nB * 3, 0.0f);
    std::vector<float> poseQ(nB * 4, 0.0f);
    std::vector<float> poseT(nB * 3, 0.0f);
    for (unsigned int i = 0; i < nB; ++i) {
        restQ[i * 4 + 0] = sd->restList[i].qx;
        restQ[i * 4 + 1] = sd->restList[i].qy;
        restQ[i * 4 + 2] = sd->restList[i].qz;
        restQ[i * 4 + 3] = sd->restList[i].qw;
        restT[i * 3 + 0] = sd->restList[i].tx;
        restT[i * 3 + 1] = sd->restList[i].ty;
        restT[i * 3 + 2] = sd->restList[i].tz;
        poseQ[i * 4 + 0] = sd->poseList[i].qx;
        poseQ[i * 4 + 1] = sd->poseList[i].qy;
        poseQ[i * 4 + 2] = sd->poseList[i].qz;
        poseQ[i * 4 + 3] = sd->poseList[i].qw;
        poseT[i * 3 + 0] = sd->poseList[i].tx;
        poseT[i * 3 + 1] = sd->poseList[i].ty;
        poseT[i * 3 + 2] = sd->poseList[i].tz;
    }

    // DQS 专用数据（每帧上传，因为依赖骨骼姿势）
    if (outAlgo == 2) {
        std::vector<float> boneDirs(nB * 3, 0.0f);
        std::vector<float> jointPos(nB * 3, 0.0f);
        unsigned int nBD = (unsigned int)sd->boneDirs.size();
        for (unsigned int i = 0; i < nBD && i < nB; ++i) {
            boneDirs[i * 3 + 0] = sd->boneDirs[i].x;
            boneDirs[i * 3 + 1] = sd->boneDirs[i].y;
            boneDirs[i * 3 + 2] = sd->boneDirs[i].z;
        }
        unsigned int nJP = (unsigned int)sd->jointPositions.size();
        for (unsigned int i = 0; i < nJP && i < nB; ++i) {
            jointPos[i * 3 + 0] = sd->jointPositions[i].x;
            jointPos[i * 3 + 1] = sd->jointPositions[i].y;
            jointPos[i * 3 + 2] = sd->jointPositions[i].z;
        }
        if (!uploadBuffer(m_boneDirsBuf, nB * 3 * sizeof(float), boneDirs.data())) return false;
        if (!uploadBuffer(m_jointPosBuf, nB * 3 * sizeof(float), jointPos.data())) return false;
    }

    if (!uploadBuffer(m_restQBuf, nB * 4 * sizeof(float), restQ.data()))      return false;
    if (!uploadBuffer(m_restTBuf, nB * 3 * sizeof(float), restT.data()))      return false;
    if (!uploadBuffer(m_poseQBuf, nB * 4 * sizeof(float), poseQ.data()))      return false;
    if (!uploadBuffer(m_poseTBuf, nB * 3 * sizeof(float), poseT.data()))      return false;

    return true;
}

// ================================================================
// enqueueCoR / DLB / DQS
// ================================================================
cl_int OptimizedSkinClusterGPUDeformer::enqueueCoR(
    MAutoCLEvent& syncEvent,
    const MGPUDeformerBuffer& inPos,
    const MGPUDeformerBuffer& outPos,
    unsigned int affect, bool identity) {

    cl_int err = CL_SUCCESS;
    MAutoCLEvent syncIn = syncEvent;
    syncEvent = MAutoCLEvent();
    MGPUEventList eventList;
    eventList.add(syncIn);

    unsigned int a = 0;
    err = m_kernelInfo.setKernelArgBuffer(a++, inPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, outPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_corBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneIdsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneWtsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneCountBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, affectMapBuffer(), err, !identity);
    err = m_kernelInfo.setKernelArg<cl_uint>(a++, affect, err);
    err = m_kernelInfo.setKernelArg<cl_int>(a++, identity ? 0 : 1, err);
    err = m_kernelInfo.setKernelArg<cl_float>(a++, envelope(), err);

    if (err != CL_SUCCESS) return err;
    return m_kernelInfo.enqueueKernel(eventList, syncEvent);
}

cl_int OptimizedSkinClusterGPUDeformer::enqueueDLB(
    MAutoCLEvent& syncEvent,
    const MGPUDeformerBuffer& inPos,
    const MGPUDeformerBuffer& outPos,
    unsigned int affect, bool identity) {

    cl_int err = CL_SUCCESS;
    MAutoCLEvent syncIn = syncEvent;
    syncEvent = MAutoCLEvent();
    MGPUEventList eventList;
    eventList.add(syncIn);

    unsigned int a = 0;
    err = m_kernelInfo.setKernelArgBuffer(a++, inPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, outPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneIdsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneWtsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneCountBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, affectMapBuffer(), err, !identity);
    err = m_kernelInfo.setKernelArg<cl_uint>(a++, affect, err);
    err = m_kernelInfo.setKernelArg<cl_int>(a++, identity ? 0 : 1, err);
    err = m_kernelInfo.setKernelArg<cl_float>(a++, envelope(), err);

    if (err != CL_SUCCESS) return err;
    return m_kernelInfo.enqueueKernel(eventList, syncEvent);
}

cl_int OptimizedSkinClusterGPUDeformer::enqueueDQS(
    MAutoCLEvent& syncEvent,
    const MGPUDeformerBuffer& inPos,
    const MGPUDeformerBuffer& outPos,
    unsigned int affect, bool identity,
    int numBones, float dqsStrength) {

    cl_int err = CL_SUCCESS;
    MAutoCLEvent syncIn = syncEvent;
    syncEvent = MAutoCLEvent();
    MGPUEventList eventList;
    eventList.add(syncIn);

    unsigned int a = 0;
    err = m_kernelInfo.setKernelArgBuffer(a++, inPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, outPos.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneIdsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneWtsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneCountBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_restTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseQBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_poseTBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_boneDirsBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, m_jointPosBuf.buffer(), err);
    err = m_kernelInfo.setKernelArgBuffer(a++, affectMapBuffer(), err, !identity);
    err = m_kernelInfo.setKernelArg<cl_uint>(a++, affect, err);
    err = m_kernelInfo.setKernelArg<cl_int>(a++, identity ? 0 : 1, err);
    err = m_kernelInfo.setKernelArg<cl_int>(a++, numBones, err);
    err = m_kernelInfo.setKernelArg<cl_float>(a++, envelope(), err);
    err = m_kernelInfo.setKernelArg<cl_float>(a++, dqsStrength, err);

    if (err != CL_SUCCESS) return err;
    return m_kernelInfo.enqueueKernel(eventList, syncEvent);
}

// ================================================================
// evaluate（★ 加缓存判断）
// ================================================================
MPxGPUDeformer::DeformerStatus OptimizedSkinClusterGPUDeformer::evaluate(
    MDataBlock& block,
    const MEvaluationNode& evaluationNode,
    const MPlug& plug,
    const MPlugArray& inputPlugs,
    const MGPUDeformerData& inputData,
    MGPUDeformerData& outputData) {

    MAutoCLEvent syncEvent;
    DeformerStatus ds = prepareEvaluation(
        block, evaluationNode, plug, inputPlugs,
        inputData, outputData, syncEvent);
    if (ds != MPxGPUDeformer::kDeformerSuccess) return ds;

    prepareAffectMapBuffer();

    unsigned int full = fullCount();
    unsigned int affect = affectCount();
    if (affect == 0) return MPxGPUDeformer::kDeformerSuccess;
    bool identity = isIdentityMap();

    if (!ensureContext()) return MPxGPUDeformer::kDeformerFailure;

    // ---- 读 sd 判断是否需要重传 ----
    MStatus status;
    MDataHandle hData = block.inputValue(
        OptimizedSkinClusterCompute::aInData, &status);
    if (!status) return MPxGPUDeformer::kDeformerFailure;
    MObject obj = hData.data();
    if (obj.isNull()) return MPxGPUDeformer::kDeformerFailure;
    MFnPluginData fnD(obj, &status);
    if (!status) return MPxGPUDeformer::kDeformerFailure;
    SkinClusterData* sdCheck = dynamic_cast<SkinClusterData*>(fnD.data(&status));
    if (!status || !sdCheck) return MPxGPUDeformer::kDeformerFailure;

    bool needVtxUpload = false;
    if (!m_vtxBuffersCached)                                       needVtxUpload = true;
    else if (sdCheck->weightVersion != m_lastWeightVersion)        needVtxUpload = true;
    else if (full != m_lastFullCount)                              needVtxUpload = true;
    else if (sdCheck->algorithmType != m_lastAlgo)                 needVtxUpload = true;

    int algo = -1;
    float dqsStr = 1.0f;
    int nBones = 0;
    if (!uploadSkinData(block, full, algo, dqsStr, nBones, needVtxUpload))
        return MPxGPUDeformer::kDeformerFailure;

    // 更新缓存版本
    m_lastWeightVersion = sdCheck->weightVersion;

    // OCD 暂仅 CPU 实现：需要每顶点 τ 的曲线求值、M_j 的 swing-twist 分解，
    // 还需把弧长 LUT 上传到 GPU；收益不确定，且两套公式极易漂移。
    // 这里显式返回失败，由 Maya 回退到 CPU 端的 deformOCD，而不是静默产出错误结果。
    if (algo == 3) {
        static bool s_ocdUnsupportedWarned = false;
        if (!s_ocdUnsupportedWarned && optSkinDebugLog()) {
            s_ocdUnsupportedWarned = true;
            MGlobal::displayWarning(
                "[optimizedSkinCluster] B-spline OCD 暂不支持 GPU 求值，已回退 CPU");
        }
        return MPxGPUDeformer::kDeformerFailure;
    }

    // kernel 选择
    MString kernelFile;
    MString kernelName;
    switch (algo) {
    case 0: kernelFile = s_pluginPath + "/kernel_cor.cl"; kernelName = "cor_deform"; break;
    case 1: kernelFile = s_pluginPath + "/kernel_dlb.cl"; kernelName = "dlb_deform"; break;
    case 2: kernelFile = s_pluginPath + "/kernel_dqs.cl"; kernelName = "dqs_deform"; break;
    default: return MPxGPUDeformer::kDeformerFailure;
    }

    if (m_lastAlgo != algo) {
        m_kernelInfo.reset();
        m_lastAlgo = algo;
    }

    if (!m_kernelInfo.loadOrUpdate(kernelFile, kernelName, (int)affect))
        return MPxGPUDeformer::kDeformerFailure;

    cl_int err = initializeOutputPositions(syncEvent);
    if (err != CL_SUCCESS) return MPxGPUDeformer::kDeformerFailure;

    switch (algo) {
    case 0: err = enqueueCoR(syncEvent, inputPositions(), outputPositions(), affect, identity); break;
    case 1: err = enqueueDLB(syncEvent, inputPositions(), outputPositions(), affect, identity); break;
    case 2: err = enqueueDQS(syncEvent, inputPositions(), outputPositions(), affect, identity, nBones, dqsStr); break;
    }

    if (err != CL_SUCCESS) return MPxGPUDeformer::kDeformerFailure;

    return finishEvaluation(syncEvent, outputData);
}

// ================================================================
// 注册信息
// ================================================================
MPxGPUDeformer* OptimizedSkinClusterGPUDeformerInfo::createGPUDeformer() {
    return new OptimizedSkinClusterGPUDeformer();
}

bool OptimizedSkinClusterGPUDeformerInfo::validateNodeInGraph(
    MDataBlock& /*block*/,
    const MEvaluationNode& /*evaluationNode*/,
    const MPlug& /*plug*/,
    MStringArray* /*messages*/) {
    return true;
}

bool OptimizedSkinClusterGPUDeformerInfo::validateNodeValues(
    MDataBlock& block,
    const MEvaluationNode& /*evaluationNode*/,
    const MPlug& /*plug*/,
    MStringArray* /*messages*/) {
    MStatus status;
    MDataHandle h = block.inputValue(
        OptimizedSkinClusterCompute::aInUseGPU, &status);
    if (!status) return false;
    return h.asBool();
}