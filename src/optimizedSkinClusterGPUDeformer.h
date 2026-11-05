#pragma once

#include <maya/MPxGPUStandardDeformer.h>
#include <maya/MGPUDeformerRegistry.h>
#include <maya/MOpenCLUtils.h>
#include <maya/MOpenCLBuffer.h>
#include <maya/MOpenCLAutoPtr.h>
#include <maya/MEvaluationNode.h>
#include <maya/MDataBlock.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>
#include <maya/MString.h>
#include <maya/MStringArray.h>
#include <maya/MObjectArray.h>
#include <maya/MStatus.h>
#include <maya/MTypeId.h>

#include <clew/clew_cl.h>
#include <vector>

/**
 * @brief optimizedSkinCluster 的 GPU 标准变形器。
 *
 * 支持的算法编号：0 = Centers of Rotation、1 = Dual Quaternion Linear Blending、
 * 2 = DQS Compensation。
 *
 * 注意：3 = B-spline OCD **不在此实现**。OCD 需要每顶点 τ 的曲线求值、
 *      M_j = S_j·W_j·R_j 的 swing-twist 分解，以及弧长 LUT 上传，
 *      目前仅由 CPU 端 deformOCD 求值；evaluate() 遇到该编号会显式返回失败，
 *      由 Maya 自动回退到 CPU 路径，以避免 CPU/GPU 两套公式漂移。
 */
class OptimizedSkinClusterGPUDeformer : public MPxGPUStandardDeformer {
public:
    OptimizedSkinClusterGPUDeformer();
    ~OptimizedSkinClusterGPUDeformer() override;

    MPxGPUDeformer::DeformerStatus evaluate(
        MDataBlock& block,
        const MEvaluationNode& evaluationNode,
        const MPlug& outputPlug,
        const MPlugArray& inputPlugs,
        const MGPUDeformerData& inputData,
        MGPUDeformerData& outputData) override;

    void terminate() override;

    static MString s_pluginPath;

private:
    bool ensureContext();
    bool uploadBuffer(MOpenCLBuffer& buf, size_t bytes, const void* data);

    bool uploadSkinData(MDataBlock& block, unsigned int fullCount,
        int& outAlgo, float& outDqsStrength, int& outNumBones,
        bool needVtxUpload);   // ★

    cl_int enqueueCoR(MAutoCLEvent& syncEvent,
        const MGPUDeformerBuffer& inPos,
        const MGPUDeformerBuffer& outPos,
        unsigned int affect, bool identity);

    cl_int enqueueDLB(MAutoCLEvent& syncEvent,
        const MGPUDeformerBuffer& inPos,
        const MGPUDeformerBuffer& outPos,
        unsigned int affect, bool identity);

    cl_int enqueueDQS(MAutoCLEvent& syncEvent,
        const MGPUDeformerBuffer& inPos,
        const MGPUDeformerBuffer& outPos,
        unsigned int affect, bool identity,
        int numBones, float dqsStrength);

    MOpenCLKernelInfo m_kernelInfo;
    cl_context        m_context = nullptr;
    int               m_lastAlgo = -1;

    // ★ 顶点级 buffer 缓存状态
    int               m_lastWeightVersion = -1;
    unsigned int      m_lastFullCount = 0;
    bool              m_vtxBuffersCached = false;

    // 通用骨骼 buffer
    MOpenCLBuffer m_boneIdsBuf;
    MOpenCLBuffer m_boneWtsBuf;
    MOpenCLBuffer m_boneCountBuf;
    MOpenCLBuffer m_restQBuf;
    MOpenCLBuffer m_restTBuf;
    MOpenCLBuffer m_poseQBuf;
    MOpenCLBuffer m_poseTBuf;

    // CoR 专用
    MOpenCLBuffer m_corBuf;

    // DQS 专用
    MOpenCLBuffer m_boneDirsBuf;
    MOpenCLBuffer m_jointPosBuf;
};

class OptimizedSkinClusterGPUDeformerInfo : public MGPUDeformerRegistrationInfo {
public:
    OptimizedSkinClusterGPUDeformerInfo() {}
    ~OptimizedSkinClusterGPUDeformerInfo() override {}

    MPxGPUDeformer* createGPUDeformer() override;

    bool validateNodeInGraph(MDataBlock& block,
        const MEvaluationNode& evaluationNode,
        const MPlug& plug,
        MStringArray* messages) override;

    bool validateNodeValues(MDataBlock& block,
        const MEvaluationNode& evaluationNode,
        const MPlug& plug,
        MStringArray* messages) override;
};