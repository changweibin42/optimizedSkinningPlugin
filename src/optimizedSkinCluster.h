#pragma once

#include <maya/MPxSkinCluster.h>
#include <maya/MTypeId.h>
#include <maya/MObject.h>
#include <maya/MStatus.h>
#include <maya/MString.h>
#include <maya/MDataBlock.h>
#include <maya/MItGeometry.h>
#include <maya/MMatrix.h>
#include <maya/MPointArray.h>
#include <maya/MPlug.h>
#include <maya/MPlugArray.h>

#include <vector>
#include <utility>
#include <unordered_map>

#include "math_core.h"
#include "algorithms.h"
#include "ocd_math.h"
#include "optimizedSkinClusterCoR.h"

class OptimizedSkinCluster : public MPxSkinCluster {
public:
    OptimizedSkinCluster();
    virtual ~OptimizedSkinCluster();

    static void* creator();
    static MStatus initialize();

    virtual void postConstructor() override;

    virtual MStatus deform(MDataBlock& block, MItGeometry& iter,
        const MMatrix& mat, unsigned int multiIndex) override;

    virtual MStatus setDependentsDirty(const MPlug& plug,
        MPlugArray& plugArray) override;

    static MTypeId id;
    static MString typeName;

    // ---- 通用属性 ----
    static MObject aAlgorithmType;
    static MObject aDQSCompStrength;
    static MObject aForceFullCoR;

    static MObject aCorTriLayerExpand;
    static MObject aCorVertLayerExpand;
    static MObject aCorIncBudgetDiv;
    static MObject aDisplayCoR;

    // ---- MOS 可调参数 ----
    static MObject aMosInfluenceRadius;
    static MObject aMosCurveMode;
    static MObject aMosEndLock;
    static MObject aMosCurveSamples;
    static MObject aMosDebugLog;

    // ---- MOS rig correction（bulge / slide） ----
    static MObject aMosInBulgeWeight;
    static MObject aMosOutBulgeWeight;
    static MObject aMosSideBulgeWeight;
    static MObject aMosCenterSlideInWeight;
    static MObject aMosCenterSlideOutWeight;

    static MObject aMosSlideInWeightPrev;
    static MObject aMosSlideInWeightNext;
    static MObject aMosSlideOutWeightPrev;
    static MObject aMosSlideOutWeightNext;
    static MObject aMosSlideOrthoInWeightPrev;
    static MObject aMosSlideOrthoInWeightNext;
    static MObject aMosSlideOrthoOutWeightPrev;
    static MObject aMosSlideOrthoOutWeightNext;

    static MObject aMosUseManualInVector;
    static MObject aMosManualInVector;

    static MObject aMosTwistWeight;

    static MObject aMosScaleWeight;
    static MObject aMosScaleMode;

    bool getCoRData(std::vector<mathcore::Vec3>& out) const;

    struct MosParams {
        double influenceRadius = 0.6;
        int    curveMode = 0;
        bool   endLock = true;
        int    curveSamples = 1024;
        bool   debugLog = false;

        // rig correction（专利 US9418465B2）：三向 bulge 由 in vector 界定的独立方向
        double inBulgeWeight = 0.0;         ///< in  bulge：内侧顶点沿 -in 向外鼓（负值 = 向内挤压）
        double outBulgeWeight = 0.0;           ///< out bulge：外侧顶点沿 -in 向外鼓
        double sideBulgeWeight = 0.0;       ///< side bulge：沿 cross(in, bone_axis) 侧向鼓
        double centerSlideInWeight = 0.0;
        double centerSlideOutWeight = 0.0;

        // ---- 方案B: 8个slide变体 ----
        double slideInWeight_prev = 0.0;
        double slideInWeight_next = 0.0;
        double slideOutWeight_prev = 0.0;
        double slideOutWeight_next = 0.0;
        double slideOrthoInWeight_prev = 0.0;
        double slideOrthoInWeight_next = 0.0;
        double slideOrthoOutWeight_prev = 0.0;
        double slideOrthoOutWeight_next = 0.0;

        // ---- 手动 in vector（覆盖 bulge 径向方向）----
        bool           useManualInVector = false;
        mathcore::Vec3 manualInVector = { 0.0, 1.0, 0.0 };

        // ---- Twist 主动控制 ----
        // [0,1]：皮肤跟随骨骼扭转的比例；1.0 = 完整跟随（默认，零回归），
        //        0.0 = 完全剥离 twist，只跟 swing（如手背不随前臂旋前旋后）
        double twistWeight = 1.0;

        // ---- Scale 主动控制 ----
        // scaleWeight ∈ [0,1]：0 = 不跟随骨骼缩放（默认，零回归），
        //                      1 = 完整跟随骨骼缩放（squash/stretch 动画）
        // scaleMode: 0 = 关闭，1 = 轴向对称（沿骨骼轴），2 = 各向异性
        double scaleWeight = 0.0;
        int    scaleMode = 1;

        bool nearlyEqual(const MosParams& rhs) const;
    };

private:
    // ================================================================
    // 四种算法的 deform 实现
    // ================================================================
    MStatus deformCoR(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);
    MStatus deformDLB(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);
    MStatus deformDQSComp(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);
    MStatus deformMOS(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);

    bool readBoneTransforms(MDataBlock&,
        std::vector<mathcore::RigidTransform>& out_rest,
        std::vector<mathcore::RigidTransform>& out_pose,
        std::vector<mathcore::Quat>& out_pose_qs);

    /// 读取骨骼完整 3×3 线性部分（含缩放/shear）。
    /// readBoneTransforms 返回的 RigidTransform 只保留旋转与平移，丢失缩放；
    /// Scale 主动控制需要真实缩放，故单独读取 matrix[] × bindPreMatrix[] 的 4×4 并提取列。
    bool readBoneLinearParts(MDataBlock& block, const MMatrix& worldMat,
        std::vector<mathcore::Vec3>& col0,
        std::vector<mathcore::Vec3>& col1,
        std::vector<mathcore::Vec3>& col2);

    // ================================================================
    // CoR 缓存（不动）
    // ================================================================
    struct CoRCache {
        bool valid = false;
        unsigned int numVertices = 0;
        std::vector<mathcore::Vec3> cors;

        unsigned int numTris = 0;
        std::vector<std::vector<unsigned int>> triVertCache;
        std::vector<optimized_skin_internal::SparseW> allW;
        std::vector<optimized_skin_internal::TriInfo> tris;
        std::vector<std::vector<optimized_skin_internal::SubTri>> triSubdivs;
        std::vector<std::vector<unsigned int>> triAdj;
        std::vector<std::vector<unsigned int>> vertToTris;
        unsigned int K = 0;
        std::vector<unsigned int> centers;
        std::vector<std::vector<unsigned int>> clusterMembers;
        std::vector<std::vector<std::pair<double, unsigned int>>> vertClusterDists;
    };
    CoRCache m_corCache;

    bool computeCoRCache(MDataBlock& block, unsigned int numVertices);
    void updateCoRCacheIncremental(MDataBlock& block,
        const std::vector<unsigned int>& dirtyList,
        int triLayers, int vertLayers);
    int computeVertLayers(unsigned int dirtyCount,
        unsigned int numVertices,
        int vertLayerOverride,
        int budgetDiv);

    std::vector<unsigned char> m_dirtyVertFlags;
    bool m_hasPlugDirty = false;
    bool m_needFullWeightScan = false;

    void markDirtyVert(unsigned int vi);
    void clearDirtyFlags();

    // ================================================================
    // 顶点权重缓存（不动）
    // ================================================================
    struct VtxCache {
        bool valid = false;
        unsigned long long weightVersion = 0;
        unsigned int numVertices = 0;
        std::vector<std::vector<std::pair<unsigned int, float>>> vertices;
    };
    VtxCache m_vtxCache;
    unsigned long long m_currentWeightVersion = 0;

    void rebuildVtxCache(MDataBlock& block, unsigned int numVertices);

    // ================================================================
    // 骨骼信息缓存（不动）
    // ================================================================
    struct BoneInfo {
        bool valid = false;
        unsigned int numBones = 0;
        std::vector<int> depths;
        std::vector<mathcore::Vec3> boneDirRest;
        std::vector<mathcore::Vec3> jointPosRest;
        unsigned long long topologySignature = 0;
    };
    BoneInfo m_boneInfo;

    void collectBoneInfo(MDataBlock& block);

    // ================================================================
    // MOS 缓存
    //
    // 数学主链（专利 US9786083B2）：
    //   deform: v_i = Σ_j (p_ij + F_ij·O_i) · w_ij          （EQ.5）
    //   bind:   O_i = m_i - Σ_j p_ij^bind · w_ij            （EQ.9 的 F_ij=I 特例）
    //   F_ij = mat3_from_quat(Cs[bone_j].q)                 （纯旋转）
    //   w_ij = B-spline 基函数在 τ 处的值
    //   p_ij = apply_transform(Cs[bone_j], cvs_rest[j])
    // ================================================================
    struct MosCurve {
        std::vector<mathcore::Vec3> cvs_rest;
        std::vector<unsigned int>  cvs_bone;
        /// 每个控制点在 chain 上的归一化骨骼参数 u_j ∈ [0,1]，与 cvs_rest 等长。
        /// 第 i 段骨骼内相对位置 r 的控制点取 u = (i + r) / 段总数，弯曲关节处恰为 0.5。
        /// 该值将作为 B-spline 内节点使用，使线性精度 Σ N_j(τ)·u_j = τ 成立，
        /// 从而在关节处把父/子基函数权重精确锁定为 0.5/0.5，消除角偏转衰减。
        std::vector<double> cvs_u;
        std::vector<double> knots;
        int degree = 3;
        std::vector<unsigned int> chain_bones;
        /// 该曲线负责的弯曲关节在 chain_bones 中的下标。
        /// per_joint 模式静态指定为 j；single_curve 模式为 -1（deform 时动态选最大旋转关节）
        int bend_joint_idx = -1;
        mathcore::Vec3 joint_anchor_bind = { 0.0, 0.0, 0.0 }; // 关节锚点（骨骼起点）
    };

    struct MosPerCurveBind {
        double tau = 0.0;                        ///< 顶点在曲线上的 B-spline 参数
        mathcore::Vec3 O = { 0.0, 0.0, 0.0 };    ///< EQ.9 解出的单偏移向量
        double w_c = 0.0;                        ///< 曲线对顶点的混合权重
        int vertex_joint = -1;                   ///< 顶点在 chain 上最近 joint 的下标
    };

    struct MosVertexBind {
        std::vector<MosPerCurveBind> per_curve;
    };

    struct MosCache {
        bool valid = false;
        unsigned int numVertices = 0;
        unsigned int n_curves = 0;
        double avg_bone_length = 1.0;
        MosParams params;
        std::vector<MosCurve> curves;
        std::vector<MosVertexBind> verts;
    };
    MosCache m_mosCache;

    MosParams readMosParams(MDataBlock& block);

    /// 构建 MOS 全量缓存；失败时调用方回退 deformDLB
    bool computeMosCache(MDataBlock& block, unsigned int numVertices,
        const MosParams& params);

    /// 控制点级 bulge / centerSlide 修正（专利 US9418465B2）。
    ///
    /// 核心原则：所有次级操作都在控制点位置 p_j 上修改，而非在 B-spline 输出上后处理。
    /// B-spline 的局部支撑让修改自动限制在对应曲线段；w_j（该控制点对弯曲关节的从属权重）
    /// 进一步把有效修改限制在关节附近的控制点上，避免影响远端控制点。
    ///
    /// 参数 j：当前控制点在 cv.cvs_rest / cv_bend_weight 中的索引
    /// 参数 cv_bend_weight：与 cv.cvs_rest 等长，每个控制点对弯曲关节的从属权重（0~1）
    mathcore::Vec3 applyMosSecondaryOps(
        const MosCurve& cv, int j, int bend_idx,
        const std::vector<double>& cv_bend_weight,
        const MosPerCurveBind& pc,
        const std::vector<mathcore::RigidTransform>& rest,
        const std::vector<mathcore::RigidTransform>& Cs,
        double avg_bone_length,
        const MosParams& P,
        const mathcore::Vec3& p_j) const;

    MPlug m_envelopePlug;
    double readEnvelopeFromNode();
    mathcore::Vec3 applyEnvelope(const mathcore::Vec3& orig,
        const mathcore::Vec3& deformed,
        double envelope);
};