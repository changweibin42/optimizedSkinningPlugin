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

#include <memory>
#include <vector>
#include <utility>
#include <unordered_map>

#include "math_core.h"
#include "algorithms.h"
#include "cor_paper.h"
#include "ddm_v4.h"
#include "ocd_math.h"
#include "mos_math.h"
#include "ocd_arc.h"
#include "ocd_hinge.h"
#include "ocd_centers.h"
#include "optimizedSkinClusterCoR.h"

class DlbGpu;

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

    MStatus preEvaluation(const MDGContext& context,
        const MEvaluationNode& evaluationNode) override;

    static MTypeId id;
    static MString typeName;

    // ---- 通用属性 ----
    static MObject aAlgorithmType;
    static MObject aDlbUseOpenCL;
    static MObject aDQSCompStrength;
    static MObject aForceFullCoR;
    // CoR Fixed Center Fit（对应参考实现 alg 8）：把外部求得、存于绑定对象
    // 空间的旋转中心偏移直接加到每个顶点的 CoR 中心上，再由同一条
    // cor_paper::vertexTransform 求值。插件本身不求解偏移。
    static MObject aFitOffsets;
    static MObject aCorTriLayerExpand;
    static MObject aCorVertLayerExpand;
    static MObject aCorIncBudgetDiv;
    static MObject aCorMosSecondaryWeight;
    // ---- OCD 铰链参数（algorithmType = 3 / 5） ----
    // 顶点 twist 跟随骨骼扭转的比例：1 = 完整跟随，0 = 完全剥离 twist。
    static MObject aOcdTwistWeight;
    // 旧版内侧避让：开启后内侧肩部按参考实现的 avoid_overlap 曲线放宽。
    static MObject aOcdAvoidOverlap;
    // OCD+Corrected CoR 的中心修正上限（世界单位）。
    static MObject aOcdCenterLimit;
    static MObject aOcdEnableCenterLimit;
    static MObject aOcdOuterGather;
    static MObject aOcdBendFlip;
    static MObject aOcdElbowGuide;
    // station=0 处折角的圆角窗口，单位是肢体半径的倍数。
    // 0 = 关闭（参考实现的硬旋转切换）。参考实现在这里有一个 15.5° 的折角：
    // 458 点的低密度网格跨过去了看不出来，16080 点就如实采出来成刀口。
    static MObject aOcdCreaseFillet;
    /// CoR 曲线引导开关（0=关闭，走原分组路径；1=开启，逐顶点积分）。
    static MObject aCorCurveGate;
    /// 曲线关联度的高斯尺度，单位是参与积分的绑定半径。
    static MObject aCorCurveSigma;
    /// 只衰减跨曲线贡献（B 组），同曲线因子恒为 1。
    static MObject aCorCurveCrossOnly;
    /// 手工曲线归属（每顶点一个 curveId）；空表示按骨骼链自动推导。
    static MObject aCorCurveAssign;
    // 显式指定主铰链的父/子骨骼名称（空 = 自动）。
    // 真实角色上如果自动选择有歧义，用它显式锁定。
    static MObject aOcdHingeParent;
    static MObject aOcdHingeChild;
    // 诊断：把实际选中的 parent/child/helper 名字写成字符串，
    // 供测试与真实角色校对。不影响 outputGeom。
    static MObject aOcdHingeReport;
    // 诊断：实际送入求解的 guide 与求出的 delta（局部归一坐标）。
    // 只读，不参与正常驱动。用来判断凹陷到底丢在哪一步。
    static MObject aOcdGuideDump;
    static MObject aOcdDeltaDump;
    static MObject aOcdCenterLimitDump;
    // 实验：中心偏移的低权重门控。
    // 子骨权重 w1 <= lo 时不用 delta，w1 >= hi 时全用，中间线性。
    // 默认 0 / 0 表示关闭（保持旧行为）。
    static MObject aOcdCenterGateLow;
    static MObject aOcdCenterGateHigh;
    // 可选的中心偏移注入（对象空间向量）。
    // 非空时直接矩阵这些偏移，跳过求解，用于与参考实现逐级对拍。
    static MObject aOcdCenterOffsets;
    // OCD 的 11 个补偿通道，顺序与参考实现的 CHANNELS 一致。
    static constexpr int ocdChannelCount = 11;
    static MObject aOcdCenterSlideIn;
    static MObject aOcdCenterSlideOut;
    static MObject aOcdSlideInPrev;
    static MObject aOcdSlideInNext;
    static MObject aOcdSlideOutPrev;
    static MObject aOcdSlideOutNext;
    static MObject aOcdSlideOrthoInPrev;
    static MObject aOcdSlideOrthoInNext;
    static MObject aOcdSlideOrthoOutPrev;
    static MObject aOcdSlideOrthoOutNext;
    static MObject aOcdBulge;
    static MObject aCorSigma;
    static MObject aCorDebugLog;
    static MObject aDdmIterations, aDdmLambda, aDdmKappa, aDdmAlpha, aDdmDebugLog;
    static MObject aDisplayCoR;

    // ---- MOS 可调参数 ----
    static MObject aMosInfluenceRadius;
    static MObject aMosCurveMode;
    static MObject aMosShapeGuide;
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

    // [DIAG] 诊断开关 bit mask：bit0=FJ_HARD, bit1=W_LINEAR, bit2=WC_UNITY
    static MObject aDiagFlags;

    bool getCoRData(std::vector<mathcore::Vec3>& out) const;

    struct MosParams {
        double influenceRadius = 1.0;
        static constexpr short arcSegments = 2;
        int    curveMode = arcSegments;
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

        // ---- CoR + MOS Secondary 专用 ----
        // 次级修正（bulge/slide）的总强度倍率，只在 algorithmType==4 时生效。
        // 1.0 = 完整叠加；0.0 = 关闭次级修正（退化为纯 CoR）；
        // 允许 >1 做夸张化。bulge/slide 各自权重仍然独立生效。
        double corMosSecondaryWeight = 1.0;

        bool nearlyEqual(const MosParams& rhs) const;
    };

private:
    // ================================================================
    // 各算法 deform 实现
    //   0 = CoR，1 = DLB，2 = DQS Compensation，3 = OCD，
    //   4 = MOS，5 = OCD + Corrected CoR，6 = DDM v4
    // ================================================================
    enum class CenteredMode { CoR, DdmV4, FixedFit };
    MStatus deformCoR(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int,
        CenteredMode mode = CenteredMode::CoR);
    MStatus deformDLB(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int,
        bool useOpenCL = false);
    std::unique_ptr<DlbGpu> m_dlbGpu;
    MStatus deformDQSComp(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);
    MStatus deformMOS(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int);
    MStatus deformOCD(MDataBlock&, MItGeometry&, const MMatrix&, unsigned int,
        bool correctCenters);

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

    struct CenteredSkinningCache {
        bool valid = false;
        unsigned int geometryIndex = 0;
        unsigned long long weightVersion = 0;
        double sigma = 0.0;
        ddm_v4::Settings ddmSettings;
        std::vector<unsigned int> influenceIds;
        std::vector<cor_paper::Weights> weights;
        std::vector<cor_paper::Weights> translationWeights;
        std::vector<mathcore::Vec3> centers;
        /// 生成这批中心时使用的曲线引导配置；任一变化都会使缓存失效。
        double curveGate = 0.0;
        double curveSigma = 0.0;
        /// 是否是「只衰减跨曲线」模式；变化时也要重算中心。
        double curveCrossOnly = -1.0;
    };
    CenteredSkinningCache m_paperCoR, m_ddmV4;

    MStatus rebuildCenteredSkinning(MDataBlock& block, unsigned int geometryIndex,
        double sigma, const ddm_v4::Settings* settings, CenteredSkinningCache& cache,
        const char*& stage);

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
        // 每个 joint 的"唯一" grandchild joint（用于当前 pose 判断内角方向）
        // -1 = leaf 或无唯一 child（多 child 或 0 child），此时 in/out 分区回退到 0.5/0.5
        std::vector<int> boneSoleChild;
        // 每个 influence 的最近「已蒙皮」祖先（logical index）；没有时为 -1。
        std::vector<int> boneParent;
        // 到那个祖先之间隔了几层关节（1 = 直接父骨，>1 表示跨过了未蒙皮关节）。
        std::vector<int> boneParentSteps;
        // 该骨骼是否还有子骨骼（用于判断它是否延续链条）。
        std::vector<char> boneHasJointChild;
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
        std::vector<ocd_math::Mat3> bindFrames;
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
        mos_math::Basis basis;
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
        bool direct = false;
        bool shapeGuide = false;
        double arcBefore = 0.0;
        double arcAfter = 0.0;
        double arcRadius = 0.0;
        std::vector<ocd_arc::Bind> arcVertices;
        unsigned int geometryIndex = 0;
        MMatrix bindSpace;
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
        const MosParams& params, const MMatrix* bindSpace = nullptr,
        unsigned int geometryIndex = 0);

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
