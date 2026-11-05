#pragma once

#include <maya/MPxData.h>
#include <maya/MTypeId.h>
#include <maya/MString.h>
#include <vector>

struct RigidData {
    float qx = 0, qy = 0, qz = 0, qw = 1;
    float tx = 0, ty = 0, tz = 0;
};

struct Vec3Data {
    float x = 0, y = 0, z = 0;
};

struct SkinVertex {
    int   boneIds[4] = { 0, 0, 0, 0 };
    float weights[4] = { 0, 0, 0, 0 };
    int   boneCount = 0;
};

class SkinClusterData : public MPxData {
public:
    static const MString typeName;
    static MTypeId       id;

    SkinClusterData();
    virtual ~SkinClusterData();

    virtual void    copy(const MPxData& other) override;
    virtual MTypeId typeId() const override;
    virtual MString name()   const override;

    static void* creator();

    // ---- 载荷 ----
    int   algorithmType = 0;
    float dqsStrength = 1.0f;
    bool  useGPU = false;
    int   weightVersion = 0;

    std::vector<RigidData>  restList;      // rest world 旋转的逆
    std::vector<RigidData>  poseList;      // C = P · R⁻¹
    std::vector<SkinVertex> vertices;
    std::vector<Vec3Data>   cors;
    std::vector<Vec3Data>   boneDirs;      // ★ 每骨骼的 rest 时"沿肢体方向"（世界空间）
    std::vector<Vec3Data>   jointPositions; // rest 时每骨骼世界位置
    std::vector<int>        depths;         // ★ 每骨骼的层级深度（0=根）
};