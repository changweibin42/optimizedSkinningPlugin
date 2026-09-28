#pragma once
#include "math_core.h"
#include <vector>
#include <unordered_map>

using namespace mathcore;

// 单个顶点的权重映射：骨骼逻辑索引 -> 权重
using WeightMap = std::unordered_map<unsigned int, double>;

// ================================================================
// 四种蒙皮算法（纯数学，无 Maya 依赖）
// 输入统一为：
//   Cs       — 每根骨骼的 C = P · R⁻¹
//   weights  — 当前顶点的权重映射
// ================================================================

// LBS
Vec3 lbs_vertex(const Vec3& pos, const WeightMap& weights,
                const std::vector<RigidTransform>& Cs);

// DLB (Dual Quaternion Linear Blending)
Vec3 dlb_vertex(const Vec3& pos, const WeightMap& weights,
                const std::vector<RigidTransform>& Cs);

// CoR (Optimized Centers of Rotation)
Vec3 cor_vertex(const Vec3& pos, const Vec3& cor,
                const WeightMap& weights,
                const std::vector<RigidTransform>& Cs);

// QLERP 混合四元数
Quat qlerp(const WeightMap& weights,
           const std::vector<RigidTransform>& Cs);

// DQS 补偿
//   rest_pos / dqs_pos — 静止姿势和 DQS 变形后的位置
//   bone_dirs         — 每根骨骼的世界 Y 轴方向
//   joint_positions   — 静止姿势下每根骨骼的世界位置
Vec3 dqs_compensate_vertex(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    double strength = 1.0);