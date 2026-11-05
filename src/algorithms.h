#pragma once
#include "math_core.h"
#include <vector>
#include <unordered_map>

using namespace mathcore;

// 单个顶点的权重映射：骨骼逻辑索引 -> 权重
using WeightMap = std::unordered_map<unsigned int, double>;

// ================================================================
// 四种蒙皮算法（纯数学，无 Maya 依赖）
// ================================================================

// LBS
Vec3 lbs_vertex(const Vec3& pos, const WeightMap& weights,
    const std::vector<RigidTransform>& Cs);

// DLB (Kavan 2008 Algorithm 1)
Vec3 dlb_vertex(const Vec3& pos, const WeightMap& weights,
    const std::vector<RigidTransform>& Cs);

// CoR (Le & Hodgins 2016 Algorithm 1，运行时)
Vec3 cor_vertex(const Vec3& pos, const Vec3& cor,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs);

// QLERP (Kavan 2008 Algorithm 2)
Quat qlerp(const WeightMap& weights,
    const std::vector<RigidTransform>& Cs);

// DQS 补偿 (Sulema & Rudenko 2020 Algorithm 1)
Vec3 dqs_compensate_vertex(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const WeightMap& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    const std::vector<int>& depths,
    double strength = 1.0);

// ================================================================
// v2 接口
// ================================================================

Quat qlerp_v2(const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs);

Vec3 cor_vertex_v2(const Vec3& pos, const Vec3& cor,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs);

Vec3 dlb_vertex_v2(const Vec3& pos,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs);

Vec3 dqs_compensate_vertex_v2(
    const Vec3& rest_pos,
    const Vec3& dqs_pos,
    const std::vector<std::pair<unsigned int, float>>& weights,
    const std::vector<RigidTransform>& Cs,
    const std::vector<Vec3>& bone_dirs,
    const std::vector<Vec3>& joint_positions,
    const std::vector<int>& depths,
    double strength);