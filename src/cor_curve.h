#pragma once

// ================================================================
// CoR 曲线引导用的骨骼链曲线（方向 A 第一版）
//
// 由绑定姿态的关节位置构造折线曲线，并给出每个顶点的：
//   · curveId      归属的链（多条）
//   · membership   对各链的归属权重（软权重，避免硬切换接缝）
//   · arcLength    在该链曲线上的绑定弧长
//
// 顶点归属按「顶点对该链的蒙皮权重总量」分配。两条链共享根骨时，
// 共享骨的权重不能单独决定归属，所以先扣掉只属于共享骨的份额，
// 再按各链独占骨的权重归一。
//
// 三角形内的积分样本在**同一 curveId 内**插值弧长；跨曲线边界先
// 插值各曲线的归属权重，避免把两条不同链上的弧长平均成一个没有
// 对应位置的参数。
// ================================================================

#include <cmath>
#include <cstddef>
#include <limits>
#include <utility>
#include <vector>

#include "math_core.h"

namespace cor_curve {

using Vec3 = mathcore::Vec3;

struct Chain {
    /// 该链包含的 influence（骨骼）索引，按父到子顺序。
    std::vector<unsigned int> bones;
    /// 折线顶点（绑定位姿的关节位置），长度 = bones.size() + 1，末点是末端关节位置。
    std::vector<Vec3> points;
    /// 累计弧长，长度与 points 相同，arc[0] = 0。
    std::vector<double> arc;
    double length = 0.0;
    /// 链的包围体，用于快速排除明显不相关的顶点。
    Vec3 boxMin{0.0, 0.0, 0.0};
    Vec3 boxMax{0.0, 0.0, 0.0};
};

struct VertexBinding {
    /// 归属权重：与 chains 等长。全部为 0 表示该顶点不参与曲线引导。
    std::vector<double> membership;
    /// 在各链上的弧长；membership 为 0 的项无意义。
    std::vector<double> arcLength;
};

struct Bind {
    std::vector<Chain> chains;
    std::vector<VertexBinding> vertices;
    /// 参与归属判断的全部骨骼数，用于把 influenceId 映射到链上。
    unsigned int influenceCount = 0;
    /// 平均链长，作为距离归一化的尺度；0 表示没有可用链。
    double meanChainLength = 0.0;
};

namespace detail {

inline double dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

inline Vec3 sub(const Vec3& a, const Vec3& b) {
    return {a.x - b.x, a.y - b.y, a.z - b.z};
}

inline Vec3 add(const Vec3& a, const Vec3& b) {
    return {a.x + b.x, a.y + b.y, a.z + b.z};
}

inline Vec3 scale(const Vec3& a, double s) {
    return {a.x * s, a.y * s, a.z * s};
}

inline double length(const Vec3& a) {
    return std::sqrt(dot(a, a));
}

/// 把 p 投影到折线上，返回累计弧长。
inline double projectArc(const Chain& chain, const Vec3& p, double* outDistance = nullptr) {
    if (chain.points.size() < 2) {
        if (outDistance) *outDistance = std::numeric_limits<double>::max();
        return 0.0;
    }
    double bestDistance = std::numeric_limits<double>::max();
    double bestArc = 0.0;
    for (size_t i = 0; i + 1 < chain.points.size(); ++i) {
        const Vec3 a = chain.points[i];
        const Vec3 ab = sub(chain.points[i + 1], a);
        const double len2 = dot(ab, ab);
        double t = 0.0;
        if (len2 > 1e-18) t = dot(sub(p, a), ab) / len2;
        t = std::max(0.0, std::min(1.0, t));
        const Vec3 closest = add(a, scale(ab, t));
        const double d = length(sub(p, closest));
        if (d < bestDistance) {
            bestDistance = d;
            bestArc = chain.arc[i] + t * (chain.arc[i + 1] - chain.arc[i]);
        }
    }
    if (outDistance) *outDistance = bestDistance;
    return bestArc;
}

inline void computeExtent(Chain& chain) {
    if (chain.points.empty()) return;
    chain.boxMin = chain.boxMax = chain.points.front();
    for (const auto& p : chain.points) {
        chain.boxMin.x = std::min(chain.boxMin.x, p.x);
        chain.boxMin.y = std::min(chain.boxMin.y, p.y);
        chain.boxMin.z = std::min(chain.boxMin.z, p.z);
        chain.boxMax.x = std::max(chain.boxMax.x, p.x);
        chain.boxMax.y = std::max(chain.boxMax.y, p.y);
        chain.boxMax.z = std::max(chain.boxMax.z, p.z);
    }
}

} // namespace detail

/// 由关节位置构造链的折线与累计弧长。
inline void buildChainGeometry(Chain& chain) {
    chain.arc.assign(chain.points.size(), 0.0);
    for (size_t i = 1; i < chain.points.size(); ++i)
        chain.arc[i] = chain.arc[i - 1] + detail::length(
            detail::sub(chain.points[i], chain.points[i - 1]));
    chain.length = chain.arc.empty() ? 0.0 : chain.arc.back();
    detail::computeExtent(chain);
}

/// 绑定期：按蒙皮权重把每个顶点分配到链上，并记录弧长。
///
/// weights[i] 是 (influenceId, weight) 列表，influenceId 是骨骼的
/// 逻辑索引，chains[b].bones 里存的也是同一套索引。
inline Bind bindVertices(const std::vector<Chain>& chains,
    const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<std::vector<std::pair<unsigned int, double>>>& weights) {
    Bind bind;
    bind.chains = chains;
    bind.vertices.resize(restPoints.size());
    double total = 0.0;
    for (const auto& chain : chains) total += std::max(chain.length, 0.0);
    bind.meanChainLength = chains.empty() ? 0.0 : total / chains.size();

    // 统计每条骨被几条链引用。共享骨（>1）不参与独占份额。
    std::vector<unsigned int> boneUse(chains.size() + 1, 0);
    unsigned int maxBone = 0;
    for (const auto& chain : chains)
        for (unsigned int bone : chain.bones) maxBone = std::max(maxBone, bone);
    std::vector<unsigned int> use(maxBone + 1, 0);
    for (const auto& chain : chains)
        for (unsigned int bone : chain.bones)
            if (bone < use.size()) ++use[bone];
    boneUse.resize(use.size());

    for (size_t v = 0; v < restPoints.size(); ++v) {
        auto& out = bind.vertices[v];
        out.membership.assign(chains.size(), 0.0);
        out.arcLength.assign(chains.size(), 0.0);
        if (v >= weights.size()) continue;

        // 每条链的独占权重 = 该链骨骼上的权重之和，但只计非共享骨。
        std::vector<double> exclusive(chains.size(), 0.0);
        for (const auto& [bone, w] : weights[v]) {
            if (bone >= use.size() || w <= 0.0) continue;
            for (size_t c = 0; c < chains.size(); ++c) {
                for (unsigned int cb : chains[c].bones) {
                    if (cb != bone) continue;
                    if (use[bone] <= 1) exclusive[c] += w;
                    break;
                }
            }
        }
        double sum = 0.0;
        for (double e : exclusive) sum += e;
        if (sum <= 1e-12) continue;

        for (size_t c = 0; c < chains.size(); ++c) {
            out.membership[c] = exclusive[c] / sum;
            if (out.membership[c] <= 0.0) continue;
            out.arcLength[c] = detail::projectArc(chains[c], restPoints[v]);
        }
    }
    return bind;
}

} // namespace cor_curve
