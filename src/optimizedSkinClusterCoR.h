#pragma once

#include <maya/MPointArray.h>

#include <vector>
#include <utility>
#include <unordered_map>

#include "math_core.h"
#include "algorithms.h"

namespace optimized_skin_internal {

    struct SparseW {
        std::vector<std::pair<unsigned int, double>> v;
        void clear() { v.clear(); }
        size_t size() const { return v.size(); }
        void buildFrom(const WeightMap& w);
        void buildFromMap(const std::unordered_map<unsigned int, double>& w);
    };

    struct TriInfo {
        mathcore::Vec3 center = { 0.0, 0.0, 0.0 };
        double area = 0.0;
        SparseW w;
    };

    struct SubTri {
        mathcore::Vec3 center = { 0.0, 0.0, 0.0 };
        double area = 0.0;
        SparseW w;
    };

    // ---------- CoR 辅助函数 ----------
    double weightL2DistSq(const SparseW& a, const SparseW& b);
    double paper_similarity(const SparseW& wp, const SparseW& wv, double sigma);
    SparseW midSparseW(const SparseW& a, const SparseW& b);
    SparseW avg3SparseW(const SparseW& a, const SparseW& b, const SparseW& c);
    bool sparseEqual(const SparseW& a, const SparseW& b);

    TriInfo buildSingleTriInfo(
        const MPointArray& points,
        const unsigned int* triVerts,
        const std::vector<SparseW>& allW);

    std::vector<SubTri> subdivideSingleTriangle(
        const MPointArray& points,
        const unsigned int* triVerts,
        const std::vector<SparseW>& allW);

    // ★ 新增：Schlömer 2011 最远点优化点集
    std::vector<unsigned int> farthestPointClustering(
        const std::vector<SparseW>& allW,
        unsigned int numClusters);

    mathcore::Vec3 computeSingleVertexCoR(
        const SparseW& wi,
        const MPointArray& points,
        unsigned int vi,
        const std::vector<std::vector<SubTri>>& triSubdivs,
        const std::vector<std::vector<unsigned int>>& triAdj,
        const std::vector<std::vector<unsigned int>>& vertToTris,
        unsigned int K,
        const std::vector<std::vector<unsigned int>>& clusterMembers,
        const std::vector<std::vector<std::pair<double, unsigned int>>>& vertClusterDists,
        const std::vector<SparseW>& allW,
        unsigned int numTris);

} // namespace optimized_skin_internal