#include "optimizedSkinClusterCoR.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace optimized_skin_internal {

    // ================================================================
    // SparseW
    // ================================================================
    void SparseW::buildFrom(const WeightMap& w) {
        v.clear();
        v.reserve(w.size());
        for (const auto& kv : w) {
            if (kv.second > 1e-12) v.push_back({ kv.first, kv.second });
        }
        std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    void SparseW::buildFromMap(const std::unordered_map<unsigned int, double>& w) {
        v.clear();
        v.reserve(w.size());
        for (const auto& kv : w) {
            if (kv.second > 1e-12) v.push_back({ kv.first, kv.second });
        }
        std::sort(v.begin(), v.end(),
            [](const auto& a, const auto& b) { return a.first < b.first; });
    }

    // ================================================================
    // weightL2DistSq
    // ================================================================
    double weightL2DistSq(const SparseW& a, const SparseW& b) {
        double s = 0.0;
        size_t i = 0, j = 0;
        size_t na = a.v.size(), nb = b.v.size();
        while (i < na && j < nb) {
            unsigned int ia = a.v[i].first;
            unsigned int ib = b.v[j].first;
            if (ia == ib) {
                double d = a.v[i].second - b.v[j].second;
                s += d * d;
                ++i; ++j;
            }
            else if (ia < ib) {
                s += a.v[i].second * a.v[i].second;
                ++i;
            }
            else {
                s += b.v[j].second * b.v[j].second;
                ++j;
            }
        }
        while (i < na) { s += a.v[i].second * a.v[i].second; ++i; }
        while (j < nb) { s += b.v[j].second * b.v[j].second; ++j; }
        return s;
    }

    // ================================================================
    // paper_similarity
    // ================================================================
    double paper_similarity(const SparseW& wp, const SparseW& wv, double sigma) {
        size_t n = 0;
        {
            size_t i = 0, j = 0;
            size_t na = wp.v.size(), nb = wv.v.size();
            while (i < na && j < nb) {
                if (wp.v[i].first == wv.v[j].first) { ++i; ++j; }
                else if (wp.v[i].first < wv.v[j].first) { ++i; }
                else { ++j; }
                ++n;
            }
            n += (na - i) + (nb - j);
        }
        if (n < 2) return 0.0;

        static thread_local std::vector<double> Wp, Wv;
        Wp.assign(n, 0.0);
        Wv.assign(n, 0.0);
        {
            size_t k = 0, i = 0, j = 0;
            size_t na = wp.v.size(), nb = wv.v.size();
            while (i < na && j < nb) {
                unsigned int ia = wp.v[i].first;
                unsigned int ib = wv.v[j].first;
                if (ia == ib) {
                    Wp[k] = wp.v[i].second; Wv[k] = wv.v[j].second;
                    ++i; ++j; ++k;
                }
                else if (ia < ib) {
                    Wp[k] = wp.v[i].second; ++i; ++k;
                }
                else {
                    Wv[k] = wv.v[j].second; ++j; ++k;
                }
            }
            while (i < na) { Wp[k] = wp.v[i].second; ++i; ++k; }
            while (j < nb) { Wv[k] = wv.v[j].second; ++j; ++k; }
        }

        double sum = 0.0;
        double inv_s2 = 1.0 / (sigma * sigma);
        for (size_t a = 0; a < n; ++a) {
            if (Wp[a] < 1e-9 && Wv[a] < 1e-9) continue;
            for (size_t b = 0; b < n; ++b) {
                if (a == b) continue;
                double d = Wp[a] * Wv[b] - Wp[b] * Wv[a];
                sum += Wp[a] * Wp[b] * Wv[a] * Wv[b] * std::exp(-d * d * inv_s2);
            }
        }
        return sum;
    }

    // ================================================================
    // midSparseW / avg3SparseW / sparseEqual
    // ================================================================
    SparseW midSparseW(const SparseW& a, const SparseW& b) {
        SparseW m;
        size_t i = 0, j = 0;
        size_t na = a.v.size(), nb = b.v.size();
        m.v.reserve(na + nb);
        while (i < na && j < nb) {
            unsigned int ia = a.v[i].first, ib = b.v[j].first;
            if (ia == ib) {
                m.v.push_back({ ia, (a.v[i].second + b.v[j].second) * 0.5 });
                ++i; ++j;
            }
            else if (ia < ib) {
                m.v.push_back({ ia, a.v[i].second * 0.5 }); ++i;
            }
            else {
                m.v.push_back({ ib, b.v[j].second * 0.5 }); ++j;
            }
        }
        while (i < na) { m.v.push_back({ a.v[i].first, a.v[i].second * 0.5 }); ++i; }
        while (j < nb) { m.v.push_back({ b.v[j].first, b.v[j].second * 0.5 }); ++j; }
        return m;
    }

    SparseW avg3SparseW(const SparseW& a, const SparseW& b, const SparseW& c) {
        std::unordered_map<unsigned int, double> tmp;
        tmp.reserve(a.v.size() + b.v.size() + c.v.size());
        for (const auto& kv : a.v) tmp[kv.first] += kv.second / 3.0;
        for (const auto& kv : b.v) tmp[kv.first] += kv.second / 3.0;
        for (const auto& kv : c.v) tmp[kv.first] += kv.second / 3.0;
        SparseW m;
        m.buildFromMap(tmp);
        return m;
    }

    bool sparseEqual(const SparseW& a, const SparseW& b) {
        if (a.v.size() != b.v.size()) return false;
        for (size_t i = 0; i < a.v.size(); ++i) {
            if (a.v[i].first != b.v[i].first) return false;
            if (std::abs(a.v[i].second - b.v[i].second) > 1e-6) return false;
        }
        return true;
    }

    // ================================================================
    // buildSingleTriInfo
    // ================================================================
    TriInfo buildSingleTriInfo(
        const MPointArray& points,
        const unsigned int* triVerts,
        const std::vector<SparseW>& allW) {

        int i0 = triVerts[0], i1 = triVerts[1], i2 = triVerts[2];
        mathcore::Vec3 v0 = { (double)points[i0].x, (double)points[i0].y, (double)points[i0].z };
        mathcore::Vec3 v1 = { (double)points[i1].x, (double)points[i1].y, (double)points[i1].z };
        mathcore::Vec3 v2 = { (double)points[i2].x, (double)points[i2].y, (double)points[i2].z };

        TriInfo ti;
        ti.center = {
            (v0.x + v1.x + v2.x) / 3.0,
            (v0.y + v1.y + v2.y) / 3.0,
            (v0.z + v1.z + v2.z) / 3.0
        };
        mathcore::Vec3 e1 = mathcore::v3_sub(v1, v0);
        mathcore::Vec3 e2 = mathcore::v3_sub(v2, v0);
        mathcore::Vec3 cr = {
            e1.y * e2.z - e1.z * e2.y,
            e1.z * e2.x - e1.x * e2.z,
            e1.x * e2.y - e1.y * e2.x
        };
        ti.area = 0.5 * mathcore::v3_len(cr);

        std::unordered_map<unsigned int, double> wsum;
        for (const auto& kv : allW[i0].v) wsum[kv.first] += kv.second;
        for (const auto& kv : allW[i1].v) wsum[kv.first] += kv.second;
        for (const auto& kv : allW[i2].v) wsum[kv.first] += kv.second;
        for (auto& kv : wsum) kv.second /= 3.0;
        ti.w.buildFromMap(wsum);
        return ti;
    }

    // ================================================================
    // subdivideSingleTriangle
    // ================================================================
    std::vector<SubTri> subdivideSingleTriangle(
        const MPointArray& points,
        const unsigned int* triVerts,
        const std::vector<SparseW>& allW) {

        const double SUBDIV_EPS_SQ = 0.1 * 0.1;
        const int    MAX_SUBDIV_DEPTH = 6;

        struct SubdivFrame {
            mathcore::Vec3 p0, p1, p2;
            SparseW w0, w1, w2;
            int depth = 0;
        };

        int i0 = triVerts[0], i1 = triVerts[1], i2 = triVerts[2];
        mathcore::Vec3 v0 = { (double)points[i0].x, (double)points[i0].y, (double)points[i0].z };
        mathcore::Vec3 v1 = { (double)points[i1].x, (double)points[i1].y, (double)points[i1].z };
        mathcore::Vec3 v2 = { (double)points[i2].x, (double)points[i2].y, (double)points[i2].z };

        std::vector<SubTri> result;
        std::vector<SubdivFrame> stack;
        stack.push_back({ v0, v1, v2, allW[i0], allW[i1], allW[i2], 0 });

        auto midP = [](const mathcore::Vec3& a, const mathcore::Vec3& b) {
            return mathcore::Vec3{ (a.x + b.x) * 0.5, (a.y + b.y) * 0.5, (a.z + b.z) * 0.5 };
            };

        while (!stack.empty()) {
            SubdivFrame f = std::move(stack.back());
            stack.pop_back();

            double d01 = weightL2DistSq(f.w0, f.w1);
            double d12 = weightL2DistSq(f.w1, f.w2);
            double d20 = weightL2DistSq(f.w2, f.w0);
            double dmax = std::max(std::max(d01, d12), d20);

            if (f.depth >= MAX_SUBDIV_DEPTH || dmax <= SUBDIV_EPS_SQ) {
                SubTri st;
                st.center = {
                    (f.p0.x + f.p1.x + f.p2.x) / 3.0,
                    (f.p0.y + f.p1.y + f.p2.y) / 3.0,
                    (f.p0.z + f.p1.z + f.p2.z) / 3.0
                };
                mathcore::Vec3 e1 = mathcore::v3_sub(f.p1, f.p0);
                mathcore::Vec3 e2 = mathcore::v3_sub(f.p2, f.p0);
                mathcore::Vec3 cr = {
                    e1.y * e2.z - e1.z * e2.y,
                    e1.z * e2.x - e1.x * e2.z,
                    e1.x * e2.y - e1.y * e2.x
                };
                st.area = 0.5 * mathcore::v3_len(cr);
                st.w = avg3SparseW(f.w0, f.w1, f.w2);
                result.push_back(std::move(st));
                continue;
            }

            if (d01 >= d12 && d01 >= d20) {
                mathcore::Vec3 pm = midP(f.p0, f.p1);
                SparseW wm = midSparseW(f.w0, f.w1);
                SubdivFrame f1, f2;
                f1.p0 = f.p0; f1.p1 = pm; f1.p2 = f.p2;
                f1.w0 = f.w0; f1.w1 = wm; f1.w2 = f.w2; f1.depth = f.depth + 1;
                f2.p0 = pm; f2.p1 = f.p1; f2.p2 = f.p2;
                f2.w0 = wm; f2.w1 = f.w1; f2.w2 = f.w2; f2.depth = f.depth + 1;
                stack.push_back(std::move(f1));
                stack.push_back(std::move(f2));
            }
            else if (d12 >= d20) {
                mathcore::Vec3 pm = midP(f.p1, f.p2);
                SparseW wm = midSparseW(f.w1, f.w2);
                SubdivFrame f1, f2;
                f1.p0 = f.p0; f1.p1 = f.p1; f1.p2 = pm;
                f1.w0 = f.w0; f1.w1 = f.w1; f1.w2 = wm; f1.depth = f.depth + 1;
                f2.p0 = f.p0; f2.p1 = pm; f2.p2 = f.p2;
                f2.w0 = f.w0; f2.w1 = wm; f2.w2 = f.w2; f2.depth = f.depth + 1;
                stack.push_back(std::move(f1));
                stack.push_back(std::move(f2));
            }
            else {
                mathcore::Vec3 pm = midP(f.p2, f.p0);
                SparseW wm = midSparseW(f.w2, f.w0);
                SubdivFrame f1, f2;
                f1.p0 = f.p0; f1.p1 = f.p1; f1.p2 = pm;
                f1.w0 = f.w0; f1.w1 = f.w1; f1.w2 = wm; f1.depth = f.depth + 1;
                f2.p0 = pm; f2.p1 = f.p1; f2.p2 = f.p2;
                f2.w0 = wm; f2.w1 = f.w1; f2.w2 = f.w2; f2.depth = f.depth + 1;
                stack.push_back(std::move(f1));
                stack.push_back(std::move(f2));
            }
        }
        return result;
    }

    // ================================================================
    // ★ Schlömer 2011：最远点优化点集（替代简单的 farthest-point）
    //    论文 §3.2 引用的方法
    // ================================================================
    std::vector<unsigned int> farthestPointClustering(
        const std::vector<SparseW>& allW,
        unsigned int numClusters) {

        unsigned int n = (unsigned int)allW.size();
        if (numClusters > n) numClusters = n;
        if (numClusters == 0) return {};

        std::vector<unsigned int> centers;
        centers.reserve(numClusters);

        // 起点：第一个顶点
        centers.push_back(0);

        std::vector<double> minDistSq(n, std::numeric_limits<double>::max());
        for (unsigned int i = 0; i < n; ++i) {
            minDistSq[i] = weightL2DistSq(allW[i], allW[0]);
        }

        for (unsigned int k = 1; k < numClusters; ++k) {
            // 找离已有中心最远的点
            unsigned int best = 0;
            double bestDist = -1.0;
            for (unsigned int i = 0; i < n; ++i) {
                if (minDistSq[i] > bestDist) {
                    bestDist = minDistSq[i];
                    best = i;
                }
            }
            centers.push_back(best);
            for (unsigned int i = 0; i < n; ++i) {
                double d = weightL2DistSq(allW[i], allW[best]);
                if (d < minDistSq[i]) minDistSq[i] = d;
            }
        }

        return centers;
    }

    // ================================================================
    // computeSingleVertexCoR
    // ================================================================
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
        unsigned int numTris) {

        const double OMEGA_SQ = 0.01;
        const double SIM_EPS = 1e-6;
        const double sigma = 0.1;
        unsigned int MAX_BFS = (unsigned int)allW.size() / 8;
        if (MAX_BFS < 2000) MAX_BFS = 2000;
        if (MAX_BFS > 20000) MAX_BFS = 20000;

        std::vector<unsigned int> visited(numTris, 0);
        unsigned int visitMark = 1;
        std::vector<unsigned int> bfsQueue;
        bfsQueue.reserve(MAX_BFS);

        // ============================================================
        // ★ 论文方法：从最近 3 个簇中收集候选种子
        //    不依赖硬阈值，保证总能找到种子
        // ============================================================
        if (vi < vertClusterDists.size() && K > 0) {
            unsigned int nCheck = std::min<unsigned int>(3, K);

            // 收集最近 3 簇的所有顶点，按权重 L2 距离排序
            std::vector<std::pair<double, unsigned int>> candidates;
            for (unsigned int c = 0; c < nCheck; ++c) {
                unsigned int ck = vertClusterDists[vi][c].second;
                for (unsigned int j : clusterMembers[ck]) {
                    double d = weightL2DistSq(wi, allW[j]);
                    candidates.push_back({ d, j });
                }
            }

            // 排序，取最近的 32 个
            std::sort(candidates.begin(), candidates.end());
            unsigned int takeCount = std::min<unsigned int>(32, (unsigned int)candidates.size());

            for (unsigned int k = 0; k < takeCount; ++k) {
                unsigned int j = candidates[k].second;
                for (unsigned int t : vertToTris[j]) {
                    if (visited[t] == visitMark) continue;
                    visited[t] = visitMark;
                    bfsQueue.push_back(t);
                    if (bfsQueue.size() >= MAX_BFS) break;
                }
                if (bfsQueue.size() >= MAX_BFS) break;
            }
        }

        // BFS 扩展
        size_t head = 0;
        while (head < bfsQueue.size() && bfsQueue.size() < MAX_BFS) {
            unsigned int t = bfsQueue[head++];
            for (unsigned int nb : triAdj[t]) {
                if (visited[nb] == visitMark) continue;
                visited[nb] = visitMark;

                bool keep = false;
                for (const SubTri& st : triSubdivs[nb]) {
                    double sim = paper_similarity(wi, st.w, sigma);
                    if (sim >= SIM_EPS) { keep = true; break; }
                }
                if (!keep) continue;

                bfsQueue.push_back(nb);
                if (bfsQueue.size() >= MAX_BFS) break;
            }
        }

        // 积分
        double nx = 0, ny = 0, nz = 0, den = 0;
        for (unsigned int t : bfsQueue) {
            for (const SubTri& st : triSubdivs[t]) {
                if (st.area < 1e-12) continue;
                double sim = paper_similarity(wi, st.w, sigma);
                if (sim < 1e-15) continue;
                double wt = sim * st.area;
                nx += wt * st.center.x;
                ny += wt * st.center.y;
                nz += wt * st.center.z;
                den += wt;
            }
        }

        if (den > 1e-15) {
            return { nx / den, ny / den, nz / den };
        }
        return { points[vi].x, points[vi].y, points[vi].z };
    }

} // namespace optimized_skin_internal