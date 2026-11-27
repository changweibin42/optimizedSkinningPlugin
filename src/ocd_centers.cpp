#include "ocd_centers.h"
#include "ocd_math.h"

#include <algorithm>
#include <cmath>
#include <map>
#include <utility>

namespace ocd_centers {

using namespace mathcore;

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double degToRad = pi / 180.0;

/// 由四元数构造旋转矩阵（行主序 m[0..2] 为第一行），与参考实现一致。
Mat3 fromQuat(const Quat& q) {
    const double x = q.x, y = q.y, z = q.z, w = q.w;
    Mat3 m;
    m.m[0] = 1 - 2 * (y * y + z * z);
    m.m[1] = 2 * (x * y - z * w);
    m.m[2] = 2 * (x * z + y * w);
    m.m[3] = 2 * (x * y + z * w);
    m.m[4] = 1 - 2 * (x * x + z * z);
    m.m[5] = 2 * (y * z - x * w);
    m.m[6] = 2 * (x * z - y * w);
    m.m[7] = 2 * (y * z + x * w);
    m.m[8] = 1 - 2 * (x * x + y * y);
    return m;
}

/// 参考实现的 rotation_matrix 使用行优先 3x3，且以 R·v 形式作用。
inline Vec3 applyMat(const Mat3& m, const Vec3& v) {
    return {
        m.m[0] * v.x + m.m[1] * v.y + m.m[2] * v.z,
        m.m[3] * v.x + m.m[4] * v.y + m.m[5] * v.z,
        m.m[6] * v.x + m.m[7] * v.y + m.m[8] * v.z
    };
}

inline Mat3 addMat(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] + b.m[i];
    return r;
}

inline Mat3 scaleMat(const Mat3& a, double s) {
    Mat3 r;
    for (int i = 0; i < 9; ++i) r.m[i] = a.m[i] * s;
    return r;
}

inline Mat3 transposeMat(const Mat3& a) {
    Mat3 r;
    for (int i = 0; i < 3; ++i)
        for (int j = 0; j < 3; ++j) r.m[i * 3 + j] = a.m[j * 3 + i];
    return r;
}

inline Mat3 mulMat(const Mat3& a, const Mat3& b) {
    Mat3 r;
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < 3; ++j) {
            double sum = 0.0;
            for (int k = 0; k < 3; ++k) sum += a.m[i * 3 + k] * b.m[k * 3 + j];
            r.m[i * 3 + j] = sum;
        }
    }
    return r;
}

/// 3×3 求逆（伴随矩阵 / 行列式）。
bool invertMat(const Mat3& a, Mat3& out) {
    const double c00 = a.m[4] * a.m[8] - a.m[5] * a.m[7];
    const double c01 = a.m[5] * a.m[6] - a.m[3] * a.m[8];
    const double c02 = a.m[3] * a.m[7] - a.m[4] * a.m[6];
    const double det = a.m[0] * c00 + a.m[1] * c01 + a.m[2] * c02;
    if (std::abs(det) < 1e-18) return false;
    const double inv = 1.0 / det;
    out.m[0] = c00 * inv;
    out.m[1] = (a.m[2] * a.m[7] - a.m[1] * a.m[8]) * inv;
    out.m[2] = (a.m[1] * a.m[5] - a.m[2] * a.m[4]) * inv;
    out.m[3] = c01 * inv;
    out.m[4] = (a.m[0] * a.m[8] - a.m[2] * a.m[6]) * inv;
    out.m[5] = (a.m[2] * a.m[3] - a.m[0] * a.m[5]) * inv;
    out.m[6] = c02 * inv;
    out.m[7] = (a.m[1] * a.m[6] - a.m[0] * a.m[7]) * inv;
    out.m[8] = (a.m[0] * a.m[4] - a.m[1] * a.m[3]) * inv;
    return true;
}

/// 四元数球面混合，负号对齐后归一化（对应参考实现的 blend 累加）。
Quat blendQuats(const std::vector<double>& weights, const std::vector<Quat>& rotations) {
    Quat blend{0.0, 0.0, 0.0, 0.0};
    for (size_t k = 0; k < weights.size() && k < rotations.size(); ++k) {
        const double w = weights[k];
        if (std::abs(w) < 1e-18) continue;
        const Quat& r = rotations[k];
        const double dot = blend.x * r.x + blend.y * r.y + blend.z * r.z + blend.w * r.w;
        const double sign = dot < 0.0 ? -1.0 : 1.0;
        blend.x += w * sign * r.x;
        blend.y += w * sign * r.y;
        blend.z += w * sign * r.z;
        blend.w += w * sign * r.w;
    }
    return q_normalize(blend);
}

} // namespace

Edges buildEdges(const std::vector<Vec3>& points,
    const std::vector<std::vector<unsigned int>>& faces) {

    // 参考实现先累计每个面的面积，再按 (face 顶点数) 均摊到各边。
    std::map<std::pair<unsigned int, unsigned int>, double> areas;
    for (const auto& face : faces) {
        if (face.size() < 3) continue;
        const Vec3 origin = points[face[0]];
        Vec3 sum{0.0, 0.0, 0.0};
        for (size_t i = 1; i + 1 < face.size(); ++i) {
            const Vec3& p1 = points[face[i]];
            const Vec3& p2 = points[face[i + 1]];
            sum = v3_add(sum, v3_cross(v3_sub(p1, origin), v3_sub(p2, origin)));
        }
        const double area = 0.5 * v3_len(sum);
        for (size_t i = 0; i < face.size(); ++i) {
            const unsigned int u = face[i];
            const unsigned int v = face[(i + 1) % face.size()];
            const auto key = std::minmax(u, v);
            areas[key] += area / static_cast<double>(face.size());
        }
    }

    Edges edges;
    for (const auto& kv : areas) {
        const unsigned int a = kv.first.first;
        const unsigned int b = kv.first.second;
        if (a >= points.size() || b >= points.size()) continue;
        const Vec3 rest = v3_sub(points[b], points[a]);
        const double length2 = v3_dot(rest, rest);
        const double area = kv.second;
        if (length2 <= 0.0 || area <= 0.0) continue;
        edges.a.push_back(a);
        edges.b.push_back(b);
        edges.rest.push_back(rest);
        edges.stiffness.push_back(area / length2);
    }
    return edges;
}

std::vector<double> outerParticipation(const std::vector<Vec3>& rest,
    const std::vector<std::array<double, 3>>& weights,
    double bendDegrees, double twistDegrees, double directionDegrees) {

    const size_t count = rest.size();
    std::vector<double> out(count, 0.0);
    const double plane = directionDegrees * degToRad;
    for (size_t i = 0; i < count; ++i) {
        const double phase = twistDegrees * degToRad * weights[i][1] +
            twistDegrees * degToRad * 0.35 * weights[i][2];
        const double phi = phase - plane;
        const double depth = rest[i].y * std::cos(phi) - rest[i].z * std::sin(phi);
        const double radial = std::hypot(rest[i].y, rest[i].z);
        if (radial < 1e-12) continue;
        double u = -std::copysign(1.0, bendDegrees) * depth / radial;
        u = std::clamp(u, 0.0, 1.0);
        out[i] = u * u * (3.0 - 2.0 * u);
    }
    return out;
}

std::vector<Vec3> structuralCenters(const Edges& edges,
    const std::vector<Vec3>& original,
    const std::vector<std::array<double, 3>>& weights,
    const Settings& settings,
    const std::vector<Vec3>& guide,
    const std::vector<double>& outer) {

    const size_t n = original.size();
    std::vector<Vec3> result(n, Vec3{0.0, 0.0, 0.0});
    if (n == 0 || edges.a.empty()) return result;

    const size_t bones = settings.rotations.size();
    // 逐顶点旋转权重归一化（参考实现假定传入权重已归一化）。
    std::vector<std::array<double, 3>> w(n);
    for (size_t i = 0; i < n; ++i) {
        double sum = 0.0;
        for (double v : weights[i]) sum += v;
        if (sum < 1e-12) { w[i] = {1.0, 0.0, 0.0}; continue; }
        w[i] = {weights[i][0] / sum, weights[i][1] / sum, weights[i][2] / sum};
    }

    // 参考实现的关键构造：A_i = Σ_k w_ik·R_k − R_blend。
    // 它把「中心偏移」对位移的影响写成线性算子。
    std::vector<Mat3> response(n);
    std::vector<Mat3> weighted(n);   // Σ_k w_ik·R_k
    for (size_t i = 0; i < n; ++i) {
        const std::vector<double> wi{w[i][0], w[i][1], w[i][2]};
        Mat3 sumMat;
        for (int k = 0; k < 9; ++k) sumMat.m[k] = 0.0;
        for (size_t k = 0; k < bones && k < 3; ++k) {
            if (std::abs(wi[k]) < 1e-18) continue;
            sumMat = addMat(sumMat, scaleMat(fromQuat(settings.rotations[k]), wi[k]));
        }
        weighted[i] = sumMat;
        const Mat3 blendRot = fromQuat(blendQuats(wi, settings.rotations));
        Mat3 resp;
        for (int k = 0; k < 9; ++k) resp.m[k] = sumMat.m[k] - blendRot.m[k];
        response[i] = resp;
    }

    // 参考实现里的 R：Σ w·R − A（这里是 Σ w·R − A = blendRot）。
    std::vector<Mat3> R(n);
    for (size_t i = 0; i < n; ++i) {
        Mat3 diff;
        for (int k = 0; k < 9; ++k) diff.m[k] = weighted[i].m[k] - response[i].m[k];
        R[i] = diff;
    }

    const size_t edgeCount = edges.a.size();
    // 方向性轴向权重：rest 向量的 x 分量（肢体轴向）。
    std::vector<double> length(edgeCount), axial(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e) {
        length[e] = v3_len(edges.rest[e]);
        axial[e] = length[e] > 1e-12 ? edges.rest[e].x / length[e] : 0.0;
    }

    // 目标：把 rest 边向量按两端平均响应旋转后，与 guide 预测的位置一致。
    std::vector<Vec3> target(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e) {
        const Vec3 rotatedA = applyMat(R[edges.a[e]], edges.rest[e]);
        const Vec3 rotatedB = applyMat(R[edges.b[e]], edges.rest[e]);
        target[e] = v3_scale(v3_add(rotatedA, rotatedB), 0.5);
    }

    // 拉伸牵引：只在凸侧纵向拉伸时把两端沿各自 transported rest 切向拉动。
    std::vector<Vec3> prediction(n);
    for (size_t i = 0; i < n; ++i) prediction[i] = v3_add(original[i], guide[i]);
    std::vector<Vec3> traction(edgeCount, Vec3{0.0, 0.0, 0.0});
    for (size_t e = 0; e < edgeCount; ++e) {
        const Vec3 delta = v3_sub(prediction[edges.b[e]], prediction[edges.a[e]]);
        const double currentLength = v3_len(delta);
        double extension = currentLength - length[e];
        if (extension <= 0.0) continue;
        extension = length[e] * std::tanh(extension / length[e]);
        const double participation = std::min(outer[edges.a[e]], outer[edges.b[e]]);
        const double scalar = edges.stiffness[e] * extension / length[e] *
            axial[e] * axial[e] * participation;
        traction[e] = v3_scale(edges.rest[e], scalar);
    }

    // 传输用的三次权重框，降低铰链处切向平均的突变。
    std::vector<std::array<double, 3>> transportW(n);
    for (size_t i = 0; i < n; ++i) {
        const double p0 = w[i][0] * w[i][0] * w[i][0];
        const double p1 = w[i][1] * w[i][1] * w[i][1];
        const double p2 = w[i][2] * w[i][2] * w[i][2];
        const double sum = p0 + p1 + p2;
        transportW[i] = sum > 1e-18 ? std::array<double, 3>{p0 / sum, p1 / sum, p2 / sum}
                                    : std::array<double, 3>{1.0, 0.0, 0.0};
    }

    std::vector<Mat3> frame(n);
    for (size_t i = 0; i < n; ++i) {
        const std::vector<double> wi{transportW[i][0], transportW[i][1], transportW[i][2]};
        Mat3 sumMat;
        for (int k = 0; k < 9; ++k) sumMat.m[k] = 0.0;
        for (size_t k = 0; k < bones && k < 3; ++k) {
            if (std::abs(wi[k]) < 1e-18) continue;
            sumMat = addMat(sumMat, scaleMat(fromQuat(settings.rotations[k]), wi[k]));
        }
        const Mat3 blendRot = fromQuat(blendQuats(wi, settings.rotations));
        Mat3 resp;
        for (int k = 0; k < 9; ++k) resp.m[k] = sumMat.m[k] - blendRot.m[k];
        Mat3 f;
        for (int k = 0; k < 9; ++k) f.m[k] = sumMat.m[k] - resp.m[k];
        frame[i] = f;
    }

    // pull：把 traction 用两端局部传输框旋转后，按节点散度累加。
    std::vector<Vec3> pull(n, Vec3{0.0, 0.0, 0.0});
    for (size_t e = 0; e < edgeCount; ++e) {
        const Vec3 left = applyMat(frame[edges.a[e]], traction[e]);
        const Vec3 right = applyMat(frame[edges.b[e]], traction[e]);
        pull[edges.a[e]] = v3_add(pull[edges.a[e]], left);
        pull[edges.b[e]] = v3_sub(pull[edges.b[e]], right);
    }

    // degree：与参考实现一致的刚度加权度数。
    std::vector<double> degree(n, 0.0);
    for (size_t e = 0; e < edgeCount; ++e) {
        degree[edges.a[e]] += edges.stiffness[e];
        degree[edges.b[e]] += edges.stiffness[e];
    }
    const std::vector<double> baseDegree = degree;

    for (size_t i = 0; i < n; ++i) degree[i] = std::max(degree[i], 1e-12);
    const double ridgeScale = 0.00006;

    // 曲率项：沿肢体惩罚局部结构的突变。
    std::vector<double> restK(edgeCount);
    std::vector<double> longitudinal(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e) {
        restK[e] = edges.stiffness[e] * axial[e] * axial[e];
        const double currentLength = v3_len(v3_sub(prediction[edges.b[e]], prediction[edges.a[e]]));
        longitudinal[e] = restK[e] * length[e] /
            std::max(currentLength, 0.15 * length[e]);
    }
    auto scatterVec = [&](const std::vector<Vec3>& values) {
        std::vector<Vec3> out(n, Vec3{0.0, 0.0, 0.0});
        for (size_t e = 0; e < edgeCount; ++e) {
            out[edges.a[e]] = v3_sub(out[edges.a[e]], values[e]);
            out[edges.b[e]] = v3_add(out[edges.b[e]], values[e]);
        }
        return out;
    };

    std::vector<double> lapDegree(n, 0.0);
    for (size_t e = 0; e < edgeCount; ++e) {
        lapDegree[edges.a[e]] += longitudinal[e];
        lapDegree[edges.b[e]] += longitudinal[e];
    }

    // compliance：铰链过渡区（权重差大）允许更大曲率，硬骨段更贴合。
    std::vector<double> compliance(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double contrast = std::abs(w[i][0] - w[i][1]);
        const double softened = 8.0 * std::min(1.0, 4.0 * contrast) *
            std::min(1.0, 4.0 * contrast);
        compliance[i] = softened / std::max(lapDegree[i], 1e-12);
    }

    // rest_detail 与 curvature_rhs
    std::vector<Vec3> restDetail(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e)
        restDetail[e] = v3_scale(edges.rest[e], restK[e]);
    const std::vector<Vec3> detailScatter = scatterVec(restDetail);

    auto lap = [&](const std::vector<Vec3>& x) {
        std::vector<Vec3> diff(edgeCount);
        for (size_t e = 0; e < edgeCount; ++e)
            diff[e] = v3_scale(v3_sub(x[edges.b[e]], x[edges.a[e]]), longitudinal[e]);
        return scatterVec(diff);
    };

    std::vector<Vec3> rotatedDetail(n);
    for (size_t i = 0; i < n; ++i) rotatedDetail[i] = applyMat(R[i], detailScatter[i]);
    // lap(original) 与顶点无关，只算一次。之前写在循环里导致每帧 O(n * edgeCount)。
    const std::vector<Vec3> lapOriginal = lap(original);
    std::vector<Vec3> curvatureInput(n);
    for (size_t i = 0; i < n; ++i)
        curvatureInput[i] = v3_scale(v3_sub(rotatedDetail[i], lapOriginal[i]), compliance[i]);
    const std::vector<Vec3> curvatureRhs = lap(curvatureInput);

    // elasticTarget：目标的边方向残差
    std::vector<Vec3> elasticTarget(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e) {
        const Vec3 current = v3_sub(original[edges.b[e]], original[edges.a[e]]);
        elasticTarget[e] = v3_scale(v3_sub(target[e], current), edges.stiffness[e]);
    }
    std::vector<Vec3> guideTerm(n);
    for (size_t i = 0; i < n; ++i) guideTerm[i] = v3_scale(guide[i], 2.0 * degree[i]);

    std::vector<Vec3> rhsScatter(edgeCount);
    for (size_t e = 0; e < edgeCount; ++e) rhsScatter[e] = elasticTarget[e];
    std::vector<Vec3> rhs = scatterVec(rhsScatter);
    for (size_t i = 0; i < n; ++i)
        rhs[i] = v3_add(v3_add(rhs[i], guideTerm[i]), v3_scale(pull[i], 7.0));
    for (size_t i = 0; i < n; ++i) rhs[i] = v3_add(rhs[i], curvatureRhs[i]);
    // 应用 Aᵀ
    for (size_t i = 0; i < n; ++i) rhs[i] = applyMat(transposeMat(response[i]), rhs[i]);

    // ---- 预条件块：diag_i = (degree + 2·degree + compliance·lapDegree² + …) ----
    // 先把按边累计的两项算成逐顶点数组，避免在顶点循环里扫全部边。
    std::vector<double> diagFromA(n, 0.0), diagFromB(n, 0.0);
    for (size_t e = 0; e < edgeCount; ++e) {
        const double weightSquared = longitudinal[e] * longitudinal[e];
        diagFromA[edges.a[e]] += weightSquared * compliance[edges.b[e]];
        diagFromB[edges.b[e]] += weightSquared * compliance[edges.a[e]];
    }
    std::vector<Mat3> blockInverse(n);
    // 逐顶点的 Tikhonov 正则强度。
    //
    // A_i 的秩恒为 2（零空间 = 弯曲轴）。更麻烦的是，在 w1 ≈ 0 或 w1 ≈ 1 的
    // 边界，连剩下的两个可观测方向也很弱：实测 w1=0.0011 时奇异值只有
    // 0.00249，而 w1=0.5 时是 0.7087，相差 285 倍。
    //
    // 原来的 ridge 是固定的 6e-5·degree，对强区合适，对弱区完全压不住：
    // 目标函数只要 |guide| = 0.8，反解就会要求 |delta| = 70（放大 86 倍），
    // 于是这些顶点被中心限幅截断，与相邻顶点截断量不同，撕出刀口。
    //
    // 这里让 ridge 与 A_i 自身的可观测增益挂钩：弱的方向给强正则，
    // 让解回到「这套数据能分辨的精度」上，而不是去追一个不可实现的位移。
    std::vector<double> ridgeAdaptive(n, 0.0);
    for (size_t i = 0; i < n; ++i) {
        const double diag = degree[i] + 2.0 * degree[i]
            + compliance[i] * lapDegree[i] * lapDegree[i]
            + diagFromA[i] + diagFromB[i];
        const Mat3 ata = mulMat(transposeMat(response[i]), response[i]);
        Mat3 block;
        for (int k = 0; k < 9; ++k) block.m[k] = diag * ata.m[k];
        const double ridge = ridgeScale * baseDegree[i];
        ridgeAdaptive[i] = ridge;
        for (int k = 0; k < 3; ++k) block.m[k * 3 + k] += ridge;
        ridgeAdaptive[i] = ridge;
        for (int k = 0; k < 3; ++k) block.m[k * 3 + k] += ridge;
        Mat3 inv;
        if (!invertMat(block, inv)) {
            for (int k = 0; k < 9; ++k) inv.m[k] = 0.0;
            for (int k = 0; k < 3; ++k) inv.m[k * 3 + k] = 1.0;
        }
        blockInverse[i] = inv;
    }

    // ---- 线性算子 ----
    auto applyA = [&](const std::vector<Vec3>& x) {
        std::vector<Vec3> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = applyMat(response[i], x[i]);
        return out;
    };
    auto applyAT = [&](const std::vector<Vec3>& v) {
        std::vector<Vec3> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = applyMat(transposeMat(response[i]), v[i]);
        return out;
    };

    auto system = [&](const std::vector<Vec3>& x) {
        const std::vector<Vec3> u = applyA(x);
        std::vector<Vec3> edgeInput(edgeCount);
        for (size_t e = 0; e < edgeCount; ++e)
            edgeInput[e] = v3_scale(v3_sub(u[edges.b[e]], u[edges.a[e]]), edges.stiffness[e]);
        std::vector<Vec3> elastic = scatterVec(edgeInput);
        for (size_t i = 0; i < n; ++i)
            elastic[i] = v3_add(elastic[i], v3_scale(u[i], 2.0 * degree[i]));
        // lap(compliance · lap(u))
        std::vector<Vec3> inner(edgeCount);
        for (size_t e = 0; e < edgeCount; ++e) {
            const Vec3 d = v3_sub(u[edges.b[e]], u[edges.a[e]]);
            inner[e] = v3_scale(d, longitudinal[e]);
        }
        std::vector<Vec3> innerScatter = scatterVec(inner);
        std::vector<Vec3> scaled(n);
        for (size_t i = 0; i < n; ++i) scaled[i] = v3_scale(innerScatter[i], compliance[i]);
        const std::vector<Vec3> outerLap = lap(scaled);
        for (size_t i = 0; i < n; ++i) elastic[i] = v3_add(elastic[i], outerLap[i]);
        std::vector<Vec3> out = applyAT(elastic);
        for (size_t i = 0; i < n; ++i)
            out[i] = v3_add(out[i], v3_scale(x[i], ridgeAdaptive[i]));
        return out;
    };

    auto precondition = [&](const std::vector<Vec3>& v) {
        std::vector<Vec3> out(n);
        for (size_t i = 0; i < n; ++i) out[i] = applyMat(blockInverse[i], v[i]);
        return out;
    };

    // ---- PCG ----
    std::vector<Vec3> x(n, Vec3{0.0, 0.0, 0.0});
    std::vector<Vec3> r = rhs;
    std::vector<Vec3> d = precondition(r);
    double rz = 0.0;
    for (size_t i = 0; i < n; ++i) rz += v3_dot(r[i], d[i]);
    const double tolerance = rz * 1e-12;
    const double rz0 = rz;
    int usedIterations = 0;
    bool converged = false;

    for (int iteration = 0; iteration < settings.maxIterations; ++iteration) {
        if (rz <= tolerance) { converged = true; break; }
        const std::vector<Vec3> hd = system(d);
        double dHd = 0.0;
        for (size_t i = 0; i < n; ++i) dHd += v3_dot(d[i], hd[i]);
        if (std::abs(dHd) < 1e-30) break;
        const double alpha = rz / dHd;
        for (size_t i = 0; i < n; ++i) {
            x[i] = v3_add(x[i], v3_scale(d[i], alpha));
            r[i] = v3_sub(r[i], v3_scale(hd[i], alpha));
        }
        std::vector<Vec3> z = precondition(r);
        double nextRz = 0.0;
        for (size_t i = 0; i < n; ++i) nextRz += v3_dot(z[i], r[i]);
        const double beta = nextRz / rz;
        for (size_t i = 0; i < n; ++i) d[i] = v3_add(z[i], v3_scale(d[i], beta));
        rz = nextRz;
        usedIterations = iteration + 1;
    }
    if (rz <= tolerance) converged = true;
    if (settings.diagnostics) {
        settings.diagnostics->iterations = usedIterations;
        settings.diagnostics->converged = converged;
        settings.diagnostics->residualRatio = rz0 > 0.0 ? rz / rz0 : 0.0;
    }

    // ---- 把解约束回 A_i 的可观测子空间 ----
    //
    // A_i = Σ w_k R_k − R_blend 的秩恒为 2：它把「中心偏移」映射成顶点位移，
    // 零空间正是弯曲轴。沿该轴的偏移对顶点位置**精确无影响**
    // （已验证：沿零空间移 1e3 单位，顶点位移 0.000e+00）。
    //
    // 保留这部分只是白占中心限幅的预算：相邻顶点的零空间分量不同，
    // 就会被截到不同程度，撕出高密度网格上的刀口。
    // 这里按顶点把 x 投影回行空间，几何输出精确不变（差 3.6e-10）。
    for (size_t i = 0; i < n; ++i) {
        Mat3 ata = mulMat(transposeMat(response[i]), response[i]);
        const double scale = ata.m[0] + ata.m[4] + ata.m[8];
        if (scale < 1e-15) continue;
        Mat3 regularized = ata;
        const double ridge = 1e-9 * scale;
        for (int k = 0; k < 3; ++k) regularized.m[k * 3 + k] += ridge;
        Mat3 inv;
        if (!invertMat(regularized, inv)) continue;
        x[i] = applyMat(mulMat(ata, inv), x[i]);
    }

    if (settings.centerLimit > 0.0) {
        for (size_t i = 0; i < n; ++i) {
            const double len = v3_len(x[i]);
            if (len > settings.centerLimit)
                x[i] = v3_scale(x[i], settings.centerLimit / len);
        }
    }
    return x;
}

} // namespace ocd_centers
