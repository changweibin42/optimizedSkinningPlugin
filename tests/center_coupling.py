# -*- coding: utf-8 -*-
r"""中心耦合：只攻「已关闭限幅、仍剩 109.78°」这一例。

已确认三层互相独立：
  层 1  s=0 形状折角   : fillet=0.4 后 178.00° -> 9.84°（已解决）
  层 2  |s|≈2.2 病态   : 109.78°，|delta|max 70.092（本脚本目标）
  层 3  限幅撕裂       : 714 顶点，本脚本一律关闭限幅

目标函数只加一项（用户方案）：

    E_new = E_original + λ Σ_(i,j) k_ij a_ij² ||Δc_j − Δc_i||²

三处落地与原 Python 求解器一致：system(x) 加 λ·L_c x、rhs 不变、
预条件块对角加 λ·center_degree。

验收按用户给的三条：
  1. |s|≈2.2 前后整段经线的最大有符号转角及位置（尖折是否真减小）
  2. 中线和相邻两圈的内外极点位移（肘窝/肘尖是否保住）
  3. 同一批顶点的 |Δc|、|A·Δc|、相邻中心差（改善来源是否合理）

    mayapy tests/center_coupling.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
FILLET = 0.4          # 关掉层 1，专攻层 2
LAMBDAS = (0.0, 0.01, 0.03, 0.1, 0.3, 1.0, 3.0, 10.0, 30.0)


# ---------------------------------------------------------------- 求解器
def structural_centers_coupled(edges, original, weights, rotations, guide, outer,
                               center_limit=None, lambda_c=0.0, max_iters=300):
    a, b, rest, k = edges
    n = len(original)
    A = ref.center_response(weights, rotations)
    AT = A.transpose(0, 2, 1)
    R = np.einsum("nk,kij->nij", weights, ref.rotation_matrix(rotations)) - A

    def apply(matrix, x):
        return (matrix @ x[:, :, None])[:, :, 0]

    def scatter(values):
        return np.column_stack([
            np.bincount(b, weights=v, minlength=n) -
            np.bincount(a, weights=v, minlength=n) for v in values.T])

    degree = np.bincount(a, weights=k, minlength=n) + \
        np.bincount(b, weights=k, minlength=n)
    ridge = 0.00006 * np.maximum(degree, 1e-12)
    target = apply(0.5 * (R[a] + R[b]), rest)

    prediction = original + guide
    length = np.linalg.norm(rest, axis=1)
    extension = np.maximum(
        np.linalg.norm(prediction[b] - prediction[a], axis=1) - length, 0.0)
    axial = rest[:, 0] / length
    extension = length * np.tanh(extension / length)
    traction = (k * extension / length * axial ** 2 *
                np.minimum(outer[a], outer[b]))[:, None] * rest

    transport_weights = weights ** 3
    transport_weights /= transport_weights.sum(axis=1, keepdims=True)
    frame = np.einsum("nk,kij->nij", transport_weights,
                      ref.rotation_matrix(rotations))
    frame -= ref.center_response(transport_weights, rotations)
    left = apply(frame[a], traction)
    right = apply(frame[b], traction)
    pull = np.column_stack([
        np.bincount(a, weights=l, minlength=n) -
        np.bincount(b, weights=r, minlength=n) for l, r in zip(left.T, right.T)])

    guide_mass = 2.0 * degree[:, None]
    rest_k = k * axial ** 2
    current_length = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    longitudinal = rest_k * length / np.maximum(current_length, 0.15 * length)

    def lap(x):
        return scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n) +
                  np.bincount(b, weights=longitudinal, minlength=n))
    compliance = 8.0 * np.minimum(1.0, 4.0 * np.abs(weights[:, 0] - weights[:, 1])) ** 2 \
        / np.maximum(lap_degree, 1e-12)
    rest_detail = scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(compliance[:, None] * (apply(R, rest_detail) - lap(original)))

    rhs = apply(AT,
                scatter(k[:, None] * (target - (original[b] - original[a]))) +
                guide_mass * guide + 7.0 * pull + curvature_rhs)

    # [NEW] 中心差惩罚：k_ij·a_ij² 加权的图拉普拉斯
    center_edge_w = k * axial ** 2

    def center_lap(x):
        return scatter(center_edge_w[:, None] * (x[b] - x[a]))

    center_degree = (np.bincount(a, weights=center_edge_w, minlength=n) +
                     np.bincount(b, weights=center_edge_w, minlength=n))

    def system(x):
        u = apply(A, x)
        elastic = scatter(k[:, None] * (u[b] - u[a])) + guide_mass * u
        out = apply(AT, elastic + lap(compliance[:, None] * lap(u))) + \
            ridge[:, None] * x
        if lambda_c > 0.0:
            out = out + lambda_c * center_lap(x)
        return out

    diag = (degree + guide_mass[:, 0] + compliance * lap_degree ** 2 +
            np.bincount(a, weights=longitudinal ** 2 * compliance[b], minlength=n) +
            np.bincount(b, weights=longitudinal ** 2 * compliance[a], minlength=n))
    diag = diag + lambda_c * center_degree          # [NEW]
    block = diag[:, None, None] * (AT @ A) + ridge[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(block)

    x = np.zeros_like(original)
    r = rhs.copy()
    d = apply(inverse, r)
    rz = np.sum(r * d)
    tolerance = rz * 1e-12
    used = 0
    for it in range(max_iters):
        if rz <= tolerance:
            break
        hd = system(d)
        denom = np.sum(d * hd)
        if abs(denom) < 1e-30:
            break
        alpha = rz / denom
        x += alpha * d
        r -= alpha * hd
        z = apply(inverse, r)
        next_rz = np.sum(z * r)
        d = z + (next_rz / rz) * d
        rz = next_rz
        used = it + 1

    if center_limit is not None:
        x *= np.minimum(1.0, center_limit /
                        np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12))
    return x, {"iters": used, "A": A}


# ---------------------------------------------------------------- 工具
def signed_turning(pts):
    u = pts[1:-1] - pts[:-2]
    v = pts[2:] - pts[1:-1]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def hinge_section_filleted(station, depth, radius, weights, angle, fillet):
    if fillet <= 0.0 or abs(station) >= fillet or angle <= 1e-8:
        return ref.hinge_section(np.atleast_1d(station), np.atleast_1d(depth),
                                 radius, weights, angle)[0]

    def raw(s):
        return ref.hinge_section(np.array([s]), np.array([depth]), radius,
                                 weights, angle)[0]

    w = fillet
    left, right = np.asarray(raw(-w), float), np.asarray(raw(w), float)
    h = min(1e-4, w * 1e-3)
    t0 = np.asarray(raw(-w + h), float) - np.asarray(raw(-w - h), float)
    t1 = np.asarray(raw(w + h), float) - np.asarray(raw(w - h), float)
    n0, n1 = np.linalg.norm(t0), np.linalg.norm(t1)
    chord = np.linalg.norm(right - left)
    if n0 < 1e-12 or n1 < 1e-12 or chord < 1e-12:
        return raw(station)
    t0 /= n0
    t1 /= n1
    u = smoothstep((station + w) / (2.0 * w))
    return ((2*u**3 - 3*u**2 + 1) * left + (u**3 - 2*u**2 + u) * (chord * t0) +
            (-2*u**3 + 3*u**2) * right + (u**3 - u**2) * (chord * t1))


def guide_with_fillet(bind, original, W, fillet):
    """原版 hinge_displacement 的逐行复刻，只把内侧轮廓替换成带圆角的版本。

    fillet=0 时必须与 ref.hinge_displacement 逐位一致 —— 这点由
    tests/guide_vs_fillet.py 的自检强制。之前那一版重写了整个目标点构造，
    与参考实现在 depth/lateral 的处理上有偏差，| guide| 差 0.1166。
    """
    station, y, z = np.asarray(bind, dtype=float).T
    plane = math.radians(180.0 if BEND < 0 else 0.0)
    phase = 0.0                      # twist = 0
    depth = y * np.cos(phase - plane) - z * np.sin(phase - plane)
    lateral = y * np.sin(phase - plane) + z * np.cos(phase - plane)
    radius = float(np.max(np.hypot(y, z)))
    if radius <= 1e-12:
        return np.zeros_like(original)
    ext = np.full_like(station, radius)
    angle = math.radians(abs(BEND))

    # 与参考实现同样的一次性求值；fillet>0 时逐点走带圆角的版本。
    if fillet <= 0.0:
        inner = ref.hinge_section(station, ext, radius, W, angle)
        outer = ref.hinge_section(station, -ext, radius, W, angle)
    else:
        inner = np.array([hinge_section_filleted(s, +e, radius, W[i:i + 1],
                                                 angle, fillet)
                          for i, (s, e) in enumerate(zip(station, ext))])
        outer = np.array([hinge_section_filleted(s, -e, radius, W[i:i + 1],
                                                 angle, fillet)
                          for i, (s, e) in enumerate(zip(station, ext))])

    neutral = (inner + outer) * 0.5
    gradient = (inner - outer) / (2.0 * radius)
    x, h = (neutral + depth[:, None] * gradient).T
    target = np.column_stack((x,
                              h * np.cos(plane) - lateral * np.sin(plane),
                              h * np.sin(plane) + lateral * np.cos(plane)))
    return target - original


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        def tube(sides, rings, half):
            pts, faces = [], []
            for row in range(rings):
                s = -half + row * (2 * half / (rings - 1))
                for j in range(sides):
                    a = 2 * math.pi * j / sides
                    pts.append((s, math.cos(a), math.sin(a)))
            for row in range(rings - 1):
                for j in range(sides):
                    p = row * sides + j
                    q = row * sides + (j + 1) % sides
                    faces.append((p, q, q + sides, p + sides))
            return pts, faces

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind_list, faces = tube(SIDES, RINGS, HALF)
        bind = np.asarray(bind_list, float)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="pv", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="el", parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type=SKIN, name="sk#")[0]
        for idx, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, idx))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                       *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
        w1 = np.array([ref.skin_weight(p[0]) for p in bind])
        for i in range(len(bind)):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w1[i], w1[i], size=2)
        s2 = om.MSelectionList()
        s2.add(elbow)
        om.MFnTransform(s2.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
        sl = om.MSelectionList()
        sl.add(shape)
        mc.setAttr(skin + ".algorithmType", 0)
        mc.refresh(force=True)
        for _ in range(2):
            mc.dgeval(shape)
            mc.refresh(force=True)
        original = np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
            sl.getDagPath(0)).getPoints(om.MSpace.kObject)])
        mc.delete(holder)

        W = np.column_stack([1.0 - w1, w1])
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        guide = guide_with_fillet(bind, original, W, FILLET)
        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        ring_idx = [r * SIDES for r in range(RINGS)]
        mid = RINGS // 2
        probe = [mid - 1, mid, mid + 1]      # 中线及相邻两圈

        def finalize(delta, A):
            return original + np.einsum("nij,nj->ni", A, delta)

        def extremes(final):
            """中线和相邻两圈的内外极点位移。"""
            vals = []
            for r in probe:
                seg = np.arange(r * SIDES, (r + 1) * SIDES)
                yb = bind[seg, 1]
                i_in = seg[int(np.argmax(yb))]
                i_out = seg[int(np.argmin(yb))]
                vals.append(np.linalg.norm(final[i_in] - original[i_in]))
                vals.append(np.linalg.norm(final[i_out] - original[i_out]))
            return vals

        print("配置：fillet=%.1f（关掉层 1），关闭限幅（关掉层 3）" % FILLET)
        print("guide 的 |guide|max = %.4f" % np.linalg.norm(guide, axis=1).max())

        # ---- 基线 λ=0 ----
        d0, info0 = structural_centers_coupled(
            edges, original, W, quats, guide, outer, center_limit=None)
        f0 = finalize(d0, info0["A"])
        t0 = signed_turning(f0[ring_idx])
        pos = stations[1:-1]
        print("\n基线 λ=0：|delta|max %.3f  iters %d"
              % (np.linalg.norm(d0, axis=1).max(), info0["iters"]))
        print("  最大 |转角| %.2f° @ s=%.3f" % (abs(t0[int(np.argmax(np.abs(t0)))]),
                                                pos[int(np.argmax(np.abs(t0)))]))
        base_turn = abs(t0[int(np.argmax(np.abs(t0)))])
        base_ext = extremes(f0)
        print("  内外极点位移 %s" % " ".join("%.3f" % v for v in base_ext))

        # ---- λ 扫描 ----
        print("\n%6s %10s %8s %10s %10s %10s %10s"
              % ("λ", "最大|转角|", "@s", "|delta|max", "相邻Δc", "|A·Δc|max", "极点最大变化"))
        rows = []
        for lam in LAMBDAS:
            if lam == 0.0:
                d, info, f = d0, info0, f0
            else:
                d, info = structural_centers_coupled(
                    edges, original, W, quats, guide, outer,
                    center_limit=None, lambda_c=lam)
                f = finalize(d, info["A"])
            turns = signed_turning(f[ring_idx])
            imax = int(np.argmax(np.abs(turns)))
            dc_gap = np.linalg.norm(np.diff(d[ring_idx], axis=0), axis=1).mean()
            ad = np.linalg.norm(np.einsum("nij,nj->ni", info["A"], d), axis=1).max()
            ext = extremes(f)
            ext_change = max(abs(x - y) for x, y in zip(ext, base_ext))
            rows.append((lam, abs(turns[imax]), pos[imax],
                         np.linalg.norm(d, axis=1).max(), dc_gap, ad,
                         ext_change, info["iters"], turns, f, d, ext))
            print("%6.2f %10.2f %8.3f %10.3f %10.4f %10.3f %10.3f"
                  % (lam, abs(turns[imax]), pos[imax],
                     np.linalg.norm(d, axis=1).max(), dc_gap, ad, ext_change))

        # ---- 逐峰对比：看尖折是减小还是移走 ----
        print("\n整段经线的峰值迁移（|s|<=3.0 内）：")
        for row in rows:
            lam, _, _, _, _, _, _, _, turns, _, _, _ = row
            inside = [i for i in range(len(pos)) if abs(pos[i]) <= 3.0]
            order = sorted(inside, key=lambda i: -abs(turns[i]))[:4]
            print("  λ=%-5.2f  %s" % (lam, "  ".join("%+.1f°@%+.2f"
                                                    % (turns[i], pos[i]) for i in order)))

        best = min(rows, key=lambda r: r[1])
        print("\n结论：")
        print("  基线 %.2f° -> 最优 λ=%.2f 时 %.2f°（降 %.1f%%）"
              % (base_turn, best[0], best[1],
                 100.0 * (base_turn - best[1]) / base_turn))
        print("  该配置 |delta|max %.3f、相邻Δc 均值 %.4f、极点最大变化 %.3f"
              % (best[3], best[4], best[6]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
