# -*- coding: utf-8 -*-
r"""局部消融：按 A_i 的可观测增益给 compliance 做平滑衰减（保留非零下限）。

固定原版引导 fillet=0、λ=0、无限幅。

衰减系数（对 compliance 整体乘一个连续系数）：
    g_i   = σ_max(A_i)                    每顶点的中心响应增益
    gn_i  = g_i / g_ref                   归一化增益（g_ref = 0.70868，w1=0.5 处）
    ken_i = smoothstep(lo, hi, gn_i)      0..1，增益低的地方小
    scale_i = floor + (1 - floor) * ken_i 保留 floor 作为非零下限

系数同时作用于右端项、线性算子、预条件器里的 compliance 项。

试验网格：
  全关（floor=0）作为已知上界，原版（scale=1）作为下界，
  中间试几档 (floor, lo, hi)。

    mayapy tests/compliance_taper.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
G_REF = 0.70868           # A 在 w1=0.5 处的奇异值

KEY_EDGES = [("入边", 4880, 4960), ("出边", 4960, 5040),
             ("邻边1", 4720, 4800), ("邻边2", 4800, 4880)]
KEY_TRIPLE = (4880, 4960, 5040)


def smoothstep(e0, e1, x):
    if e1 <= e0:
        return np.where(x >= e1, 1.0, 0.0)
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def structural_centers_taper(edges, original, weights, rotations, guide, outer,
                             lambda_c=0.0, taper=None, max_iters=300):
    """structural_centers 的 taper 版。taper 是逐顶点的 compliance 系数。"""
    a, b, rest, k = edges
    n = len(original)
    A = ref.center_response(weights, rotations)
    AT = A.transpose(0, 2, 1)
    R = np.einsum("nk,kij->nij", weights, ref.rotation_matrix(rotations)) - A

    def apply(matrix, x):
        return (matrix @ x[:, :, None])[:, :, 0]

    def scatter(values):
        return np.stack([
            np.bincount(b, weights=v, minlength=n) -
            np.bincount(a, weights=v, minlength=n) for v in values.T], axis=1)

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

    tw = weights ** 3
    tw /= tw.sum(axis=1, keepdims=True)
    frame = np.einsum("nk,kij->nij", tw, ref.rotation_matrix(rotations))
    frame -= ref.center_response(tw, rotations)
    pull = np.stack([
        np.bincount(a, weights=l, minlength=n) -
        np.bincount(b, weights=r, minlength=n)
        for l, r in zip(*[apply(frame[e], traction).T for e in (a, b)])], axis=1)

    guide_mass = 2.0 * degree[:, None]
    rest_k = k * axial ** 2
    current_length = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    longitudinal = rest_k * length / np.maximum(current_length, 0.15 * length)

    def lap(x):
        return scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n) +
                  np.bincount(b, weights=longitudinal, minlength=n))

    base_com = 8.0 * np.minimum(1.0, 4.0 * np.abs(weights[:, 0] - weights[:, 1])) ** 2 \
        / np.maximum(lap_degree, 1e-12)
    com = base_com * (taper if taper is not None else 1.0)

    rest_detail = scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(com[:, None] * (apply(R, rest_detail) - lap(original)))

    rhs = apply(AT,
                scatter(k[:, None] * (target - (original[b] - original[a]))) +
                guide_mass * guide + 7.0 * pull + curvature_rhs)

    def system(x):
        u = apply(A, x)
        elastic = scatter(k[:, None] * (u[b] - u[a])) + guide_mass * u
        out = apply(AT, elastic + lap(com[:, None] * lap(u))) + ridge[:, None] * x
        if lambda_c > 0.0:
            out = out + lambda_c * scatter(k[:, None] * axial[:, None] ** 2 *
                                           (x[b] - x[a]))
        return out

    diag = (degree + guide_mass[:, 0] + com * lap_degree ** 2 +
            np.bincount(a, weights=longitudinal ** 2 * com[b], minlength=n) +
            np.bincount(b, weights=longitudinal ** 2 * com[a], minlength=n))
    block = diag[:, None, None] * (AT @ A) + ridge[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(block)

    x = np.zeros_like(original)
    r = rhs.copy()
    d = apply(inverse, r)
    rz = np.sum(r * d)
    tol = rz * 1e-12
    for _ in range(max_iters):
        if rz <= tol:
            break
        hd = system(d)
        denom = np.sum(d * hd)
        if abs(denom) < 1e-30:
            break
        alpha = rz / denom
        x += alpha * d
        r -= alpha * hd
        z = apply(inverse, r)
        nz = np.sum(z * r)
        d = z + (nz / rz) * d
        rz = nz
    return x, {"A": A}


def project_ratio(final, guide, a, b):
    bg = np.array([guide[b][0] - guide[a][0], guide[b][1] - guide[a][1]])
    bf = np.array([final[b][0] - final[a][0], final[b][1] - final[a][1]])
    den = float(bg @ bg)
    return (float(bf @ bg) / den if den > 1e-18 else float("nan"),
            float(np.linalg.norm(bf)))


def turning_at(final, i0, i1, i2):
    u = np.array([final[i1][0] - final[i0][0], final[i1][1] - final[i0][1]])
    v = np.array([final[i2][0] - final[i1][0], final[i2][1] - final[i1][1]])
    return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0],
                                   u[0] * v[0] + u[1] * v[1]))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind_list, faces, bind, original, w1 = gvf.build_scene(mc, om)
        W = np.column_stack([1.0 - w1, w1])
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        g_plain = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
        T_plain = original + g_plain

        A = ref.center_response(W, quats)
        gain = np.array([np.linalg.svd(np.asarray(A[i], float),
                                       compute_uv=False)[0]
                         for i in range(len(bind))])
        gn = gain / G_REF
        print("增益分布：min %.4f  p5 %.4f  p25 %.4f  median %.4f  max %.4f"
              % (gn.min(), np.percentile(gn, 5), np.percentile(gn, 25),
                 np.median(gn), gn.max()))

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])
        all_s = stations[1:]
        reg = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]

        def evaluate(tag, taper):
            d, info = structural_centers_taper(
                edges, original, W, quats, g_plain, outer,
                lambda_c=0.0, taper=taper)
            final = original + np.einsum("nij,nj->ni", info["A"], d)
            pr = {t: project_ratio(final, T_plain, a, b) for t, a, b in KEY_EDGES}
            tv = turning_at(final, *KEY_TRIPLE)
            turn = gvf.signed_turning_2d(final[meridian])
            imax = max(reg, key=lambda i: abs(turn[i]))
            mid_id = int(meridian[imax + 1])
            el = np.linalg.norm(np.diff(final[meridian], axis=0), axis=1)
            ext = []
            mid = RINGS // 2
            for r in (mid - 1, mid, mid + 1):
                seg = np.arange(r * SIDES, (r + 1) * SIDES)
                yb = bind[seg, 1]
                for pick in (int(np.argmax(yb)), int(np.argmin(yb))):
                    ii = seg[pick]
                    ext.append(float(np.linalg.norm(final[ii] - original[ii])))
            return dict(tag=tag, pr=pr, tv=tv, mx=abs(turn[imax]), mid=mid_id,
                        at=all_s[imax], el=float(min(el[imax], el[imax + 1])),
                        ext=ext, dmax=float(np.linalg.norm(d, axis=1).max()))

        trials = [("原版（scale=1）", None)]
        trials.append(("全关（scale=0）", np.zeros(len(bind))))
        for floor, lo, hi in ((0.2, 0.05, 0.6), (0.3, 0.05, 0.6),
                              (0.2, 0.1, 0.8), (0.4, 0.05, 0.6),
                              (0.3, 0.15, 1.0)):
            kern = smoothstep(lo, hi, gn)
            trials.append(("floor=%.1f lo=%.2f hi=%.2f" % (floor, lo, hi),
                           floor + (1.0 - floor) * kern))

        print("\n" + "=" * 100)
        print("compliance 平滑衰减（原版引导 fillet=0，λ=0，无限幅）")
        print("=" * 100)
        print("%-22s %10s %10s %10s %10s %10s %9s %8s"
              % ("配置", "入边4880→", "出边→5040", "邻1 4720→", "邻2 4800→",
                 "4960转角", "最大转角", "中点ID"))
        rows = []
        for tag, taper in trials:
            r = evaluate(tag, taper)
            rows.append(r)
            print("%-22s %10.4f %10.4f %10.4f %10.4f %10.2f %9.2f %8d"
                  % (tag, r["pr"]["入边"][0], r["pr"]["出边"][0],
                     r["pr"]["邻边1"][0], r["pr"]["邻边2"][0],
                     r["tv"], r["mx"], r["mid"]))

        print("\n整段峰值位置与最短边长、内外极点：")
        print("%-22s %10s %10s %12s" % ("配置", "峰值@s", "最短边长", "极点最大变化"))
        base_ext = rows[0]["ext"]
        for r in rows:
            extd = max(abs(x - y) for x, y in zip(r["ext"], base_ext))
            print("%-22s %10.3f %10.5f %12.4f"
                  % (r["tag"], r["at"], r["el"], extd))

        # ---- 判定 ----
        print("\n" + "=" * 100)
        print("判定（相对原版 109.78° 与全关 124.29°）")
        print("=" * 100)
        base = rows[0]
        best = None
        for r in rows[1:]:
            good_main = r["pr"]["入边"][0] > base["pr"]["入边"][0]
            good_nb = r["pr"]["邻边1"][0] >= -3.1238 - 1e-9
            good_peak = r["mx"] < base["mx"]
            ok = good_main and good_nb and good_peak
            print("%-22s 入边改善 %-5s 邻边不劣于全关 %-5s 峰值下降 %-5s  %s"
                  % (r["tag"], good_main, good_nb, good_peak,
                     "**同时满足**" if ok else ""))
            if ok and (best is None or r["mx"] < best["mx"]):
                best = r
        print()
        if best:
            print("采纳候选：%s" % best["tag"])
            print("  入边 %.4f -> %.4f，最大转角 %.2f° -> %.2f°，极点变化 %.4f"
                  % (base["pr"]["入边"][0], best["pr"]["入边"][0],
                     base["mx"], best["mx"],
                     max(abs(x - y) for x, y in zip(best["ext"], base["ext"]))))
        else:
            print("没有任何 taper 同时满足三项；按判据停止扫 compliance。")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
