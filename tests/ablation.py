# -*- coding: utf-8 -*-
r"""结构项消融：定位是哪个求解项把 b- 的切向投影压到 -1.261。

固定产生 109.78° 的 T_fil、权重、旋转、顶点 ID，λ=0，关闭限幅。

三组：
  基线            复现 109.78° 与 b- 的切向投影比 -1.261
  关闭纵向差分      compliance 置零（rhs / 线性算子 / 预条件器同步）
  关闭凸侧牵引      rhs 里的牵引系数 7 -> 0

判据：b- 的切向投影比 = (b-_final · t_hat) / |b-_guide|
  该值 > -1 表示这条边还有正向投影（未折返）。
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import center_coupling as cc  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
FILLET = 0.4


def structural_centers_ablate(edges, original, weights, rotations, guide, outer,
                              lambda_c=0.0, use_compliance=True,
                              traction_gain=7.0, max_iters=300):
    """cc.structural_centers_coupled 的消融版。只改两处开关。"""
    a, b, rest, k = edges
    n = len(original)
    A = ref.center_response(weights, rotations)
    AT = A.transpose(0, 2, 1)
    R = np.einsum("nk,kij->nij", weights, ref.rotation_matrix(rotations)) - A

    def apply(matrix, x):
        return (matrix @ x[:, :, None])[:, :, 0]

    def tri_scatter(values):
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
    left = apply(frame[a], traction)
    right = apply(frame[b], traction)
    pull = np.stack([
        np.bincount(a, weights=l, minlength=n) -
        np.bincount(b, weights=r, minlength=n) for l, r in zip(left.T, right.T)],
        axis=1)

    guide_mass = 2.0 * degree[:, None]
    rest_k = k * axial ** 2
    current_length = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    longitudinal = rest_k * length / np.maximum(current_length, 0.15 * length)

    def lap(x):
        return tri_scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n) +
                  np.bincount(b, weights=longitudinal, minlength=n))

    if use_compliance:
        com = 8.0 * np.minimum(1.0, 4.0 * np.abs(weights[:, 0] - weights[:, 1])) ** 2 \
            / np.maximum(lap_degree, 1e-12)
    else:
        com = np.zeros(n)

    rest_detail = tri_scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(com[:, None] * (apply(R, rest_detail) - lap(original)))

    rhs = apply(AT,
                tri_scatter(k[:, None] * (target - (original[b] - original[a]))) +
                guide_mass * guide + traction_gain * pull + curvature_rhs)

    center_edge_w = k * axial ** 2

    def center_lap(x):
        return tri_scatter(center_edge_w[:, None] * (x[b] - x[a]))

    center_degree = (np.bincount(a, weights=center_edge_w, minlength=n) +
                     np.bincount(b, weights=center_edge_w, minlength=n))

    def system(x):
        u = apply(A, x)
        elastic = tri_scatter(k[:, None] * (u[b] - u[a])) + guide_mass * u
        out = apply(AT, elastic + lap(com[:, None] * lap(u))) + ridge[:, None] * x
        if lambda_c > 0.0:
            out = out + lambda_c * center_lap(x)
        return out

    diag = (degree + guide_mass[:, 0] + com * lap_degree ** 2 +
            np.bincount(a, weights=longitudinal ** 2 * com[b], minlength=n) +
            np.bincount(b, weights=longitudinal ** 2 * com[a], minlength=n))
    diag = diag + lambda_c * center_degree
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
        # 不用 last-iterate 的 rz，交给外层判断
        if rz <= tol:
            break
    return x, {"A": A}


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
        g_fil = cc.guide_with_fillet(bind, original, W, FILLET)
        T_fil = original + g_fil
        A = ref.center_response(W, quats)

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])

        # ---- 先核对三点各自的 A（排除诊断复用中点矩阵）----
        print("=" * 78)
        print("核对：三点各自的 A 奇异值（按各自顶点 ID 重算）")
        print("=" * 78)
        k_peak = 61
        idx3 = [meridian[k_peak], meridian[k_peak + 1], meridian[k_peak + 2]]
        for tag, idx in zip(("i-1", "i", "i+1"), idx3):
            sv = np.linalg.svd(np.asarray(A[idx], float), compute_uv=False)
            print("  %-4s 顶点 %5d  station %+.3f  w1=%.4f  奇异值 %s"
                  % (tag, idx, stations[[k_peak, k_peak + 1, k_peak + 2][
                      ("i-1", "i", "i+1").index(tag)]], w1[idx],
                     np.array2string(sv, precision=6)))

        def metrics(final, label):
            pts = final[meridian]
            turn = gvf.signed_turning_2d(pts)
            all_s = stations[1:]
            imax = int(np.argmax(np.abs(turn)))
            # T_fil 上的引导边
            bg_m = T_fil[meridian[k_peak + 1]] - T_fil[meridian[k_peak]]
            bg_p = T_fil[meridian[k_peak + 2]] - T_fil[meridian[k_peak + 1]]
            tm = bg_m / np.linalg.norm(bg_m)
            tp = bg_p / np.linalg.norm(bg_p)
            fm = final[meridian[k_peak + 1]] - final[meridian[k_peak]]
            fp = final[meridian[k_peak + 2]] - final[meridian[k_peak + 1]]
            proj_m = float(fm @ tm) / np.linalg.norm(bg_m)
            proj_p = float(fp @ tp) / np.linalg.norm(bg_p)
            # 极点位移（中线及相邻两圈）
            mid = RINGS // 2
            ext = []
            for r in (mid - 1, mid, mid + 1):
                seg = np.arange(r * SIDES, (r + 1) * SIDES)
                yb = bind[seg, 1]
                for pick in (int(np.argmax(yb)), int(np.argmin(yb))):
                    i = seg[pick]
                    ext.append(float(np.linalg.norm(final[i] - original[i])))
            return dict(label=label, mx=abs(turn[imax]), at=all_s[imax],
                        proj_m=proj_m, proj_p=proj_p,
                        len_m=float(np.linalg.norm(fm)),
                        len_p=float(np.linalg.norm(fp)),
                        ext=ext)

        groups = (
            ("基线", dict(use_compliance=True, traction_gain=7.0)),
            ("关闭纵向差分", dict(use_compliance=False, traction_gain=7.0)),
            ("关闭凸侧牵引", dict(use_compliance=True, traction_gain=0.0)),
            ("两者都关", dict(use_compliance=False, traction_gain=0.0)),
        )

        print("\n" + "=" * 78)
        print("消融结果（λ=0，关闭限幅）")
        print("=" * 78)
        print("%-14s %10s %8s %12s %12s %10s %10s"
              % ("组别", "最大|转角|", "@s", "b-投影比", "b+投影比", "|b-|", "|b+|"))
        rows = []
        for label, kw in groups:
            d, info = structural_centers_ablate(
                edges, original, W, quats, g_fil, outer, lambda_c=0.0, **kw)
            final = original + np.einsum("nij,nj->ni", info["A"], d)
            m = metrics(final, label)
            rows.append(m)
            print("%-14s %10.2f %8.3f %12.4f %12.4f %10.5f %10.5f"
                  % (label, m["mx"], m["at"], m["proj_m"], m["proj_p"],
                     m["len_m"], m["len_p"]))

        print("\n极点位移（中线及相邻两圈的 max-y / min-y 顶点）：")
        print("%-14s %s" % ("组别", "  ".join("r%+d" % r for r in (-1, 0, 1))))
        base_ext = rows[0]["ext"]
        for m in rows:
            diff = max(abs(x - y) for x, y in zip(m["ext"], base_ext))
            print("%-14s %s   相对基线最大变化 %.4f"
                  % (m["label"],
                     "  ".join("%.3f/%.3f" % (m["ext"][i * 2], m["ext"][i * 2 + 1])
                               for i in range(3)), diff))

        print("\n判读：")
        for m in rows[1:]:
            ok = m["proj_m"] > -1.0
            print("  %-14s b- 投影比 %+.4f  %s   最大转角 %.2f° @ s=%.3f"
                  % (m["label"], m["proj_m"],
                     "(已恢复正向投影)" if ok else "(仍在折返)",
                     m["mx"], m["at"]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
