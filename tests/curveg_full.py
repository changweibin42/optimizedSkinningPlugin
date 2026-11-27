# -*- coding: utf-8 -*-
r"""方向 A 第一轮：曲线因子加进 CoR 中心积分，A/B 对比。

阶段：
  1. 用插件 alg0 的输出作为「原版 CoR」基准（已存成 _aban_*.npy）
  2. Python 复刻 cor_paper::computeCenters，先核对能否复现 alg0
  3. 打开曲线因子，比较最终几何

CoR 的变形公式（Disney CoR）：先求中心 c_i，再让顶点绕 c_i 做
权重混合旋转，最后把顶点放回：v_i = c_i + R_blend(c_i→x_i)(x_i - c_i)

曲线因子只作用在**求中心**的积分上：
    g = exp( -(Δs/R)² / (2σ²) ),  σ=0.1, Δs = 采样点参数 - 顶点参数
Δs 按 radius 归一化，所以跨模型作用范围一致。

    mayapy tests/curveg_full.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

HALF, BEND = 6.0, 146.1265
SIGMA_OVER_R = 0.1
CASES = (("lo", 24, 19), ("hi", 80, 201))


def load(tag):
    bind = np.load(os.path.join(HERE, "_aban_%s_bind.npy" % tag))
    posed = np.load(os.path.join(HERE, "_aban_%s_posed.npy" % tag))
    w1 = np.load(os.path.join(HERE, "_aban_%s_w1.npy" % tag))
    faces = []
    with open(os.path.join(HERE, "_aban_%s_faces.txt" % tag)) as fh:
        for line in fh:
            faces.append(tuple(int(x) for x in line.split()))
    return bind, posed, w1, faces


def quads_to_tris(faces):
    tris = []
    for a, b, c, d in faces:
        tris.append((a, b, c))
        tris.append((a, c, d))
    return tris


def cor_centers(points, weights, tris, sigma, gate_sigma=None, radius=1.0,
                curve=None):
    """cor_paper::computeCenters 的 Python 复刻（含可选曲线因子）。

    原实现按骨骼对累积三角形上的 mass/moment。这里同样：
      每个三角形取质心 cen 与权重均值 wm
      对每个骨骼对 (i,j)，mass += area·wm_i·wm_j
                              moment += area·wm_i·wm_j·cen
    曲线因子乘在 (i,j) 对每个顶点 v 的贡献上。
    """
    points = np.asarray(points, float)
    weights = np.asarray(weights, float)
    n = len(points)
    nb = weights.shape[1]
    if curve is None:
        curve = (points[:, 0] - points[:, 0].min()) / \
            max(points[:, 0].max() - points[:, 0].min(), 1e-12)

    mom = np.zeros((n, 3))
    mass = np.zeros(n)

    tri_arr = np.asarray(tris)
    cen = points[tri_arr].mean(axis=1)              # (T,3)
    wm = weights[tri_arr].mean(axis=1)              # (T,nb)
    e1 = points[tri_arr[:, 1]] - points[tri_arr[:, 0]]
    e2 = points[tri_arr[:, 2]] - points[tri_arr[:, 0]]
    area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)   # (T,)
    good = area > 1e-18
    tri_arr, cen, wm, area = tri_arr[good], cen[good], wm[good], area[good]

    # 采样点的曲线参数
    sp = curve[tri_arr].mean(axis=1)                # (T,)

    for i in range(nb):
        for j in range(i + 1, nb):
            m = area * wm[:, i] * wm[:, j]          # (T,)
            if not np.any(m > 0):
                continue
            for b in range(3):
                v = tri_arr[:, b]
                g = np.ones(len(v))
                if gate_sigma and gate_sigma > 0:
                    d = (sp - curve[v]) / radius
                    g = np.exp(-(d * d) / (2.0 * gate_sigma ** 2))
                contrib = m * g
                mom[:, 0] += np.bincount(v, weights=contrib * cen[:, 0],
                                         minlength=n)
                mom[:, 1] += np.bincount(v, weights=contrib * cen[:, 1],
                                         minlength=n)
                mom[:, 2] += np.bincount(v, weights=contrib * cen[:, 2],
                                         minlength=n)
                mass += np.bincount(v, weights=contrib, minlength=n)

    out = points.copy()
    ok = mass > 0
    out[ok] = mom[ok] / mass[ok, None]
    return out, ok


def apply_cor(points, centers, weights, quats, scale=1.0):
    """v_i = c_i + R_blend·(x_i - c_i)，R_blend 由 quats 按权重混合。"""
    R = np.einsum("nk,kij->nij", weights,
                  np.stack([ref.rotation_matrix(q) for q in quats]))
    d = (points - centers)[:, :, None]
    return centers + np.einsum("nij,njk->nik", R, d)[:, :, 0]


def meridian_stats(pts, sides, rings, half=HALF):
    """沿经线的最大有符号转角、峰值 station、最短边长、反向边数。"""
    st = np.array([-half + r * (2 * half / (rings - 1)) for r in range(rings)])
    idx = np.array([r * sides for r in range(rings)])
    P = pts[idx]
    u = P[1:-1] - P[:-2]
    v = P[2:] - P[1:-1]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    turn = np.degrees(np.arctan2(cross, dot))
    el = np.linalg.norm(np.diff(P, axis=0), axis=1)
    Pg = pts[idx]
    # 反向边：最终边与 bind 边方向点积为负
    bi = np.array([r * sides for r in range(rings)])
    return dict(turn=turn, st=st[1:-1], el=el, st_edges=st, P=P)


def main():
    for tag, sides, rings in CASES:
        bind, posed, w1, faces = load(tag)
        rad = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
        W = np.column_stack([1.0 - w1, w1])
        tris = quads_to_tris(faces)
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]

        print("\n" + "=" * 80)
        print("%s 点（radius %.4f，三角形 %d）" % (tag, rad, len(tris)))
        print("=" * 80)

        base_c, ok = cor_centers(bind, W, tris, 0.1)
        base_v = apply_cor(bind, base_c, W, quats)
        err = np.linalg.norm(base_v - posed, axis=1)
        print("  Python 复刻 vs 插件 alg0：mean %.4e  max %.4e  %s"
              % (err.mean(), err.max(),
                 "**一致**" if err.max() < 1e-4 else "**不一致**"))

        gated_c, ok2 = cor_centers(bind, W, tris, 0.1,
                                   gate_sigma=SIGMA_OVER_R, radius=rad)
        gated_v = apply_cor(bind, gated_c, W, quats)

        dc = np.linalg.norm(gated_c - base_c, axis=1)
        dv = np.linalg.norm(gated_v - base_v, axis=1)
        print("  曲线因子带来的中心位移：mean %.4e  max %.4e"
              % (dc.mean(), dc.max()))
        print("  顶点位移变化：mean %.4e  max %.4e" % (dv.mean(), dv.max()))

        for name, pts in (("原 CoR", base_v), ("曲线引导", gated_v)):
            m = meridian_stats(pts, sides, rings)
            turn, st = m["turn"], m["st"]
            imax = int(np.argmax(np.abs(turn)))
            reg = [i for i in range(len(st)) if 1.8 <= abs(st[i]) <= 3.0]
            jmax = max(reg, key=lambda i: abs(turn[i])) if reg else imax
            print("    %-8s 全段最大 %7.2f°@%+.2f  过渡区最大 %7.2f°@%+.2f  "
                  "最短边 %.5f  中心唯一值 %d"
                  % (name, abs(turn[imax]), st[imax],
                     abs(turn[jmax]), st[jmax], m["el"].min(),
                     len(np.unique(np.round(base_c, 6)))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
