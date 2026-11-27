# -*- coding: utf-8 -*-
r"""沿实际数据流排查 s=0.96 的 0.1166：锁定峰值顶点，逐层比较。

用户给出的关系：固定 depth、radius 和外轨时，

    ΔT(s,d) = (1 + d/R)/2 · ΔI(s)

所以如果实际送进公式的 ΔI(0.96) = 0，该项在 s=0.96 应严格为零。
先查实际数据流是否这样。

    mayapy tests/fillet_trace.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import center_coupling as cc  # noqa: E402

BEND = 146.1265
SIDES, RINGS, HALF = 80, 201, 6.0
FILLET = 0.4


def main():
    bind = []
    for row in range(RINGS):
        s = -HALF + row * (2 * HALF / (RINGS - 1))
        for j in range(SIDES):
            a = 2 * math.pi * j / SIDES
            bind.append((s, math.cos(a), math.sin(a)))
    bind = np.asarray(bind, float)
    w1 = np.array([ref.skin_weight(p[0]) for p in bind])
    W = np.column_stack([1.0 - w1, w1])
    angle = math.radians(abs(BEND))
    radius = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
    plane = math.radians(180.0 if BEND < 0 else 0.0)

    g_ref = np.asarray(ref.hinge_displacement(bind, bind, W, BEND, 0.0, 0.0))
    g4 = cc.guide_with_fillet(bind, bind, W, FILLET)
    d = np.linalg.norm(g4 - g_ref, axis=1)
    worst = int(np.argmax(d))
    print("峰值顶点 ID = %d  station = %+.4f  周向索引 j = %d  w1 = %.4f"
          % (worst, bind[worst][0], worst % SIDES, w1[worst]))
    print("  |ΔT| = %.6f" % d[worst])
    print("  bind = (%+.5f, %+.5f, %+.5f)" % tuple(bind[worst]))

    # 该顶点的实际输入
    station = np.asarray(bind[:, 0], float)
    y = np.asarray(bind[:, 1], float)
    z = np.asarray(bind[:, 2], float)
    depth_all = y * np.cos(0.0 - plane) - z * np.sin(0.0 - plane)
    lateral_all = y * np.sin(0.0 - plane) + z * np.cos(0.0 - plane)
    ext = np.full_like(station, radius)

    inner_ref = ref.hinge_section(station, ext, radius, W, angle)
    outer_ref = ref.hinge_section(station, -ext, radius, W, angle)
    inner_f = np.array([cc.hinge_section_filleted(s, +e, radius, W[i:i + 1],
                                                  angle, FILLET)
                        for i, (s, e) in enumerate(zip(station, ext))])
    outer_f = np.array([cc.hinge_section_filleted(s, -e, radius, W[i:i + 1],
                                                  angle, FILLET)
                        for i, (s, e) in enumerate(zip(station, ext))])

    i = worst
    s_i = float(station[i])
    dI = inner_f[i] - inner_ref[i]
    dO = outer_f[i] - outer_ref[i]
    print("\n该顶点处的实际数据流：")
    print("  station = %+.4f  depth = %+.5f  radius = %.5f" % (s_i, depth_all[i], radius))
    print("  inner_ref = (%+.6f, %+.6f)" % tuple(inner_ref[i]))
    print("  inner_fil = (%+.6f, %+.6f)" % tuple(inner_f[i]))
    print("  Δinner    = (%+.3e, %+.3e)  |Δinner| = %.3e"
          % (dI[0], dI[1], np.linalg.norm(dI)))
    print("  outer_ref = (%+.6f, %+.6f)" % tuple(outer_ref[i]))
    print("  outer_fil = (%+.6f, %+.6f)" % tuple(outer_f[i]))
    print("  Δouter    = (%+.3e, %+.3e)  |Δouter| = %.3e"
          % (dO[0], dO[1], np.linalg.norm(dO)))

    # 用用户的公式核对
    dd = float(depth_all[i])
    coef = (1.0 + dd / radius) / 2.0
    pred = coef * dI + (1.0 - coef) * dO
    print("\n按 ΔT = (1+d/R)/2·ΔI + (1-d/R)/2·ΔO 预测（含外轨项）：")
    print("  (1+d/R)/2 = %.6f   depth/R = %+.4f" % (coef, dd / radius))
    print("  预测 ΔT = (%+.6e, %+.6e)  |预测| = %.3e"
          % (pred[0], pred[1], np.linalg.norm(pred)))

    # 实际 ΔT
    def target(g):
        return bind + g
    dT = (target(g4) - target(g_ref))[i]
    print("  实际 ΔT = (%+.6e, %+.6e)  |实际| = %.3e"
          % (dT[0], dT[1], np.linalg.norm(dT)))

    # 直接看 hinge_section 在若干 station 处的差（应为 0）
    print("\nhinge_section 在若干 station 处的 Δinner / Δouter（j=0 那一列）：")
    print("%10s %14s %14s %14s" % ("station", "|Δinner|", "|Δouter|", "|ΔT| 实测"))
    for s_probe in (-1.0, -0.5, 0.0, 0.5, 0.9, 0.96, 1.0, 1.5, 1.62):
        # j=0 顶点的 ID = 环序数 * SIDES。用 station 的取值找环。
        ring_of = np.round((station + HALF) / (2 * HALF / (RINGS - 1))).astype(int)
        candidates = np.nonzero(ring_of ==
                                int(np.round((s_probe + HALF) /
                                             (2 * HALF / (RINGS - 1)))))[0]
        if len(candidates) == 0:
            continue
        st_act = station[candidates[0]]
        sel = candidates[np.argmin(np.abs(station[candidates] - s_probe))]
        dmag = np.linalg.norm(inner_f[sel] - inner_ref[sel])
        omag = np.linalg.norm(outer_f[sel] - outer_ref[sel])
        print("%10.3f %14.3e %14.3e %14.3e"
              % (st_act, dmag, omag, d[sel]))


    print("\n结论线索：若 s=0.96 处 |Δinner| 与 |Δouter| 都为 0，")
    print("但 |ΔT| 非零，则差异来自数据流的其它环节（如 W 的切片或 ext）。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
