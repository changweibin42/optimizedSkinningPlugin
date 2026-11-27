# -*- coding: utf-8 -*-
"""核实 fillet=0.4 的影响范围：为什么 |station|<0.4 的窗口会影响到 ±1.56。"""
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

    g_ref = np.asarray(ref.hinge_displacement(bind, bind, W, BEND, 0.0, 0.0))
    g4 = cc.guide_with_fillet(bind, bind, W, 0.4)
    d = np.linalg.norm(g4 - g_ref, axis=1)

    stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                         for r in range(RINGS)])
    print("每环的引导差异（|station| <= 2.0）：")
    print("%10s %12s %12s" % ("station", "max diff", "min diff"))
    for r in range(RINGS):
        seg = slice(r * SIDES, (r + 1) * SIDES)
        if abs(stations[r]) > 2.0:
            continue
        print("%10.3f %12.3e %12.3e"
              % (stations[r], d[seg].max(), d[seg].min()))

    # 直接看 hinge_section 的内外侧在 |s|<0.4 外是否变化
    angle = math.radians(abs(BEND))
    radius = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
    print("\nhinge_section 内轨在 |s| 略大于窗口时是否被改动：")
    for s in (-1.56, -0.9, -0.4, -0.36, -0.2, 0.0, 0.2, 0.36, 0.4, 0.9):
        i = int(np.argmin(np.abs(bind[:, 0] - s)))
        a1 = ref.hinge_section(np.array([s]), np.array([radius]), radius,
                               W[i:i + 1], angle)[0]
        a2 = cc.hinge_section_filleted(s, radius, radius, W[i:i + 1], angle, 0.4)
        print("  s=%+6.2f  ref=(%+.4f,%+.4f)  fil=(%+.4f,%+.4f)  差 %.3e"
              % (s, a1[0], a1[1], a2[0], a2[1],
                 np.linalg.norm(np.asarray(a1) - np.asarray(a2))))
    return 0


if __name__ == "__main__":
    sys.exit(main())
