# -*- coding: utf-8 -*-
r"""自检：guide_with_fillet(fillet=0) 必须逐位复现 hinge_displacement。

按用户要求比较逐顶点的 T_fil(0) − T_ref **向量模**的最大值（不是 |guide| 之差）；
若有差异，打印最大误差顶点的 ID，并按内侧轮廓 / 外侧轮廓 / 中性层 / 目标点
逐层比较，定位偏差来自哪一层。

    mayapy tests/fillet_selfcheck.py
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


def main():
    bind_list, faces = ref.cylinder() if False else (None, None)
    # 与其它脚本同一份网格
    bind = []
    for row in range(RINGS):
        s = -HALF + row * (2 * HALF / (RINGS - 1))
        for j in range(SIDES):
            a = 2 * math.pi * j / SIDES
            bind.append((s, math.cos(a), math.sin(a)))
    bind = np.asarray(bind, float)
    w1 = np.array([ref.skin_weight(p[0]) for p in bind])
    W = np.column_stack([1.0 - w1, w1])

    # original 用 bind 本身即可（比较的是两条 guide 的差，与 original 无关）
    original = bind.copy()

    g_ref = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
    g_f0 = cc.guide_with_fillet(bind, original, W, 0.0)

    T_ref = original + g_ref
    T_f0 = original + g_f0
    diff = np.linalg.norm(T_f0 - T_ref, axis=1)
    worst = int(np.argmax(diff))

    print("fillet=0 与原版的逐顶点向量差：")
    print("  max = %.3e   mean = %.3e"
          % (diff.max(), diff.mean()))
    if diff.max() < 1e-12:
        print("  PASS：逐位一致")
        exact = True
    else:
        print("  FAIL：不一致")
        print("  最大误差顶点 ID = %d  station = %+.4f  w1 = %.4f"
              % (worst, bind[worst][0], w1[worst]))
        print("    T_ref = (%+.6f, %+.6f, %+.6f)" % tuple(T_ref[worst]))
        print("    T_f0  = (%+.6f, %+.6f, %+.6f)" % tuple(T_f0[worst]))
        exact = False

        # 逐层比较
        station, y, z = bind.T
        radius = float(np.max(np.hypot(y, z)))
        angle = math.radians(abs(BEND))
        ext = np.full_like(station, radius)
        inner_ref = ref.hinge_section(station, ext, radius, W, angle)
        outer_ref = ref.hinge_section(station, -ext, radius, W, angle)
        inner_f = np.array([cc.hinge_section_filleted(s, +e, radius, W[i:i + 1],
                                                      angle, 0.0)
                            for i, (s, e) in enumerate(zip(station, ext))])
        outer_f = np.array([cc.hinge_section_filleted(s, -e, radius, W[i:i + 1],
                                                      angle, 0.0)
                            for i, (s, e) in enumerate(zip(station, ext))])
        print("  分层最大差：")
        print("    内侧轮廓 inner : %.3e" % np.abs(inner_f - inner_ref).max())
        print("    外侧轮廓 outer : %.3e" % np.abs(outer_f - outer_ref).max())
        neu_ref = (inner_ref + outer_ref) * 0.5
        neu_f = (inner_f + outer_f) * 0.5
        print("    中性层 neutral : %.3e" % np.abs(neu_f - neu_ref).max())
        g_ref2 = (neu_ref + (y * np.cos(math.radians(180)) -
                             z * np.sin(math.radians(180)))[:, None] *
                  ((inner_ref - outer_ref) / (2 * radius)))
        print("    目标点 target  : %.3e" % np.abs(T_f0 - T_ref).max())

    # 顺带打印非零圆角相对原版的差异，作为后续复跑的依据
    g_f4 = cc.guide_with_fillet(bind, original, W, 0.4)
    T_f4 = original + g_f4
    d4 = np.linalg.norm(T_f4 - T_ref, axis=1)
    print("\nfillet=0.4 相对原版的引导差异：max %.4f  mean %.4f"
          % (d4.max(), d4.mean()))
    print("  （圆角只作用于 |station| < 0.4，应只在 station=0 两圈附近非零）")
    affected = np.nonzero(d4 > 1e-9)[0]
    if len(affected):
        sts = sorted({round(float(bind[i][0]), 3) for i in affected})
        print("  受影响的 station：%s" % sts[:12])
    return 0 if exact else 1


if __name__ == "__main__":
    sys.exit(main())
