# -*- coding: utf-8 -*-
"""与坐标系无关的逐环圆度：用环自身质心 + 最佳拟合平面。

之前用 hypot 或绑定基投影测径向，在 x>0 一侧会被子骨 146 度旋转污染
（径向会莫名涨到 1.8）。这里只用一个环内的 80 个点本身：
  · 质心 c
  · 最佳拟合平面的法向（环的 PCA 最小主成分方向）
  · 各点到 c 在该平面内投影的距离
  · 圆度 = std(dist) / mean(dist)

    python tests/ring_roundness.py
"""
import json
import math
import sys

import numpy as np

HERE = r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin"


def ring_roundness(P, rings, ring_ids, sides=80):
    """返回 {ring: (roundness, mean_radius, n)}"""
    out = {}
    for rid in ring_ids:
        m = np.where(rings == rid)[0]
        if len(m) != sides:
            continue
        Q = P[m]
        c = Q.mean(axis=0)
        d = Q - c
        # 最佳拟合平面：最小奇异值方向为法线
        try:
            _, _, vt = np.linalg.svd(d, full_matrices=False)
        except np.linalg.LinAlgError:
            continue
        nrm = vt[2]
        # 投影到平面内的两个基
        e1 = vt[0]
        e2 = vt[1]
        u = d @ e1
        v = d @ e2
        r = np.hypot(u, v)
        mr = r.mean()
        if mr < 1e-12:
            continue
        out[rid] = (float(r.std() / mr), float(mr), len(m))
    return out


def main():
    case = json.load(open(r"D:\tmp\ocd_case3.json"))
    exp = json.load(open(HERE + r"\tests\_clamp_exp.json"))

    bind = np.asarray(case["mesh"]["bindPoints"], float)
    rings = np.round(bind[:, 0] / 0.12).astype(int)
    ring_ids = np.unique(rings)

    p16 = np.asarray(exp["p16"], float)
    p100 = np.asarray(exp["p100"], float)

    # 夹具里的两组（门控 0/0, limit 16）
    cor = np.asarray(case["points"]["cor"], float)
    fin = np.asarray(case["points"]["final"], float)

    sets = [("CoR(算法0)", cor), ("final(算法5,限16)", fin),
            ("限100", p100)]
    res = [ring_roundness(P, rings, ring_ids) for _, P in sets]

    print("逐环圆度 std/mean（坐标系无关，越小越圆）")
    print("%8s %6s %12s %12s %12s %10s" %
          ("x", "n", "CoR", "alg5限16", "alg5限100", "16-100"))
    for rid in ring_ids:
        x0 = rid * 0.12
        if not (-4.5 <= x0 <= 4.5):
            continue
        if rid not in res[0]:
            continue
        r0 = res[0][rid]
        r1 = res[1][rid]
        r2 = res[2][rid]
        flag = ""
        if r1[0] > r0[0] * 1.3:
            flag = " ★比CoR差"
        print("%8.2f %6d %12.4f %12.4f %12.4f %+10.4f%s"
              % (x0, r0[2], r0[0], r1[0], r2[0], r2[0] - r1[0], flag))

    print("\n=== 关键环对照 ===")
    for x0 in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        rid = int(round(x0 / 0.12))
        if rid not in res[0]:
            continue
        a, b, c = res[0][rid][0], res[1][rid][0], res[2][rid][0]
        print("  x=%+5.1f  CoR=%.4f  限16=%.4f  限100=%.4f   "
              "alg5/CoR=%.2fx" % (x0, a, b, c, b / max(a, 1e-12)))


if __name__ == "__main__":
    main()
