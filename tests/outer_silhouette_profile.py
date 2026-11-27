"""外侧轮廓剖面：沿 -Z 看（截图视角），逐 x 取最小 y。

对比 CoR / 关限幅 final，找凹陷位置与深度。

    python tests/outer_silhouette_profile.py
"""
import json
import math
import sys

import numpy as np

SIDES = 80
STEP = 0.12


def main():
    d = json.load(open(r"D:\tmp\ocd_case3.json"))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)

    rings = np.round(bind[:, 0] / STEP).astype(int)
    rid_list = sorted(set(rings.tolist()))

    print("外侧轮廓（世界 XY，逐环取 min y；y 越小越靠外）")
    print("%8s %10s %12s %12s %12s %12s"
          % ("x", "w1", "CoR_miny", "final_miny", "Δ", "内侧ymax(参考)"))

    rows = []
    for rid in rid_list:
        m = np.where(rings == rid)[0]
        x0 = rid * STEP
        if abs(x0) > 4.2:
            continue
        yc = cor[m, 1]
        yf = fin[m, 1]
        rows.append((x0, yc.min(), yf.min(), yc.max(), yf.max(), len(m)))
        print("%8.3f %10.4f %12.5f %12.5f %+12.5f %12.5f"
              % (x0, 0.0, yc.min(), yf.min(), yf.min() - yc.min(), yf.max()))

    print("\n外侧轮廓二阶差分（正=局部内凹）")
    print("%8s %14s %14s" % ("x", "CoR", "final"))
    for i in range(1, len(rows) - 1):
        x0, yc0, yf0 = rows[i - 1][0], rows[i - 1][1], rows[i - 1][2]
        x1, yc1, yf1 = rows[i][0], rows[i][1], rows[i][2]
        x2, yc2, yf2 = rows[i + 1][0], rows[i + 1][1], rows[i + 1][2]
        if abs(x1) > 3.8:
            continue
        dc = yc2 - 2 * yc1 + yc0
        df = yf2 - 2 * yf1 + yf0
        print("%8.3f %14.6e %14.6e" % (x1, dc, df))

    # 位移方向：final 相对 CoR 在每个 x 的外侧位移
    print("\n外侧位移分解 (final - CoR)：")
    print("%8s %14s %14s" % ("x", "Δy", "Δx"))
    for x0, yc, yf, _, _, _ in rows:
        if abs(x0) > 3.8:
            continue
        m = np.where(rings == round(x0 / STEP))[0]
        i = m[np.argmin(cor[m, 1])]
        print("%8.3f %14.5f %14.5f" % (x0, fin[i, 1] - cor[i, 1],
                                       fin[i, 0] - cor[i, 0]))


if __name__ == "__main__":
    main()
