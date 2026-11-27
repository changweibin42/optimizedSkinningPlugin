"""16080 点夹具：w1≈0.5 那一圈的实际顶点排列，是否沿内侧轮廓、
是否出现位置折返。

判据：
  · 该环 80 个点是否近似共面圆（圆度）
  · 沿 j 方向（环内）的相邻距离是否均匀（不均有=侧面褶皱）
  · 该环与相邻环的中心距离、边长
  · 内侧经线（j=0）上 相邻环的 x 序列是否折返

    python tests/check_midring_fold.py
"""
import json
import math
import sys

import numpy as np

D = r"D:\tmp\ocd_case3.json"
HERE = r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests"


def main():
    d = json.load(open(D))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)
    SIDES = 80

    # 环索引：x 步长 0.12
    rings = np.round(bind[:, 0] / 0.12).astype(int)
    ring_ids = np.unique(rings)
    R0 = int(np.argmin(np.abs(ring_ids * 0.12)))  # x≈0

    print("环数 %d，每环 %d 点，x≈0 的环号 %d (x=%.3f)"
          % (len(ring_ids), SIDES, R0, ring_ids[R0] * 0.12))

    def ring_members(rid):
        return np.where(rings == rid)[0]

    def stats(P, rid):
        m = ring_members(rid)
        Q = P[m]
        c = Q.mean(axis=0)
        # Newell 法向 + 面内半径
        d2 = Q - c
        nx = np.zeros(3)
        for j in range(len(Q)):
            a = d2[j]
            b = d2[(j + 1) % len(Q)]
            nx += np.cross(a, b)
        nrm = nx / max(np.linalg.norm(nx), 1e-14)
        rad = []
        for p in d2:
            h = np.dot(p, nrm)
            rad.append(np.linalg.norm(p - nrm * h))
        rad = np.asarray(rad)
        return c, nrm, rad

    print("\n" + "=" * 88)
    print("w1≈0.5 区域各环的圆度与中心")
    print("=" * 88)
    print("  %8s %8s %12s %12s %12s"
          % ("x", "n", "CoR圆度", "final圆度", "中心距CoR"))
    for k in range(R0 - 4, R0 + 5):
        rid = ring_ids[k]
        if len(ring_members(rid)) != SIDES:
            continue
        c0, n0, r0 = stats(cor, rid)
        c1, n1, r1 = stats(fin, rid)
        rd0 = r0.std() / r0.mean()
        rd1 = r1.std() / r1.mean()
        dc = np.linalg.norm(c1 - c0)
        print("  %8.3f %8d %12.6f %12.6f %12.6f"
              % (rid * 0.12, len(ring_members(rid)), rd0, rd1, dc))

    print("\n" + "=" * 88)
    print("x=0 那一环：环内相邻点距离（j 方向均匀性）")
    print("=" * 88)
    rid = ring_ids[R0]
    m = ring_members(rid)
    order = np.argsort(np.degrees(np.arctan2(bind[m, 2], bind[m, 1])))
    mm = m[order]
    for nm, P in (("CoR", cor), ("final", fin)):
        Q = P[mm]
        seg = [np.linalg.norm(Q[j + 1] - Q[j]) for j in range(len(Q) - 1)]
        seg.append(np.linalg.norm(Q[0] - Q[-1]))
        seg = np.asarray(seg)
        print("  %-6s 段长: 均=%.5f  min=%.5f  max=%.5f  std/mean=%.4f"
              % (nm, seg.mean(), seg.min(), seg.max(), seg.std() / seg.mean()))

    print("\n" + "=" * 88)
    print("内侧经线（j=0，即 bind 的 +Y 方向）沿 x 的最终位置")
    print("=" * 88)
    # 找每环中最接近 +Y 的点
    print("  %8s %14s %14s %14s"
          % ("x", "CoR(y,z)", "final(y,z)", "移动"))
    for k in range(R0 - 3, R0 + 4):
        rid = ring_ids[k]
        m = ring_members(rid)
        if len(m) != SIDES:
            continue
        # +Y 方向
        j = np.argmax(bind[m, 1])
        i = m[j]
        dy = fin[i, 1] - cor[i, 1]
        dz = fin[i, 2] - cor[i, 2]
        print("  %8.3f  (%+7.4f,%+7.4f)  (%+7.4f,%+7.4f)  (%+7.4f,%+7.4f)"
              % (rid * 0.12, cor[i, 1], cor[i, 2], fin[i, 1], fin[i, 2], dy, dz))


if __name__ == "__main__":
    main()
