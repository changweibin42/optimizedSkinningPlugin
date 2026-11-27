"""用世界空间 XY 轮廓对比（与截图视角一致）。

图像是沿 -Z 看，所以轮廓 = 每个 x 列上的最小 y（下边界）。
比较三条曲线：
  · CoR（原始旋转中心输出）
  · guide 目标（= cor + guide 换算到世界）
  · final（当前算法 5，关限幅）

若 guide 已经凹陷 → hinge_section 外侧轮廓的问题
若 guide 饱满而 final 凹陷 → 中心求解没实现目标

    python tests/silhouette_compare.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref


def main():
    d = json.load(open(r"D:\tmp\ocd_case3.json"))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)

    child = [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(d["hinge"]["axis"], float)
    axis = axis / np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)

    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

    def from_local(L):
        return (pivot[None, :]
                + R * (L[:, 0:1] * axis[None, :]
                       + L[:, 1:2] * bY[None, :]
                       + L[:, 2:3] * bZ[None, :]))

    cor_l = to_local(cor)
    fin_l = to_local(fin)

    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    print("bend=%.4f  R=%.6f  骨骼列=%s" % (bend, R, used))

    guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    guided_world = cor + R * (guide[:, 0:1] * axis[None, :]
                              + guide[:, 1:2] * bY[None, :]
                              + guide[:, 2:3] * bZ[None, :])

    # 下轮廓：每个 x 列的最小 y
    def silhouette(P, xlo=-4.0, xhi=4.0, bins=81):
        edges = np.linspace(xlo, xhi, bins + 1)
        xs, ys = [], []
        for a, b in zip(edges[:-1], edges[1:]):
            m2 = (P[:, 0] >= a) & (P[:, 0] < b)
            if m2.sum() < 4:
                continue
            xs.append(0.5 * (a + b))
            ys.append(P[m2, 1].min())
        return np.asarray(xs), np.asarray(ys)

    xc, yc = silhouette(cor)
    xg, yg = silhouette(guided_world)
    xf, yf = silhouette(fin)

    print("\n下轮廓对比（世界 XY 投影，y 越小越靠外）")
    print("%8s %14s %14s %14s %14s"
          % ("x", "CoR", "guide目标", "final", "final-guide"))
    for x0, a, b, c in zip(xc, yc, yg, yf):
        if abs(x0) > 3.5:
            continue
        print("%8.3f %14.6f %14.6f %14.6f %+14.6f"
              % (x0, a, b, c, c - b))

    # 用二阶差分找轮廓的凹陷
    print("\n下轮廓二阶差分（负=局部向外凸，正=局部向内凹）")
    print("%8s %14s %14s" % ("x", "CoR", "final"))
    for i in range(1, len(xc) - 1):
        if abs(xc[i]) > 3.5:
            continue
        dc = yc[i + 1] - 2 * yc[i] + yc[i - 1]
        df = yf[i + 1] - 2 * yf[i] + yf[i - 1]
        print("%8.3f %14.6e %14.6e" % (xc[i], dc, df))

    # 统计：final 相对 guide 的亏空
    gap = yf - yg
    print("\nfinal 相对 guide 的下轮廓亏空：")
    print("  均=%.5f  max=%.5f @x=%.2f" % (gap.mean(), gap.max(), xc[np.argmax(gap)]))
    print("  （正值表示 final 比目标更靠内，即外轮廓欠饱满）")


if __name__ == "__main__":
    main()
