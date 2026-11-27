"""量化外轮廓亏空：逐环外极点，对比 base CoR / guide目标 / 最终输出。

判断：
  · 若 guide 就已经凹陷 → hinge_section 的外侧轮廓（depth<0 的 arc 分支）
  · 若 guide 饱满但 final 凹陷 → structuralCenters 没实现目标（响应不足/牵引不够）

外极点 = 局部系 depth 最小的顶点（depth = y，basisY=+Y，outer 在 y<0 侧）。

    python tests/outer_contour_audit.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

D = r"D:\tmp\ocd_case3.json"
R = None


def main():
    d = json.load(open(D))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)
    SIDES = 80
    STEP = 0.12

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

    cor_l = to_local(cor)
    fin_l = to_local(fin)

    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]
    print("R=%.6f  骨骼列=%s  bend=?" % (R, used))

    # 姿态角：从子骨矩阵取
    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    print("bend=%.4f deg" % bend)

    q = ref.joint_quaternion(bend, 0.0, 0.0)
    rotations = [(0.0, 0.0, 0.0, 1.0), q]
    guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    guided_l = cor_l + guide

    print("\n外极点轨迹（depth 极小者；depth 越负越靠外，= 饱满）")
    print("%8s %8s %6s %12s %12s %12s %12s"
          % ("x", "w1", "n", "CoR depth", "guide depth", "final depth", "final-guide"))
    rings = np.round(bind[:, 0] / STEP).astype(int)
    prev = None
    for k in range(-24, 25):
        rid = int(round(k))
        m2 = np.where(rings == rid)[0]
        if len(m2) != SIDES:
            continue
        x0 = rid * STEP
        # 该环外极点（depth 最小）
        dcor = cor_l[m2, 1]
        dgui = guided_l[m2, 1]
        dfin = fin_l[m2, 1]
        i_o = int(np.argmin(dcor))
        gap = dfin[i_o] - dgui[i_o]
        print("%8.3f %8.4f %6d %12.5f %12.5f %12.5f %+12.5f"
              % (x0, W[m2[0], 1], len(m2),
                 dcor[i_o], dgui[i_o], dfin[i_o], gap))

    # 用"轮廓包络"衡量：每环外极点沿 basisY 的坐标
    print("\n外轮廓（用 |depth| 最大者，单位=R）")
    print("%8s %12s %12s %12s %14s"
          % ("x", "CoR", "guide", "final", "final/guide"))
    for k in range(-24, 25):
        rid = int(round(k))
        m2 = np.where(rings == rid)[0]
        if len(m2) != SIDES:
            continue
        x0 = rid * STEP
        a = -cor_l[m2, 1].min()
        b = -guided_l[m2, 1].min()
        c = -fin_l[m2, 1].min()
        print("%8.3f %12.5f %12.5f %12.5f %14.4f"
              % (x0, a, b, c, c / max(b, 1e-9)))


if __name__ == "__main__":
    main()
