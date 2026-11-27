"""修正版：正侧顶点先撤掉子骨旋转，再进局部系。

夹具里 cor/final 是对象空间；x>0 的部分已绕 Z 转了 -bend。
必须先按顶点所属骨的权重撤旋转，才能和 bind 系的 guide 比较。

    python tests/guide_fulfillment_fixed.py
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

    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]
    w1 = W[:, 1] if W.shape[1] > 1 else np.zeros(n)

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))

    # 单骨近似：x>0 且 w1>0.99 的顶点撤掉整角旋转
    cb, sb = math.cos(math.radians(bend)), math.sin(math.radians(bend))
    print("bend=%.4f  cos=%.5f sin=%.5f  R=%.6f" % (bend, cb, sb, R))

    def unrotate(P):
        Q = P.copy()
        # 绕 Z 撤掉 bend
        x = Q[:, 0] * cb + Q[:, 1] * sb
        y = -Q[:, 0] * sb + Q[:, 1] * cb
        Q[:, 0], Q[:, 1] = x, y
        return Q

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

    # 只对 w1 接近 1 的顶点撤旋转，其余保持（混合区本来就该在两者之间）
    def prep(P):
        Q = P.copy()
        m1 = w1 > 0.99
        Q[m1] = unrotate(P[m1])
        return to_local(Q)

    cor_l = prep(cor)

    # guide 用参考实现
    guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    gn = np.linalg.norm(guide, axis=1)

    print("\n|guide| 按 x 分段（修正后）")
    print("%12s %6s %12s %12s" % ("x", "n", "|guide|均", "|guide|max"))
    for lo, hi in ((-3.6, -3.0), (-3.0, -2.4), (-2.4, -1.8), (-1.8, -1.2),
                   (-1.2, -0.6), (-0.6, 0.0), (0.0, 0.6), (0.6, 1.2),
                   (1.2, 1.8), (1.8, 2.4), (2.4, 3.0), (3.0, 3.6),
                   (3.6, 12.1)):
        m2 = (bind[:, 0] >= lo) & (bind[:, 0] < hi)
        if m2.sum() < 20:
            continue
        print("%12s %6d %12.5f %12.5f"
              % ("%.1f..%.1f" % (lo, hi), int(m2.sum()),
                 gn[m2].mean(), gn[m2].max()))

    print("\n纯单骨段（w1>0.99 或 w1<0.01）的 |guide|：")
    for lab, m2 in (("w1<0.01", w1 < 0.01), ("w1>0.99", w1 > 0.99)):
        if m2.sum() == 0:
            continue
        print("  %-9s n=%5d  |guide|均=%.6f max=%.6f"
              % (lab, int(m2.sum()), gn[m2].mean(), gn[m2].max()))


if __name__ == "__main__":
    main()
