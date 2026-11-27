"""逐顶点：final 相对 guide 目标的实现率，按 x 分段。

避免坐标系歧义：只比较"同一顶点"的三个世界坐标。
  实现率 = |final - cor| / |guide_world|
  若 < 1，说明中心求解没把顶点送到目标位置。

    python tests/guide_fulfillment.py
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

    cor_l = to_local(cor)

    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))

    guide_l = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    # 世界空间位移：局部归一 → 世界向量（不加 pivot 平移）
    guide_w = R * (guide_l[:, 0:1] * axis[None, :]
                   + guide_l[:, 1:2] * bY[None, :]
                   + guide_l[:, 2:3] * bZ[None, :])

    actual = fin - cor
    gn = np.linalg.norm(guide_w, axis=1)
    an = np.linalg.norm(actual, axis=1)
    ok = gn > 1e-6
    rate = np.where(ok, an / np.maximum(gn, 1e-12), 1.0)

    # 投影到 guide 方向的有符号实现量
    gh = guide_w / np.maximum(gn[:, None], 1e-12)
    along = np.where(ok, np.einsum("ij,ij->i", actual, gh), 0.0)

    print("bend=%.4f  R=%.6f" % (bend, R))
    print("\n按 x 分段：guide 需求 vs 实际实现")
    print("%10s %6s %12s %12s %12s %12s %12s"
          % ("x", "n", "|guide|均", "|actual|均", "沿向实现均", "实现率", "比例<0.5"))

    for lo, hi in ((-3.6, -3.0), (-3.0, -2.4), (-2.4, -1.8), (-1.8, -1.2),
                   (-1.2, -0.6), (-0.6, 0.0), (0.0, 0.6), (0.6, 1.2),
                   (1.2, 1.8), (1.8, 2.4), (2.4, 3.0), (3.0, 3.6)):
        m2 = (bind[:, 0] >= lo) & (bind[:, 0] < hi) & ok
        if m2.sum() < 20:
            continue
        print("%10s %6d %12.5f %12.5f %12.5f %12.4f %12.4f"
              % ("%.1f..%.1f" % (lo, hi), int(m2.sum()),
                 gn[m2].mean(), an[m2].mean(),
                 along[m2].mean(), rate[m2].mean(),
                 float((rate[m2] < 0.5).mean())))

    print("\n整体:")
    print("  实现率 均=%.4f  中位=%.4f" % (rate[ok].mean(), np.median(rate[ok])))
    print("  沿向实现/需求 均=%.4f" % (along[ok].sum() / gn[ok].sum()))
    print("  需求最大的 10 个顶点：")
    order = np.argsort(-gn)[:10]
    for i in order:
        print("    v%-6d x=%+7.2f w1=%.4f  |guide|=%.4f  |actual|=%.4f  实现率=%.3f"
              % (i, bind[i, 0], W[i, 1], gn[i], an[i], rate[i]))


if __name__ == "__main__":
    main()
