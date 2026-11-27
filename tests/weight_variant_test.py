"""验证：Python 预览的"好效果"是否来自 preview_influences 那套宽过渡权重。

对比三组，同一网格、同一姿态、同一算法(5)：
  A) 场景真实权重（16080 点网格上原样的 w1）
  B) preview_influences 风格：w = 0.75*skin_weight(s) + 0.25*smooth(0.5+s/9.0)
  C) 纯 skin_weight(s)

若 B 明显更饱满，则 Python 效果来自权重设计，与求解器无关。

    python tests/weight_variant_test.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

D = r"D:\tmp\ocd_case3.json"
STEP = 0.12
R = 0.984706


def main():
    d = json.load(open(D))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    faces = d["mesh"]["faces"]
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)

    child = [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(d["hinge"]["axis"], float)
    axis /= np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)
    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    RR = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / RR, (rad / RR) @ bY, (rad / RR) @ bZ])

    def to_local(P):
        o = np.asarray(P, float) - pivot
        return np.column_stack([(o @ axis) / RR, (o @ bY) / RR, (o @ bZ) / RR])

    cor_l = to_local(cor)

    Wreal = np.zeros((n, 2))
    for i, row in enumerate(d["weightsSparse"]):
        Wreal[i, 0] = float(row.get("0", 0.0))
        Wreal[i, 1] = float(row.get("1", 0.0))

    m = np.asarray(child["worldMatrix"], float)
    bend = math.degrees(math.atan2(m[4], m[0]))
    q = ref.joint_quaternion(bend, 0.0, 0.0)
    rots = [(0.0, 0.0, 0.0, 1.0), q]
    edges = ref.local_edges(bind_l.tolist(), faces)

    s = bind_l[:, 0] * RR   # 还原成"station"尺度（半径单位）
    # 场景真实权重的过渡宽度
    w1r = Wreal[:, 1]
    nz = np.abs(s)[(w1r > 0.02) & (w1r < 0.98)]
    print("真实权重过渡区 |station| 最大 = %.3f R" % (nz.max() if nz.size else 0))

    def make(spec):
        if spec == "real":
            return Wreal
        if spec == "skin":
            w = np.array([ref.skin_weight(x) for x in s])
        elif spec == "preview":
            w = np.array([0.75 * ref.skin_weight(x) + 0.25 * ref.smooth(0.5 + x / 9.0)
                          for x in s])
        w = np.clip(w, 0.0, 1.0)
        return np.column_stack([1.0 - w, w])

    rings = np.round(bind[:, 0] / STEP).astype(int)

    def run(W):
        guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))
        dd = np.asarray(ref.structural_centers(edges, cor_l, W, rots, guide,
                                               outer, None))
        A = ref.center_response(W, rots)
        pred = cor_l + np.einsum("nij,nj->ni", A, dd)
        # 转回世界 y
        y = pivot[1] + RR * (pred[:, 1] * bY[1] + pred[:, 2] * bZ[1]
                             + pred[:, 0] * axis[1])
        return y, np.linalg.norm(dd, axis=1)

    print("\n%8s %10s %12s %12s %12s" % ("x", "CoR", "real", "skin", "preview"))
    res = {}
    for spec in ("real", "skin", "preview"):
        W = make(spec)
        y, dn = run(W)
        res[spec] = y
        print("", end="")
        if spec == "preview":
            pass
    # 重新打印成表
    for spec in ("real", "skin", "preview"):
        W = make(spec)
        y, dn = run(W)
        res[spec] = (y, dn)

    print("\n%8s %10s %12s %12s %12s %14s"
          % ("x", "CoR外极点", "real", "skin", "preview", "|delta|均"))
    for rid in sorted(set(rings.tolist())):
        x0 = rid * STEP
        if not (-2.7 <= x0 <= 0.7):
            continue
        m2 = np.where(rings == rid)[0]
        i0 = m2[np.argmin(cor_l[m2, 1])]
        row = "%8.3f %10.5f" % (x0, cor_l[i0, 1])
        for spec in ("real", "skin", "preview"):
            y, dn = res[spec]
            row += " %12.5f" % y[i0]
        print(row)

    print("\n各权重方案的外极点 vs CoR 的 RMS：")
    for spec in ("real", "skin", "preview"):
        y, dn = res[spec]
        dev = []
        for rid in sorted(set(rings.tolist())):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            m2 = np.where(rings == rid)[0]
            i0 = m2[np.argmin(cor_l[m2, 1])]
            dev.append(y[i0] - cor_l[i0, 1])
        dev = np.asarray(dev)
        print("  %-8s RMS=%.5f  均值=%+.5f  |delta|均=%.4f"
              % (spec, math.sqrt((dev ** 2).mean()), dev.mean(),
                 dn.mean()))


if __name__ == "__main__":
    main()
