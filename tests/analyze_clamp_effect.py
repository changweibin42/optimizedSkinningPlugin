# -*- coding: utf-8 -*-
"""用固定顶点 ID 判定：限幅是否影响红圈。

两层指标：
  · x≈3 同一圈的圆度 / 径向
  · 纵向折痕：沿某条纵向经线的有符号转角，以及相邻环的径向连续性

    python tests/analyze_clamp_effect.py
"""
import json
import math
import sys

import numpy as np

HERE = r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin"


def main():
    case = json.load(open(r"D:\tmp\ocd_case3.json"))
    exp = json.load(open(HERE + r"\tests\_clamp_exp.json"))

    bind = np.asarray(case["mesh"]["bindPoints"], float)
    n = len(bind)
    p16 = np.asarray(exp["p16"], float)
    p100 = np.asarray(exp["p100"], float)
    clamped = set(exp["clamped16"])
    moved = set(exp["moved"])
    diffs = np.asarray(exp["diffs"], float)

    child = [b for b in case["bones"] if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(case["hinge"]["axis"], float)
    axis /= np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)
    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = float(np.linalg.norm(rad, axis=1).max())

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

    l16, l100, lb = to_local(p16), to_local(p100), to_local(bind)
    r16 = np.linalg.norm(l16[:, 1:], axis=1)
    r100 = np.linalg.norm(l100[:, 1:], axis=1)

    print("触限 %d  移动 %d  交集 %d" % (len(clamped), len(moved),
                                        len(clamped & moved)))

    # 用 bind 的 x 值分组（环），步长 0.12
    rings = np.round(bind[:, 0] / 0.12).astype(int)
    ring_ids = np.unique(rings)
    print("环数 %d  每环点数 %d" % (len(ring_ids), int((rings == ring_ids[0]).sum())))

    def ring_members(ring):
        return np.where(rings == ring)[0]

    print("\n=== 指标 1: 逐环圆度（限16 vs 限100）===")
    print("%8s %6s %10s %10s %10s %10s %10s"
          % ("x", "n", "限16圆度", "限100圆度", "限16径向", "限100径向", "位移max"))
    for rid in ring_ids:
        x0 = rid * 0.12
        if not (2.4 <= abs(x0) <= 4.2):
            continue
        m = ring_members(rid)
        if len(m) < 8:
            continue
        s16 = np.std(r16[m]) / max(1e-12, r16[m].mean())
        s100 = np.std(r100[m]) / max(1e-12, r100[m].mean())
        print("%8.2f %6d %10.4f %10.4f %10.4f %10.4f %10.4f"
              % (x0, int(len(m)), s16, s100, r16[m].mean(), r100[m].mean(),
                 diffs[m].max()))

    # 纵向折痕：固定"侧面"顶点（按环内序号），沿环序列看径向
    print("\n=== 指标 2: 同一环内序号的纵向径向剖面 ===")
    full_rings = [rid for rid in ring_ids if len(ring_members(rid)) == 80]
    print("完整 80 点环数 = %d" % len(full_rings))
    for side in (0, 20, 40, 60):
        print("\n  环内序号 %d（按 y 排序）:" % side)
        print("  %8s %10s %10s %10s" % ("x", "r16", "r100", "Δ"))
        for rid in full_rings:
            x0 = rid * 0.12
            if not (-4.2 <= x0 <= 4.2):
                continue
            m = ring_members(rid)
            o = np.argsort(bind[m, 1])
            i = m[o[side]]
            print("  %8.2f %10.4f %10.4f %+10.4f"
                  % (x0, r16[i], r100[i], r100[i] - r16[i]))

    # 红圈候选：|x| 在 2.5-4 之间、位移最大的顶点
    print("\n=== 位置 16 vs 100 变化最大的顶点 ===")
    order = np.argsort(-diffs)[:10]
    for i in order:
        print("  v%-6d x=%+7.2f st/R=%+7.2f w1=%.4f  Δ=%.6f  %s"
              % (i, bind[i, 0], lb[i, 0],
                 float(np.load(HERE + r"\tests\_W.npy")[i, 1]),
                 diffs[i], "触限" if i in clamped else ""))


if __name__ == "__main__":
    main()
