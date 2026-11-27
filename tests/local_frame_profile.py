# -*- coding: utf-8 -*-
"""在正确的铰链局部系里测截面半径。

cor/final 里 x>0 的顶点绕 Z 转了 146°，截面法线不再是 X。
必须把顶点变换回"绑定铰链局部系"再取径向，否则 hypot(y,z) 毫无意义。

局部系（与插件 toLocalPoint 一致）：
    offset = p - childPivot
    x_l = dot(offset, axis) / R
    y_l = dot(offset, basisY) / R
    z_l = dot(offset, basisZ) / R
其中 axis 是 parent->child 的绑定轴，basisY 由父骨绑定旋转导出。
父骨 joint1 绑定时是单位旋转，所以 basisY = +Y，basisZ = +Z。

    python tests/local_frame_profile.py D:/tmp/ocd_case3.json
"""
import json
import math
import sys

import numpy as np


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
    d = json.load(open(path))

    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    fin = np.asarray(d["points"]["final"], float)
    n = len(bind)

    pivot = np.asarray(
        [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]["bindPosition"], float)
    axis = np.asarray(d["hinge"]["axis"], float)
    axis /= np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)

    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = np.linalg.norm(rad, axis=1).max()
    print("childPivot=%s  axis=%s  R=%.6f" % (pivot, np.round(axis, 3), R))
    print("bind station 范围 %.3f .. %.3f  (归一化 %.2f .. %.2f)"
          % (st.min(), st.max(), st.min() / R, st.max() / R))

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

    lc = to_local(cor)
    lf = to_local(fin)
    lb = to_local(bind)

    # 验证：bind 的径向必须恒为 1
    rb = np.linalg.norm(lb[:, 1:], axis=1)
    print("bind 径向: min=%.6f max=%.6f  (应恒为 1)" % (rb.min(), rb.max()))

    rc = np.linalg.norm(lc[:, 1:], axis=1)
    rf = np.linalg.norm(lf[:, 1:], axis=1)

    W = np.load("tests/_W.npy")
    w1 = W[:, 1]

    print("\n== 逐 station 切片的局部径向（归一化）==")
    print("%8s %8s %10s %10s %10s %10s"
          % ("st/R", "n", "r_cor", "r_final", "std_r", "w1均"))
    for x0 in np.arange(-4.5, 4.51, 0.25):
        m = np.abs(lb[:, 0] - x0) < 0.09
        if m.sum() < 8:
            continue
        s = np.std(rf[m]) / max(1e-12, rf[m].mean())
        print("%8.2f %8d %10.4f %10.4f %10.4f %10.4f"
              % (x0, int(m.sum()), rc[m].mean(), rf[m].mean(), s, w1[m].mean()))

    print("\n== 圆度 std/mean（越小越圆）==")
    for x0 in (-3.0, -2.0, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 3.0):
        m = np.abs(lb[:, 0] - x0) < 0.09
        if m.sum() < 8:
            continue
        sc = np.std(rc[m]) / max(1e-12, rc[m].mean())
        sf = np.std(rf[m]) / max(1e-12, rf[m].mean())
        print("  st/R=%+5.1f  n=%5d  CoR圆度=%.4f  final圆度=%.4f  %s"
              % (x0, int(m.sum()), sc, sf, "变差" if sf > sc * 1.15 else ""))


if __name__ == "__main__":
    main()
