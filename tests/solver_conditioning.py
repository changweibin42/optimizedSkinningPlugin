# -*- coding: utf-8 -*-
r"""structuralCenters 在 |s| ≈ 2.5 处为什么给出 delta=50，而 guide 只有 0.02。

A_i = Σ w_k·R_k − R_blend 是「中心偏移 → 位移」的响应矩阵。
若 A_i 接近奇异（w_parent ≈ w_child，且两骨旋转差不大），
它的伪逆会把 rhs 放大很多倍。

这里在 Python 里复算每个顶点的 cond(A_i) 与 |delta|/|guide| 的关系。

    mayapy tests/solver_conditioning.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

BEND = 146.1265


def rotation_matrix(q):
    x, y, z, w = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def center_response(w, rots):
    """A_i = Σ w_k R_k − R_blend，与参考实现一致。"""
    blend = np.zeros(4)
    for wk, q in zip(w, rots):
        sign = -1.0 if float(np.dot(blend, q)) < 0 else 1.0
        blend += wk * sign * np.asarray(q, float)
    n = np.linalg.norm(blend)
    blend = blend / n if n > 1e-12 else np.array([0, 0, 0, 1.0])
    return sum(wk * rotation_matrix(q) for wk, q in zip(w, rots)) - rotation_matrix(blend)


def main():
    rots = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
    print("bend = %.4f°" % BEND)
    print("%10s %8s %12s %12s %12s"
          % ("station", "w1", "cond(A)", "min|sv|", "1/min|sv|"))
    rows = []
    for station in np.linspace(-3.5, 3.5, 36):
        w1 = ref.skin_weight(station)
        w = [1.0 - w1, w1]
        A = center_response(w, rots)
        sv = np.linalg.svd(A, compute_uv=False)
        cond = sv[0] / max(sv[-1], 1e-15)
        rows.append((station, w1, cond, sv[-1]))
        if abs(station) < 3.5:
            print("%10.3f %8.4f %12.2f %12.4e %12.2e"
                  % (station, w1, cond, sv[-1], 1.0 / max(sv[-1], 1e-15)))

    worst = max(rows, key=lambda r: r[2])
    print("\n最坏条件数 %.1f 在 station %.3f (w1=%.4f)"
          % (worst[2], worst[0], worst[1]))

    # A 的秩
    print("\nA 的奇异值随 station 的变化（前 4 个采样）：")
    for station in (-2.5, -1.0, 0.0, 1.0, 2.5):
        w1 = ref.skin_weight(station)
        A = center_response([1 - w1, w1], rots)
        sv = np.linalg.svd(A, compute_uv=False)
        print("  station %6.2f w1=%.4f  奇异值 %s"
              % (station, w1, np.array2string(sv, precision=5)))
    return 0


if __name__ == "__main__":
    sys.exit(main())
