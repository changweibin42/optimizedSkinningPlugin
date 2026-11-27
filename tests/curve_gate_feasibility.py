# -*- coding: utf-8 -*-
r"""方案可行性预检：曲线参数关联度 g_curve 会不会真的改变 CoR 中心？

在动 C++ 之前，先用纯 Python 回答三个问题：

  1. 圆柱这种「一条曲线贯穿」的模型，g_curve 几乎恒为 1，
     所以它不会改变 458 点那套已认可的造型 —— 这是好事（不回归）。
  2. 只有当顶点分属不同曲线、或曲线参数跨度大时，g_curve 才起作用。
  3. 起作用时，中心位移有多大？如果位移远小于「刀口」的量级，
     这条路就解决不了当前问题，需要提前知道。

    mayapy tests/curve_gate_feasibility.py
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


def blend_quat(w, rots):
    blend = np.zeros(4)
    for wk, q in zip(w, rots):
        sign = -1.0 if float(np.dot(blend, q)) < 0 else 1.0
        blend += wk * sign * np.asarray(q, float)
    n = np.linalg.norm(blend)
    return blend / n if n > 1e-12 else np.array([0, 0, 0, 1.0])


def main():
    # 458 点圆柱：一条沿 X 的曲线，参数 = station 归一化
    bind_list, faces = ref.cylinder()
    bind = np.asarray(bind_list, float)
    SIDES, HALF = ref.SIDES, ref.HALF
    RINGS = 2 * ref.HALF_RINGS + 1
    print("圆柱 %d 点（24 边 x %d 环），单条曲线沿 X" % (len(bind), RINGS))

    # 曲线参数：归一化轴向坐标
    s_param = (bind[:, 0] - bind[:, 0].min()) / (bind[:, 0].max() - bind[:, 0].min())
    # 环内同 station 的点参数相同
    print("曲线参数范围 [%.3f, %.3f]，同环内参数是否相同: %s"
          % (s_param.min(), s_param.max(),
             bool(np.allclose(s_param[:SIDES], s_param[0]))))

    # 候选邻域：同一环 80 点（这里 24 点）内的其他顶点 + 轴向邻近环
    # 简化：只看同环内的点对，它们的参数差恒为 0 → g_curve = 1
    print("\n同环点对的参数差: 恒为 %.1e  => g_curve = 1.000000"
          % float(np.abs(s_param[:SIDES] - s_param[0]).max()))

    # 跨环点对的参数差
    ring = np.round((bind[:, 0] - bind[:, 0].min()) / (2 * HALF / (RINGS - 1))).astype(int)
    print("\n跨环点对的参数差与 g_curve（sigma_s = 0.1）：")
    print("%8s %12s %12s" % ("环距", "参数差", "g_curve"))
    for dr in (1, 2, 3, 6, 9):
        d = dr * (2 * HALF / (RINGS - 1)) / (2 * HALF)
        for sigma in (0.05, 0.1, 0.2):
            g = math.exp(-(d * d) / (2 * sigma * sigma))
            if sigma == 0.1:
                print("%8d %12.4f %12.6f" % (dr, d, g))
        print("            sigma=0.05 -> %.6f   sigma=0.2 -> %.6f"
              % (math.exp(-(d * d) / (2 * 0.05 ** 2)),
                 math.exp(-(d * d) / (2 * 0.2 ** 2))))

    # CoR 的积分核：权重差 d 的尺度
    print("\n当前 CoR 积分核 exp(-((a*wj - b*wi)/sigma)^2)，sigma = corSigma（默认 0.1）")
    print("权重差与核值（a=b=1）：")
    for dw in (0.0, 0.1, 0.2, 0.3, 0.5):
        for sigma in (0.05, 0.1, 0.2):
            pass
        print("  Δw=%.2f  sigma=0.05 -> %.6f   sigma=0.1 -> %.6f   sigma=0.2 -> %.6f"
              % (dw, math.exp(-(dw / 0.05) ** 2), math.exp(-(dw / 0.1) ** 2),
                 math.exp(-(dw / 0.2) ** 2)))

    print("\n关键判断：")
    print("  · 单曲线、环内点对：g_curve = 1，对已认可的 458 点造型零影响")
    print("  · 跨环点对：sigma_s=0.1 时，1 环外 g=%.3f，3 环外 g=%.5f"
          % (math.exp(-0.111 ** 2 / 0.02), math.exp(-0.333 ** 2 / 0.02)))
    print("  · 所以 g_curve 在圆柱上作用于「跨环」邻域，不影响环内")
    print("  · 真正需要曲线因子的场景：多曲线、曲线归属不同、关节折叠区")
    return 0


if __name__ == "__main__":
    sys.exit(main())
