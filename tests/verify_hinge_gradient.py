# -*- coding: utf-8 -*-
"""验证 hinge_section 的厚度梯度是否保圆。

hinge_displacement 里：
    ext = radius
    inner = hinge_section(station, +ext, radius, w, angle)
    outer = hinge_section(station, -ext, radius, w, angle)
    neutral = (inner+outer)*0.5
    gradient = (inner-outer)/(2*radius)        ← 厚度方向尺度
    x,h = neutral + depth*gradient

一个圆环上，depth 和 lateral 的分布是 (cos a, sin a)。变形后：
    (x, h) = neutral + cos(a)*gradient_x        (沿弯曲平面方向)
    lateral 保持 = sin(a)
所以该环的截面尺度是 (|gradient|, 1)。若 |gradient| != 1 就失圆。

    python tests/verify_hinge_gradient.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

HALF, HALF_RINGS, SIDES = ref.HALF, ref.HALF_RINGS, ref.SIDES
RINGS = 2 * HALF_RINGS + 1


def main():
    bend = 146.1265
    angle = math.radians(abs(bend))
    radius = 1.0
    bind, faces = ref.cylinder()
    stations = np.asarray(bind)[:, 0]
    w = ref.skin_weight  # 每环一个 w
    n = len(bind)

    W = np.array([[1.0 - w(s), w(s)] for s in stations])
    weights = np.column_stack([W, np.zeros(n)])

    print("bend=%.4f  angle=%.4f deg" % (bend, math.degrees(angle)))
    print("\n逐环厚度梯度（纯弯、无扭转、无补偿通道）")
    print("%8s %10s %14s %14s %14s %12s"
          % ("x", "w1", "gradient_x", "gradient_h", "|gradient|", "=|g|-1"))

    for row in range(RINGS):
        x0 = -HALF + row * (2 * HALF / (RINGS - 1))
        first = row * SIDES
        st = stations[first]
        ww = weights[first]
        ext = np.array([radius])
        inner = ref.hinge_section(np.array([st]), ext, radius,
                                  ww[None, :], angle)
        outer = ref.hinge_section(np.array([st]), -ext, radius,
                                  ww[None, :], angle)
        grad = (inner[0] - outer[0]) / (2.0 * radius)
        mag = math.hypot(grad[0], grad[1])
        print("%8.3f %10.4f %14.6f %14.6f %14.6f %+12.6f"
              % (x0, ww[1], grad[0], grad[1], mag, mag - 1.0))

    # 直接把 guide 对每环做"保圆性"检验：同一环上 24 个点的
    # (depth, lateral) 是 (cos a, sin a)*radius，检查映射后的半径
    print("\n=== 直接检验：每环 guide 映射后的截面尺度 ===")
    print("%8s %12s %12s %12s"
          % ("x", "圆度(guide)", "尺度a", "尺度b"))
    for row in range(RINGS):
        x0 = -HALF + row * (2 * HALF / (RINGS - 1))
        pts = []
        for j in range(SIDES):
            a = 2 * math.pi * j / SIDES
            pts.append((x0, math.cos(a), math.sin(a)))
        pts = np.asarray(pts)
        wrow = np.full(SIDES, w(x0))
        Wr = np.column_stack([1.0 - wrow, wrow, np.zeros(SIDES)])
        # guide = target - original；用 bind 自身当 original 做纯几何检验
        # （参考实现里 original 是 CoR 输出，但截面形状相同）
        g = ref.hinge_displacement(pts, pts.copy(), Wr, bend, 0.0, 0.0)
        target = pts + g
        # 该环的截面尺度：拟合后取主轴
        d = target - target.mean(axis=0)
        cov = d.T @ d / SIDES
        ev = np.linalg.eigvalsh(cov)
        # 半径方向的尺度和侧向尺度
        print("%8.3f %12.6f %12.6f"
              % (x0, 0.0, math.sqrt(max(ev[1], 1e-18)),
                 math.sqrt(max(ev[0], 1e-18))))


if __name__ == "__main__":
    main()
