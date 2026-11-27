# -*- coding: utf-8 -*-
r"""解析检查 hinge_section 在 station→0± 两侧的连续性。

hinge_section(station, depth, radius, weights, angle):
    distance = |station|
    u = min(distance/support, 1)
    ...
    if station <= 0: return (-outward, height)
    else:            return (cos(a)*outward - sin(a)*height,
                             sin(a)*outward + cos(a)*height)

在 station=0 两侧：outward 和 height 是 distance=|station| 的偶函数 → 相同值。
但映射不同：
    x(-0) = -outward0
    x(+0) = cos(a)*outward0 - sin(a)*height0
两者之差 = outward0*(1+cos a) - sin(a)*height0
除非 outward0*(1+cos a) == sin(a)*height0，否则位置不连续。

同时检查导数：d/d(station)。

    python tests/verify_hinge_continuity.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref


def main():
    print("hinge_section 在 station=0 两侧的连续性与导数")
    print("=" * 78)

    for bend in (90.0, 118.0, 120.0, 146.1265):
        angle = math.radians(abs(bend))
        print("\n=== bend=%.4f  (angle=%.4f rad) ===" % (bend, angle))

        # 参考实现里 support 由整条肢体推断；这里用一个典型值
        for radius, support in ((1.0, 4.06), (1.0, 3.0), (1.0, 2.6)):
            print("\n  radius=%.2f  support=%.3f" % (radius, support))
            print("  %10s %14s %14s %14s %14s"
                  % ("depth", "x(-0+)", "x(+0+)", "位置跳变", "跳变/radius"))
            for depth in (-1.0, -0.5, 0.0, 0.5, 1.0):
                eps = 1e-9
                # station 从负侧趋近 0 和从正侧
                left = ref.hinge_section(np.array([-eps]), np.array([depth]),
                                         radius, np.array([[0.0, 0.0]]), angle)
                right = ref.hinge_section(np.array([+eps]), np.array([depth]),
                                          radius, np.array([[0.0, 0.0]]), angle)
                # 也取 station 精确 0
                zero = ref.hinge_section(np.array([0.0]), np.array([depth]),
                                         radius, np.array([[0.0, 0.0]]), angle)
                jump = math.hypot(left[0][0] - right[0][0],
                                  left[0][1] - right[0][1])
                print("  %10.2f %14.6f %14.6f %14.6e %14.6e"
                      % (depth, left[0][0], right[0][0], jump, jump / radius))

        # 解析式验证
        print("\n  解析验证：跳变 = |outward0*(1+cos a) - sin(a)*height0|")
        # 在 station=0, u=0
        for depth in (-1.0, 0.0, 1.0):
            a = angle
            outward0 = 0.0
            # 从公式：distance=0 → u=0
            # shift = max(depth,0)*tan(a/2) / sqrt(1+...)
            raw = max(depth, 0.0) * math.tan(a / 2)
            limit = 0.95 * min(support, 2.6)
            shift = raw / math.sqrt(1 + (raw / limit) ** 2) if limit > 1e-12 else 0.0
            release = 0.0  # u=0 → release=0
            inner_travel = shift * 1.0
            outward0 = 0.0 + inner_travel
            height0 = depth
            if a > 1e-8:
                height0 += (shift / math.tan(a / 2) - max(depth, 0.0)) * 1.0 * 1.0
            # depth<0 走 arc 分支
            if depth < 0:
                outer_radius = -depth
                fillet = outer_radius * (1 + 0.45 * math.cos(a / 2))
                tangent_st = 0.45 * outer_radius * math.sin(a / 2)
                arc_len = fillet * a / 2
                extra = arc_len - tangent_st
                side = 1.0
                gather = 0.85 * math.sin(a / 2) * side
                opening = 0.0
                material = 0.0 * (1 - gather) + opening
                join = 0.0
                progress = material + extra * join
                remaining = max(arc_len - progress, 0)
                phi = remaining / fillet if fillet > 1e-12 else 0
                outward0 = tangent_st - fillet * math.sin(phi)
                height0 = depth + 2 * fillet * math.sin(phi / 2) ** 2
            pred = abs(outward0 * (1 + math.cos(a)) - math.sin(a) * height0)
            print("    depth=%+.1f  outward0=%.6f height0=%.6f  预测跳变=%.6f"
                  % (depth, outward0, height0, pred))

        # 导数检查：d x/d station 在 0+ 与 0-
        print("\n  导数（d x/d station）：")
        eps = 1e-7
        print("  %10s %16s %16s %16s"
              % ("depth", "左导数", "右导数", "比值"))
        for depth in (-1.0, 0.0, 1.0):
            def f(s):
                return ref.hinge_section(np.array([s]), np.array([depth]),
                                         radius, np.array([[0.0, 0.0]]), angle)[0][0]
            dl = (f(-eps) - f(-2 * eps)) / eps
            dr = (f(2 * eps) - f(eps)) / eps
            ratio = dr / dl if abs(dl) > 1e-12 else float("inf")
            print("  %10.2f %16.6f %16.6f %16.6f" % (depth, dl, dr, ratio))


if __name__ == "__main__":
    main()
