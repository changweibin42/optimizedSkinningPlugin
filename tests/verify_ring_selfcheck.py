# -*- coding: utf-8 -*-
r"""修正版自检：同一次求值、同一环，打印 angle / k / 三维极点间距 / 圆度。

自检关系：
  P± = N ± R·g，故 |P+ - P-| = 2Rk；归一化后应为 2R。
  R=1 时：原间距=2k，归一化间距=2。

上一版我逐环只传单点调用 hinge_section，support 退化为单点，
所以 146.13 度第一圈得到 k=1.432427（错），正确值是 1.469886。

    python tests/verify_ring_selfcheck.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

HALF, HALF_RINGS, SIDES = ref.HALF, ref.HALF_RINGS, ref.SIDES
RINGS = 2 * HALF_RINGS + 1
R = 1.0


def main():
    bind, faces = ref.cylinder()
    P = np.asarray(bind, float)
    stations = P[:, 0]
    w = np.array([ref.skin_weight(s) for s in stations])
    W = np.column_stack([1.0 - w, w])
    n = len(P)

    print("R=%.1f  SIDES=%d  RINGS=%d" % (R, SIDES, RINGS))

    for bend in (118.0, 120.0, 146.1265):
        angle = math.radians(abs(bend))
        ext = np.full_like(stations, R)
        inner = ref.hinge_section(stations, ext, R, W, angle)
        outer = ref.hinge_section(stations, -ext, R, W, angle)
        neutral = (inner + outer) * 0.5
        grad = (inner - outer) / (2.0 * R)
        k = np.linalg.norm(grad, axis=1)

        print("\n=== bend=%.4f  angle=%.4f deg ===" % (bend, math.degrees(angle)))
        print("%8s %8s %10s %14s %14s %12s %12s"
              % ("x", "w1", "k", "原极点间距", "归一间距", "原圆度", "归一圆度"))
        for row in range(RINGS):
            i = row * SIDES
            x0 = stations[i]
            if abs(x0) > 2.7:
                continue
            # 三维极点（只用该环自身，g 是该环的梯度）
            p_plus = neutral[i] + R * grad[i]
            p_minus = neutral[i] - R * grad[i]
            d_raw = float(np.linalg.norm(p_plus - p_minus))
            g_unit = grad[i] / max(k[i], 1e-12)
            q_plus = neutral[i] + R * g_unit
            q_minus = neutral[i] - R * g_unit
            d_unit = float(np.linalg.norm(q_plus - q_minus))

            # 该环 24 点的圆度（两种梯度）
            pts = []
            for j in range(SIDES):
                a = 2 * math.pi * j / SIDES
                pts.append((math.cos(a), math.sin(a)))
            pts = np.asarray(pts)
            def ring_round(gg):
                xy = neutral[i][None, :] + pts[:, 0:1] * gg[None, :]
                c = xy.mean(axis=0)
                r = np.linalg.norm(xy - c, axis=1)
                return float(r.std() / r.mean())
            r0 = ring_round(grad[i])
            r1 = ring_round(g_unit)

            print("%8.3f %8.4f %10.6f %14.6f %14.6f %12.6f %12.6f"
                  % (x0, W[i, 1], k[i], d_raw, d_unit, r0, r1))

        # 断言检查
        i = 1 * SIDES  # x=-5.333? 用第一圈 |x|=0.667
        row_first = int(round((HALF - HALF / HALF_RINGS + HALF) / (2 * HALF / (RINGS - 1))))
        # 直接找 x≈+0.667
        row = min(range(RINGS), key=lambda r: abs(stations[r * SIDES] - HALF / HALF_RINGS))
        i = row * SIDES
        print("  自检 @x=%+.3f: 2Rk=%.6f  实测极点间距=%.6f  归一间距=%.6f"
              % (stations[i], 2 * R * k[i],
                 float(np.linalg.norm((neutral[i] + R * grad[i]) - (neutral[i] - R * grad[i]))),
                 float(np.linalg.norm((neutral[i] + R * grad[i] / k[i])
                                      - (neutral[i] - R * grad[i] / k[i])))))


if __name__ == "__main__":
    main()
