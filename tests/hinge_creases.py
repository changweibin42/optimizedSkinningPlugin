# -*- coding: utf-8 -*-
r"""扫 hinge_section 的整条内轨，找出所有折角（而不仅是 station=0 那一个）。

结论会决定改造范围：如果折角有多个，只圆 station=0 是不够的。

    mayapy tests/hinge_creases.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

BEND = 146.1265


def main():
    angle = math.radians(BEND)
    # 与插件走的 support 一致的量级：两条腿网格上 support ≈ 2.5~3
    for support in (2.5, 2.75, 3.0):
        stations = np.linspace(-5.0, 5.0, 2001)
        w1 = np.array([ref.skin_weight(s) for s in stations])
        w2 = np.column_stack([1.0 - w1, w1])
        radius = 1.0
        ext = np.full_like(stations, radius)
        inner = ref.hinge_section(stations, ext, radius, w2, angle)
        outer = ref.hinge_section(stations, -ext, radius, w2, angle)
        print("\nsupport = %.3f   弯角 %.2f°" % (support, BEND))
        for name, prof in (("inner", inner), ("outer", outer)):
            d = np.diff(prof, axis=0)
            n = np.linalg.norm(d, axis=1)
            keep = n > 1e-14
            d, n = d[keep], n[keep]
            u = d / n[:, None]
            cosang = np.clip((u[1:] * u[:-1]).sum(axis=1), -1, 1)
            turn = np.degrees(np.arccos(cosang))
            seg_mid = (stations[1:] + stations[:-1]) * 0.5
            mid = seg_mid[keep]
            mid = (mid[1:] + mid[:-1]) * 0.5
            # 找局部尖峰
            peaks = []
            for k in range(1, len(turn) - 1):
                if turn[k] > 5.0 and turn[k] >= turn[k - 1] and turn[k] >= turn[k + 1]:
                    peaks.append((mid[k], turn[k]))
            print("  %s: 最大 %.2f°   尖峰(>5°): %s"
                  % (name, turn.max(),
                     ", ".join("%.2f@%.2f" % (t, s) for s, t in peaks[:8])
                     or "无"))
    return 0


if __name__ == "__main__":
    sys.exit(main())
