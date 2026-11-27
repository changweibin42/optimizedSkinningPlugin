# -*- coding: utf-8 -*-
r"""折角归因：24x19 与 80x201 是不是同一条形状，以及折角有多硬。

做法：把 24x19 的每个顶点，算它到 80x201 折线（同一条经线）的距离。
若全部 ≈ 0，则两条折线共线，形状完全相同，差别只在采样密度。

再算 station=0 处的真实切向夹角（取极小 epsilon 的单侧导数），
证明折角出自解析模型本身，不是移植误差。

用户的图1（458 点，看着圆滑）与图2（16080 点，肘部刀口）正是
这份数据的两种采样。

    python tests/crease_shape_identity.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

BEND = 146.1265


def tube(sides, rings, half):
    return [(-half + row * (2 * half / (rings - 1)),
             math.cos(2 * math.pi * j / sides),
             math.sin(2 * math.pi * j / sides))
            for row in range(rings) for j in range(sides)]


def meridian_xy(pose, sides, rings, half, j):
    grid = tube(sides, rings, half)
    out = np.asarray(pose.mesh(grid), float)
    pts = np.asarray([out[r * sides + j] for r in range(rings)])
    st = np.asarray([-half + r * (2 * half / (rings - 1)) for r in range(rings)])
    return pts, st


def point_to_polyline(p, poly):
    """点到折线的最短距离（2D）。"""
    best = 1e9
    for a, b in zip(poly[:-1], poly[1:]):
        ab = b - a
        L2 = float(ab @ ab)
        if L2 < 1e-18:
            d = np.linalg.norm(p - a)
        else:
            t = max(0.0, min(1.0, float((p - a) @ ab) / L2))
            d = np.linalg.norm(p - (a + t * ab))
        best = min(best, d)
    return best


def tangent_angle(st_axis, fn, station=0.0, eps=1e-6):
    """station 处左右单侧导数，返回两切向夹角（度）。"""
    left = (np.asarray(fn(station), float) - np.asarray(fn(station - eps), float))
    right = (np.asarray(fn(station + eps), float) - np.asarray(fn(station), float))
    lu, lv = np.linalg.norm(left), np.linalg.norm(right)
    if lu < 1e-15 or lv < 1e-15:
        return None
    u, v = left / lu, right / lv
    return math.degrees(math.acos(max(-1.0, min(1.0, float(u @ v)))))


def main():
    pose = ref.Elbow(BEND, 0.0, 0.0, ref.ZERO, False)
    print("bend = %.4f°   Elbow 解析模型" % BEND)
    print("=" * 78)

    for j_tag, pick in (("y=+1 (INNER 轨侧)", lambda s: 0),
                        ("y=-1 (OUTER 轨侧)", lambda s: s // 2)):
        print("\n--- %s ---" % j_tag)
        coarse, st_c = meridian_xy(pose, 24, 19, ref.HALF, pick(24))
        dense, st_d = meridian_xy(pose, 80, 201, ref.HALF, pick(80))

        # 只比 (x, y) 平面内的形状（z=0 附近）
        poly = dense[:, :2]
        worst = 0.0
        for p in coarse[:, :2]:
            worst = max(worst, point_to_polyline(p, poly))
        print("  24x19 顶点到 80x201 折线的最大距离: %.3e" % worst)

        # 反方向也测：80x201 到 24x19 折线（24x19 是折线，会切掉尖角）
        poly_c = coarse[:, :2]
        worst2 = 0.0
        for p in dense[:, :2]:
            worst2 = max(worst2, point_to_polyline(p, poly_c))
        print("  80x201 顶点到 24x19 折线的最大距离: %.3e"
              "   ← 这就是被切掉的尖角高度" % worst2)

        # station=0 的真实切向夹角
        def fn(s, jidx=pick(80)):
            a = 2 * math.pi * jidx / 80
            return pose.point((s, math.cos(a), math.sin(a)))[:2]

        ang = tangent_angle(None, fn, 0.0, 1e-7)
        print("  station=0 处左右切向夹角（eps=1e-7）: %s"
              % ("%.2f°" % ang if ang is not None else "不可分"))

        # 采样能分辨出的三点转角
        for sides2, rings2 in ((24, 19), (80, 201), (160, 401)):
            pts2, st2 = meridian_xy(pose, sides2, rings2, ref.HALF, pick(sides2))
            k = int(np.argmin(np.abs(st2)))
            if 0 < k < len(pts2) - 1:
                u = pts2[k] - pts2[k - 1]
                v = pts2[k + 1] - pts2[k]
                lu, lv = np.linalg.norm(u), np.linalg.norm(v)
                t = math.degrees(math.acos(max(-1.0, min(1.0,
                    float(u @ v) / (lu * lv)))))
                print("    %3d边 x %3d环（环距 %.4f）三点转角 %.2f°"
                      % (sides2, rings2, 2 * ref.HALF / (rings2 - 1), t))

    # hinge_section 是否也有同样的尖角
    print("\n--- hinge_section（alg5 默认 guide 用的那套）---")
    n = 9
    st = np.linspace(-0.3, 0.3, n)
    w2 = np.column_stack([1.0 - np.array([ref.skin_weight(s) for s in st]),
                          np.array([ref.skin_weight(s) for s in st])])
    rad = 1.0
    ext = np.full_like(st, rad)
    inner = ref.hinge_section(st, ext, rad, w2, math.radians(BEND))
    outer = ref.hinge_section(st, -ext, rad, w2, math.radians(BEND))
    for name, prof in (("inner", inner), ("outer", outer)):
        k = n // 2
        u = prof[k] - prof[k - 1]
        v = prof[k + 1] - prof[k]
        lu, lv = np.linalg.norm(u), np.linalg.norm(v)
        if lu < 1e-15 or lv < 1e-15:
            continue
        t = math.degrees(math.acos(max(-1.0, min(1.0,
            float(u @ v) / (lu * lv)))))
        print("  %s: 环距 %.4f 时三点转角 %.2f°" % (name, st[1] - st[0], t))
    return 0


if __name__ == "__main__":
    sys.exit(main())
