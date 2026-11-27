# -*- coding: utf-8 -*-
r"""原型 v3：在折角处做局部圆角（Hermite 过渡），而不是整段位置插值。

为什么 v1/v2 失败：参考实现里 station>0 分支是对**整条曲线**做一次绕原点的
镜像（矩阵 [[-cb,-sb],[-sb,cb]]，det=-1，是反射）。折角点 C 恰好落在镜面
（反射轴）上，所以位置连续、但左右切向差 15.48°。把「未反射」和「已反射」
的位置直接插值，等于让点在 4.5 个单位跨度上来回，必然爆曲率。

v3 正确做法：折角是**局部**的。取 |s| < w 的窗口，两端点 P0=orig(-w)、
P1=orig(+w) 相距很近（2w·sin(θ/2)），用三次 Hermite 连接，端点切向取原曲线
的单侧切向 —— 得到 C1 连续、曲率有界的圆角。窗口外完全沿用原实现。

w 越小，圆角越局部、对 458 点造型的改动越小，但高密度下越可能看不清；
w 越大越平滑，但越改形状。这一版扫 w，给出取舍曲线。

    mayapy tests/fillet_prototype.py tests/_fillet.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/_fillet.json"
HALF = ref.HALF
BEND = 146.1265


class FilletElbow(ref.Elbow):
    """在 station=0 的折角处用 Hermite 圆角替换。"""

    window = 0.0

    def _rail(self, station, offsets, pole):
        w = self.window
        if w <= 0.0 or abs(station) >= w:
            return super()._rail(station, offsets, pole)
        # 端点
        p0 = np.array(super()._rail(-w, offsets, pole), float)
        p1 = np.array(super()._rail(+w, offsets, pole), float)
        # 单侧切向（用解析函数的中心差分，步长远小于窗口）
        h = min(1e-4, w * 1e-3)
        t0 = (np.array(super()._rail(-w + h, offsets, pole), float)
              - np.array(super()._rail(-w - h, offsets, pole)
                         if w - h > 0 else super()._rail(-w, offsets, pole), float))
        t1 = (np.array(super()._rail(w + h, offsets, pole)
                       if w + h < HALF else super()._rail(w, offsets, pole), float)
              - np.array(super()._rail(w - h, offsets, pole), float))
        n0 = np.linalg.norm(t0)
        n1 = np.linalg.norm(t1)
        if n0 < 1e-12 or n1 < 1e-12:
            return super()._rail(station, offsets, pole)
        t0 /= n0
        t1 /= n1
        chord = float(np.linalg.norm(p1 - p0))
        # Hermite：t∈[0,1]
        u = (station + w) / (2.0 * w)
        u = max(0.0, min(1.0, u))
        u2 = u * u
        u3 = u2 * u
        h00 = 2 * u3 - 3 * u2 + 1
        h10 = u3 - 2 * u2 + u
        h01 = -2 * u3 + 3 * u2
        h11 = u3 - u2
        pt = h00 * p0 + h10 * (chord * t0) + h01 * p1 + h11 * (chord * t1)
        return (float(pt[0]), float(pt[1]))


def tube(sides, rings, half):
    return [(-half + row * (2 * half / (rings - 1)),
             math.cos(2 * math.pi * j / sides),
             math.sin(2 * math.pi * j / sides))
            for row in range(rings) for j in range(sides)]


def turning(p, q, r):
    u = np.asarray(q) - np.asarray(p)
    v = np.asarray(r) - np.asarray(q)
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-15 or lv < 1e-15:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0,
        float(u @ v) / (lu * lv)))))


def prof(P, sides, rings, j):
    st = [-HALF + r * (2 * HALF / (rings - 1)) for r in range(rings)]
    idx = [r * sides + j for r in range(rings)]
    return [(st[k], turning(P[idx[k - 1]], P[idx[k]], P[idx[k + 1]]))
            for k in range(1, rings - 1)]


def build(window, sides, rings):
    pose = FilletElbow(BEND, 0.0, 0.0, ref.ZERO, False)
    pose.window = window
    return np.asarray(pose.mesh(tube(sides, rings, HALF)), float), pose


def main():
    print("bend = %.4f°   Hermite 圆角原型（纯 Python，不含插件）" % BEND)
    print("=" * 92)
    base58, _ = build(0.0, 24, 19)
    base80, _ = build(0.0, 80, 201)
    print("原始 Elbow：458 内轨 %.2f°   16080 内轨 %.2f°   16080 外轨 %.2f°"
          % (max(t for _, t in prof(base58, 24, 19, 0)),
             max(t for _, t in prof(base80, 80, 201, 0)),
             max(t for _, t in prof(base80, 80, 201, 40))))

    print("\n扫圆角窗口 w：")
    print("%7s %10s %10s %10s %11s %11s %10s"
          % ("w", "458内轨", "16080内轨", "16080外轨",
             "458最大偏差", "16080偏差", "折角°"))
    rows = []
    for w in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.6, 0.8):
        P58, pose58 = build(w, 24, 19)
        P80, pose80 = build(w, 80, 201)
        t58 = max(t for _, t in prof(P58, 24, 19, 0))
        t80 = max(t for _, t in prof(P80, 80, 201, 0))
        t80o = max(t for _, t in prof(P80, 80, 201, 40))
        dev58 = float(np.linalg.norm(P58 - base58, axis=1).max())
        dev80 = float(np.linalg.norm(P80 - base80, axis=1).max())
        eps = 1e-7
        p0 = np.array(pose80.point((0.0, 1.0, 0.0))[:2])
        pl = np.array(pose80.point((-eps, 1.0, 0.0))[:2])
        pr = np.array(pose80.point((eps, 1.0, 0.0))[:2])
        dl, dr = p0 - pl, pr - p0
        if np.linalg.norm(dl) < 1e-14 or np.linalg.norm(dr) < 1e-14:
            kink = float("nan")
        else:
            u, v = dl / np.linalg.norm(dl), dr / np.linalg.norm(dr)
            kink = 180.0 - math.degrees(math.acos(
                max(-1.0, min(1.0, float(u @ v)))))
        rows.append(dict(v=3, window=w, t458=t58, t16080_in=t80,
                         t16080_out=t80o, kink=kink,
                         dev458=dev58, dev16080=dev80))
        print("%7.2f %10.2f %10.2f %10.2f %11.4f %11.4f %10.3f"
              % (w, t58, t80, t80o, dev58, dev80, kink))

    with open(OUT, "w") as h:
        json.dump(rows, h, default=float)
    print("\nwrote", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
