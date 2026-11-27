# -*- coding: utf-8 -*-
r"""修正版：gradient 必须传整条肢体的数组，support 是全局量。

上一版逐环只传单点时，hinge_section 内部
    mixed / reach / support
全部退化，所以第一圈 k 不等于整条肢体一次算出的值。

本脚本：
  A) 全数组一次算 gradient，复核 0..150 度的 k 表
  B) 验证 k -> 圆度 的解析关系（24 点椭圆）
  C) 单变量实验：把梯度归一化（单位厚度方向搬运绑定 depth）
     比圆度 + 内外极点位移 + 纵向经线转角

    python tests/verify_gradient_normalized.py
"""
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

HALF, HALF_RINGS, SIDES = ref.HALF, ref.HALF_RINGS, ref.SIDES
RINGS = 2 * HALF_RINGS + 1
RADIUS = 1.0


def build(weights3=False):
    bind, faces = ref.cylinder()
    P = np.asarray(bind, float)
    stations = P[:, 0]
    w = np.array([ref.skin_weight(s) for s in stations])
    if weights3:
        cols = []
        for wi in w:
            cols.append(ref.rig_weights(wi, True))
        W = np.asarray(cols, float)
    else:
        W = np.column_stack([1.0 - w, w])
    return P, faces, stations, W


def gradient_full(stations, W, bend, radius=RADIUS):
    """整条肢体一次算，返回 (neutral, gradient)，形状 (n,2)。"""
    angle = math.radians(abs(bend))
    ext = np.full_like(stations, radius)
    inner = ref.hinge_section(stations, ext, radius, W, angle)
    outer = ref.hinge_section(stations, -ext, radius, W, angle)
    neutral = (inner + outer) * 0.5
    grad = (inner - outer) / (2.0 * radius)
    return neutral, grad


def ellipse_roundness(k, n=SIDES):
    """24 点椭圆 (k*cos a, sin a) 到质心的距离 std/mean。"""
    a = 2 * math.pi * np.arange(n) / n
    r = np.hypot(k * np.cos(a), np.sin(a))
    return float(r.std() / r.mean())


def ring_of(x0):
    return int(round((x0 + HALF) / (2 * HALF / (RINGS - 1))))


def main():
    P, faces, stations, W = build(False)
    n = len(P)

    # ---------- A) 角度扫描，复核 k 表 ----------
    print("=== A) 整条肢体一次算出的 k = |gradient| ===")
    print("%9s %12s %14s %14s %12s"
          % ("bend", "中线k", "首圈k(|x|=.667)", "全部环max k", "max环圆度"))
    angles = [0, 30, 60, 90, 118, 120, 135, 146.13, 150]
    for bend in angles:
        neutral, grad = gradient_full(stations, W, bend)
        mag = np.linalg.norm(grad, axis=1)
        mid = ring_of(0.0)
        first = ring_of(HALF * 1.0 / HALF_RINGS)  # x=+0.667
        imax = int(np.argmax(mag))
        # 该环的 k（环内 24 点 k 相同吗？检查）
        ring_idx = [imax // SIDES * SIDES + j for j in range(SIDES)]
        kring = mag[ring_idx]
        print("%9.2f %12.6f %14.6f %14.6f %12.6f"
              % (bend, mag[mid], mag[first], mag.max(),
                 ellipse_roundness(float(kring.mean()))))

    # ---------- B) 你给的校正：k=1.436166 -> 0.1252? ----------
    print("\n=== B) k -> 圆度 解析关系（24 点椭圆）===")
    for k in (1.432427, 1.436166, 1.469890, 1.470, 1.49):
        print("  k=%.6f  圆度=%.6f" % (k, ellipse_roundness(k)))
    print("  你的表: k=1.436166 -> 0.1252, k=1.46989 -> 0.1329")

    # 实测 146.13 度那一环的 k 与圆度
    neutral, grad = gradient_full(stations, W, 146.13)
    mag = np.linalg.norm(grad, axis=1)
    print("\n  146.13 度逐环 k：")
    for row in range(RINGS):
        x0 = -HALF + row * (2 * HALF / (RINGS - 1))
        i = row * SIDES
        if abs(x0) <= 2.7:
            print("    x=%+7.3f  w1=%.4f  k=%.6f  圆度=%.6f"
                  % (x0, W[i, 1], mag[i], ellipse_roundness(mag[i])))

    # ---------- C) 单变量实验：梯度归一化 ----------
    print("\n=== C) 梯度归一化（单位厚度方向）===")

    def guide_targets(bend, unit):
        angle = math.radians(abs(bend))
        y = P[:, 1].copy()
        z = P[:, 2].copy()
        depth = y
        lateral = z
        ext = np.full_like(stations, RADIUS)
        inner = ref.hinge_section(stations, ext, RADIUS, W, angle)
        outer = ref.hinge_section(stations, -ext, RADIUS, W, angle)
        neutral = (inner + outer) * 0.5
        grad = (inner - outer) / (2.0 * RADIUS)
        if unit:
            m = np.linalg.norm(grad, axis=1, keepdims=True)
            grad = grad / np.maximum(m, 1e-12)
        x = neutral[:, 0] + depth * grad[:, 0]
        h = neutral[:, 1] + depth * grad[:, 1]
        return np.column_stack([x, h, lateral]), neutral, grad

    for bend in (118.0, 120.0, 146.13):
        T0, N0, G0 = guide_targets(bend, False)
        T1, N1, G1 = guide_targets(bend, True)

        print("\n  --- bend=%.2f ---" % bend)

        def ring_round(T, row):
            Q = T[row * SIDES:(row + 1) * SIDES]
            c = Q.mean(axis=0)
            r = np.linalg.norm(Q - c, axis=1)
            return float(r.std() / r.mean()), float(r.mean())

        print("  %8s %10s %14s %14s %12s"
              % ("x", "w1", "原圆度", "归一化圆度", "改善"))
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            if abs(x0) > 2.7:
                continue
            r0, m0 = ring_round(T0, row)
            r1, m1 = ring_round(T1, row)
            print("  %8.3f %10.4f %14.6f %14.6f %+12.6f"
                  % (x0, W[row * SIDES, 1], r0, r1, r1 - r0))

        # 内外极点位移：a=0 是 depth=+1（内侧/肘窝一侧），a=pi 是 depth=-1
        j_in = 0
        j_out = SIDES // 2
        print("\n  极点位移（归一化 vs 原，沿 h 方向）：")
        print("  %8s %14s %14s %12s"
              % ("x", "内侧点Δh", "外侧点Δh", "厚度变化"))
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            if abs(x0) > 2.7:
                continue
            i_in = row * SIDES + j_in
            i_out = row * SIDES + j_out
            dh_in = T1[i_in, 1] - T0[i_in, 1]
            dh_out = T1[i_out, 1] - T0[i_out, 1]
            thick0 = T0[i_in, 1] - T0[i_out, 1]
            thick1 = T1[i_in, 1] - T1[i_out, 1]
            print("  %8.3f %14.5f %14.5f %+12.5f"
                  % (x0, dh_in, dh_out, thick1 - thick0))

        # 纵向经线：a=0（内侧经线）上的径向距离与相邻环转角
        print("\n  内侧经线 (j=0) 纵向轮廓：")
        print("  %8s %12s %12s %12s"
              % ("x", "原|N->T|", "归一|N->T|", "差值"))
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            if abs(x0) > 2.7:
                continue
            i = row * SIDES
            print("  %8.3f %12.5f %12.5f %+12.5f"
                  % (x0, T0[i, 1], T1[i, 1], T1[i, 1] - T0[i, 1]))


if __name__ == "__main__":
    main()
