# -*- coding: utf-8 -*-
r"""A_i 的零空间：中心偏移里有一个方向对顶点位置完全没有影响。

推导：p(c) = Σ w_k T_k(c) + R_blend (rest − c)
          = (Σ w_k R_k − R_blend) c + 常数
          = A_i c + 常数
对 c 是**严格线性**的，所以 A_i 的零空间方向对顶点位置精确无影响。

这意味着：求解出的 delta 若含很大的零空间分量，几何上看不见，
但**会挤占限幅**（|delta| 被截到 16），截断时把可观测部分也一起缩了，
于是相邻顶点因为零空间分量不同而被截到不同程度 —— 就是刀口。

本脚本验证：
  1. A_i 的零空间方向是什么
  2. 插件解出的 delta 里，零空间分量占多大
  3. 投影掉零空间后，被限幅的顶点会少多少

    mayapy tests/nullspace_probe.py
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
    print("\n=== 1. A_i 的零空间方向 ===")
    print("%10s %8s %28s %14s"
          % ("station", "w1", "零空间向量", "可见奇异值"))
    nulls = []
    for station in (-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0):
        w1 = ref.skin_weight(station)
        A = center_response([1 - w1, w1], rots)
        U, S, Vt = np.linalg.svd(A)
        n = Vt[-1]
        nulls.append(n)
        print("%10.3f %8.4f %28s %14s"
              % (station, w1,
                 "(%+.4f, %+.4f, %+.4f)" % tuple(n),
                 "[%.4f, %.4f]" % (S[0], S[1])))

    # 零空间方向是否与弯曲轴一致（绕 Z 弯 => 轴为 +Z）
    print("\n与 +Z 轴的夹角：")
    for station, n in zip((-3.0, -2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5, 3.0), nulls):
        c = abs(float(n[2]))
        print("  station %6.2f   |n·Z| = %.6f" % (station, c))

    print("\n=== 2. 投影掉零空间的效果（用参考求解器复算） ===")
    # 用参考实现的圆柱，算 delta，再分解
    bind_list, faces = ref.cylinder()
    bind = np.asarray(bind_list, float)
    SIDES, HALF = ref.SIDES, ref.HALF
    RINGS = 2 * ref.HALF_RINGS + 1
    eff = np.array([ref.preview_influences(p[0], True) for p in bind])
    edges = ref.local_edges(bind_list, faces)
    q = ref.joint_quaternion(BEND, 0, 0)
    rots3 = [ref.joint_quaternion(0, 0, 0), q,
             ref.joint_quaternion(BEND * 0.5, 0.0, 0.0)]

    # original：用 alg0 的输出近似（这里只关心量级）
    cor = np.zeros_like(bind)
    for i in range(len(bind)):
        w = eff[i]
        blend = np.zeros(4)
        for wk, qq in zip(w, rots3):
            sign = -1.0 if float(np.dot(blend, qq)) < 0 else 1.0
            blend += wk * sign * np.asarray(qq, float)
        n = np.linalg.norm(blend)
        blend /= n if n > 1e-12 else 1.0
        Rb = rotation_matrix(blend)
        cor[i] = (sum(wk * rotation_matrix(qq) for wk, qq in zip(w, rots3))) @ bind[i] \
            + Rb @ (bind[i] - bind[i])
    guide = np.asarray(ref.hinge_displacement(bind, cor, eff, BEND, 0.0, 0.0))
    outer = np.asarray(ref.outer_participation(bind, eff, BEND, 0.0, 0.0))
    delta = np.asarray(ref.structural_centers(edges, cor, eff, rots3,
                                              guide, outer, 16.0))
    print("  |delta| max %.3f  mean %.3f" %
          (np.linalg.norm(delta, axis=1).max(),
           np.linalg.norm(delta, axis=1).mean()))

    # 分解
    ratio = []
    for i in range(len(bind)):
        A = center_response(eff[i], rots3)
        U, S, Vt = np.linalg.svd(A)
        n = Vt[-1]
        d = delta[i]
        dn = float(d @ n)
        dr = d - dn * n
        ratio.append((np.linalg.norm(dn), np.linalg.norm(dr),
                      np.linalg.norm(d)))
    ratio = np.asarray(ratio)
    nz = ratio[:, 2] > 1e-9
    print("  零空间分量 / 总长度 的中位数: %.4f"
          % float(np.median(ratio[nz, 0] / ratio[nz, 2])))
    print("  去掉零空间后 |delta| max %.3f（原 %.3f）"
          % (float(ratio[nz, 1].max()), float(ratio[nz, 2].max())))
    n_clamp_before = int((ratio[nz, 2] > 15.99).sum())
    n_clamp_after = int((ratio[nz, 1] > 15.99).sum())
    print("  被限幅顶点: 原 %d -> 去掉零空间后 %d"
          % (n_clamp_before, n_clamp_after))
    return 0


if __name__ == "__main__":
    sys.exit(main())
