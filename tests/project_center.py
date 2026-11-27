# -*- coding: utf-8 -*-
r"""验证：把 delta 投影到 A_i 的可观测子空间，顶点位置精确不变。

这是「只优化 CoR 中心」这条路的核心前提：
沿 A_i 零空间（弯曲轴）的中心偏移对顶点位置**没有任何影响**，
所以投影掉它不改变几何，只是把限幅预算还给可观测方向。

    mayapy tests/project_center.py
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


def center_response(w, rots):
    return sum(wk * rotation_matrix(q) for wk, q in zip(w, rots)) \
        - rotation_matrix(blend_quat(w, rots))


def vertex_transform(center, w, rots):
    """参考实现的 vertexTransform：返回 (rot, trans)，p' = rot·p + trans。"""
    rot = rotation_matrix(blend_quat(w, rots))
    posed = sum(wk * (rotation_matrix(q) @ center) for wk, q in zip(w, rots))
    return rot, posed - rot @ center


def main():
    rots = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
    print("bend = %.4f°" % BEND)

    # 取几个代表顶点，验证投影前后顶点位置一致
    print("\n投影前/后，同一 rest 点的输出位置差：")
    print("%10s %8s %14s %14s" % ("station", "w1", "投影前|Δ|", "投影后|Δ|"))
    rng = np.random.default_rng(0)
    worst = 0.0
    for station in (-2.5, -2.0, -1.0, 0.0, 1.0, 2.0, 2.5):
        w1 = ref.skin_weight(station)
        w = [1.0 - w1, w1]
        A = center_response(w, rots)
        # 造一个含大零空间分量的 delta
        n = np.linalg.svd(A)[2][-1]
        d0 = A @ np.array([0.3, -0.2, 0.1])      # 可观测部分
        d = d0 + 40.0 * n                        # 加大量零空间分量
        rest = np.array([station, 1.0, 0.0])

        def eval_at(center):
            rot, trans = vertex_transform(center, w, rots)
            return rot @ rest + trans

        base = eval_at(np.zeros(3))
        before = np.linalg.norm(eval_at(d) - base)
        # 投影
        ata = A.T @ A
        scale = np.trace(ata)
        P = ata @ np.linalg.inv(ata + 1e-9 * scale * np.eye(3))
        dp = P @ d
        after = np.linalg.norm(eval_at(dp) - base)
        worst = max(worst, abs(before - after))
        print("%10.2f %8.4f %14.6e %14.6e" % (station, w1, before, after))

    print("\n最大偏差: %.3e  （应当只剩 ridge 的 1e-9 量级影响）" % worst)

    # 零空间分量对几何的影响应当恒为 0
    print("\n零空间向量本身对顶点位置的影响：")
    for station in (-2.0, 0.0, 2.0):
        w1 = ref.skin_weight(station)
        w = [1.0 - w1, w1]
        A = center_response(w, rots)
        n = np.linalg.svd(A)[2][-1]
        rest = np.array([station, 1.0, 0.0])
        rot, trans = vertex_transform(np.zeros(3), w, rots)
        base = rot @ rest + trans
        moved = np.linalg.norm((rot @ rest + trans) if False else
                               (vertex_transform(1e3 * n, w, rots)[0] @ rest
                                + vertex_transform(1e3 * n, w, rots)[1]) - base)
        print("  station %6.2f  |n| = 1e3 时位移 %.3e" % (station, moved))
    return 0


if __name__ == "__main__":
    sys.exit(main())
