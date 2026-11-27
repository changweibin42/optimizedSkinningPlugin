# -*- coding: utf-8 -*-
"""验证 final - original = A_i * delta 在真实夹具上是否成立。

A_i = sum_k w_ik R_k - R_blend。
本夹具所有旋转都绕世界 Z 轴，于是 A_i 第三行必为 0，
因此 (final - original).z 必须恒为 0 —— 这是不需要 delta 就能证伪的预测。

另外反解 delta 并与 |final-original| 对比，量化"中心响应"的放大倍数。

    python tests/verify_center_response_identity.py D:/tmp/ocd_case3.json
"""
import json
import math
import sys

import numpy as np


def mat3_from_maya(m):
    """Maya MMatrix 是行主序，前 3 行是 3x3 部分。"""
    return np.array([[m[0], m[1], m[2]],
                     [m[4], m[5], m[6]],
                     [m[8], m[9], m[10]]], float)


def quat_from_mat3(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w], float)


def quat_to_mat3(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    if n < 1e-18:
        return np.eye(3)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], float)


def blend_rotation(weights, quats):
    """复刻 cor_paper::vertexTransform 的同半球累加 + 归一化。"""
    s = np.zeros(4, float)
    for w, q in zip(weights, quats):
        if w == 0.0:
            continue
        sign = -1.0 if float(np.dot(s, q)) < 0.0 else 1.0
        s += w * sign * q
    n = float(np.linalg.norm(s))
    return s / n if n > 1e-18 else np.array([0.0, 0.0, 0.0, 1.0])


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
    d = json.load(open(path))

    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    final = np.asarray(d["points"]["final"], float)
    n = len(bind)
    nb = len(d["bones"])

    print("顶点 %d  骨骼 %d  姿态 %s" % (n, nb, d["scene"]))

    # bindToPose 旋转 = poseQ * restQ
    quats = []
    for b in d["bones"]:
        Rpose = mat3_from_maya(b["worldMatrix"])
        Rrest = mat3_from_maya(b["bindPreMatrix"])
        q = quat_from_mat3(Rpose)          # poseQ
        qr = quat_from_mat3(Rrest)         # restQ
        # 四元数乘：q_pose * q_rest
        x1, y1, z1, w1 = q
        x2, y2, z2, w2 = qr
        prod = np.array([
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
        ])
        quats.append(prod)
        ang = 2.0 * math.degrees(math.acos(max(-1.0, min(1.0, abs(prod[3])))))
        print("  %-14s bindToPose 转角 %8.3f deg" % (b["short"], ang))

    # 权重矩阵
    W = np.zeros((n, nb), float)
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    sums = W.sum(axis=1)
    print("\n权重和: min=%.6f max=%.6f  偏离1的顶点数=%d"
          % (sums.min(), sums.max(), int(np.count_nonzero(np.abs(sums - 1.0) > 1e-6))))
    for k in range(nb):
        nz = int(np.count_nonzero(W[:, k] > 1e-12))
        print("  骨 %d 非零权重顶点 %5d  权重和 %.1f" % (k, nz, W[:, k].sum()))

    # 逐顶点 A_i
    Rmats = [quat_to_mat3(q) for q in quats]
    disp = final - cor
    Az = np.zeros(n)
    Axy_inv_ok = np.zeros(n, bool)
    delta_xy = np.zeros((n, 2))

    for i in range(n):
        wi = W[i]
        active = [k for k in range(nb) if wi[k] > 0.0]
        sumMat = np.zeros((3, 3))
        for k in active:
            sumMat += wi[k] * Rmats[k]
        qb = blend_rotation([wi[k] for k in active], [quats[k] for k in active])
        Rb = quat_to_mat3(qb)
        A = sumMat - Rb
        Az[i] = A[2, 2]
        # 限制在 XY 平面
        Axy = A[:2, :2]
        det = Axy[0, 0] * Axy[1, 1] - Axy[0, 1] * Axy[1, 0]
        if abs(det) > 1e-14:
            Axy_inv_ok[i] = True
            delta_xy[i] = np.linalg.solve(Axy, disp[i, :2])

    print("\n== 预测 1: A_i 第三行必须为 0（旋转都绕 Z）==")
    print("  max |A_zz| = %.3e" % np.abs(Az).max())

    print("\n== 预测 2: (final - original).z 必须为 0 ==")
    dz = np.abs(disp[:, 2])
    print("  max |dz| = %.4e   mean = %.4e" % (dz.max(), dz.mean()))
    bad = np.argsort(-dz)[:8]
    print("  最大 8 个顶点: %s" % ", ".join(str(int(i)) for i in bad))
    for i in bad:
        print("    v%-6d w=%s  dz=%+.4e" % (i, np.round(W[i], 4), disp[i, 2]))

    print("\n== 预测 3: 反解 delta_xy 与位移放大倍数 ==")
    ok = Axy_inv_ok
    print("  可反解顶点 %d / %d" % (int(ok.sum()), n))
    mag_d = np.linalg.norm(delta_xy[ok], axis=1)
    mag_p = np.linalg.norm(disp[ok, :2], axis=1)
    print("  |delta_xy| 均=%.4f max=%.4f" % (mag_d.mean(), mag_d.max()))
    print("  |disp_xy|  均=%.4f max=%.4f" % (mag_p.mean(), mag_p.max()))

    # 按 w1 分箱看放大倍数
    w1 = W[:, 1] if nb > 1 else np.zeros(n)
    print("\n%8s %8s %12s %12s %12s" % ("w1区间", "顶点数", "|disp|均", "|delta|均", "放大倍数"))
    edges = [0.0, 0.02, 0.05, 0.1, 0.2, 0.3, 0.5, 0.7, 0.9, 1.0]
    for a, b in zip(edges[:-1], edges[1:]):
        m = ok & (w1 >= a) & (w1 < b) if b < 1.0 else ok & (w1 >= a) & (w1 <= b)
        if m.sum() < 10:
            continue
        pd = np.linalg.norm(disp[m, :2], axis=1).mean()
        dd = np.linalg.norm(delta_xy[m], axis=1).mean()
        print("%8s %8d %12.5f %12.5f %12.1f"
              % ("%.2f-%.2f" % (a, b), int(m.sum()), pd, dd, dd / max(pd, 1e-12)))

    np.save("tests/_delta_xy.npy", delta_xy)
    np.save("tests/_W.npy", W)
    print("\n保存 tests/_delta_xy.npy, tests/_W.npy")


if __name__ == "__main__":
    main()
