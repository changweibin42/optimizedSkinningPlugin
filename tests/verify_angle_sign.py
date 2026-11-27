# -*- coding: utf-8 -*-
"""验证 make_python_delta.py 的符号简化是否改变了 delta。

矩阵实际是绕 Z 的 -146.13 度。脚本用 abs() 取正、direction=0 重建。
参考实现里 bend<0 会让 plane += pi，弯面朝向差 180 度。

对比三种输入的 delta：
  A) ang=+146.13, direction=0    （脚本当前做法）
  B) ang=+146.13, direction=180  （等价于负弯：plane 加 pi）
  C) 真实相对四元数（绕 Z -146.13）

    python tests/verify_angle_sign.py D:/tmp/ocd_case3.json
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref


def quat_from_mat3(R):
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        return np.array([(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s,
                         (R[1, 0] - R[0, 1]) / s, 0.25 * s])
    if R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        return np.array([0.25 * s, (R[0, 1] + R[1, 0]) / s,
                         (R[0, 2] + R[2, 0]) / s, (R[2, 1] - R[1, 2]) / s])
    if R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        return np.array([(R[0, 1] + R[1, 0]) / s, 0.25 * s,
                         (R[1, 2] + R[2, 1]) / s, (R[0, 2] - R[2, 0]) / s])
    s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
    return np.array([(R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s,
                     0.25 * s, (R[1, 0] - R[0, 1]) / s])


def qmat(q):
    x, y, z, w = q
    n = math.sqrt(x * x + y * y + z * z + w * w)
    x, y, z, w = x / n, y / n, z / n, w / n
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])


def main():
    d = json.load(open(sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"))

    bind = np.asarray(d["mesh"]["bindPoints"], float)
    faces = d["mesh"]["faces"]
    n = len(bind)
    bones = d["bones"]

    child = [b for b in bones if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(d["hinge"]["axis"], float)
    axis /= np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)

    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

    cor = np.asarray(d["points"]["cor"], float)
    rc = cor - pivot
    cor_l = np.column_stack([(rc @ axis) / R, (rc @ bY) / R, (rc @ bZ) / R])

    W = np.zeros((n, len(bones)))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(bones)) if W[:, j].max() > 0.0]
    W = W[:, used]
    print("使用骨骼列:", used, " 权重列数", W.shape[1])

    # 真实相对旋转
    Rpose = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    Rbind = np.linalg.inv(np.asarray(child["bindPreMatrix"], float).reshape(4, 4))[:3, :3]
    relRot = Rpose @ Rbind.T
    q_true = quat_from_mat3(relRot)
    ang_true = math.degrees(math.atan2(relRot[1, 0], relRot[0, 0]))
    print("\n真实相对旋转: 绕 Z %.4f deg  q=%s" % (ang_true, np.round(q_true, 6)))

    ids = [i for i in range(len(bones)) if i in used]
    idParent = used[0]
    idChild = used[1] if len(used) > 1 else used[0]

    edges = ref.local_edges(bind_l.tolist(), faces)

    def solve(label, ang, direction, q_use, rot_parent=None):
        rp = rot_parent if rot_parent is not None else (0.0, 0.0, 0.0, 1.0)
        rotations = [rp, q_use]
        if len(used) > 2:
            rotations.append(q_use)
        rotations = rotations[:len(used)]
        g = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, ang, 0.0, direction))
        o = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, direction))
        dd = np.asarray(ref.structural_centers(edges, cor_l, W, rotations, g, o, 16.0))
        return g, o, dd, rotations

    gA, oA, dA, rA = solve("A", 146.13, 0.0, ref.joint_quaternion(146.13, 0.0, 0.0))
    gB, oB, dB, rB = solve("B", 146.13, 180.0, ref.joint_quaternion(146.13, 0.0, 180.0))
    gC, oC, dC, rC = solve("C", ang_true, 0.0, tuple(q_true))

    print("\n== outer 参与度对比（凸侧/凹侧翻转的直接指标）==")
    print("  A mean=%.4f   B mean=%.4f   C mean=%.4f"
          % (oA.mean(), oB.mean(), oC.mean()))
    print("  A 与 C 的差 max=%.4f" % np.abs(oA - oC).max())
    print("  B 与 C 的差 max=%.4f" % np.abs(oB - oC).max())

    print("\n== guide 对比 ==")
    for nm, g in (("A", gA), ("B", gB), ("C", gC)):
        print("  %s |guide|均=%.6f" % (nm, np.linalg.norm(g, axis=1).mean()))
    print("  A vs C 差 max=%.6e  mean=%.6e" % (np.abs(gA - gC).max(), np.abs(gA - gC).mean()))
    print("  B vs C 差 max=%.6e  mean=%.6e" % (np.abs(gB - gC).max(), np.abs(gB - gC).mean()))

    print("\n== delta 对比 ==")
    for nm, dd in (("A", dA), ("B", dB), ("C", dC)):
        print("  %s |delta|均=%.6f max=%.6f"
              % (nm, np.linalg.norm(dd, axis=1).mean(), np.linalg.norm(dd, axis=1).max()))
    print("  A vs C 差 max=%.6e  mean=%.6e" % (np.abs(dA - dC).max(), np.abs(dA - dC).mean()))
    print("  B vs C 差 max=%.6e  mean=%.6e" % (np.abs(dB - dC).max(), np.abs(dB - dC).mean()))


if __name__ == "__main__":
    main()
