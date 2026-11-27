# -*- coding: utf-8 -*-
"""用外部 Python（有 numpy）算 Python 独立生成的 delta，并转成对象空间向量。

读取 mayapy 第一阶段导出的 cppGuide/cppDelta/centerLimit，
用参考实现重新生成 guide 与 delta（不喂 C++ 的 guide，独立生成），
再换算成插件 ocdCenterOffsets 要求的对象空间向量。

    python tests/make_python_delta.py tests/_stage1.json tests/_python_delta.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

import ocd_elbow_reference_latest as ref


def mat_inverse(m):
    aug = np.hstack([m, np.eye(4)])
    for col in range(4):
        pivot = int(np.argmax(np.abs(aug[col:, col]))) + col
        if abs(aug[pivot, col]) < 1e-12:
            raise ValueError("奇异")
        aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]
        for row in range(4):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]
    return aug[:, 4:]


def main():
    stage = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
        HERE, "_stage1.json")
    target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
        HERE, "_python_delta.json")
    source = r"D:\tmp\ocd_case3.json"

    case = json.load(open(source))
    st_data = json.load(open(stage))

    bind = np.asarray(case["mesh"]["bindPoints"], float)
    n = len(bind)
    faces = case["mesh"]["faces"]
    bones = case["bones"]
    by = {b["short"]: b for b in bones}
    child = by["L_Ankle_jnt"]

    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(case["hinge"]["axis"], float)
    axis = axis / np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.array([0.0, 0.0, 1.0])

    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

    # original = C++ 的 CoR 输出（第一遍沿用，只换 delta 来源）
    cor = np.asarray(case["points"]["cor"], float)
    rc = cor - pivot
    cor_l = np.column_stack([(rc @ axis) / R, (rc @ bY) / R, (rc @ bZ) / R])

    W = np.zeros((n, len(bones)))
    for i, row in enumerate(case["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(bones)) if W[:, j].max() > 0.0]
    W = W[:, used]

    # 相对旋转由矩阵反解，不猜 bend 符号
    child_pose = np.asarray(child["worldMatrix"], float).reshape(4, 4)
    child_bind = mat_inverse(np.asarray(child["bindPreMatrix"],
                                       float).reshape(4, 4))
    rel_rot = child_pose[:3, :3] @ child_bind[:3, :3].T
    trace = float(np.trace(rel_rot))
    ang = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))
    # C++ 在 hingeDisplacement / structuralCenters 内部用 radians(abs(bend))，
    # 且四元数用 joint_quaternion(bend, twist, direction) 构造。
    # 这里直接选与矩阵匹配的符号，再交给下游。
    rel_deg = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))
    q_pos = ref.joint_quaternion(rel_deg, 0.0, 0.0)
    q_neg = ref.joint_quaternion(-rel_deg, 0.0, 0.0)

    def _qmat(qq):
        x, y, z, w = qq
        return np.array([
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ])

    err_pos = float(np.abs(_qmat(q_pos) - rel_rot).max())
    err_neg = float(np.abs(_qmat(q_neg) - rel_rot).max())
    if err_neg < err_pos:
        rel_deg = -rel_deg
    # C++ 对 bend 取 abs，所以送入参考实现的是绝对值。
    ang = abs(rel_deg)
    q = ref.joint_quaternion(ang, 0.0, 0.0)
    print("  相对旋转 |%.4f|度 (rel_deg=%.4f)  q=%s"
          % (ang, rel_deg, np.round(q, 6)))

    edges = ref.local_edges(bind_l.tolist(), faces)
    rotations = [(0.0, 0.0, 0.0, 1.0), q]

    # Python 独立生成 guide（不喂 C++ 的）
    py_guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, ang, 0.0, 0.0))
    outer = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, 0.0))
    limit = float(st_data["centerLimit"][0]) if st_data.get("centerLimit") else 16.0
    py_delta = np.asarray(ref.structural_centers(
        edges, cor_l, W, rotations, py_guide, outer, limit))

    # 与 C++ 对比（诊断用）
    cpp_guide = np.asarray(st_data["cppGuide"], float).reshape(n, 3)
    cpp_delta = np.asarray(st_data["cppDelta"], float).reshape(n, 3)
    print("\n== guide 对比（Python 独立生成 vs C++）==")
    dg = np.linalg.norm(py_guide - cpp_guide, axis=1)
    print("  |py|均=%.6f  |cpp|均=%.6f  差 max=%.6f mean=%.6f  @v%d"
          % (np.linalg.norm(py_guide, axis=1).mean(),
             np.linalg.norm(cpp_guide, axis=1).mean(),
             dg.max(), dg.mean(), int(np.argmax(dg))))
    print("== delta 对比（Python 独立生成 vs C++）==")
    dd = np.linalg.norm(py_delta - cpp_delta, axis=1)
    print("  |py|均=%.6f  |cpp|均=%.6f  差 max=%.6f mean=%.6f  @v%d"
          % (np.linalg.norm(py_delta, axis=1).mean(),
             np.linalg.norm(cpp_delta, axis=1).mean(),
             dd.max(), dd.mean(), int(np.argmax(dd))))

    # 局部归一 -> 对象空间（本夹具 worldMat 为单位阵）
    obj = (py_delta[:, 0:1] * axis[None, :]
           + py_delta[:, 1:2] * bY[None, :]
           + py_delta[:, 2:3] * bZ[None, :]) * R
    json.dump({"objectDelta": obj.ravel().tolist(),
               "localDelta": py_delta.ravel().tolist(),
               "pyGuide": py_guide.ravel().tolist(),
               "radius": R, "bend": ang},
              open(target, "w"))
    print("\nwrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
