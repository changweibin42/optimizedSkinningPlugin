# -*- coding: utf-8 -*-
"""用 C++ 实际导出的 guide/delta，判定它内部用的是哪种 (bend, direction)。

候选：
  A) bend=+146.13, direction=0     ← make_python_delta.py 当前做法
  B) bend=+146.13, direction=180   ← 等价于绕 Z 负转（矩阵实际方向）
  C) 真实相对四元数（绕 Z -146.13）

    python tests/verify_cpp_uses_which_angle.py D:/tmp/ocd_case3.json tests/_stage1.json
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


def main():
    case = json.load(open(sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"))
    stage = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "tests/_stage1.json"))

    bind = np.asarray(case["mesh"]["bindPoints"], float)
    faces = case["mesh"]["faces"]
    n = len(bind)
    bones = case["bones"]

    child = [b for b in bones if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(case["hinge"]["axis"], float)
    axis /= np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)

    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    R = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

    cor = np.asarray(case["points"]["cor"], float)
    rc = cor - pivot
    cor_l = np.column_stack([(rc @ axis) / R, (rc @ bY) / R, (rc @ bZ) / R])

    W = np.zeros((n, len(bones)))
    for i, row in enumerate(case["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(bones)) if W[:, j].max() > 0.0]
    W = W[:, used]

    Rpose = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    Rbind = np.linalg.inv(np.asarray(child["bindPreMatrix"], float).reshape(4, 4))[:3, :3]
    relRot = Rpose @ Rbind.T
    q_true = quat_from_mat3(relRot)
    ang_true = math.degrees(math.atan2(relRot[1, 0], relRot[0, 0]))

    # ---- C++ 实际导出 ----
    cg = np.asarray(stage["cppGuide"], float).reshape(n, 3)
    cd = np.asarray(stage["cppDelta"], float).reshape(n, 3)
    print("C++ 导出: |guide|均=%.6f  |delta|均=%.6f  |delta|max=%.6f"
          % (np.linalg.norm(cg, axis=1).mean(),
             np.linalg.norm(cd, axis=1).mean(),
             np.linalg.norm(cd, axis=1).max()))
    print("stage1 里记录的 centerLimit: %s" % stage.get("centerLimit"))
    print("stage1 键: %s" % list(stage.keys()))

    edges = ref.local_edges(bind_l.tolist(), faces)

    def solve(ang, direction, q_use):
        rotations = [(0.0, 0.0, 0.0, 1.0), q_use][:len(used)]
        g = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, ang, 0.0, direction))
        o = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, direction))
        dd = np.asarray(ref.structural_centers(edges, cor_l, W, rotations, g, o, 16.0))
        return g, o, dd

    cands = [
        ("A +146.13 dir=0  ", 146.13, 0.0, ref.joint_quaternion(146.13, 0.0, 0.0)),
        ("B +146.13 dir=180", 146.13, 180.0, ref.joint_quaternion(146.13, 0.0, 180.0)),
        ("C true quat      ", ang_true, 0.0, tuple(q_true)),
    ]

    print("\n%20s %12s %14s %14s" % ("候选", "|guide|均", "vs C++ guide", "vs C++ delta"))
    for nm, ang, dirc, q in cands:
        g, o, dd = solve(ang, dirc, q)
        dg = np.abs(g - cg).max()
        ddm = np.abs(dd - cd).max()
        print("%20s %12.6f %14.6e %14.6e"
              % (nm, np.linalg.norm(g, axis=1).mean(), dg, ddm))

    print("\n== 结论判据 ==")
    best = None
    for nm, ang, dirc, q in cands:
        g, o, dd = solve(ang, dirc, q)
        err = np.abs(dd - cd).max()
        if best is None or err < best[1]:
            best = (nm, err)
    print("  与 C++ 最接近的是: %s  (delta max 差 %.3e)" % best)


if __name__ == "__main__":
    main()
