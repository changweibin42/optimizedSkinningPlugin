# -*- coding: utf-8 -*-
r"""方案 A：用导出的夹具对拍 Python 参考实现，判定红圈凹陷归属。

严格复现 164d594 的归一化（src/optimizedSkinCluster.cpp:2964-3012, 3115, 3332）：

  parentBindRotation = inverse(rest[parent].q)   # rest 存的是逆
  referenceUp        = parentBindRotation * (0,1,0)
  basisY             = normalize(referenceUp - axis*dot(referenceUp, axis))
  basisZ             = normalize(cross(axis, basisY))
  relative           = bindWorld - bindChildPivot
  station            = dot(relative, axis)       # 世界单位
  maxRadius          = max |relative - axis*station|
  toLocal(o)         = (dot(o,axis), dot(o,basisY), dot(o,basisZ)) / maxRadius
  centerLimit_local  = ocdCenterLimit / maxRadius

本例已核实 joint1 绑定世界旋转 = 单位阵，axis=+X，故 basisY=+Y、basisZ=+Z。
关节中心在世界 x=0，child 枢轴也在 x=0，所以世界 x=0 -> 局部 x=0。

第一遍不重写基础 CoR：original 直接取 C++ 算法 0 的输出，只对拍
「给定同一份 original，Python 求出的 delta 是否与 C++ 一致」。

用法：
    python tests/ocd_alg5_replay_compare.py D:/tmp/ocd_case3.json
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
            raise ValueError("奇异矩阵")
        aug[[col, pivot]] = aug[[pivot, col]]
        aug[col] /= aug[col, col]
        for row in range(4):
            if row != col:
                aug[row] -= aug[row, col] * aug[col]
    return aug[:, 4:]


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
    d = json.load(open(source))

    bind = np.asarray(d["mesh"]["bindPoints"], float)
    n = len(bind)
    bones = d["bones"]
    by_name = {b["short"]: b for b in bones}

    # ---------- 稀疏权重 -> 稠密，只保留真正有权重的骨 ----------
    full = np.zeros((n, len(bones)))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            full[i, int(k)] = float(v)
    used = [j for j in range(len(bones)) if full[:, j].max() > 0.0]
    W = full[:, used]
    used_bones = [bones[j] for j in used]
    print("== 实际生效的骨骼 ==")
    for j, b in zip(used, used_bones):
        print("  [%d] %-14s bind=%s  权重和=%.1f"
              % (j, b["short"], b["bindPosition"], full[:, j].sum()))

    # ---------- 铰链基（严格按源码） ----------
    parent = by_name["joint1"]
    child = by_name["L_Ankle_jnt"]

    parent_bind = mat_inverse(np.asarray(parent["bindPreMatrix"], float).reshape(4, 4))
    parent_rot = parent_bind[:3, :3]

    axis_world = np.asarray(d["hinge"]["axis"], float)
    axis_world = axis_world / np.linalg.norm(axis_world)

    upward = parent_rot @ np.array([0.0, 1.0, 0.0])
    basisY = upward - axis_world * float(upward @ axis_world)
    if np.linalg.norm(basisY) < 1e-6:
        side = parent_rot @ np.array([0.0, 0.0, 1.0])
        basisY = side - axis_world * float(side @ axis_world)
    basisY = basisY / np.linalg.norm(basisY)
    basisZ = np.cross(axis_world, basisY)
    basisZ = basisZ / np.linalg.norm(basisZ)

    print("\n== 铰链基（按源码复算）==")
    print("  axis   = %s" % np.round(axis_world, 6))
    print("  basisY = %s" % np.round(basisY, 6))
    print("  basisZ = %s" % np.round(basisZ, 6))

    pivot = np.asarray(child["bindPosition"], float)
    print("  bindChildPivot = %s" % np.round(pivot, 4))

    # ---------- 归一化 ----------
    relative = bind - pivot
    station_world = relative @ axis_world
    radial_world = relative - np.outer(station_world, axis_world)
    maxRadius = float(np.linalg.norm(radial_world, axis=1).max())
    print("\n== 归一化 ==")
    print("  maxRadius = %.6f" % maxRadius)
    print("  station(世界) %.4f .. %.4f" % (station_world.min(), station_world.max()))

    def to_local(points_world):
        rel = np.asarray(points_world, float) - pivot
        return np.stack([
            (rel @ axis_world) / maxRadius,
            (rel @ basisY) / maxRadius,
            (rel @ basisZ) / maxRadius,
        ], axis=1)

    # 自检：关节中心世界 x=0 -> 局部 x=0
    check0 = to_local(np.array([[0.0, 0.0, 0.0]]))[0]
    print("  自检 世界(0,0,0) -> 局部 x = %.6f  (应为 0)" % check0[0])
    check12 = to_local(np.array([[12.0, 0.0, 0.0]]))[0]
    print("  自检 世界(12,0,0) -> 局部 x = %.6f  (应为 %.4f)"
          % (check12[0], 12.0 / maxRadius))

    bind_local = to_local(bind)
    cor_local = to_local(d["points"]["cor"])
    ocd_local = to_local(d["points"]["ocd"])
    cpp_local = to_local(d["points"]["final"])

    # ---------- 弯曲角 ----------
    child_pose = np.asarray(child["worldMatrix"], float).reshape(4, 4)
    child_bind = mat_inverse(np.asarray(child["bindPreMatrix"], float).reshape(4, 4))
    rel_rot = child_pose[:3, :3] @ child_bind[:3, :3].T
    cosb = max(-1.0, min(1.0, (np.trace(rel_rot) - 1.0) / 2.0))
    bend = math.degrees(math.acos(cosb))
    if rel_rot[0, 1] < 0.0:
        bend = -bend
    print("\n== 弯曲角（从矩阵反解）= %.3f 度 ==" % bend)

    # ---------- Python 参考求解 ----------
    faces = d["mesh"]["faces"]
    edges = ref.local_edges(bind_local.tolist(), faces)
    rotations = [ref.joint_quaternion(0, 0, 0),
                 ref.joint_quaternion(bend, 0, 0)]

    original = cor_local
    guide = np.asarray(ref.hinge_displacement(
        bind_local, original, W, bend, 0.0, 0.0))
    outer = np.asarray(ref.outer_participation(
        bind_local, W, bend, 0.0, 0.0))

    center_limit_world = float(d["attributes"]["ocdCenterLimit"])
    center_limit_local = center_limit_world / maxRadius
    print("  centerLimit 世界=%.4f -> 局部=%.6f"
          % (center_limit_world, center_limit_local))

    delta = np.asarray(ref.structural_centers(
        edges, original, W, rotations, guide, outer, center_limit_local))

    py_final = original + delta
    py_guided = original + guide

    # ---------- 有符号径向（统一测量轴 = 局部 yz 平面） ----------
    def radial(points):
        return np.linalg.norm(points[:, 1:3], axis=1)

    stations_local = bind_local[:, 0]
    uniq = np.unique(np.round(stations_local, 5))
    width = float(np.median(np.diff(uniq)))
    keys = np.round((stations_local - uniq[0]) / width).astype(int)

    print("\n== 三条轮廓（局部，正=向外鼓）==")
    print("%9s %7s %9s %10s %10s %11s %10s"
          % ("station", "w1", "R_cor", "guide_dR", "py_dR", "cpp_dR", "cpp-py"))
    rows = []
    for k in sorted(set(keys.tolist())):
        sel = keys == k
        s = stations_local[sel].mean()
        if abs(s) > 2.2:
            continue
        Rc = radial(cor_local[sel]).mean()
        g = (radial(py_guided[sel]) - radial(cor_local[sel])).mean()
        p = (radial(py_final[sel]) - radial(cor_local[sel])).mean()
        c = (radial(cpp_local[sel]) - radial(cor_local[sel])).mean()
        w1 = W[sel, 1].mean()
        rows.append((s, w1, Rc, g, p, c))
        print("%9.3f %7.3f %9.4f %10.4f %10.4f %11.4f %10.4f"
              % (s, w1, Rc, g, p, c, c - p))

    arr = np.array(rows)
    print("\n== 判定 ==")
    for label, col in (("引导形态", 3), ("Python 最终", 4), ("C++ 最终", 5)):
        idx = int(np.argmin(arr[:, col]))
        print("  %-12s 最凹 station %7.3f  径向变化 %+.4f"
              % (label, arr[idx, 0], arr[idx, col]))
    gap = np.abs(arr[:, 5] - arr[:, 4])
    worst = int(np.argmax(gap))
    print("  C++ 与 Python 最大径向差 %.4f 在 station %.3f"
          % (gap[worst], arr[worst, 0]))

    cpp_deep = int(np.argmin(arr[:, 5]))
    py_deep = int(np.argmin(arr[:, 4]))
    if abs(arr[py_deep, 4] - arr[cpp_deep, 5]) < 0.02:
        print("  => Python 复现了 C++ 的凹陷：不是移植差异")
    else:
        print("  => Python 与 C++ 有实质差异：需查移植")
    return 0


if __name__ == "__main__":
    sys.exit(main())
