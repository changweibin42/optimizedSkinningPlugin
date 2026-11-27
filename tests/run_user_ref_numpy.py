"""用外部 Python（有 numpy）跑 ocd_elbow_reference.py 的图元算法。

只算数值，不建 Maya 场景。输出给 mayapy 端做最终求值或直接对比。
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"D:\optimizedSkinningPlugin")
import ocd_elbow_reference as ref

src = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
out_path = sys.argv[2] if len(sys.argv) > 2 else r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests\_py_real.json"

d = json.load(open(src))
bind = np.asarray(d["mesh"]["bindPoints"], float)
n = len(bind)
bend = 146.1265

bones = d["bones"]
by = {b["short"]: b for b in bones}
child = by["L_Ankle_jnt"]
pivot = np.asarray(child["bindPosition"], float)
axis = np.asarray(d["hinge"]["axis"], float)
axis = axis / np.linalg.norm(axis)

rel = bind - pivot
st = rel @ axis
rad = rel - np.outer(st, axis)
R = float(np.linalg.norm(rad, axis=1).max())
print("maxRadius = %.4f" % R)

# 归一化局部坐标（与插件同口径）
bind_l = np.column_stack([st / R, (rad / R) @ np.array([0., 1., 0.]),
                          (rad / R) @ np.array([0., 0., 1.])])

# 权重（两列）
w1 = np.array([float(r.get("1", 0.0)) for r in d["weightsSparse"]])
W = np.column_stack([1.0 - w1, w1])

cor = np.asarray(d["points"]["cor"], float)
rc = cor - pivot
cor_l = np.column_stack([(rc @ axis) / R, (rc @ np.array([0., 1., 0.])) / R,
                         (rc @ np.array([0., 0., 1.])) / R])

faces = d["mesh"]["faces"]
print("计算 local_edges ...")
edges = ref.local_edges(bind_l.tolist(), faces)

print("计算 hinge_displacement ...")
q = ref.joint_quaternion(bend, 0.0, 0.0)
rotations = [(0.0, 0.0, 0.0, 1.0), q]
guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
print("  |guide| 均=%.4f max=%.4f" % (np.linalg.norm(guide, axis=1).mean(),
                                      np.linalg.norm(guide, axis=1).max()))

print("计算 outer_participation ...")
outer = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))
print("  outer 范围 %.4f .. %.4f" % (outer.min(), outer.max()))

print("计算 structural_centers ...")
delta = np.asarray(ref.structural_centers(edges, cor_l, W, rotations, guide,
                                          outer, 16.0))
dm = np.linalg.norm(delta, axis=1)
print("  |delta| 均=%.4f max=%.4f" % (dm.mean(), dm.max()))

# 把 delta 从局部归一换回对象空间（给注入用）
obj = (delta[:, 0:1] * axis[None, :]
       + delta[:, 1:2] * np.array([0., 1., 0.])[None, :]
       + delta[:, 2:3] * np.array([0., 0., 1.])[None, :]) * R

json.dump({
    "guide": guide.ravel().tolist(),
    "delta": delta.ravel().tolist(),
    "objectDelta": obj.ravel().tolist(),
    "radius": R,
    "bend": bend,
}, open(out_path, "w"))
print("wrote", out_path)

# 与 C++ 的对比（若有 stage1）
import os
stage_path = os.path.join(os.path.dirname(out_path), "_stage1.json")
if os.path.exists(stage_path):
    st_data = json.load(open(stage_path))
    cg = np.asarray(st_data["cppGuide"], float).reshape(n, 3)
    cd = np.asarray(st_data["cppDelta"], float).reshape(n, 3)
    dg = np.linalg.norm(guide - cg, axis=1)
    dd = np.linalg.norm(delta - cd, axis=1)
    print("\n== 与 C++ 对比 ==")
    print("  guide  差 max=%.3e mean=%.3e" % (dg.max(), dg.mean()))
    print("  delta  差 max=%.3e mean=%.3e" % (dd.max(), dd.mean()))
    print("  C++ |delta| 均=%.4f" % np.linalg.norm(cd, axis=1).mean())
