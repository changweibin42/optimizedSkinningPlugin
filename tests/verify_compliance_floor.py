"""用 monkey-patch 方式改 compliance，其余完全走参考实现。

参考实现的 structural_centers 里 compliance 是内联计算的，
所以这里复制该函数源码、仅替换 compliance 一行，保证其他逻辑逐字不变。
"""
import json
import math
import sys

import numpy as np

HERE = r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests"
sys.path.insert(0, HERE)
sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\scripts")
import ocd_elbow_reference_latest as ref

case = json.load(open(r"D:\tmp\ocd_case3.json"))
stage = json.load(open(HERE + r"\_stage1.json"))

bind = np.asarray(case["mesh"]["bindPoints"], float)
n = len(bind)
bones = case["bones"]
by = {b["short"]: b for b in bones}
child = by["L_Ankle_jnt"]
pivot = np.asarray(child["bindPosition"], float)
axis = np.asarray(case["hinge"]["axis"], float)
axis /= np.linalg.norm(axis)
rel = bind - pivot
st = rel @ axis
rad = rel - np.outer(st, axis)
R = float(np.linalg.norm(rad, axis=1).max())
bY = np.array([0.0, 1.0, 0.0])
bZ = np.array([0.0, 0.0, 1.0])
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

faces = case["mesh"]["faces"]
edges = ref.local_edges(bind_l.tolist(), faces)
ang = 146.1265
q = ref.joint_quaternion(ang, 0.0, 0.0)
rotations = [(0.0, 0.0, 0.0, 1.0), q]
cpp_guide = np.asarray(stage["cppGuide"], float).reshape(n, 3)
cpp_delta = np.asarray(stage["cppDelta"], float).reshape(n, 3)
limit = float(stage["centerLimit"][0])
outer = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, 0.0))

import inspect
src = inspect.getsource(ref.structural_centers)
print("== 确认参考实现里的 compliance 行 ==")
for line in src.split("\n"):
    if "compliance" in line and "8.0" in line:
        print("  ", line.strip())

# 构造 floors=0 版本应当与参考实现完全一致（自检）
def make_fn(floor):
    body = src
    body = body.replace("def structural_centers(",
                        "def custom_structural_centers(", 1)
    old = "compliance = 8.0*np.minimum(1.,4.*abs(weights[:,0]-weights[:,1]))**2/np.maximum(lap_degree,1e-12)"
    assert old in body, "找不到 compliance 行，源码可能已变"
    new = ("_shape = np.minimum(1.,4.*abs(weights[:,0]-weights[:,1]))**2\n"
           "    compliance = 8.0*({f}+(1.0-{f})*_shape)/np.maximum(lap_degree,1e-12)"
           ).format(f=repr(float(floor)))
    body = body.replace(old, new)
    ns = dict(ref.__dict__)
    exec(compile(body, "<custom>", "exec"), ns)
    return ns["custom_structural_centers"]


def roundness(pts, xw):
    out = {}
    for x0 in [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]:
        sel = np.abs(xw - x0) < 0.06
        if sel.sum() < 10:
            continue
        r = np.linalg.norm(pts[sel][:, 1:3], axis=1)
        out[x0] = (r.std() / max(1e-9, r.mean()), r.mean())
    return out


xw = bind_l[:, 0] * R

# 自检
fn0 = make_fn(0.0)
d0 = fn0(edges, cor_l, W, rotations, cpp_guide, outer, limit)
print("\n自检 floor=0 vs 参考实现: max 差 = %.3e"
      % np.abs(d0 - ref.structural_centers(edges, cor_l, W, rotations,
                                           cpp_guide, outer, limit)).max())
print("自检 floor=0 vs C++ delta: max 差 = %.3e"
      % np.abs(d0 - cpp_delta).max())

print("\n== compliance 下限扫描：圆度（越小越圆）==")
xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
print("%8s" % "floor" + "".join("  x=%+.1f" % v for v in xs))
base = None
for floor in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
    fn = make_fn(floor)
    delta = fn(edges, cor_l, W, rotations, cpp_guide, outer, limit)
    pts = cor_l + delta
    rn = roundness(pts, xw)
    if base is None:
        base = rn
    line = "%8.2f" % floor
    for x0 in xs:
        line += " %6.3f" % rn[x0][0] if x0 in rn else "      -"
    print(line)

print("\n== 径向均值 ==")
print("%8s" % "floor" + "".join("  x=%+.1f" % v for v in xs))
for floor in (0.0, 0.1, 0.3, 1.0):
    fn = make_fn(floor)
    delta = fn(edges, cor_l, W, rotations, cpp_guide, outer, limit)
    pts = cor_l + delta
    rn = roundness(pts, xw)
    line = "%8.2f" % floor
    for x0 in xs:
        line += " %6.3f" % rn[x0][1] if x0 in rn else "      -"
    print(line)

# 对照 Elbow 解析模型
bind_list = [tuple(p) for p in bind]
analytic = np.asarray([list(p) for p in
                       ref.Elbow(ang, 0.0, 0.0, ref.ZERO, False).mesh(bind_list)])
al = np.column_stack([
    ((analytic - pivot) @ axis) / R,
    ((analytic - pivot) @ bY) / R,
    ((analytic - pivot) @ bZ) / R,
])
rn = roundness(al, xw)
print("\n== 对照：Elbow 解析模型 ==")
print("%8s" % "解析" + "".join("  x=%+.1f" % v for v in xs))
line = "%8s" % ""
for x0 in xs:
    line += " %6.3f" % rn[x0][0] if x0 in rn else "      -"
print(line)
line = "%8s" % "径向"
for x0 in xs:
    line += " %6.3f" % rn[x0][1] if x0 in rn else "      -"
print(line)
