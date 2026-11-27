"""查 x=+0.5 与 x=-0.5 的不对称来源。

把 rhs 分解成各贡献项，比较两处的大小。
参考实现 rhs =
    apply(AT, scatter(k*(target-(original[b]-original[a])))
          + guide_mass*guide + 7.0*pull + curvature_rhs)
"""
import json
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
a, b, rest, k = edges
ang = 146.1265
q = ref.joint_quaternion(ang, 0.0, 0.0)
rotations = [(0.0, 0.0, 0.0, 1.0), q]
cpp_guide = np.asarray(stage["cppGuide"], float).reshape(n, 3)
cpp_delta = np.asarray(stage["cppDelta"], float).reshape(n, 3)
limit = float(stage["centerLimit"][0])
outer = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, 0.0))

A = ref.center_response(W, rotations)
AT = A.transpose(0, 2, 1)
Rm = np.einsum("nk,kij->nij", W, ref.rotation_matrix(rotations)) - A


def apply(matrix, x):
    return (matrix @ x[:, :, None])[:, :, 0]


def scatter(values):
    return np.column_stack([
        np.bincount(b, weights=v, minlength=n)
        - np.bincount(a, weights=v, minlength=n) for v in values.T])


degree = np.bincount(a, weights=k, minlength=n) + np.bincount(b, weights=k, minlength=n)
target = apply(0.5 * (Rm[a] + Rm[b]), rest)

prediction = cor_l + cpp_guide
length = np.linalg.norm(rest, axis=1)
extension = np.maximum(
    np.linalg.norm(prediction[b] - prediction[a], axis=1) - length, 0.0)
axial = rest[:, 0] / length
extension = length * np.tanh(extension / length)
traction = (k * extension / length * axial ** 2
            * np.minimum(outer[a], outer[b]))[:, None] * rest

tw = W ** 3
tw /= tw.sum(axis=1, keepdims=True)
frame = np.einsum("nk,kij->nij", tw, ref.rotation_matrix(rotations))
frame -= ref.center_response(tw, rotations)
left = apply(frame[a], traction)
right = apply(frame[b], traction)
pull = np.column_stack([
    np.bincount(a, weights=l, minlength=n)
    - np.bincount(b, weights=r, minlength=n) for l, r in zip(left.T, right.T)])

guide_mass = 2.0 * degree[:, None]
rest_k = k * axial ** 2
cur_len = np.linalg.norm(prediction[b] - prediction[a], axis=1)
longitudinal = rest_k * length / np.maximum(cur_len, 0.15 * length)
lap_degree = (np.bincount(a, weights=longitudinal, minlength=n)
              + np.bincount(b, weights=longitudinal, minlength=n))
compliance = (8.0 * np.minimum(1., 4. * np.abs(W[:, 0] - W[:, 1])) ** 2
              / np.maximum(lap_degree, 1e-12))


def lap(x):
    return scatter(longitudinal[:, None] * (x[b] - x[a]))


rest_detail = scatter(rest_k[:, None] * rest)
curv_input = compliance[:, None] * (apply(Rm, rest_detail) - lap(cor_l))
curvature_rhs = lap(curv_input)

term_elastic = apply(AT, scatter(k[:, None] * (target - (cor_l[b] - cor_l[a]))))
term_guide = apply(AT, guide_mass * cpp_guide)
term_pull = apply(AT, 7.0 * pull)
term_curv = apply(AT, curvature_rhs)

xw = bind_l[:, 0] * R

print("== 各 rhs 项在 x=±0.5 处的量级 ==")
print("%6s %4s %12s %12s %12s %12s %12s"
      % ("x", "n", "elastic", "guide", "pull", "curvature", "合计"))
for x0 in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
    sel = np.abs(xw - x0) < 0.06
    if sel.sum() < 10:
        continue
    m = lambda arr: np.linalg.norm(arr[sel], axis=1).mean()
    total = m(term_elastic) + m(term_guide) + m(term_pull) + m(term_curv)
    print("%6.2f %4d %12.4f %12.4f %12.4f %12.4f %12.4f"
          % (x0, sel.sum(), m(term_elastic), m(term_guide), m(term_pull),
             m(term_curv), total))

print("\n== 对称性检查：x=+0.5 vs x=-0.5 ==")
sp = np.abs(xw - 0.5) < 0.06
sn = np.abs(xw + 0.5) < 0.06
print("%-12s %12s %12s %10s" % ("项", "x=+0.5", "x=-0.5", "比值"))
for label, arr in (("elastic", term_elastic), ("guide", term_guide),
                   ("pull", term_pull), ("curvature", term_curv),
                   ("delta", cpp_delta), ("guide向量", cpp_guide)):
    mp = np.linalg.norm(arr[sp], axis=1).mean()
    mn = np.linalg.norm(arr[sn], axis=1).mean()
    print("%-12s %12.4f %12.4f %10.3f" % (label, mp, mn, mp / max(1e-12, mn)))

print("\n== 权重与几何在两点 ==")
for x0 in (-0.5, 0.5):
    sel = np.abs(xw - x0) < 0.06
    print("  x=%.1f  w0=%.4f w1=%.4f  |w0-w1|=%.4f  compliance均值=%.4e"
          % (x0, W[sel, 0].mean(), W[sel, 1].mean(),
             np.abs(W[sel, 0] - W[sel, 1]).mean(), compliance[sel].mean()))
    print("         outer=%.4f  degree=%.4f  longitudinal=%.4f"
          % (outer[sel].mean(), degree[sel].mean(), longitudinal[sel].mean()))

# guide 的径向与轴向分量在两点
print("\n== guide 分解在 x=±0.5 ==")
rlen = np.linalg.norm(rad, axis=1)
runi = rad / np.maximum(rlen, 1e-12)[:, None]
gx = cpp_guide @ axis
gr = np.einsum("vi,vi->v", cpp_guide, runi)
for x0 in (-0.5, 0.5):
    sel = np.abs(xw - x0) < 0.06
    print("  x=%.1f  guide轴向=%+.4f  径向=%+.4f  模长=%.4f"
          % (x0, gx[sel].mean(), gr[sel].mean(),
             np.linalg.norm(cpp_guide[sel], axis=1).mean()))
