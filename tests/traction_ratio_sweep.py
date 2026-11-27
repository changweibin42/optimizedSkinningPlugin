"""扫描 traction 系数与 guide_mass，看能否让 A·delta 贴合 guide。

rhs = AT[scatter(k*(target-(orig_b-orig_a))) + guide_mass*guide + T*pull + curvature]
guide_mass = M * degree

目标：dy_final / guide_y 尽量接近 1（不过冲）。
用线性近似 A·delta 预测，避免每次跑 Maya。

    python tests/traction_ratio_sweep.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

D = r"D:\tmp\ocd_case3.json"
STEP = 0.12


def solve(edges, original, weights, rotations, guide, outer,
          center_limit=None, M=2.0, T=7.0):
    a, b, rest, k = edges
    n = len(original)
    A = ref.center_response(weights, rotations)
    AT = A.transpose(0, 2, 1)
    R = np.einsum("nk,kij->nij", weights, ref.rotation_matrix(rotations)) - A

    def apply(matrix, x):
        return (matrix @ x[:, :, None])[:, :, 0]

    def scatter(values):
        return np.column_stack([
            np.bincount(b, weights=v, minlength=n)
            - np.bincount(a, weights=v, minlength=n) for v in values.T])

    degree = np.bincount(a, weights=k, minlength=n) + np.bincount(b, weights=k, minlength=n)
    ridge = 0.00006 * np.maximum(degree, 1e-12)
    target = apply(0.5 * (R[a] + R[b]), rest)

    prediction = original + guide
    length = np.linalg.norm(rest, axis=1)
    extension = np.maximum(
        np.linalg.norm(prediction[b] - prediction[a], axis=1) - length, 0.)
    axial = rest[:, 0] / length
    extension = length * np.tanh(extension / length)
    traction = (k * extension / length * axial ** 2
                * np.minimum(outer[a], outer[b]))[:, None] * rest

    tw = weights ** 3
    tw = tw / tw.sum(axis=1, keepdims=True)
    frame = np.einsum("nk,kij->nij", tw, ref.rotation_matrix(rotations))
    frame -= ref.center_response(tw, rotations)
    left = apply(frame[a], traction)
    right = apply(frame[b], traction)
    pull = np.column_stack([
        np.bincount(a, weights=l, minlength=n) - np.bincount(b, weights=r, minlength=n)
        for l, r in zip(left.T, right.T)])

    guide_mass = M * degree[:, None]
    rest_k = k * axial ** 2
    current_length = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    longitudinal = rest_k * length / np.maximum(current_length, 0.15 * length)

    def lap(x):
        return scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n)
                  + np.bincount(b, weights=longitudinal, minlength=n))
    compliance = (8.0 * np.minimum(1., 4. * np.abs(weights[:, 0] - weights[:, 1])) ** 2
                  / np.maximum(lap_degree, 1e-12))
    rest_detail = scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(compliance[:, None] * (apply(R, rest_detail) - lap(original)))
    rhs = apply(AT, scatter(k[:, None] * (target - (original[b] - original[a])))
                + guide_mass * guide + T * pull + curvature_rhs)

    def system(x):
        u = apply(A, x)
        elastic = scatter(k[:, None] * (u[b] - u[a])) + guide_mass * u
        return apply(AT, elastic + lap(compliance[:, None] * lap(u))) + ridge[:, None] * x

    diag = (degree + guide_mass[:, 0] + compliance * lap_degree ** 2
            + np.bincount(a, weights=longitudinal ** 2 * compliance[b], minlength=n)
            + np.bincount(b, weights=longitudinal ** 2 * compliance[a], minlength=n))
    block = diag[:, None, None] * (AT @ A) + ridge[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(block)
    x = np.zeros_like(original)
    r = rhs.copy()
    d = apply(inverse, r)
    rz = np.sum(r * d)
    tol = rz * 1e-12
    for _ in range(100):
        if rz <= tol:
            break
        hd = system(d)
        alpha = rz / np.sum(d * hd)
        x += alpha * d
        r -= alpha * hd
        z = apply(inverse, r)
        nr = np.sum(z * r)
        d = z + (nr / rz) * d
        rz = nr
    if center_limit is not None:
        x *= np.minimum(1., center_limit / np.maximum(
            np.linalg.norm(x, axis=1, keepdims=True), 1e-12))
    return x


def main():
    d = json.load(open(D))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    faces = d["mesh"]["faces"]
    cor = np.asarray(d["points"]["cor"], float)
    n = len(bind)

    child = [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]
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

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

    cor_l = to_local(cor)
    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    q = ref.joint_quaternion(bend, 0.0, 0.0)
    rots = [(0.0, 0.0, 0.0, 1.0), q]
    guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    outer_p = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))
    edges = ref.local_edges(bind_l.tolist(), faces)

    rings = np.round(bind[:, 0] / STEP).astype(int)
    poles = []
    for rid in sorted(set(rings.tolist())):
        x0 = rid * STEP
        if not (-2.0 <= x0 <= 0.2):
            continue
        m2 = np.where(rings == rid)[0]
        i = m2[np.argmin(cor_l[m2, 1])]
        poles.append((x0, i))
    print("bend=%.4f  外极点 %d 个" % (bend, len(poles)))

    A = ref.center_response(W, rots)

    print("\n%6s %8s |" % ("M", "T") + "".join(" %9.2f" % x for x, _ in poles[:12]))
    print("  " + " " * 16 + "".join(" %9s" % "ratio" for _ in poles[:12]))

    for M, T in ((2.0, 7.0), (2.0, 3.0), (2.0, 1.0), (2.0, 0.0),
                 (5.0, 7.0), (10.0, 7.0), (20.0, 7.0), (50.0, 7.0),
                 (20.0, 1.0), (50.0, 0.0)):
        dd = solve(edges, cor_l, W, rots, guide, outer_p, None, M=M, T=T)
        pred = np.einsum("nij,nj->ni", A, dd)
        line = "%6.1f %8.1f |" % (M, T)
        for x0, i in poles[:12]:
            gv = guide[i, 1]
            pv = pred[i, 1]
            r = pv / gv if abs(gv) > 1e-9 else float("nan")
            line += " %9.2f" % r
        print(line)

    print("\n（ratio = 实际 y 位移 / guide 要求；1.0 为理想）")


if __name__ == "__main__":
    main()
