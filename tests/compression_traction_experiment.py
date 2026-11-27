"""实验：给 structural_centers 加"压缩牵引"（对称于现有拉伸牵引）。

现有 traction 只处理 extension>0（拉长），压缩的边完全没有约束。
外侧轮廓被压扁正是因为压缩无约束。

改为：
    ext = current - length
    if ext > 0: 拉伸项（现有）
    else:       压缩项，沿 rest 切向把两端推开

在 16080 夹具上比较：外侧轮廓、红线区域（x≈-0.5）、以及 118/120 度是否退化。

    python tests/compression_traction_experiment.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

D = r"D:\tmp\ocd_case3.json"
SIDES = 80
STEP = 0.12


def patched_structural_centers(edges, original, weights, rotations, guide,
                               outer, center_limit=None,
                               compression_gain=0.0):
    """在原版基础上加压缩牵引项。compression_gain=0 等价原版。"""
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
    current = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    delta_len = current - length
    axial = rest[:, 0] / length

    # 拉伸项（原版）
    ext_t = np.maximum(delta_len, 0.0)
    ext_t = length * np.tanh(ext_t / length)
    traction_t = (k * ext_t / length * axial ** 2
                  * np.minimum(outer[a], outer[b]))[:, None] * rest

    # 压缩项（新增）：压缩量同样经 tanh 饱和，沿 rest 切向反号
    ext_c = np.maximum(-delta_len, 0.0)
    ext_c = length * np.tanh(ext_c / length)
    traction_c = (-compression_gain * k * ext_c / length * axial ** 2
                  * np.minimum(outer[a], outer[b]))[:, None] * rest

    traction = traction_t + traction_c

    tw = weights ** 3
    tw = tw / tw.sum(axis=1, keepdims=True)
    frame = np.einsum("nk,kij->nij", tw, ref.rotation_matrix(rotations))
    frame -= ref.center_response(tw, rotations)
    left = apply(frame[a], traction)
    right = apply(frame[b], traction)
    pull = np.column_stack([
        np.bincount(a, weights=l, minlength=n) - np.bincount(b, weights=r, minlength=n)
        for l, r in zip(left.T, right.T)])

    guide_mass = 2.0 * degree[:, None]
    rest_k = k * axial ** 2
    longitudinal = rest_k * length / np.maximum(current, 0.15 * length)

    def lap(x):
        return scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n)
                  + np.bincount(b, weights=longitudinal, minlength=n))
    compliance = (8.0 * np.minimum(1., 4. * np.abs(weights[:, 0] - weights[:, 1])) ** 2
                  / np.maximum(lap_degree, 1e-12))
    rest_detail = scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(compliance[:, None] * (apply(R, rest_detail) - lap(original)))
    rhs = apply(AT, scatter(k[:, None] * (target - (original[b] - original[a])))
                + guide_mass * guide + 7.0 * pull + curvature_rhs)

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
    axis = axis / np.linalg.norm(axis)
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
        for k2, v in row.items():
            W[i, int(k2)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    print("bend=%.4f  R=%.6f" % (bend, R))

    q = ref.joint_quaternion(bend, 0.0, 0.0)
    rots = [(0.0, 0.0, 0.0, 1.0), q]
    guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
    outer_p = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))
    edges = ref.local_edges(bind_l.tolist(), faces)

    rings = np.round(bind[:, 0] / STEP).astype(int)

    for gain in (0.0, 1.0, 2.0, 3.0, 5.0):
        dd = patched_structural_centers(edges, cor_l, W, rots, guide, outer_p,
                                        None, compression_gain=gain)
        # 用 A·delta 预测顶点位移（线性近似），得到外侧轮廓
        A = ref.center_response(W, rots)
        pred = cor_l + np.einsum("nij,nj->ni", A, dd)
        # 转回世界 y
        y_world = pivot[1] + R * (pred[:, 1] * bY[1] + pred[:, 2] * bZ[1]
                                  + pred[:, 0] * axis[1])
        print("\n=== compression_gain=%.1f  |delta|均=%.4f ==="
              % (gain, np.linalg.norm(dd, axis=1).mean()))
        print("  %8s %14s %14s" % ("x", "CoR_miny", "pred_miny"))
        for rid in sorted(set(rings.tolist())):
            x0 = rid * STEP
            if not (-2.6 <= x0 <= 0.7):
                continue
            m2 = np.where(rings == rid)[0]
            print("  %8.3f %14.5f %14.5f"
                  % (x0, cor[m2, 1].min(), y_world[m2].min()))


if __name__ == "__main__":
    main()
