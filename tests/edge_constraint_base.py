# -*- coding: utf-8 -*-
r"""structural_centers 的可复用求解器骨架（Python 版）。

与参考实现逐行对应，额外把右端项 rhs0 与线性算子 system0 暴露出来，
方便在上面叠加额外惩罚项（中心耦合、边投影约束等）。

    build_solver(edges, original, weights, rotations, guide, outer, lambda_c)
      -> dict(A, apply, system0, rhs0, inverse, original, a, b, k)

    solve_pcg(ctx, rhs, system) -> x
"""
import numpy as np

import ocd_elbow_reference_latest as ref


def build_solver(edges, original, weights, rotations, guide, outer,
                 lambda_c=0.0):
    a, b, rest, k = edges
    n = len(original)
    A = ref.center_response(weights, rotations)
    AT = A.transpose(0, 2, 1)
    R = np.einsum("nk,kij->nij", weights, ref.rotation_matrix(rotations)) - A

    def apply(matrix, x):
        return (matrix @ x[:, :, None])[:, :, 0]

    def scatter(values):
        return np.stack([
            np.bincount(b, weights=v, minlength=n) -
            np.bincount(a, weights=v, minlength=n) for v in values.T], axis=1)

    degree = np.bincount(a, weights=k, minlength=n) + \
        np.bincount(b, weights=k, minlength=n)
    ridge = 0.00006 * np.maximum(degree, 1e-12)
    target = apply(0.5 * (R[a] + R[b]), rest)

    prediction = original + guide
    length = np.linalg.norm(rest, axis=1)
    extension = np.maximum(
        np.linalg.norm(prediction[b] - prediction[a], axis=1) - length, 0.0)
    axial = rest[:, 0] / length
    extension = length * np.tanh(extension / length)
    traction = (k * extension / length * axial ** 2 *
                np.minimum(outer[a], outer[b]))[:, None] * rest

    tw = weights ** 3
    tw /= tw.sum(axis=1, keepdims=True)
    frame = np.einsum("nk,kij->nij", tw, ref.rotation_matrix(rotations))
    frame -= ref.center_response(tw, rotations)
    left = apply(frame[a], traction)
    right = apply(frame[b], traction)
    pull = np.stack([
        np.bincount(a, weights=lp, minlength=n) -
        np.bincount(b, weights=rp, minlength=n)
        for lp, rp in zip(left.T, right.T)], axis=1)

    guide_mass = 2.0 * degree[:, None]
    rest_k = k * axial ** 2
    current_length = np.linalg.norm(prediction[b] - prediction[a], axis=1)
    longitudinal = rest_k * length / np.maximum(current_length, 0.15 * length)

    def lap(x):
        return scatter(longitudinal[:, None] * (x[b] - x[a]))

    lap_degree = (np.bincount(a, weights=longitudinal, minlength=n) +
                  np.bincount(b, weights=longitudinal, minlength=n))
    com = 8.0 * np.minimum(1.0, 4.0 * np.abs(weights[:, 0] - weights[:, 1])) ** 2 \
        / np.maximum(lap_degree, 1e-12)
    rest_detail = scatter(rest_k[:, None] * rest)
    curvature_rhs = lap(com[:, None] * (apply(R, rest_detail) - lap(original)))

    rhs0 = apply(AT,
                 scatter(k[:, None] * (target - (original[b] - original[a]))) +
                 guide_mass * guide + 7.0 * pull + curvature_rhs)

    center_edge_w = k * axial ** 2

    def system0(x):
        u = apply(A, x)
        elastic = scatter(k[:, None] * (u[b] - u[a])) + guide_mass * u
        out = apply(AT, elastic + lap(com[:, None] * lap(u))) + ridge[:, None] * x
        if lambda_c > 0.0:
            out = out + lambda_c * scatter(center_edge_w[:, None] *
                                           (x[b] - x[a]))
        return out

    diag = (degree + guide_mass[:, 0] + com * lap_degree ** 2 +
            np.bincount(a, weights=longitudinal ** 2 * com[b], minlength=n) +
            np.bincount(b, weights=longitudinal ** 2 * com[a], minlength=n))
    if lambda_c > 0.0:
        diag = diag + lambda_c * (
            np.bincount(a, weights=center_edge_w, minlength=n) +
            np.bincount(b, weights=center_edge_w, minlength=n))
    block = diag[:, None, None] * (AT @ A) + ridge[:, None, None] * np.eye(3)
    inverse = np.linalg.inv(block)

    return dict(A=A, apply=apply, system0=system0, rhs0=rhs0,
                inverse=inverse, original=original, a=a, b=b, k=k,
                latitude=dict(degree=degree, ridge=ridge, com=com,
                              longitudinal=longitudinal, guide_mass=guide_mass))


def solve_pcg(ctx, rhs, system, max_iters=400):
    inverse = ctx["inverse"]
    apply = ctx["apply"]
    x = np.zeros_like(ctx["original"])
    r = rhs.copy()
    d = apply(inverse, r)
    rz = np.sum(r * d)
    tol = rz * 1e-12
    for _ in range(max_iters):
        if rz <= tol:
            break
        hd = system(d)
        den = np.sum(d * hd)
        if abs(den) < 1e-30:
            break
        alpha = rz / den
        x += alpha * d
        r -= alpha * hd
        z = apply(inverse, r)
        nz = np.sum(z * r)
        d = z + (nz / rz) * d
        rz = nz
    return x
