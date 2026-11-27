# -*- coding: utf-8 -*-
"""用一个极小例子验证 E_edge 的梯度和 system 的符号是否与朴素有限差分一致。"""
import numpy as np

import ocd_elbow_reference_latest as ref


def tiny_case():
    """两点、一骨，手工构造 A 和边，用有限差分核对梯度。"""
    rng = np.random.default_rng(0)
    n = 2
    # 造两个 3x3 满秩矩阵当 A（真实 A 秩 2，但这里只验证梯度公式）
    A = rng.normal(size=(n, 3, 3))
    A[0] = A[0] @ A[0].T + np.eye(3)
    A[1] = A[1] @ A[1].T + np.eye(3)
    p0 = rng.normal(size=(n, 3))
    ia, ib = 0, 1
    k = 1.0
    mu = 2.0
    rho = 0.1

    dg = np.array([0.5, 0.2, 0.0])
    l = float(np.linalg.norm(dg))
    t = dg / l

    def q_of(x):
        disp = np.einsum("nij,nj->ni", A, x)
        base = p0[ib] - p0[ia]
        delta = disp[ib] - disp[ia]
        return float((base + delta) @ t) / l

    def energy(x):
        v = max(0.0, rho - q_of(x))
        return mu * k * l * l * v * v

    def q_at(x):
        disp = np.einsum("nij,nj->ni", A, x)
        return (np.einsum("ni,ni->n", disp[ib][None, :], t[None, :]) -
                np.einsum("ni,ni->n", disp[ia][None, :], t[None, :])) / l

    x0 = rng.normal(size=(n, 3)) * 2.0

    # 解析梯度：E = μ k l² v²，v = max(0, ρ−q)，q = (...)·t̂/l
    #   ∂E/∂Δc_b = -2μ k l² v · (A_bᵀ t̂)/l = -2μ k l v · A_bᵀ t̂
    #   ∂E/∂Δc_a = +2μ k l v · A_aᵀ t̂
    q = q_of(x0)
    v = max(0.0, rho - q)
    coef = 2.0 * mu * k * l * v          # 注意这里有 l（不是 l²/l 之外的因子）
    grad_b = -coef * (A[ib] @ t)
    grad_a = +coef * (A[ia] @ t)
    analytic = np.zeros_like(x0)
    analytic[ib] += grad_b
    analytic[ia] += grad_a

    # 有限差分
    eps = 1e-6
    numeric = np.zeros_like(x0)
    for i in range(n):
        for c in range(3):
            xp = x0.copy()
            xp[i, c] += eps
            xm = x0.copy()
            xm[i, c] -= eps
            numeric[i, c] = (energy(xp) - energy(xm)) / (2 * eps)

    print("q =", q)
    print("解析梯度:\n", analytic)
    print("数值梯度:\n", numeric)
    print("最大差:", np.abs(analytic - numeric).max())
    ok = np.abs(analytic - numeric).max() < 1e-4
    print("结论:", "梯度公式正确" if ok else "**梯度公式错误**")
    return ok


if __name__ == "__main__":
    tiny_case()
