# -*- coding: utf-8 -*-
r"""显式构造 B 与 B^TB，验证对称正定，并给出正确的算子实现。

B 的形状 (m, 3n)，第 e 行：
    B[e, 3*eb+:] = +t̂_e^T A_eb / l_e
    B[e, 3*ea+:] = -t̂_e^T A_ea / l_e

惩罚 Hessian = B^T W B（W 对角），必然对称半正定。
先在小规模（少数边）上验证，再放大。

    mayapy tests/btb_check.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402
from edge_constraint_base import build_solver  # noqa: E402

HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201


def build_B(A, ea, eb, et, el, n, sqrt_k=None):
    """B 的第 e 行（对应 E = mu k l^2 (q-rho)^2，Bx-d = sqrt(k) l (q-rho)）：

        B[e, 3*ea:] = -sqrt(k_e) · t̂_e^T A_ea
        B[e, 3*eb:] = +sqrt(k_e) · t̂_e^T A_eb

    注意不带 1/l，也不带 l：l 的因子由 d = sqrt(k) l (rho - q_base) 承担。
    """
    m = len(ea)
    sk = np.ones(m) if sqrt_k is None else np.asarray(sqrt_k, float)
    B = np.zeros((m, 3 * n))
    for e in range(m):
        B[e, 3 * ea[e]:3 * ea[e] + 3] = -sk[e] * (et[e] @ A[ea[e]])
        B[e, 3 * eb[e]:3 * eb[e] + 3] = +sk[e] * (et[e] @ A[eb[e]])
    return B


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind_list, faces, bind, original, w1 = gvf.build_scene(mc, om)
        W = np.column_stack([1.0 - w1, w1])
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        g = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))

        ctx = build_solver(edges, original, W, quats, g, outer)
        A = ctx["A"]
        a_e, b_e = ctx["a"], ctx["b"]
        n = len(original)

        ids = [(4880, 4960), (4720, 4800), (4800, 4880)]
        sel = np.array([np.nonzero((a_e == i) & (b_e == j))[0][0]
                        for i, j in ids])
        ea, eb = a_e[sel], b_e[sel]
        dg = g[eb] - g[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        print("测试边：%s" % [tuple(x) for x in zip(ea.tolist(), eb.tolist())])

        B = build_B(A, ea, eb, et, el, n)
        print("B 形状 %s（只有 %d 行，不要构造 3n x 3n 的 H）" % (B.shape, B.shape[0]))

        # 只在小矩阵上验证性质：B B^T 是 (m, m)，Gram 矩阵必对称半正定。
        G = B @ B.T
        print("B B^T 形状 %s  对称偏差（相对）: %.3e"
              % (G.shape, np.abs(G - G.T).max() / max(np.abs(G).max(), 1e-30)))
        ev = np.linalg.eigvalsh(G)
        print("B B^T 特征值：min %.6e  max %.6e （min 应 >= 0）"
              % (ev.min(), ev.max()))
        ev2 = np.linalg.eigvalsh(B.T @ B) if B.shape[1] <= 3000 else None
        if ev2 is not None:
            print("B^TB 特征值 min %.6e" % ev2.min())

        from edge_constraint_base import solve_pcg
        x0 = solve_pcg(ctx, ctx["rhs0"], ctx["system0"])
        print("\n基线 |x|max %.4f" % np.linalg.norm(x0, axis=1).max())

        def pen_exact(xx, w):
            v = xx.reshape(-1)
            return (B.T @ ((B @ v) * w)).reshape(n, 3)

        # 核对：bincount 实现是否等价
        def pen_bincount(xx, w):
            disp = np.einsum("nij,nj->ni", A, xx)
            bx = (np.einsum("ni,ni->n", disp[eb], et) -
                  np.einsum("ni,ni->n", disp[ea], et)) / el
            coef = w * bx / el
            fb = np.einsum("nij,nj->ni", A[eb], et * coef[:, None])
            fa = np.einsum("nij,nj->ni", A[ea], et * coef[:, None])
            return np.stack([
                np.bincount(eb, weights=fb[:, c], minlength=n) -
                np.bincount(ea, weights=fa[:, c], minlength=n)
                for c in range(3)], axis=1)

        rng = np.random.default_rng(5)
        for w in (1.0, 100.0):
            u = rng.normal(size=(n, 3))
            d = np.abs(pen_exact(u, w) - pen_bincount(u, w)).max()
            print("  w=%5.0f  bicount 与 B^TB 的最大差: %.3e  %s"
                  % (w, d, "等价" if d < 1e-8 * max(1, np.abs(u).max()) else "**不等价**"))

        print("\n用显式 B^TB 求解：")
        for w in (1e-2, 1e0, 1e1, 1e2, 1e3, 1e4):
            def system(xx, w=w):
                return ctx["system0"](xx) + pen_exact(xx, w)
            x = solve_pcg(ctx, ctx["rhs0"], system, max_iters=250)
            fin = np.linalg.norm(x, axis=1).max()
            print("  w=%.0e  |x|max %.6e  %s"
                  % (w, fin, "OK" if fin < 1e3 else "**发散**"))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
