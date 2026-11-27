# -*- coding: utf-8 -*-
r"""正确的自检：半正定用 u^T H u（同一向量），对称性用 u^T H v vs v^T H u。

B 的第 e 行作用在 x：
    (Bx)_e = [ t̂ᵀ(A_eb x_eb - A_ea x_ea) ] / l
    (BᵀWB x)_eb = + W_e / l² · A_ebᵀ t̂ [t̂ᵀ(A_eb x_eb - A_ea x_ea)]
    (BᵀWB x)_ea = - W_e / l² · A_eaᵀ t̂ [...]

其中 W_e = 2μ k l v（对单边测试取 2μk·1·v）。注意 BᵀB 自带 1/l²。

    mayapy tests/pen_form_check.py
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

        sel = np.array([np.nonzero((a_e == 4880) & (b_e == 4960))[0][0],
                        np.nonzero((a_e == 4720) & (b_e == 4800))[0][0]])
        ea, eb = a_e[sel], b_e[sel]
        dg = g[eb] - g[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        print("测试边 %d 条, l = %s" % (len(sel), np.array2string(el, precision=5)))

        def pen(x, w):
            """(Bᵀ W B) x，W_e = w（对单边测试）。BᵀB 自带 1/l²。"""
            disp = np.einsum("nij,nj->ni", A, x)
            bx = (np.einsum("ni,ni->n", disp[eb], et) -
                  np.einsum("ni,ni->n", disp[ea], et)) / el
            coef = w * bx / el          # 注意额外的 1/l
            fb = np.einsum("nij,nj->ni", A[eb], et * coef[:, None])
            fa = np.einsum("nij,nj->ni", A[ea], et * coef[:, None])
            return np.stack([
                np.bincount(eb, weights=fb[:, c], minlength=n) -
                np.bincount(ea, weights=fa[:, c], minlength=n)
                for c in range(3)], axis=1)

        rng = np.random.default_rng(3)
        print("\n半正定（u^T H u）与对称性（u^T H v vs v^T H u）：")
        for w in (1e-2, 1e0, 1e3, 1e5):
            min_q, max_asym = np.inf, 0.0
            for _ in range(6):
                u = rng.normal(size=(n, 3))
                v = rng.normal(size=(n, 3))
                q_uu = float(np.sum(u * pen(u, w)))
                min_q = min(min_q, q_uu)
                a1 = float(np.sum(u * pen(v, w)))
                a2 = float(np.sum(v * pen(u, w)))
                max_asym = max(max_asym, abs(a1 - a2))
            scale = max(abs(w), 1e-30)
            print("  w=%.0e  u^T H u 最小 %+.4e（相对 %.2e）  对称偏差 %.3e（相对 %.2e）"
                  % (w, min_q, min_q / scale, max_asym, max_asym / scale))

        # ---- 用正确形式逐 w 求解 ----
        print("\n逐 w 求解（w 是单边权重，非 μ）：")
        for w in (1e-6, 1e-4, 1e-2, 1e0, 1e1, 1e2, 1e3):
            def system(xx, w=w):
                return ctx["system0"](xx) + pen(xx, w)
            from edge_constraint_base import solve_pcg
            x = solve_pcg(ctx, ctx["rhs0"], system, max_iters=250)
            fin = np.linalg.norm(x, axis=1).max()
            print("  w=%.0e  |x|max %.6e  %s"
                  % (w, fin, "OK" if fin < 1e3 else "**发散**"))
            if fin > 1e3:
                break
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
