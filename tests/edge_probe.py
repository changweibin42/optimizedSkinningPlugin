# -*- coding: utf-8 -*-
r"""最小可行验证：惩罚项加到极小 μ 时，PCG 是否仍然收敛。

如果 μ=1e-8 也发散，说明惩罚算子的实现有根本问题；
如果只有大 μ 发散，说明是量级/条件数问题。

    mayapy tests/edge_probe.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402
from edge_constraint_base import build_solver, solve_pcg  # noqa: E402

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

        # 基线
        x0 = solve_pcg(ctx, ctx["rhs0"], ctx["system0"])
        print("基线：|x0|max %.4f" % np.linalg.norm(x0, axis=1).max())

        # 只取少数几条边做测试（KEY 边）
        sel = np.array([np.nonzero((a_e == 4880) & (b_e == 4960))[0][0]])
        print("测试边数 %d" % len(sel))
        ea, eb = a_e[sel], b_e[sel]
        dg = g[eb] - g[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]

        def s_of(x):
            disp = np.einsum("nij,nj->ni", A, x)
            return ((np.einsum("ni,ni->n", disp[eb], et) -
                     np.einsum("ni,ni->n", disp[ea], et)) /
                    np.maximum(el, 1e-15))

        def pen(x, w):
            d = s_of(x)
            coef = w * d
            fb = np.einsum("nij,nj->ni", A[eb], et * coef[:, None])
            fa = np.einsum("nij,nj->ni", A[ea], et * coef[:, None])
            return np.stack([
                np.bincount(eb, weights=fb[:, c], minlength=n) -
                np.bincount(ea, weights=fa[:, c], minlength=n)
                for c in range(3)], axis=1)

        # 半正定自检（单边）
        rng = np.random.default_rng(1)
        for w in (1e-6, 1e-2, 1.0, 1e3):
            vals = []
            asym = 0.0
            for _ in range(4):
                u = rng.normal(size=(n, 3))
                v = rng.normal(size=(n, 3))
                q1 = float(np.sum(u * pen(v, w)))
                vals.append(q1)
                asym = max(asym, abs(q1 - float(np.sum(v * pen(u, w)))))
            print("  w=%.0e  x·Hy 最小 %+.4e  对称偏差 %.3e"
                  % (w, min(vals), asym))

        # 逐 μ 试解，看是否收敛
        print("\n逐 μ（w 是惩罚权重，注意不是 mu）")
        for w in (1e-8, 1e-6, 1e-4, 1e-2, 1e0, 1e2):
            def system(xx, w=w):
                return ctx["system0"](xx) + pen(xx, w)
            x = solve_pcg(ctx, ctx["rhs0"], system, max_iters=200)
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
