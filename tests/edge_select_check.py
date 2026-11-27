# -*- coding: utf-8 -*-
"""查边选择：为什么 q_base 会到 ±38？"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402

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
        edges = ref.local_edges(bind_list, faces)
        a_e, b_e, rest_e, k_e = edges
        g_plain = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))

        st_a, st_b = bind[a_e, 0], bind[b_e, 0]
        print("总边数 %d" % len(a_e))

        for name, m in (
            ("全部", np.ones(len(a_e), bool)),
            ("st_b>st_a", (st_b - st_a) > 1e-6),
            ("跨环", np.abs(st_a - st_b) > 1e-6),
            ("|s|<3", np.maximum(np.abs(st_a), np.abs(st_b)) < 3.0),
            ("不跨中线", (st_a * st_b) > 0.0),
        ):
            print("  %-12s %6d" % (name, int(m.sum())))

        mask = ((st_b - st_a) > 1e-6) & (np.abs(st_a - st_b) > 1e-6) & \
               (np.maximum(np.abs(st_a), np.abs(st_b)) < 3.0) & \
               ((st_a * st_b) > 0.0)
        sel = np.nonzero(mask)[0]
        dg = g_plain[b_e[sel]] - g_plain[a_e[sel]]
        el = np.linalg.norm(dg, axis=1)
        print("\n选中 %d 条边" % len(sel))
        print("  引导边长 l: min %.6f  median %.6f  max %.6f"
              % (el.min(), np.median(el), el.max()))

        base = original[b_e[sel]] - original[a_e[sel]]
        et = dg / np.maximum(el, 1e-15)[:, None]
        q_base = np.einsum("ni,ni->n", base, et) / np.maximum(el, 1e-15)
        print("  q_base: min %.4f  median %.4f  max %.4f"
              % (q_base.min(), np.median(q_base), q_base.max()))

        # 看最极端的几条边
        order = np.argsort(-np.abs(q_base))[:6]
        print("\n最极端的 6 条边：")
        print("%10s %10s %12s %12s %12s %12s"
              % ("i", "j", "station_i", "station_j", "l", "q_base"))
        for t in order:
            i, j = a_e[sel][t], b_e[sel][t]
            print("%10d %10d %12.4f %12.4f %12.6f %12.4f"
                  % (i, j, bind[i, 0], bind[j, 0], el[t], q_base[t]))

        # 这些边是真正的「沿经线」边吗？
        print("\n周向位置 j = id %% %d：" % SIDES)
        for t in order[:4]:
            i, j = a_e[sel][t], b_e[sel][t]
            di = original[j] - original[i]
            print("  边 %d->%d  周向 %d->%d  原始边向量 (%.4f, %.4f, %.4f)"
                  % (i, j, i % SIDES, j % SIDES, di[0], di[1], di[2]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
