# -*- coding: utf-8 -*-
r"""消融：fillet=0 的原版引导下，compliance 是否仍控制 s=-2.28 的边翻转。

固定原版 hinge_displacement、λ=0、关闭限幅。只切原 compliance 与 compliance=0。

记录：
  · s=-2.28 与 s=-2.52 两处的左右边【有符号投影比】和最终边长
  · 同一过渡区（|s| ∈ [1.8, 3.0]）的最大转角
  · s=0 的原有深折单列（原版引导在那里本来就有折，别让它盖住过渡区）

    mayapy tests/ablation_plain.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402
import ablation as abl  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201


def signed_projection(vec, guide_edge):
    """vec 在 guide_edge 方向上的有符号投影，除以 guide_edge 长度。

    > 0 表示仍有正向投影；< -1 表示已完全翻到反向。
    """
    lg = float(np.linalg.norm(guide_edge))
    if lg < 1e-15:
        return float("nan")
    return float(vec @ guide_edge) / (lg * lg)


def probe_at(stations, meridian, s_target):
    """返回 (k, i0, i1, i2)，其中 stations[k] == s_target（环索引，不是转角索引）。

    转角 turn[j] 对应 stations[j+1]（由点 j, j+1, j+2 构成）；
    b- 是 i0->i1，b+ 是 i1->i2，两条边的中点都在 stations[k+1] 附近。
    """
    k = int(np.argmin(np.abs(stations - s_target)))
    return k, meridian[k], meridian[k + 1], meridian[k + 2]


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

        # 原版引导（fillet=0）
        g_plain = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
        T_plain = original + g_plain
        print("原版引导：|guide|max %.6f" % np.linalg.norm(g_plain, axis=1).max())

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])

        # 先看原版引导自身在过渡区的转角（确认它平顺）
        turn_g = gvf.signed_turning_2d(T_plain[meridian])
        all_s = stations[1:]
        region = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]
        print("原版引导在 |s|∈[1.8,3.0] 的最大 |转角| %.3f°"
              % max(abs(turn_g[i]) for i in region))
        print("原版引导在 s≈0 的转角 %.2f°（这是原版自带的深折，单列）"
              % turn_g[int(np.argmin(np.abs(all_s)))])

        groups = (
            ("原 compliance", dict(use_compliance=True, traction_gain=7.0)),
            ("compliance=0", dict(use_compliance=False, traction_gain=7.0)),
        )

        probes = (-2.52, -2.28, -2.22)
        print("\n" + "=" * 88)
        print("消融：原版引导 / λ=0 / 无限幅")
        print("=" * 88)
        results = {}
        for label, kw in groups:
            d, info = abl.structural_centers_ablate(
                edges, original, W, quats, g_plain, outer, lambda_c=0.0, **kw)
            final = original + np.einsum("nij,nj->ni", info["A"], d)
            turn_f = gvf.signed_turning_2d(final[meridian])
            results[label] = (final, turn_f, d)

            print("\n--- %s ---" % label)
            print("  |delta|max %.3f" % np.linalg.norm(d, axis=1).max())
            print("  %8s %12s %12s %12s %12s %10s"
                  % ("边中点s", "b-投影比", "b+投影比", "|b-|", "|b+|", "转角°"))
            for s_target in probes:
                k, i0, i1, i2 = probe_at(stations, meridian, s_target)
                bg_m = T_plain[i1] - T_plain[i0]
                bg_p = T_plain[i2] - T_plain[i1]
                fm = final[i1] - final[i0]
                fp = final[i2] - final[i1]
                pm = signed_projection(fm, bg_m)
                pp = signed_projection(fp, bg_p)
                # 转角 turn_f[k] 由 i0,i1,i2 构成，对应 stations[k+1]
                print("  %8.3f %12.4f %12.4f %12.5f %12.5f %10.2f"
                      % (stations[k + 1], pm, pp,
                         float(np.linalg.norm(fm)), float(np.linalg.norm(fp)),
                         turn_f[k]))
            reg = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]
            imax = max(reg, key=lambda i: abs(turn_f[i]))
            print("  过渡区最大 |转角| %.2f° @ s=%.3f" % (abs(turn_f[imax]),
                                                          all_s[imax]))
            i0z = int(np.argmin(np.abs(all_s)))
            print("  s≈0 转角 %.2f°（单列，与非过渡区比较）"
                  % float(turn_f[i0z]))

        # ---- 对比 ----
        print("\n" + "=" * 88)
        print("关键判断：-2.28 的 b- 是否由反向转为正向？尖峰是否迁到 -2.52？")
        print("=" * 88)
        for s_target in (-2.52, -2.28):
            k, i0, i1, i2 = probe_at(stations, meridian, s_target)
            print("\n  边中点 s = %.3f（环 %.3f）" % (stations[k + 1], stations[k]))
            print("    %-16s %12s %12s %12s"
                  % ("组别", "b-投影比", "|b-|", "转角°"))
            for label, _ in groups:
                final, turn_f, _d = results[label]
                bg_m = T_plain[i1] - T_plain[i0]
                fm = final[i1] - final[i0]
                print("    %-16s %12.4f %12.5f %12.2f"
                      % (label, signed_projection(fm, bg_m),
                         float(np.linalg.norm(fm)), turn_f[k]))

        ka, i0a, i1a, _i2a = probe_at(stations, meridian, -2.28)
        kb, i0b, i1b, _i2b = probe_at(stations, meridian, -2.52)
        fa = results["原 compliance"][0]
        fb = results["compliance=0"][0]
        pa_before = signed_projection(fa[i1a] - fa[i0a], T_plain[i1a] - T_plain[i0a])
        pa_after = signed_projection(fb[i1a] - fb[i0a], T_plain[i1a] - T_plain[i0a])
        a = results["原 compliance"][1]
        b = results["compliance=0"][1]
        reg_a = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]
        mx_a = max(reg_a, key=lambda i: abs(a[i]))
        mx_b = max(reg_a, key=lambda i: abs(b[i]))
        print("\n  判读：")
        print("    -2.28 的 b- 投影比：%.4f -> %.4f  （%s）"
              % (pa_before, pa_after,
                 "由反转变为正向" if pa_after > 0 > pa_before else "未改变方向"))
        print("    过渡区尖峰：原 compliance %.2f°@s=%.3f  ->  compliance=0 %.2f°@s=%.3f"
              % (abs(a[mx_a]), all_s[mx_a], abs(b[mx_b]), all_s[mx_b]))
        if abs(pa_before) > 1.0 and pa_after > -1.0:
            print("    -> 复现了尖峰迁移，且独立于圆角。")
        else:
            print("    -> 未复现；上一轮结论需限定在非零圆角那组输入。")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
