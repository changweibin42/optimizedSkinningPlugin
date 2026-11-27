# -*- coding: utf-8 -*-
r"""把 s=-2.280 的三点完整对齐：顶点 ID -> 边向量 -> 投影比 -> 转角。

解决上一轮的索引矛盾：+0.4970 到底是 b- 还是 b+。

对产生 109.78° 的那次原版求值（fillet=0, λ=0, 无限幅），打印：
  · i0/i1/i2 的顶点 ID、绑定 station、最终坐标
  · b- = P1-P0 与 b+ = P2-P1 的引导/最终二维向量
  · 每条边的投影比 dot(b_final, b_guide)/dot(b_guide, b_guide)
  · 两条最终边算出的 atan2(cross, dot)，复算 109.78°
  · 各自从引导边转到最终边的有符号角度，看两偏转能否重建尖峰

    mayapy tests/align_triple.py
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


def signed_angle(a, b):
    """a 转到 b 的有符号角度（度），弯面 XY 内。"""
    return math.degrees(math.atan2(a[0] * b[1] - a[1] * b[0], a[0] * b[0] + a[1] * b[1]))


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
        g_plain = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
        T_plain = original + g_plain

        d, info = abl.structural_centers_ablate(
            edges, original, W, quats, g_plain, outer,
            lambda_c=0.0, use_compliance=True, traction_gain=7.0)
        final = original + np.einsum("nij,nj->ni", info["A"], d)

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])

        # 找最终网格在过渡区的尖峰（环索引）
        turn = gvf.signed_turning_2d(final[meridian])
        all_s = stations[1:]
        reg = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]
        kpeak = max(reg, key=lambda i: abs(turn[i]))
        print("尖峰：转角 %+.2f°  由点 k=%d,%d,%d 构成，转角对应 station[%d]=%.3f"
              % (turn[kpeak], kpeak, kpeak + 1, kpeak + 2, kpeak + 1,
                 stations[kpeak + 1]))
        print("  （turn[j] 是点 j,j+1,j+2 的转角，记在 stations[j+1]）\n")

        # 用环索引对齐：i0 = 环 kpeak，i1 = 环 kpeak+1，i2 = 环 kpeak+2
        kidx = [kpeak, kpeak + 1, kpeak + 2]
        ids = [int(meridian[k]) for k in kidx]
        label = ["i0", "i1", "i2"]
        print("三点身份：")
        print("%4s %8s %10s %10s   %-34s %-34s"
              % ("", "顶点ID", "环索引", "station", "引导 T_guide", "最终 final"))
        for tag, k, i in zip(label, kidx, ids):
            print("%4s %8d %10d %10.3f   (%+8.5f,%+8.5f,%+8.5f)   (%+8.5f,%+8.5f,%+8.5f)"
                  % (tag, i, k, stations[k],
                     T_plain[i][0], T_plain[i][1], T_plain[i][2],
                     final[i][0], final[i][1], final[i][2]))

        # 两条边
        print("\n两条边（b- = i0->i1，b+ = i1->i2）：")
        print("%4s %-30s %-30s %10s %10s" %
              ("边", "引导边 XY", "最终边 XY", "投影比", "最终边长"))
        rows = []
        for tag, (a, b) in (("b-", (ids[0], ids[1])), ("b+", (ids[1], ids[2]))):
            bg = np.array([T_plain[b][0] - T_plain[a][0],
                           T_plain[b][1] - T_plain[a][1]])
            bf = np.array([final[b][0] - final[a][0],
                           final[b][1] - final[a][1]])
            proj = float(bf @ bg) / float(bg @ bg)
            rows.append((tag, bg, bf, proj))
            print("%4s (%+8.5f,%+8.5f)          (%+8.5f,%+8.5f)          %10.4f %10.5f"
                  % (tag, bg[0], bg[1], bf[0], bf[1], proj, np.linalg.norm(bf)))

        # 用两条最终边复算转角
        bf_m = rows[0][2]
        bf_p = rows[1][2]
        cross = bf_m[0] * bf_p[1] - bf_m[1] * bf_p[0]
        dot = bf_m[0] * bf_p[0] + bf_m[1] * bf_p[1]
        print("\n用两条最终边复算转角：atan2(%.6f, %.6f) = %+.4f°"
              % (cross, dot, math.degrees(math.atan2(cross, dot))))
        print("  （gvf 的 signed_turning 给出 %+.4f°，两者应一致）" % turn[kpeak])

        # 各自从引导边转到最终边的有符号角度
        print("\n各边从引导转到最终的有符号偏转：")
        for tag, bg, bf, proj in rows:
            d_ang = signed_angle(bg, bf)
            print("  %s: %+8.3f°   （投影比 %+.4f，%s）"
                  % (tag, d_ang, proj,
                     "正向" if proj > 0 else "已反向"))
        d_m = signed_angle(rows[0][1], rows[0][2])
        d_p = signed_angle(rows[1][1], rows[1][2])
        guide_ang = signed_angle(rows[0][1], rows[1][1])
        print("\n  引导边之间的夹角: %+.3f°" % guide_ang)
        print("  两个偏转之和 + 引导夹角 = %+.3f° + %+.3f° + %+.3f° = %+.3f°"
              % (d_m, d_p, guide_ang, d_m + guide_ang - d_p))
        print("  （转角 ≈ 引导夹角 + b+偏转 - b-偏转，因为 b- 是入边）")

        # 判据：两边是否都正向
        both_pos = all(r[3] > 0 for r in rows)
        print("\n判据：两条最终边对各自引导边的投影是否都为正？ %s" % both_pos)
        if both_pos:
            print("  -> 是。109.78° 不是边折返，而是两条边朝相反方向横向偏转而夹出。")
            print("     两偏转的符号相反：b- %+.3f°, b+ %+.3f°" % (d_m, d_p))
        else:
            neg = [r[0] for r in rows if r[3] <= 0]
            print("  -> 否。%s 已反向，属于边折返。" % neg)

        # 顺带把 -2.46 也对齐一次，确认它属于哪条边
        print("\n" + "=" * 78)
        print("对照：环 -2.46 附近（上一轮 -2.7152 的位置）")
        print("=" * 78)
        k46 = int(np.argmin(np.abs(stations - (-2.46))))
        ids46 = [int(meridian[k46]), int(meridian[k46 + 1]), int(meridian[k46 + 2])]
        print("%4s %8s %10s" % ("", "顶点ID", "station"))
        for tag, k, i in zip(label, [k46, k46 + 1, k46 + 2], ids46):
            print("%4s %8d %10.3f" % (tag, i, stations[k]))
        print("%4s %12s %10s" % ("边", "投影比", "最终边长"))
        for tag, (a, b) in (("b-", (ids46[0], ids46[1])),
                            ("b+", (ids46[1], ids46[2]))):
            bg = np.array([T_plain[b][0] - T_plain[a][0],
                           T_plain[b][1] - T_plain[a][1]])
            bf = np.array([final[b][0] - final[a][0],
                           final[b][1] - final[a][1]])
            proj = float(bf @ bg) / float(bg @ bg)
            print("%4s %12.4f %10.5f" % (tag, proj, np.linalg.norm(bf)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
