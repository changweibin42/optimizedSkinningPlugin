# -*- coding: utf-8 -*-
r"""compliance 消融（原版引导，λ=0，无限幅），全部用顶点 ID 锁定。

基线已对齐（原 compliance 组）：
  入边 4880 -> 4960   投影比 -0.2608
  出边 4960 -> 5040   投影比 +0.4970
  中间顶点 4960 转角  +109.783°

本脚本只跑原 compliance / compliance=0 两组，逐项对比。
所有边都写成「起点 ID -> 终点 ID」，转角单独标中间顶点 ID。

    mayapy tests/ablation_ids.py
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

# 固定索引（用户指定）
KEY_EDGES = [
    ("入边", 4880, 4960),
    ("出边", 4960, 5040),
    ("邻边1", 4720, 4800),
    ("邻边2", 4800, 4880),
]
KEY_TRIPLE = (4880, 4960, 5040)


def project_ratio(final, guide, a, b):
    """(P_b - P_a)·(G_b - G_a) / |G_b - G_a|²，弯面 XY。"""
    bg = np.array([guide[b][0] - guide[a][0], guide[b][1] - guide[a][1]])
    bf = np.array([final[b][0] - final[a][0], final[b][1] - final[a][1]])
    den = float(bg @ bg)
    return (float(bf @ bg) / den if den > 1e-18 else float("nan"),
            float(np.linalg.norm(bf)))


def turning_at(final, i0, i1, i2):
    u = np.array([final[i1][0] - final[i0][0], final[i1][1] - final[i0][1]])
    v = np.array([final[i2][0] - final[i1][0], final[i2][1] - final[i1][1]])
    return math.degrees(math.atan2(u[0] * v[1] - u[1] * v[0],
                                   u[0] * v[0] + u[1] * v[1]))


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

        groups = (
            ("原 compliance", dict(use_compliance=True, traction_gain=7.0)),
            ("compliance=0", dict(use_compliance=False, traction_gain=7.0)),
        )

        results = {}
        for label, kw in groups:
            d, info = abl.structural_centers_ablate(
                edges, original, W, quats, g_plain, outer, lambda_c=0.0, **kw)
            final = original + np.einsum("nij,nj->ni", info["A"], d)
            results[label] = (final, d)

        # ---------------- 逐边对比 ----------------
        print("=" * 84)
        print("compliance 消融（原版引导 fillet=0，λ=0，无限幅）")
        print("=" * 84)
        print("\n边（起点ID -> 终点ID）：")
        print("%-6s %18s %12s %12s %12s %12s"
              % ("", "边", "原投影比", "关后投影比", "原|b|", "关后|b|"))
        for tag, a, b in KEY_EDGES:
            pa, la = project_ratio(results["原 compliance"][0], T_plain, a, b)
            pb, lb = project_ratio(results["compliance=0"][0], T_plain, a, b)
            arrow = ""
            if pa < 0 <= pb:
                arrow = "  <<< 由反转变为正向"
            elif pb < pa:
                arrow = "  <<< 更负"
            print("%-6s %18s %12.4f %12.4f %12.5f %12.5f%s"
                  % (tag, "%d -> %d" % (a, b), pa, pb, la, lb, arrow))

        print("\n中间顶点转角（4880 -> 4960 -> 5040）：")
        ta = turning_at(results["原 compliance"][0], *KEY_TRIPLE)
        tb = turning_at(results["compliance=0"][0], *KEY_TRIPLE)
        print("  原 compliance %+9.4f°    compliance=0 %+9.4f°" % (ta, tb))

        # ---------------- 过渡区整体 ----------------
        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])
        all_s = stations[1:]
        reg = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]

        print("\n过渡区 |s|∈[1.8,3.0] 的整体指标：")
        print("%-16s %12s %10s %14s %10s"
              % ("组别", "最大|转角|", "中点ID", "峰值 station", "最短边长"))
        summary = {}
        for label, _ in groups:
            final = results[label][0]
            turn = gvf.signed_turning_2d(final[meridian])
            imax = max(reg, key=lambda i: abs(turn[i]))
            mid_id = int(meridian[imax + 1])
            el = np.linalg.norm(np.diff(final[meridian], axis=0), axis=1)
            el_reg = el[imax:imax + 2] if imax + 2 <= len(el) else el[imax:]
            summary[label] = (abs(turn[imax]), mid_id, all_s[imax],
                              float(el_reg.min()))
            print("%-16s %12.2f %10d %14.3f %10.5f"
                  % (label, abs(turn[imax]), mid_id, all_s[imax],
                     float(el_reg.min())))

        # ---------------- 判读 ----------------
        print("\n" + "=" * 84)
        print("判读")
        print("=" * 84)
        pa0, _ = project_ratio(results["原 compliance"][0], T_plain, 4880, 4960)
        pa1, _ = project_ratio(results["compliance=0"][0], T_plain, 4880, 4960)
        po0, _ = project_ratio(results["原 compliance"][0], T_plain, 4960, 5040)
        po1, _ = project_ratio(results["compliance=0"][0], T_plain, 4960, 5040)
        print("  入边 4880 -> 4960：%+.4f -> %+.4f" % (pa0, pa1))
        if pa1 > 0 > pa0:
            print("    -> 入边由负变正：去掉纵向差分足以解除这条边的折返。")
        elif pa1 < pa0:
            print("    -> 入边更负：原 compliance 项在缓解这条边的折返。")
        else:
            print("    -> 入边变化但不跨零，需看整体。")
        print("  出边 4960 -> 5040：%+.4f -> %+.4f" % (po0, po1))
        mx0 = summary["原 compliance"][0]
        mx1 = summary["compliance=0"][0]
        print("  过渡区最大转角：%.2f°（中点 %d, s=%.3f） -> %.2f°（中点 %d, s=%.3f）"
              % (mx0, summary["原 compliance"][1], summary["原 compliance"][2],
                 mx1, summary["compliance=0"][1], summary["compliance=0"][2]))
        print()
        if pa1 > 0 > pa0:
            if mx1 > mx0:
                print("  => 入边恢复正向，但整段最大转角上升（%.2f -> %.2f）。"
                      % (mx0, mx1))
                print("     按判据：保留原 compliance，用这次结果定位误差转移位置。")
            else:
                print("  => 入边恢复正向且整段最大转角下降，值得继续。")
        else:
            print("  => 去掉 compliance 不能解除入边折返；该项不是唯一责任方。")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
