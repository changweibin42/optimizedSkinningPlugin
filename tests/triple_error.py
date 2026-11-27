# -*- coding: utf-8 -*-
r"""查 s=-2.28 的相邻三点误差，定位 109.78° 是怎么在输出里形成的。

用产生该结果的同一次求值数据（λ=0，关闭限幅，fillet=0.4 的 guide）。

对相邻三点 i-1, i, i+1：
    e_i = final_i - T_fil_i            （求解器引入的误差）
    b-_final = b-_guide + (e_i - e_i-1)
    b+_final = b+_guide + (e_i+1 - e_i)

打印每条边的：引导边长、最终边长、边方向变化，
以及误差差分在引导边切向/法向上的投影。

    mayapy tests/triple_error.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import center_coupling as cc  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
FILLET = 0.4


def decompose(vec, tangent, normal):
    """把 vec 投影到 (tangent, normal) 上，返回 (t, n)。"""
    return float(vec @ tangent), float(vec @ normal)


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

        g_fil = cc.guide_with_fillet(bind, original, W, FILLET)
        T_fil = original + g_fil
        A = ref.center_response(W, quats)
        delta, info = cc.structural_centers_coupled(
            edges, original, W, quats, g_fil, outer, center_limit=None)
        final = original + np.einsum("nij,nj->ni", A, delta)

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])

        # 找最终网格的尖折位置
        turn = gvf.signed_turning_2d(final[meridian])
        kpeak = int(np.argmax(np.abs(turn)))
        s_peak = stations[kpeak + 1]
        print("最终网格最大 |转角| %.2f° @ s=%.3f（索引 k=%d）"
              % (abs(turn[kpeak]), s_peak, kpeak))

        # 三点：k, k+1, k+2 对应 station 索引
        # turn[k] 是 station[k+1] 处的转角，由点 k, k+1, k+2 构成
        i0, i1, i2 = meridian[kpeak], meridian[kpeak + 1], meridian[kpeak + 2]
        print("三点: station %+.3f / %+.3f / %+.3f"
              % (stations[kpeak], stations[kpeak + 1], stations[kpeak + 2]))

        e = final - T_fil
        print("\n误差 |e| = |final - T_guide|：")
        for tag, idx in (("i-1", i0), ("i", i1), ("i+1", i2)):
            print("  %-4s station %+.3f  |e| = %.5f  e = (%+.5f, %+.5f, %+.5f)"
                  % (tag, stations[[kpeak, kpeak + 1, kpeak + 2][
                      ("i-1", "i", "i+1").index(tag)]], np.linalg.norm(e[idx]),
                     e[idx][0], e[idx][1], e[idx][2]))

        print("\n两条边（负侧 b- = i-1 -> i，正侧 b+ = i -> i+1）：")
        print("%6s %12s %12s %12s %12s %10s"
              % ("边", "引导边长", "最终边长", "缩短量", "边方向变化°", "剩余比例"))
        for tag, (a, b) in (("b-", (i0, i1)), ("b+", (i1, i2))):
            bg = T_fil[b] - T_fil[a]
            bf = final[b] - final[a]
            lg = float(np.linalg.norm(bg))
            lf = float(np.linalg.norm(bf))
            # 边方向变化（两向量夹角）
            cosv = float(bg @ bf) / max(lg * lf, 1e-15)
            ang = math.degrees(math.acos(max(-1.0, min(1.0, cosv))))
            print("%6s %12.5f %12.5f %12.5f %12.2f %10.3f"
                  % (tag, lg, lf, lg - lf, ang, lf / max(lg, 1e-15)))

        # 误差差分分解
        print("\n误差差分 (e_b - e_a) 在引导边切向/法向上的投影：")
        print("%6s %10s %12s %12s %12s %12s"
              % ("边", "引导边长", "误差差分", "切向分量", "法向分量", "法向/边长"))
        for tag, (a, b) in (("b-", (i0, i1)), ("b+", (i1, i2))):
            bg = T_fil[b] - T_fil[a]
            lg = float(np.linalg.norm(bg))
            t_hat = bg / max(lg, 1e-15)
            # 弯面内法向：XY 平面里把切向转 90°
            n_hat = np.array([-t_hat[1], t_hat[0], 0.0])
            n_hat /= max(np.linalg.norm(n_hat), 1e-15)
            de = e[b] - e[a]
            dt, dn = decompose(de, t_hat, n_hat)
            print("%6s %10.5f %12.5f %12.5f %12.5f %12.3f"
                  % (tag, lg, float(np.linalg.norm(de)), dt, dn,
                     abs(dn) / max(lg, 1e-15)))

        # 判定
        print("\n判读：")
        bg_m = T_fil[i1] - T_fil[i0]
        bg_p = T_fil[i2] - T_fil[i1]
        lgm = float(np.linalg.norm(bg_m))
        lgp = float(np.linalg.norm(bg_p))
        dem = e[i1] - e[i0]
        dep = e[i2] - e[i1]
        tm = bg_m / max(lgm, 1e-15)
        tp = bg_p / max(lgp, 1e-15)
        dtm = float(dem @ tm)
        dtp = float(dep @ tp)
        nm = np.array([-tm[1], tm[0], 0.0])
        np_ = np.array([-tp[1], tp[0], 0.0])
        dnm = float(dem @ nm)
        dnp = float(dep @ np_)
        print("  b- 切向误差/边长 = %+.3f  （接近 -1 说明边在塌缩）" % (dtm / lgm))
        print("  b+ 切向误差/边长 = %+.3f" % (dtp / lgp))
        print("  b- 法向误差/边长 = %+.3f" % (dnm / lgm))
        print("  b+ 法向误差/边长 = %+.3f" % (dnp / lgp))
        if abs(dnm / lgm) > 0.5 or abs(dnp / lgp) > 0.5:
            print("  -> 法向误差主导：输出把边扭到了侧面")
            if dnm * dnp < 0:
                print("     且左右两侧法向误差反向 -> 集中成尖角")
        if dtm / lgm < -0.5 or dtp / lgp < -0.5:
            print("  -> 切向误差主导：边在塌缩，进一步缩短就会折返")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
