# -*- coding: utf-8 -*-
r"""核验：固定 C=16，现有 2220 条候选边子集是否可行？

不做最小化，直接问可行性：
    存在 dc 使得  q_ij(dc) >= rho（这 2220 条），且 ||dc_i|| <= 16 ？

方法（不需要凸锥求解器）：对**单个约束**做上界估计。
    一条边 e 要满足 q_e >= rho，等价于 t̂ᵀ(A_b dc_b - A_a dc_a) >= d_e。
    若 ||dc|| <= C，则左边最大只能到 C·(||A_bᵀt̂|| + ||A_aᵀt̂||)。
    对任意一组边，把它们叠加时的上界也是可算的（用 Cauchy-Schwarz）。

更直接的做法：把 2220 条写成 B dc >= d，检查
    max_{||dc||<=C} min_e (B dc)_e
若这个最大值 < min d_e，则不可行。这是凹最大化，可用梯度上升近似，
上界用对偶：min_{lam>=0, ||Bᵀlam||<=1} lamᵀd 。

这里用后者的一个**可验证下界**：
    对任意 lam >= 0，若 ||Bᵀ lam|| <= 1/C，则 lamᵀd 是原问题最优值的下界
    （弱对偶）。

    mayapy tests/verify_c16.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402

HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
RHO = 0.1
C_BUDGET = 16.0


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
        g_plain = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
        A = ref.center_response(W, quats)
        a_e, b_e = edges[0], edges[1]
        n = len(original)

        st_a, st_b = bind[a_e, 0], bind[b_e, 0]
        cand = ((st_b - st_a) > 1e-6) & (np.abs(st_a - st_b) > 1e-6) & \
               (np.maximum(np.abs(st_a), np.abs(st_b)) < 3.0) & \
               ((st_a * st_b) > 0.0)
        bind_len = np.linalg.norm(bind[b_e] - bind[a_e], axis=1)
        l_all = np.linalg.norm(g_plain[b_e] - g_plain[a_e], axis=1)
        sel = np.nonzero(cand & (l_all > 0.15 * np.maximum(bind_len, 1e-12)))[0]

        ea, eb = a_e[sel], b_e[sel]
        dg = g_plain[eb] - g_plain[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        d_c = RHO * el - np.einsum("ni,ni->n", original[eb] - original[ea], et)

        # ---- 单位核对 ----
        radius = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
        print("=" * 86)
        print("单位核对")
        print("=" * 86)
        print("  夹具 radius = %.6f（绑定位 y,z 的最大模）" % radius)
        print("  bind 范围: x [%.3f, %.3f]  y [%.3f, %.3f]  z [%.3f, %.3f]"
              % (bind[:, 0].min(), bind[:, 0].max(),
                 bind[:, 1].min(), bind[:, 1].max(),
                 bind[:, 2].min(), bind[:, 2].max()))
        print("  reference 的 center_limit 是「世界单位 / radius」，生产 16.0 世界单位")
        print("  -> 归一化后 C = 16.0 / %.6f = %.4f" % (radius, 16.0 / radius))

        # ---- 2220 条子集（与上一轮相同的选取顺序）----
        order = np.argsort(-d_c)          # 违约量从大到小
        sub = order[:2220]
        print("\n子集：%d 条（按 d 降序取前 2220）" % len(sub))
        print("  d_e 范围: %.4f .. %.4f" % (d_c[sub].min(), d_c[sub].max()))

        sea, seb = ea[sub], eb[sub]
        set_, sel_ = et[sub], el[sub]
        sd = d_c[sub]

        # ---- 单约束上界（最严格的必要条件）----
        # q_e >= rho 需要  t̂ᵀ(A_b dc_b - A_a dc_a) >= d_e
        # 在 ||dc||<=C 下左边 <= C·|| [A_bᵀt̂ ; -A_aᵀt̂] || （两顶点合起的范数）
        Pa = np.einsum("ni,nij->nj", set_, A[seb])   # (m,3)
        Qa = np.einsum("ni,nij->nj", set_, A[sea])
        reach = C_BUDGET * np.sqrt(np.sum(Pa ** 2, axis=1) + np.sum(Qa ** 2, axis=1))
        feasible_single = reach >= sd
        print("\n单约束可达性（每个顶点预算 %.1f）：" % C_BUDGET)
        print("  可达上限 >= 需求 的边：%d / %d" % (int(feasible_single.sum()), len(sub)))
        bad = np.nonzero(~feasible_single)[0]
        if len(bad):
            print("  ** 有 %d 条边即使把全部预算用在它涉及的 2 个顶点上也做不到 **"
                  % len(bad))
            print("  %10s %10s %10s %12s %12s"
                  % ("i", "j", "station", "需要 d", "可达上限"))
            for t in bad[np.argsort(-sd[bad])][:8]:
                print("  %10d %10d %10.3f %12.5f %12.5f"
                      % (sea[t], seb[t], bind[sea[t], 0], sd[t], reach[t]))
        if len(bad) == 0:
            print("  单约束层面都可达 —— 需要看叠加后的联合可行性。")

        # ---- 联合不可行的直接证据：找一个必须被违反的约束组 ----
        # 若 v 个顶点的中心偏移被若干条边以相反方向拉，可用能量论证。
        # 这里用更直接的实测：在预算内做投影梯度，看能达到的最小 q。
        print("\n在 ||dc_i|| <= %.1f 约束下做投影梯度（最大化最小 q）：" % C_BUDGET)
        x = np.zeros((n, 3))
        lr = 0.05
        best_min_q = np.inf
        for step in range(600):
            r = np.einsum("ni,ni->n",
                          np.einsum("nij,nj->ni", A[seb], x[seb]) -
                          np.einsum("nij,nj->ni", A[sea], x[sea]), set_)
            q = (sd - r) / np.maximum(sel_, 1e-15)   # 需求缺口
            viol = np.maximum(0.0, q)                 # q>0 表示还没满足
            cur_min = float(np.min((r) / np.maximum(sel_, 1e-15)))
            best_min_q = max(best_min_q, cur_min) if step == 0 else best_min_q
            idx_v = np.nonzero(viol > 0)[0]
            if len(idx_v) == 0:
                print("  第 %d 步：全部满足" % step)
                break
            # 只推最违约的那批
            top = idx_v[np.argsort(-viol[idx_v])][:200]
            vec = set_[top] * viol[top, None]
            gj = np.einsum("nji,nj->ni", A[seb[top]], vec)
            gi = np.einsum("nji,nj->ni", A[sea[top]], vec)
            gr = np.stack([
                np.bincount(seb[top], weights=gj[:, c], minlength=n) -
                np.bincount(sea[top], weights=gi[:, c], minlength=n)
                for c in range(3)], axis=1)
            x = x + lr * gr
            nn = np.linalg.norm(x, axis=1, keepdims=True)
            x = x * np.minimum(1.0, C_BUDGET / np.maximum(nn, 1e-12))
            if step % 150 == 0:
                lr *= 0.6

        r = np.einsum("ni,ni->n",
                      np.einsum("nij,nj->ni", A[seb], x[seb]) -
                      np.einsum("nij,nj->ni", A[sea], x[sea]), set_)
        q_fin = r / np.maximum(sel_, 1e-15)
        nviol = int((q_fin < RHO).sum())
        Cused = float(np.linalg.norm(x, axis=1).max())
        print("  结果：C 用到 %.4f  仍违约 %d / %d 条  最小 q = %+.4f"
              % (Cused, nviol, len(sub), q_fin.min()))
        if nviol:
            print("  -> 在 C=%.0f 内无法让这个子集全部满足：生产预算下不可行。"
                  % C_BUDGET)
        else:
            print("  -> 在 C=%.0f 内可达（子集可行）。" % C_BUDGET)
        return 0
    finally:
        maya.standalone.uninitialize()


default_scale = 1.0

if __name__ == "__main__":
    sys.exit(main())
