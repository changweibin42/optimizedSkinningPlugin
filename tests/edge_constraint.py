# -*- coding: utf-8 -*-
r"""直接约束最终边的有符号投影（用户方案）。

未知量仍是每顶点的 CoR 中心偏移 Δc。

    q_ij = [(p_j0 - p_i0) + A_j·Δc_j - A_i·Δc_i] · t̂_ij / l_ij
    E_edge = μ Σ_ij k_ij l_ij² [max(0, ρ - q_ij)]²

梯度已用有限差分核对（tests/edge_grad_check.py，最大差 3.9e-08）：
    ∂E/∂Δc_b = -2μ k l v A_bᵀ t̂
    ∂E/∂Δc_a = +2μ k l v A_aᵀ t̂

两个必须处理的坑：
  1. q 的定义在 l→0 时发散（引导把两点压到一起）。按 l > 0.15·bind_len 过滤。
  2. 惩罚的 Hessian 必须是半正定，否则 PCG 发散。这里显式用 BᵀB 形式并自检。

    mayapy tests/edge_constraint.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402
import compliance_taper as tap  # noqa: E402
from edge_constraint_base import build_solver, solve_pcg  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
RHO = 0.1
MU_LIST = (0.3, 1.0, 3.0, 10.0, 30.0, 100.0)

KEY_EDGES = [("入边", 4880, 4960), ("出边", 4960, 5040),
             ("邻边1", 4720, 4800), ("邻边2", 4800, 4880)]


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

        ctx = build_solver(edges, original, W, quats, g_plain, outer)
        A = ctx["A"]
        a_e, b_e = ctx["a"], ctx["b"]
        n = len(original)

        # ---- 选边 ----
        st_a, st_b = bind[a_e, 0], bind[b_e, 0]
        mask = ((st_b - st_a) > 1e-6) & (np.abs(st_a - st_b) > 1e-6) & \
               (np.maximum(np.abs(st_a), np.abs(st_b)) < 3.0) & \
               ((st_a * st_b) > 0.0)
        bind_len = np.linalg.norm(bind[b_e] - bind[a_e], axis=1)
        dg_all = g_plain[b_e] - g_plain[a_e]
        l_all = np.linalg.norm(dg_all, axis=1)
        usable = l_all > 0.15 * np.maximum(bind_len, 1e-12)
        n_drop = int((mask & ~usable).sum())
        mask = mask & usable
        sel = np.nonzero(mask)[0]

        ea, eb = a_e[sel], b_e[sel]
        ek = ctx["k"][sel]
        dg = g_plain[eb] - g_plain[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        base = original[eb] - original[ea]
        q_base = np.einsum("ni,ni->n", base, et) / np.maximum(el, 1e-15)
        print("候选边 %d 条（丢弃 l<=0.15*bind_len 的 %d 条）" % (len(sel), n_drop))
        print("  l: min %.6f median %.6f max %.6f"
              % (el.min(), np.median(el), el.max()))
        print("  q_base: min %+.4f  median %+.4f  max %+.4f"
              % (q_base.min(), np.median(q_base), q_base.max()))

        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])
        all_s = stations[1:]
        reg = [i for i in range(len(all_s)) if 1.8 <= abs(all_s[i]) <= 3.0]
        mid = RINGS // 2

        def evaluate(final, d, label):
            pr = {t: tap.project_ratio(final, T_plain, p, q)
                  for t, p, q in KEY_EDGES}
            turn = gvf.signed_turning_2d(final[meridian])
            imax = max(reg, key=lambda i: abs(turn[i]))
            ell = np.linalg.norm(np.diff(final[meridian], axis=0), axis=1)
            ext = []
            for r in (mid - 1, mid, mid + 1):
                seg = np.arange(r * SIDES, (r + 1) * SIDES)
                yb = bind[seg, 1]
                for pick in (int(np.argmax(yb)), int(np.argmin(yb))):
                    ii = seg[pick]
                    ext.append(float(np.linalg.norm(final[ii] - original[ii])))
            return dict(label=label, pr=pr, mx=abs(turn[imax]),
                        mid=int(meridian[imax + 1]), at=all_s[imax],
                        el=float(min(ell[imax], ell[imax + 1])), ext=ext,
                        dmax=float(np.linalg.norm(d, axis=1).max()))

        def s_of(x):
            disp = np.einsum("nij,nj->ni", A, x)
            d_sp = (np.einsum("ni,ni->n", disp[eb], et) -
                    np.einsum("ni,ni->n", disp[ea], et)) / np.maximum(el, 1e-15)
            return d_sp, q_base + d_sp

        def pen_apply(x, mu_eff, v):
            """H_pen·x = BᵀB x，B 的第 e 行权 sqrt(2 μ k l v)/l。半正定。"""
            d_sp, _q = s_of(x)
            coef = 2.0 * mu_eff * ek * el * v * d_sp
            fb = np.einsum("nij,nj->ni", A[eb], et * coef[:, None])
            fa = np.einsum("nij,nj->ni", A[ea], et * coef[:, None])
            return np.stack([
                np.bincount(eb, weights=fb[:, c], minlength=n) -
                np.bincount(ea, weights=fa[:, c], minlength=n)
                for c in range(3)], axis=1)

        def pen_grad(x, mu_eff):
            _s, q = s_of(x)
            v = np.maximum(0.0, RHO - q)
            coef = 2.0 * mu_eff * ek * el * v
            gb = np.einsum("nij,nj->ni", A[eb], et * coef[:, None])
            ga = np.einsum("nij,nj->ni", A[ea], et * coef[:, None])
            return np.stack([
                np.bincount(eb, weights=-gb[:, c], minlength=n) +
                np.bincount(ea, weights=ga[:, c], minlength=n)
                for c in range(3)], axis=1)

        # ---- 半正定 + 对称性自检 ----
        rng = np.random.default_rng(0)
        v_test = np.ones(len(sel))
        minq, asym = np.inf, 0.0
        for _ in range(5):
            xt_ = rng.normal(size=(n, 3))
            yt_ = rng.normal(size=(n, 3))
            qv = float(np.sum(xt_ * pen_apply(yt_, 1.0, v_test)))
            minq = min(minq, qv)
            asym = max(asym, abs(qv - float(np.sum(yt_ * pen_apply(xt_, 1.0, v_test)))))
        print("  H_pen 自检：x·Hy 最小 %+.4e（应 >= 0）, 对称偏差 %.3e"
              % (minq, asym))

        # ---- 基线 ----
        x0 = solve_pcg(ctx, ctx["rhs0"], ctx["system0"])
        f0 = original + np.einsum("nij,nj->ni", A, x0)
        r0 = evaluate(f0, x0, "原版")
        _s0, q0 = s_of(x0)
        n0 = int((q0 < RHO).sum())

        # ---- taper 对照 ----
        gain = np.array([np.linalg.svd(np.asarray(A[i], float),
                                       compute_uv=False)[0] for i in range(n)])
        kern = tap.smoothstep(0.05, 0.60, gain / tap.G_REF)
        xt, infot = tap.structural_centers_taper(
            edges, original, W, quats, g_plain, outer,
            lambda_c=0.0, taper=0.4 + 0.6 * kern)
        ft = original + np.einsum("nij,nj->ni", infot["A"], xt)
        rt = evaluate(ft, xt, "taper floor=0.4")

        print("\n" + "=" * 108)
        print("边投影约束（原版引导 fillet=0，λ=0，无限幅，ρ=%.1f）" % RHO)
        print("=" * 108)
        print("%-18s %10s %10s %10s %10s %9s %8s %10s %8s"
              % ("配置", "入边4880→", "出边→5040", "邻1 4720→", "邻2 4800→",
                 "最大转角", "中点ID", "|delta|max", "违约边"))
        for r, nv in ((r0, n0), (rt, -1)):
            print("%-18s %10.4f %10.4f %10.4f %10.4f %9.2f %8d %10.2f %8s"
                  % (r["label"], r["pr"]["入边"][0], r["pr"]["出边"][0],
                     r["pr"]["邻边1"][0], r["pr"]["邻边2"][0],
                     r["mx"], r["mid"], r["dmax"],
                     str(nv) if nv >= 0 else "-"))

        scale_e = float(np.mean(ek * el * el))
        print("\n归一化尺度 mean(k l^2) = %.6e（mu 会除以它）" % scale_e)

        print("\n逐 μ（Gauss-Newton，最多 8 轮外层）：")
        rows = []
        for mu in MU_LIST:
            mu_eff = mu / max(scale_e, 1e-30)
            x = x0.copy()
            n_out = 0
            for outer_it in range(8):
                n_out = outer_it + 1
                _s, q = s_of(x)
                v = np.maximum(0.0, RHO - q)

                def system(xx, v=v, mu_eff=mu_eff):
                    return ctx["system0"](xx) + pen_apply(xx, mu_eff, v)

                rhs_adj = ctx["rhs0"] - pen_grad(x, mu_eff) + \
                    pen_apply(x, mu_eff, v)
                x_new = solve_pcg(ctx, rhs_adj, system)
                rel = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-12)
                x = x_new
                if rel < 1e-6:
                    break

            final = original + np.einsum("nij,nj->ni", A, x)
            r = evaluate(final, x, "mu=%.1f" % mu)
            _s, q = s_of(x)
            nv = int((q < RHO).sum())
            rows.append((r, nv))
            print("%-18s %10.4f %10.4f %10.4f %10.4f %9.2f %8d %10.2f %8d (%d轮)"
                  % (r["label"], r["pr"]["入边"][0], r["pr"]["出边"][0],
                     r["pr"]["邻边1"][0], r["pr"]["邻边2"][0],
                     r["mx"], r["mid"], r["dmax"], nv, n_out))
            print("     最短边长 %.5f  极点变化 %.4f"
                  % (r["el"], max(abs(u - z) for u, z in zip(r["ext"], r0["ext"]))))

        print("\n" + "=" * 108)
        print("判定")
        print("=" * 108)
        print("  原版：入边 %+.4f / 邻边1 %+.4f / 最大 %.2f° / 最短边 %.5f"
              % (r0["pr"]["入边"][0], r0["pr"]["邻边1"][0], r0["mx"], r0["el"]))
        print("  目标：入边与邻边1 同时 >0，4640 无新峰，最短边回升")
        for r, nv in rows:
            ok = (r["pr"]["入边"][0] > 0 and r["pr"]["邻边1"][0] > 0
                  and r["mx"] < r0["mx"])
            print("    %-10s 入边 %+.4f 邻边1 %+.4f 最大 %.2f° 最短边 %.5f  %s"
                  % (r["label"], r["pr"]["入边"][0], r["pr"]["邻边1"][0],
                     r["mx"], r["el"], "**达标**" if ok else ""))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
