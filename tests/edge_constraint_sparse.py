# -*- coding: utf-8 -*-
r"""边投影约束：稀疏 B（不构造矩阵），按用户给的规范化形式。

一条活跃边 i->j：
    B_ij = sqrt(k) [ -t̂^T A_i ,  t̂^T A_j ]
    d_ij = sqrt(k) [ rho·l - t̂^T (p_j^0 - p_i^0) ]
于是 (Bx - d)_ij = sqrt(k)·l·(q_ij - rho)，
而 mu(Bx-d)² = mu k l² (q-rho)²，与要求一致（长度因子在组装时已约掉）。

正规方程：算子 += mu B^TB，右端 += mu B^Td。

B 每行 6 个非零元，用 einsum 直接算，不物化矩阵。

    mayapy tests/edge_constraint_sparse.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import guide_vs_fillet as gvf  # noqa: E402
import compliance_taper as tap  # noqa: E402
from edge_constraint_base import build_solver, solve_pcg  # noqa: E402
from btb_check import build_B  # noqa: E402

HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
RHO = 0.1
W_LIST = (1.0, 10.0, 100.0)

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

        # ---------- 选边 ----------
        st_a, st_b = bind[a_e, 0], bind[b_e, 0]
        cand_mask = ((st_b - st_a) > 1e-6) & (np.abs(st_a - st_b) > 1e-6) & \
                    (np.maximum(np.abs(st_a), np.abs(st_b)) < 3.0) & \
                    ((st_a * st_b) > 0.0)
        bind_len = np.linalg.norm(bind[b_e] - bind[a_e], axis=1)
        dg_all = g_plain[b_e] - g_plain[a_e]
        l_all = np.linalg.norm(dg_all, axis=1)
        usable = l_all > 0.15 * np.maximum(bind_len, 1e-12)
        n_short = int((cand_mask & ~usable).sum())
        sel = np.nonzero(cand_mask & usable)[0]

        ea, eb = a_e[sel], b_e[sel]
        ek = ctx["k"][sel]
        dg = g_plain[eb] - g_plain[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        base = original[eb] - original[ea]
        proj_base = np.einsum("ni,ni->n", base, et)
        q_base = proj_base / np.maximum(el, 1e-15)

        # 目标 E = mu k l^2 (q - rho)^2，其中 q = (proj_base + t̂ᵀAΔx) / l。
        # 写成 ||Bx - d||^2 时，B 必须带 sqrt(k)·l（不是 sqrt(k)），否则
        # 少的那个 l^2 会让 w 的量纲整体偏掉。
        #   B_ij = sqrt(k)·l · [ -t̂ᵀA_i , t̂ᵀA_j ] / l ... 等价于 sqrt(k)·l/l
        # 为清楚起见取  Bx_e = sqrt(k)·l·(q_e) 的变部分 = sqrt(k)·l·(t̂ᵀAΔx)/l
        #                                     = sqrt(k)·t̂ᵀAΔx
        # 于是 B 行 = sqrt(k)·[ -t̂ᵀA_i, t̂ᵀA_j ]（不带 1/l，也不带 l），
        # 而 d_e = sqrt(k)·l·(rho - q_base) 。这样
        #   Bx - d = sqrt(k)·l·(q - rho)   ✓ 与 mu k l^2 (q-rho)^2 一致
        sk = np.sqrt(ek)
        d_vec = sk * el * (RHO - q_base)

        print("候选 %d 条；因 l<=0.15*bind_len 过滤掉 %d 条（这些仍纳入最终经线检查）"
              % (len(sel), n_short))
        print("  l: %.6f/%.6f/%.6f   q_base: %+.3f/%+.3f/%+.3f"
              % (el.min(), np.median(el), el.max(),
                 q_base.min(), np.median(q_base), q_base.max()))
        print("  d 范围 %+.4f..%+.4f（>0 表示初始违约）"
              % (d_vec.min(), d_vec.max()))

        # ---------- 稀疏 Bx / B^Ty（每条边 6 个非零元，不物化矩阵） ----------
        def Bx(x, ii, jj, tt, ss):
            """(Bx)_e = sqrt(k_e) · t̂^T (A_j x_j - A_i x_i)"""
            disp_i = np.einsum("nij,nj->ni", A[ii], x[ii])
            disp_j = np.einsum("nij,nj->ni", A[jj], x[jj])
            return ss * (np.einsum("ni,ni->n", disp_j, tt) -
                         np.einsum("ni,ni->n", disp_i, tt))

        def BTy(y, mu, ii, jj, tt, ss):
            """(B^T y)_i = -mu*sqrt(k) A_i^T t y ;  (B^T y)_j = +mu*sqrt(k) A_j^T t y

            必须用 A^T（nji），不是 A（nij）：B 的行是 t^T A，转置后是 A^T t。
            用错时顶点同时在 ea 与 eb 会出现 5e-3 量级的差。
            """
            vec = tt * (mu * ss * y)[:, None]
            gj = np.einsum("nji,nj->ni", A[jj], vec)
            gi = np.einsum("nji,nj->ni", A[ii], vec)
            return np.stack([
                np.bincount(jj, weights=gj[:, c], minlength=n) -
                np.bincount(ii, weights=gi[:, c], minlength=n)
                for c in range(3)], axis=1)

        # ---------- 对拍：三条边 vs 显式 B ----------
        print("\n" + "=" * 96)
        print("对拍：稀疏 Bx/B^Ty vs 显式 B（三条边）")
        print("=" * 96)
        ids = [(4880, 4960), (4720, 4800), (4800, 4880)]
        sel3 = np.array([np.nonzero((a_e == i) & (b_e == j))[0][0] for i, j in ids])
        ea3, eb3 = a_e[sel3], b_e[sel3]
        dg3 = g_plain[eb3] - g_plain[ea3]
        el3 = np.linalg.norm(dg3, axis=1)
        et3 = dg3 / el3[:, None]
        sk3 = np.sqrt(ctx["k"][sel3])
        B3 = build_B(A, ea3, eb3, et3, el3, n, sqrt_k=sk3)

        rng = np.random.default_rng(7)
        xt = rng.normal(size=(n, 3))
        yt = rng.normal(size=len(sel3))

        bx_sparse = Bx(xt, ea3, eb3, et3, sk3)
        bx_dense = B3 @ xt.reshape(-1)
        print("  Bx：稀疏 vs 显式 最大差 %.3e" % np.abs(bx_sparse - bx_dense).max())

        bty_sparse = BTy(yt, 1.0, ea3, eb3, et3, sk3)
        bty_dense = (B3.T @ yt).reshape(n, 3)
        print("  B^Ty：稀疏 vs 显式 最大差 %.3e"
              % np.abs(bty_sparse - bty_dense).max())

        # ---------- 基线与 taper 对照 ----------
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

        x0 = solve_pcg(ctx, ctx["rhs0"], ctx["system0"])
        f0 = original + np.einsum("nij,nj->ni", A, x0)
        r0 = evaluate(f0, x0, "原版")

        print("\n" + "=" * 100)
        print("完整边约束（fillet=0，lam=0，无限幅，rho=%.1f）" % RHO)
        print("=" * 100)
        print("%-22s %9s %9s %9s %9s %8s %7s %8s %7s %6s"
              % ("配置", "入边4880→", "出边→5040", "邻1 4720→", "邻2 4800→",
                 "最大转角", "峰值ID", "|dc|max", "最短边", "违约"))
        def show(r, nv):
            print("%-22s %9.4f %9.4f %9.4f %9.4f %8.2f %7d %8.2f %7.5f %6s"
                  % (r["label"], r["pr"]["入边"][0], r["pr"]["出边"][0],
                     r["pr"]["邻边1"][0], r["pr"]["邻边2"][0],
                     r["mx"], r["mid"], r["dmax"], r["el"],
                     str(nv) if nv is not None else "-"))
        _qb0 = q_base
        nv0 = int((q_base + 0.0 < RHO).sum())
        show(r0, int((q_base < RHO).sum()))

        # ---------- 固定点迭代 ----------
        print("\n逐 w（每轮重扫整个过渡区，直到活跃集稳定，最多 12 轮）：")
        rows = []
        for w in W_LIST:
            x = x0.copy()
            prev_active = None
            n_out = 0
            for outer in range(12):
                n_out = outer + 1
                resid = Bx(x, ea, eb, et, sk) - d_vec   # sqrt(k) l (q-rho)
                active = resid > 0.0                    # 违约：q < rho
                idx_act = np.nonzero(active)[0]
                if len(idx_act) == 0:
                    break
                if prev_active is not None and np.array_equal(active, prev_active):
                    break
                prev_active = active

                ia, ib = ea[idx_act], eb[idx_act]
                tt, ss = et[idx_act], sk[idx_act]
                dd = d_vec[idx_act]

                def system(xx, ia=ia, ib=ib, tt=tt, ss=ss, mm=w):
                    return ctx["system0"](xx) + BTy(Bx(xx, ia, ib, tt, ss),
                                                    mm, ia, ib, tt, ss)

                rhs_adj = ctx["rhs0"] + BTy(dd, w, ia, ib, tt, ss)
                x_new = solve_pcg(ctx, rhs_adj, system, max_iters=300)
                rel = np.linalg.norm(x_new - x) / max(np.linalg.norm(x), 1e-12)
                x = x_new
                if rel < 1e-7:
                    break

            final = original + np.einsum("nij,nj->ni", A, x)
            r = evaluate(final, x, "w=%.0f" % w)
            resid = Bx(x, ea, eb, et, sk) - d_vec
            # resid = sqrt(k) l (q - rho)  =>  q = rho + resid/(sqrt(k) l)
            q_now = RHO + resid / np.maximum(sk * el, 1e-15)
            nv = int((q_now < RHO).sum())
            rows.append((r, nv, q_now, n_out))
            show(r, nv)
            print("      最小投影比 %+.4f  过滤短边 %d 条  外层次数 %d"
                  % (q_now.min(), n_short, n_out))
            print("      极点变化 %.4f  （相对原版）"
                  % max(abs(u - z) for u, z in zip(r["ext"], r0["ext"])))

        print("\n" + "=" * 100)
        print("判定")
        print("=" * 100)
        print("  原版：入边 %+.4f  邻1 %+.4f  最大 %.2f°  最短边 %.5f"
              % (r0["pr"]["入边"][0], r0["pr"]["邻边1"][0], r0["mx"], r0["el"]))
        for r, nv, q_now, n_out in rows:
            ok = (r["pr"]["入边"][0] > 0 and r["pr"]["邻边1"][0] > 0
                  and r["mx"] < r0["mx"] and r["el"] > r0["el"])
            print("    %-8s 入边 %+.4f  邻1 %+.4f  最大 %.2f°  最短边 %.5f  %s"
                  % (r["label"], r["pr"]["入边"][0], r["pr"]["邻边1"][0],
                     r["mx"], r["el"], "**达标**" if ok else ""))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
