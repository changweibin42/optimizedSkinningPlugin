# -*- coding: utf-8 -*-
r"""可行性：让候选边全部 q >= rho，最少需要多大中心行程？

问题      min ||dc||   s.t.  B dc >= d   （B 是 m x 3n，每行 6 个非零元）

不用罚函数（它在本问题上病态到 1e15），改用**精确的最小范数解**：

    约束是线性的，等式情形下最小范数解为  dc = Bᵀ (B Bᵀ)^{-1} d
    不等式情形用「活跃集」迭代：先解等式，检查哪些约束仍违反，
    把它们加入等式集，重解。

因为 m ~ 7000 而已，B Bᵀ 是 m x m 稠密矩阵，内存会爆。
所以每轮只取**尚未满足的约束子集**（通常几十到几百条），
在子集上解等式最小范数问题，再全局检查。

    mayapy tests/center_feasibility.py
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
PROD_LIMIT = 16.0


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
        dg_all = g_plain[b_e] - g_plain[a_e]
        l_all = np.linalg.norm(dg_all, axis=1)
        sel = np.nonzero(cand & (l_all > 0.15 * np.maximum(bind_len, 1e-12)))[0]

        ea, eb = a_e[sel], b_e[sel]
        dg = g_plain[eb] - g_plain[ea]
        el = np.linalg.norm(dg, axis=1)
        et = dg / np.maximum(el, 1e-15)[:, None]
        proj_base = np.einsum("ni,ni->n", original[eb] - original[ea], et)
        q_base = proj_base / el
        d_c = RHO * el - proj_base          # 需要 B dc >= d_c

        print("候选边 %d 条" % len(sel))
        print("  q_base: min %+.3f  median %+.3f  max %+.3f"
              % (q_base.min(), np.median(q_base), q_base.max()))
        print("  初始违约（q_base < %.1f）: %d 条"
              % (RHO, int((d_c > 0).sum())))

        def row(e):
            """B 的第 e 行（稀疏表示）：返回 [(idx, vec3), ...] 与标量 d。"""
            return [(-1, ea[e]), (+1, eb[e])], d_c[e]

        def Bdense(idx):
            """对给定索引集构造稠密小矩阵 (len(idx), 3n) —— 只在子集上做。"""
            m = len(idx)
            Bs = np.zeros((m, 3 * n))
            for r, e in enumerate(idx):
                Bs[r, 3 * ea[e]:3 * ea[e] + 3] = -(et[e] @ A[ea[e]])
                Bs[r, 3 * eb[e]:3 * eb[e] + 3] = +(et[e] @ A[eb[e]])
            return Bs

        # ---- 活跃集迭代（稀疏 Gram，不构造稠密 B）----
        active = list(np.nonzero(d_c > 0)[0][:300])
        x = np.zeros((n, 3))
        print("\n活跃集迭代（每轮最多约 300 条，共 25 轮）：")
        print("%4s %8s %10s %12s %12s"
              % ("轮", "活跃", "新违约", "C=max|dc|", "关键边q"))
        keyidx = {}
        for tag, i, j in (("入边", 4880, 4960), ("邻边1", 4720, 4800)):
            m = np.nonzero((ea == i) & (eb == j))[0]
            if len(m):
                keyidx[tag] = int(m[0])

        def gram_and_row(idx):
            """正确算 G = B Bᵀ（m×m）与 P,Q。

            B 行 r 在 eb_r 处是 +p_r，在 ea_r 处是 -q_r（p,q 是 3 维）。
            G[r,s] = (+p_r)·(+p_s)·[eb_r==eb_s]
                   - (+p_r)·(-q_s)·[eb_r==ea_s]
                   - (-q_r)·(+p_s)·[ea_r==eb_s]
                   + (-q_r)·(-q_s)·[ea_r==ea_s]
            用掩码逐组累加，避免错误的整体矩阵乘。
            """
            m = len(idx)
            P = np.einsum("ni,nij->nj", et[idx], A[eb[idx]])   # (m,3)
            Q = np.einsum("ni,nij->nj", et[idx], A[ea[idx]])   # (m,3)
            G = np.zeros((m, m))
            ra = ea[idx]
            rb = eb[idx]

            def add(mask_r, mask_s, sign):
                # mask_r/mask_s 是 (m,m) 布尔
                if not mask_r.any():
                    return
                dotP = P @ P.T
                dotQ = Q @ Q.T
                dotPQ = P @ Q.T
                G[mask_r & mask_s] += sign * dotP[mask_r & mask_s]

            # 四种组合，全部用正确的向量对
            dPP = P @ P.T
            dQQ = Q @ Q.T
            dPQ = P @ Q.T
            dQP = Q @ P.T
            same_bb = rb[:, None] == rb[None, :]
            same_ba = rb[:, None] == ra[None, :]
            same_ab = ra[:, None] == rb[None, :]
            same_aa = ra[:, None] == ra[None, :]
            G += np.where(same_bb, dPP, 0.0)
            G -= np.where(same_ba, dPQ, 0.0)
            G -= np.where(same_ab, dQP, 0.0)
            G += np.where(same_aa, dQQ, 0.0)
            return G, P, Q

        for it in range(25):
            idx = np.array(sorted(set(active)), int)
            G, P, Q = gram_and_row(idx)
            dvec = d_c[idx]
            G += np.eye(len(idx)) * (1e-10 * np.trace(G) / max(len(idx), 1) + 1e-18)
            lam = np.linalg.solve(G, dvec)
            x = np.zeros((n, 3))
            contribB = np.einsum("n,nj->nj", lam, P)
            contribA = np.einsum("n,nj->nj", lam, Q)
            for c in range(3):
                x[:, c] = (np.bincount(eb[idx], weights=contribB[:, c], minlength=n) -
                           np.bincount(ea[idx], weights=contribA[:, c], minlength=n))

            r = np.einsum("ni,ni->n",
                          np.einsum("nij,nj->ni", A[eb], x[eb]) -
                          np.einsum("nij,nj->ni", A[ea], x[ea]), et)
            q_now = (proj_base + r) / el
            viol = np.nonzero(q_now < RHO)[0]
            Cmax = float(np.linalg.norm(x, axis=1).max())
            ks = " ".join("%s=%+.2f" % (t, q_now[keyidx[t]]) for t in keyidx)
            print("%4d %8d %10d %12.2f %12s"
                  % (it + 1, len(idx), len(viol), Cmax, ks))

            if len(viol) == 0:
                print("\n*** 第 %d 轮达到可行：所有候选边 q >= %.1f ***"
                      % (it + 1, RHO))
                break
            known = set(idx.tolist())
            newv = [int(v) for v in viol if int(v) not in known]
            active = idx.tolist() + newv[:80]

        # ---- 结论 ----
        r = np.einsum("ni,ni->n",
                      np.einsum("nij,nj->ni", A[eb], x[eb]) -
                      np.einsum("nij,nj->ni", A[ea], x[ea]), et)
        q_fin = (proj_base + r) / el
        nviol = int((q_fin < RHO).sum())
        Cmax = float(np.linalg.norm(x, axis=1).max())
        print("\n" + "=" * 88)
        print("结论")
        print("=" * 88)
        print("  候选 %d 条，违约 %d 条" % (len(sel), nviol))
        print("  最小中心行程 C = %.4f（生产限额 %.1f，比值 %.3f）"
              % (Cmax, PROD_LIMIT, Cmax / PROD_LIMIT))
        print("  最小 q = %+.4f（目标 >= %.1f）" % (q_fin.min(), RHO))
        for t in keyidx:
            print("  %-5s 基线 %+.4f -> %+.4f"
                  % (t, q_base[keyidx[t]], q_fin[keyidx[t]]))
        print()
        if nviol == 0:
            if Cmax <= PROD_LIMIT:
                print("  -> 在生产预算内可行。下一步在预算与边约束下恢复形状目标。")
            else:
                print("  -> 可行但需 %.1f 倍预算。固定预算后看哪些边必须放松。"
                      % (Cmax / PROD_LIMIT))
        else:
            print("  -> 活跃集迭代未收敛到可行。")
            bad = np.argsort(q_fin)[:6]
            print("     最差的边：")
            for t in bad:
                print("       %d->%d  station %+.3f  q=%+.3f  l=%.6f"
                      % (ea[t], eb[t], bind[ea[t], 0], q_fin[t], el[t]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
