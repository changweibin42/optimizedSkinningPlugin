# -*- coding: utf-8 -*-
r"""第 1 步：确认同一个顶点的滑动缺口。

用户指出的关键点：`0.8` 和 `0.040` 只有落在**同一点、同一单位**下才可直接比较。
之前我把 |guide|max（顶点 8880, s=+0.66）和 |delta|max（顶点 11320, s=+2.46）
混作一处，是错的。

这一版沿经线逐点记录：
  · guide 的纵向切向分量（沿经线单位切向 t_i 投影）
  · 该点实际纵向位移
  · A_i，以及沿 t_i 能产生的最大位移 C·||A_iᵀ t_i||
  · 与 C·σ_max(A_i) 的对比（后者不是这次的滑动方向）

同时记录 C=16 与 C=inf 两种情况下的角点预算，因为 109.78° 是关闭限幅的结果。

    mayapy tests/slide_gap.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402
import center_coupling as cc  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201
FILLET = 0.4       # 关掉层 1（形状折角），只看层 2
CENTER_LIMIT = 16.0


def build_scene(mc, om, ref):
    """建圆柱 + 插件，取 alg0 输出作为 original；返回 (bind, faces, original, W)。"""
    def tube(sides, rings, half):
        pts, faces = [], []
        for row in range(rings):
            s = -half + row * (2 * half / (rings - 1))
            for j in range(sides):
                a = 2 * math.pi * j / sides
                pts.append((s, math.cos(a), math.sin(a)))
        for row in range(rings - 1):
            for j in range(sides):
                p = row * sides + j
                q = row * sides + (j + 1) % sides
                faces.append((p, q, q + sides, p + sides))
        return pts, faces

    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
    bind_list, faces = tube(SIDES, RINGS, HALF)
    bind = np.asarray(bind_list, float)
    holder = mc.createNode("transform", name="h")
    geo = mc.createNode("transform", name="g", parent=holder)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    prev = mc.createNode("joint", name="pv", parent=holder)
    mc.setAttr(prev + ".translateX", -HALF)
    elbow = mc.createNode("joint", name="el", parent=prev)
    mc.setAttr(elbow + ".translateX", HALF)
    skin = mc.deformer(shape, type=SKIN, name="sk#")[0]
    for idx, jnt in enumerate((prev, elbow)):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, idx))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
    w1 = np.array([ref.skin_weight(p[0]) for p in bind])
    for i in range(len(bind)):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                   1.0 - w1[i], w1[i], size=2)
    s2 = om.MSelectionList()
    s2.add(elbow)
    om.MFnTransform(s2.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
    sl = om.MSelectionList()
    sl.add(shape)
    mc.setAttr(skin + ".algorithmType", 0)
    mc.refresh(force=True)
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    original = np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        sl.getDagPath(0)).getPoints(om.MSpace.kObject)])
    mc.delete(holder)
    return bind_list, faces, bind, original, w1


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind_list, faces, bind, original, w1 = build_scene(mc, om, ref)
        W = np.column_stack([1.0 - w1, w1])
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        guide = cc.guide_with_fillet(bind, original, W, FILLET)

        A = ref.center_response(W, quats)     # (n,3,3)
        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        ring_idx = [r * SIDES for r in range(RINGS)]
        meridian = np.asarray(ring_idx)

        # 求解（λ=0，按用户要求）
        delta, info = cc.structural_centers_coupled(
            edges, original, W, quats, guide, outer, center_limit=None)
        final = original + np.einsum("nij,nj->ni", A, delta)
        u = final - original                       # 实际顶点位移

        # 沿经线的单位切向 t_i（用最终位置的差分，中心差分）
        T = np.zeros((len(meridian), 3))
        for k in range(len(meridian)):
            lo = max(0, k - 1)
            hi = min(len(meridian) - 1, k + 1)
            v = final[meridian[hi]] - final[meridian[lo]]
            nv = np.linalg.norm(v)
            T[k] = v / nv if nv > 1e-12 else np.array([1.0, 0.0, 0.0])

        print("C = %.1f (生产限额)；λ=0；fillet=%.1f（层 1 已关）" %
              (CENTER_LIMIT, FILLET))
        print("|delta|max = %.3f   iters = %d\n"
              % (np.linalg.norm(delta, axis=1).max(), info["iters"]))

        print("%8s %8s %10s %10s %10s %10s %10s %10s"
              % ("s", "w1", "guide_t", "actual_t", "gap", "|delta|",
                 "C*|A^T*t|", "C*svmax"))
        rows = []
        for k in range(1, len(meridian) - 1):
            s = stations[k]
            if abs(s) > 3.0:
                continue
            i = int(meridian[k])
            t = T[k]
            gi = guide[i]
            ui = u[i]
            g_t = float(gi @ t)
            u_t = float(ui @ t)
            gap = g_t - u_t
            Ai = np.asarray(A[i], float)
            # 沿切向 t 能产生的最大位移：||Aᵀ t|| · C
            at = Ai.T @ t
            cap_t = CENTER_LIMIT * float(np.linalg.norm(at))
            sv = np.linalg.svd(Ai, compute_uv=False)
            cap_sv = CENTER_LIMIT * float(sv[0])
            rows.append((s, w1[i], g_t, u_t, gap,
                         float(np.linalg.norm(delta[i])), cap_t, cap_sv))
            print("%8.3f %8.4f %10.4f %10.4f %10.4f %10.3f %10.4f %10.4f"
                  % rows[-1])

        rows = np.asarray(rows)
        # s≈±2.28 处
        print("\n关键点位（|s| 在 2.2~2.4 之间）：")
        sel = (np.abs(rows[:, 0]) > 2.2) & (np.abs(rows[:, 0]) < 2.4)
        for r in rows[sel]:
            print("  s=%+.3f w1=%.4f  guide_t=%+.4f actual_t=%+.4f 缺口=%+.4f"
                  "  |delta|=%.2f  切向可产生 %.4f  (σmax 口径 %.4f)"
                  % (r[0], r[1], r[2], r[3], r[4], r[5], r[6], r[7]))

        print("\n同一里程碑判断（切向口径）：")
        worst = rows[np.argsort(-np.abs(rows[:, 4]))[:5]]
        print("  缺口最大的 5 个点：")
        for r in worst:
            status = "可实现" if abs(r[4]) <= r[6] + 1e-9 else "超出可实现上限"
            print("    s=%+.3f w1=%.4f 缺口 %+.4f  切向上限 %.4f  -> %s"
                  % (r[0], r[1], r[4], r[6], status))

        print("\n对 |delta| = 70 的分解：")
        big = np.argsort(-rows[:, 5])[0]
        r = rows[big]
        print("  |delta| 最大的经线点 s=%+.3f w1=%.4f" % (r[0], r[1]))
        print("    |delta| = %.2f  >> C = %.1f，说明关闭限幅时解远超预算"
              % (r[5], CENTER_LIMIT))

        # 102° 尖折处的 A 结构
        turns = cc.signed_turning(final[meridian])
        pos = stations[1:-1]
        imax = int(np.argmax(np.abs(turns)))
        s_peak = pos[imax]
        i_peak = int(meridian[imax + 1])
        print("\n尖折位置 s=%+.3f  转角 %+.2f°" % (s_peak, turns[imax]))
        print("  该点 w1=%.4f  |guide|=%.4f  |delta|=%.3f"
              % (w1[i_peak], float(np.linalg.norm(guide[i_peak])),
                 float(np.linalg.norm(delta[i_peak]))))
        sv = np.linalg.svd(np.asarray(A[i_peak], float), compute_uv=False)
        print("  该点 A 的奇异值: %s  （秩 %d）"
              % (np.array2string(sv, precision=5), int((sv > 1e-8).sum())))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
