# -*- coding: utf-8 -*-
r"""对拍：原版 hinge_displacement vs guide_with_fillet，以及最终网格。

按用户要求先做两件校正：

  1. 不让 guide_t 的翻号冒充「引导折返」。判据改成测 **T = original + guide
     的相邻边**：用沿 station 递增的同一弯曲平面内的有符号转角。
  2. 切线基准不在最终网格上取（那是用结果解释输入）。

流程：
  第 0 步  自检：fillet=0 的 guide_with_fillet 必须逐位复现 hinge_displacement。
           并确认两者都返回「相对位移」，所以 original 只加一次。
  第 1 步  三条线对拍：T_ref / T_fillet / final。
  第 2 步  按用户给的三分支判断落在哪一支。

    mayapy tests/guide_vs_fillet.py
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
FILLET = 0.4


def signed_turning_2d(pts):
    """pts 是沿 station 递增的 (n,3)。返回长度 n-2 的有符号转角（度），弯面 XY。"""
    u = pts[1:-1] - pts[:-2]
    v = pts[2:] - pts[1:-1]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def edge_lengths(pts):
    return np.linalg.norm(np.diff(pts, axis=0), axis=1)


def build_scene(mc, om):
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

        bind_list, faces, bind, original, w1 = build_scene(mc, om)
        W = np.column_stack([1.0 - w1, w1])

        # ---------- 第 0 步：自检 ----------
        print("=" * 78)
        print("第 0 步  自检：guide_with_fillet(fillet=0) 是否逐位复现原版")
        print("=" * 78)
        g_ref = np.asarray(ref.hinge_displacement(bind, original, W, BEND, 0.0, 0.0))
        g_zero = cc.guide_with_fillet(bind, original, W, 0.0)
        d0 = float(np.linalg.norm(g_ref - g_zero, axis=1).max())
        print("  两者最大差: %.3e   (%s)"
              % (d0, "逐位一致" if d0 < 1e-12 else "**不一致**"))
        print("  两者都是相对位移（target - original）：")
        print("    |g_ref + original - original - g_ref| = %.3e"
              % float(np.linalg.norm((original + g_ref) - original - g_ref, axis=1).max()))

        g_fil = cc.guide_with_fillet(bind, original, W, FILLET)
        print("  fillet=%.1f 与原版引导的差异: |g| 最大差 %.4f"
              % (FILLET, float(np.linalg.norm(g_ref - g_fil, axis=1).max())))

        T_ref = original + g_ref
        T_fil = original + g_fil

        # ---------- 第 1 步：求解，得最终网格 ----------
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        A = ref.center_response(W, quats)
        delta, info = cc.structural_centers_coupled(
            edges, original, W, quats, g_fil, outer, center_limit=None)
        final = original + np.einsum("nij,nj->ni", A, delta)
        print("\n  求解：|delta|max %.3f  iters %d" %
              (np.linalg.norm(delta, axis=1).max(), info["iters"]))

        # ---------- 第 2 步：三条线的对拍表 ----------
        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        meridian = np.array([r * SIDES for r in range(RINGS)])

        def table(line, label, s_min, s_max):
            pts = line[meridian]
            turn = signed_turning_2d(pts)          # 长度 RINGS-2，对应 station[1..-2]
            el = edge_lengths(pts)                 # 长度 RINGS-1，对应 edge k->k+1
            sel = [k for k in range(len(turn))
                   if s_min <= stations[k + 1] <= s_max]
            print("\n%s   最大 |转角| %.2f° @ s=%.3f"
                  % (label, abs(turn[np.argmax(np.abs(turn))]),
                     stations[1:][np.argmax(np.abs(turn))]))
            print("%8s %10s %10s %10s" % ("s", "转角(deg)", "左边长", "右边长"))
            for k in sel:
                # 转角在 station[k+1]，左边长是 el[k]，右边长是 el[k+1]
                print("%8.3f %10.2f %10.5f %10.5f"
                      % (stations[k + 1], turn[k], el[k], el[k + 1]))
            return turn, el

        print("\n" + "=" * 78)
        print("第 1 步  三条线对拍（同一 batch 顶点，弯面 XY 有符号转角）")
        print("=" * 78)
        for lo, hi, tag in ((2.16, 2.58, "正侧"), (-2.58, -2.16, "负侧")):
            print("\n" + "#" * 78)
            print("# 区间 %s: s ∈ [%.2f, %.2f]" % (tag, lo, hi))
            print("#" * 78)
            t1, e1 = table(T_ref, "原版引导 T_ref = original + hinge_displacement",
                           lo, hi)
            t2, e2 = table(T_fil, "实验引导 T_fil = original + guide_with_fillet",
                           lo, hi)
            t3, e3 = table(final, "实验最终 final", lo, hi)

            # 合并成一张紧凑表
            print("\n>>> 合并表（%s）" % tag)
            print("%8s %12s %12s %12s %10s %10s"
                  % ("s", "原版引导", "fillet引导", "最终", "左边长", "右边长"))
            s_each = stations[1:]
            for k in range(len(s_each)):
                if not (lo <= s_each[k] <= hi):
                    continue
                print("%8.3f %12.2f %12.2f %12.2f %10.5f %10.5f"
                      % (s_each[k], t1[k], t2[k], t3[k], e3[k], e3[k + 1]))

        # ---------- 第 3 步：判读 ----------
        print("\n" + "=" * 78)
        print("第 2 步  判读")
        print("=" * 78)
        all_s = stations[1:]
        t1 = signed_turning_2d(T_ref[meridian])
        t2 = signed_turning_2d(T_fil[meridian])
        t3 = signed_turning_2d(final[meridian])
        m1 = abs(t1[np.argmax(np.abs(t1))])
        m2 = abs(t2[np.argmax(np.abs(t2))])
        m3 = abs(t3[np.argmax(np.abs(t3))])
        p1 = all_s[np.argmax(np.abs(t1))]
        p2 = all_s[np.argmax(np.abs(t2))]
        p3 = all_s[np.argmax(np.abs(t3))]
        print("  原版引导  最大 %.2f° @ s=%.3f" % (m1, p1))
        print("  fillet引导最大 %.2f° @ s=%.3f" % (m2, p2))
        print("  最终      最大 %.2f° @ s=%.3f" % (m3, p3))
        print()
        if m1 < 20.0 and m2 >= 20.0:
            print("  -> 分支 1：原版平顺、fillet 版已尖折。查 guide_with_fillet。")
        elif m1 >= 20.0 and m2 >= 20.0:
            print("  -> 分支 2：两版引导都在同一位置尖折。问题已在几何目标层，")
            print("     需回查该侧轨道的相邻采样点与材料顺序。")
        elif m1 < 20.0 and m2 < 20.0:
            print("  -> 分支 3：两版引导都平顺，只有最终网格尖折。")
            print("     这才满足「转查结构项」的判据。")
        else:
            print("  -> 混合：原版尖折但 fillet 版更平，需单独分析。")
        print("  从引导到最终的增量：%.2f°（fillet 引导） / %.2f°（原版引导）"
              % (m3 - m2, m3 - m1))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
