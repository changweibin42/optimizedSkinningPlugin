# -*- coding: utf-8 -*-
r"""单片参照：A/B 谁更接近「无跨片串扰」的正确位置。

方法（用户指定）：
  保持骨骼矩阵、绑定点空间位置、蒙皮权重、corSigma、三角化一致，
  让每片**分别单独**参与原生 alg0 的中心积分（保留原顶点映射，不移到原点）。

  参照 = 两片各自独立求出的结果拼回同一顶点数组。

对比：
  · 基础中心：A、B 相对参照的中心误差，重点看片间 X 方向
  · 最终顶点：A、B 相对参照的平均 / 最大偏差
  · 两片间距：z=-0.4 与 z=+0.4，比较 A、B 与参照的间距

    mayapy tests/curveg_plate_ref.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

SKIN = "optimizedSkinCluster"
BEND = 90.0
GAP = 0.8
LENGTH = 4.0
WIDTH = 0.5
NX, NZ = 5, 21


def build_plates(mc, om, plates):
    """建 plates 指定的窄条（0=左，1=右），返回 bind/faces/plateRanges。

    plates 是参与本次构建的片号列表。片的 X 偏移始终按原始 GAP 计算，
    所以「单独构建某一片」时它的空间位置与两片共存时完全一样。
    """
    pts, faces = [], []
    ranges = {}
    for plate in plates:
        xoff = (-GAP * 0.5) if plate == 0 else (+GAP * 0.5)
        base = len(pts)
        for iz in range(NZ):
            z = -LENGTH * 0.5 + iz * LENGTH / (NZ - 1)
            for ix in range(NX):
                x = xoff + (ix - (NX - 1) * 0.5) * WIDTH / (NX - 1)
                pts.append((x, 0.0, z))
        for iz in range(NZ - 1):
            for ix in range(NX - 1):
                a = base + iz * NX + ix
                faces.append((a, a + 1, a + NX + 1, a + NX))
        ranges[plate] = (base, len(pts))
    return np.asarray(pts, float), faces, ranges, len(pts)


def make_scene(mc, om, label):
    holder = mc.createNode("transform", name="%s_holder" % label)
    geo = mc.createNode("transform", name="%s_geo" % label, parent=holder)
    root = mc.createNode("joint", name="%s_root" % label, parent=holder)
    mc.setAttr(root + ".translateZ", -LENGTH * 0.5)
    tip = mc.createNode("joint", name="%s_tip" % label, parent=root)
    mc.setAttr(tip + ".translateZ", LENGTH)
    return holder, geo, root, tip


def read(mc, om, shape):
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    sl = om.MSelectionList()
    sl.add(shape)
    return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        sl.getDagPath(0)).getPoints(om.MSpace.kObject)])


def run_case(mc, om, plates, gate, crossOnly, label):
    """建一个只含 plates 的场景，跑 alg0，返回顶点数组与权重。"""
    bind, faces, ranges, total = build_plates(mc, om, plates)
    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
    holder, geo, root, tip = make_scene(mc, om, label)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [k for f in faces for k in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    skin = mc.deformer(shape, type=SKIN, name="%s_sk" % label)[0]
    for idx, jnt in enumerate((root, tip)):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, idx))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
    zn = bind[:, 2] / (LENGTH * 0.5)
    w1 = np.array([ref.skin_weight(v * 6.0) for v in zn])
    for i in range(len(bind)):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                   1.0 - w1[i], w1[i], size=2)
    # 手工归属：按片号
    assign = []
    for plate in plates:
        base, end = ranges[plate]
        assign.extend([float(plate)] * (end - base))
    mc.setAttr(skin + ".corCurveAssign", assign, type="doubleArray")

    s = om.MSelectionList()
    s.add(tip)
    om.MFnTransform(s.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)

    mc.setAttr(skin + ".algorithmType", 0)
    mc.setAttr(skin + ".corSigma", 0.1)
    mc.setAttr(skin + ".corCurveGate", 1.0 if gate else 0.0)
    mc.setAttr(skin + ".corCurveCrossOnly", bool(crossOnly))
    mc.refresh(force=True)
    P = read(mc, om, shape)
    return P, bind, ranges, w1


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        # ---- 两片共存：A（原生）、B（跨片衰减）----
        PA, bindA, rangesA, wA = run_case(mc, om, [0, 1], gate=False,
                                          crossOnly=False, label="bothA")
        PB, _, _, _ = run_case(mc, om, [0, 1], gate=True, crossOnly=True,
                               label="bothB")

        # ---- 单片参照：每片单独构建，空间位置不变 ----
        P0, bind0, ranges0, w0 = run_case(mc, om, [0], gate=False,
                                          crossOnly=False, label="only0")
        P1, bind1, ranges1, w1s = run_case(mc, om, [1], gate=False,
                                           crossOnly=False, label="only1")

        # 拼回同一顶点数组（两片共存时的顺序：片0 在前，片1 在后）
        n0 = NX * NZ
        n1 = NX * NZ
        assert len(PA) == n0 + n1, (len(PA), n0 + n1)
        assert len(P0) == n0 and len(P1) == n1, (len(P0), len(P1))

        # 两片共存与单片的绑定顺序一致？
        same_bind0 = np.allclose(bindA[:n0], bind0)
        same_bind1 = np.allclose(bindA[n0:], bind1)
        print("绑定位置核对：片0 一致 %s  片1 一致 %s" % (same_bind0, same_bind1))
        print("权重核对：片0 max 差 %.3e  片1 max 差 %.3e"
              % (np.abs(wA[:n0] - w0).max(), np.abs(wA[n0:] - w1s).max()))

        PREF = np.vstack([P0, P1])

        print("\n" + "=" * 76)
        print("最终顶点相对「单片参照」的偏差")
        print("=" * 76)
        print("%-6s %14s %14s %14s %14s"
              % ("", "平均偏差", "最大偏差", "片0 最大", "片1 最大"))
        for tag, P in (("A 原生", PA), ("B 衰减", PB)):
            d = np.linalg.norm(P - PREF, axis=1)
            print("%-6s %14.6e %14.6e %14.6e %14.6e"
                  % (tag, d.mean(), d.max(), d[:n0].max(), d[n0:].max()))

        # ---- 片间间距 ----
        print("\n" + "=" * 76)
        print("片间间距（同一 Z 处，片1 的 X 减片0 的 X）")
        print("=" * 76)
        print("%8s %12s %12s %12s %14s %14s"
              % ("z", "参照", "A", "B", "A-参照", "B-参照"))
        for iz in range(0, NZ, 4):
            i0 = iz * NX + NX // 2
            i1 = n0 + iz * NX + NX // 2
            g_ref = PREF[i1][0] - PREF[i0][0]
            g_a = PA[i1][0] - PA[i0][0]
            g_b = PB[i1][0] - PB[i0][0]
            print("%8.3f %12.6f %12.6f %12.6f %14.6f %14.6f"
                  % (bind0[i0][2], g_ref, g_a, g_b, g_a - g_ref, g_b - g_ref))

        # ---- 片间 X 方向分量 ----
        print("\n" + "=" * 76)
        print("片间（X）方向的偏差分量")
        print("=" * 76)
        for tag, P in (("A 原生", PA), ("B 衰减", PB)):
            dx = P[:, 0] - PREF[:, 0]
            print("  %-6s X 方向：mean %+.6e  max|.| %.6e  片0 mean %+.6e  片1 mean %+.6e"
                  % (tag, dx.mean(), np.abs(dx).max(),
                     dx[:n0].mean(), dx[n0:].mean()))

        print("\n" + "=" * 76)
        print("判定")
        print("=" * 76)
        dA = np.linalg.norm(PA - PREF, axis=1)
        dB = np.linalg.norm(PB - PREF, axis=1)
        dXA = np.abs(PA[:, 0] - PREF[:, 0])
        dXB = np.abs(PB[:, 0] - PREF[:, 0])
        print("  平均偏差：A %.6e  B %.6e  ->  %s 更接近参照"
              % (dA.mean(), dB.mean(), "B" if dB.mean() < dA.mean() else "A"))
        print("  片间分量：A %.6e  B %.6e  ->  %s 更接近参照"
              % (dXA.mean(), dXB.mean(), "B" if dXB.mean() < dXA.mean() else "A"))
        print()
        if dB.max() < 1e-6:
            print("  B 在容差内复现单片参照 -> 跨片样本被完全屏蔽。")
        elif dB.mean() < dA.mean():
            print("  B 比 A 更接近参照，但未完全复现 -> 部分衰减在起作用。")
        else:
            print("  B 没有比 A 更接近参照。")
            print("  注意：也可能是我的「跨片」判定与参照构造不一致，需查归属。")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
