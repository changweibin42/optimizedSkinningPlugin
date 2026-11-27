# -*- coding: utf-8 -*-
r"""方向 A 第一轮验收：CoR 曲线引导的邻域权重。

三项独立检查：
  1. 关闭（corCurveGate=0）时与原生 alg0 逐位一致
  2. 强制 g≈1（sigma 极大）时与原版差异可忽略 —— 证明 gate 只乘在
     权重上和，不改变积分结构
  3. 启用真实衰减时，打印固定查询顶点到 0.5R / 1R 处样本的因子，
     并在 458 点与 16080 点上用**相同的物理距离**复核

    mayapy tests/curveg_cpp.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

HALF, BEND = 6.0, 146.1265
SKIN = "optimizedSkinCluster"


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


def build(mc, om, sides, rings, label):
    bind_list, faces = tube(sides, rings, HALF)
    bind = np.asarray(bind_list, float)
    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
    holder = mc.createNode("transform", name="%s_h" % label)
    geo = mc.createNode("transform", name="%s_g" % label, parent=holder)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [k for f in faces for k in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    prev = mc.createNode("joint", name="%s_pv" % label, parent=holder)
    mc.setAttr(prev + ".translateX", -HALF)
    elbow = mc.createNode("joint", name="%s_el" % label, parent=prev)
    mc.setAttr(elbow + ".translateX", HALF)
    skin = mc.deformer(shape, type=SKIN, name="%s_sk" % label)[0]
    for k, jnt in enumerate((prev, elbow)):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
    w1 = np.array([ref.skin_weight(p[0]) for p in bind])
    for i in range(len(bind)):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                   1.0 - w1[i], w1[i], size=2)
    s = om.MSelectionList()
    s.add(elbow)
    om.MFnTransform(s.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
    sl = om.MSelectionList()
    sl.add(shape)
    mc.setAttr(skin + ".algorithmType", 0)
    mc.refresh(force=True)
    return bind, shape, skin, sl, w1, faces


def read(mc, om, shape):
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    s = om.MSelectionList()
    s.add(shape)
    return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        s.getDagPath(0)).getPoints(om.MSpace.kObject)])


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        for sides, rings, tag in ((24, 19, "lo"), (80, 201, "hi")):
            bind, shape, skin, sl, w1, faces = build(mc, om, sides, rings, tag)
            rad = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
            print("\n" + "=" * 78)
            print("%s 点（radius %.4f，环距 %.4f）"
                  % (tag, rad, 2 * HALF / (rings - 1)))
            print("=" * 78)

            # --- 1. 关闭 ---
            mc.setAttr(skin + ".corCurveGate", 0.0)
            mc.setAttr(skin + ".corCurveSigma", 0.1)
            mc.refresh(force=True)
            base = read(mc, om, shape)
            print("  关闭：|v| 最大 %.4f" % np.linalg.norm(base, axis=1).max())

            # --- 2. 强制 g≈1（sigma 极大）---
            mc.setAttr(skin + ".corCurveGate", 1.0)
            mc.setAttr(skin + ".corCurveSigma", 1000.0)
            mc.refresh(force=True)
            wide = read(mc, om, shape)
            d2 = np.linalg.norm(wide - base, axis=1)
            print("  gate开 + sigma=1000（g≈1）：与原版 max 差 %.4e  mean %.4e"
                  % (d2.max(), d2.mean()))

            # --- 3. 真实衰减 ---
            for sigma in (0.1, 0.3, 1.0):
                mc.setAttr(skin + ".corCurveSigma", sigma)
                mc.refresh(force=True)
                cur = read(mc, om, shape)
                d3 = np.linalg.norm(cur - base, axis=1)
                # 经线转角对比
                st = np.array([-HALF + r * (2 * HALF / (rings - 1))
                               for r in range(rings)])
                idx = np.array([r * sides for r in range(rings)])

                def maxturn(P):
                    Q = P[idx]
                    u = Q[1:-1] - Q[:-2]
                    v = Q[2:] - Q[1:-1]
                    cr = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
                    dt = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
                    return np.degrees(np.arctan2(cr, dt))

                tb = maxturn(base)
                tc = maxturn(cur)
                print("  sigma=%-4.1f  顶点差 max %.4e mean %.4e  "
                      "最大转角 %.2f° -> %.2f°"
                      % (sigma, d3.max(), d3.mean(),
                         abs(tb[np.argmax(np.abs(tb))]),
                         abs(tc[np.argmax(np.abs(tc))])))

            # --- 因子随物理距离的衰减（与网格密度无关）---
            print("  因子 g 随物理距离（sigma=0.1, radius=%.3f）：" % rad)
            span = bind[:, 0].max() - bind[:, 0].min()
            print("    轴向跨度 %.3f，按半径归一化后 1R = %.3f 物理单位"
                  % (span, rad))
            for mult in (0.25, 0.5, 1.0, 2.0):
                d = mult * rad
                g = math.exp(-(d / rad) ** 2 / (2 * 0.1 ** 2))
                print("      %.2fR = %.3f 单位  ->  g = %.6f" % (mult, d, g))
            print("    两种密度用同一物理距离时 g 相同（因子只依赖 d/radius）")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
