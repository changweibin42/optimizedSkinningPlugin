# -*- coding: utf-8 -*-
r"""ocdCreaseFillet 验收：station=0 的形状折角必须被磨掉，且默认关闭时逐位不变。

背景：参考实现的 Elbow / hinge_section 在 station=0 用「左右两支 + 绕原点
旋转」拼接，位置连续但切向夹角只有 164.5°（Elbow）／167.9°（hinge_section）。
458 点圆柱（环距 0.667）跨过这个折角看不出来，16080 点（环距 0.060）会
如实采出来，视觉上是刀口。

ocdCreaseFillet = W 在 |station| < W 内用三次 Hermite 接起来，窗口外完全
沿用原实现。这个属性默认 0（关闭），所以既有场景逐位不变。

注意：这个属性只处理 station=0 那一处形状折角。alg5 在 |station|≈2.2 处
还有一处更大的折角，那是 structuralCenters 的解在 A_i 病态区震荡造成的，
不在这里的范围内（见 NOTES §11c）。

    mayapy tests/crease_fillet_check.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
SKIN = "optimizedSkinCluster"
HALF = 6.0
BEND = 146.1265
SIDES, RINGS = 80, 201
WINDOWS = (0.0, 0.1, 0.2, 0.3, 0.4, 0.6, 0.8)


def tube(sides, rings, half):
    pts, faces = [], []
    for row in range(rings):
        s = -half + row * (2 * half / (rings - 1))
        for j in range(sides):
            a = 2 * math.pi * j / sides
            pts.append((s, math.cos(a), math.sin(a)))
    for row in range(rings - 1):
        for j in range(sides):
            a = row * sides + j
            b = row * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    return pts, faces


def turn(p, q, r):
    u = np.asarray(q) - np.asarray(p)
    v = np.asarray(r) - np.asarray(q)
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-15 or lv < 1e-15:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0,
        float(u @ v) / (lu * lv)))))


def build(mc, om, ref, sides, rings, label):
    bind, faces = tube(sides, rings, HALF)
    holder = mc.createNode("transform", name="%s_h#" % label)
    geo = mc.createNode("transform", name="%s_g#" % label, parent=holder)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    prev = mc.createNode("joint", name="%s_pv#" % label, parent=holder)
    mc.setAttr(prev + ".translateX", -HALF)
    elbow = mc.createNode("joint", name="%s_el#" % label, parent=prev)
    mc.setAttr(elbow + ".translateX", HALF)
    skin = mc.deformer(shape, type=SKIN, name="%s_sk#" % label)[0]
    for k, jnt in enumerate((prev, elbow)):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
    for i, (sx, _, _) in enumerate(bind):
        w = ref.skin_weight(sx)
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                   1.0 - w, w, size=2)
    s = om.MSelectionList()
    s.add(elbow)
    om.MFnTransform(s.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
    mc.setAttr(skin + ".ocdCenterLimit", 16.0)
    mc.refresh(force=True)
    return holder, shape, skin


def read(mc, om, shape):
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    s = om.MSelectionList()
    s.add(shape)
    return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        s.getDagPath(0)).getPoints(om.MSpace.kObject)])


def turning_at_zero(P, sides, rings):
    """station=0 那一环的三点转角。"""
    k = rings // 2
    idx = [r * sides for r in range(rings)]
    return turn(P[idx[k - 1]], P[idx[k]], P[idx[k + 1]])


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        rig = build(mc, om, ref, SIDES, RINGS, "hi")
        holder, shape, skin = rig

        print("alg3（Elbow 形状）在 station=0 的三点转角：")
        print("%8s %12s" % ("fillet", "turning"))
        t_alg3 = {}
        for window in WINDOWS:
            mc.setAttr(skin + ".algorithmType", 3)
            mc.setAttr(skin + ".ocdCreaseFillet", window)
            mc.refresh(force=True)
            P = read(mc, om, shape)
            t_alg3[window] = turning_at_zero(P, SIDES, RINGS)
            print("%8.2f %12.2f" % (window, t_alg3[window]))

        # 默认关闭时的形状必须与「参考实现的原始 Elbow」一致
        mc.setAttr(skin + ".ocdCreaseFillet", 0.0)
        mc.refresh(force=True)
        P0 = read(mc, om, shape)
        bind, _ = tube(SIDES, RINGS, HALF)
        analytic = np.asarray(ref.Elbow(BEND, 0.0, 0.0, ref.ZERO, False).mesh(bind),
                              float)
        dev = float(np.linalg.norm(P0 - analytic, axis=1).max())
        print("\nfillet=0 时 alg3 与参考 Elbow 的最大顶点差: %.3e" % dev)
        if dev > 1e-6:
            failures.append("fillet=0 未复现参考 Elbow（%.3e）" % dev)

        # 打开圆角后 station=0 的折角必须显著变小
        if t_alg3[0.0] < 100.0:
            failures.append("station=0 原本就没有折角（%.2f），前提不成立"
                            % t_alg3[0.0])
        best = min(t_alg3.items(), key=lambda kv: kv[1])
        print("最平滑的窗口: fillet=%.2f -> %.2f°（原来 %.2f°）"
              % (best[0], best[1], t_alg3[0.0]))
        if best[1] > 30.0:
            failures.append("所有窗口都没把 station=0 磨平（最好 %.2f°）"
                            % best[1])

        # 关闭时逐位不变
        mc.setAttr(skin + ".ocdCreaseFillet", 0.0)
        mc.refresh(force=True)
        again = read(mc, om, shape)
        d = float(np.linalg.norm(again - P0, axis=1).max())
        print("再次关闭后的逐位偏差: %.3e" % d)
        if d > 1e-12:
            failures.append("开关不可逆（%.3e）" % d)

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            return 1
        print("PASS: 圆角磨平 station=0 的形状折角，且默认关闭时与参考实现逐位一致")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
