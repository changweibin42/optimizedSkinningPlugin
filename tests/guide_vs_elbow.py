# -*- coding: utf-8 -*-
r"""导出 C++ 的 guide，与「Elbow.point(bindLocal) − originalLocal」逐点对比。

关键判据：纯父骨顶点（w1 = 0）的 CoR 输出就是绑定位置，
所以 originalLocal == bindLocal，此时
    guide 应当精确等于 Elbow.point(bindLocal) - bindLocal
若这一组对不上，说明 bindLocal / Elbow 调用的约定有偏差。

    mayapy tests/guide_vs_elbow.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import numpy as np
        import ocd_elbow_reference_latest as ref

        HALF = ref.HALF
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1
        bend = 135.0

        bind_list, faces = ref.cylinder()
        bind = np.asarray(bind_list, float)
        n = len(bind)
        eff = np.array([ref.preview_influences(p[0], True) for p in bind])
        nb = eff.shape[1]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        s = om.MSelectionList()
        s.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind_list],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=s.getDependNode(0))
        shp = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                               fullPath=True)[0]
        prev = mc.createNode("joint", name="prev", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(elbow + ".translateX", 2 * HALF)
        tip = mc.createNode("joint", name="tip", parent=elbow)
        mc.setAttr(tip + ".translateX", HALF)
        helper = mc.createNode("joint", name="helper", parent=prev)
        mc.setAttr(helper + ".translateX", HALF)

        bones = [prev, elbow, helper]
        skin = mc.deformer(shp, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate(bones):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
        for i in range(n):
            for k in range(nb):
                mc.setAttr("%s.weightList[%d].weights[%d]" % (skin, i, k),
                           float(eff[i, k]))

        q = ref.joint_quaternion(bend, 0.0, 0.0)
        s1 = om.MSelectionList()
        s1.add(elbow)
        om.MFnTransform(s1.getDagPath(0)).setRotation(om.MQuaternion(*q),
                                                      om.MSpace.kTransform)
        hq = ref.joint_quaternion(bend * 0.5, 0.0, 0.0)
        s2 = om.MSelectionList()
        s2.add(helper)
        om.MFnTransform(s2.getDagPath(0)).setRotation(om.MQuaternion(*hq),
                                                      om.MSpace.kTransform)

        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".ocdElbowGuide", True)
        mc.setAttr(skin + ".ocdEnableCenterLimit", True)
        mc.setAttr(skin + ".ocdCenterLimit", 16.0)
        mc.setAttr(skin + ".corDebugLog", True)
        for _ in range(3):
            mc.getAttr(shp + ".outMesh")
        print("REPORT:", mc.getAttr(skin + ".ocdHingeReport"))

        gd = mc.getAttr(skin + ".ocdGuideDump")
        print("guide dump len:", len(gd) if gd else 0)
        if not gd:
            return 1
        G = np.asarray(gd, float).reshape(n, 3)

        # --- 复算 bindLocal，与 C++ 的 toLocalPoint 同口径 ---
        # 子骨在 x=+HALF，joint1 绑定旋转为单位 → basisY=+Y, basisZ=+Z
        axis = np.array([1.0, 0.0, 0.0])
        bY = np.array([0.0, 1.0, 0.0])
        bZ = np.cross(axis, bY)
        child_pivot = np.array([HALF, 0.0, 0.0])
        rel = bind - child_pivot
        st = rel @ axis
        rad = rel - np.outer(st, axis)
        R = float(np.linalg.norm(rad, axis=1).max())
        bind_local = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])
        print("R = %.6f" % R)

        # --- Elbow.point(bindLocal) ---
        pose = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
        elbow_pts = np.asarray([pose.point(tuple(p)) for p in bind_local], float)

        w1 = eff[:, 1]
        pure_parent = w1 < 1e-9
        print("纯父骨顶点(w1=0): %d" % int(pure_parent.sum()))

        # 对这些顶点 originalLocal == bindLocal，期望 guide = elbow - bind
        exp_g = elbow_pts - bind_local
        d = np.abs(G - exp_g)
        dm = np.linalg.norm(G - exp_g, axis=1)

        print("\n[判据] 纯父骨顶点：guide vs (Elbow.point(bindLocal) - bindLocal)")
        print("  max|Δ|=%.6f  mean|Δ|=%.6f  最大处 v%d"
              % (dm[pure_parent].max(), dm[pure_parent].mean(),
                 int(np.where(pure_parent)[0][np.argmax(dm[pure_parent])])))

        print("\n逐环对比（取每环 j=0 与 j=SIDES/2）：")
        print("%8s %6s %28s %28s"
              % ("x", "w1", "C++ guide (y,z)", "期望 guide (y,z)"))
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) < 1e-9 or abs(x0) > 4.1:
                continue
            for j in (0, SIDES // 2):
                i = rid * SIDES + j
                print("%8.3f %6.3f  (%+9.5f,%+9.5f)   (%+9.5f,%+9.5f)"
                      % (x0, w1[i], G[i, 1], G[i, 2], exp_g[i, 1], exp_g[i, 2]))

        print("\n按 w1 分段统计 |G - exp_g|：")
        for lo, hi in ((0.0, 1e-9), (1e-9, 0.1), (0.1, 0.4),
                       (0.4, 0.6), (0.6, 0.9), (0.9, 1.0 - 1e-9)):
            m = (w1 >= lo) & (w1 < hi)
            if m.sum() < 4:
                continue
            print("  w1∈[%.2f,%.2f) n=%4d  mean|Δ|=%.5f  max=%.5f"
                  % (lo, hi, int(m.sum()), dm[m].mean(), dm[m].max()))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
