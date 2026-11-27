# -*- coding: utf-8 -*-
"""用截面半径（坐标系无关）对比 preview 的 CoR_CurveGuided 与 C++ alg5。

上次的错误：把 Elbow.mesh()（肢体局部系）和 deformer 输出（对象空间，
关节在 x=±6）直接求最大差，得到 17.7 的假差异。截面半径与质心路径
不受整体平移/旋转影响，才是可比量。

    mayapy tests/preview_vs_cpp_radius.py
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
        THREE = True
        bend = 135.0

        bind_list, faces = ref.cylinder()
        bind = np.asarray(bind_list, float)
        n = len(bind)
        weights = np.array([ref.rig_weights(ref.skin_weight(p[0]), THREE)
                            for p in bind])
        effective = np.array([ref.preview_influences(p[0], THREE)
                              for p in bind])
        nb = weights.shape[1]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind_list],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shp = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                               fullPath=True)[0]
        prev = mc.createNode("joint", name="prev", parent=holder)
        elbow = mc.createNode("joint", name="elbow", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", 2 * HALF)
        bones = [prev, elbow]
        helper = None
        if THREE:
            helper = mc.createNode("joint", name="helper", parent=prev)
            mc.setAttr(helper + ".translateX", HALF)
            bones.append(helper)

        skin = mc.deformer(shp, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate(bones):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")

        def setw(W):
            for i in range(n):
                for k in range(nb):
                    mc.setAttr("%s.weightList[%d].weights[%d]" % (skin, i, k),
                               float(W[i, k]))

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s = om.MSelectionList()
            s.add(shp)
            return np.asarray([[p.x, p.y, p.z] for p in
                               om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)])

        def inject(vals):
            s = om.MSelectionList()
            s.add(skin)
            om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        # 姿态：与 show() 一致
        q = ref.joint_quaternion(bend, 0.0, 0.0)
        s = om.MSelectionList()
        s.add(elbow)
        om.MFnTransform(s.getDagPath(0)).setRotation(om.MQuaternion(*q),
                                                     om.MSpace.kTransform)
        if THREE:
            half = ref.joint_quaternion(bend * 0.5, 0.0, 0.0)
            s2 = om.MSelectionList()
            s2.add(helper)
            om.MFnTransform(s2.getDagPath(0)).setRotation(
                om.MQuaternion(*half), om.MSpace.kTransform)

        # --- CoR 格子 ---
        setw(weights)
        mc.setAttr(skin + ".algorithmType", 0)
        inject([])
        cor = read()

        # --- CurveGuided 格子：alg8 + effective + Python delta ---
        setw(effective)
        mc.setAttr(skin + ".algorithmType", 8)
        inject([])
        base = read()
        rots = [ref.joint_quaternion(0, 0, 0), q]
        if THREE:
            rots.append(ref.joint_quaternion(bend * 0.5, 0.0, 0.0))
        edges = ref.local_edges(bind_list, faces)
        # CurveGuided 的 original 就是它自己零偏移的输出（对象空间，
        # 但参考实现的 structural_centers 期望局部链空间）。
        # show() 里 original 直接取自 guided_mesh 的点，也就是对象空间。
        guide = np.asarray(ref.hinge_displacement(bind, base, effective,
                                                  bend, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind, effective, bend, 0.0, 0.0))
        delta = np.asarray(ref.structural_centers(edges, base, effective, rots,
                                                  guide, outer, 16.0))
        inject(list(delta.ravel()))
        guided = read()

        # --- C++ alg5：两种 guide 来源，用场景真实两骨权重（预览用三骨）---
        setw(effective)
        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".ocdEnableCenterLimit", False)
        mc.setAttr(skin + ".ocdElbowGuide", False)
        inject([])
        cpp_hinge = read()
        mc.setAttr(skin + ".ocdElbowGuide", True)
        cpp_elbow = read()

        # --- OCD 格子（Elbow.mesh）---
        pose = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
        elbow_pts = np.asarray(pose.mesh(bind_list), float)

        def ring_stats(P, rid):
            m = slice(rid * SIDES, rid * SIDES + SIDES)
            Q = P[m]
            c = Q.mean(axis=0)
            r = np.linalg.norm(Q - c, axis=1)
            return float(r.mean()), float(r.std() / max(r.mean(), 1e-12)), c

        names = [("CoR", cor), ("CurveGuided", guided), ("OCD(Elbow)", elbow_pts),
                 ("cpp hinge", cpp_hinge), ("cpp elbow", cpp_elbow)]
        print("bend=%.0f  three_bone=%s  458 点圆柱" % (bend, THREE))
        print("\n截面平均半径（坐标系无关；>1 表示外扩）")
        print("%8s" % "x" + "".join(" %12s" % nm for nm, _ in names))
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) < 1e-9 or abs(x0) > 4.1:
                continue
            line = "%8.3f" % x0
            for nm, P in names:
                line += " %12.5f" % ring_stats(P, rid)[0]
            print(line)

        print("\n截面圆度 std/mean")
        print("%8s" % "x" + "".join(" %12s" % nm for nm, _ in names))
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) < 1e-9 or abs(x0) > 4.1:
                continue
            line = "%8.3f" % x0
            for nm, P in names:
                line += " %12.6f" % ring_stats(P, rid)[1]
            print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
