# -*- coding: utf-8 -*-
"""复现 show() 的 CoR_CurveGuided 格子，导出它的最终顶点。

对比：
  · OCD 格子       = Elbow.mesh(bind)          （纯 Python 写点）
  · CoR 格子       = alg 0
  · CurveGuided 格子 = alg 8 + preview_influences 权重 + Python delta

若 CurveGuided 与 OCD 接近，则说明 alg 8 这条路径确实能复现 OCD 外形，
且关键在 preview_influences 权重（不是场景真实权重）。

    mayapy tests/repro_show_curveguided.py
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

        bind_list, faces = ref.cylinder()
        bind = np.asarray(bind_list, float)
        n = len(bind)
        bend = 118.0

        weights = np.array([ref.rig_weights(ref.skin_weight(p[0]), THREE)
                            for p in bind])
        effective = np.array([ref.preview_influences(p[0], THREE)
                              for p in bind])
        nb = weights.shape[1]
        print("three_bone=%s  bones=%d  n=%d  bend=%.1f"
              % (THREE, nb, n, bend))

        # 参考：Elbow.mesh
        pose = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
        elbow_pts = np.asarray(pose.mesh(bind_list), float)

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
        tip = mc.createNode("joint", name="tip", parent=elbow)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", 2 * HALF)
        mc.setAttr(tip + ".translateX", HALF)
        bones = [prev, elbow]
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

        # 关节姿态：与 show() 一致
        q = ref.joint_quaternion(bend, 0.0, 0.0)
        s = om.MSelectionList()
        s.add(elbow)
        om.MFnTransform(s.getDagPath(0)).setRotation(om.MQuaternion(*q),
                                                     om.MSpace.kTransform)
        if THREE:
            half = ref.joint_quaternion(bend * 0.5, 0.0, 0.0)
            s2 = om.MSelectionList()
            s2.add(helper)
            om.MFnTransform(s2.getDagPath(0)).setRotation(om.MQuaternion(*half),
                                                          om.MSpace.kTransform)

        # --- CoR 格子（alg 0，用 weights）---
        setw(weights)
        mc.setAttr(skin + ".algorithmType", 0)
        inject([])
        cor = read()

        # --- CurveGuided 格子（alg 8，用 effective + Python delta）---
        setw(effective)
        mc.setAttr(skin + ".algorithmType", 8)
        inject([])
        guided_base = read()

        # 用参考实现算 delta
        bind_l = bind.copy()
        cor_l = cor.copy()   # 近似：用 alg0 输出当 original
        # 更准确：CurveGuided 的 original 是它自己零偏移时的输出
        cor_l = guided_base.copy()

        rots = [ref.joint_quaternion(0, 0, 0), q]
        if THREE:
            rots.append(ref.joint_quaternion(bend * 0.5, 0.0, 0.0))
        edges = ref.local_edges(bind_list, faces)
        guide = np.asarray(ref.hinge_displacement(bind, cor_l, effective,
                                                  bend, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind, effective, bend, 0.0, 0.0))
        delta = np.asarray(ref.structural_centers(edges, cor_l, effective, rots,
                                                  guide, outer, 16.0))
        inject(list(delta.ravel()))
        guided = read()

        def diff(A, B):
            return float(np.abs(A - B).max())

        print("\n最大顶点差：")
        print("  CurveGuided vs OCD(Elbow) : %.5f" % diff(guided, elbow_pts))
        print("  CurveGuided vs CoR        : %.5f" % diff(guided, cor))
        print("  CoR vs OCD(Elbow)         : %.5f" % diff(cor, elbow_pts))

        # 外极点与圆度
        def rnd(P, rid):
            m = slice(rid * SIDES, rid * SIDES + SIDES)
            Q = P[m]
            c = Q.mean(axis=0)
            r = np.linalg.norm(Q - c, axis=1)
            return float(r.std() / max(r.mean(), 1e-12)), float(r.mean())

        print("\n%8s %22s %22s %22s"
              % ("x", "OCD(Elbow)", "CurveGuided", "CoR"))
        print("%8s %10s %12s %10s %12s %10s %12s"
              % ("", "圆度", "半径", "圆度", "半径", "圆度", "半径"))
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) < 1e-6 or abs(x0) > 4.1:
                continue
            a = rnd(elbow_pts, rid)
            b = rnd(guided, rid)
            c = rnd(cor, rid)
            print("%8.3f %10.6f %12.5f %10.6f %12.5f %10.6f %12.5f"
                  % (x0, a[0], a[1], b[0], b[1], c[0], c[1]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
