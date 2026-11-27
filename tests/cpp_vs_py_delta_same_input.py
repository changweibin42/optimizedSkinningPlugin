# -*- coding: utf-8 -*-
"""验证：C++ alg5 与「alg8 + Python delta」在同一输入下是否一致。

若一致 → C++ 求解器没问题，差异来自 guide/权重
若不一致 → C++ 的 structural_centers 有数值差异

    mayapy tests/cpp_vs_py_delta_same_input.py
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
        effective = np.array([ref.preview_influences(p[0], THREE) for p in bind])
        nb = effective.shape[1]

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
        for i in range(n):
            for k in range(nb):
                mc.setAttr("%s.weightList[%d].weights[%d]" % (skin, i, k),
                           float(effective[i, k]))
        for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(a, node=skin, exists=True):
                mc.setAttr("%s.%s" % (skin, a), 0.0)

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

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s3 = om.MSelectionList()
            s3.add(shp)
            return np.asarray([[p.x, p.y, p.z] for p in
                               om.MFnMesh(s3.getDagPath(0)).getPoints(om.MSpace.kObject)])

        def inject(vals):
            s4 = om.MSelectionList()
            s4.add(skin)
            om.MFnDependencyNode(s4.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        def stats(P):
            out = []
            for rid in range(RINGS):
                Q = P[rid * SIDES:(rid + 1) * SIDES]
                c = Q.mean(axis=0)
                r = np.linalg.norm(Q - c, axis=1)
                out.append((float(r.mean()), float(r.std() / max(r.mean(), 1e-12))))
            return out

        # 1) CoR 基准（alg0）
        mc.setAttr(skin + ".algorithmType", 0)
        inject([])
        cor = read()

        # 2) C++ alg5，Elbow guide
        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".ocdEnableCenterLimit", False)
        mc.setAttr(skin + ".ocdElbowGuide", True)
        inject([])
        cpp5 = read()
        # 抓 C++ 的 delta dump
        dv = mc.getAttr(skin + ".ocdDeltaDump")
        cpp_delta = np.asarray(dv, float).reshape(n, 3) if dv else None
        if cpp_delta is not None:
            print("C++ |delta|均=%.5f max=%.5f"
                  % (np.linalg.norm(cpp_delta, axis=1).mean(),
                     np.linalg.norm(cpp_delta, axis=1).max()))

        # 3) C++ alg8 + Python delta（同 guide：用 Elbow）
        # 先取 alg5 零偏移输出作为 original
        mc.setAttr(skin + ".algorithmType", 8)
        inject([])
        orig8 = read()

        # Python 侧：Elbow guide
        rots = [ref.joint_quaternion(0, 0, 0), q]
        if THREE:
            rots.append(ref.joint_quaternion(bend * 0.5, 0.0, 0.0))
        edges = ref.local_edges(bind_list, faces)
        elbow_obj = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
        elbow_pts = np.asarray(elbow_obj.mesh(bind_list), float)
        guide = elbow_pts - orig8
        outer = np.asarray(ref.outer_participation(bind, effective, bend, 0.0, 0.0))
        py_delta = np.asarray(ref.structural_centers(edges, orig8, effective,
                                                     rots, guide, outer, 16.0))
        print("Py  |delta|均=%.5f max=%.5f"
              % (np.linalg.norm(py_delta, axis=1).mean(),
                 np.linalg.norm(py_delta, axis=1).max()))
        if cpp_delta is not None:
            d = np.abs(py_delta - cpp_delta)
            print("\|Py-C++\| delta: max=%.6f mean=%.6f" % (d.max(), d.mean()))

        inject(list(py_delta.ravel()))
        py8 = read()

        print("\n最大顶点差: cpp9(alg5) vs alg8+PyDelta = %.6f"
              % np.abs(cpp5 - py8).max())

        sc = stats(cor)
        s5 = stats(cpp5)
        s8 = stats(py8)
        print("\n%8s %10s %12s %12s %12s"
              % ("x", "CoR半径", "cpp alg5", "alg8+Py", "OCD(Elbow)"))
        se = stats(elbow_pts)
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) < 1e-9 or abs(x0) > 4.1:
                continue
            print("%8.3f %10.5f %12.5f %12.5f %12.5f"
                  % (x0, sc[rid][0], s5[rid][0], s8[rid][0], se[rid][0]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
