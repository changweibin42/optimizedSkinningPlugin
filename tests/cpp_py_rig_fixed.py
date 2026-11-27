# -*- coding: utf-8 -*-
r"""rig 层级必须与 show() 一致：prev -> elbow -> tip，helper 是 prev 的另一子节点。

上一版把 elbow 挂在 holder 下，导致 C++ 只能看到 prev->helper 这一对，
把 helper 当 child、elbow 当 helper，A_i 全错，delta 解在错误空间里。

    mayapy tests/cpp_py_rig_fixed.py
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

        # 与 show() 完全一致的层级
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
        mc.setAttr(skin + ".corDebugLog", True)
        for _ in range(3):
            mc.getAttr(shp + ".outMesh")
        print("REPORT:", mc.getAttr(skin + ".ocdHingeReport"))

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s3 = om.MSelectionList()
            s3.add(shp)
            return np.asarray([[p.x, p.y, p.z] for p in
                               om.MFnMesh(s3.getDagPath(0)).getPoints(om.MSpace.kObject)])

        def stats(P):
            out = []
            for rid in range(RINGS):
                Q = P[rid * SIDES:(rid + 1) * SIDES]
                c = Q.mean(axis=0)
                r = np.linalg.norm(Q - c, axis=1)
                out.append((float(r.mean()), float(r.std() / max(r.mean(), 1e-12))))
            return out

        base = read()
        pose = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
        elbow_pts = np.asarray(pose.mesh(bind_list), float)
        se = stats(elbow_pts)
        sb = stats(base)

        print("\n%8s %12s %12s %10s"
              % ("x", "CoR半径", "alg5(Elbow)", "OCD(Elbow)"))
        for lim, en in ((16, True), (64, True), (0, False)):
            mc.setAttr(skin + ".ocdEnableCenterLimit", 1 if en else 0)
            mc.setAttr(skin + ".ocdCenterLimit", float(lim) if lim > 0 else 16.0)
            mc.setAttr(skin + ".ocdElbowGuide", True)
            P = read()
            sp = stats(P)
            print("\n--- limit=%s en=%s ---" % (lim, en))
            for rid in range(RINGS):
                x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
                if abs(x0) < 1e-9 or abs(x0) > 4.1:
                    continue
                print("%8.3f %12.5f %12.5f %10.5f"
                      % (x0, sb[rid][0], sp[rid][0], se[rid][0]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
