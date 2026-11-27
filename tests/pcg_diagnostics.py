# -*- coding: utf-8 -*-
"""Read the PCG diagnostics: did the solve converge?"""
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness"
                   r"\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref  # noqa: E402

SKIN = "optimizedSkinCluster"
HALF, BEND = 6.0, 146.1265
SIDES, RINGS = 80, 201


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


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = tube(SIDES, RINGS, HALF)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        s = om.MSelectionList()
        s.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=s.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="pv", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="el", parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type=SKIN, name="sk#")[0]
        for k, j in enumerate((prev, elbow)):
            mc.connectAttr(j + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                       *mc.getAttr(j + ".worldInverseMatrix[0]"), type="matrix")
        for i, (sx, _, _) in enumerate(bind):
            w = ref.skin_weight(sx)
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w, w, size=2)
        s2 = om.MSelectionList()
        s2.add(elbow)
        om.MFnTransform(s2.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
        sl = om.MSelectionList()
        sl.add(shape)

        def read():
            for _ in range(2):
                mc.dgeval(shape)
                mc.refresh(force=True)
            return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
                sl.getDagPath(0)).getPoints(om.MSpace.kObject)])

        mc.setAttr(skin + ".algorithmType", 5)
        for enabled in (True, False):
            mc.setAttr(skin + ".ocdEnableCenterLimit", enabled)
            mc.refresh(force=True)
            read()
            info = mc.getAttr(skin + ".ocdCenterLimitDump")
            d = np.asarray(mc.getAttr(skin + ".ocdDeltaDump"), float).reshape(-1, 3)
            print("limitEnabled=%-5s  limit=%.4f  iters=%d  converged=%s  "
                  "residualRatio=%.3e  |delta|max=%.3f"
                  % (enabled, info[0], int(info[1]), bool(info[2] > 0.5),
                     info[3], np.linalg.norm(d, axis=1).max()))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
