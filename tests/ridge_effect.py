# -*- coding: utf-8 -*-
"""Does the adaptive ridge tame |delta| and remove the crease?"""
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


def turn(p, q, r):
    u = np.asarray(q) - np.asarray(p)
    v = np.asarray(r) - np.asarray(q)
    lu, lv = np.linalg.norm(u), np.linalg.norm(v)
    if lu < 1e-15 or lv < 1e-15:
        return None
    return math.degrees(math.acos(max(-1.0, min(1.0,
        float(u @ v) / (lu * lv)))))


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

        st = [-HALF + r * (2 * HALF / (RINGS - 1)) for r in range(RINGS)]
        idx = [r * SIDES for r in range(RINGS)]

        mc.setAttr(skin + ".algorithmType", 5)
        for enabled, fillet in ((True, 0.0), (True, 0.4), (False, 0.0),
                                (False, 0.4)):
            mc.setAttr(skin + ".ocdEnableCenterLimit", enabled)
            mc.setAttr(skin + ".ocdCreaseFillet", fillet)
            mc.refresh(force=True)
            P = read()
            d = np.asarray(mc.getAttr(skin + ".ocdDeltaDump"), float).reshape(-1, 3)
            info = mc.getAttr(skin + ".ocdCenterLimitDump")
            ts = [turn(P[idx[k - 1]], P[idx[k]], P[idx[k + 1]])
                  for k in range(1, RINGS - 1)]
            mx = max(t for t in ts if t is not None)
            worst_k = max(range(1, RINGS - 1),
                          key=lambda k: ts[k - 1] if ts[k - 1] else -1)
            nc = int((np.linalg.norm(d, axis=1) > info[0] - 1e-6).sum()) \
                if enabled else 0
            print("limit=%-5s fillet=%.1f  |delta|max=%8.3f  iters=%3d  "
                  "被截=%4d  内轨最大转角=%7.2f @ s=%.3f"
                  % (enabled, fillet, np.linalg.norm(d, axis=1).max(),
                     int(info[1]), nc, mx, st[worst_k]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
