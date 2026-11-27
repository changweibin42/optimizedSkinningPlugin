# -*- coding: utf-8 -*-
r"""隔离「折痕是不是密度造成的」：同一条圆柱，只改环数。

已确认：纯 Python 的 Elbow 在 80边×201环 上是平滑的（最大三点转角 6.89°，
单调变化），所以解析形状本身没有尖角。但插件在 test04（80×201）上出现
174.75° 的折返，在 458 点圆柱（24×19）上只有 40.76°。

这里把两种密度都交给插件，权重、弯角、接法完全相同，只改环数。

    mayapy tests/crease_vs_density.py tests/_density.json
"""
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = sys.argv[1] if len(sys.argv) > 1 else "tests/_density.json"
SKIN = "optimizedSkinCluster"
HALF = 6.0
BEND = 146.1265
CASES = ((24, 19), (80, 201))


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


def turning(p, q, r):
    u = [q[i] - p[i] for i in range(3)]
    v = [r[i] - q[i] for i in range(3)]
    lu = math.sqrt(sum(x * x for x in u))
    lv = math.sqrt(sum(x * x for x in v))
    if lu < 1e-12 or lv < 1e-12:
        return None
    d = sum(u[i] * v[i] for i in range(3)) / (lu * lv)
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        report = {}

        for sides, rings in CASES:
            bind, faces = tube(sides, rings, HALF)
            n = len(bind)

            # rig 与 show() 一致：prev 在 -HALF，elbow 是其子节点（世界 0）
            holder = mc.createNode("transform", name="h%d_%d" % (sides, rings))
            geo = mc.createNode("transform", name="g%d_%d" % (sides, rings),
                                parent=holder)
            sel = om.MSelectionList()
            sel.add(geo)
            om.MFnMesh().create([om.MPoint(*p) for p in bind],
                                [len(f) for f in faces],
                                [j for f in faces for j in f],
                                parent=sel.getDependNode(0))
            shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                     fullPath=True)[0]
            prev = mc.createNode("joint", name="pv%d_%d" % (sides, rings),
                                 parent=holder)
            mc.setAttr(prev + ".translateX", -HALF)
            elbow = mc.createNode("joint", name="el%d_%d" % (sides, rings),
                                  parent=prev)
            mc.setAttr(elbow + ".translateX", HALF)

            skin = mc.deformer(shape, type=SKIN, name="sk%d_%d#" % (sides, rings))[0]
            for k, jnt in enumerate((prev, elbow)):
                mc.connectAttr(jnt + ".worldMatrix[0]",
                               "%s.matrix[%d]" % (skin, k))
                mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                           *mc.getAttr(jnt + ".worldInverseMatrix[0]"),
                           type="matrix")
            for i, (sx, _, _) in enumerate(bind):
                w = ref.skin_weight(sx)
                mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                           1.0 - w, w, size=2)

            # 摆姿势
            s = om.MSelectionList()
            s.add(elbow)
            om.MFnTransform(s.getDagPath(0)).setRotation(
                om.MQuaternion(*ref.joint_quaternion(BEND, 0.0, 0.0)),
                om.MSpace.kTransform)
            mc.refresh(force=True)

            def read():
                for _ in range(2):
                    mc.dgeval(shape)
                    mc.refresh(force=True)
                sl = om.MSelectionList()
                sl.add(shape)
                return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
                    sl.getDagPath(0)).getPoints(om.MSpace.kObject)])

            def measure(P, j):
                idx = [r * sides + j for r in range(rings)]
                ts = [turning(P[idx[k - 1]], P[idx[k]], P[idx[k + 1]])
                      for k in range(1, rings - 1)]
                vals = [t for t in ts if t is not None]
                st = [-HALF + (k + 1) * (2 * HALF / (rings - 1))
                      for k in range(1, rings - 1)]
                return ts, st, (max(vals) if vals else 0.0), \
                    (float(np.percentile(vals, 99)) if vals else 0.0)

            for alg in (0, 3, 5):
                for limit in ("16", "4", "off"):
                    mc.setAttr(skin + ".algorithmType", alg)
                    if limit == "off":
                        mc.setAttr(skin + ".ocdEnableCenterLimit", False)
                        mc.setAttr(skin + ".ocdCenterLimit", 16.0)
                    else:
                        mc.setAttr(skin + ".ocdEnableCenterLimit", True)
                        mc.setAttr(skin + ".ocdCenterLimit", float(limit))
                    mc.refresh(force=True)
                    P = read()
                    key = "%dx%d_alg%d_lim%s" % (sides, rings, alg, limit)
                    entry = {"sides": sides, "rings": rings, "alg": alg,
                             "limit": limit, "spacing": 2 * HALF / (rings - 1)}
                    for j, tag in ((0, "outer"), (sides // 2, "inner")):
                        ts, st, mx, p99 = measure(P, j)
                        entry[tag] = {"turning": ts, "station": st,
                                      "max": mx, "p99": p99}
                    if alg == 5:
                        d = mc.getAttr(skin + ".ocdDeltaDump")
                        if d:
                            d3 = np.asarray(d, float).reshape(-1, 3)
                            entry["deltaMax"] = float(np.linalg.norm(d3, axis=1).max())
                            entry["deltaMean"] = float(np.linalg.norm(d3, axis=1).mean())
                    report[key] = entry
                    print("%-20s 内 %7.2f°  外 %7.2f°  环距 %.4f%s"
                          % (key, entry["inner"]["max"], entry["outer"]["max"],
                             entry["spacing"],
                             ("  |delta|max %.2f" % entry["deltaMax"])
                             if "deltaMax" in entry else ""))
            mc.delete(holder)

        with open(OUT, "w") as handle:
            json.dump(report, handle, default=float)
        print("\nwrote", OUT)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
