# -*- coding: utf-8 -*-
"""从 C++ 导出 guide，检查纯单骨段的 |guide| 是否为 0。

若 w1>0.99 的 |guide| 达 10，则 C++ 的 guide 确实有这个问题。

    mayapy tests/dump_cpp_guide_check.py
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

        d = json.load(open(r"D:\tmp\ocd_case3.json"))
        bind = [tuple(p) for p in d["mesh"]["bindPoints"]]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        joints = []
        for i, bn in enumerate(bones):
            if i == 0:
                nd = mc.createNode("joint", name="j0", parent=holder)
                mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
            else:
                nd = mc.createNode("joint", name="j%d" % i, parent=joints[-1])
                mc.setAttr(nd + ".translateX",
                           bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
            joints.append(nd)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="s#")[0]
        for i, (nd, bn) in enumerate(zip(joints, bones)):
            mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                       *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
        for v, row in enumerate(sparse):
            for k, val in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k), float(val))
        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".ocdEnableCenterLimit", False)
        for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(a, node=skin, exists=True):
                mc.setAttr("%s.%s" % (skin, a), 0.0)
        mc.setAttr(joints[1] + ".rotateZ", -147.2124)

        for _ in range(3):
            mc.refresh(force=True)
            mc.dgeval(shape)

        gd = mc.getAttr(skin + ".ocdGuideDump")
        dd = mc.getAttr(skin + ".ocdDeltaDump")
        lm = mc.getAttr(skin + ".ocdCenterLimitDump")
        print("guide 长度 %d  delta 长度 %d  limit %s"
              % (len(gd) if gd else 0, len(dd) if dd else 0, lm))
        if not gd:
            print("★ guide dump 为空")
            return 1

        w1 = [float(r.get("1", 0.0)) for r in sparse]
        x = [p[0] for p in bind]

        import collections
        def stats(mask, label):
            gm = [math.sqrt(gd[3 * i] ** 2 + gd[3 * i + 1] ** 2 + gd[3 * i + 2] ** 2)
                  for i in range(n) if mask[i]]
            if not gm:
                print("  %-14s 无顶点" % label)
                return
            gm.sort()
            print("  %-14s n=%5d  均=%.6f  中位=%.6f  max=%.6f"
                  % (label, len(gm), sum(gm) / len(gm), gm[len(gm) // 2], gm[-1]))

        print("\nC++ 导出的 |guide|：")
        stats([w < 0.01 for w in w1], "w1<0.01")
        stats([w > 0.99 for w in w1], "w1>0.99")
        stats([0.4 < w < 0.6 for w in w1], "w1 in (0.4,0.6)")

        print("\n按 x 分段：")
        print("  %12s %6s %12s %12s" % ("x", "n", "|guide|均", "max"))
        for lo, hi in ((-3.6, -3.0), (-2.4, -1.8), (-1.2, -0.6), (-0.6, 0.0),
                       (0.0, 0.6), (0.6, 1.2), (1.2, 1.8), (1.8, 2.4),
                       (2.4, 3.0), (3.0, 3.6), (3.6, 12.1)):
            m2 = [i for i in range(n) if lo <= x[i] < hi]
            if len(m2) < 20:
                continue
            gm = [math.sqrt(gd[3 * i] ** 2 + gd[3 * i + 1] ** 2 + gd[3 * i + 2] ** 2)
                  for i in m2]
            print("  %12s %6d %12.5f %12.5f"
                  % ("%.1f..%.1f" % (lo, hi), len(m2),
                     sum(gm) / len(gm), max(gm)))

        # delta 的情况
        print("\nC++ 导出的 |delta|：")
        stats_d = lambda mask: [math.sqrt(dd[3 * i] ** 2 + dd[3 * i + 1] ** 2
                                          + dd[3 * i + 2] ** 2)
                                for i in range(n) if mask[i]]
        for lab, mask in (("w1<0.01", [w < 0.01 for w in w1]),
                          ("w1>0.99", [w > 0.99 for w in w1])):
            v = stats_d(mask)
            if v:
                print("  %-14s n=%5d  均=%.6f max=%.6f"
                      % (lab, len(v), sum(v) / len(v), max(v)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
