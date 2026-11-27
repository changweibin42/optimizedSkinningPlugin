# -*- coding: utf-8 -*-
"""对比 alg5 的两种 guide 来源：Elbow vs hingeDisplacement。

    mayapy tests/elbow_guide_compare.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
STEP = 0.12


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
        h = mc.createNode("transform", name="h")
        g = mc.createNode("transform", name="g", parent=h)
        s = om.MSelectionList()
        s.add(g)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=s.getDependNode(0))
        shp = mc.listRelatives(g, shapes=True, noIntermediate=True,
                               fullPath=True)[0]
        js = []
        for i, bn in enumerate(bones):
            if i == 0:
                nd = mc.createNode("joint", name="j0", parent=h)
                mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
            else:
                nd = mc.createNode("joint", name="j%d" % i, parent=js[-1])
                mc.setAttr(nd + ".translateX",
                           bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
            js.append(nd)
        sk = mc.deformer(shp, type="optimizedSkinCluster", name="s#")[0]
        for i, (nd, bn) in enumerate(zip(js, bones)):
            mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (sk, i))
            mc.setAttr("%s.bindPreMatrix[%d]" % (sk, i),
                       *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
        for v, row in enumerate(sparse):
            for k, val in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]" % (sk, v, k), float(val))
        for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(a, node=sk, exists=True):
                mc.setAttr("%s.%s" % (sk, a), 0.0)
        mc.setAttr(js[1] + ".rotateZ", -147.2124)

        print("ocdElbowGuide 属性存在: %s  默认 %s"
              % (mc.attributeQuery("ocdElbowGuide", node=sk, exists=True),
                 mc.getAttr(sk + ".ocdElbowGuide")))

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s2 = om.MSelectionList()
            s2.add(shp)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s2.getDagPath(0)).getPoints(om.MSpace.kObject)]

        mc.setAttr(sk + ".algorithmType", 0)
        cor = read()

        mc.setAttr(sk + ".algorithmType", 5)
        mc.setAttr(sk + ".ocdEnableCenterLimit", False)
        mc.setAttr(sk + ".ocdElbowGuide", False)
        hinge_guide = read()
        mc.setAttr(sk + ".ocdElbowGuide", True)
        elbow_guide = read()

        rings = [round(p[0] / STEP) for p in bind]

        def prof(P, rid):
            m = [i for i in range(n) if rings[i] == rid]
            return min(P[i][1] for i in m), max(P[i][1] for i in m)

        def rnd(P, rid):
            m = [i for i in range(n) if rings[i] == rid]
            c = [sum(P[i][k] for i in m) / len(m) for k in range(3)]
            r = [math.sqrt(sum((P[i][k] - c[k]) ** 2 for k in range(3))) for i in m]
            mr = sum(r) / len(r)
            v = sum((x - mr) ** 2 for x in r) / len(r)
            return math.sqrt(v) / max(1e-12, mr)

        print("\n外轮廓（逐环 min y，越接近 CoR 越饱满）")
        print("%8s %10s %12s %12s %10s %10s"
              % ("x", "CoR", "hingeG", "elbowG", "d(hinge)", "d(elbow)"))
        a5 = a8 = 0.0
        cnt = 0
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            c0 = prof(cor, rid)[0]
            h5 = prof(hinge_guide, rid)[0]
            e8 = prof(elbow_guide, rid)[0]
            a5 += (h5 - c0) ** 2
            a8 += (e8 - c0) ** 2
            cnt += 1
            print("%8.3f %10.5f %12.5f %12.5f %+10.5f %+10.5f"
                  % (x0, c0, h5, e8, h5 - c0, e8 - c0))
        print("\n外轮廓相对 CoR 的 RMS: hinge=%.5f  elbow=%.5f"
              % (math.sqrt(a5 / max(cnt, 1)), math.sqrt(a8 / max(cnt, 1))))

        print("\n圆度对比")
        print("%8s %10s %10s %10s" % ("x", "CoR", "hingeG", "elbowG"))
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.4 <= x0 <= 2.4):
                continue
            print("%8.3f %10.6f %10.6f %10.6f"
                  % (x0, rnd(cor, rid), rnd(hinge_guide, rid),
                     rnd(elbow_guide, rid)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
