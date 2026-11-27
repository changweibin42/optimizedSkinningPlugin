# -*- coding: utf-8 -*-
"""458 点圆柱上：alg8 + Python delta vs alg0(纯CoR) vs alg5。

这是你在 Python 预览里看到效果的那个尺度。若这里 alg8 明显优于纯 CoR，
说明效果只在小尺度成立，16080 网格上不成立。

    mayapy tests/cyl_alg8_check.py
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
        import ocd_elbow_reference_latest as ref

        s1 = json.load(open(os.path.join(HERE, "_cyl1.json")))
        s2 = json.load(open(os.path.join(HERE, "_cyl2.json")))
        bind = [tuple(p) for p in s1["bind"]]
        faces = s1["faces"]
        weights = s1["weights"]
        n = len(bind)
        HALF = s1["HALF"]
        bend = s1["bend"]
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1

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
        prev = mc.createNode("joint", name="prev", parent=h)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", HALF)
        sk = mc.deformer(shp, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (sk, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (sk, k), *inv, type="matrix")
        for i, w in enumerate(weights):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (sk, i),
                       float(w[0]), float(w[1]), size=2)
        for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(a, node=sk, exists=True):
                mc.setAttr("%s.%s" % (sk, a), 0.0)
        mc.setAttr(elbow + ".rotateZ", bend)

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s2_ = om.MSelectionList()
            s2_.add(shp)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s2_.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def inject_fit(vals):
            s3 = om.MSelectionList()
            s3.add(sk)
            om.MFnDependencyNode(s3.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        mc.setAttr(sk + ".algorithmType", 0)
        inject_fit([])
        cor = read()

        mc.setAttr(sk + ".algorithmType", 5)
        mc.setAttr(sk + ".ocdEnableCenterLimit", False)
        a5 = read()

        # delta 是局部归一空间，R=1（本夹具），直接当对象空间向量用
        mc.setAttr(sk + ".algorithmType", 8)
        inject_fit(s2["delta"])
        a8 = read()

        def rnd(P, rid):
            m = [rid * SIDES + j for j in range(SIDES)]
            c = [sum(P[i][k] for i in m) / SIDES for k in range(3)]
            r = [math.sqrt(sum((P[i][k] - c[k]) ** 2 for k in range(3))) for i in m]
            mr = sum(r) / len(r)
            v = sum((x - mr) ** 2 for x in r) / len(r)
            return math.sqrt(v) / max(1e-12, mr), mr

        def prof(P, rid):
            m = [rid * SIDES + j for j in range(SIDES)]
            return min(P[i][1] for i in m), max(P[i][1] for i in m)

        print("bend=%.2f HALF=%.1f  458 点圆柱" % (bend, HALF))
        print("\n%8s %12s %12s %12s %12s %12s %12s"
              % ("x", "CoR圆度", "alg5圆度", "alg8圆度", "CoR外", "alg5外", "alg8外"))
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) > 4.1:
                continue
            r0, m0 = rnd(cor, rid)
            r5, m5 = rnd(a5, rid)
            r8, m8 = rnd(a8, rid)
            _, o0 = prof(cor, rid)
            _, o5 = prof(a5, rid)
            _, o8 = prof(a8, rid)
            print("%8.3f %12.6f %12.6f %12.6f %12.5f %12.5f %12.5f"
                  % (x0, r0, r5, r8, o0, o5, o8))

        # 外轮廓相对 CoR 的偏差
        d5 = d8 = 0.0
        cnt = 0
        for rid in range(RINGS):
            x0 = -HALF + rid * (2 * HALF / (RINGS - 1))
            if abs(x0) > 4.1:
                continue
            _, o0 = prof(cor, rid)
            _, o5 = prof(a5, rid)
            _, o8 = prof(a8, rid)
            d5 += (o5 - o0) ** 2
            d8 += (o8 - o0) ** 2
            cnt += 1
        print("\n外轮廓相对 CoR 的 RMS: alg5=%.5f  alg8=%.5f"
              % (math.sqrt(d5 / max(cnt, 1)), math.sqrt(d8 / max(cnt, 1))))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
