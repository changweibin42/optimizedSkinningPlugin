# -*- coding: utf-8 -*-
"""support 对照：原 support≈4.06 vs support=2.6，只比两处。

  · x=-2.0 的 42.16° 尖峰是否下降
  · 相邻圈是否出现新尖峰

    mayapy tests/support_compare.py
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

        d1 = json.load(open(os.path.join(HERE, "_la1.json")))
        d2 = json.load(open(os.path.join(HERE, "_la2_fixed.json")))
        bind = [tuple(p) for p in d1["bind"]]
        faces = d1["faces"]
        n = len(bind)
        HALF = d1["HALF"]
        weights = d1["weights"]
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1

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
        prev = mc.createNode("joint", name="prev", parent=holder)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
        for i, w in enumerate(weights):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       float(w[0]), float(w[1]), size=2)
        for attr in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(attr, node=skin, exists=True):
                mc.setAttr("%s.%s" % (skin, attr), 0.0)

        def read(algo, inj=None):
            mc.setAttr(skin + ".algorithmType", algo)
            s = om.MSelectionList()
            s.add(skin)
            om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                "ocdCenterOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(
                        om.MDoubleArray(inj if inj is not None else [])))
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shape)
            s2 = om.MSelectionList()
            s2.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s2.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def turning(P, row, j):
            if row <= 0 or row >= RINGS - 1:
                return None
            a = P[(row - 1) * SIDES + j]
            b = P[row * SIDES + j]
            c = P[(row + 1) * SIDES + j]
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - b[i] for i in range(3)]
            lu = math.sqrt(sum(x * x for x in u))
            lv = math.sqrt(sum(x * x for x in v))
            if lu < 1e-12 or lv < 1e-12:
                return None
            dd = sum(u[i] * v[i] for i in range(3)) / (lu * lv)
            return math.degrees(math.acos(max(-1.0, min(1.0, dd))))

        print("最终经线转角对比（row 为中心）")
        for bend_s in ("118.0", "120.0", "146.1265"):
            bend = float(bend_s)
            mc.setAttr(elbow + ".rotateZ", bend)
            dd = d2[bend_s]
            f_raw = read(5, dd["deltaRaw"])
            f_unit = read(5, dd["deltaUnit"])

            print("\n" + "=" * 70)
            print("bend=%.4f" % bend)
            print("=" * 70)
            for j, jn in ((0, "内侧 j=0"), (SIDES // 2, "外侧 j=%d" % (SIDES // 2))):
                vals = []
                for row in range(1, RINGS - 1):
                    t = turning(f_raw, row, j)
                    if t is None:
                        continue
                    x0 = -HALF + row * (2 * HALF / (RINGS - 1))
                    vals.append((x0, t))
                mx = max(vals, key=lambda p: p[1])
                print("  %s: 原梯度 max=%.3f° @x=%+.3f" % (jn, mx[1], mx[0]))

            # 逐环比 x=-2.0 附近
            print("\n  内侧 j=0 逐环（原梯度 → 单位梯度）：")
            print("  %8s %8s %10s %10s %10s" % ("x", "w1", "原梯度", "单位梯度", "差值"))
            for row in range(1, RINGS - 1):
                x0 = -HALF + row * (2 * HALF / (RINGS - 1))
                if abs(x0) > 2.7:
                    continue
                t0 = turning(f_raw, row, 0)
                t1 = turning(f_unit, row, 0)
                if t0 is None or t1 is None:
                    continue
                print("  %8.3f %8.4f %10.3f %10.3f %+10.3f"
                      % (x0, weights[row * SIDES][1], t0, t1, t1 - t0))

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
