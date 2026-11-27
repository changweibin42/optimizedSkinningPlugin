# -*- coding: utf-8 -*-
r"""B 阶段：把 gate 做成正式属性后，确认开关属性确实生效。

    mayapy tests/ocd_gate_attribute.py D:/tmp/ocd_case3.json
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

        src = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(src))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)
        bend = 146.1265

        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = child["bindPosition"]
        axis = d["hinge"]["axis"]
        na = math.sqrt(sum(v * v for v in axis))
        axis = [v / na for v in axis]

        def rl(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return math.sqrt(sum((r[i] - axis[i] * s) ** 2 for i in range(3)))

        R = max(rl(p) for p in bind)
        xw = [p[0] for p in bind]

        def to_local(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return (s / R, r[1] / R, r[2] / R)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        holder = mc.createNode("transform", name="ga_h")
        geo = mc.createNode("transform", name="ga_g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        mc.sets(geo, edit=True, forceElement="initialShadingGroup")
        joints = []
        for i, bn in enumerate(bones):
            if i == 0:
                nd = mc.createNode("joint", name="ga_j0", parent=holder)
                mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
            else:
                nd = mc.createNode("joint", name="ga_j%d" % i, parent=joints[-1])
                mc.setAttr(nd + ".translateX",
                           bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
            joints.append(nd)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="ga_s#")[0]
        for i, (nd, bn) in enumerate(zip(joints, bones)):
            mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                       *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
        for v, row in enumerate(sparse):
            for k, val in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k),
                           float(val))
        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(joints[1] + ".rotateZ", bend)
        mc.refresh(force=True)

        print("== 属性存在性 ==")
        for name in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            exists = mc.attributeQuery(name, node=skin, exists=True)
            val = mc.getAttr("%s.%s" % (skin, name)) if exists else None
            print("  %-20s exists=%s  value=%s" % (name, exists, val))

        def read():
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def stats(pts):
            out = {}
            for x0 in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
                s = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
                if len(s) < 10:
                    continue
                rr = []
                for i in s:
                    l = to_local(pts[i])
                    rr.append(math.sqrt(l[1] ** 2 + l[2] ** 2))
                m = sum(rr) / len(rr)
                v = sum((q - m) ** 2 for q in rr) / len(rr)
                out[x0] = (math.sqrt(v) / max(1e-9, m), m)
            return out

        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
        print("\n== 通过属性开关 gate：圆度 ==")
        print("%18s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label, lo, hi in (("关闭 (0,0)", 0.0, 0.0),
                              ("0.35 / 0.50", 0.35, 0.50),
                              ("0.45 / 0.60", 0.45, 0.60)):
            mc.setAttr(skin + ".ocdCenterGateLow", lo)
            mc.setAttr(skin + ".ocdCenterGateHigh", hi)
            mc.dgeval(shape)
            mc.refresh(force=True)
            st = stats(read())
            line = "%18s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][0] if x0 in st else "      -"
            print(line)

        print("\n== 径向均值 ==")
        print("%18s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label, lo, hi in (("关闭 (0,0)", 0.0, 0.0),
                              ("0.35 / 0.50", 0.35, 0.50),
                              ("0.45 / 0.60", 0.45, 0.60)):
            mc.setAttr(skin + ".ocdCenterGateLow", lo)
            mc.setAttr(skin + ".ocdCenterGateHigh", hi)
            mc.dgeval(shape)
            mc.refresh(force=True)
            st = stats(read())
            line = "%18s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][1] if x0 in st else "      -"
            print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
