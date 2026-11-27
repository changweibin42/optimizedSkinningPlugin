# -*- coding: utf-8 -*-
"""确认 ocdCenterOffsets 注入是否真的生效。

λ=0 和 λ=1 的径向完全相同，可疑。检查：
  1) 注入全零 vs 注入空数组 vs 不注入 三者是否不同
  2) 注入 2*delta 是否与 delta 不同
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

        src = r"D:\tmp\ocd_case3.json"
        d = json.load(open(src))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)
        bend = 146.1265
        stage = json.load(open(os.path.join(HERE, "_stage1.json")))
        delta_local = stage["cppDelta"]

        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = child["bindPosition"]
        axis = d["hinge"]["axis"]
        na = math.sqrt(sum(v * v for v in axis))
        axis = [v / na for v in axis]
        bY, bZ = [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]

        def rl(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return math.sqrt(sum((r[i] - axis[i] * s) ** 2 for i in range(3)))

        R = max(rl(p) for p in bind)
        xw = [p[0] for p in bind]

        obj = []
        for i in range(n):
            lx, ly, lz = (delta_local[3 * i], delta_local[3 * i + 1],
                          delta_local[3 * i + 2])
            for k in range(3):
                obj.append(R * (lx * axis[k] + ly * bY[k] + lz * bZ[k]))

        def to_local(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return (s / R, r[1] / R, r[2] / R)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        def build(name):
            holder = mc.createNode("transform", name=name + "_h")
            geo = mc.createNode("transform", name=name + "_g", parent=holder)
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
                    nd = mc.createNode("joint", name=name + "_j0", parent=holder)
                    mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
                else:
                    nd = mc.createNode("joint", name="%s_j%d" % (name, i),
                                       parent=joints[-1])
                    mc.setAttr(nd + ".translateX",
                               bn["bindPosition"][0]
                               - bones[i - 1]["bindPosition"][0])
                joints.append(nd)
            skin = mc.deformer(shape, type="optimizedSkinCluster",
                               name=name + "_s#")[0]
            for i, (nd, bn) in enumerate(zip(joints, bones)):
                mc.connectAttr(nd + ".worldMatrix[0]",
                               "%s.matrix[%d]" % (skin, i))
                mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                           *[float(v) for v in bn["bindPreMatrix"]],
                           type="matrix")
            for v, row in enumerate(sparse):
                for k, val in row.items():
                    mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k),
                               float(val))
            mc.setAttr(skin + ".algorithmType", 5)
            mc.setAttr(joints[1] + ".rotateZ", bend)
            mc.refresh(force=True)
            mc.dgeval(shape)
            return holder, shape, skin

        def read(shape):
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def inject(skin, values):
            s = om.MSelectionList()
            s.add(skin)
            om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                "ocdCenterOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(values)))
            mc.refresh(force=True)

        def r_at(pts, x0):
            sel = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
            rr = []
            for i in sel:
                l = to_local(pts[i])
                rr.append(math.sqrt(l[1] ** 2 + l[2] ** 2))
            return sum(rr) / len(rr)

        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
        tests = [
            ("不注入", None),
            ("注入空数组", []),
            ("注入全零", [0.0] * (3 * n)),
            ("注入 1*delta", list(obj)),
            ("注入 2*delta", [2.0 * v for v in obj]),
            ("注入 -1*delta", [-v for v in obj]),
        ]
        print("%16s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label, values in tests:
            holder, shape, skin = build("t" + str(abs(hash(label)) % 10000))
            if values is not None:
                inject(skin, values)
            pts = read(shape)
            line = "%16s" % label
            for x0 in xs:
                line += " %7.4f" % r_at(pts, x0)
            print(line)
            mc.delete(holder)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
