# -*- coding: utf-8 -*-
"""注入全零 delta，看圆度是否改善。用插件真实输出度量。

如果置零后圆度 = 解析模型，说明 delta 是圆度损失的来源；
如果置零后仍然差，说明问题在 CoR 求值本身。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def build(mc, om, bind, faces, bones, sparse, name, bend):
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
                       bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
        joints.append(nd)
    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name=name + "_s#")[0]
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
    mc.dgeval(shape)
    return holder, shape, skin, joints


def read(om, shape):
    sel = om.MSelectionList()
    sel.add(shape)
    return [[p.x, p.y, p.z] for p in
            om.MFnMesh(sel.getDagPath(0)).getPoints(om.MSpace.kObject)]


def inject(mc, om, skin, values):
    sel = om.MSelectionList()
    sel.add(skin)
    nd = om.MFnDependencyNode(sel.getDependNode(0))
    nd.findPlug("ocdCenterOffsets", False).setMObject(
        om.MFnDoubleArrayData().create(om.MDoubleArray(values)))
    mc.refresh(force=True)


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

        def to_local(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return (s / R, r[1] / R, r[2] / R)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        cases = {}
        # 自动
        h, sh, sk, jt = build(mc, om, bind, faces, bones, sparse, "auto", bend)
        cases["auto"] = read(om, sh)
        mc.delete(h)
        # 注入全零
        h, sh, sk, jt = build(mc, om, bind, faces, bones, sparse, "zero", bend)
        inject(mc, om, sk, [0.0] * (3 * n))
        cases["zero"] = read(om, sh)
        mc.delete(h)
        # 注入 C++ delta（应复现 auto）
        stage = json.load(open(os.path.join(HERE, "_stage1.json")))
        h, sh, sk, jt = build(mc, om, bind, faces, bones, sparse, "cppd", bend)
        inject(mc, om, sk, stage["cppDelta"])
        cases["delta"] = read(om, sh)
        mc.delete(h)

        xw = [p[0] for p in bind]

        def ring_stats(pts):
            out = {}
            for x0 in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
                sel = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
                if len(sel) < 10:
                    continue
                rr = []
                for i in sel:
                    l = to_local(pts[i])
                    rr.append(math.sqrt(l[1] ** 2 + l[2] ** 2))
                mean = sum(rr) / len(rr)
                var = sum((v - mean) ** 2 for v in rr) / len(rr)
                out[x0] = (math.sqrt(var) / max(1e-9, mean), mean)
            return out

        print("== 圆度（插件真实输出，越小越圆）==")
        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
        print("%10s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label in ("auto", "zero", "delta"):
            st = ring_stats(cases[label])
            line = "%10s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][0] if x0 in st else "      -"
            print(line)

        print("\n== 径向均值 ==")
        print("%10s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label in ("auto", "zero", "delta"):
            st = ring_stats(cases[label])
            line = "%10s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][1] if x0 in st else "      -"
            print(line)

        # 解析模型对照
        import ocd_elbow_reference_latest as ref
        analytic = [list(p) for p in
                    ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False).mesh(bind)]
        st = ring_stats(analytic)
        line = "%10s" % "解析"
        for x0 in xs:
            line += " %6.3f" % st[x0][0] if x0 in st else "      -"
        print(line)
        line = "%10s" % "解析径向"
        for x0 in xs:
            line += " %6.3f" % st[x0][1] if x0 in st else "      -"
        print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
