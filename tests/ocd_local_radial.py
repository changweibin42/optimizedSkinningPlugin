# -*- coding: utf-8 -*-
r"""在插件自己的铰链局部坐标系里度量凹陷。

插件已经把每个顶点转到「关节在原点、骨骼沿 +X、半径 1」的局部系。
在那个系里，radial = hypot(local.y, local.z) 才是真正的截面半径，
而且它由插件计算，口径不可能错。

做法：在 Maya 里重建夹具，读 final 后手工做同样的局部变换
（因为 C++ 不导出局部坐标，这里按同一公式复算），
再比较 bind / cor / final 三者的 local 径向。

    mayapy tests/ocd_local_radial.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def build(mc, om, bind, faces, bones, sparse, name):
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
    for i, b in enumerate(bones):
        if i == 0:
            n = mc.createNode("joint", name=name + "_j0", parent=holder)
            mc.setAttr(n + ".translateX", b["bindPosition"][0])
        else:
            n = mc.createNode("joint", name="%s_j%d" % (name, i),
                              parent=joints[-1])
            mc.setAttr(n + ".translateX",
                       b["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
        joints.append(n)
    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name=name + "_s#")[0]
    for i, (n, b) in enumerate(zip(joints, bones)):
        mc.connectAttr(n + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                   *[float(v) for v in b["bindPreMatrix"]], type="matrix")
    for v, row in enumerate(sparse):
        for k, val in row.items():
            mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k),
                       float(val))
    mc.setAttr(skin + ".algorithmType", 5)
    return holder, shape, skin, joints


def read(om, shape):
    sel = om.MSelectionList()
    sel.add(shape)
    return [[p.x, p.y, p.z] for p in
            om.MFnMesh(sel.getDagPath(0)).getPoints(om.MSpace.kObject)]


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

        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = child["bindPosition"]
        axis = d["hinge"]["axis"]
        n_ax = math.sqrt(sum(v * v for v in axis))
        axis = [v / n_ax for v in axis]

        # 半径（插件公式）：max |relative - axis*station|
        def radial_len(p):
            rel = [p[i] - pivot[i] for i in range(3)]
            s = sum(rel[i] * axis[i] for i in range(3))
            return math.sqrt(sum((rel[i] - axis[i] * s) ** 2 for i in range(3)))

        R = max(radial_len(p) for p in bind)
        print("maxRadius = %.6f" % R)

        def to_local(p):
            rel = [p[i] - pivot[i] for i in range(3)]
            s = sum(rel[i] * axis[i] for i in range(3))
            # basisY=+Y, basisZ=+Z
            return (s / R, rel[1] / R, rel[2] / R)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        results = {}
        for algo, label in ((0, "cor"), (3, "ocd"), (5, "final")):
            holder, shape, skin, joints = build(mc, om, bind, faces, bones,
                                                sparse, label)
            mc.setAttr(skin + ".algorithmType", algo)
            mc.setAttr(joints[1] + ".rotateZ", 146.126)
            mc.refresh(force=True)
            mc.dgeval(shape)
            results[label] = read(om, shape)
            mc.delete(holder)

        # 局部径向
        def local_radial(pts):
            out = []
            for p in pts:
                l = to_local(p)
                out.append(math.sqrt(l[1] ** 2 + l[2] ** 2))
            return out

        Rb = local_radial(bind)
        Rc = local_radial(results["cor"])
        Ro = local_radial(results["ocd"])
        Rf = local_radial(results["final"])

        print("== 局部径向（插件口径，bind 时应恒为 1）==")
        print("  bind:  min=%.5f max=%.5f" % (min(Rb), max(Rb)))

        x = [p[0] for p in bind]
        print("\n%9s %10s %10s %10s %10s %10s"
              % ("x", "Rbind", "Rcor", "Rocd", "Rfin", "fin-bind"))
        for target in [i * 0.5 for i in range(-12, 13)]:
            sel = [j for j in range(len(bind)) if abs(x[j] - target) < 0.031]
            if len(sel) < 10:
                continue
            m = lambda arr: sum(arr[j] for j in sel) / len(sel)
            print("%9.2f %10.4f %10.4f %10.4f %10.4f %10.4f"
                  % (target, m(Rb), m(Rc), m(Ro), m(Rf),
                     m([Rf[j] - Rb[j] for j in range(len(bind))])))

        dd = [Rf[j] - Rb[j] for j in range(len(bind))]
        print("\n== 极值 ==")
        print("  fin-bind min=%+.4f max=%+.4f" % (min(dd), max(dd)))
        i = dd.index(min(dd))
        print("  最凹 v%d x=%.3f Rbind=%.4f Rcor=%.4f Rfin=%.4f"
              % (i, x[i], Rb[i], Rc[i], Rf[i]))
        near = [j for j in range(len(bind)) if abs(x[j]) <= 2.0]
        print("\n关节区 |x|<=2: fin-bind min=%+.4f max=%+.4f"
              % (min(dd[j] for j in near), max(dd[j] for j in near)))
        j = min(near, key=lambda k: dd[k])
        print("  关节区最凹 v%d x=%.3f Rbind=%.4f Rcor=%.4f Rfin=%.4f"
              % (j, x[j], Rb[j], Rc[j], Rf[j]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
