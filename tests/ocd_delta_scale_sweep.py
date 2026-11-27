# -*- coding: utf-8 -*-
r"""按比例缩小 delta，看形状是否更接近解析模型。

假设：delta 过拟合（|delta|=1.59 而 |guide|=0.15），全局缩小应改善。
用注入通道 λ*delta，不改代码。

    mayapy tests/ocd_delta_scale_sweep.py D:/tmp/ocd_case3.json
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

        src = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(src))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)
        bend = 146.1265

        stage = json.load(open(os.path.join(HERE, "_stage1.json")))
        delta_local = stage["cppDelta"]          # 局部归一坐标，长度 3n
        assert len(delta_local) == 3 * n

        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = child["bindPosition"]
        axis = d["hinge"]["axis"]
        na = math.sqrt(sum(v * v for v in axis))
        axis = [v / na for v in axis]
        bY = [0.0, 1.0, 0.0]
        bZ = [0.0, 0.0, 1.0]

        def rl(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return math.sqrt(sum((r[i] - axis[i] * s) ** 2 for i in range(3)))

        R = max(rl(p) for p in bind)
        xw = [p[0] for p in bind]

        # 局部归一 -> 对象空间（worldMat 为单位阵，故对象空间即世界）
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

        # 解析模型（Python 里实际显示的）
        analytic = [list(p) for p in
                    ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False).mesh(bind)]

        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]

        def stats(pts):
            out = {}
            for x0 in xs:
                sel = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
                if len(sel) < 10:
                    continue
                rr = []
                for i in sel:
                    l = to_local(pts[i])
                    rr.append(math.sqrt(l[1] ** 2 + l[2] ** 2))
                m = sum(rr) / len(rr)
                v = sum((q - m) ** 2 for q in rr) / len(rr)
                out[x0] = (math.sqrt(v) / max(1e-9, m), m)
            return out

        st_ana = stats(analytic)

        print("== 注入 λ*delta：径向均值（括号内为圆度）==")
        print("%8s" % "λ" + "".join("   x=%+.1f      " % v for v in xs))

        base = None
        rows = []
        for lam in (0.0, 0.25, 0.5, 0.75, 1.0):
            holder, shape, skin = build("l%d" % int(lam * 100))
            if lam > 0.0:
                inject(skin, [lam * v for v in obj])
            pts = read(shape)
            st = stats(pts)
            if base is None:
                base = st
            line = "%8.2f" % lam
            for x0 in xs:
                if x0 in st:
                    line += " %6.3f(%5.3f)" % (st[x0][1], st[x0][0])
                else:
                    line += "        -      "
            print(line)
            rows.append((lam, st))
            mc.delete(holder)

        print("\n%8s" % "解析" + "".join("   x=%+.1f      " % v for v in xs))
        line = "%8s" % ""
        for x0 in xs:
            if x0 in st_ana:
                line += " %6.3f(%5.3f)" % (st_ana[x0][1], st_ana[x0][0])
            else:
                line += "        -      "
        print(line)

        # 与解析模型的总偏差
        print("\n== 与解析模型的径向总偏差 ==")
        for lam, st in rows:
            tot = sum(abs(st[x][1] - st_ana[x][1]) for x in xs if x in st and x in st_ana)
            totr = sum(abs(st[x][0] - st_ana[x][0]) for x in xs if x in st and x in st_ana)
            print("  λ=%.2f  径向偏差和=%.4f  圆度偏差和=%.4f" % (lam, tot, totr))
        tot = sum(abs(st_ana[x][1] - st_ana[x][1]) for x in xs)
        print("  （解析模型自身为 0）")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
