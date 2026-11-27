# -*- coding: utf-8 -*-
"""扫描 gate 阈值，用插件真实输出测圆度。

    mayapy tests/ocd_gate_sweep.py
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
        xw = [p[0] for p in bind]

        def to_local(p):
            r = [p[i] - pivot[i] for i in range(3)]
            s = sum(r[i] * axis[i] for i in range(3))
            return (s / R, r[1] / R, r[2] / R)

        def build(name, gate_env=None):
            if gate_env is None:
                os.environ.pop("DSH_OCD_GATE", None)
            else:
                os.environ["DSH_OCD_GATE"] = gate_env
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
            return holder, shape

        def read(shape):
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def stats(pts):
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

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
        print("== gate 扫描：圆度（越小越圆）==")
        print("%14s" % "" + "".join("  x=%+.1f" % v for v in xs))

        configs = [("off", None), ("0.20,0.35", "0.20,0.35"),
                   ("0.35,0.50", "0.35,0.50"), ("0.40,0.55", "0.40,0.55"),
                   ("0.45,0.60", "0.45,0.60"), ("0.50,0.70", "0.50,0.70")]
        for label, env in configs:
            holder, shape = build("g" + label.replace(",", "_").replace(".", ""),
                                  env)
            st = stats(read(shape))
            line = "%14s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][0] if x0 in st else "      -"
            print(line)
            mc.delete(holder)

        print("\n== 同一批的径向均值 ==")
        print("%14s" % "" + "".join("  x=%+.1f" % v for v in xs))
        for label, env in configs:
            holder, shape = build("r" + label.replace(",", "_").replace(".", ""),
                                  env)
            st = stats(read(shape))
            line = "%14s" % label
            for x0 in xs:
                line += " %6.3f" % st[x0][1] if x0 in st else "      -"
            print(line)
            mc.delete(holder)

        # 解析模型对照
        os.environ.pop("DSH_OCD_GATE", None)
        analytic = [list(p) for p in
                    ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False).mesh(bind)]
        st = stats(analytic)
        line = "%14s" % "解析模型"
        for x0 in xs:
            line += " %6.3f" % st[x0][0] if x0 in st else "      -"
        print(line)
        line = "%14s" % "解析径向"
        for x0 in xs:
            line += " %6.3f" % st[x0][1] if x0 in st else "      -"
        print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
