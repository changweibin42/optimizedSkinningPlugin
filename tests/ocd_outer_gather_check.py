# -*- coding: utf-8 -*-
"""ocdOuterGather 验收：外轮廓是否变饱满。

    mayapy tests/ocd_outer_gather_check.py
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
        mc.setAttr(sk + ".algorithmType", 5)
        mc.setAttr(sk + ".ocdEnableCenterLimit", False)
        for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(a, node=sk, exists=True):
                mc.setAttr("%s.%s" % (sk, a), 0.0)
        mc.setAttr(js[1] + ".rotateZ", -147.2124)

        ok = mc.attributeQuery("ocdOuterGather", node=sk, exists=True)
        print("属性 ocdOuterGather 存在: %s  默认 %s"
              % (ok, mc.getAttr(sk + ".ocdOuterGather") if ok else "N/A"))
        if not ok:
            return 1

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

        rings = [round(p[0] / STEP) for p in bind]
        out = {}
        for gv in (0.85, 0.4, 0.0):
            mc.setAttr(sk + ".ocdOuterGather", float(gv))
            out[gv] = read()

        print("\n外极点 y（越小越靠外，越接近 CoR = 越饱满）")
        print("%8s %10s %10s %10s %10s"
              % ("x", "CoR", "G=0.85", "G=0.40", "G=0.00"))
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            m2 = [i for i in range(n) if rings[i] == rid]
            i_o = min(m2, key=lambda i: cor[i][1])
            yc = cor[i_o][1]
            y85 = out[0.85][i_o][1]
            y40 = out[0.4][i_o][1]
            y00 = out[0.0][i_o][1]
            print("%8.3f %10.5f %10.5f %10.5f %10.5f"
                  % (x0, yc, y85, y40, y00))

        # 与 CoR 的偏差统计
        print("\n与 CoR 外轮廓的偏差")
        rows = []
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            m2 = [i for i in range(n) if rings[i] == rid]
            i_o = min(m2, key=lambda i: cor[i][1])
            rows.append((x0, cor[i_o][1], out[0.85][i_o][1],
                         out[0.4][i_o][1], out[0.0][i_o][1]))
        for k, gname in ((1, "G=0.85"), (2, "G=0.40"), (3, "G=0.00")):
            dev = [abs(r[k] - r[1]) for r in rows]
            rms = math.sqrt(sum(d * d for d in dev) / len(dev))
            print("  %-8s 均=%.5f  max=%.5f  RMS=%.5f"
                  % (gname, sum(dev) / len(dev), max(dev), rms))

        # 内侧不应变化
        print("\n内侧极点 y（应基本不变）")
        print("%8s %10s %10s %10s" % ("x", "G=0.85", "G=0.40", "G=0.00"))
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            m2 = [i for i in range(n) if rings[i] == rid]
            i_i = max(m2, key=lambda i: cor[i][1])
            print("%8.3f %10.5f %10.5f %10.5f"
                  % (x0, out[0.85][i_i][1], out[0.4][i_i][1], out[0.0][i_i][1]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
