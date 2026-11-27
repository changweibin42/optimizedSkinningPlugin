# -*- coding: utf-8 -*-
r"""在导出夹具上重建场景，读 ocdGuideDump / ocdDeltaDump，定位凹陷归属。

    mayapy tests/ocd_delta_dump.py D:/tmp/ocd_case3.json out.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.getcwd(), "tests", "_delta_dump.json")
        d = json.load(open(source))

        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        holder = mc.createNode("transform", name="dd_holder")
        geo = mc.createNode("transform", name="dd_geo", parent=holder)
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
        for index, bone in enumerate(bones):
            if index == 0:
                node = mc.createNode("joint", name=bone["short"], parent=holder)
                mc.setAttr(node + ".translateX", bone["bindPosition"][0])
            else:
                node = mc.createNode("joint", name=bone["short"],
                                     parent=joints[-1])
                local = (bone["bindPosition"][0]
                         - bones[index - 1]["bindPosition"][0])
                mc.setAttr(node + ".translateX", local)
            mc.setAttr(node + ".radius", 0.09)
            joints.append(node)

        skin = mc.deformer(shape, type="optimizedSkinCluster",
                           name="dd_skin#")[0]
        for index, (node, bone) in enumerate(zip(joints, bones)):
            mc.connectAttr(node + ".worldMatrix[0]",
                           "%s.matrix[%d]" % (skin, index))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, index),
                       *[float(v) for v in bone["bindPreMatrix"]], type="matrix")
        for vertex, row in enumerate(sparse):
            for key, value in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]"
                           % (skin, vertex, key), float(value))

        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".corDebugLog", True)
        mc.setAttr(skin + ".ocdCenterLimit", 16.0)

        # 姿态：joint1 不动，L_Ankle_jnt 转 146.126 度（绕 Z）
        mc.setAttr(joints[1] + ".rotateZ", 146.126)
        mc.refresh(force=True)
        mc.dgeval(shape)

        # 与导出对比，确认复现
        got = [tuple(p) for p in om.MFnMesh(
            (lambda s: (s.add(shape), s.getDagPath(0))[1])(
                om.MSelectionList())).getPoints(om.MSpace.kObject)]
        want = d["points"]["final"]
        worst = max(dist(g, w) for g, w in zip(got, want))
        print("复现偏差 max = %.6e" % worst)

        guide = list(mc.getAttr(skin + ".ocdGuideDump") or [])
        delta = list(mc.getAttr(skin + ".ocdDeltaDump") or [])
        limit = list(mc.getAttr(skin + ".ocdCenterLimitDump") or [])
        print("guide 长度=%d  delta 长度=%d  limit=%s"
              % (len(guide), len(delta), limit))

        n = len(bind)
        guide_pts = [[guide[3 * i], guide[3 * i + 1], guide[3 * i + 2]]
                     for i in range(n)]
        delta_pts = [[delta[3 * i], delta[3 * i + 1], delta[3 * i + 2]]
                     for i in range(n)]

        gm = [math.sqrt(sum(v * v for v in g)) for g in guide_pts]
        dm = [math.sqrt(sum(v * v for v in g)) for g in delta_pts]
        print("|guide| max=%.6f mean=%.6f" % (max(gm), sum(gm) / len(gm)))
        print("|delta| max=%.6f mean=%.6f" % (max(dm), sum(dm) / len(dm)))

        json.dump({"guide": guide_pts, "delta": delta_pts, "limit": limit},
                  open(target, "w"))
        print("wrote", target)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
