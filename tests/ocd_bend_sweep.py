# -*- coding: utf-8 -*-
r"""角度扫描：凹陷随弯曲角如何变化？

若凹陷只在超过某角度后出现且单调增长，说明是算法的角度极限，
而不是某个可以修的 bug。

    mayapy tests/ocd_bend_sweep.py [case.json]
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
                       b["bindPosition"][0]
                       - bones[i - 1]["bindPosition"][0])
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

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        # 只统计真正在管壁上的顶点：排除两端封盖与轴线附近。
        Rbind = [math.hypot(p[1], p[2]) for p in bind]
        Rmax = max(Rbind)
        # 管壁顶点：半径接近最大且不在端部。
        ends = max(abs(p[0]) for p in bind)
        wall = [i for i, r in enumerate(Rbind)
                if r > 0.95 * Rmax and abs(bind[i][0]) < ends * 0.95]

        print("== 角度扫描（算法5）==")
        print("%8s %10s %10s %10s %10s"
              % ("bend", "最小径向", "塌陷", "关节区最小", "关节区塌陷"))
        prev = None
        for bend in (30.0, 60.0, 90.0, 110.0, 120.0, 130.0, 140.0, 146.0,
                     155.0, 165.0):
            holder, shape, skin, joints = build(mc, om, bind, faces, bones,
                                                sparse, "b%d" % int(bend))
            mc.setAttr(joints[1] + ".rotateZ", bend)
            mc.refresh(force=True)
            mc.dgeval(shape)
            pts = read(om, shape)

            radial = {i: math.hypot(pts[i][1], pts[i][2]) for i in wall}
            rmin = min(radial.values())
            # 关节区：世界 x 在 [-1.5, 1.5]
            near = {i: r for i, r in radial.items() if abs(bind[i][0]) <= 1.5}
            nmin = min(near.values()) if near else float("nan")
            print("%8.1f %10.4f %10.4f %10.4f %10.4f"
                  % (bend, rmin, rmin - Rmax, nmin, nmin - Rmax))
            mc.delete(holder)

        # 对照：算法0（纯CoR）
        print("\n== 对照：算法0 ==")
        print("%8s %10s %10s" % ("bend", "最小径向", "塌陷"))
        for bend in (60.0, 118.0, 146.0):
            holder, shape, skin, joints = build(mc, om, bind, faces, bones,
                                                sparse, "c%d" % int(bend))
            mc.setAttr(skin + ".algorithmType", 0)
            mc.setAttr(joints[1] + ".rotateZ", bend)
            mc.refresh(force=True)
            mc.dgeval(shape)
            pts = read(om, shape)
            rmin = min(math.hypot(pts[i][1], pts[i][2]) for i in wall)
            print("%8.1f %10.4f %10.4f" % (bend, rmin, rmin - Rmax))
            mc.delete(holder)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
