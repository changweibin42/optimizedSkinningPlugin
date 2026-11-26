# -*- coding: utf-8 -*-
r"""绑定之后转动/平移父骨，肘部形状必须刚体跟随。

这是审计里指出的第一个隔离测试。旧实现把「已随父骨运动的世界轴」投影到
「绑定基」上，所以父骨一转，direction / 弯面 / 内外侧参与度都会变；算法 3
的目标还会留在绑定位置不跟随。

做法：绑定时父子骨骼都不动，之后只旋转父骨（prev），保持肘关节的局部弯曲
不变。正确行为是网格输出在**世界空间**里等于同一刚体变换作用在原输出上，
也就是肘部折痕随整条手臂一起走，局部形状不变。

用法：
    mayapy tests/maya_ocd_parent_anim.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import ocd_elbow_reference as ref

TOLERANCE = 2.0e-3
BEND = 118.0
# 父骨（肩）在绑定之后的位姿：(translate, rotate度)。
PARENT_POSES = (
    ((0.0, 0.0, 0.0), (0.0, 0.0, 30.0)),
    ((0.0, 0.0, 0.0), (25.0, -40.0, 15.0)),
    ((4.0, -3.0, 6.0), (25.0, -40.0, 15.0)),
)


def build_rig(mc, om, bind, faces):
    holder = mc.createNode("transform", name="par_holder#")
    geo = mc.createNode("transform", name="par_geo#", parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True, fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")
    prev = mc.createNode("joint", name="par_shoulder#", parent=holder)
    elbow = mc.createNode("joint", name="par_elbow#", parent=prev)
    tip = mc.createNode("joint", name="par_tip#", parent=elbow)
    for joint, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
        mc.setAttr(joint + ".translateX", dx)
        mc.setAttr(joint + ".radius", 0.09)
    return {"holder": holder, "geo": geo, "shape": shape,
            "prev": prev, "elbow": elbow}


def attach_skin(mc, rig, algorithm, bind):
    skin = mc.deformer(rig["shape"], type="optimizedSkinCluster",
                       name="par_skin#")[0]
    # bindPreMatrix 只在绑定时取一次：这一版故意不随父骨更新。
    for k, joint in enumerate((rig["prev"], rig["elbow"])):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, (station, _, _) in enumerate(bind):
        w = ref.skin_weight(station)
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, vertex),
                   1.0 - w, w, size=2)
    mc.setAttr(skin + ".algorithmType", algorithm)
    result = mc.listRelatives(rig["geo"], shapes=True,
                              noIntermediate=True, fullPath=True)[0]
    return skin, result


def world_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    return [om.MPoint(p) for p in
            om.MFnMesh(selection.getDagPath(0)).getPoints(om.MSpace.kWorld)]


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()

        for algorithm, label in ((0, "CoR(0)"), (3, "OCD(3)"), (5, "OCD+CoR(5)")):
            rig = build_rig(mc, om, bind, faces)
            _, result = attach_skin(mc, rig, algorithm, bind)

            # 绑定时父骨不动，只弯肘。
            selection = om.MSelectionList()
            selection.add(rig["elbow"])
            om.MFnTransform(selection.getDagPath(0)).setRotation(
                om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)),
                om.MSpace.kTransform)
            mc.refresh(force=True)
            base = world_points(om, result)

            shoulder = rig["prev"]
            # worldMatrix 返回 16 个浮点，行主序；平移在 12..14。
            flat = mc.getAttr(shoulder + ".worldMatrix[0]")
            pivot = om.MPoint(flat[12], flat[13], flat[14])
            # 骨骼自己的 translateX 已经是 -HALF，不能直接覆盖。
            bind_translate = mc.getAttr(shoulder + ".translate")[0]

            for translate, rotation in PARENT_POSES:
                # 父骨在绑定之后才动：bindPreMatrix 保持绑定时的值不变。
                mc.setAttr(shoulder + ".translate",
                           bind_translate[0] + translate[0],
                           bind_translate[1] + translate[1],
                           bind_translate[2] + translate[2])
                mc.setAttr(shoulder + ".rotate",
                           rotation[0], rotation[1], rotation[2])
                mc.refresh(force=True)
                moved = world_points(om, result)

                euler = om.MEulerRotation(
                    math.radians(rotation[0]), math.radians(rotation[1]),
                    math.radians(rotation[2]), om.MEulerRotation.kXYZ)
                rot = euler.asMatrix()
                expected = []
                for point in base:
                    relative = point - pivot
                    rotated = relative * rot
                    expected.append(om.MPoint(rotated.x + pivot.x,
                                              rotated.y + pivot.y,
                                              rotated.z + pivot.z))

                expected = [om.MPoint(p.x + translate[0], p.y + translate[1],
                                      p.z + translate[2]) for p in expected]
                worst = 0.0
                for got, want in zip(moved, expected):
                    delta = got - want
                    worst = max(worst, math.sqrt(delta.x * delta.x +
                                                 delta.y * delta.y +
                                                 delta.z * delta.z))
                ok = worst <= TOLERANCE
                print("%-12s parent t=%s r=%s  刚体跟随偏差 %.6f  %s"
                      % (label, translate, rotation, worst,
                         "OK" if ok else "MISMATCH"))
                if not ok:
                    failures.append("%s t=%s r=%s 偏差 %.3e"
                                    % (label, translate, rotation, worst))
                mc.setAttr(shoulder + ".translate", *bind_translate)
                mc.setAttr(shoulder + ".rotate", 0, 0, 0)
                mc.refresh(force=True)

            mc.delete(rig["holder"])

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            print("FAILURES:", len(failures))
            return 1
        print("PASS: 绑定后转动父骨，三个算法的肘部形状均刚体跟随")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
