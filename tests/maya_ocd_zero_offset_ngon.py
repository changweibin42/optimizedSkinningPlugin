# -*- coding: utf-8 -*-
r"""零偏移一致性：算法 5（显式注入全零偏移）必须等于算法 0。

这是三角化统一之后的守门测试。算法 0 用 Maya 的 getTriangles() 做 CoR
积分，算法 5 之前从 face[0] 扇形三角化；四边面上两者选同一条对角线，
n-gon 上则不一定。这里用**带八边形端盖的圆柱**（真实 n-gon）来暴露差异。

不含 numpy，直接跑：
    mayapy tests/maya_ocd_zero_offset_ngon.py
可加 --quad 只测四边面圆柱作对照。
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
BENDS = (118.0,)


def ngon_cylinder(half_rings=9, sides=8, star=False):
    """与参考圆柱同样的管身，但两端是单个 n 边形端盖。

    star=True 时截面半径交替（星形），端盖变成**凹**多边形。凹 n-gon 上
    Maya 的 getTriangles() 会做耳切，与 face[0] 扇形三角化选出不同的三角形，
    因此这个用例才有区分力。
    """
    points, faces = [], []
    rings = 2 * half_rings + 1
    for row in range(rings):
        s = ref.HALF * (row - half_rings) / half_rings
        for index in range(sides):
            angle = 2.0 * math.pi * index / sides
            radius = 1.0
            if star:
                radius = 1.0 if index % 2 == 0 else 0.55
            points.append((s, radius * math.cos(angle), radius * math.sin(angle)))
    for row in range(rings - 1):
        for index in range(sides):
            a = row * sides + index
            b = row * sides + (index + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    last_ring = (rings - 1) * sides
    start_cap = [index for index in range(sides - 1, -1, -1)]
    end_cap = [last_ring + index for index in range(sides)]
    faces.append(tuple(start_cap))
    faces.append(tuple(end_cap))
    return points, faces


def build(mc, om, bind, faces, label):
    holder = mc.createNode("transform", name="zo_%s_holder#" % label)
    geo = mc.createNode("transform", name="zo_%s_geo#" % label, parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True, fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")
    prev = mc.createNode("joint", name="zo_%s_prev#" % label, parent=holder)
    elbow = mc.createNode("joint", name="zo_%s_elbow#" % label, parent=prev)
    tip = mc.createNode("joint", name="zo_%s_tip#" % label, parent=elbow)
    for joint, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
        mc.setAttr(joint + ".translateX", dx)
        mc.setAttr(joint + ".radius", 0.09)
    return {"holder": holder, "geo": geo, "shape": shape,
            "prev": prev, "elbow": elbow}


def asymmetric_weight(station, vertex):
    """同一环内权重不同：积分才依赖三角化。

    端盖顶点同属一个环，若权重只随 station 变化，各三角形贡献相同，
    总面积即可决定积分，fan 与耳切会给出同一个中心。这里按顶点号扰动量级，
    让三角化选择真正影响 computeCenters。
    """
    base = ref.skin_weight(station)
    jitter = 0.45 * ((vertex * 13) % 17) / 17.0
    return min(0.98, max(0.02, base * (0.55 + jitter)))


def attach(mc, rig, algorithm, bind, asymmetric=False):
    skin = mc.deformer(rig["shape"], type="optimizedSkinCluster",
                       name="zo_skin#")[0]
    for k, joint in enumerate((rig["prev"], rig["elbow"])):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, (station, _, _) in enumerate(bind):
        w = (asymmetric_weight(station, vertex) if asymmetric
             else ref.skin_weight(station))
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, vertex),
                   1.0 - w, w, size=2)
    mc.setAttr(skin + ".algorithmType", algorithm)
    result = mc.listRelatives(rig["geo"], shapes=True,
                              noIntermediate=True, fullPath=True)[0]
    return skin, result


def set_zero_offsets(om, skin, count):
    selection = om.MSelectionList()
    selection.add(skin)
    plug = om.MFnDependencyNode(selection.getDependNode(0)).findPlug(
        "ocdCenterOffsets", False)
    values = om.MDoubleArray([0.0] * (3 * count))
    plug.setMObject(om.MFnDoubleArrayData().create(values))


def read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [(p.x, p.y, p.z) for p in mesh.getPoints(om.MSpace.kObject)]


def worst_deviation(first, second):
    worst = 0.0
    for a, b in zip(first, second):
        d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        worst = max(worst, d)
    return worst


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        shapes = {"ngon": ngon_cylinder(),
                  "star": ngon_cylinder(star=True)}
        if "--quad" in sys.argv:
            shapes["quad"] = ref.cylinder()

        for name, (bind, faces) in shapes.items():
            asymmetric = name.startswith("star")
            ngon_count = sum(1 for face in faces if len(face) > 4)
            print("mesh=%s  verts=%d faces=%d  n-gon faces=%d  asymmetric=%s"
                  % (name, len(bind), len(faces), ngon_count, asymmetric))
            for bend in BENDS:
                cor = build(mc, om, bind, faces, "cor_" + name)
                _, cor_result = attach(mc, cor, 0, bind, asymmetric)
                mc.setAttr(cor["elbow"] + ".rotateZ", bend)
                mc.refresh(force=True)
                cor_points = read_points(om, cor_result)

                zero = build(mc, om, bind, faces, "zero_" + name)
                zero_skin, zero_result = attach(mc, zero, 5, bind, asymmetric)
                mc.setAttr(zero["elbow"] + ".rotateZ", bend)
                set_zero_offsets(om, zero_skin, len(bind))
                mc.refresh(force=True)
                zero_points = read_points(om, zero_result)

                deviation = worst_deviation(zero_points, cor_points)
                ok = deviation <= TOLERANCE
                print("  bend=%5.1f  零偏移 algo5 vs algo0 偏差 %.6f  %s"
                      % (bend, deviation, "OK" if ok else "MISMATCH"))
                if not ok:
                    failures.append("%s bend=%.0f 偏差 %.3e"
                                    % (name, bend, deviation))
                mc.delete(cor["holder"])
                mc.delete(zero["holder"])

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            print("FAILURES:", len(failures))
            return 1
        print("PASS: n-gon 上零偏移算法 5 与算法 0 的顶点输出一致")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
