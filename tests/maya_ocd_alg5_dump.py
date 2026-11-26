# -*- coding: utf-8 -*-
r"""逐级对拍的一端：在 Maya 里导出 originalCoR，或注入真值后比较。

关键约束：**权重由本脚本决定并导出，truth 与 inject 都直接消费同一份**。
这样即使三骨有效权重的推导函数（preview_influences）与老的两骨权重不同，
两端也不会各用一套。boneCount、每个顶点的权重列都写进 JSON。

    # 1) 导出（--three-bone 表示用参考的 preview_influences 三骨权重）
    mayapy tests/maya_ocd_alg5_dump.py --dump tests/_alg5_original.json [--three-bone]

    # 2) numpy 环境算 guide 与 centerOffset
    python tests/ocd_alg5_truth.py tests/_alg5_original.json tests/_alg5_truth.json

    # 3) 注入真值，与插件自动求解比较（权重从 truth 里读，不需要再传 --three-bone）
    mayapy tests/maya_ocd_alg5_dump.py --inject tests/_alg5_truth.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import ocd_elbow_reference as ref
# 1285f9a 版才有 preview_influences（三骨有效权重）。
import ocd_elbow_reference_latest as ref_latest

BENDS = (118.0, 150.0)
TOLERANCE = 2.0e-3


# ----------------------------------------------------------------------
# 权重：唯一来源
# ----------------------------------------------------------------------

def vertex_weights(station, three_bone):
    """返回该顶点的权重元组；三骨时用参考的 preview_influences。"""
    if three_bone:
        return tuple(ref_latest.preview_influences(station, True))
    w = ref.skin_weight(station)
    return (1.0 - w, w)


def rig_weights(bind, three_bone):
    return [vertex_weights(p[0], three_bone) for p in bind]


# ----------------------------------------------------------------------
# 场景
# ----------------------------------------------------------------------

def build_rig(mc, om, bind, faces, label, bone_count):
    holder = mc.createNode("transform", name="dump_%s_holder#" % label)
    geo = mc.createNode("transform", name="dump_%s_geo#" % label, parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True, fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")
    prev = mc.createNode("joint", name="dump_%s_prev#" % label, parent=holder)
    elbow = mc.createNode("joint", name="dump_%s_elbow#" % label, parent=prev)
    tip = mc.createNode("joint", name="dump_%s_tip#" % label, parent=elbow)
    for joint, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
        mc.setAttr(joint + ".translateX", dx)
        mc.setAttr(joint + ".radius", 0.09)
    rig = {"holder": holder, "geo": geo, "shape": shape,
           "prev": prev, "elbow": elbow, "helper": None}
    if bone_count >= 3:
        # 参考实现的 helper：prev 的另一个子骨，位置与 elbow 重合。
        helper = mc.createNode("joint", name="dump_%s_helper#" % label, parent=prev)
        mc.setAttr(helper + ".translateX", ref.HALF)
        mc.setAttr(helper + ".radius", 0.09)
        rig["helper"] = helper
    return rig


def attach_skin(mc, rig, algorithm, weights):
    """按传入的权重矩阵写节点；boneCount 由权重列数决定。"""
    skin = mc.deformer(rig["shape"], type="optimizedSkinCluster",
                       name="dump_skin#")[0]
    joints = [rig["prev"], rig["elbow"]]
    if len(weights[0]) >= 3:
        joints.append(rig["helper"])
    for k, joint in enumerate(joints):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, row in enumerate(weights):
        mc.setAttr("%s.weightList[%d].weights[0:%d]" % (skin, vertex, len(row) - 1),
                   *row, size=len(row))
    mc.setAttr(skin + ".algorithmType", algorithm)
    result = mc.listRelatives(rig["geo"], shapes=True,
                              noIntermediate=True, fullPath=True)[0]
    return skin, result


def read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [[p.x, p.y, p.z] for p in mesh.getPoints(om.MSpace.kObject)]


def pose_elbow(mc, om, rig, bend, helper_rotation):
    selection = om.MSelectionList()
    selection.add(rig["elbow"])
    om.MFnTransform(selection.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(bend, 0, 0)), om.MSpace.kTransform)
    if rig["helper"] is not None and helper_rotation is not None:
        helper_sel = om.MSelectionList()
        helper_sel.add(rig["helper"])
        om.MFnTransform(helper_sel.getDagPath(0)).setRotation(
            om.MQuaternion(*helper_rotation), om.MSpace.kTransform)
    mc.refresh(force=True)


def set_offsets(om, skin, values):
    selection = om.MSelectionList()
    selection.add(skin)
    plug = om.MFnDependencyNode(selection.getDependNode(0)).findPlug(
        "ocdCenterOffsets", False)
    plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray(values)))


def max_deviation(first, second):
    worst = 0.0
    for a, b in zip(first, second):
        d = math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)
        worst = max(worst, d)
    return worst


# ----------------------------------------------------------------------
# 模式
# ----------------------------------------------------------------------

def dump(path, three_bone):
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()
        weights = rig_weights(bind, three_bone)
        bone_count = len(weights[0])

        payload = {
            "bind": bind,
            "faces": [list(f) for f in faces],
            "vertices": len(bind),
            "boneCount": bone_count,
            "weights": [list(row) for row in weights],
            "cases": {},
        }
        for bend in BENDS:
            helper_rotation = (ref.joint_quaternion(bend * 0.5, 0, 0)
                               if bone_count >= 3 else None)
            rig = build_rig(mc, om, bind, faces, "dump", bone_count)
            _, result = attach_skin(mc, rig, 0, weights)
            pose_elbow(mc, om, rig, bend, helper_rotation)
            payload["cases"][str(bend)] = {
                "bend": bend,
                "original": read_points(om, result),
                "helperRotation": list(helper_rotation) if helper_rotation else None,
            }
            mc.delete(rig["holder"])

        with open(path, "w") as handle:
            json.dump(payload, handle)
        print("dumped %d bends, boneCount=%d -> %s"
              % (len(BENDS), bone_count, path))
        return 0
    finally:
        maya.standalone.uninitialize()


def inject(path):
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = 0
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()

        with open(path) as handle:
            truth = json.load(handle)
        # 权重与骨数完全取自 truth，避免夹具与求解各用一套。
        weights = [tuple(row) for row in truth["weights"]]
        bone_count = truth["boneCount"]
        if bone_count != len(weights[0]):
            print("FAIL: boneCount 与权重列数不一致")
            return 1

        for bend in BENDS:
            case = truth["cases"][str(bend)]
            delta = case["delta"]
            helper_rotation = case.get("helperRotation")

            inj = build_rig(mc, om, bind, faces, "injected", bone_count)
            inj_skin, inj_result = attach_skin(mc, inj, 5, weights)
            pose_elbow(mc, om, inj, bend, helper_rotation)
            set_offsets(om, inj_skin, [v for row in delta for v in row])
            injected = read_points(om, inj_result)

            auto = build_rig(mc, om, bind, faces, "auto", bone_count)
            _, auto_result = attach_skin(mc, auto, 5, weights)
            pose_elbow(mc, om, auto, bend, helper_rotation)
            solved = read_points(om, auto_result)

            # 顶点空间偏差；与 |delta| 不是同一个量，分开陈述。
            vertex_deviation = max_deviation(solved, injected)
            delta_length = max(math.sqrt(sum(v * v for v in row)) for row in delta)
            print("bend=%5.1f  boneCount=%d  注入 delta 与自动求解："
                  "顶点最大偏差 %.6f   (|delta|max=%.4f)"
                  % (bend, bone_count, vertex_deviation, delta_length))
            if vertex_deviation > TOLERANCE:
                print("  FAIL: 顶点输出不一致")
                failures += 1
            mc.delete(inj["holder"])
            mc.delete(auto["holder"])

        print("FAILURES:", failures)
        return 0 if failures == 0 else 1
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    if "--dump" in sys.argv:
        sys.exit(dump(sys.argv[sys.argv.index("--dump") + 1],
                      "--three-bone" in sys.argv))
    if "--inject" in sys.argv:
        sys.exit(inject(sys.argv[sys.argv.index("--inject") + 1]))
    print(__doc__)
    sys.exit(2)
