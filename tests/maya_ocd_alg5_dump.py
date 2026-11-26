# -*- coding: utf-8 -*-
"""逐级对拍的一端：在 Maya 里导出 originalCoR，或注入真值后比较。

分两步使用，中间由 `tests/ocd_alg5_truth.py`（外部 numpy 环境）算真值：

    # 1) 导出每个弯曲角的零偏移 CoR 输出（= 算法 5 的 original）
    mayapy tests/maya_ocd_alg5_dump.py --dump tests/_alg5_original.json

    # 2) 用 numpy 环境算 guide 与 centerOffset
    python tests/ocd_alg5_truth.py tests/_alg5_original.json tests/_alg5_truth.json

    # 3) 注入真值，与插件自动求解的结果比较
    mayapy tests/maya_ocd_alg5_dump.py --inject tests/_alg5_truth.json

第 2 步能跑通就说明 Python 参考实现的 guide 与 delta 已经独立算出；第 3 步
若「注入 == 自动求解」，则插件的数据流与参考一致。
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

BENDS = (118.0, 150.0)


def build_rig(mc, om, bind, faces, label):
    holder = mc.createNode("transform", name="dump_%s_holder#" % label)
    geo = mc.createNode("transform", name="dump_%s_geo#" % label, parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind], [len(f) for f in faces],
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
    return {"holder": holder, "geo": geo, "shape": shape,
            "prev": prev, "elbow": elbow}


def attach_skin(mc, rig, algorithm, bind):
    skin = mc.deformer(rig["shape"], type="optimizedSkinCluster",
                       name="dump_skin#")[0]
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


def read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [[p.x, p.y, p.z] for p in mesh.getPoints(om.MSpace.kObject)]


def pose_elbow(mc, om, rig, bend):
    selection = om.MSelectionList()
    selection.add(rig["elbow"])
    om.MFnTransform(selection.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(bend, 0, 0)), om.MSpace.kTransform)
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


def run(mode, path):
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()

        if mode == "dump":
            payload = {"bind": bind,
                       "faces": [list(f) for f in faces],
                       "vertices": len(bind),
                       "cases": {}}
            for bend in BENDS:
                rig = build_rig(mc, om, bind, faces, "dump")
                _, result = attach_skin(mc, rig, 0, bind)
                pose_elbow(mc, om, rig, bend)
                payload["cases"][str(bend)] = {
                    "bend": bend,
                    "original": read_points(om, result),
                }
                mc.delete(rig["holder"])
            with open(path, "w") as handle:
                json.dump(payload, handle)
            print("dumped original for %d bends -> %s" % (len(BENDS), path))
            return 0

        # inject 模式
        with open(path) as handle:
            truth = json.load(handle)
        failures = 0
        for bend in BENDS:
            case = truth["cases"][str(bend)]
            delta = case["delta"]

            inj = build_rig(mc, om, bind, faces, "injected")
            inj_skin, inj_result = attach_skin(mc, inj, 5, bind)
            pose_elbow(mc, om, inj, bend)
            set_offsets(om, inj_skin, [v for row in delta for v in row])
            injected = read_points(om, inj_result)

            auto = build_rig(mc, om, bind, faces, "auto")
            _, auto_result = attach_skin(mc, auto, 5, bind)
            pose_elbow(mc, om, auto, bend)
            solved = read_points(om, auto_result)

            deviation = max_deviation(solved, injected)
            worst_delta = max(math.sqrt(sum(v * v for v in row)) for row in delta)
            print("bend=%5.1f  |delta|max=%8.4f  注入 vs 自动求解 最大偏差 %.6f"
                  % (bend, worst_delta, deviation))
            if deviation > 2.0e-3:
                print("  FAIL: 插件自动求解与 Python 真值不一致")
                failures += 1
            mc.delete(inj["holder"])
            mc.delete(auto["holder"])

        print("FAILURES:", failures)
        return 0 if failures == 0 else 1
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    if "--dump" in sys.argv:
        sys.exit(run("dump", sys.argv[sys.argv.index("--dump") + 1]))
    if "--inject" in sys.argv:
        sys.exit(run("inject", sys.argv[sys.argv.index("--inject") + 1]))
    print(__doc__)
    sys.exit(2)
