# -*- coding: utf-8 -*-
r"""长径比实验：保持半径、权重形态、姿态不变，只压缩纵向长度。

判定「delta 量级失控」是否由长径比导致。
参考实现的测试模型归一后是「长度 2、半径 1」，本夹具是「长度 24.4、半径 1」。

做法：把 bind 的 x 乘以 scale（station 同比例缩小），半径不动，
权重按原 station 的值查表（因为权重只依赖 station）。
然后读新的 |delta| 与凹陷深度。

    mayapy tests/ocd_aspect_experiment.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def build_and_measure(mc, om, bind, faces, bones, sparse, bend, aspect_scale):
    """按给定纵向缩放重建，返回 (|delta| 统计, 最终点)。"""
    holder = mc.createNode("transform", name="ax_holder")
    geo = mc.createNode("transform", name="ax_geo", parent=holder)
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
            node = mc.createNode("joint", name="ax_j%d" % index, parent=holder)
            mc.setAttr(node + ".translateX", bone["bindPosition"][0])
        else:
            node = mc.createNode("joint", name="ax_j%d" % index,
                                 parent=joints[-1])
            local = (bone["bindPosition"][0]
                     - bones[index - 1]["bindPosition"][0])
            mc.setAttr(node + ".translateX", local)
        joints.append(node)

    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name="ax_skin#")[0]
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
    mc.setAttr(skin + ".corDebugLog", False)
    mc.setAttr(joints[1] + ".rotateZ", bend)
    mc.refresh(force=True)
    mc.dgeval(shape)

    # 诊断属性只在 deformOCD 里写；强制拉一次输出实体确保它跑过。
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
        cur = om.MSelectionList()
        cur.add(shape)
        om.MFnMesh(cur.getDagPath(0)).getPoints(om.MSpace.kObject)
    try:
        mc.getAttr(skin + ".ocdDeltaDump")
    except Exception:
        pass

    delta = list(mc.getAttr(skin + ".ocdDeltaDump") or [])
    guide = list(mc.getAttr(skin + ".ocdGuideDump") or [])
    limit = list(mc.getAttr(skin + ".ocdCenterLimitDump") or [])

    sel2 = om.MSelectionList()
    sel2.add(shape)
    final = [(p.x, p.y, p.z) for p in
             om.MFnMesh(sel2.getDagPath(0)).getPoints(om.MSpace.kObject)]

    dm = [math.sqrt(delta[3 * i] ** 2 + delta[3 * i + 1] ** 2
                    + delta[3 * i + 2] ** 2) for i in range(len(bind))]
    gm = [math.sqrt(guide[3 * i] ** 2 + guide[3 * i + 1] ** 2
                    + guide[3 * i + 2] ** 2) for i in range(len(bind))]

    mc.delete(holder)
    return {
        "delta_max": max(dm), "delta_mean": sum(dm) / len(dm),
        "guide_max": max(gm), "guide_mean": sum(gm) / len(gm),
        "limit": limit[0] if limit else 0.0,
        "final": final,
    }


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(source))
        bind0 = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        # 半径与当前半径
        rad = [math.hypot(p[1], p[2]) for p in bind0]
        R0 = max(rad)
        half_len0 = max(abs(p[0]) for p in bind0)
        print("原始: 半长=%.3f  半径=%.4f  长径比=%.2f"
              % (half_len0, R0, 2 * half_len0 / R0))

        bend = 146.126
        results = []
        for scale in (1.0, 0.5, 0.25, 1.0 / 6.0, 1.0 / 12.19):
            bind = [[p[0] * scale, p[1], p[2]] for p in bind0]
            half_len = half_len0 * scale
            ratio = 2 * half_len / R0
            stats = build_and_measure(mc, om, bind, faces, bones, sparse,
                                      bend, scale)
            # 凹陷：径向最小值
            fin = stats["final"]
            Rf = [math.hypot(p[1], p[2]) for p in fin]
            Rb = [math.hypot(b[1], b[2]) for b in bind]
            drop = min(Rf) - max(Rb)
            results.append((ratio, stats, drop))
            print("scale=%.4f  长径比=%6.2f  |delta|max=%8.4f mean=%7.4f  "
                  "|guide|max=%6.4f  限幅=%7.3f  半径最小=%.4f"
                  % (scale, ratio, stats["delta_max"], stats["delta_mean"],
                     stats["guide_max"], stats["limit"], min(Rf)))

        print("\n== 判定 ==")
        print("  长径比  |delta|max  |delta|mean  |guide|max  半径最小")
        for ratio, stats, _ in results:
            print("  %7.2f  %9.4f  %10.4f  %10.4f" % (
                ratio, stats["delta_max"], stats["delta_mean"],
                stats["guide_max"]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
