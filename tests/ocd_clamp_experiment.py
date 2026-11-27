# -*- coding: utf-8 -*-
r"""限幅实验：同一个节点，只改 ocdCenterLimit，用固定顶点 ID 判定。

判据（严格按预期）：
  · 上限 16 时未触限的顶点，提高上限后最终位置必须在数值容差内不变。
  · 370 个触限顶点是唯一允许改变位置的集合。
  · 红圈若不受影响，限幅就不可能是那道褶皱的成因。

只做两遍：16 和 100。

    mayapy tests/ocd_clamp_experiment.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

OUT = os.path.join(HERE, "_clamp_exp.json")


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        src = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(src))
        bind = [tuple(p) for p in d["mesh"]["bindPoints"]]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="clmp_h")
        geo = mc.createNode("transform", name="clmp_g", parent=holder)
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
                nd = mc.createNode("joint", name="clmp_j0", parent=holder)
                mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
            else:
                nd = mc.createNode("joint", name="clmp_j%d" % i, parent=joints[-1])
                mc.setAttr(nd + ".translateX",
                           bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
            joints.append(nd)

        skin = mc.deformer(shape, type="optimizedSkinCluster", name="clmp_s#")[0]
        for i, (nd, bn) in enumerate(zip(joints, bones)):
            mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                       *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
        for v, row in enumerate(sparse):
            for k, val in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k), float(val))

        mc.setAttr(skin + ".algorithmType", 5)
        # 显式关门控，消除夹具缺字段的歧义
        for attr in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(attr, node=skin, exists=True):
                mc.setAttr("%s.%s" % (skin, attr), 0.0)
        # 确认注入为空
        plug = om.MFnDependencyNode(
            om.MSelectionList().add(skin).getDependNode(0)).findPlug(
                "ocdCenterOffsets", False)
        plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray()))

        bend = d["attributes"].get("algorithmType", 5)
        mc.setAttr(joints[1] + ".rotateZ", -146.1265)

        print("plugin:", mc.pluginInfo("optimizedSkinningPlugin", q=True, path=True))
        print("gate:", [mc.getAttr("%s.%s" % (skin, a)) if mc.attributeQuery(
            a, node=skin, exists=True) else None
            for a in ("ocdCenterGateLow", "ocdCenterGateHigh")])

        def evaluate(limit):
            mc.setAttr(skin + ".ocdCenterLimit", float(limit))
            for _ in range(3):
                mc.refresh(force=True)
                mc.dgeval(shape)
            s = om.MSelectionList()
            s.add(shape)
            pts = [[p.x, p.y, p.z] for p in
                   om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]
            delta = None
            if mc.attributeQuery("ocdDeltaDump", node=skin, exists=True):
                v = mc.getAttr(skin + ".ocdDeltaDump")
                if v:
                    delta = list(v)
            lim = None
            if mc.attributeQuery("ocdCenterLimitDump", node=skin, exists=True):
                v = mc.getAttr(skin + ".ocdCenterLimitDump")
                if v:
                    lim = list(v)
            # 触限判定：用 delta 的模与 dump 的限额比较
            clamped = []
            if delta and lim:
                L = lim[0]
                for i in range(n):
                    m = math.sqrt(delta[3 * i] ** 2 + delta[3 * i + 1] ** 2
                                  + delta[3 * i + 2] ** 2)
                    if m >= L * (1.0 - 1e-6):
                        clamped.append(i)
            return pts, delta, lim, clamped

        print("\n=== 上限 16 ===")
        p16, d16, l16, c16 = evaluate(16.0)
        print("  centerLimitDump = %s" % l16)
        print("  delta 长度 = %s" % (len(d16) if d16 else None))
        print("  触限顶点数 = %d" % len(c16))
        if c16:
            print("  触限顶点 x 范围: %.2f .. %.2f"
                  % (min(bind[i][0] for i in c16), max(bind[i][0] for i in c16)))

        print("\n=== 上限 100 ===")
        p100, d100, l100, c100 = evaluate(100.0)
        print("  centerLimitDump = %s" % l100)
        print("  触限顶点数 = %d" % len(c100))

        # ---- 逐顶点位置差 ----
        diffs = [math.sqrt(sum((p16[i][k] - p100[i][k]) ** 2 for k in range(3)))
                 for i in range(n)]
        moved = [i for i in range(n) if diffs[i] > 1e-7]
        print("\n=== 位置变化 ===")
        print("  16 vs 100: 有变化的顶点 %d / %d" % (len(moved), n))
        print("  最大位置差 = %.6e" % max(diffs))
        if moved:
            xs = sorted(set(round(bind[i][0], 2) for i in moved))
            print("  变化顶点的 x 值: %s" % xs[:25])
            print("  x 范围 %.2f .. %.2f" % (min(bind[i][0] for i in moved),
                                             max(bind[i][0] for i in moved)))

        # ---- 关键判据 1：未触限顶点必须不动 ----
        notclamped = [i for i in range(n) if i not in set(c16)]
        mx_nc = max((diffs[i] for i in notclamped), default=0.0)
        print("\n=== 判据 1: 上限 16 时未触限的顶点 ===")
        print("  数量 %d" % len(notclamped))
        print("  放宽到 100 后最大位置差 = %.6e  %s"
              % (mx_nc, "通过（不变）" if mx_nc < 1e-6 else "★违反预期★"))

        # ---- 关键判据 2：触限顶点变化了多少 ----
        if c16:
            mc_moved = [diffs[i] for i in c16]
            print("\n=== 判据 2: 370 个触限顶点 ===")
            print("  位置差 均=%.6e max=%.6e" % (sum(mc_moved) / len(mc_moved),
                                                max(mc_moved)))

        # 保存
        json.dump({"p16": p16, "p100": p100,
                   "delta16": d16, "delta100": d100,
                   "limit16": l16, "limit100": l100,
                   "clamped16": c16, "clamped100": c100,
                   "moved": moved, "diffs": diffs},
                  open(OUT, "w"))
        print("\nwrote", OUT)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
