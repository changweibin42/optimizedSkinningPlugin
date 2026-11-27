"""最终判定：把 delta 置零（通过注入全零偏移），看凹陷是否消失。

ocdCenterOffsets 注入会绕过 offsetsLocal，于是 centers = base + 0 = base，
即退化为「纯 CoR 用 baseCenters」。
但我们要的是「CoR + guide 但不要 delta」，只能改代码。

先用两次注入做隔离：
  A) 注入全零  -> centers = base（纯 CoR 基中心）
  B) 不注入    -> centers = base + delta（当前行为）
比较两者的径向轮廓，看 delta 贡献了多少凹陷。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        source = r"D:\tmp\ocd_case3.json"
        d = json.load(open(source))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="z_holder")
        geo = mc.createNode("transform", name="z_geo", parent=holder)
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
                n = mc.createNode("joint", name="zj0", parent=holder)
                mc.setAttr(n + ".translateX", b["bindPosition"][0])
            else:
                n = mc.createNode("joint", name="zj%d" % i, parent=joints[-1])
                mc.setAttr(n + ".translateX",
                           b["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
            joints.append(n)

        skin = mc.deformer(shape, type="optimizedSkinCluster", name="z_skin#")[0]
        for i, (n, b) in enumerate(zip(joints, bones)):
            mc.connectAttr(n + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                       *[float(v) for v in b["bindPreMatrix"]], type="matrix")
        for v, row in enumerate(sparse):
            for k, val in row.items():
                mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k),
                           float(val))

        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(joints[1] + ".rotateZ", 146.126)
        mc.refresh(force=True)
        mc.dgeval(shape)

        def read():
            s = om.MSelectionList()
            s.add(shape)
            return [(p.x, p.y, p.z) for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        # A: 不注入（当前行为）
        a = read()

        # B: 注入全零
        node = om.MSelectionList()
        node.add(skin)
        plug = om.MFnDependencyNode(node.getDependNode(0)).findPlug(
            "ocdCenterOffsets", False)
        plug.setMObject(om.MFnDoubleArrayData().create(
            om.MDoubleArray([0.0] * (3 * len(bind)))))
        mc.refresh(force=True)
        mc.dgeval(shape)
        b_pts = read()

        # 径向
        rad = [math.hypot(p[1], p[2]) for p in bind]
        Rmax = max(rad)
        wall = [i for i, r in enumerate(rad) if r > 0.9 * Rmax]

        def prof(pts):
            return [math.hypot(pts[i][1], pts[i][2]) for i in wall]

        pa, pb = prof(a), prof(b_pts)
        print("== 注入全零 vs 不注入 ==")
        print("  不注入 (base+delta) 径向 min=%.4f max=%.4f" % (min(pa), max(pa)))
        print("  注入零 (只 base)    径向 min=%.4f max=%.4f" % (min(pb), max(pb)))
        print("  差异 max=%.4f" % max(abs(x - y) for x, y in zip(pa, pb)))

        # 更直接：算法0/3/5 三者对比
        print("\n== 三种算法 ==")
        plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray([])))
        for algo, label in ((0, "CoR"), (3, "OCD"), (5, "OCD+CoR")):
            mc.setAttr(skin + ".algorithmType", algo)
            mc.refresh(force=True)
            mc.dgeval(shape)
            p = prof(read())
            print("  %-8s 径向 min=%.4f max=%.4f mean=%.4f"
                  % (label, min(p), max(p), sum(p) / len(p)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
