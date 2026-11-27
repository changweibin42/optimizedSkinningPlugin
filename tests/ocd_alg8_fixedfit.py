# -*- coding: utf-8 -*-
"""验证 alg 8（CoR Fixed Center Fit）语义。

三项：
  1) corFitOffsets 为空  → 必须与 alg 0 逐位相同
  2) 注入全零           → 必须与 alg 0 逐位相同
  3) 注入已知偏移       → 必须等于 alg 0 的基础上做 A·offset 的位移
                          （A = Σw·R − R_blend，即中心线性响应）

    mayapy tests/ocd_alg8_fixedfit.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


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
        mc.setAttr(js[1] + ".rotateZ", -147.2124)

        print("corFitOffsets 属性存在: %s"
              % mc.attributeQuery("corFitOffsets", node=sk, exists=True))

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s2 = om.MSelectionList()
            s2.add(shp)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s2.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def inject(vals):
            s3 = om.MSelectionList()
            s3.add(sk)
            om.MFnDependencyNode(s3.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        def diff(a, b):
            return max(math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for k in range(3)))
                       for i in range(n))

        # 1) alg 0
        mc.setAttr(sk + ".algorithmType", 0)
        inject([])
        base = read()

        # 2) alg 8，偏移为空
        mc.setAttr(sk + ".algorithmType", 8)
        inject([])
        a8_empty = read()
        print("\n[1] alg8 空偏移 vs alg0 : %.3e  %s"
              % (diff(a8_empty, base),
                 "通过" if diff(a8_empty, base) < 1e-9 else "★失败★"))

        # 3) alg 8，全零
        inject([0.0] * (3 * n))
        a8_zero = read()
        print("[2] alg8 全零偏移 vs alg0: %.3e  %s"
              % (diff(a8_zero, base),
                 "通过" if diff(a8_zero, base) < 1e-9 else "★失败★"))

        # 4) 注入已知偏移，检验是否 = alg0 + A·offset
        # 选 20 个混合区顶点，沿 +Y 加 0.1
        mixed = [i for i in range(n)
                 if 0.2 < float(sparse[i].get("1", 0.0)) < 0.8][:20]
        vals = [0.0] * (3 * n)
        for i in mixed:
            vals[3 * i + 1] = 0.1
        inject(vals)
        a8_off = read()

        # 期望：A·offset，A 只用两骨
        child = [b for b in bones if b["short"] == "L_Ankle_jnt"][0]
        wm = [float(v) for v in child["worldMatrix"]]
        bend = math.degrees(math.atan2(wm[4], wm[0]))
        cb, sb = math.cos(math.radians(bend)), math.sin(math.radians(bend))
        Rc = [[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]]

        worst = 0.0
        for i in mixed:
            w1 = float(sparse[i].get("1", 0.0))
            w0 = 1.0 - w1
            # blend 用同半球累加后的归一化；纯 Z 旋转时等价于
            # 角度的加权和：atan2(w1*sb, w0 + w1*cb)
            ang = math.atan2(w1 * sb, w0 + w1 * cb)
            cbq, sbq = math.cos(ang), math.sin(ang)
            Rb = [[cbq, -sbq, 0.0], [sbq, cbq, 0.0], [0.0, 0.0, 1.0]]
            A = [[w1 * Rc[a][b] - Rb[a][b] for b in range(3)] for a in range(3)]
            off = [0.0, 0.1, 0.0]
            pred = [sum(A[a][b] * off[b] for b in range(3)) for a in range(3)]
            got = [a8_off[i][k] - base[i][k] for k in range(3)]
            err = math.sqrt(sum((pred[k] - got[k]) ** 2 for k in range(3)))
            worst = max(worst, err)
        print("[3] alg8 偏移效果 vs A·offset 预测: max 误差 %.3e  %s"
              % (worst, "通过" if worst < 2e-3 else "★偏差大★"))

        # 打印几个样本
        print("\n样本（alg0 → alg8 位移 vs 预测）：")
        for i in mixed[:5]:
            w1 = float(sparse[i].get("1", 0.0))
            got = [a8_off[i][k] - base[i][k] for k in range(3)]
            ang = math.atan2(w1 * sb, (1 - w1) + w1 * cb)
            cbq, sbq = math.cos(ang), math.sin(ang)
            Rb = [[cbq, -sbq, 0.0], [sbq, cbq, 0.0], [0.0, 0.0, 1.0]]
            A = [[w1 * Rc[a][b] - Rb[a][b] for b in range(3)] for a in range(3)]
            pred = [sum(A[a][b] * [0.0, 0.1, 0.0][b] for b in range(3))
                    for a in range(3)]
            print("  v%-6d w1=%.3f  got=(%+.5f,%+.5f,%+.5f)  pred=(%+.5f,%+.5f,%+.5f)"
                  % (i, w1, got[0], got[1], got[2], pred[0], pred[1], pred[2]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
