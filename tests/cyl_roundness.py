# -*- coding: utf-8 -*-
"""在 HALF=6 圆柱（458 点，24 边）上测逐环圆度：CoR vs alg5。

如果 458 点的圆柱圆度良好而 16080 点网格圆度差 0.13，
说明问题与网格密度/环上点数有关，而不是算法本身。

    mayapy tests/cyl_roundness.py
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
        import ocd_elbow_reference_latest as ref

        HALF = ref.HALF
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1
        bend = 146.1265
        bind, faces = ref.cylinder()
        n = len(bind)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="prev", parent=holder)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")

        weights = []
        for i, p in enumerate(bind):
            w = ref.skin_weight(p[0])
            rw = ref.rig_weights(w, False)
            weights.append([float(v) for v in rw])
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       float(rw[0]), float(rw[1]), size=2)
        mc.setAttr(elbow + ".rotateZ", bend)

        def read(algo):
            mc.setAttr(skin + ".algorithmType", algo)
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shape)
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        cor = read(0)
        ocd = read(5)

        # 逐环：cylinder() 是 RINGS x SIDES + 2 极点
        def roundness(P, row):
            first = row * SIDES
            Q = [P[first + j] for j in range(SIDES)]
            cx = sum(q[0] for q in Q) / SIDES
            cy = sum(q[1] for q in Q) / SIDES
            cz = sum(q[2] for q in Q) / SIDES
            c = (cx, cy, cz)
            # 平面基：用环内两两方向做 PCA 太麻烦，直接用法向 = 环质心到下一环质心的方向
            pts = [(q[0] - c[0], q[1] - c[1], q[2] - c[2]) for q in Q]
            # 用 SVD 的简化版：协方差矩阵最小特征向量
            import math as _m
            cov = [[sum(p[i] * p[j] for p in pts) for j in range(3)] for i in range(3)]
            # 幂迭代找最小特征向量：用伴随矩阵的零空间近似
            # 简化：法向取 cross(p0-c, p_{SIDES//4}-c)
            a = pts[0]
            b = pts[SIDES // 4]
            nrm = (a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2],
                   a[0] * b[1] - a[1] * b[0])
            ln = _m.sqrt(sum(v * v for v in nrm))
            if ln < 1e-12:
                return None
            nrm = tuple(v / ln for v in nrm)
            r = []
            for p in pts:
                d = sum(p[i] * nrm[i] for i in range(3))
                proj = tuple(p[i] - nrm[i] * d for i in range(3))
                r.append(_m.sqrt(sum(v * v for v in proj)))
            mr = sum(r) / len(r)
            if mr < 1e-12:
                return None
            var = sum((v - mr) ** 2 for v in r) / len(r)
            return _m.sqrt(var) / mr, mr

        print("HALF=%.1f  SIDES=%d  RINGS=%d  bend=%.2f" % (HALF, SIDES, RINGS, bend))
        print("\n%8s %12s %12s %10s" % ("x", "CoR圆度", "alg5圆度", "差值"))
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            rc = roundness(cor, row)
            ro = roundness(ocd, row)
            if rc is None or ro is None:
                continue
            print("%8.2f %12.6f %12.6f %+10.6f"
                  % (x0, rc[0], ro[0], ro[0] - rc[0]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
