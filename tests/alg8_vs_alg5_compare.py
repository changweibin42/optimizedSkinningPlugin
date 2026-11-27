# -*- coding: utf-8 -*-
r"""alg 8 (corFitOffsets) 与 alg 5 (内部求解) 的对拍。

步骤：
  1) 用 1285f9a 的参考实现，在真实夹具上从零偏移的 CoR 出发求 delta
  2) 把 delta 按"绑定对象空间向量"写进 corFitOffsets，用 alg 8 求值 → A
  3) 同一姿态用 alg 5 内部求解求值 → B
  4) 对比 A/B 与 alg 0（纯 CoR），看谁更贴近 CoR 的饱满外轮廓

注意：delta 的空间约定。参考实现的 structural_centers 输出在"局部铰链系"
（半径归一）。插件 alg 8 的 corFitOffsets 按绑定对象空间向量解释：
    offset_object = R * (dx*axis + dy*basisY + dz*basisZ)
本夹具 worldMat 为单位阵，故对象空间 == 世界空间。

    mayapy tests/alg8_vs_alg5_compare.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build(mc, om, bind, faces, bones, sparse, name="t"):
    h = mc.createNode("transform", name=name + "_h")
    g = mc.createNode("transform", name=name + "_g", parent=h)
    s = om.MSelectionList()
    s.add(g)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=s.getDependNode(0))
    shp = mc.listRelatives(g, shapes=True, noIntermediate=True, fullPath=True)[0]
    js = []
    for i, bn in enumerate(bones):
        if i == 0:
            nd = mc.createNode("joint", name=name + "_j0", parent=h)
            mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
        else:
            nd = mc.createNode("joint", name="%s_j%d" % (name, i), parent=js[-1])
            mc.setAttr(nd + ".translateX",
                       bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
        js.append(nd)
    sk = mc.deformer(shp, type="optimizedSkinCluster", name=name + "_s#")[0]
    for i, (nd, bn) in enumerate(zip(js, bones)):
        mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (sk, i))
        mc.setAttr("%s.bindPreMatrix[%d]" % (sk, i),
                   *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
    for v, row in enumerate(sparse):
        for k, val in row.items():
            mc.setAttr("%s.weightList[%d].weights[%s]" % (sk, v, k), float(val))
    for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
        if mc.attributeQuery(a, node=sk, exists=True):
            mc.setAttr("%s.%s" % (sk, a), 0.0)
    return h, shp, sk, js


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        case = json.load(open(r"D:\tmp\ocd_case3.json"))
        bind = [tuple(p) for p in case["mesh"]["bindPoints"]]
        faces = case["mesh"]["faces"]
        bones = case["bones"]
        sparse = case["weightsSparse"]
        n = len(bind)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        h, shp, sk, js = build(mc, om, bind, faces, bones, sparse)
        mc.setAttr(js[1] + ".rotateZ", -147.2124)

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s = om.MSelectionList()
            s.add(shp)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def inject(vals):
            s = om.MSelectionList()
            s.add(sk)
            om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        # --- 1) 纯 CoR ---
        mc.setAttr(sk + ".algorithmType", 0)
        inject([])
        cor = read()

        # --- 2) alg 5 内部求解 ---
        mc.setAttr(sk + ".algorithmType", 5)
        mc.setAttr(sk + ".ocdEnableCenterLimit", False)
        a5 = read()

        # --- 3) 参考实现求 delta（外部注入用）---
        # 若上面 step1/step2 已导出过，直接复用；否则现在算。
        import ocd_elbow_reference_latest as ref
        import numpy as np

        child = [b for b in bones if b["short"] == "L_Ankle_jnt"][0]
        pivot = np.asarray(child["bindPosition"], float)
        axis = np.asarray(case["hinge"]["axis"], float)
        axis = axis / np.linalg.norm(axis)
        bY = np.array([0.0, 1.0, 0.0])
        bZ = np.cross(axis, bY)
        bZ /= np.linalg.norm(bZ)
        rel = np.asarray(bind, float) - pivot
        st = rel @ axis
        rad = rel - np.outer(st, axis)
        R = float(np.linalg.norm(rad, axis=1).max())
        bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

        def to_local(P):
            o = np.asarray(P, float) - pivot
            return np.column_stack([(o @ axis) / R, (o @ bY) / R, (o @ bZ) / R])

        cor_l = to_local(cor)
        W = np.zeros((n, len(bones)))
        for i, row in enumerate(sparse):
            for k, v in row.items():
                W[i, int(k)] = float(v)
        used = [j for j in range(len(bones)) if W[:, j].max() > 0.0]
        W = W[:, used]

        wm = [float(v) for v in child["worldMatrix"]]
        bend = math.degrees(math.atan2(wm[4], wm[0]))
        print("bend=%.4f  R=%.6f  骨骼列=%s" % (bend, R, used))

        q = ref.joint_quaternion(bend, 0.0, 0.0)
        rots = [(0.0, 0.0, 0.0, 1.0), q]
        guide = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))
        edges = ref.local_edges(bind_l.tolist(), faces)
        py_delta = np.asarray(ref.structural_centers(
            edges, cor_l, W, rots, guide, outer, None))   # 不限幅

        print("Python delta: |delta|均=%.5f max=%.5f"
              % (np.linalg.norm(py_delta, axis=1).mean(),
                 np.linalg.norm(py_delta, axis=1).max()))

        # 局部归一 → 对象空间向量（本夹具 worldMat 为单位阵）
        obj = (py_delta[:, 0:1] * axis[None, :]
               + py_delta[:, 1:2] * bY[None, :]
               + py_delta[:, 2:3] * bZ[None, :]) * R

        # --- 4) alg 8 + Python delta ---
        mc.setAttr(sk + ".algorithmType", 8)
        inject(list(obj.ravel()))
        a8 = read()

        # --- 对比 ---
        def diff(A, B):
            return max(math.sqrt(sum((A[i][k] - B[i][k]) ** 2 for k in range(3)))
                       for i in range(n))

        print("\n=== 三套结果的两两最大顶点差 ===")
        print("  alg5  vs alg0(纯CoR)   : %.5f" % diff(a5, cor))
        print("  alg8  vs alg0(纯CoR)   : %.5f" % diff(a8, cor))
        print("  alg8  vs alg5          : %.5f" % diff(a8, a5))

        # 外轮廓对比（逐环 min y）
        STEP = 0.12
        rings = [round(p[0] / STEP) for p in bind]
        print("\n=== 外轮廓（逐环 min y；越接近 CoR 越饱满）===")
        print("%8s %10s %10s %10s %10s %10s"
              % ("x", "CoR", "alg5", "alg8+Py", "a5-CoV", "a8-CoV"))
        s5 = s8 = 0.0
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.7 <= x0 <= 0.7):
                continue
            m2 = [i for i in range(n) if rings[i] == rid]
            yc = min(cor[i][1] for i in m2)
            y5 = min(a5[i][1] for i in m2)
            y8 = min(a8[i][1] for i in m2)
            s5 += (y5 - yc) ** 2
            s8 += (y8 - yc) ** 2
            print("%8.3f %10.5f %10.5f %10.5f %+10.5f %+10.5f"
                  % (x0, yc, y5, y8, y5 - yc, y8 - yc))
        print("\n外轮廓相对 CoR 的偏差 RMS: alg5=%.5f  alg8=%.5f"
              % (math.sqrt(s5 / max(1, len(rings))),
                 math.sqrt(s8 / max(1, len(rings)))))

        # 同一环圆度
        def ring_round(P, rid):
            m2 = [i for i in range(n) if rings[i] == rid]
            c = [sum(P[i][k] for i in m2) / len(m2) for k in range(3)]
            r = [math.sqrt(sum((P[i][k] - c[k]) ** 2 for k in range(3))) for i in m2]
            mr = sum(r) / len(r)
            v = sum((x - mr) ** 2 for x in r) / len(r)
            return math.sqrt(v) / max(1e-12, mr)

        print("\n=== 圆度（逐环）===")
        print("%8s %10s %10s %10s" % ("x", "CoR", "alg5", "alg8+Py"))
        for rid in sorted(set(rings)):
            x0 = rid * STEP
            if not (-2.4 <= x0 <= 2.4):
                continue
            print("%8.3f %10.6f %10.6f %10.6f"
                  % (x0, ring_round(cor, rid), ring_round(a5, rid),
                     ring_round(a8, rid)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
