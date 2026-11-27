# -*- coding: utf-8 -*-
r"""验证 C++ 的 A_i 是否与 Python 一致。

A_i = Σ_k w_ik R_k − R_blend（在铰链局部系里）。
判据：对同一个 guide，C++ 解出的 delta 与 Python 解出的 delta 之比
     应当接近 1；若相差很多，说明 A_i 不同。

绕过 A_i 的直接办法：给单个顶点注入已知的 delta，
看 C++ 的最终位移是否符合 A_i·delta。但 A_i 未知，故反向：
比较「C++ delta」与「Python delta」，并用 C++ 的 delta 在 Python 里算 A·delta，
看能否解释 C++ 的实际顶点位移。

    mayapy tests/verify_A_matrix.py
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
        import numpy as np
        import ocd_elbow_reference_latest as ref

        HALF = ref.HALF
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1
        bend = 135.0

        bind_list, faces = ref.cylinder()
        bind = np.asarray(bind_list, float)
        n = len(bind)
        eff = np.array([ref.preview_influences(p[0], True) for p in bind])
        nb = eff.shape[1]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        s = om.MSelectionList()
        s.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind_list],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=s.getDependNode(0))
        shp = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                               fullPath=True)[0]
        prev = mc.createNode("joint", name="prev", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(elbow + ".translateX", 2 * HALF)
        tip = mc.createNode("joint", name="tip", parent=elbow)
        mc.setAttr(tip + ".translateX", HALF)
        helper = mc.createNode("joint", name="helper", parent=prev)
        mc.setAttr(helper + ".translateX", HALF)
        bones = [prev, elbow, helper]
        skin = mc.deformer(shp, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate(bones):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
        for i in range(n):
            for k in range(nb):
                mc.setAttr("%s.weightList[%d].weights[%d]" % (skin, i, k),
                           float(eff[i, k]))
        q = ref.joint_quaternion(bend, 0.0, 0.0)
        s1 = om.MSelectionList()
        s1.add(elbow)
        om.MFnTransform(s1.getDagPath(0)).setRotation(om.MQuaternion(*q),
                                                      om.MSpace.kTransform)
        hq = ref.joint_quaternion(bend * 0.5, 0.0, 0.0)
        s2 = om.MSelectionList()
        s2.add(helper)
        om.MFnTransform(s2.getDagPath(0)).setRotation(om.MQuaternion(*hq),
                                                      om.MSpace.kTransform)

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shp)
            s3 = om.MSelectionList()
            s3.add(shp)
            return np.asarray([[p.x, p.y, p.z] for p in
                               om.MFnMesh(s3.getDagPath(0)).getPoints(om.MSpace.kObject)])

        def inject(vals):
            s4 = om.MSelectionList()
            s4.add(skin)
            om.MFnDependencyNode(s4.getDependNode(0)).findPlug(
                "corFitOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(vals)))
            mc.refresh(force=True)

        # CoR 基准与 alg5
        mc.setAttr(skin + ".algorithmType", 0)
        inject([])
        cor = read()

        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".ocdElbowGuide", True)
        mc.setAttr(skin + ".ocdEnableCenterLimit", True)
        mc.setAttr(skin + ".ocdCenterLimit", 16.0)
        a5 = read()
        dv = mc.getAttr(skin + ".ocdDeltaDump")
        cpp_delta = np.asarray(dv, float).reshape(n, 3)

        # 用 alg8 注入 C++ 自己的 delta，看是否复现 alg5
        mc.setAttr(skin + ".algorithmType", 8)
        inject(list(cpp_delta.ravel()))
        a8_cpp = read()
        print("alg8(C++delta) vs alg5: max=%.6f"
              % np.abs(a8_cpp - a5).max())

        # Python 侧 A·delta，delta 用 C++ 的
        bind_l = bind.copy()  # R=1，关节在原点等价
        # toLocalPoint：相对 child pivot
        cp = np.array([HALF, 0.0, 0.0])
        def to_local(P):
            o = P - cp
            return np.column_stack([o[:, 0], o[:, 1], o[:, 2]])
        cor_l = to_local(cor)
        rots = [ref.joint_quaternion(0, 0, 0), q,
                ref.joint_quaternion(bend * 0.5, 0.0, 0.0)]
        A = ref.center_response(eff, rots)
        pred = np.einsum("nij,nj->ni", A, cpp_delta)

        # C++ 的实际位移（局部系），与 pred 比较
        a5_l = to_local(a5)
        actual = a5_l - cor_l
        err = np.linalg.norm(pred - actual, axis=1)
        print("\n用 C++ 的 delta 在 Python 里算 A·delta，与 C++ 实际位移比较：")
        print("  mean|Δ|=%.5f  max=%.5f" % (err.mean(), err.max()))
        print("  |actual|均=%.5f  |A·delta|均=%.5f"
              % (np.linalg.norm(actual, axis=1).mean(),
                 np.linalg.norm(pred, axis=1).mean()))

        w1 = eff[:, 1]
        print("\n按 w1 分段：")
        print("  %14s %6s %12s %12s %12s"
              % ("w1", "n", "|actual|", "|A·delta|", "|Δ|"))
        for lo, hi in ((0, 1e-9), (1e-9, 0.1), (0.1, 0.4), (0.4, 0.6),
                       (0.6, 0.9), (0.9, 1.0)):
            m = (w1 >= lo) & (w1 < hi) if hi < 1.0 else (w1 >= lo) & (w1 <= hi)
            if m.sum() < 4:
                continue
            print("  %14s %6d %12.5f %12.5f %12.5f"
                  % ("[%.2f,%.2f)" % (lo, hi), int(m.sum()),
                     np.linalg.norm(actual[m], axis=1).mean(),
                     np.linalg.norm(pred[m], axis=1).mean(),
                     err[m].mean()))

        # delta 的方向分布
        print("\n|delta| 按 w1 分段：")
        for lo, hi in ((0, 1e-9), (1e-9, 0.1), (0.1, 0.4), (0.4, 0.6),
                       (0.6, 0.9), (0.9, 1.0)):
            m = (w1 >= lo) & (w1 < hi) if hi < 1.0 else (w1 >= lo) & (w1 <= hi)
            if m.sum() < 4:
                continue
            print("  [%.2f,%.2f) n=%4d |delta|均=%.4f max=%.4f"
                  % (lo, hi, int(m.sum()),
                     np.linalg.norm(cpp_delta[m], axis=1).mean(),
                     np.linalg.norm(cpp_delta[m], axis=1).max()))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
