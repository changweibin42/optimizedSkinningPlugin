# -*- coding: utf-8 -*-
r"""决定性实验：在 HALF=6 圆柱上直接对比三种形状。

  A) Elbow 解析模型（Python 5 模型窗口里 "OCD" 那一格实际显示的东西）
  B) CoR + delta（structuralCenters 求解，C++ alg5/alg8 实际算的东西）
  C) 纯 CoR（无 delta）

如果 B 有褶皱而 A 没有，则褶皱 100% 来自 structuralCenters，
与尺度无关，与 C++ 移植无关。

    mayapy tests/ocd_cylinder_solver_vs_analytic.py
"""
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

        sys.path.insert(0, HERE)
        import ocd_elbow_reference_latest as ref

        HALF = ref.HALF
        RINGS = 2 * ref.HALF_RINGS + 1
        SIDES = ref.SIDES
        bend = 146.1265
        print("HALF=%.2f rings=%d sides=%d bend=%.2f"
              % (HALF, RINGS, SIDES, bend))

        bind_pts, faces = ref.cylinder()
        n = len(bind_pts)
        bind = [tuple(p) for p in bind_pts]
        import numpy as np
        P = np.asarray(bind, float)

        # 权重：与 show() 完全一致
        w = np.array([ref.skin_weight(p[0]) for p in bind])
        W = np.array([ref.rig_weights(wi, False) for wi in w])
        print("权重 w1 范围 %.4f .. %.4f" % (W[:, 1].min(), W[:, 1].max()))

        edges = ref.local_edges(bind, faces)
        q = ref.joint_quaternion(bend, 0.0, 0.0)
        rotations = [(0.0, 0.0, 0.0, 1.0), q]

        # 纯 CoR（alg0）作为 original
        # 用 Elbow 自身的 section 定义算 CoR 很麻烦，这里直接在 Maya 里算
        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        holder = mc.createNode("transform", name="g_h")
        geo = mc.createNode("transform", name="g_g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        mc.sets(geo, edit=True, forceElement="initialShadingGroup")

        prev = mc.createNode("joint", name="g_prev", parent=holder)
        elbow = mc.createNode("joint", name="g_elbow", parent=prev)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", HALF)

        skin = mc.deformer(shape, type="optimizedSkinCluster", name="g_s#")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
        for v in range(n):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, v),
                       float(W[v, 0]), float(W[v, 1]), size=2)
        mc.setAttr(skin + ".algorithmType", 0)      # 纯 CoR
        mc.setAttr(elbow + ".rotateZ", bend)
        mc.refresh(force=True)
        mc.dgeval(shape)

        def read():
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        def inject(values):
            s = om.MSelectionList()
            s.add(skin)
            om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                "ocdCenterOffsets", False).setMObject(
                    om.MFnDoubleArrayData().create(om.MDoubleArray(values)))
            mc.refresh(force=True)

        # 纯 CoR
        inject([])
        cor = read()

        # Python 求解
        original = np.asarray(cor, float)
        guide = np.asarray(ref.hinge_displacement(bind, original, W, bend, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind, W, bend, 0.0, 0.0))
        delta = np.asarray(ref.structural_centers(edges, original, W, rotations,
                                                  guide, outer, 16.0))
        print("|guide|均=%.5f  |delta|均=%.5f  ratio=%.1f"
              % (np.linalg.norm(guide, axis=1).mean(),
                 np.linalg.norm(delta, axis=1).mean(),
                 np.linalg.norm(delta, axis=1).mean() /
                 max(1e-12, np.linalg.norm(guide, axis=1).mean())))

        # A) Elbow 解析
        analytic = [list(p) for p in ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False).mesh(bind)]

        # B) CoR + delta
        inject(delta.ravel().tolist())
        solved = read()

        # 圆度（在局部 station 上，撤掉骨骼旋转）
        th = math.radians(bend)
        ct, st = math.cos(th), math.sin(th)

        def unrotate(pts):
            out = []
            for i, p in enumerate(pts):
                x, y, z = p
                if bind[i][0] > 0:
                    x, y = ct * x + st * y, -st * x + ct * y
                out.append((x, y, z))
            return out

        def stats(pts):
            o = {}
            for x0 in (-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0):
                idx = [i for i in range(n)
                       if abs(bind[i][0] - x0) < HALF / ref.HALF_RINGS / 2]
                if len(idx) < 8:
                    continue
                rr = [math.hypot(pts[i][1], pts[i][2]) for i in idx]
                m = sum(rr) / len(rr)
                v = sum((t - m) ** 2 for t in rr) / len(rr)
                o[x0] = (math.sqrt(v) / max(1e-12, m), m)
            return o

        xs = [-6.0, -4.0, -2.0, -1.0, 0.0, 1.0, 2.0, 4.0, 6.0]
        sa = stats(analytic)
        sc = stats(unrotate(cor))
        ss = stats(unrotate(solved))

        print("\n== 圆度 std/mean（越小越圆）==")
        print("%12s" % "" + "".join("  x=%+.0f" % v for v in xs))
        for name, s in (("A 解析Elbow", sa), ("B CoR+delta", ss), ("C 纯CoR", sc)):
            line = "%12s" % name
            for x0 in xs:
                line += " %6.3f" % s[x0][0] if x0 in s else "      -"
            print(line)

        print("\n== 径向均值 ==")
        print("%12s" % "" + "".join("  x=%+.0f" % v for v in xs))
        for name, s in (("A 解析Elbow", sa), ("B CoR+delta", ss), ("C 纯CoR", sc)):
            line = "%12s" % name
            for x0 in xs:
                line += " %6.3f" % s[x0][1] if x0 in s else "      -"
            print(line)

        # 与解析的偏差
        print("\n== B vs A/C 的径向差 ==")
        for x0 in xs:
            if x0 in sa and x0 in ss and x0 in sc:
                print("  x=%+.0f  A=%.4f  B=%.4f (Δ%+.4f)  C=%.4f (Δ%+.4f)"
                      % (x0, sa[x0][1], ss[x0][1], ss[x0][1] - sa[x0][1],
                         sc[x0][1], sc[x0][1] - sa[x0][1]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
