# -*- coding: utf-8 -*-
"""复现 Python 验证的 report()：把 Elbow 解析模型、CoR、OCD+CoR 三者对比。

关键：Elbow.mesh() 是**解析模型**（OUTER/INNER 那组 150° 拟合值），
它不使用 delta。所以「Python 版没有凹陷」可能指的是解析模型本身。

    mayapy tests/ocd_elbow_vs_solver.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        bind, faces = ref.cylinder()
        holder = mc.createNode("transform", name="ev_holder")
        geo = mc.createNode("transform", name="ev_geo", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        mc.sets(geo, edit=True, forceElement="initialShadingGroup")

        prev = mc.createNode("joint", name="ev_prev", parent=holder)
        elbow = mc.createNode("joint", name="ev_elbow", parent=prev)
        tip = mc.createNode("joint", name="ev_tip", parent=elbow)
        for j, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
            mc.setAttr(j + ".translateX", dx)
            mc.setAttr(j + ".radius", 0.09)

        skin = mc.deformer(shape, type="optimizedSkinCluster",
                           name="ev_skin#")[0]
        for k, j in enumerate((prev, elbow)):
            mc.connectAttr(j + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                       *mc.getAttr(j + ".worldInverseMatrix[0]"), type="matrix")
        gw = []
        for v, (s, _, _) in enumerate(bind):
            w = ref.skin_weight(s)
            gw.append(w)
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, v),
                       1.0 - w, w, size=2)

        def read():
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        # 局部径向（插件口径）：joint 在原点、沿 +X、半径 1
        def local_radial(pts, bend):
            th = math.radians(bend)
            out = []
            for p in pts:
                x, y, z = p
                # 转到 bind 系：撤销子骨旋转（绕 Z，正角）
                c, s = math.cos(th), math.sin(th)
                bx = c * x + s * y
                by = -s * x + c * y
                sgn = 1.0 if bx >= 0 else -1.0
                out.append(math.hypot(by, z))
            return out

        print("== 三种模型的关节区径向（bend=150 与 146.1）==")
        for bend in (146.126, 150.0):
            mc.setAttr(elbow + ".rotateZ", bend)
            mc.refresh(force=True)

            mc.setAttr(skin + ".algorithmType", 0)
            mc.dgeval(shape)
            mc.refresh(force=True)
            cor = read()

            mc.setAttr(skin + ".algorithmType", 5)
            mc.dgeval(shape)
            mc.refresh(force=True)
            fin = read()

            analytic = [list(p) for p in
                        ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False).mesh(bind)]

            Rc = local_radial(cor, bend)
            Rf = local_radial(fin, bend)
            Ra = local_radial(analytic, bend)

            xs = [p[0] for p in bind]
            near = [i for i in range(len(bind)) if abs(xs[i]) < 2.0]
            print("\n bend=%.2f" % bend)
            print("  %-10s %10s %10s %10s" % ("", "min", "max", "mean"))
            for label, R in (("CoR", Rc), ("OCD+CoR", Rf),
                             ("Elbow解析", Ra)):
                vals = [R[i] for i in near]
                print("  %-10s %10.4f %10.4f %10.4f"
                      % (label, min(vals), max(vals), sum(vals) / len(vals)))

            # 逐环
            print("  %8s %10s %10s %10s" % ("x", "CoR", "OCD+CoR", "解析"))
            for target in [i * 0.5 for i in range(-4, 5)]:
                s = [i for i in range(len(bind)) if abs(xs[i] - target) < 0.031]
                if len(s) < 10:
                    continue
                m = lambda R: sum(R[i] for i in s) / len(s)
                print("  %8.2f %10.4f %10.4f %10.4f"
                      % (target, m(Rc), m(Rf), m(Ra)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
