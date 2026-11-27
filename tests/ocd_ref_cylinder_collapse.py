"""在参考圆珠上跑完整算法5（C++），看它是否也凹陷。

这是最关键的对照：同一个插件、同一套算法，只有几何不同。
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference as ref

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()

        holder = mc.createNode("transform", name="rc_holder")
        geo = mc.createNode("transform", name="rc_geo", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        mc.sets(geo, edit=True, forceElement="initialShadingGroup")

        prev = mc.createNode("joint", name="rc_prev", parent=holder)
        elbow = mc.createNode("joint", name="rc_elbow", parent=prev)
        tip = mc.createNode("joint", name="rc_tip", parent=elbow)
        for j, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
            mc.setAttr(j + ".translateX", dx)
            mc.setAttr(j + ".radius", 0.09)

        skin = mc.deformer(shape, type="optimizedSkinCluster",
                           name="rc_skin#")[0]
        for k, j in enumerate((prev, elbow)):
            mc.connectAttr(j + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                       *mc.getAttr(j + ".worldInverseMatrix[0]"), type="matrix")
        for v, (s, _, _) in enumerate(bind):
            w = ref.skin_weight(s)
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, v),
                       1.0 - w, w, size=2)

        mc.setAttr(skin + ".algorithmType", 5)
        mc.setAttr(skin + ".corDebugLog", False)

        print("== 参考圆珠，不同弯曲角下的径向最小值 ==")
        print("%8s %10s %10s %10s" % ("bend", "R_bind", "R_final", "塌陷"))
        R_bind = [math.hypot(p[1], p[2]) for p in bind]
        for bend in (60.0, 90.0, 118.0, 146.0, 160.0):
            mc.setAttr(elbow + ".rotateZ", bend)
            mc.refresh(force=True)
            mc.dgeval(shape)
            sel2 = om.MSelectionList()
            sel2.add(shape)
            pts = [(p.x, p.y, p.z) for p in
                   om.MFnMesh(sel2.getDagPath(0)).getPoints(om.MSpace.kObject)]
            Rf = [math.hypot(p[1], p[2]) for p in pts]
            # 只统计真正在管壁上的点（bind 半径接近最大）
            Rmax = max(R_bind)
            wall = [i for i, r in enumerate(R_bind) if r > 0.9 * Rmax]
            rw = sorted(Rf[i] for i in wall)
            print("%8.1f %10.4f %10.4f %10.4f  (n=%d)"
                  % (bend, Rmax, rw[0], rw[0] - Rmax, len(wall)))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
