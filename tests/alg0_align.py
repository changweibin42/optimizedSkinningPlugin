# -*- coding: utf-8 -*-
r"""先对齐算法 0 的坐标与矩阵约定：纯单骨点对拍。

用户在 616f636 的源码里确认：
  · deformCoR 用 Maya 行向量约定：p_out = p_bind · M
  · 变换在**世界空间**做（restWorld = p * worldMat，最后 * worldInverse）
  · vertexTransform 返回完整仿射变换（含平移），不只旋转

纯单骨点：旋转中心对最终位置没有贡献，所以
    p_out = p_bind · M_geo · M_bindPre · M_pose · M_geo^{-1}
如果这个点都差几个单位，问题就在约定/空间，不在中心积分。

    mayapy tests/alg0_align.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

HALF, BEND = 6.0, 146.1265


def maya_mat_to_np(m):
    """Maya MMatrix 按行主序读出来是 16 个数；转成 numpy 4x4（行向量约定）。"""
    return np.asarray(m, float).reshape(4, 4)


def rowvec_apply(p, M):
    """Maya 行向量：p_out = p_row · M。"""
    v = np.append(np.asarray(p, float), 1.0)
    return (v @ M)[:3]


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind = np.load(os.path.join(HERE, "_aban_lo_bind.npy"))
        posed = np.load(os.path.join(HERE, "_aban_lo_posed.npy"))
        w1 = np.load(os.path.join(HERE, "_aban_lo_w1.npy"))
        sides, rings = 24, 19

        # 找纯单骨点：w1 最大（子骨全权）和 w1 最小（父骨全权）
        i_child = int(np.argmax(w1))
        i_parent = int(np.argmin(w1))
        print("单骨点：child 顶点 %d (w1=%.6f, station %+.3f)"
              % (i_child, w1[i_child], bind[i_child][0]))
        print("        parent 顶点 %d (w1=%.6f, station %+.3f)"
              % (i_parent, w1[i_parent], bind[i_parent][0]))

        # 重建场景取矩阵
        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        faces = []
        with open(os.path.join(HERE, "_aban_lo_faces.txt")) as fh:
            for line in fh:
                faces.append(tuple(int(x) for x in line.split()))
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [k for f in faces for k in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="pv", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="el", parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="sk")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                       *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
        for i in range(len(bind)):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w1[i], w1[i], size=2)
        s = om.MSelectionList()
        s.add(elbow)
        om.MFnTransform(s.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
        mc.setAttr(skin + ".algorithmType", 0)
        mc.refresh(force=True)
        for _ in range(2):
            mc.dgeval(shape)
            mc.refresh(force=True)

        # 收集矩阵
        geo_mat = maya_mat_to_np(mc.getAttr(geo + ".worldMatrix[0]"))
        prev_mat = maya_mat_to_np(mc.getAttr(prev + ".worldMatrix[0]"))
        elbow_mat = maya_mat_to_np(mc.getAttr(elbow + ".worldMatrix[0]"))
        print("\n几何世界矩阵:\n%s" % np.array2string(geo_mat, precision=4))

        # 用行向量约定独立复算单骨点
        print("\n" + "=" * 78)
        print("单骨点独立复算：bindToPose = bindPreMatrix * worldMatrix")
        print("  bindPreMatrix 就是绑定时的世界逆矩阵，不要再求逆")
        print("=" * 78)
        for tag, idx, jnt_mat, slot in (
                ("parent", i_parent, prev_mat, 0), ("child", i_child, elbow_mat, 1)):
            pre = maya_mat_to_np(mc.getAttr("%s.bindPreMatrix[%d]" % (skin, slot)))
            # bindToPose = bindPreMatrix * matrix（行向量约定，左到右）
            bindToPose = pre @ jnt_mat
            # 世界空间：p_out = p * geo * bindToPose * geo^-1
            M = geo_mat @ bindToPose @ np.linalg.inv(geo_mat)
            p = rowvec_apply(bind[idx], M)
            print("  %-6s 顶点 %5d  bind (%+.4f,%+.4f,%+.4f)"
                  % (tag, idx, *bind[idx]))
            print("         bindToPose =\n%s"
                  % np.array2string(bindToPose, precision=4))
            print("         复算 (%+.4f,%+.4f,%+.4f)" % tuple(p))
            print("         插件 (%+.4f,%+.4f,%+.4f)" % tuple(posed[idx]))
            print("         差 %.4e" % np.linalg.norm(p - posed[idx]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
