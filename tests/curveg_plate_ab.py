# -*- coding: utf-8 -*-
r"""方向 A 专用夹具：权重配方相同、曲线归属不同。

用户要求的设定：两片平行、互不连通的窄条，横向错开一小段。
沿长度复制**同一套混权分布**，绑到同两根骨骼，每片指定独立 curveId。
这样骨骼权重完全无法区分两片，只有显式归属能区分。

三组：
  A  原生 alg0
  B  只衰减跨曲线贡献（同曲线 g=1）
  C  本轮不做

三项指标：
  1. 跨曲线积分份额：乘 g 前 / 乘 g 后（需要在插件里统计，这里先用
     可观测的代理：中心偏移 + 顶点位移）
  2. 中心偏移：两个中心沿片间方向的移动
  3. 最终顶点位移

    mayapy tests/curveg_plate_ab.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

SKIN = "optimizedSkinCluster"
BEND = 90.0
GAP = 0.8          # 两片的横向错开量
LENGTH = 4.0
WIDTH = 0.5


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        # 两片窄条：沿 Z 方向，各自在 X 方向错开 ±GAP/2
        # 每片 nx x nz 顶点，两片不连通
        nx, nz = 5, 21
        pts, faces = [], []
        plateRanges = []
        for plate, xoff in ((0, -GAP * 0.5), (1, +GAP * 0.5)):
            base = len(pts)
            for iz in range(nz):
                z = -LENGTH * 0.5 + iz * LENGTH / (nz - 1)
                for ix in range(nx):
                    x = xoff + (ix - (nx - 1) * 0.5) * WIDTH / (nx - 1)
                    pts.append((x, 0.0, z))
            for iz in range(nz - 1):
                for ix in range(nx - 1):
                    a = base + iz * nx + ix
                    b = a + 1
                    c = a + nx + 1
                    d = a + nx
                    faces.append((a, b, c, d))
            plateRanges.append((base, len(pts)))
        bind = np.asarray(pts, float)

        holder = mc.createNode("transform", name="holder")
        geo = mc.createNode("transform", name="geo", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [k for f in faces for k in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]

        # 两根骨：沿 Z 的父骨 + 子骨
        root = mc.createNode("joint", name="root", parent=holder)
        mc.setAttr(root + ".translateZ", -LENGTH * 0.5)
        tip = mc.createNode("joint", name="tip", parent=root)
        mc.setAttr(tip + ".translateZ", LENGTH)

        skin = mc.deformer(shape, type=SKIN, name="sk")[0]
        for idx, jnt in enumerate((root, tip)):
            mc.connectAttr(jnt + ".worldMatrix[0]",
                           "%s.matrix[%d]" % (skin, idx))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                       *mc.getAttr(jnt + ".worldInverseMatrix[0]"),
                       type="matrix")

        # 关键：两片用**完全相同**的权重配方（只依赖 Z，不依赖 X 也不依赖片号）
        w1 = np.array([ref.skin_weight(p[2] * (2.0 / LENGTH)
                                       if False else p[2]) for p in bind])
        # 直接用 Z 归一化到 [-1,1] 再送 skin_weight
        zn = bind[:, 2] / (LENGTH * 0.5)
        w1 = np.array([ref.skin_weight(v * 6.0) for v in zn])
        for i in range(len(bind)):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w1[i], w1[i], size=2)

        # 手工曲线归属：片 0 -> curveId 0，片 1 -> curveId 1
        assign = [0.0] * len(bind)
        for base, end in (plateRanges[1],):
            for i in range(base, end):
                assign[i] = 1.0
        mc.setAttr(skin + ".corCurveAssign", assign, type="doubleArray")

        # 弯曲姿态
        s = om.MSelectionList()
        s.add(tip)
        om.MFnTransform(s.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)),
            om.MSpace.kTransform)

        mc.setAttr(skin + ".algorithmType", 0)
        mc.setAttr(skin + ".corSigma", 0.1)

        def read():
            for _ in range(2):
                mc.dgeval(shape)
                mc.refresh(force=True)
            sl = om.MSelectionList()
            sl.add(shape)
            return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
                sl.getDagPath(0)).getPoints(om.MSpace.kObject)])

        print("夹具：%d 点，两片各 %d 点，横向间距 %.2f，权重配方完全相同"
              % (len(bind), nx * nz, GAP))
        print("  片0 x∈[%.2f, %.2f]  片1 x∈[%.2f, %.2f]"
              % (bind[plateRanges[0][0]][0], bind[plateRanges[0][1] - 1][0],
                 bind[plateRanges[1][0]][0], bind[plateRanges[1][1] - 1][0]))
        print("  权重只依赖 Z，两片在相同 Z 处权重完全一致")

        mc.setAttr(skin + ".corCurveGate", 0.0)
        mc.refresh(force=True)
        PA = read()

        mc.setAttr(skin + ".corCurveGate", 1.0)
        mc.setAttr(skin + ".corCurveCrossOnly", True)
        mc.refresh(force=True)
        PB = read()

        d = np.linalg.norm(PB - PA, axis=1)
        print("\n" + "=" * 72)
        print("A/B 结果")
        print("=" * 72)
        print("  B 相对 A 的顶点位移：max %.6e  mean %.6e" % (d.max(), d.mean()))

        # 逐片的位移统计
        for p, (base, end) in enumerate(plateRanges):
            seg = np.arange(base, end)
            print("  片%d：max %.4e  mean %.4e"
                  % (p, d[seg].max(), d[seg].mean()))

        # 片间方向（X）的位移分量：这个最能反映「另一片是否把中心拉偏」
        dx = PB[:, 0] - PA[:, 0]
        print("\n  X 方向（片间）位移分量：")
        for p, (base, end) in enumerate(plateRanges):
            seg = np.arange(base, end)
            print("  片%d：mean %+.4e  max|.| %.4e"
                  % (p, dx[seg].mean(), np.abs(dx[seg]).max()))

        # 同 Z 处两片的相对位移变化（是否被拉向彼此）
        print("\n  同 Z 处两片的 X 位置变化（负=靠拢，正=分开）：")
        print("  %8s %12s %12s %12s"
              % ("z", "片0 dx", "片1 dx", "间距变化"))
        for iz in range(0, nz, 4):
            i0 = plateRanges[0][0] + iz * nx + nx // 2
            i1 = plateRanges[1][0] + iz * nx + nx // 2
            gapA = PA[i1][0] - PA[i0][0]
            gapB = PB[i1][0] - PB[i0][0]
            print("  %8.3f %+12.5f %+12.5f %+12.5f"
                  % (bind[i0][2], dx[i0], dx[i1], gapB - gapA))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
