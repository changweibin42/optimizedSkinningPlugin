# -*- coding: utf-8 -*-
r"""方向 A 第一轮：把曲线因子加进 CoR 的中心积分，做 A/B。

分两步：
  1. 用插件（alg0）取原 CoR 的最终顶点位置，作为「原版」基准
  2. 用纯 Python 复刻 cor_paper::computeCenters，先核对关掉曲线因子时
     能否复现原 CoR，再打开曲线因子看几何变化

曲线因子（跨模型按半径归一化）：
    g_ij = exp( -(Δs/radius)² / (2σ²) ),  σ = 0.1
    Δs = 采样点曲线参数 - 顶点曲线参数（沿骨骼轴的归一化弧长）

    mayapy tests/curveg_ab.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

HALF, BEND = 6.0, 146.1265
SIGMA_OVER_R = 0.1
CASES = ((24, 19, "lo"), (80, 201, "hi"))


def tube(sides, rings, half):
    pts, faces = [], []
    for row in range(rings):
        s = -half + row * (2 * half / (rings - 1))
        for j in range(sides):
            a = 2 * math.pi * j / sides
            pts.append((s, math.cos(a), math.sin(a)))
    for row in range(rings - 1):
        for j in range(sides):
            p = row * sides + j
            q = row * sides + (j + 1) % sides
            faces.append((p, q, q + sides, p + sides))
    return pts, faces


def scene(mc, om, sides, rings, label):
    """建圆柱 + 两骨 + 插件，返回 bind, posed(alg0), w1, faces。"""
    bind_list, faces = tube(sides, rings, HALF)
    bind = np.asarray(bind_list, float)
    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
    holder = mc.createNode("transform", name="%s_h" % label)
    geo = mc.createNode("transform", name="%s_g" % label, parent=holder)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [k for f in faces for k in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    prev = mc.createNode("joint", name="%s_pv" % label, parent=holder)
    mc.setAttr(prev + ".translateX", -HALF)
    elbow = mc.createNode("joint", name="%s_el" % label, parent=prev)
    mc.setAttr(elbow + ".translateX", HALF)
    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name="%s_sk" % label)[0]
    for k, jnt in enumerate((prev, elbow)):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
    w1 = np.array([ref.skin_weight(p[0]) for p in bind])
    for i in range(len(bind)):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                   1.0 - w1[i], w1[i], size=2)
    s = om.MSelectionList()
    s.add(elbow)
    om.MFnTransform(s.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
    sl = om.MSelectionList()
    sl.add(shape)
    mc.setAttr(skin + ".algorithmType", 0)
    mc.refresh(force=True)
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    posed = np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        sl.getDagPath(0)).getPoints(om.MSpace.kObject)])
    # 取插件算出的中心（若节点暴露）
    centers = None
    for attr in ("ocdCenterDump", "ocdCentersDump", "coRCentersDump"):
        if mc.attributeQuery(attr, node=skin, exists=True):
            try:
                centers = np.asarray(mc.getAttr("%s.%s" % (skin, attr)),
                                     float).reshape(-1, 3)
                break
            except Exception:
                pass
    mc.delete(holder)
    return bind, posed, w1, faces, centers


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        for sides, rings, tag in CASES:
            bind, posed, w1, faces, centers = scene(mc, om, sides, rings, tag)
            rad = float(np.max(np.hypot(bind[:, 1], bind[:, 2])))
            print("\n" + "=" * 76)
            print("%s 点  radius=%.4f  faces=%d" % (tag, rad, len(faces)))
            print("=" * 76)
            print("  alg0 顶点位移（相对 bind）：mean %.4f  max %.4f"
                  % (np.linalg.norm(posed - bind, axis=1).mean(),
                     np.linalg.norm(posed - bind, axis=1).max()))
            if centers is None:
                print("  插件未暴露中心 dump 属性（算法类型的实现可能是闭式，无缓存）")
            else:
                print("  取到中心 %s" % (centers.shape,))
            # 保存供后续纯 Python 对比
            np.save(os.path.join(HERE, "_aban_%s_bind.npy" % tag), bind)
            np.save(os.path.join(HERE, "_aban_%s_posed.npy" % tag), posed)
            np.save(os.path.join(HERE, "_aban_%s_w1.npy" % tag), w1)
            with open(os.path.join(HERE, "_aban_%s_faces.txt" % tag), "w") as fh:
                for f in faces:
                    fh.write(" ".join(str(x) for x in f) + "\n")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
