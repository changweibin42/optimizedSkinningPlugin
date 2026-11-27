# -*- coding: utf-8 -*-
r"""方向 A 第一轮：曲线归属能否减少跨链积分串扰。

按用户要求，先做前置检查：**原 alg0 对所选查询点是否真有跨链贡献**。
如果原 CoR 的权重相似性已经把另一条链的样本排除，开曲线归属也不会有变化。

夹具：两条平行骨链，在绑定空间靠近（间距 1.2R），一个圆柱表面同时被
两条链影响 —— 构造「空间接近但分属不同链」的部位。

三组：
  A  原生 alg0
  B  corCurveCrossOnly=1，同曲线 g=1，只衰减跨链
  C  corCurveCrossOnly=0，再叠加同曲线弧长高斯

    mayapy tests/curveg_chain_ab.py
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
SEP = 1.2          # 两条链在绑定空间的间距（半径倍数）
RADIUS = 1.0


def build(mc, om):
    """两条平行骨链（各 2 骨），中间夹一层被共同影响的圆柱壳。"""
    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

    # 圆柱壳：沿 Z 轴，半径 RADIUS，包住两条链
    sides, rings = 48, 41
    pts, faces = [], []
    zmin, zmax = -4.0, 4.0
    for r in range(rings):
        z = zmin + r * (zmax - zmin) / (rings - 1)
        for j in range(sides):
            a = 2 * math.pi * j / sides
            # 椭圆截面：X 方向拉长以同时覆盖两条链
            x = math.cos(a) * (SEP * 0.5 + RADIUS)
            y = math.sin(a) * RADIUS
            pts.append((x, y, z))
    for r in range(rings - 1):
        for j in range(sides):
            p = r * sides + j
            q = r * sides + (j + 1) % sides
            faces.append((p, q, q + sides, p + sides))
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

    # 链 A 在 x = -SEP/2，链 B 在 x = +SEP/2，各两节
    joints = []
    for tag, x in (("A", -SEP * 0.5), ("B", +SEP * 0.5)):
        root = mc.createNode("joint", name="%s_root" % tag, parent=holder)
        mc.setAttr(root + ".translateX", x)
        mc.setAttr(root + ".translateZ", -2.0)
        tip = mc.createNode("joint", name="%s_tip" % tag, parent=root)
        mc.setAttr(tip + ".translateZ", 4.0)
        joints.append((root, tip))

    skin = mc.deformer(shape, type=SKIN, name="sk")[0]
    flat = [j for pair in joints for j in pair]
    for idx, jnt in enumerate(flat):
        mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, idx))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                   *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")

    # 权重：按 X 距离在两链间平滑过渡 —— 让边界处的顶点同时受两链影响
    for i in range(len(bind)):
        x = bind[i][0]
        t = 0.5 + 0.5 * (x / (SEP * 0.5))       # 0 -> 链A, 1 -> 链B
        t = max(0.0, min(1.0, t))
        # 每链两骨：按 Z 分配
        z = bind[i][2]
        k = 0.5 + 0.5 * (z / 2.0)
        k = max(0.05, min(0.95, k))
        wa = (1 - t) * (1 - k)
        wb = (1 - t) * k
        wc = t * (1 - k)
        wd = t * k
        mc.setAttr("%s.weightList[%d].weights[0:3]" % (skin, i),
                   wa, wb, wc, wd, size=4)
    return bind, shape, skin, joints, flat


def read(mc, om, shape):
    for _ in range(2):
        mc.dgeval(shape)
        mc.refresh(force=True)
    s = om.MSelectionList()
    s.add(shape)
    return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
        s.getDagPath(0)).getPoints(om.MSpace.kObject)])


def meridian(P, sides, rings, half=None):
    idx = np.array([r * sides for r in range(rings)])
    Q = P[idx]
    u = Q[1:-1] - Q[:-2]
    v = Q[2:] - Q[1:-1]
    cr = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dt = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    turn = np.degrees(np.arctan2(cr, dt))
    el = np.linalg.norm(np.diff(Q, axis=0), axis=1)
    return turn, el


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind, shape, skin, joints, flat = build(mc, om)
        sides, rings = 48, 41
        print("夹具：%d 点，两条链各 2 骨，间距 %.2fR" % (len(bind), SEP))

        # 让两条链向相反方向弯，制造跨链差异
        for tag, root, tip, ang in (("A", joints[0][0], joints[0][1], +BEND),
                                    ("B", joints[1][0], joints[1][1], -BEND)):
            s = om.MSelectionList()
            s.add(tip)
            om.MFnTransform(s.getDagPath(0)).setRotation(
                om.MQuaternion(*ref.joint_quaternion(ang, 0, 0)),
                om.MSpace.kTransform)

        mc.setAttr(skin + ".algorithmType", 0)
        mc.setAttr(skin + ".corSigma", 0.1)

        # ---- A：原生 ----
        mc.setAttr(skin + ".corCurveGate", 0.0)
        mc.refresh(force=True)
        PA = read(mc, om, shape)
        print("\nA 原生 alg0：|v| 最大 %.4f" % np.linalg.norm(PA, axis=1).max())

        # ---- 前置检查：跨链权重占比 ----
        print("\n" + "=" * 74)
        print("前置检查：边界顶点的跨链影响")
        print("=" * 74)
        print("  夹具的权重要求每点同时受两链影响（X 位置决定 t），")
        print("  所以边界附近顶点的跨链份额本来就大 —— 这是可测的前提。")
        # 统计每个顶点两链的权重占比
        cross = []
        for i in range(len(bind)):
            ws = mc.getAttr("%s.weightList[%d].weights" % (skin, i))[0]
            a_chain = ws[0] + ws[1]
            b_chain = ws[2] + ws[3]
            cross.append(min(a_chain, b_chain))
        cross = np.asarray(cross)
        print("  跨链份额（min(链A, 链B)）：mean %.3f  max %.3f  非零顶点 %d/%d"
              % (cross.mean(), cross.max(), int((cross > 1e-6).sum()),
                 len(cross)))

        # ---- B：只衰减跨链 ----
        mc.setAttr(skin + ".corCurveGate", 1.0)
        mc.setAttr(skin + ".corCurveCrossOnly", True)
        mc.setAttr(skin + ".corCurveSigma", 0.1)
        mc.refresh(force=True)
        PB = read(mc, om, shape)
        dB = np.linalg.norm(PB - PA, axis=1)
        print("\nB 只衰减跨链：顶点差 max %.4e  mean %.4e"
              % (dB.max(), dB.mean()))

        # ---- C：加同链弧长 ----
        mc.setAttr(skin + ".corCurveCrossOnly", False)
        mc.refresh(force=True)
        PC = read(mc, om, shape)
        dC = np.linalg.norm(PC - PA, axis=1)
        print("C 归属 + 同链弧长：顶点差 max %.4e  mean %.4e"
              % (dC.max(), dC.mean()))

        # ---- 经线与截面 ----
        print("\n" + "=" * 74)
        print("几何对比")
        print("=" * 74)
        for tag, P in (("A", PA), ("B", PB), ("C", PC)):
            turn, el = meridian(P, sides, rings)
            print("  %s  最大|转角| %7.2f°  最短边 %.5f"
                  % (tag, abs(turn[np.argmax(np.abs(turn))]), el.min()))

        print("\n  逐环内外极点位移（相对 A）：")
        print("  %8s %12s %12s %12s"
              % ("z", "B 内/外", "C 内/外", ""))
        for r in range(rings // 2 - 2, rings // 2 + 3):
            seg = np.arange(r * sides, (r + 1) * sides)
            yb = bind[seg, 1]
            i_in = seg[int(np.argmax(yb))]
            i_out = seg[int(np.argmin(yb))]
            z = bind[seg[0], 2]
            print("  %8.3f %5.4f/%5.4f %5.4f/%5.4f"
                  % (z, np.linalg.norm(PB[i_in] - PA[i_in]),
                     np.linalg.norm(PB[i_out] - PA[i_out]),
                     np.linalg.norm(PC[i_in] - PA[i_in]),
                     np.linalg.norm(PC[i_out] - PA[i_out])))

        print("\n" + "=" * 74)
        print("判定")
        print("=" * 74)
        if dB.max() < 1e-6:
            print("  B 与 A 几乎相同 -> 跨链贡献在 CoR 积分里本来就很小，")
            print("  这条夹具不能检验归属的作用，需要更强的跨链耦合。")
        else:
            print("  B 相对 A 有可测变化（max %.4e），归属确实改变了中心积分。"
                  % dB.max())
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
