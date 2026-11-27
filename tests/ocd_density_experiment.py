# -*- coding: utf-8 -*-
r"""采样密度对照：保持半径、长度、姿态、权重函数，只改轴向分段数。

判断红圈褶皱是否随网格密度变化。若随密度显著变化，
说明 structuralCenters 的边求和项没有按物理长度归一。

    mayapy tests/ocd_density_experiment.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def make_cylinder(half_len=12.0, radius=0.9847, rings=201, sides=80):
    """沿 X 的细长圆柱，环数可调。rings = 轴向环数。"""
    points, faces = [], []
    for i in range(rings):
        x = -half_len + 2.0 * half_len * i / (rings - 1)
        for j in range(sides):
            a = 2.0 * math.pi * j / sides
            points.append((x, radius * math.cos(a), radius * math.sin(a)))
    for i in range(rings - 1):
        for j in range(sides):
            a = i * sides + j
            b = i * sides + (j + 1) % sides
            faces.append((a, b, b + sides, a + sides))
    # 两端封口
    first = tuple(range(sides - 1, -1, -1))
    last = tuple((rings - 1) * sides + j for j in range(sides))
    faces.append(first)
    faces.append(last)
    return points, faces


def reference_weight(x, half_len):
    """与你的夹具形状接近的平滑权重：以 x=0 为中心过渡。"""
    # 归一化到 [-1,1] 再取 smoothstep，过渡宽度与你的网格一致（约 7.44/24）
    t = x / half_len
    w = 0.5 + 0.5 * (t / 0.31)
    w = max(0.0, min(1.0, w))
    return w * w * (3.0 - 2.0 * w)


def build(mc, om, bind, faces, name, bend, weights):
    holder = mc.createNode("transform", name=name + "_holder")
    geo = mc.createNode("transform", name=name + "_geo", parent=holder)
    sel = om.MSelectionList()
    sel.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=sel.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")

    prev = mc.createNode("joint", name=name + "_prev", parent=holder)
    elbow = mc.createNode("joint", name=name + "_elbow", parent=prev)
    tip = mc.createNode("joint", name=name + "_tip", parent=elbow)
    for j, dx in ((prev, -12.0), (elbow, 12.0), (tip, 12.0)):
        mc.setAttr(j + ".translateX", dx)
        mc.setAttr(j + ".radius", 0.5)

    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name=name + "_skin#")[0]
    for k, j in enumerate((prev, elbow)):
        mc.connectAttr(j + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                   *mc.getAttr(j + ".worldInverseMatrix[0]"), type="matrix")
    for v, w in enumerate(weights):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, v),
                   1.0 - w, w, size=2)
    mc.setAttr(skin + ".algorithmType", 5)
    mc.setAttr(elbow + ".rotateZ", bend)
    mc.refresh(force=True)
    mc.dgeval(shape)
    return holder, shape, skin


def read(om, shape):
    sel = om.MSelectionList()
    sel.add(shape)
    return [[p.x, p.y, p.z] for p in
            om.MFnMesh(sel.getDagPath(0)).getPoints(om.MSpace.kObject)]


def turning_profile(bind, pts, sides, rings, half_len):
    """沿一条经线（j=0）计算径向轮廓与相邻段转角。"""
    prof = []
    for i in range(rings):
        v = i * sides
        prof.append((bind[v][0], math.hypot(pts[v][1], pts[v][2])))
    angles = []
    for k in range(1, len(prof) - 1):
        x0, r0 = prof[k - 1]
        x1, r1 = prof[k]
        x2, r2 = prof[k + 1]
        v1 = (x1 - x0, r1 - r0)
        v2 = (x2 - x1, r2 - r1)
        n1 = math.hypot(*v1)
        n2 = math.hypot(*v2)
        if n1 < 1e-12 or n2 < 1e-12:
            angles.append((x1, 0.0))
            continue
        c = max(-1.0, min(1.0, (v1[0] * v2[0] + v1[1] * v2[1]) / (n1 * n2)))
        angles.append((x1, math.degrees(math.acos(c))))
    return prof, angles


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        half_len = 12.0
        radius = 0.9847
        sides = 80
        bend = 146.126

        print("== 密度对照：只改轴向环数 ==")
        print("%6s %9s %10s %10s %12s %12s"
              % ("rings", "环距", "最小径向", "塌陷", "最大转角", "转角@红圈"))
        results = []
        for rings in (51, 101, 201, 401):
            bind, faces = make_cylinder(half_len, radius, rings, sides)
            weights = [reference_weight(p[0], half_len) for p in bind]
            holder, shape, skin = build(mc, om, bind, faces,
                                        "d%d" % rings, bend, weights)
            pts = read(om, shape)
            prof, angles = turning_profile(bind, pts, sides, rings, half_len)
            radial = [r for _, r in prof]
            collapse = min(radial) - radius
            worst = max(a for _, a in angles)
            # 红圈位置：x 在 [-1, 1] 附近（关节中心）
            near = [a for x, a in angles if abs(x) <= 1.5]
            near_worst = max(near) if near else 0.0
            results.append((rings, 2 * half_len / (rings - 1), min(radial),
                            collapse, worst, near_worst, angles, prof))
            print("%6d %9.4f %10.4f %10.4f %12.3f %12.3f"
                  % (rings, 2 * half_len / (rings - 1), min(radial),
                     collapse, worst, near_worst))
            mc.delete(holder)

        print("\n== 判定 ==")
        worst_turn = [r[4] for r in results]
        near_turn = [r[5] for r in results]
        print("  最大转角: %s" % ["%.2f" % v for v in worst_turn])
        print("  红圈转角: %s" % ["%.2f" % v for v in near_turn])
        ratio = max(near_turn) / max(1e-9, min(near_turn))
        print("  红圈转角变化倍数 = %.2f" % ratio)
        if ratio > 1.5:
            print("  => 褶皱随密度显著变化：structuralCenters 未按物理长度归一")
            failures.append("密度敏感 (倍数 %.2f)" % ratio)
        else:
            print("  => 褶皱对密度不敏感：不是采样密度问题")

        # 输出最细网格的轮廓以便看形状
        rings, spacing, rmin, collapse, wt, nt, angles, prof = results[-1]
        print("\n最细网格(rings=%d)的径向轮廓（x 在 [-2,2]）:" % rings)
        print("%9s %10s %10s" % ("x", "径向", "转角"))
        for k, (x, r) in enumerate(prof):
            if abs(x) > 2.0:
                continue
            a = 0.0
            for ax, aa in angles:
                if abs(ax - x) < spacing * 0.5:
                    a = aa
                    break
            print("%9.4f %10.4f %10.3f" % (x, r, a))

        print()
        if failures:
            for m in failures:
                print("NOTE:", m)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
