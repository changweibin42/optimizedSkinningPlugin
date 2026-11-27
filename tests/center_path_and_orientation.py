# -*- coding: utf-8 -*-
"""测中心轨迹与截面朝向，定位纵向折痕来源。

对引导形状和最终 CoR 分别，逐环求：
  · 环质心（中心轨迹）
  · 环最佳拟合平面法向（截面朝向）
再沿 x 看它们的二阶差分（突变点）。

    mayapy tests/center_path_and_orientation.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def fit_ring(P, row, sides):
    """返回 (质心, 单位法向, 面内半径 list)。"""
    idx = [row * sides + j for j in range(sides)]
    Q = [P[i] for i in idx]
    c = [sum(q[k] for q in Q) / sides for k in range(3)]
    d = [[q[k] - c[k] for k in range(3)] for q in Q]

    def cross(a, b):
        return [a[1] * b[2] - a[2] * b[1],
                a[2] * b[0] - a[0] * b[2],
                a[0] * b[1] - a[1] * b[0]]

    def dot(a, b):
        return sum(a[i] * b[i] for i in range(3))

    def norm(a):
        m = math.sqrt(dot(a, a))
        return [x / m for x in a], m

    # Newell 法向（对圆环稳定）
    nx = [0.0, 0.0, 0.0]
    for j in range(sides):
        a = d[j]
        b = d[(j + 1) % sides]
        cr = cross(a, b)
        for k in range(3):
            nx[k] += cr[k]
    nrm, ln = norm(nx) if math.sqrt(dot(nx, nx)) > 1e-14 else ([0.0, 0.0, 1.0], 1.0)

    rad = []
    for p in d:
        h = dot(p, nrm)
        proj = [p[k] - nrm[k] * h for k in range(3)]
        rad.append((math.sqrt(dot(proj, proj)), h))
    return c, nrm, rad


def angle_between(a, b):
    d = sum(a[i] * b[i] for i in range(3))
    return math.degrees(math.acos(max(-1.0, min(1.0, d))))


def second_diff(values):
    """沿序列取二阶差分（数量级）。"""
    out = []
    for i in range(1, len(values) - 1):
        out.append(values[i + 1] - 2 * values[i] + values[i - 1])
    return out


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        d1 = json.load(open(os.path.join(HERE, "_la1.json")))
        d2 = json.load(open(os.path.join(HERE, "_la2.json")))
        bind = [tuple(p) for p in d1["bind"]]
        faces = d1["faces"]
        n = len(bind)
        HALF = d1["HALF"]
        weights = d1["weights"]
        SIDES = ref.SIDES
        RINGS = 2 * ref.HALF_RINGS + 1

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="prev", parent=holder)
        elbow = mc.createNode("joint", name="elbow", parent=prev)
        mc.setAttr(prev + ".translateX", -HALF)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="s#")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
        for i, w in enumerate(weights):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       float(w[0]), float(w[1]), size=2)
        for attr in ("ocdCenterGateLow", "ocdCenterGateHigh"):
            if mc.attributeQuery(attr, node=skin, exists=True):
                mc.setAttr("%s.%s" % (skin, attr), 0.0)

        def read(algo, inj=None):
            mc.setAttr(skin + ".algorithmType", algo)
            if inj is not None:
                s = om.MSelectionList()
                s.add(skin)
                om.MFnDependencyNode(s.getDependNode(0)).findPlug(
                    "ocdCenterOffsets", False).setMObject(
                        om.MFnDoubleArrayData().create(om.MDoubleArray(inj)))
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shape)
            s = om.MSelectionList()
            s.add(shape)
            return [[p.x, p.y, p.z] for p in
                    om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]

        bend_s = "146.1265"
        bend = float(bend_s)
        dd = d2[bend_s]
        mc.setAttr(elbow + ".rotateZ", bend)

        base = read(0, [])

        # 引导形状 = base + guide(原始梯度)
        G = dd["guideRaw"]
        guided = [[base[i][k] + G[3 * i + k] for k in range(3)] for i in range(n)]

        fin = read(5, dd["deltaRaw"])

        sets = [("BaseCoR  ", base), ("引导      ", guided), ("最终CoR   ", fin)]

        print("bend=146.1265  环数 %d  每环 %d 点" % (RINGS, SIDES))

        # ---- 中心轨迹 ----
        print("\n" + "=" * 96)
        print("中心轨迹（环质心）与截面朝向（Newell 法向）")
        print("=" * 96)
        hdr = "  %8s %8s |" % ("x", "w1")
        for nm, _ in sets:
            hdr += " %-28s |" % nm
        print(hdr)
        print("  %18s|" % "" + "".join(" %10s %16s |" % ("质心", "法向变化°") for _ in sets))

        prev_n = {nm: None for nm, _ in sets}
        rows = []
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            if abs(x0) > 2.7:
                continue
            line = "  %8.3f %8.4f |" % (x0, weights[row * SIDES][1])
            rec = {}
            for nm, P2 in sets:
                c, nrm, rad = fit_ring(P2, row, SIDES)
                ch = 0.0
                if prev_n[nm] is not None:
                    ch = angle_between(prev_n[nm], nrm)
                prev_n[nm] = nrm
                rec[nm] = (c, nrm, ch)
                line += " (%7.3f,%7.3f) %14.3f |" % (c[0], c[1], ch)
            rows.append((x0, rec))
            print(line)

        # ---- 中心轨迹二阶差分 ----
        print("\n" + "=" * 96)
        print("中心轨迹二阶差分（突变指标，标量为 |c_i+1 - 2c_i + c_i-1|）")
        print("=" * 96)
        print("  %8s |" % "x" + "".join(" %18s |" % nm for nm, _ in sets))
        for nm, _ in sets:
            pass
        for idx in range(1, len(rows) - 1):
            x0 = rows[idx][0]
            line = "  %8.3f |" % x0
            for nm, _ in sets:
                a = rows[idx - 1][1][nm][0]
                b = rows[idx][1][nm][0]
                c = rows[idx + 1][1][nm][0]
                dd2 = math.sqrt(sum((c[k] - 2 * b[k] + a[k]) ** 2 for k in range(3)))
                line += " %18.5f |" % dd2
            print(line)

        # ---- 朝向二阶差分 ----
        print("\n" + "=" * 96)
        print("截面朝向变化（与前环法向的夹角，度）")
        print("=" * 96)
        for nm, _ in sets:
            vals = [r[1][nm][2] for r in rows]
            xs = [r[0] for r in rows]
            print("\n  --- %s ---" % nm)
            print("  %8s %10s" % ("x", "转角"))
            for x0, v in zip(xs, vals):
                print("  %8.3f %10.3f" % (x0, v))
            mx = max(vals)
            imx = vals.index(mx)
            print("  最大 %.3f° @x=%+.3f" % (mx, xs[imx]))

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
