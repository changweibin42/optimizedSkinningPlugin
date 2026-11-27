# -*- coding: utf-8 -*-
r"""纵向验收：118 / 120 / 146.1265 三档，红圈经线的有符号转角。

四项对比（每档角度）：
  1) 原引导        original + guide(原始梯度)
  2) 归一化引导    original + guide(单位梯度)
  3) 原最终 CoR    注入原始 delta
  4) 归一化最终     注入归一化 delta

转角定义：固定一条经线（环内序号 j），沿 x 排序后，
用相邻三点夹角衡量"折痕"。也报告曲率（二阶差分）。

    mayapy tests/longitudinal_angle.py tests/_cyl1.json
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
        import ocd_elbow_reference_latest as ref

        s1 = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tests/_cyl1.json"))
        bind = [tuple(p) for p in s1["bind"]]
        faces = s1["faces"]
        n = len(bind)
        HALF = s1["HALF"]
        weights = s1["weights"]
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

        def read():
            for _ in range(2):
                mc.refresh(force=True)
                mc.dgeval(shape)
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

        def turning(P, row, j):
            """以环 j 为中心的三点转角（度），沿 x 方向。"""
            if row <= 0 or row >= RINGS - 1:
                return None
            a = P[(row - 1) * SIDES + j]
            b = P[row * SIDES + j]
            c = P[(row + 1) * SIDES + j]
            u = [b[i] - a[i] for i in range(3)]
            v = [c[i] - b[i] for i in range(3)]
            lu = math.sqrt(sum(x * x for x in u))
            lv = math.sqrt(sum(x * x for x in v))
            if lu < 1e-12 or lv < 1e-12:
                return None
            d = sum(u[i] * v[i] for i in range(3)) / (lu * lv)
            return math.degrees(math.acos(max(-1.0, min(1.0, d))))

        import numpy as _np
        P = _np.asarray(bind, float)
        stations = P[:, 0]
        w1 = _np.array([w[1] for w in weights])
        W2 = _np.column_stack([1.0 - w1, w1])

        for bend in (118.0, 120.0, 146.1265):
            mc.setAttr(elbow + ".rotateZ", bend)
            mc.setAttr(skin + ".algorithmType", 0)
            inject([])
            base = read()

            angle = math.radians(abs(bend))
            ext = _np.full_like(stations, 1.0)
            inner = ref.hinge_section(stations, ext, 1.0, W2, angle)
            outer = ref.hinge_section(stations, -ext, 1.0, W2, angle)
            neutral = (inner + outer) * 0.5
            grad = (inner - outer) / 2.0
            k = _np.linalg.norm(grad, axis=1, keepdims=True)
            grad_u = grad / _np.maximum(k, 1e-12)

            # 用 hinge_displacement 的完整 3D 路径，只换 gradient
            def guide_with(gg):
                y = P[:, 1]
                z = P[:, 2]
                depth = y
                lateral = z
                x = neutral[:, 0] + depth * gg[:, 0]
                h = neutral[:, 1] + depth * gg[:, 1]
                target = _np.column_stack([x, h, lateral])
                # 参考实现还要减去 original；这里 original 用 bind 自身
                return target - P

            g_raw = guide_with(grad)
            g_unit = guide_with(grad_u)
            guided_raw = [[base[i][kk] + g_raw[i][kk] for kk in range(3)]
                          for i in range(n)]
            guided_unit = [[base[i][kk] + g_unit[i][kk] for kk in range(3)]
                           for i in range(n)]

            # 求 delta：需要局部归一坐标
            child_pivot = (HALF, 0.0, 0.0)
            R = 1.0
            bind_l = _np.column_stack([
                stations / R, P[:, 1] / R, P[:, 2] / R])

            edges = ref.local_edges(bind.tolist() if False else bind_l.tolist(), faces)
            q = ref.joint_quaternion(bend, 0.0, 0.0)
            rots = [(0.0, 0.0, 0.0, 1.0), q]
            outer_p = _np.asarray(ref.outer_participation(bind_l, W2, bend, 0.0, 0.0))

            def solve_delta(base_pts, guide_pts):
                base_l = _np.column_stack([
                    _np.asarray([p[0] for p in base_pts]) / R,
                    _np.asarray([p[1] for p in base_pts]) / R,
                    _np.asarray([p[2] for p in base_pts]) / R])
                gl = _np.column_stack([
                    _np.asarray([p[0] for p in guide_pts]) / R,
                    _np.asarray([p[1] for p in guide_pts]) / R,
                    _np.asarray([p[2] for p in guide_pts]) / R])
                return ref.structural_centers(edges, base_l, W2, rots,
                                              gl - base_l, outer_p, 16.0)

            d_raw = solve_delta(base, guided_raw)
            d_unit = solve_delta(base, guided_unit)

            mc.setAttr(skin + ".algorithmType", 5)
            inject(list(_np.asarray(d_raw).ravel()))
            fin_raw = read()
            inject(list(_np.asarray(d_unit).ravel()))
            fin_unit = read()

            print("\n" + "=" * 74)
            print("bend=%.4f" % bend)
            print("=" * 74)
            # 找红圈：129 度附近（内侧），用 j=0 和 j=SIDES/2
            for j, jn in ((0, "内侧 j=0"), (SIDES // 2, "外侧 j=%d" % (SIDES // 2))):
                print("\n  --- %s 的纵向转角（度）---" % jn)
                print("  %8s %8s %10s %10s %10s %10s"
                      % ("x", "w1", "原引导", "归一引导", "原最终", "归一最终"))
                for row in range(1, RINGS - 1):
                    x0 = -HALF + row * (2 * HALF / (RINGS - 1))
                    if abs(x0) > 2.6:
                        continue
                    t = [turning(P2, row, j)
                         for P2 in (guided_raw, guided_unit, fin_raw, fin_unit)]
                    if any(v is None for v in t):
                        continue
                    print("  %8.3f %8.4f %10.3f %10.3f %10.3f %10.3f"
                          % (x0, w1[row * SIDES], t[0], t[1], t[2], t[3]))

                # 尖峰指标：最大转角
                mx = [0.0] * 4
                for row in range(1, RINGS - 1):
                    t = [turning(P2, row, j)
                         for P2 in (guided_raw, guided_unit, fin_raw, fin_unit)]
                    if any(v is None for v in t):
                        continue
                    for q2 in range(4):
                        mx[q2] = max(mx[q2], t[q2])
                print("  最大转角: 原引导 %.3f  归一引导 %.3f  原最终 %.3f  归一最终 %.3f"
                      % tuple(mx))

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
