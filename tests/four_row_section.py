# -*- coding: utf-8 -*-
r"""四行最小决定性实验（458 点圆柱，HALF=6，x=±0.67 环，圆度 0.1305）。

在同一坐标系、同一截面测法下比较：
  1) Base CoR            = original（算法 0）
  2) 算法5 引导           = original + guide  （局部归一空间，转换到世界）
  3) Python 优化 CoR      = 注入 Python 独立求的 delta 后的最终网格
  4) C++ 自动 alg5        = 自动求值后的最终网格

每行给：
  · 圆度 std/mean
  · 沿环一周的有符号径向残差（最深点方位）
  · 截面最佳拟合平面的离面误差（区分翘曲与面内压扁）

门控显式 0/0，与 7e901b9 语义一致。

    mayapy tests/four_row_section.py tests/_cyl1.json tests/_cyl2.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

SIDES = 24
HALF_RINGS = 9
RINGS = 2 * HALF_RINGS + 1


def section_metrics(P, row, sides=SIDES):
    """用一整个环的顶点，求圆度 / 有符号径向残差 / 离面误差。

    全部基于该环自身：质心 + PCA 最佳拟合平面。
    """
    idx = [row * sides + j for j in range(sides)]
    Q = [P[i] for i in idx]
    c = [sum(q[k] for q in Q) / sides for k in range(3)]
    d = [[q[k] - c[k] for k in range(3)] for q in Q]

    # 3x3 协方差，取最小特征向量的近似：用 SVD 的替代——幂迭代求最大，
    # 再在正交补里求次大，剩下就是法向。
    cov = [[sum(p[i] * p[j] for p in d) for j in range(3)] for i in range(3)]

    def matvec(m, v):
        return [sum(m[i][j] * v[j] for j in range(3)) for i in range(3)]

    def norm(v):
        return math.sqrt(sum(x * x for x in v))

    # 三个基向量幂迭代
    basis = []
    work = [list(row) for row in cov]
    for _ in range(3):
        v = [1.0, 0.3, 0.7]
        for _ in range(80):
            v = matvec(work, v)
            n = norm(v)
            if n < 1e-18:
                break
            v = [x / n for x in v]
        lam = norm(matvec(work, v))
        basis.append((v, lam))
        # 收缩：work -= lam * v v^T
        for i in range(3):
            for j in range(3):
                work[i][j] -= lam * v[i] * v[j]
    nrm = basis[2][0]
    e1 = basis[0][0]
    e2 = basis[1][0]

    rad = []
    out_of_plane = []
    for p in d:
        h = sum(p[i] * nrm[i] for i in range(3))
        u = sum(p[i] * e1[i] for i in range(3))
        v = sum(p[i] * e2[i] for i in range(3))
        rad.append(math.hypot(u, v))
        out_of_plane.append(h)
    mr = sum(rad) / sides
    var = sum((r - mr) ** 2 for r in rad) / sides
    rd = math.sqrt(var) / max(1e-12, mr)
    oop = max(abs(h) for h in out_of_plane)
    return {
        "roundness": rd,
        "mean_radius": mr,
        "residual": [r - mr for r in rad],
        "signed_residual": rad,
        "out_of_plane_max": oop,
        "centroid": c,
        "basis": (e1, e2, nrm),
    }


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        s1 = json.load(open(sys.argv[1] if len(sys.argv) > 1 else "tests/_cyl1.json"))
        sv = json.load(open(sys.argv[2] if len(sys.argv) > 2 else "tests/_cyl2.json"))

        bind = [tuple(p) for p in s1["bind"]]
        faces = s1["faces"]
        n = len(bind)
        HALF = s1["HALF"]
        bend = s1["bend"]
        weights = s1["weights"]
        py_delta = sv["delta"]
        py_guide = sv["guide"]

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
        mc.setAttr(elbow + ".rotateZ", bend)
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

        # 1) Base CoR
        mc.setAttr(skin + ".algorithmType", 0)
        inject([])
        base = read()

        # 2) 引导形状 = original + guide（guide 是局部归一空间的位移）
        # child 关节在 (HALF, 0, 0)，半径 = 所有顶点到轴的径向距离最大者
        child_pivot = (HALF, 0.0, 0.0)
        R = 0.0
        for p in bind:
            d = (p[0] - child_pivot[0], p[1] - child_pivot[1], p[2] - child_pivot[2])
            r = math.hypot(d[1], d[2])
            if r > R:
                R = r
        G = py_guide
        # guide 是局部归一空间；本夹具 basisY=+Y, basisZ=+Z，故只需乘 R
        guided = [[base[i][k] + G[3 * i + k] * R for k in range(3)]
                  for i in range(n)]

        # 3) Python delta 注入（alg5）
        mc.setAttr(skin + ".algorithmType", 5)
        inject([float(v) * R for v in py_delta])
        py_out = read()

        # 4) C++ 自动
        inject([])
        cpp_out = read()

        rows = [("1 Base CoR      ", base), ("2 引导 original+guide", guided),
                ("3 Python 优化CoR", py_out), ("4 C++ 自动 alg5 ", cpp_out)]

        print("HALF=%.1f SIDES=%d RINGS=%d bend=%.3f R=%.6f" % (HALF, SIDES, RINGS, bend, R))
        print("门控 0/0；3/4 都用 alg5")

        target_rows = [0, 1, 2, 3, 4, 5] + [RINGS - 1 - k for k in range(6)]
        target_rows = sorted(set(target_rows))

        print("\n=== 圆度 std/mean ===")
        print("%8s %18s %18s %18s %18s" % ("x", "1 BaseCoR", "2 引导",
                                           "3 Py优化CoR", "4 C++自动"))
        for row in range(RINGS):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            vals = []
            for _, P2 in rows:
                m = section_metrics(P2, row)
                vals.append(m["roundness"])
            mark = "  ← 已知 0.1305" if abs(abs(x0) - 0.667) < 0.05 else ""
            print("%8.2f %18.6f %18.6f %18.6f %18.6f%s"
                  % (x0, vals[0], vals[1], vals[2], vals[3], mark))

        print("\n=== 关键环详细（x≈±0.67）===")
        for row in (3, 4, RINGS - 1 - 3):
            x0 = -HALF + row * (2 * HALF / (RINGS - 1))
            print("\n  --- row=%d  x=%+.3f ---" % (row, x0))
            print("  %16s %12s %12s %14s %12s"
                  % ("对象", "圆度", "平均半径", "离面max", "最深方位"))
            for nm, P2 in rows:
                m = section_metrics(P2, row)
                res = m["residual"]
                # 最深点方位（用环内序号）
                k = res.index(min(res))
                ang = 360.0 * k / SIDES
                print("  %16s %12.6f %12.6f %14.6e %8.1f°"
                      % (nm, m["roundness"], m["mean_radius"],
                         m["out_of_plane_max"], ang))

        print("\n=== 环内带符号径向残差（row=%d, x≈-0.67）===" % 3)
        print("  %6s %10s %10s %10s %10s" % ("j", "1基CoR", "2引导", "3Py", "4C++"))
        for j in range(SIDES):
            line = "  %6d" % j
            for nm, P2 in rows:
                m = section_metrics(P2, 3)
                line += " %10.5f" % m["residual"][j]
            print(line)

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
