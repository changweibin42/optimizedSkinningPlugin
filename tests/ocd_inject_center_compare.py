# -*- coding: utf-8 -*-
r"""同一张网格，只换中心偏移来源，比较红圈那条纵线。

三步对照（全部由插件完成最终 CoR 求值）：
  A) 自动求解                       —— 当前有褶皱的结果
  B) 注入 C++ 自己导出的 delta       —— 应复现 A，验证单位/基底/注入路径
  C) 注入 Python 独立生成的 delta    —— 回答「换回 Python 中心后褶皱还在不在」

局部归一化偏移 -> 对象空间向量的换算（与插件 localOffsetToWorld 一致）：
  world = radius * (local.x*axis + local.y*basisY + local.z*basisZ)
本夹具 worldMat 为单位阵，故对象空间向量 == 世界向量。

褶皱度量（不用单点径向）：
  * 同一经线上相邻段的**有符号转角**（折线的最直接指标）
  * 中间几圈相对两侧弦线的**向内偏离**

    mayapy tests/ocd_inject_center_compare.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def mat_inverse(m):
    aug = [list(m[r * 4:r * 4 + 4]) + [1.0 if r == k else 0.0 for k in range(4)]
           for r in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(4):
            if row == col:
                continue
            factor = aug[row][col]
            if factor:
                aug[row] = [v - factor * w for v, w in zip(aug[row], aug[col])]
    return [aug[r][4 + c] for r in range(4) for c in range(4)]


def build(mc, om, bind, faces, bones, sparse, name):
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

    joints = []
    for index, bone in enumerate(bones):
        if index == 0:
            node = mc.createNode("joint", name=name + "_j0", parent=holder)
            mc.setAttr(node + ".translateX", bone["bindPosition"][0])
        else:
            node = mc.createNode("joint", name="%s_j%d" % (name, index),
                                 parent=joints[-1])
            mc.setAttr(node + ".translateX",
                       bone["bindPosition"][0]
                       - bones[index - 1]["bindPosition"][0])
        joints.append(node)

    skin = mc.deformer(shape, type="optimizedSkinCluster",
                       name=name + "_skin#")[0]
    for index, (node, bone) in enumerate(zip(joints, bones)):
        mc.connectAttr(node + ".worldMatrix[0]",
                       "%s.matrix[%d]" % (skin, index))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, index),
                   *[float(v) for v in bone["bindPreMatrix"]], type="matrix")
    for vertex, row in enumerate(sparse):
        for key, value in row.items():
            mc.setAttr("%s.weightList[%d].weights[%s]"
                       % (skin, vertex, key), float(value))
    mc.setAttr(skin + ".algorithmType", 5)
    mc.setAttr(joints[1] + ".rotateZ", 146.126)
    mc.refresh(force=True)
    mc.dgeval(shape)
    return holder, shape, skin


def read_points(om, shape):
    sel = om.MSelectionList()
    sel.add(shape)
    return [(p.x, p.y, p.z) for p in
            om.MFnMesh(sel.getDagPath(0)).getPoints(om.MSpace.kObject)]


def inject(mc, om, skin, values):
    sel = om.MSelectionList()
    sel.add(skin)
    node = om.MFnDependencyNode(sel.getDependNode(0))
    plug = node.findPlug("ocdCenterOffsets", False)
    plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray(values)))
    sel2 = om.MSelectionList()
    mc.refresh(force=True)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    import numpy as np
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(source))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        # ---------- 铰链基与归一化 ----------
        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = np.asarray(child["bindPosition"], float)
        axis = np.asarray(d["hinge"]["axis"], float)
        axis = axis / np.linalg.norm(axis)
        bY = np.array([0.0, 1.0, 0.0])
        bZ = np.array([0.0, 0.0, 1.0])
        B = np.asarray(bind, float)
        rel = B - pivot
        st = rel @ axis
        rad = rel - np.outer(st, axis)
        R = float(np.linalg.norm(rad, axis=1).max())
        bind_l = np.column_stack([st / R, (rad / R) @ bY, (rad / R) @ bZ])

        W = np.zeros((n, len(bones)))
        for i, row in enumerate(sparse):
            for k, v in row.items():
                W[i, int(k)] = float(v)
        used = [j for j in range(len(bones)) if W[:, j].max() > 0.0]
        W = W[:, used]

        # ---------- A) 自动 ----------
        holder_a, shape_a, skin_a = build(mc, om, bind, faces, bones, sparse, "ca")
        auto = read_points(om, shape_a)

        cpp_guide = list(mc.getAttr(skin_a + ".ocdGuideDump") or [])
        cpp_delta = list(mc.getAttr(skin_a + ".ocdDeltaDump") or [])
        limit = list(mc.getAttr(skin_a + ".ocdCenterLimitDump") or [])
        print("== 诊断属性 ==")
        print("  guide 长度=%d  delta 长度=%d  centerLimit=%s"
              % (len(cpp_guide), len(cpp_delta), limit))
        if len(cpp_delta) != 3 * n:
            # 强制再求值一次
            for _ in range(3):
                mc.dgeval(shape_a)
                mc.refresh(force=True)
                read_points(om, shape_a)
            cpp_delta = list(mc.getAttr(skin_a + ".ocdDeltaDump") or [])
            cpp_guide = list(mc.getAttr(skin_a + ".ocdGuideDump") or [])
            limit = list(mc.getAttr(skin_a + ".ocdCenterLimitDump") or [])
            print("  重取: guide=%d delta=%d limit=%s"
                  % (len(cpp_guide), len(cpp_delta), limit))

        delta_cpp = np.asarray(cpp_delta, float).reshape(n, 3)
        guide_cpp = np.asarray(cpp_guide, float).reshape(n, 3)

        # 局部归一 -> 对象空间（worldMat 为单位阵）
        def to_object(local):
            return (local[:, 0:1] * axis[None, :]
                    + local[:, 1:2] * bY[None, :]
                    + local[:, 2:3] * bZ[None, :]) * R

        obj_cpp = to_object(delta_cpp)

        # ---------- B) 注入 C++ 自己的 delta ----------
        holder_b, shape_b, skin_b = build(mc, om, bind, faces, bones, sparse, "cb")
        inject(mc, om, skin_b, obj_cpp.ravel().tolist())
        mc.dgeval(shape_b)
        mc.refresh(force=True)
        inj_cpp = read_points(om, shape_b)

        # ---------- C) 注入 Python 生成的 delta ----------
        cor_l = np.column_stack([
            ((np.asarray(d["points"]["cor"], float) - pivot) @ axis) / R,
            ((np.asarray(d["points"]["cor"], float) - pivot) @ bY) / R,
            ((np.asarray(d["points"]["cor"], float) - pivot) @ bZ) / R,
        ])
        # 相对旋转直接由矩阵反解，不猜 bend 符号
        child_pose = np.asarray(child["worldMatrix"], float).reshape(4, 4)
        child_bind = np.asarray(mat_inverse(child["bindPreMatrix"]),
                               float).reshape(4, 4)
        rel_rot = child_pose[:3, :3] @ child_bind[:3, :3].T
        trace = rel_rot[0, 0] + rel_rot[1, 1] + rel_rot[2, 2]
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (trace - 1) / 2))))
        # 转到四元数（(x,y,z,w)）
        if abs(1.0 + trace) < 1e-6:
            q = (0.0, 0.0, 0.0, 1.0)
        else:
            s = math.sqrt(max(0.0, trace + 1.0)) * 2.0
            q = ((rel_rot[2, 1] - rel_rot[1, 2]) / s,
                 (rel_rot[0, 2] - rel_rot[2, 0]) / s,
                 (rel_rot[1, 0] - rel_rot[0, 1]) / s,
                 0.25 * s)
        print("\n相对旋转: 角度=%.3f 度  四元数=%s" % (ang, np.round(q, 6)))

        edges = ref.local_edges(bind_l.tolist(), faces)
        rotations = [(0.0, 0.0, 0.0, 1.0), q]
        py_guide = np.asarray(ref.hinge_displacement(
            bind_l, cor_l, W, ang, 0.0, 0.0))
        outer = np.asarray(ref.outer_participation(bind_l, W, ang, 0.0, 0.0))
        py_delta = np.asarray(ref.structural_centers(
            edges, cor_l, W, rotations, py_guide, outer,
            float(limit[0]) if limit else 16.0))

        print("\n== guide / delta 差异（Python 自生成 vs C++）==")
        for name, a, b in (("guide", np.asarray(cpp_guide, float).reshape(n, 3),
                            py_guide),
                           ("delta", delta_cpp, py_delta)):
            diff = np.linalg.norm(a - b, axis=1)
            print("  %-6s |a|均=%.6f |b|均=%.6f  差 max=%.6f mean=%.6f"
                  % (name, np.linalg.norm(a, axis=1).mean(),
                     np.linalg.norm(b, axis=1).mean(), diff.max(), diff.mean()))
            worst = int(np.argmax(diff))
            print("         差最大顶点 v%d  station=%.3f"
                  % (worst, bind_l[worst, 0]))

        holder_c, shape_c, skin_c = build(mc, om, bind, faces, bones, sparse, "cc")
        obj_py = to_object(py_delta)
        inject(mc, om, skin_c, obj_py.ravel().tolist())
        mc.dgeval(shape_c)
        mc.refresh(force=True)
        inj_py = read_points(om, shape_c)

        # ---------- 三条线比较 ----------
        def dist(a, b):
            return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

        print("\n== A/B/C 顶点差 ==")
        for label, pts in (("B(注入C++delta) vs A(自动)", inj_cpp),
                           ("C(注入Py delta) vs A(自动)", inj_py),
                           ("C vs B", inj_py)):
            ref_pts = auto if label != "C vs B" else inj_cpp
            dmax, wid = 0.0, -1
            for i in range(n):
                dd = dist(pts[i], ref_pts[i])
                if dd > dmax:
                    dmax, wid = dd, i
            print("  %-28s max=%.6f  v%d  station=%.3f"
                  % (label, dmax, wid, bind_l[wid, 0]))

        # ---------- 经线轮廓与转角 ----------
        # 选几条经线：按 yz 平面角度分组
        ang_all = np.degrees(np.arctan2(bind_l[:, 2], bind_l[:, 1]))
        print("\n== 经线转角（折痕指标，越大越折）==")
        print("%8s %8s %10s %10s %10s"
              % ("angle", "n", "A最大转角", "B", "C"))
        best = None
        for target in (-90, -45, 0, 45, 90, 135, 180, -135):
            sel = np.abs(((ang_all - target + 180) % 360) - 180) < 4.0
            if sel.sum() < 20:
                continue
            idx = np.where(sel)[0]
            idx = idx[np.argsort(bind_l[idx, 0])]

            def turning(pts):
                prof = np.array([[bind_l[i, 0],
                                  math.hypot(pts[i][1], pts[i][2])] for i in idx])
                worst = 0.0
                for k in range(1, len(prof) - 1):
                    v1 = prof[k] - prof[k - 1]
                    v2 = prof[k + 1] - prof[k]
                    n1 = np.linalg.norm(v1)
                    n2 = np.linalg.norm(v2)
                    if n1 < 1e-12 or n2 < 1e-12:
                        continue
                    cosang = float(v1 @ v2 / (n1 * n2))
                    cosang = max(-1.0, min(1.0, cosang))
                    t = math.degrees(math.acos(cosang))
                    if t > worst:
                        worst = t
                return worst

            ta, tb, tc = turning(auto), turning(inj_cpp), turning(inj_py)
            print("%8d %8d %10.2f %10.2f %10.2f" % (target, len(idx), ta, tb, tc))
            if best is None or ta > best[1]:
                best = (target, ta, idx)

        if best:
            target, ta, idx = best
            print("\n折角最大的经线 = %d 度（%.2f 度转角），逐点轮廓：" % (target, ta))
            print("%9s %10s %10s %10s %10s"
                  % ("station", "A径向", "B径向", "C径向", "A转角"))
            prof_a = [(bind_l[i, 0], math.hypot(auto[i][1], auto[i][2]))
                      for i in idx]
            prof_b = [(bind_l[i, 0], math.hypot(inj_cpp[i][1], inj_cpp[i][2]))
                      for i in idx]
            prof_c = [(bind_l[i, 0], math.hypot(inj_py[i][1], inj_py[i][2]))
                      for i in idx]
            for k in range(1, len(prof_a) - 1):
                v1 = np.array(prof_a[k]) - np.array(prof_a[k - 1])
                v2 = np.array(prof_a[k + 1]) - np.array(prof_a[k])
                n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
                t = 0.0
                if n1 > 1e-12 and n2 > 1e-12:
                    t = math.degrees(math.acos(max(-1.0, min(1.0,
                        float(v1 @ v2 / (n1 * n2))))))
                if abs(prof_a[k][0]) <= 2.0:
                    print("%9.3f %10.4f %10.4f %10.4f %10.2f"
                          % (prof_a[k][0], prof_a[k][1], prof_b[k][1],
                             prof_c[k][1], t))

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
