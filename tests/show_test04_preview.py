# -*- coding: utf-8 -*-
r"""把参考预览 show() 搬到 test04 场景上。

参考实现的 Elbow / hinge_section 全部按"半径 = 1、关节在原点"建模，
所以这里把 test04 的真实网格按 1/R 缩放后再进参考数学，造型比例不变。

用法（Maya 脚本编辑器，先打开 test04.ma）：
    import runpy
    g = runpy.run_path(r"D:\optimizedSkinningPlugin\cor-local-structure-d78d264\scripts\ocd_elbow_reference.py")
    ns = g
    exec(open(r"<本文件路径>", encoding="utf-8").read(), ns)

或直接用独立模式（从夹具读数据，不需要打开场景）：
    mayapy tests/show_test04_preview.py --standalone

面板（与 show() 一致的六格，另加第七格）：
  1 Classic_LBS   原生 skinCluster
  2 DQ_Skin       原生 skinCluster（DQS）
  3 CoR           alg 0
  4 DQS_Com       alg 2
  5 CurveGuided   alg 8 + 你的真实权重 + Python 求得的 delta
  6 OCD           Elbow.mesh() 直接写点
  7 OCD+CoR       alg 5（插件内部求解，用真实权重）

    python tests/show_test04_preview.py --check   # 只打印数值，不建窗口
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
FIXTURE = r"D:\tmp\ocd_case3.json"
REF_SCRIPT = r"D:\optimizedSkinningPlugin\cor-local-structure-d78d264\scripts\ocd_elbow_reference.py"


def load_ref():
    import runpy
    if not os.path.exists(REF_SCRIPT):
        raise RuntimeError("找不到参考脚本: " + REF_SCRIPT)
    return runpy.run_path(REF_SCRIPT)


def prepare():
    """读 test04 夹具，缩放到半径 1，返回 (bind_unit, faces, weights2, R, bend)。"""
    d = json.load(open(FIXTURE))
    bind = [[float(v) for v in p] for p in d["mesh"]["bindPoints"]]
    faces = [[int(v) for v in f] for f in d["mesh"]["faces"]]

    child = [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]
    pivot = [float(v) for v in child["bindPosition"]]
    axis = [float(v) for v in d["hinge"]["axis"]]
    na = math.sqrt(sum(v * v for v in axis))
    axis = [v / na for v in axis]
    bY = [0.0, 1.0, 0.0]
    bZ = [axis[1] * bY[2] - axis[2] * bY[1],
          axis[2] * bY[0] - axis[0] * bY[2],
          axis[0] * bY[1] - axis[1] * bY[0]]

    # 半径 = 到轴的最大径向距离
    R = 0.0
    for p in bind:
        rel = [p[i] - pivot[i] for i in range(3)]
        st = sum(rel[i] * axis[i] for i in range(3))
        rad = [rel[i] - axis[i] * st for i in range(3)]
        R = max(R, math.sqrt(sum(v * v for v in rad)))
    if R <= 0:
        raise RuntimeError("半径为零")

    # 归一化到半径 1；坐标轴保持 (axis, bY, bZ)
    bind_u = []
    for p in bind:
        rel = [p[i] - pivot[i] for i in range(3)]
        st = sum(rel[i] * axis[i] for i in range(3))
        rad = [rel[i] - axis[i] * st for i in range(3)]
        bind_u.append([st / R, sum(rad[i] * bY[i] for i in range(3)) / R,
                       sum(rad[i] * bZ[i] for i in range(3)) / R])

    weights = []
    for row in d["weightsSparse"]:
        weights.append([float(row.get("0", 0.0)), float(row.get("1", 0.0))])

    wm = [float(v) for v in child["worldMatrix"]]
    bend = math.degrees(math.atan2(wm[4], wm[0]))
    return bind_u, faces, weights, R, bend, axis, bY, bZ


def build_and_report(ref, bind_u, faces, weights, R, bend, do_window, panel_mode):
    import maya.cmds as mc
    from maya.api import OpenMaya as om
    import numpy as np

    n = len(bind_u)
    P = np.asarray(bind_u, float)
    SIDES = 80
    HALF = float(np.abs(P[:, 0]).max())      # 归一化后的半长
    weights3 = np.column_stack([np.asarray(weights, float),
                                np.zeros(n)])
    W = weights3
    print("归一化：R=%.6f  n=%d  半长=%.3f  弯角=%.2f°"
          % (R, n, HALF, bend))

    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

    group = mc.createNode("transform", name="test04_preview#")
    counts = [len(f) for f in faces]
    indices = [j for f in faces for j in f]
    pts = [om.MPoint(*p) for p in bind_u]

    layout = (("Classic_LBS", -1, 2), ("DQ_Skin", 1, 2),
              ("CoR", -1, 1), ("DQS_Com", 1, 1),
              ("CoR_CurveGuided", -1, 0), ("OCD", 1, 0),
              ("OCD_CoR", 0, -1))
    rigs, result_shapes = {}, {}
    for label, col, row in layout:
        holder = mc.createNode("transform", name="ref_%s#" % label, parent=group)
        geo = mc.createNode("transform", name="%s_geo#" % label, parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create(pts, counts, indices, parent=sel.getDependNode(0))
        shp = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                               fullPath=True)[0]
        mc.sets(geo, edit=True, forceElement="initialShadingGroup")

        prev = mc.createNode("joint", name="%s_prev#" % label, parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="%s_elbow#" % label, parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        tip = mc.createNode("joint", name="%s_tip#" % label, parent=elbow)
        mc.setAttr(tip + ".translateX", HALF)
        helper = mc.createNode("joint", name="%s_helper#" % label, parent=prev)
        mc.setAttr(helper + ".translateX", HALF)   # 与 elbow 共点

        if label in ("Classic_LBS", "DQ_Skin"):
            skin = mc.skinCluster(prev, elbow, geo, toSelectedBones=True,
                                  skinMethod=0 if label == "Classic_LBS" else 1,
                                  maximumInfluences=2, normalizeWeights=1,
                                  name="%s_skin#" % label)[0]
            for i in range(n):
                mc.skinPercent(skin, "%s.vtx[%d]" % (geo, i),
                               transformValue=[(prev, float(weights[i][0])),
                                               (elbow, float(weights[i][1]))])
        elif label == "OCD":
            # 纯 Mesh，由 Elbow.mesh 直接写点，不挂 deformer
            skin = None
        else:
            skin = mc.deformer(geo, type="optimizedSkinCluster",
                               name="%s_skin#" % label)[0]
            for k, jnt in enumerate((prev, elbow, helper)):
                mc.connectAttr(jnt + ".worldMatrix[0]",
                               "%s.matrix[%d]" % (skin, k))
                inv = mc.getAttr(jnt + ".worldInverseMatrix[0]")
                mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inv, type="matrix")
            for i in range(n):
                for k in range(3):
                    mc.setAttr("%s.weightList[%d].weights[%d]" % (skin, i, k),
                               float(W[i, k]))
            algo = {"CoR": 0, "DQS_Com": 2, "CoR_CurveGuided": 8,
                    "OCD_CoR": 5}[label]
            mc.setAttr(skin + ".algorithmType", algo)
            if label == "DQS_Com":
                mc.setAttr(skin + ".dqsCompStrength", 1.0)
            if label == "OCD_CoR":
                mc.setAttr(skin + ".ocdEnableCenterLimit", False)
                mc.setAttr(skin + ".ocdElbowGuide", False)
            for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
                if mc.attributeQuery(a, node=skin, exists=True):
                    mc.setAttr("%s.%s" % (skin, a), 0.0)
        rigs[label] = (holder, geo, shp, skin, prev, elbow, helper)
        result_shapes[label] = shp

    # 姿态：elbow 绕 Z 转 bend，helper 转一半
    q = ref["joint_quaternion"](bend, 0.0, 0.0)
    for label, (holder, geo, shp, skin, prev, elbow, helper) in rigs.items():
        s1 = om.MSelectionList()
        s1.add(elbow)
        om.MFnTransform(s1.getDagPath(0)).setRotation(
            om.MQuaternion(*q), om.MSpace.kTransform)
        hq = ref["joint_quaternion"](bend * 0.5, 0.0, 0.0)
        s2 = om.MSelectionList()
        s2.add(helper)
        om.MFnTransform(s2.getDagPath(0)).setRotation(
            om.MQuaternion(*hq), om.MSpace.kTransform)

    def read_shp(shp):
        for _ in range(2):
            mc.refresh(force=True)
            mc.dgeval(shp)
        s = om.MSelectionList()
        s.add(shp)
        return np.asarray([[p.x, p.y, p.z] for p in
                           om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)])

    # --- CurveGuided：先取零偏移输出当 original，再算 delta 注入 ---
    label = "CoR_CurveGuided"
    holder, geo, shp, skin, prev, elbow, helper = rigs[label]
    plug = om.MFnDependencyNode(
        om.MSelectionList().add(skin).getDependNode(0)).findPlug("corFitOffsets", False)
    plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray()))
    original = read_shp(shp)

    rots = [ref["joint_quaternion"](0, 0, 0), q,
            ref["joint_quaternion"](bend * 0.5, 0.0, 0.0)]
    edges = ref["local_edges"](bind_u, faces)
    guide = np.asarray(ref["hinge_displacement"](P, original, W[:2] if False else W,
                                                 bend, 0.0, 0.0))
    outer = np.asarray(ref["outer_participation"](P, W, bend, 0.0, 0.0))
    delta = np.asarray(ref["structural_centers"](edges, original, W, rots,
                                                 guide, outer, 16.0))
    plug.setMObject(om.MFnDoubleArrayData().create(
        om.MDoubleArray(delta.ravel().tolist())))
    mc.refresh(force=True)
    print("CurveGuided delta: |d|均=%.4f max=%.4f" %
          (np.linalg.norm(delta, axis=1).mean(),
           np.linalg.norm(delta, axis=1).max()))

    # --- OCD 格：Elbow.mesh 直接写点 ---
    label = "OCD"
    holder, geo, shp, skin, prev, elbow, helper = rigs[label]
    pose = ref["Elbow"](bend, 0.0, 0.0, ref["ZERO"], False)
    elbow_pts = np.asarray(pose.mesh(bind_u), float)
    s = om.MSelectionList()
    s.add(shp)
    om.MFnMesh(s.getDagPath(0)).setPoints(
        [om.MPoint(*p) for p in elbow_pts], om.MSpace.kObject)

    # --- 汇总数值 ---
    results = {}
    for label in layout:
        lab = label[0]
        if lab == "OCD":
            results[lab] = elbow_pts
        else:
            results[lab] = read_shp(rigs[lab][2])

    rings = np.round(P[:, 0] * (SIDES / 2.0) / 1.0).astype(int)  # 占位，下面按唯一值重算
    xs = np.unique(np.round(P[:, 0], 4))
    order = np.argsort(P[:, 0])
    # 用 80 点/环的假设不成立，改按 x 值分组
    groups = {}
    for i, v in enumerate(np.round(P[:, 0], 4)):
        groups.setdefault(v, []).append(i)

    def ring_stats(arr, idx):
        Q = arr[idx]
        c = Q.mean(axis=0)
        r = np.linalg.norm(Q - c, axis=1)
        return float(r.mean()), float(r.std() / max(r.mean(), 1e-12))

    names = [l[0] for l in layout]
    print("\n%9s %5s" % ("x/R", "n") +
          "".join(" %14s" % nm for nm in names))
    print("%9s %5s" % ("", "") + "".join(" %7s %6s" % ("半径", "圆度")
                                          for _ in names))
    picked = [v for v in sorted(groups) if abs(v) <= 3.0 and len(groups[v]) >= 20]
    for v in picked[::2]:
        idx = groups[v]
        line = "%9.3f %5d" % (v, len(idx))
        for nm in names:
            r, rd = ring_stats(results[nm], idx)
            line += " %7.4f %6.4f" % (r, rd)
        print(line)

    if do_window:
        try:
            win = mc.window("test04ElbowPreview", title="test04 × 参考预览",
                            widthHeight=(1300, 760))
            pane = mc.paneLayout(parent=win, configuration="vertical2",
                                 paneSize=[(1, 70, 100), (2, 30, 100)])
            cam_xf, cam = mc.camera(name="test04PreviewCam#", orthographic=True)
            cam_xf = mc.parent(cam_xf, group)[0]
            panel = mc.modelPanel(parent=pane, label="test04", camera=cam_xf,
                                  menuBarVisible=False)
            ed = mc.modelPanel(panel, query=True, modelEditor=True)
            mc.modelEditor(ed, e=True, displayAppearance="smoothShaded",
                           grid=False, displayTextures=False,
                           wireframeOnShaded=True, displayLights="default")
            mc.select(clear=True)
            mc.showWindow(win)
            print("\n预览窗口已创建；面板标签:", names)
        except Exception as exc:
            print("\n窗口创建失败(仅数值模式):", exc)
    return results, names


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        ref = load_ref()
        bind_u, faces, weights, R, bend, axis, bY, bZ = prepare()
        build_and_report(ref, bind_u, faces, weights, R, bend,
                         do_window=False, panel_mode=None)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
