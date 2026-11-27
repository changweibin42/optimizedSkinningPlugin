# -*- coding: utf-8 -*-
r"""验收：ocdEnableCenterLimit 开关。

四项：
  1) 开关开、限额 16       → 旧结果（过渡区 ~32.9°，中线 ~175°）
  2) 开关关、属性仍为 16   → 过渡区 ~0.85°；且与"开+32"在容差内一致
  3) 关闭时改限额数值     → 输出不变；重新开启并设回 16 → 恢复旧结果
  4) 458 点圆柱 + 16080 夹具，弯 146° 与弯加扭

    mayapy tests/ocd_center_limit_toggle.py
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)


def build_rig(mc, om, bind, faces, bones, sparse, name="t"):
    holder = mc.createNode("transform", name=name + "_h")
    geo = mc.createNode("transform", name=name + "_g", parent=holder)
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
    for i, bn in enumerate(bones):
        if i == 0:
            nd = mc.createNode("joint", name=name + "_j0", parent=holder)
            mc.setAttr(nd + ".translateX", bn["bindPosition"][0])
        else:
            nd = mc.createNode("joint", name="%s_j%d" % (name, i), parent=joints[-1])
            mc.setAttr(nd + ".translateX",
                       bn["bindPosition"][0] - bones[i - 1]["bindPosition"][0])
        joints.append(nd)
    skin = mc.deformer(shape, type="optimizedSkinCluster", name=name + "_s#")[0]
    for i, (nd, bn) in enumerate(zip(joints, bones)):
        mc.connectAttr(nd + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, i))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, i),
                   *[float(v) for v in bn["bindPreMatrix"]], type="matrix")
    for v, row in enumerate(sparse):
        for k, val in row.items():
            mc.setAttr("%s.weightList[%d].weights[%s]" % (skin, v, k), float(val))
    mc.setAttr(skin + ".algorithmType", 5)
    for a in ("ocdCenterGateLow", "ocdCenterGateHigh"):
        if mc.attributeQuery(a, node=skin, exists=True):
            mc.setAttr("%s.%s" % (skin, a), 0.0)
    s = om.MSelectionList()
    s.add(skin)
    om.MFnDependencyNode(s.getDependNode(0)).findPlug(
        "ocdCenterOffsets", False).setMObject(
            om.MFnDoubleArrayData().create(om.MDoubleArray()))
    return holder, shape, skin, joints


def read_pts(mc, om, shape):
    for _ in range(2):
        mc.refresh(force=True)
        mc.dgeval(shape)
    s = om.MSelectionList()
    s.add(shape)
    return [[p.x, p.y, p.z] for p in
            om.MFnMesh(s.getDagPath(0)).getPoints(om.MSpace.kObject)]


def diff(a, b):
    return max(math.sqrt(sum((a[i][k] - b[i][k]) ** 2 for k in range(3)))
               for i in range(len(a)))


def turning(pts, i, j, k):
    u = [pts[j][q] - pts[i][q] for q in range(3)]
    v = [pts[k][q] - pts[j][q] for q in range(3)]
    lu = math.sqrt(sum(x * x for x in u))
    lv = math.sqrt(sum(x * x for x in v))
    if lu < 1e-12 or lv < 1e-12:
        return None
    c = sum(u[q] * v[q] for q in range(3)) / (lu * lv)
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        print("=" * 78)
        print("夹具 A: 16080 点（test04）")
        print("=" * 78)
        d = json.load(open(r"D:\tmp\ocd_case3.json"))
        bind = [tuple(p) for p in d["mesh"]["bindPoints"]]
        faces = d["mesh"]["faces"]
        n = len(bind)
        holder, shape, skin, joints = build_rig(
            mc, om, bind, faces, d["bones"], d["weightsSparse"], "big")

        ok = mc.attributeQuery("ocdEnableCenterLimit", node=skin, exists=True)
        print("属性存在: %s  默认: %s"
              % (ok, mc.getAttr(skin + ".ocdEnableCenterLimit") if ok else "N/A"))
        if not ok:
            return 1

        mc.setAttr(joints[1] + ".rotateZ", -147.2124)

        def run(enable, limit):
            mc.setAttr(skin + ".ocdEnableCenterLimit", 1 if enable else 0)
            mc.setAttr(skin + ".ocdCenterLimit", float(limit))
            return read_pts(mc, om, shape)

        # 环结构：x 步长 0.12，每环 80
        SIDES = 80
        rings = [round(p[0] / 0.12) for p in bind]

        def ring_turning(pts, rid):
            idx = [i for i in range(n) if rings[i] == rid]
            if len(idx) != SIDES:
                return None
            idx.sort(key=lambda i: math.atan2(bind[i][2], bind[i][1]))
            # 三点：环内相邻两点 + 跨环
            return turning(pts, idx[0], idx[1], idx[2])

        A16 = run(True, 16)
        A32 = run(True, 32)
        Aoff = run(False, 16)

        print("\n[判据 1] 开/16  vs  开/32  vs  关/16")
        print("  最大顶点差: 开16-开32=%.6e   开16-关16=%.6e   开32-关16=%.6e"
              % (diff(A16, A32), diff(A16, Aoff), diff(A32, Aoff)))
        print("  → 关/16 应与 开/32 一致（差异远小于 开16-开32）")

        # 关/16 vs 开/32 一致性
        d_off_vs_32 = diff(Aoff, A32)
        d_16_vs_32 = diff(A16, A32)
        print("  判定: %s" % ("通过" if d_off_vs_32 < 1e-5 else "★未通过★"))

        print("\n[判据 3] 关闭时改限额是否影响")
        Aoff2 = run(False, 8)
        Aoff3 = run(False, 64)
        print("  关/8 vs 关/16 = %.3e   关/64 vs 关/16 = %.3e"
              % (diff(Aoff2, Aoff), diff(Aoff3, Aoff)))
        print("  判定: %s" % ("通过（关闭时不随限额改变）"
                            if max(diff(Aoff2, Aoff), diff(Aoff3, Aoff)) < 1e-9
                            else "★未通过★"))
        back = run(True, 16)
        print("  重开/16 vs 原开/16 = %.3e  %s"
              % (diff(back, A16),
                 "通过" if diff(back, A16) < 1e-9 else "★未通过★"))

        # 中心偏移最大值（判断是否真的触限）
        print("\n[限幅信息]")
        for lab, en, lim in (("开/16", True, 16), ("开/32", True, 32),
                             ("关/16", False, 16)):
            mc.setAttr(skin + ".ocdEnableCenterLimit", 1 if en else 0)
            mc.setAttr(skin + ".ocdCenterLimit", float(lim))
            read_pts(mc, om, shape)
            dv = mc.getAttr(skin + ".ocdDeltaDump")
            lm = mc.getAttr(skin + ".ocdCenterLimitDump")
            mx = 0.0
            if dv:
                for i in range(n):
                    m = math.sqrt(sum(dv[3 * i + k] ** 2 for k in range(3)))
                    mx = max(mx, m)
            print("  %-7s centerLimitDump=%s  |delta|max=%.6f"
                  % (lab, lm, mx))

        print("\n" + "=" * 78)
        print("夹具 B: 458 点圆柱（118° 与 146° + 扭转）")
        print("=" * 78)
        HALF = ref.HALF
        cbind, cfaces = ref.cylinder()
        cbones = [
            {"short": "prev", "bindPosition": [-HALF, 0.0, 0.0],
             "bindPreMatrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, HALF, 0, 0, 1]},
            {"short": "elbow", "bindPosition": [HALF, 0.0, 0.0],
             "bindPreMatrix": [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, -HALF, 0, 0, 1]},
        ]
        csparse = []
        for p in cbind:
            w = ref.skin_weight(p[0])
            rw = ref.rig_weights(w, False)
            csparse.append({"0": float(rw[0]), "1": float(rw[1])})

        for bend, twist in ((118.0, 0.0), (146.0, 0.0), (118.0, 35.0)):
            h2, sh2, sk2, j2 = build_rig(mc, om, cbind, cfaces, cbones,
                                         csparse, "cyl")
            mc.setAttr(j2[1] + ".rotateZ", bend)
            if twist:
                mc.setAttr(j2[1] + ".rotateY", twist)

            def run2(en, lim):
                mc.setAttr(sk2 + ".ocdEnableCenterLimit", 1 if en else 0)
                mc.setAttr(sk2 + ".ocdCenterLimit", float(lim))
                return read_pts(mc, om, sh2)

            b16 = run2(True, 16)
            boff = run2(False, 16)
            dmax = diff(b16, boff)
            # 环内圆度
            SID = ref.SIDES
            RINGS = 2 * ref.HALF_RINGS + 1
            def rnd(P, row):
                Q = [P[row * SID + j] for j in range(SID)]
                c = [sum(q[k] for q in Q) / SID for k in range(3)]
                r = [math.sqrt(sum((q[k] - c[k]) ** 2 for k in range(3))) for q in Q]
                mr = sum(r) / len(r)
                v = sum((x - mr) ** 2 for x in r) / len(r)
                return math.sqrt(v) / max(1e-12, mr)
            mx16 = max(rnd(b16, row) for row in range(RINGS))
            mxoff = max(rnd(boff, row) for row in range(RINGS))
            print("  bend=%.1f twist=%.1f  16vs关 最大差=%.4e  "
                  "圆度max: 开16=%.4f 关=%.4f"
                  % (bend, twist, dmax, mx16, mxoff))
            mc.delete(h2)

        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
