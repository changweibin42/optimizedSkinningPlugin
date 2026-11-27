# -*- coding: utf-8 -*-
r"""用导出的 JSON 重建同一场景并重放，用于确认复现与参数扫描。

只用纯 Python（mayapy 没有 numpy）。

    mayapy tests/replay_exported_case.py D:/tmp/ocd_case3.json [--sweep]

先验证重建后的三个算法输出与 JSON 里的一致；一致就说明可以在这份夹具上
安全地做实验，比如扫 centerLimit 看红圈是否移动。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def mat_inverse(m):
    """4x4 行主序纯 Python 求逆。"""
    aug = [[float(m[r * 4 + c]) for c in range(4)]
           + [1.0 if r == k else 0.0 for k in range(4)] for r in range(4)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("矩阵奇异")
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


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def build_scene(mc, om, payload):
    bind = payload["mesh"]["bindPoints"]
    faces = payload["mesh"]["faces"]
    bones = payload["bones"]
    sparse = payload["weightsSparse"]

    holder = mc.createNode("transform", name="rp_holder")
    geo = mc.createNode("transform", name="rp_geo", parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                             fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")

    # DAG 链：joint1 -> L_Ankle_jnt -> joint3，按导出里的 bind 位置摆放。
    joints = []
    for index, bone in enumerate(bones):
        if index == 0:
            node = mc.createNode("joint", name=bone["short"], parent=holder)
            mc.setAttr(node + ".translateX", bone["bindPosition"][0])
        else:
            node = mc.createNode("joint", name=bone["short"], parent=joints[-1])
            local = (bone["bindPosition"][0]
                     - bones[index - 1]["bindPosition"][0])
            mc.setAttr(node + ".translateX", local)
        mc.setAttr(node + ".radius", 0.09)
        joints.append(node)

    skin = mc.deformer(shape, type="optimizedSkinCluster", name="rp_skin#")[0]
    for index, (node, bone) in enumerate(zip(joints, bones)):
        mc.connectAttr(node + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, index))
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, index),
                   *[float(v) for v in bone["bindPreMatrix"]], type="matrix")

    for vertex, row in enumerate(sparse):
        for key, value in row.items():
            mc.setAttr("%s.weightList[%d].weights[%s]"
                       % (skin, vertex, key), float(value))

    mc.setAttr(skin + ".algorithmType", 5)
    mc.setAttr(skin + ".corDebugLog", True)
    return holder, geo, shape, skin, joints


def apply_pose(mc, bones, joints):
    """按导出里的 worldMatrix 设置旋转，返回实际使用的角度。"""
    used = []
    for index, (node, bone, joint) in enumerate(zip(bones, bones, joints)):
        target = bone["worldMatrix"]
        # 只处理绕 Z 的情形；其余骨骼按父级继承。
        c, s = float(target[0]), float(target[1])
        if abs(c - 1.0) < 1e-6 and abs(s) < 1e-6:
            used.append(0.0)
            continue
        angle = math.degrees(math.atan2(s, c))
        for candidate in (angle, -angle):
            mc.setAttr(joint + ".rotateZ", candidate)
            mc.refresh(force=True)
            wm = mc.getAttr(joint + ".worldMatrix[0]")
            if abs(float(wm[0]) - c) < 1e-4 and abs(float(wm[1]) - s) < 1e-4:
                used.append(candidate)
                break
        else:
            used.append(angle)
    mc.refresh(force=True)
    return used


def read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [(float(p.x), float(p.y), float(p.z))
            for p in mesh.getPoints(om.MSpace.kObject)]


def evaluate(mc, om, shape, skin, algorithm):
    mc.setAttr(skin + ".algorithmType", algorithm)
    mc.dgeval(shape)
    mc.refresh(force=True)
    return read_points(om, shape)


def ring_stats(bind, points, axis, origin, weight1, width, count):
    """返回每环 (station, mean|disp|, mean 径向变化, 混权数)。"""
    buckets = {}
    for vertex, point in enumerate(bind):
        station = sum((point[i] - origin[i]) * axis[i] for i in range(3))
        key = int(round(station / width))
        buckets.setdefault(key, []).append(vertex)

    radial_bind = {}
    rows = []
    for key in sorted(buckets):
        members = buckets[key]
        disp = [dist(points[v], bind[v]) for v in members]
        radial = []
        for vertex in members:
            rb = math.hypot(bind[vertex][1], bind[vertex][2])
            rp = math.hypot(points[vertex][1], points[vertex][2])
            radial.append(rp - rb)
        rows.append({
            "station": sum(sum((bind[v][i] - origin[i]) * axis[i]
                               for i in range(3)) for v in members) / len(members),
            "disp": sum(disp) / len(disp),
            "radial": sum(radial) / len(radial),
            "mixed": sum(1 for v in members if weight1[v] > 0.0),
            "count": len(members),
        })
    return rows


def fold_score(rows, low, high):
    """过渡区的最大相邻增量（每单位 station），越大越像折点。"""
    best = 0.0
    best_station = 0.0
    for a, b in zip(rows, rows[1:]):
        if not (low <= b["station"] <= high):
            continue
        span = b["station"] - a["station"]
        if span <= 1e-9:
            continue
        slope = (b["disp"] - a["disp"]) / span
        if slope > best:
            best = slope
            best_station = b["station"]
    return best, best_station


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        payload = json.load(open(source))
        bind = payload["mesh"]["bindPoints"]
        bones = payload["bones"]
        sparse = payload["weightsSparse"]
        axis = payload["hinge"]["axis"]
        origin = payload["hinge"]["origin"]
        weight1 = [float(row.get("1", 0.0)) for row in sparse]

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder, geo, shape, skin, joints = build_scene(mc, om, payload)
        angles = apply_pose(mc, bones, joints)
        print("设置的关节角度(deg):", [round(a, 2) for a in angles])

        # ---- 1. 复现验证 ----
        print("\n== 复现验证（与 JSON 对比）==")
        reproduced = {}
        for algorithm, key in ((0, "cor"), (3, "ocd"), (5, "final")):
            got = evaluate(mc, om, shape, skin, algorithm)
            reproduced[key] = got
            want = payload["points"][key]
            worst = max(dist(g, w) for g, w in zip(got, want))
            rel = max(dist(g, w) / max(1e-9, dist(w, bind[i]))
                      for i, (g, w) in enumerate(zip(got, want)))
            print("  algo%d: max 偏差 %.6e  (最大相对偏差 %.2e)"
                  % (algorithm, worst, rel))

        # ---- 2. 环表 ----
        stations = sorted({round(sum((p[i] - origin[i]) * axis[i]
                                     for i in range(3)), 5) for p in bind})
        gaps = sorted(b - a for a, b in zip(stations, stations[1:]) if b - a > 1e-6)
        width = gaps[len(gaps) // 2]
        print("\n环宽 %.5f  环数约 %d" % (width, len(stations)))

        print("\n%9s %9s %9s %9s %9s" % ("station", "|cor|", "|ocd|", "|final|",
                                          "径向fin"))
        rows_cor = ring_stats(bind, reproduced["cor"], axis, origin, weight1,
                              width, len(bind))
        rows_ocd = ring_stats(bind, reproduced["ocd"], axis, origin, weight1,
                              width, len(bind))
        rows_fin = ring_stats(bind, reproduced["final"], axis, origin, weight1,
                              width, len(bind))
        for a, b, c in zip(rows_cor, rows_ocd, rows_fin):
            if not (7.0 <= c["station"] <= 14.0):
                continue
            if int(round(c["station"] / width)) % 3:
                continue
            print("%9.3f %9.4f %9.4f %9.4f %9.4f"
                  % (c["station"], a["disp"], b["disp"], c["disp"],
                     c["radial"]))

        score, where = fold_score(rows_fin, 8.0, 11.0)
        print("\n折点强度(过渡区最大斜率) = %.4f 出现在 station %.3f"
              % (score, where))

        # ---- 3. centerLimit 扫描 ----
        if "--sweep" in sys.argv:
            print("\n== centerLimit 扫描（看折点是否随限额移动）==")
            print("%10s %10s %12s %12s" % ("limit", "折点强度", "折点station",
                                            "最凹 station"))
            for limit in (2.0, 4.0, 8.0, 16.0, 32.0, 64.0):
                mc.setAttr(skin + ".ocdCenterLimit", limit)
                points = evaluate(mc, om, shape, skin, 5)
                rows = ring_stats(bind, points, axis, origin, weight1, width,
                                  len(bind))
                score, where = fold_score(rows, 8.0, 11.0)
                deepest = min(rows, key=lambda r: r["radial"])
                print("%10.1f %10.4f %12.3f %8.3f(%.4f)"
                      % (limit, score, where, deepest["station"],
                         deepest["radial"]))

        print("\n完成。")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
