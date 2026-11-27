# -*- coding: utf-8 -*-
r"""在真实 test04.ma 上确认：插件单独跑同样的输入，alg5 复现参考预览的
CoR_CurveGuided（1.2508R @ x/R=-1.219），不是 1.03R。

两种接法都会测，因为这决定了「1.03R」是从哪来的：

  A. 插件建在 pCylinderTargetShape（rest 网格，上游，没有别的 deformer）
     —— 用户实际用法。alg5 应与预览逐环一致。
  B. 插件建在 pCylinderShape2（原生 skinCluster 下游，输入已是姿势）
     —— mc.deformer() 的默认插入位置。此时「rest 网格」已经是摆好姿势的
     形状，中心求解在错误空间里做，输出被再变形一次，外扩量掉到 1.03R 附近。
     NOTES §5 记的「同样的输入只有 1.03R」是这个接法。

两个 deformer 都用夹具里的两骨权重（与场景 skinCluster1 一致，差 0.000e+00）。

用法（Maya 2024）：
    set "MAYA_PLUG_IN_PATH=D:\...\optimizedSkinningPlugin\build\Release"
    set "PYTHONPATH=D:\...\optimizedSkinningPlugin\.pylibs"
    mayapy tests\real_target_curveguided.py

退出码 0 = A 复现预览（最大差 < 0.01）且 B 明显偏离。
"""
import json
import math
import sys

import numpy as np

SCENE = r"D:\maya2024\scenes\ndj\test04.ma"
FIXTURE = r"D:\tmp\ocd_case3.json"
SKIN = "optimizedSkinCluster"
TOLERANCE = 0.01

# show_test04_preview.py --check 在 bend=-147.2124° 的 CoR_CurveGuided
# （已除以同一环的纯 CoR 半径）
PREVIEW = {
    -2.925: 1.0286, -2.681: 1.0606, -2.437: 1.0940, -2.193: 1.1302,
    -1.950: 1.1680, -1.706: 1.2057, -1.462: 1.2354, -1.219: 1.2508,
    -0.975: 1.2550, -0.731: 1.2508, -0.487: 1.2406, -0.244: 1.2239,
    0.000: 1.1991, 0.244: 1.2239, 0.487: 1.2406, 0.731: 1.2508,
    0.975: 1.2550, 1.219: 1.2508, 1.462: 1.2354, 1.706: 1.2057,
    1.950: 1.1680, 2.194: 1.1302, 2.437: 1.0940, 2.681: 1.0606,
    2.925: 1.0286,
}
# 预览的 OCD 格（alg 3，同一姿态）
PREVIEW_OCD = {
    -2.925: 1.0220, -2.681: 1.0307, -2.437: 1.0431, -2.193: 1.0585,
    -1.950: 1.0773, -1.706: 1.1012, -1.462: 1.1290, -1.219: 1.1602,
    -0.975: 1.1906, -0.731: 1.2137, -0.487: 1.2223, -0.244: 1.2245,
    0.000: 1.2195, 0.244: 1.2245, 0.487: 1.2223, 0.731: 1.2137,
    0.975: 1.1906, 1.219: 1.1602, 1.462: 1.1290, 1.706: 1.1012,
    1.950: 1.0773, 2.194: 1.0585, 2.437: 1.0431, 2.681: 1.0307,
    2.925: 1.0220,
}


def buckets(stations):
    """按中位环宽分桶。用最小间隔会把浮点噪声拆成假环（export_ocd_case.py）。"""
    uniq = np.unique(np.round(stations, 5))
    gaps = sorted(b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6)
    width = gaps[len(gaps) // 2]
    grouped = {}
    for i, s in enumerate(stations):
        grouped.setdefault(int(math.floor((s - uniq[0]) / width + 0.5)), []).append(i)
    return grouped, width, float(uniq[0])


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        case = json.load(open(FIXTURE))
        bind = np.asarray(case["mesh"]["bindPoints"], float)
        child = [b for b in case["bones"] if b["short"] == "L_Ankle_jnt"][0]
        world = child["worldMatrix"]
        bend = math.degrees(math.atan2(world[4], world[0]))
        axis = np.asarray(case["hinge"]["axis"], float)
        axis /= np.linalg.norm(axis)
        pivot = np.asarray(child["bindPosition"], float)
        rel = bind - pivot
        station = rel @ axis
        radius = float(np.linalg.norm(rel - np.outer(station, axis), axis=1).max())
        grouped, width, base = buckets(station)
        centre = {k: base + k * width for k in grouped}
        print("R = %.6f  环数 = %d  环宽 = %.4f  bend = %.4f°"
              % (radius, len(grouped), width, bend))

        weights = [{int(k): float(v) for k, v in row.items() if float(v) > 0.0}
                   for row in case["weightsSparse"]]

        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        mc.file(SCENE, open=True, force=True)
        bones = [(0, "joint1"), (1, "L_Ankle_jnt"), (2, "joint3")]
        bind_pre = {i: [float(v) for v in
                        mc.getAttr("skinCluster1.bindPreMatrix[%d]" % i)]
                    for i, _ in bones}
        # 场景打开时是零姿势；这里摆成夹具那一帧（负弯 147.2124°）。
        mc.setAttr("L_Ankle_jnt.rotateZ", bend)
        mc.refresh(force=True)

        def ring(points, idx):
            q = points[idx]
            c = q.mean(axis=0)
            return float(np.linalg.norm(q - c, axis=1).mean())

        def attach(shape, name):
            skin = mc.deformer(shape, type=SKIN, name=name)[0]
            for index, joint in bones:
                mc.connectAttr(joint + ".worldMatrix[0]",
                               "%s.matrix[%d]" % (skin, index))
                mc.setAttr("%s.bindPreMatrix[%d]" % (skin, index),
                           *bind_pre[index], type="matrix")
            for vertex, row in enumerate(weights):
                for index, value in row.items():
                    if index < 3:
                        mc.setAttr("%s.weightList[%d].weights[%d]"
                                   % (skin, vertex, index), value)
            mc.setAttr(skin + ".corDebugLog", True)
            return skin

        def read(shape):
            for _ in range(2):
                mc.dgeval(shape)
                mc.refresh(force=True)
            sel = om.MSelectionList()
            sel.add(shape)
            return np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
                sel.getDagPath(0)).getPoints(om.MSpace.kObject)])

        def profile(shape, skin):
            """返回 {x/R: (cor/R, fig/cor, 预览fig, ocd/cor)}。"""
            mc.refresh(force=True)
            mc.setAttr(skin + ".algorithmType", 5)
            mc.refresh(force=True)
            hinge = mc.getAttr(skin + ".ocdHingeReport")
            rows = {}
            for alg, key in ((0, "cor"), (3, "ocd"), (5, "fig")):
                mc.setAttr(skin + ".algorithmType", alg)
                mc.refresh(force=True)
                rows[key] = read(shape)
            out = {}
            for bucket in sorted(grouped):
                x = round(centre[bucket] / radius, 3)
                if x not in PREVIEW or len(grouped[bucket]) < 60:
                    continue
                c = ring(rows["cor"], grouped[bucket])
                f = ring(rows["fig"], grouped[bucket])
                o = ring(rows["ocd"], grouped[bucket])
                out[x] = (c / radius, f / c, PREVIEW[x], o / c, PREVIEW_OCD[x])
            return hinge, out

        # ---- A. 上游：直接建在 rest 网格上（用户实际用法） ----
        target = "pCylinderTargetShape"
        upstream = attach(target, "target_skin#")
        print("\nA target=%s  deformer 输入: %s"
              % (target, mc.listConnections(upstream + ".input[0].inputGeometry",
                                            source=True, destination=False,
                                            plugs=True)))
        hinge_a, rows_a = profile(target, upstream)
        print("A hinge:", hinge_a)
        print("%9s %10s %10s %10s | %10s %10s"
              % ("x/R", "cor/R", "fig/cor", "预览fig", "差", "ocd/cor"))
        worst = 0.0
        for x in sorted(rows_a):
            c, ratio, preview, ocd, ocd_preview = rows_a[x]
            worst = max(worst, abs(ratio - preview))
            print("%9.3f %10.4f %10.4f %10.4f %+10.4f %10.4f"
                  % (x, c, ratio, preview, ratio - preview, ocd))
        print("A 最大差 %.4f" % worst)
        if worst > TOLERANCE:
            failures.append("A 上游接法未复现预览，最大差 %.4f" % worst)

        # ---- B. 下游：建在原生 skinCluster 之后（默认插入位置） ----
        native = "pCylinderShape2"
        mc.delete(upstream)
        downstream = attach(native, "downstream_skin#")
        source = mc.listConnections(downstream + ".input[0].inputGeometry",
                                    source=True, destination=False, plugs=True)
        print("\nB target=%s  deformer 输入: %s" % (native, source))
        hinge_b, rows_b = profile(native, downstream)
        print("B hinge:", hinge_b)
        print("%9s %10s %10s %10s"
              % ("x/R", "fig/cor", "预览fig", "fig绝对/R"))
        worst_b = 0.0
        for x in sorted(rows_b):
            c, ratio, preview, ocd, ocd_preview = rows_b[x]
            worst_b = max(worst_b, abs(ratio - preview))
            print("%9.3f %10.4f %10.4f %10.4f"
                  % (x, ratio, preview, c * ratio))
        print("B 最大差 %.4f（接在下游时偏离预览，这是接法问题，不是求解器问题）"
              % worst_b)
        if worst_b < TOLERANCE:
            failures.append("B 下游接法也复现了预览，说明 A/B 无差别，"
                            "1.03R 另有原因")

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            print("FAILURES:", len(failures))
            return 1
        print("PASS: 插件单独跑同样输入复现预览（A 最大差 %.4f）；"
              "接在原生蒙皮下游时偏离（B 最大差 %.4f）" % (worst, worst_b))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
