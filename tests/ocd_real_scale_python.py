# -*- coding: utf-8 -*-
r"""在 test04 的真实网格上跑 ocd_elbow_reference.py 的算法。

目的：判断褶皱是 Python 算法本身的问题，还是 C++ 复现的问题。

关键：不假设尺度，直接按你场景的几何算：
  1) Elbow 解析模型 —— 喂真实 bind（这是 Python 里"显示的"东西）
  2) hinge_displacement + structural_centers —— Python 算出的 center 偏移
  3) 用 Python 自己的 delta 驱动同一个 Maya 节点，看最终形状

    mayapy tests/ocd_real_scale_python.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, r"D:\optimizedSkinningPlugin")


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        # 用你仓库里那份原始脚本
        sys.path.insert(0, r"D:\optimizedSkinningPlugin")
        import ocd_elbow_reference as userref
        print("加载:", userref.__file__)
        print("HALF=%.3f  REACH=%.3f  HALF_RINGS=%d  SIDES=%d"
              % (userref.HALF, userref.REACH, userref.HALF_RINGS,
                 userref.SIDES))

        src = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        d = json.load(open(src))
        bind = [tuple(p) for p in d["mesh"]["bindPoints"]]
        n = len(bind)
        bend = 146.1265

        bones = d["bones"]
        by = {b["short"]: b for b in bones}
        child = by["L_Ankle_jnt"]
        pivot = child["bindPosition"]
        axis = d["hinge"]["axis"]
        na = math.sqrt(sum(v * v for v in axis))
        axis = [v / na for v in axis]

        R = max(math.sqrt(sum((p[i] - pivot[i]) ** 2 for i in range(3)))
                for p in bind)
        print("场景 maxRadius = %.4f" % R)
        print("场景长度 = %.1f (x: %.1f..%.1f)"
              % (max(p[0] for p in bind) - min(p[0] for p in bind),
                 min(p[0] for p in bind), max(p[0] for p in bind)))

        # ---------- 1) Elbow 解析模型 ----------
        print("\n== 1) Elbow 解析模型（喂真实 bind）==")
        try:
            e = userref.Elbow(bend, 0.0, 0.0, userref.ZERO, False)
            analytic = [list(p) for p in e.mesh(bind)]
            print("  成功")
        except Exception as exc:
            print("  失败:", exc)
            analytic = None

        # ---------- 2) Python 的 guide + delta ----------
        print("\n== 2) Python 的 hinge_displacement + structural_centers ==")
        W = []
        for row in d["weightsSparse"]:
            w0 = 1.0 - float(row.get("1", 0.0))
            w1 = float(row.get("1", 0.0))
            W.append([w0, w1])
        # 归一化到单位半径的局部坐标
        bind_l = [[(p[0] - pivot[0]) / R, (p[1] - pivot[1]) / R,
                   (p[2] - pivot[2]) / R] for p in bind]

        import numpy as np
        bind_l = np.asarray(bind_l)
        W = np.asarray(W)

        cor = np.asarray(d["points"]["cor"])
        cor_l = np.column_stack([
            ((cor - np.asarray(pivot)) @ np.asarray(axis)) / R,
            (cor - np.asarray(pivot))[:, 1] / R,
            (cor - np.asarray(pivot))[:, 2] / R,
        ])

        faces = d["mesh"]["faces"]
        try:
            edges = userref.local_edges(bind_l.tolist(), faces)
            print("  local_edges 成功")
        except Exception as exc:
            print("  local_edges 失败:", exc)
            edges = None

        if edges is not None:
            q = userref.joint_quaternion(bend, 0.0, 0.0)
            rotations = [(0.0, 0.0, 0.0, 1.0), q]
            try:
                guide = np.asarray(userref.hinge_displacement(
                    bind_l, cor_l, W, bend, 0.0, 0.0))
                print("  hinge_displacement 成功, |guide|均=%.4f"
                      % np.linalg.norm(guide, axis=1).mean())
            except Exception as exc:
                print("  hinge_displacement 失败:", exc)
                guide = None
            if guide is not None:
                try:
                    outer = np.asarray(userref.outer_participation(
                        bind_l, W, bend, 0.0, 0.0))
                    delta = np.asarray(userref.structural_centers(
                        edges, cor_l, W, rotations, guide, outer, 16.0))
                    print("  structural_centers 成功, |delta|均=%.4f max=%.4f"
                          % (np.linalg.norm(delta, axis=1).mean(),
                             np.linalg.norm(delta, axis=1).max()))
                except Exception as exc:
                    print("  structural_centers 失败:", exc)
                    delta = None

        # ---------- 3) 圆度比较 ----------
        xw = [p[0] for p in bind]

        def roundness(pts):
            out = {}
            for x0 in (-6.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0):
                sel = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
                if len(sel) < 10:
                    continue
                rr = [math.hypot(pts[i][1], pts[i][2]) for i in sel]
                m = sum(rr) / len(rr)
                v = sum((q - m) ** 2 for q in rr) / len(rr)
                out[x0] = (math.sqrt(v) / max(1e-9, m), m)
            return out

        xs = [-6.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 6.0]
        print("\n== 圆度（越小越圆）==")
        print("%24s" % "" + "".join("  x=%+.1f" % v for v in xs))

        # 插件输出（用同一口径：撤旋转）
        final = d["points"]["final"]
        th = math.radians(bend)
        ct, st = math.cos(th), math.sin(th)
        loc = []
        for i, p in enumerate(final):
            x, y, z = p
            if bind[i][0] >= 0:
                x, y = ct * x + st * y, -st * x + ct * y
            loc.append((x, y, z))
        sp = roundness(loc)
        line = "%24s" % "插件算法5 (test04)"
        for x0 in xs:
            line += " %5.3f" % sp[x0][0] if x0 in sp else "     -"
        print(line)

        if analytic is not None:
            sa = roundness(analytic)
            line = "%24s" % "Elbow解析(真实bind)"
            for x0 in xs:
                line += " %5.3f" % sa[x0][0] if x0 in sa else "     -"
            print(line)
            line = "%24s" % "Elbow径向"
            for x0 in xs:
                line += " %5.3f" % sa[x0][1] if x0 in sa else "     -"
            print(line)

        print("\n%24s" % "" + "".join("  x=%+.1f" % v for v in xs))
        line = "%24s" % "插件径向"
        for x0 in xs:
            line += " %5.3f" % sp[x0][1] if x0 in sp else "     -"
        print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
