# -*- coding: utf-8 -*-
r"""在你的真实场景尺度上跑 Python 实现，对比它的两个输出：

  A) Elbow 解析模型（Python 里实际显示的）    —— pose.mesh(bind)
  B) hinge_displacement + structural_centers  —— Python 里算出来但只注入不显示的

如果 A 平滑而 B 不平滑，说明 Python 实现本身也有这个问题，
只是显示的是 A 所以看不出来。这就把问题从 C++ 移植转回算法本身。

    mayapy tests/ocd_python_on_real_scale.py D:/tmp/ocd_case3.json
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om
        import ocd_elbow_reference_latest as ref

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

        # 真实尺度：半径 0.9847，长度 24
        R_true = max(math.sqrt(
            (p[0] - pivot[0]) ** 2 + (p[1] - pivot[1]) ** 2 + (p[2] - pivot[2]) ** 2)
            for p in bind)
        print("真实尺度：maxRadius=%.4f" % R_true)
        print("Elbow 假设：单位半径、长度约 2")

        # ---- A) Elbow 解析模型，直接把真实 bind 喂进去 ----
        print("\n== A) Elbow 解析模型（直接喂真实尺度 bind）==")
        try:
            e = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
            A = [list(p) for p in e.mesh(bind)]
            print("  成功，顶点 %d" % len(A))
            zs = [p[2] for p in A]
            ys = [p[1] for p in A]
            print("  输出范围: y[%.3f,%.3f] z[%.3f,%.3f]"
                  % (min(ys), max(ys), min(zs), max(zs)))
        except Exception as exc:
            print("  失败:", exc)
            A = None

        # ---- B) 把 bind 归一化后再喂 Elbow ----
        print("\n== B) bind 归一化到单位半径后喂 Elbow ==")
        bind_n = [[(p[0] - pivot[0]) / R_true,
                   (p[1] - pivot[1]) / R_true,
                   (p[2] - pivot[2]) / R_true] for p in bind]
        try:
            e2 = ref.Elbow(bend, 0.0, 0.0, ref.ZERO, False)
            B = [list(p) for p in e2.mesh(bind_n)]
            print("  成功，顶点 %d" % len(B))
            zs = [p[2] for p in B]
            ys = [p[1] for p in B]
            print("  输出范围: y[%.3f,%.3f] z[%.3f,%.3f]"
                  % (min(ys), max(ys), min(zs), max(zs)))
        except Exception as exc:
            print("  失败:", exc)
            B = None

        # ---- 圆度分析 ----
        xw = [p[0] for p in bind]

        def roundness(pts, scale=1.0):
            out = {}
            for x0 in (-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0):
                sel = [i for i in range(n) if abs(xw[i] - x0) < 0.06]
                if len(sel) < 10:
                    continue
                rr = [math.hypot(pts[i][1], pts[i][2]) for i in sel]
                m = sum(rr) / len(rr)
                v = sum((q - m) ** 2 for q in rr) / len(rr)
                out[x0] = (math.sqrt(v) / max(1e-9, m), m * scale)
            return out

        xs = [-3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0]
        print("\n== 对照（variance/mean，越小越圆）==")
        print("%22s" % "" + "".join("  x=%+.1f" % v for v in xs))

        # 插件真实输出
        final = d["points"]["final"]
        th = math.radians(bend)
        ct, st = math.cos(th), math.sin(th)
        child_side = [p[0] >= 0 for p in bind]
        loc = []
        for i, p in enumerate(final):
            x, y, z = p
            if child_side[i]:
                x, y = ct * x + st * y, -st * x + ct * y
            loc.append((x, y, z))
        st_plugin = roundness(loc)
        line = "%22s" % "插件算法5"
        for x0 in xs:
            line += " %5.3f" % st_plugin[x0][0] if x0 in st_plugin else "     -"
        print(line)

        # Elbow 真实尺度
        if A is not None:
            st_a = roundness(A)
            line = "%22s" % "Elbow(真实尺度)"
            for x0 in xs:
                line += " %5.3f" % st_a[x0][0] if x0 in st_a else "     -"
            print(line)
            line = "%22s" % "Elbow径向(真实)"
            for x0 in xs:
                line += " %5.3f" % st_a[x0][1] if x0 in st_a else "     -"
            print(line)

        # Elbow 归一化
        if B is not None:
            st_b = roundness(B, R_true)
            line = "%22s" % "Elbow(归一后xR)"
            for x0 in xs:
                line += " %5.3f" % st_b[x0][0] if x0 in st_b else "     -"
            print(line)
            line = "%22s" % "Elbow径向(归一)"
            for x0 in xs:
                line += " %5.3f" % st_b[x0][1] if x0 in st_b else "     -"
            print(line)

        # 插件径向
        print("\n%22s" % "" + "".join("  x=%+.1f" % v for v in xs))
        line = "%22s" % "插件径向"
        for x0 in xs:
            line += " %5.3f" % st_plugin[x0][1] if x0 in st_plugin else "     -"
        print(line)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
