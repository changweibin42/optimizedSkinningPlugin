# -*- coding: utf-8 -*-
r"""确认角色场景的 491 条 nurbsCurve 是什么，以及能否作为 Influence Curve 用。

要查：
  1. 这些曲线的命名与归属（是控制曲线？毛囊？还是蒙皮用的）
  2. 主干网格是哪个（最大、被最多骨影响的那个）
  3. 该网格的 skinCluster 有哪些影响骨，链条能否分组
  4. 是否存在「权重相似但分属不同链」的区域

    mayapy tests/probe_character2.py
"""
import os
import sys
from collections import Counter

SCENE = (r"D:\maya2024\scenes\ndj\character\phsm_rat_girl_c\01_skin"
         r"\phsm_rat_girl_c.rig.rigging.v011.ma")


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(SCENE, open=True, force=True, loadReferenceDepth="all")

        # ---- 1. 曲线的命名族 ----
        curves = mc.ls(type="nurbsCurve", noIntermediate=True) or []
        print("nurbsCurve 总数 %d" % len(curves))
        names = [c.split("|")[-1].replace("Shape", "") for c in curves]
        # 取名字前缀（去掉数字）看族群
        def fam(n):
            base = "".join(ch for ch in n if not ch.isdigit()).rstrip("_")
            return base.split("_")[0] if "_" in base else base
        fams = Counter(fam(n) for n in names)
        print("  命名族群（前 12）：")
        for k, v in fams.most_common(12):
            print("    %-28s %3d" % (k, v))

        # ---- 2. 主体网格 ----
        meshes = mc.ls(type="mesh", noIntermediate=True) or []
        biggest = []
        for sh in meshes:
            try:
                n = mc.polyEvaluate(sh, vertex=True)
            except Exception:
                n = -1
            if n > 0:
                biggest.append((n, sh))
        biggest.sort(reverse=True)
        print("\n最大的 6 个网格：")
        for n, sh in biggest[:6]:
            # 找它的 skinCluster
            par = mc.listRelatives(sh, parent=True, fullPath=True)
            sk = None
            if par:
                hist = mc.listHistory(par[0]) or []
                for h in hist:
                    if mc.nodeType(h) == "skinCluster":
                        sk = h
                        break
            ninf = -1
            if sk:
                try:
                    ninf = len(mc.skinCluster(sk, q=True, influence=True) or [])
                except Exception:
                    pass
            print("    %-38s verts %6d  skin=%s  influences %d"
                  % (sh, n, sk if sk else "-", ninf))

        # ---- 3. 主体网格的骨骼链分组 ----
        target = biggest[0][1] if biggest else None
        if target:
            par = mc.listRelatives(target, parent=True, fullPath=True)[0]
            sk = None
            for h in (mc.listHistory(par) or []):
                if mc.nodeType(h) == "skinCluster":
                    sk = h
                    break
            print("\n主体网格 %s 的 skinCluster: %s" % (target, sk))
            if sk:
                infs = mc.skinCluster(sk, q=True, influence=True) or []
                print("  影响骨 %d 个：" % len(infs))
                # 按命名族分组
                groups = Counter()
                for j in infs:
                    nm = j.split("|")[-1]
                    key = nm.split("_")[0]
                    groups[key] += 1
                for k, v in groups.most_common(14):
                    print("    %-22s %2d" % (k, v))
                # 空间接近的跨族骨对
                pos = {}
                for j in infs:
                    try:
                        pos[j] = mc.xform(j, q=True, ws=True, t=True)
                    except Exception:
                        pass
                close = []
                js = sorted(pos)
                for a in range(len(js)):
                    for b in range(a + 1, len(js)):
                        ja, jb = js[a], js[b]
                        fa = ja.split("|")[-1].split("_")[0]
                        fb = jb.split("|")[-1].split("_")[0]
                        if fa == fb:
                            continue
                        d = sum((x - y) ** 2 for x, y in
                                zip(pos[ja], pos[jb])) ** 0.5
                        close.append((d, ja, jb))
                close.sort()
                print("\n  跨族空间接近的骨对（前 8）：")
                for d, a, b in close[:8]:
                    print("    %7.3f  %-30s <-> %s"
                          % (d, a.split("|")[-1], b.split("|")[-1]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
