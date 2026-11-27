# -*- coding: utf-8 -*-
r"""探查真实角色场景，判断能否用于方向 A 的验证。

要回答三件事：
  1. 有多骨链吗？（曲线引导需要链）
  2. 「权重相似但归属不同」这类区域是否存在？（两条相对骨链在空间接近）
  3. 归属数据能否可靠取得？

    mayapy tests/probe_character.py
"""
import os
import sys

SCENE = (r"D:\maya2024\scenes\ndj\character\phsm_rat_girl_c\01_skin"
         r"\phsm_rat_girl_c.rig.rigging.v011.ma")


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        print("打开 %s ..." % os.path.basename(SCENE))
        mc.file(SCENE, open=True, force=True, loadReferenceDepth="none")

        joints = mc.ls(type="joint") or []
        meshes = mc.ls(type="mesh", noIntermediate=True) or []
        skins = mc.ls(type="skinCluster") or []
        curves = mc.ls(type="nurbsCurve", noIntermediate=True) or []
        print("\n  joints %d  meshes %d  skinClusters %d  nurbsCurves %d"
              % (len(joints), len(meshes), len(skins), len(curves)))

        # 网格规模
        for sh in meshes[:5]:
            try:
                n = mc.polyEvaluate(sh, vertex=True)
            except Exception:
                n = -1
            print("    mesh %-40s verts %6d" % (sh, n))

        # 骨骼层级：找最长的链
        parents = {}
        for j in joints:
            par = mc.listRelatives(j, parent=True, type="joint") or []
            parents[j] = par[0] if par else None
        children = {}
        for j in joints:
            children[j] = mc.listRelatives(j, children=True, type="joint") or []

        def depth(j, seen=None):
            if seen is None:
                seen = set()
            if j in seen:
                return 0
            seen.add(j)
            kids = children.get(j, [])
            return 1 + max([depth(k, seen) for k in kids], default=0)

        roots = [j for j in joints if parents.get(j) is None]
        print("\n  根关节 %d 个" % len(roots))
        lengths = []
        for r in roots:
            lengths.append((depth(r), r))
        lengths.sort(reverse=True)
        print("  最深的 5 条链：")
        for d, r in lengths[:5]:
            print("    depth %2d  %s" % (d, r.split("|")[-1]))

        # 空间接近的不同链关节对（跨链、距离小）
        pos = {}
        for j in joints:
            try:
                pos[j] = mc.xform(j, q=True, ws=True, t=True)
            except Exception:
                pass
        # 用名字前缀区分左右/部位
        def rootOf(j):
            cur = j
            while parents.get(cur):
                cur = parents[cur]
            return cur

        close = []
        names = sorted(pos)
        for a in range(len(names)):
            for b in range(a + 1, len(names)):
                ja, jb = names[a], names[b]
                if rootOf(ja) == rootOf(jb):
                    continue
                pa, pb = pos[ja], pos[jb]
                d = sum((x - y) ** 2 for x, y in zip(pa, pb)) ** 0.5
                close.append((d, ja, jb))
        close.sort()
        print("\n  跨链空间接近的关节对（前 8）：")
        for d, a, b in close[:8]:
            print("    %7.3f  %-32s <-> %s"
                  % (d, a.split("|")[-1], b.split("|")[-1]))

        # 蒙皮权重统计：有多少顶点受 2 条以上「不同根链」影响
        if skins:
            sk = skins[0]
            print("\n  skinCluster: %s" % sk)
            try:
                infs = mc.skinCluster(sk, q=True, influence=True) or []
                print("    influences %d" % len(infs))
            except Exception as exc:
                print("    读 influences 失败: %s" % exc)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
