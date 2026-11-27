# -*- coding: utf-8 -*-
r"""探查场景：找出带多组骨骼链 / 多曲线的模型，用于方向 A 的曲线归属验证。

需要的是「两条曲线在空间上接近、但对应不同材料区域」的部位，
典型是双肘/双膝相对、或者手臂贴着身体。

    mayapy tests/probe_scenes.py <scene.ma> [...]
"""
import os
import sys

SCENES = [
    r"D:\maya2024\scenes\ndj\test01.ma",
    r"D:\maya2024\scenes\ndj\test05.ma",
    r"D:\maya2024\scenes\ndj\test08.ma",
    r"D:\maya2024\scenes\ndj\test03.ma",
]


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc

        for path in (sys.argv[1:] or SCENES):
            if not os.path.exists(path):
                print("跳过（不存在）: %s" % path)
                continue
            try:
                mc.file(path, open=True, force=True)
            except Exception as exc:
                print("打开失败 %s: %s" % (os.path.basename(path), exc))
                continue
            print("\n" + "=" * 74)
            print(os.path.basename(path))
            print("=" * 74)
            joints = mc.ls(type="joint") or []
            meshes = mc.ls(type="mesh", noIntermediate=True) or []
            skins = mc.ls(type="skinCluster") or []
            curves = mc.ls(type="nurbsCurve", noIntermediate=True) or []
            print("  joints %d  meshes %d  skinClusters %d  nurbsCurves %d"
                  % (len(joints), len(meshes), len(skins), len(curves)))
            if meshes:
                for sh in meshes[:3]:
                    try:
                        n = mc.polyEvaluate(sh, vertex=True)
                    except Exception:
                        n = -1
                    print("    mesh %-28s verts %5d" % (sh, n))
            # 骨骼命名分组，看是否有左右/多链
            names = [j.split("|")[-1] for j in joints]
            prefixes = sorted({n.split("_")[0] for n in names if "_" in n})
            print("    joint 前缀: %s" % ", ".join(prefixes[:12]))
            # 找可弯曲的多骨链（连续父子 joint 且长度 > 2）
            chains = []
            for j in joints:
                kids = mc.listRelatives(j, children=True, type="joint") or []
                if kids:
                    chains.append((j.split("|")[-1], len(kids)))
            print("    有子 joint 的节点 %d 个" % len(chains))
            # 骨骼轴距离：找空间接近的关节对
            if len(joints) >= 2:
                pos = {}
                for j in joints:
                    try:
                        pos[j] = mc.xform(j, q=True, ws=True, t=True)
                    except Exception:
                        pass
                names_l = sorted(pos)
                close = []
                for i in range(len(names_l)):
                    for k in range(i + 1, len(names_l)):
                        a, b = pos[names_l[i]], pos[names_l[k]]
                        d = sum((x - y) ** 2 for x, y in zip(a, b)) ** 0.5
                        if d < 2.0:
                            close.append((d, names_l[i].split("|")[-1],
                                          names_l[k].split("|")[-1]))
                close.sort()
                if close:
                    print("    空间接近的关节对（<2 单位）前 5：")
                    for d, a, b in close[:5]:
                        print("      %6.3f  %s <-> %s" % (d, a, b))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
