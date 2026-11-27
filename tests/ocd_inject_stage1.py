# -*- coding: utf-8 -*-
r"""第一步（mayapy，无 numpy）：
  A) 自动求解 -> 导出顶点
  B) 注入 C++ 自身 delta -> 导出顶点（验证注入路径）
  C) 注入外部 Python 算好的 delta -> 导出顶点

外部 Python 先算好 python_delta.json（见 make_python_delta.py）。

    mayapy tests/ocd_inject_stage1.py D:/tmp/ocd_case3.json tests/_stage1.json
"""
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, HERE)


def build(mc, om, bind, faces, bones, sparse, name, bend):
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
    mc.setAttr(joints[1] + ".rotateZ", bend)
    mc.refresh(force=True)
    mc.dgeval(shape)
    return holder, shape, skin


def read_points(om, shape):
    sel = om.MSelectionList()
    sel.add(shape)
    return [[p.x, p.y, p.z] for p in
            om.MFnMesh(sel.getDagPath(0)).getPoints(om.MSpace.kObject)]


def inject(mc, om, skin, values):
    sel = om.MSelectionList()
    sel.add(skin)
    node = om.MFnDependencyNode(sel.getDependNode(0))
    plug = node.findPlug("ocdCenterOffsets", False)
    plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray(values)))
    mc.refresh(force=True)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
        target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(
            os.getcwd(), "tests", "_stage1.json")
        py_delta_path = sys.argv[3] if len(sys.argv) > 3 else os.path.join(
            os.getcwd(), "tests", "_python_delta.json")

        d = json.load(open(source))
        bind = d["mesh"]["bindPoints"]
        faces = d["mesh"]["faces"]
        bones = d["bones"]
        sparse = d["weightsSparse"]
        n = len(bind)
        bend = 146.126

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

        out = {"bend": bend, "n": n}

        # ---- A) 自动 ----
        holder_a, shape_a, skin_a = build(mc, om, bind, faces, bones, sparse,
                                          "ca", bend)
        for _ in range(3):
            mc.dgeval(shape_a)
            mc.refresh(force=True)
            read_points(om, shape_a)
        out["auto"] = read_points(om, shape_a)
        guide = list(mc.getAttr(skin_a + ".ocdGuideDump") or [])
        delta = list(mc.getAttr(skin_a + ".ocdDeltaDump") or [])
        limit = list(mc.getAttr(skin_a + ".ocdCenterLimitDump") or [])
        out["cppGuide"] = guide
        out["cppDelta"] = delta
        out["centerLimit"] = limit
        print("A) 自动: 顶点=%d  guide=%d  delta=%d  limit=%s"
              % (len(out["auto"]), len(guide), len(delta), limit))

        # ---- B) 注入 C++ 自身 delta ----
        holder_b, shape_b, skin_b = build(mc, om, bind, faces, bones, sparse,
                                          "cb", bend)
        inject(mc, om, skin_b, delta)
        for _ in range(3):
            mc.dgeval(shape_b)
            mc.refresh(force=True)
        out["injCpp"] = read_points(om, shape_b)
        print("B) 注入 C++ delta: 顶点=%d" % len(out["injCpp"]))

        # ---- C) 注入 Python 的 delta ----
        if os.path.exists(py_delta_path):
            payload = json.load(open(py_delta_path))
            py_obj = payload["objectDelta"]
            if len(py_obj) == 3 * n:
                holder_c, shape_c, skin_c = build(mc, om, bind, faces, bones,
                                                  sparse, "cc", bend)
                inject(mc, om, skin_c, py_obj)
                for _ in range(3):
                    mc.dgeval(shape_c)
                    mc.refresh(force=True)
                out["injPy"] = read_points(om, shape_c)
                print("C) 注入 Python delta: 顶点=%d" % len(out["injPy"]))
            else:
                print("C) 跳过：Python delta 长度 %d != %d"
                      % (len(py_obj), 3 * n))
        else:
            print("C) 跳过：找不到", py_delta_path)

        json.dump(out, open(target, "w"))
        print("wrote", target)
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
