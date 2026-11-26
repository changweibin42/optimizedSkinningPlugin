# -*- coding: utf-8 -*-
r"""主铰链选择的回归：高权重 helper、influence 顺序、未蒙皮中间关节。

三件事各自都会让「求解器输出平滑网格、但跟随了错误的弯曲角」：

A. helper 权重高于 elbow。两者有同一个 DAG 父骨（prev），都满足父子关系；
   若只按累计权重选，helper 会被选成主铰链的 child。
B. 交换 influence 接入顺序（helper 的 logical index 变成 0）。选择结果不应
   依赖 matrix[] 的接入顺序。
C. 中间夹一个没有蒙皮的关节：上臂 → 未蒙皮中间关节 → 前臂。该子骨仍应
   沿真实 DAG 找到最近蒙皮祖先，而不是被当成根节点跳过。

判定方式：打开调试日志，解析插件打印的 hinge parent/child 名称。

用法：
    mayapy tests/maya_ocd_hinge_select.py
"""
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import ocd_elbow_reference as ref

HINGE_PATTERN = re.compile(r"parent=(\S+) child=(\S+) helper=(\S+)")
# 高权重 helper 场景：helper 累计权重明显大于 elbow。
WEIGHTS_HELPER_DOMINANT = (0.5, 0.1, 0.4)


def make_mesh(mc, om, parent, name, bind, faces):
    geo = mc.createNode("transform", name=name, parent=parent)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True, fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")
    return geo, shape


def add_joint(mc, parent, name, dx):
    joint = mc.createNode("joint", name=name, parent=parent)
    mc.setAttr(joint + ".translateX", dx)
    mc.setAttr(joint + ".radius", 0.09)
    return joint


def connect_skin(mc, shape, joints, weights, algorithm):
    skin = mc.deformer(shape, type="optimizedSkinCluster", name="sel_skin#")[0]
    for k, joint in enumerate(joints):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, row in enumerate(weights):
        mc.setAttr("%s.weightList[%d].weights[0:%d]" % (skin, vertex, len(row) - 1),
                   *row, size=len(row))
    mc.setAttr(skin + ".algorithmType", algorithm)
    return skin


def build_coincident_case(mc, om, bind, faces, label, swap_order):
    """prev -> elbow 与 prev -> helper 共点；helper 权重更大。"""
    holder = mc.createNode("transform", name="sel_%s_holder#" % label)
    geo, shape = make_mesh(mc, om, holder, "sel_%s_geo#" % label, bind, faces)
    prev = add_joint(mc, holder, "sel_%s_prev#" % label, -ref.HALF)
    elbow = add_joint(mc, prev, "sel_%s_elbow#" % label, ref.HALF)
    tip = add_joint(mc, elbow, "sel_%s_tip#" % label, ref.HALF)
    helper = add_joint(mc, prev, "sel_%s_helper#" % label, ref.HALF)

    if swap_order:
        # helper 先接入 -> helper 的 logical index 变成 0。
        joints = [helper, prev, elbow]
        rows = [(WEIGHTS_HELPER_DOMINANT[2], WEIGHTS_HELPER_DOMINANT[0],
                 WEIGHTS_HELPER_DOMINANT[1])] * len(bind)
    else:
        joints = [prev, elbow, helper]
        rows = [WEIGHTS_HELPER_DOMINANT] * len(bind)

    skin = connect_skin(mc, shape, joints, rows, 3)
    mc.setAttr(elbow + ".rotateZ", 118.0)
    mc.setAttr(helper + ".rotateZ", 59.0)
    mc.refresh(force=True)
    return holder, skin, shape


def build_unskinned_middle_case(mc, om, bind, faces, label):
    """上臂 -> 未蒙皮中间关节 -> 前臂；只把上臂与前臂接入蒙皮。"""
    holder = mc.createNode("transform", name="sel_%s_holder#" % label)
    geo, shape = make_mesh(mc, om, holder, "sel_%s_geo#" % label, bind, faces)
    upper = add_joint(mc, holder, "sel_%s_upper#" % label, -ref.HALF)
    middle = add_joint(mc, upper, "sel_%s_middle#" % label, ref.HALF)
    fore = add_joint(mc, middle, "sel_%s_fore#" % label, 0.0)
    tip = add_joint(mc, fore, "sel_%s_tip#" % label, ref.HALF)

    rows = []
    for vertex, (station, _, _) in enumerate(bind):
        w = ref.skin_weight(station)
        rows.append((1.0 - w, w))
    skin = connect_skin(mc, shape, [upper, fore], rows, 3)
    mc.setAttr(fore + ".rotateZ", 118.0)
    mc.refresh(force=True)
    return holder, skin, shape, middle, fore


def run_algorithm(mc, skin, shape, om, algorithm):
    """跑一次 deform，返回插件写下的 hinge 报告字符串。"""
    mc.setAttr(skin + ".algorithmType", algorithm)
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    for _ in range(3):
        mc.refresh(force=True)
        mesh.getPoints(om.MSpace.kObject)
    return mc.getAttr(skin + ".ocdHingeReport") or ""


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    failures = []
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = ref.cylinder()

        # 让插件写 hinge 报告：corDebugLog 打开即可。
        def enable_debug(skin):
            mc.setAttr(skin + ".corDebugLog", True)

        # ---------------- A / B：高权重 helper + 顺序交换 ----------------
        expected_child_tail = "elbow"
        for swap in (False, True):
            label = "swap" if swap else "normal"
            holder, skin, shape = build_coincident_case(
                mc, om, bind, faces, label, swap)
            enable_debug(skin)
            mc.setAttr(skin + ".ocdTwistWeight", 1.0)

            report = run_algorithm(mc, skin, shape, om, 3)
            match = HINGE_PATTERN.search(report)
            if match:
                parent, child, helper = match.groups()
                ok = expected_child_tail in child
                print("A/B order=%-6s hinge parent=%s child=%s helper=%s  %s"
                      % (label, parent, child, helper, "OK" if ok else "WRONG"))
                if not ok:
                    failures.append("order=%s 选中 child=%s，应为 elbow" % (label, child))
            else:
                print("A/B order=%-6s 报告为空: %r" % (label, report))
                failures.append("order=%s 没有 hinge 报告" % label)
            mc.delete(holder)

        # ---------------- C：未蒙皮中间关节 ----------------
        holder, skin, shape, middle, fore = build_unskinned_middle_case(
            mc, om, bind, faces, "unskinned")
        enable_debug(skin)
        report = run_algorithm(mc, skin, shape, om, 3)
        match = HINGE_PATTERN.search(report)
        if match:
            parent, child, helper = match.groups()
            ok = "fore" in child and "upper" in parent
            print("C   unskinned-middle hinge parent=%s child=%s  %s"
                  % (parent, child, "OK" if ok else "WRONG"))
            if not ok:
                failures.append("未蒙皮中间关节：选中 %s -> %s" % (parent, child))
        else:
            print("C   报告为空（子骨被当成根节点跳过？）: %r" % report)
            failures.append("未蒙皮中间关节：没有 hinge 报告")
        mc.delete(holder)

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            print("FAILURES:", len(failures))
            return 1
        print("PASS: 高权重 helper、influence 顺序、未蒙皮中间关节均选对主铰链")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
