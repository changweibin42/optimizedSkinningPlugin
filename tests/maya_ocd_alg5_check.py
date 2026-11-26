# -*- coding: utf-8 -*-
r"""算法 5（OCD+CorrectedCoR）的核心校验，不依赖 numpy。

Maya 自带的 Python 没有 numpy，所以这个脚本只做不需要参考实现的检查，
真正的逐级对拍由 `tests/ocd_alg5_truth.py`（外部 numpy 环境）配合
`tests/maya_ocd_alg5_dump.py` 完成。

这里验证四件事，任一失败都退出非零：

A. original 语义 —— 把 `ocdCenterOffsets` 显式置零后，算法 5 必须等于
   普通 CoR（算法 0）。这直接证明 `original` 是「零偏移时当前姿态的
   CoR 输出」，而不是绑定位置。
B. 修正确实生效 —— 自动求解的算法 5 必须与算法 0 有可测量差异，且偏移
   幅度落在合理范围（默认中心上限 16 个肢体半径内）。
C. 刚性不变性 —— 给整条 rig（网格 + 骨骼）套一个平移与旋转后，输出在
   网格自身空间里必须完全不变。世界坐标轴参与求解时这条必然失败。
D. 补偿通道 —— 参考实现的 hinge_displacement 不读这 11 个通道，所以算法 3
   必须响应、算法 5 必须不响应。两者都写成断言，避免以后把二级编辑重复
   计算一次。

用法：
    set "MAYA_PLUG_IN_PATH=...\build\Release"
    mayapy tests/maya_ocd_alg5_check.py
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
# scripts/ 提供模型与权重函数，tests/ 提供 1285f9a 版参考实现。
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)

import ocd_elbow_reference as ref
import ocd_elbow_reference_latest as ref_latest

TOLERANCE_IDENTITY = 5.0e-3     # 零偏移算法 5 == CoR
TOLERANCE_RIGID = 5.0e-3        # 刚性变换后的不变性
BENDS = (118.0, 150.0)
# --three-bone：用参考的三骨有效权重与 helper 骨。
THREE_BONE = "--three-bone" in sys.argv


# ----------------------------------------------------------------------
# 场景
# ----------------------------------------------------------------------

def build_rig(mc, om, bind, faces, label, parent_transform=None):
    holder = mc.createNode("transform", name="chk_%s_holder#" % label)
    if parent_transform is not None:
        translate, rotate_degrees = parent_transform
        mc.setAttr(holder + ".translate", *translate)
        mc.setAttr(holder + ".rotate", *rotate_degrees)

    geo = mc.createNode("transform", name="chk_%s_geo#" % label, parent=holder)
    selection = om.MSelectionList()
    selection.add(geo)
    om.MFnMesh().create([om.MPoint(*p) for p in bind],
                        [len(f) for f in faces],
                        [j for f in faces for j in f],
                        parent=selection.getDependNode(0))
    shape = mc.listRelatives(geo, shapes=True, noIntermediate=True, fullPath=True)[0]
    mc.sets(geo, edit=True, forceElement="initialShadingGroup")

    prev = mc.createNode("joint", name="chk_%s_prev#" % label, parent=holder)
    elbow = mc.createNode("joint", name="chk_%s_elbow#" % label, parent=prev)
    tip = mc.createNode("joint", name="chk_%s_tip#" % label, parent=elbow)
    for joint, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
        mc.setAttr(joint + ".translateX", dx)
        mc.setAttr(joint + ".radius", 0.09)
    rig = {"holder": holder, "geo": geo, "shape": shape,
           "prev": prev, "elbow": elbow, "helper": None}
    if THREE_BONE:
        helper = mc.createNode("joint", name="chk_%s_helper#" % label, parent=prev)
        mc.setAttr(helper + ".translateX", ref.HALF)
        mc.setAttr(helper + ".radius", 0.09)
        rig["helper"] = helper
    return rig


def attach_skin(mc, rig, algorithm, bind):
    skin = mc.deformer(rig["shape"], type="optimizedSkinCluster",
                       name="chk_skin#")[0]
    joints = [rig["prev"], rig["elbow"]]
    if rig["helper"] is not None:
        joints.append(rig["helper"])
    for k, joint in enumerate(joints):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, (station, _, _) in enumerate(bind):
        if rig["helper"] is not None:
            row = ref_latest.preview_influences(station, True)
        else:
            w = ref.skin_weight(station)
            row = (1.0 - w, w)
        mc.setAttr("%s.weightList[%d].weights[0:%d]" % (skin, vertex, len(row) - 1),
                   *row, size=len(row))
    mc.setAttr(skin + ".algorithmType", algorithm)
    result = mc.listRelatives(rig["geo"], shapes=True,
                              noIntermediate=True, fullPath=True)[0]
    return skin, result


def read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [(p.x, p.y, p.z) for p in mesh.getPoints(om.MSpace.kObject)]


def pose_elbow(mc, om, rig, bend):
    selection = om.MSelectionList()
    selection.add(rig["elbow"])
    om.MFnTransform(selection.getDagPath(0)).setRotation(
        om.MQuaternion(*ref.joint_quaternion(bend, 0, 0)), om.MSpace.kTransform)
    if rig["helper"] is not None:
        helper_sel = om.MSelectionList()
        helper_sel.add(rig["helper"])
        om.MFnTransform(helper_sel.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(bend * 0.5, 0, 0)),
            om.MSpace.kTransform)
    mc.refresh(force=True)


def distance(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def max_deviation(first, second):
    return max(distance(a, b) for a, b in zip(first, second))


def rms(first, second):
    total = sum(distance(a, b) ** 2 for a, b in zip(first, second))
    return math.sqrt(total / max(len(first), 1))


def set_offsets(mc, om, skin, values):
    """通过 MPlug 写入 doubleArray；cmds.setAttr 对数组属性不可靠。"""
    selection = om.MSelectionList()
    selection.add(skin)
    plug = om.MFnDependencyNode(selection.getDependNode(0)).findPlug(
        "ocdCenterOffsets", False)
    plug.setMObject(om.MFnDoubleArrayData().create(om.MDoubleArray(values)))


# ----------------------------------------------------------------------
# 主流程
# ----------------------------------------------------------------------

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
        vertex_count = len(bind)
        print("bone set: %s" % ("three-bone (preview_influences)"
                                         if THREE_BONE else "two-bone (skin_weight)"))

        # ---------------- A + B ----------------
        print("=== A. 零偏移的算法 5 必须等于 CoR（验证 original 语义）===")
        print("=== B. 自动求解必须产生受控的修正 ===")
        reference_by_bend = {}
        for bend in BENDS:
            cor_rig = build_rig(mc, om, bind, faces, "cor")
            _, cor_result = attach_skin(mc, cor_rig, 0, bind)
            pose_elbow(mc, om, cor_rig, bend)
            cor_points = read_points(om, cor_result)

            zero_rig = build_rig(mc, om, bind, faces, "zero")
            zero_skin, zero_result = attach_skin(mc, zero_rig, 5, bind)
            pose_elbow(mc, om, zero_rig, bend)
            set_offsets(mc, om, zero_skin, [0.0] * (3 * vertex_count))
            zero_points = read_points(om, zero_result)

            auto_rig = build_rig(mc, om, bind, faces, "auto")
            _, auto_result = attach_skin(mc, auto_rig, 5, bind)
            pose_elbow(mc, om, auto_rig, bend)
            auto_points = read_points(om, auto_result)

            identity_error = max_deviation(zero_points, cor_points)
            correction = rms(auto_points, cor_points)
            print("bend=%5.1f  零偏移==CoR 偏差 %.6f   自动求解 vs CoR rms %.4f  max %.4f"
                  % (bend, identity_error, correction,
                     max_deviation(auto_points, cor_points)))

            if identity_error > TOLERANCE_IDENTITY:
                failures.append("bend=%.0f: 零偏移算法 5 与 CoR 差 %.3e"
                                % (bend, identity_error))
            if not (1.0e-4 < correction < 16.0):
                failures.append("bend=%.0f: 修正幅度异常 rms=%.3e" % (bend, correction))

            reference_by_bend[bend] = cor_points
            for rig in (cor_rig, zero_rig, auto_rig):
                mc.delete(rig["holder"])

        # ---------------- C ----------------
        print()
        print("=== C. 整套 rig 平移 + 旋转后，输出在网格空间里必须不变 ===")
        parent_transform = ((13.0, -7.0, 25.0), (21.0, -58.0, 37.0))
        for algorithm, label in ((0, "CoR(0)"), (3, "OCD(3)"), (5, "OCD+CoR(5)")):
            plain_rig = build_rig(mc, om, bind, faces, "plain")
            _, plain_result = attach_skin(mc, plain_rig, algorithm, bind)
            pose_elbow(mc, om, plain_rig, 118.0)
            plain = read_points(om, plain_result)

            moved_rig = build_rig(mc, om, bind, faces, "moved",
                                  parent_transform=parent_transform)
            _, moved_result = attach_skin(mc, moved_rig, algorithm, bind)
            pose_elbow(mc, om, moved_rig, 118.0)
            moved = read_points(om, moved_result)

            deviation = max_deviation(moved, plain)
            print("%-12s 刚性不变性偏差 %.6f  %s"
                  % (label, deviation,
                     "OK" if deviation <= TOLERANCE_RIGID else "MISMATCH"))
            if deviation > TOLERANCE_RIGID:
                failures.append("%s: 刚性变换偏差 %.3e" % (label, deviation))
            mc.delete(plain_rig["holder"])
            mc.delete(moved_rig["holder"])

        # ---------------- D ----------------
        # 参考实现的 hinge_displacement 只用 hinge_section，不读取 11 个补偿通道，
        # 所以算法 5 的 guide 天然不受它们影响。这里把这条行为显式断言下来：
        #   · 算法 3 必须响应补偿通道
        #   · 算法 5 必须不响应（否则就是把二级编辑重复算了一次）
        print()
        print("=== D. 补偿通道：算法 3 必须响应，算法 5 必须不响应（与参考一致）===")
        for algorithm, label, should_respond in ((3, "OCD(3)", True),
                                                 (5, "OCD+CoR(5)", False)):
            base_rig = build_rig(mc, om, bind, faces, "chan0")
            base_skin, base_result = attach_skin(mc, base_rig, algorithm, bind)
            pose_elbow(mc, om, base_rig, 118.0)
            baseline = read_points(om, base_result)

            pushed_rig = build_rig(mc, om, bind, faces, "chan1")
            pushed_skin, pushed_result = attach_skin(mc, pushed_rig, algorithm, bind)
            pose_elbow(mc, om, pushed_rig, 118.0)
            mc.setAttr(pushed_skin + ".ocdBulge", 0.5)
            mc.setAttr(pushed_skin + ".ocdCenterSlideIn", 0.4)
            mc.refresh(force=True)
            pushed = read_points(om, pushed_result)

            change = max_deviation(pushed, baseline)
            responds = change > 5.0e-3
            print("%-12s 补偿通道生效幅度 %.6f  %s"
                  % (label, change,
                     "OK" if responds == should_respond else "UNEXPECTED"))
            if responds != should_respond:
                failures.append(
                    "%s: 补偿通道%s（实际幅度 %.3e）"
                    % (label, "应生效但未生效" if should_respond else "不应生效却生效",
                       change))
            mc.delete(base_rig["holder"])
            mc.delete(pushed_rig["holder"])

        print()
        if failures:
            for message in failures:
                print("FAIL:", message)
            print("FAILURES:", len(failures))
            return 1
        print("PASS: original 语义、修正幅度、刚性不变性、补偿通道均符合预期")
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
