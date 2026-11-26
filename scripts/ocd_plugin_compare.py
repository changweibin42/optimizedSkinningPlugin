# -*- coding: utf-8 -*-
"""Maya 内逐顶点对比：Python 参考 vs 插件 OCD / OCD+CorrectedCoR。

沿用 scripts/ocd_elbow_reference.py 的圆柱模型、两段骨骼与权重函数。三格并排：

    左：Python 参考（Elbow 直接写回网格）
    中：插件 OCD            (algorithmType 3)
    右：插件 OCD+CorrectedCoR (algorithmType 5)

关键实现细节：所有格子里的**求值几何与骨骼都建在原点**，只有显示代理带网格
偏移。这样插件内部的世界空间求解（structural_centers / buildEdges）对三格看到
完全相同的输入，对比才有意义——参考脚本的预览窗口也是这么做的。

用法（Maya 脚本编辑器）：
    import sys; sys.path.insert(0, r"...\\scripts")
    import ocd_plugin_compare as c; c.show()

命令行（无界面，打印数值）：
    mayapy scripts/ocd_plugin_compare.py --check
"""
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ocd_elbow_reference as ref

HALF = ref.HALF
CHANNELS = ref.CHANNELS
VIDEO = ref.VIDEO
ZERO = ref.ZERO
skin_weight = ref.skin_weight
Elbow = ref.Elbow
joint_quaternion = ref.joint_quaternion
cylinder = ref.cylinder

WIN = "OCDPluginCompareWindow"

# 三格的网格中心偏移（cm）。求值仍在原点，这里只用于显示代理。
COLUMN_OFFSET = 16.0

CHANNEL_ATTRS = (
    "ocdCenterSlideIn", "ocdCenterSlideOut",
    "ocdSlideInPrev", "ocdSlideInNext",
    "ocdSlideOutPrev", "ocdSlideOutNext",
    "ocdSlideOrthoInPrev", "ocdSlideOrthoInNext",
    "ocdSlideOrthoOutPrev", "ocdSlideOrthoOutNext",
    "ocdBulge",
)


# ----------------------------------------------------------------------
# 场景搭建
# ----------------------------------------------------------------------

def _linear(cm):
    """把厘米换算成当前 UI 单位。"""
    from maya.api import OpenMaya as om
    return om.MDistance(cm, om.MDistance.kCentimeters).asUnits(om.MDistance.uiUnit())


def _make_mesh(mc, om, parent, name, bind, faces):
    """按参考实现的拓扑建一个圆柱网格，返回 (transform, shape 全路径)。"""
    transform = mc.createNode("transform", name=name, parent=parent)
    selection = om.MSelectionList()
    selection.add(transform)
    counts = [len(f) for f in faces]
    indices = [j for f in faces for j in f]
    points = [om.MPoint(*p) for p in bind]
    created = om.MFnMesh().create(points, counts, indices,
                                  parent=selection.getDependNode(0))
    shape = mc.rename(om.MFnDagNode(created).fullPathName(), name + "Shape#")
    mc.sets(transform, edit=True, forceElement="initialShadingGroup")
    return transform, shape


def _make_chain(mc, parent, name, bind_height):
    """建 prev / elbow / tip 三节骨骼，返回 (prev, elbow, tip)。"""
    prev = mc.createNode("joint", name=name + "_prev#", parent=parent)
    elbow = mc.createNode("joint", name=name + "_elbow#", parent=prev)
    tip = mc.createNode("joint", name=name + "_tip#", parent=elbow)
    for joint, dx in ((prev, -HALF), (elbow, HALF), (tip, HALF)):
        mc.setAttr(joint + ".translateX", _linear(dx))
        mc.setAttr(joint + ".radius", 0.09)
    return prev, elbow, tip


def _attach_skin(mc, shape, prev, elbow, algorithm, bind):
    """挂 optimizedSkinCluster，按参考实现的 skin_weight 写权重。"""
    skin = mc.deformer(shape, type="optimizedSkinCluster", name="cmp_skin#")[0]
    for k, joint in enumerate((prev, elbow)):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, (station, _, _) in enumerate(bind):
        w = skin_weight(station)
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, vertex),
                   1.0 - w, w, size=2)
    mc.setAttr(skin + ".algorithmType", algorithm)
    return skin


def _read_points(om, shape):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    return [(p.x, p.y, p.z) for p in mesh.getPoints(om.MSpace.kObject)]


def _dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def _stats(got, want):
    if not got or len(got) != len(want):
        return float("inf"), float("inf")
    errors = [_dist(got[i], want[i]) for i in range(len(got))]
    return math.sqrt(sum(e * e for e in errors) / len(errors)), max(errors)


def _build_scene(mc, om, group, bind, faces):
    """建好三格，返回一个可直接驱动的上下文。"""
    cells = {}

    # ---- 左：Python 参考。网格直接由 Elbow 写点。----
    ref_geo, ref_shape = _make_mesh(mc, om, group, "cmp_Reference", bind, faces)
    ref_prev, ref_elbow, _ = _make_chain(mc, group, "cmp_Reference", bind)
    cells["reference"] = {"geo": ref_geo, "shape": ref_shape,
                          "prev": ref_prev, "elbow": ref_elbow}

    # ---- 中：插件 OCD (3) ----
    ocd_geo, ocd_shape = _make_mesh(mc, om, group, "cmp_OCD3", bind, faces)
    ocd_prev, ocd_elbow, _ = _make_chain(mc, group, "cmp_OCD3", bind)
    ocd_skin = _attach_skin(mc, ocd_shape, ocd_prev, ocd_elbow, 3, bind)
    ocd_result = mc.listRelatives(ocd_geo, shapes=True,
                                  noIntermediate=True, fullPath=True)[0]
    cells["ocd"] = {"geo": ocd_geo, "shape": ocd_shape, "skin": ocd_skin,
                    "result": ocd_result, "prev": ocd_prev, "elbow": ocd_elbow}

    # ---- 右：插件 OCD+CorrectedCoR (5) ----
    cor_geo, cor_shape = _make_mesh(mc, om, group, "cmp_OCDCoR5", bind, faces)
    cor_prev, cor_elbow, _ = _make_chain(mc, group, "cmp_OCDCoR5", bind)
    cor_skin = _attach_skin(mc, cor_shape, cor_prev, cor_elbow, 5, bind)
    cor_result = mc.listRelatives(cor_geo, shapes=True,
                                  noIntermediate=True, fullPath=True)[0]
    cells["cor"] = {"geo": cor_geo, "shape": cor_shape, "skin": cor_skin,
                    "result": cor_result, "prev": cor_prev, "elbow": cor_elbow}

    # ---- 显示代理：把每格结果平移到自己的网格位置 ----
    for index, key in enumerate(("reference", "ocd", "cor")):
        cell = cells[key]
        display = mc.createNode("transform", name=key + "_display#", parent=group)
        mc.setAttr(display + ".translateX", _linear((index - 1) * COLUMN_OFFSET))
        proxy = mc.createNode("mesh", name=key + "_displayShape#", parent=display)
        if key == "reference":
            mc.connectAttr(cell["shape"] + ".outMesh", proxy + ".inMesh")
        else:
            mc.connectAttr(cell["result"] + ".outMesh", proxy + ".inMesh")
        mc.sets(proxy, edit=True, forceElement="initialShadingGroup")
        cell["display"] = display

    # ---- 三格共用同一组关节旋转 ----
    driver = cells["reference"]["elbow"]
    for key in ("ocd", "cor"):
        mc.connectAttr(driver + ".rotate", cells[key]["elbow"] + ".rotate")

    return cells, driver


def show():
    """打开对比窗口。"""
    import maya.cmds as mc
    import maya.api.OpenMaya as om

    if mc.window(WIN, exists=True):
        mc.deleteUI(WIN, window=True)

    if "optimizedSkinCluster" not in (mc.allNodeTypes() or []):
        try:
            mc.loadPlugin("optimizedSkinningPlugin", quiet=True)
        except RuntimeError as exc:
            raise RuntimeError(
                "请先在插件管理器加载 optimizedSkinningPlugin.mll，再打开对比窗口。") from exc

    bind, faces = cylinder()
    selection = mc.ls(selection=True, long=True) or []
    win = mc.window(WIN, title="OCD 参考 vs 插件  |  逐顶点对比",
                    widthHeight=(1280, 760))
    group = mc.createNode("transform", name="OCD_compare#")
    group_id = mc.ls(group, uuid=True)[0]
    history = [(group, group_id)]

    def clean(*_):
        for name, uid in history:
            if mc.objExists(name) and mc.ls(name, uuid=True) == [uid]:
                mc.delete(name)

    mc.scriptJob(uiDeleted=[win, clean], runOnce=True)

    try:
        cells, driver = _build_scene(mc, om, group, bind, faces)
        history.extend((cells[key]["skin"], mc.ls(cells[key]["skin"], uuid=True)[0])
                       for key in ("ocd", "cor"))

        ref_sel = om.MSelectionList()
        ref_sel.add(cells["reference"]["shape"])
        ref_mesh = om.MFnMesh(ref_sel.getDagPath(0))
        driver_sel = om.MSelectionList()
        driver_sel.add(driver)
        driver_xf = om.MFnTransform(driver_sel.getDagPath(0))

        # ---------------- 视图 ----------------
        cam_xf, cam = mc.camera(name="OCD_compareCamera#", orthographic=True)
        cam_xf = mc.parent(cam_xf, group)[0]
        panes = mc.paneLayout(parent=win, configuration="vertical2",
                              paneSize=[(1, 68, 100), (2, 32, 100)])
        panel = mc.modelPanel(parent=panes, label="OCD compare", camera=cam_xf,
                              menuBarVisible=False)
        editor = mc.modelPanel(panel, query=True, modelEditor=True)
        mc.modelEditor(editor, e=True, displayAppearance="smoothShaded",
                       grid=False, displayTextures=False,
                       wireframeOnShaded=True, displayLights="default")
        mc.select(clear=True)
        mc.isolateSelect(panel, state=True)
        for key in ("reference", "ocd", "cor"):
            mc.isolateSelect(panel, addDagObject=cells[key]["display"])
        # 骨骼隐藏，只看皮肤结果。
        for key in ("reference", "ocd", "cor"):
            for joint in ("prev", "elbow"):
                mc.setAttr(cells[key][joint] + ".visibility", False)

        # ---------------- 控制面板 ----------------
        scroll = mc.scrollLayout(parent=panes, childResizable=True)
        col = mc.columnLayout(parent=scroll, adjustableColumn=True, rowSpacing=3)
        mc.text(parent=col,
                label="左：Python 参考　　中：插件 OCD (3)　　右：插件 OCD+CoR (5)\n"
                      "三格共用同一根驱动骨骼与同一份绑定网格；求值均在原点。")
        report = mc.text(parent=col, label="", align="left")

        handles = []

        def slider(label, low, high, value):
            return mc.floatSliderGrp(parent=col, label=label, field=True,
                                     minValue=low, maxValue=high,
                                     fieldMinValue=low, fieldMaxValue=high,
                                     value=value, precision=2,
                                     columnWidth3=(120, 52, 100),
                                     adjustableColumn=3,
                                     dragCommand=lambda *_: redraw(),
                                     changeCommand=lambda *_: redraw())

        def set_values(items, values):
            for item, value in zip(items, values):
                mc.floatSliderGrp(item, e=True, value=value)

        def get_values(items):
            return [mc.floatSliderGrp(item, q=True, value=True) for item in items]

        def apply_channels(values, overlap):
            for key in ("ocd", "cor"):
                skin = cells[key]["skin"]
                mc.setAttr(skin + ".ocdAvoidOverlap", overlap)
                for attr, value in zip(CHANNEL_ATTRS, values):
                    mc.setAttr(skin + "." + attr, value)

        motion = [slider("弯曲", -150, 150, 118),
                  slider("扭转", -180, 180, 0),
                  slider("弯曲方向", -180, 180, 0)]
        avoid = mc.checkBox(parent=col, label="旧版内侧避让", value=False,
                            changeCommand=lambda *_: redraw())

        last_rotation = [None]

        def redraw(*_):
            bend, twist, direction = get_values(motion)
            values = get_values(handles)
            overlap = mc.checkBox(avoid, q=True, value=True)

            # 参考形状：Python 求值后写回参考网格。
            pose = Elbow(bend, twist, direction, values, overlap)
            ref_mesh.setPoints([om.MPoint(*p) for p in pose.mesh(bind)],
                               om.MSpace.kObject)

            # 驱动骨骼：与参考姿态一致。
            quat = joint_quaternion(bend, twist, direction)
            if quat != last_rotation[0]:
                last_rotation[0] = quat
                driver_xf.setRotation(om.MQuaternion(*quat), om.MSpace.kTransform)

            apply_channels(values, overlap)
            mc.refresh(force=True)

            target = _read_points(om, cells["reference"]["shape"])
            ocd_rms, ocd_max = _stats(_read_points(om, cells["ocd"]["result"]), target)
            cor_rms, cor_max = _stats(_read_points(om, cells["cor"]["result"]), target)
            mc.text(report, e=True, label=(
                "弯曲 %6.1f°  扭转 %6.1f°  方向 %6.1f°\n"
                "OCD (3)      RMS %8.6f   最大 %8.6f\n"
                "OCD+CoR (5)  RMS %8.6f   最大 %8.6f"
                % (bend, twist, direction, ocd_rms, ocd_max, cor_rms, cor_max)))

        def set_pose(bend, twist=0.0, direction=0.0):
            set_values(motion, (bend, twist, direction))
            redraw()

        row = mc.rowLayout(parent=col, numberOfColumns=5)
        for name, args in (("伸直", (0,)), ("90°", (90,)), ("片头", (118,)),
                           ("深弯", (145,)), ("弯＋扭", (105, 70, 35))):
            mc.button(parent=row, label=name,
                      command=lambda *_, a=args: set_pose(*a))

        def camera_pose(yaw, elev, center, depth, width):
            rotation = om.MEulerRotation(-math.radians(elev),
                                         math.radians(yaw), 0.0)
            back = om.MVector(0, 0, depth).rotateBy(rotation.asQuaternion())
            mc.xform(cam_xf, objectSpace=True, translation=[_linear(v) for v in
                     (center[0] + back.x, center[1] + back.y, center[2] + back.z)])
            cam_sel = om.MSelectionList()
            cam_sel.add(cam_xf)
            om.MFnTransform(cam_sel.getDagPath(0)).setRotation(
                rotation, om.MSpace.kTransform)
            mc.setAttr(cam + ".orthographic", True)
            mc.setAttr(cam + ".orthographicWidth", _linear(width))
            mc.setAttr(cam + ".centerOfInterest", _linear(depth))

        def frame_view(*_):
            oblique = mc.checkBox(sideview, q=True, value=True)
            yaw, elev = (-22, 12) if oblique else (0, 0)
            if mc.checkBox(closeup, q=True, value=True):
                camera_pose(yaw, elev, (0, -8, 0), 42, 52)
            else:
                camera_pose(yaw, elev, (0, 0, 0), 90, 76)

        row = mc.rowLayout(parent=col, numberOfColumns=3)
        closeup = mc.checkBox(parent=row, label="肘部特写", value=True,
                              changeCommand=lambda *_: frame_view())
        sideview = mc.checkBox(parent=row, label="斜视", value=False,
                               changeCommand=lambda *_: frame_view())
        mc.checkBox(parent=row, label="线框", value=True,
                    changeCommand=lambda value: mc.modelEditor(
                        editor, e=True, wireframeOnShaded=value))

        mc.separator(parent=col, height=6, style="in")
        mc.text(parent=col, label="补偿通道（与参考脚本同名同序）")
        handles.extend(slider(name, -1.0, 1.0, 0.0) for name in CHANNELS)

        row = mc.rowLayout(parent=col, numberOfColumns=2)
        mc.button(parent=row, label="全部归零",
                  command=lambda *_: (set_values(handles, ZERO), redraw()))
        mc.button(parent=row, label="视频参数",
                  command=lambda *_: (set_values(handles, VIDEO), redraw()))
        mc.text(parent=col, label="关闭窗口会清理本次预览创建的所有节点。")

        frame_view()
        redraw()
        mc.scriptJob(parent=win, attributeChange=[driver + ".rotate",
                                                  lambda *_: redraw()],
                     allChildren=True)
        mc.showWindow(win)
        frame_view()
    except Exception:
        if mc.window(WIN, exists=True):
            mc.deleteUI(WIN, window=True)
        raise
    finally:
        mc.select(clear=True)
        kept = [node for node in selection if mc.objExists(node)]
        if kept:
            mc.select(kept, replace=True)
    return win


# ----------------------------------------------------------------------
# 无界面数值检查
# ----------------------------------------------------------------------

def check():
    """无界面跑一遍，打印各弯曲角下插件与参考的逐顶点误差。"""
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind, faces = cylinder()
        group = mc.createNode("transform", name="OCD_compareCheck#")
        cells, driver = _build_scene(mc, om, group, bind, faces)

        ref_sel = om.MSelectionList()
        ref_sel.add(cells["reference"]["shape"])
        ref_mesh = om.MFnMesh(ref_sel.getDagPath(0))
        driver_sel = om.MSelectionList()
        driver_sel.add(driver)
        driver_xf = om.MFnTransform(driver_sel.getDagPath(0))

        failures = 0
        for bend in (0, 60, 90, 118, 145):
            pose = Elbow(bend)
            ref_mesh.setPoints([om.MPoint(*p) for p in pose.mesh(bind)],
                               om.MSpace.kObject)
            driver_xf.setRotation(
                om.MQuaternion(*joint_quaternion(bend, 0, 0)),
                om.MSpace.kTransform)
            mc.refresh(force=True)

            target = _read_points(om, cells["reference"]["shape"])
            ocd_rms, ocd_max = _stats(_read_points(om, cells["ocd"]["result"]), target)
            cor_rms, cor_max = _stats(_read_points(om, cells["cor"]["result"]), target)
            print("bend=%3d  OCD rms=%.6f max=%.6f | OCD+CoR rms=%.6f max=%.6f"
                  % (bend, ocd_rms, ocd_max, cor_rms, cor_max))
            if ocd_rms > 1e-3:
                print("  FAIL: OCD deviates from the reference")
                failures += 1

        mc.delete(group)
        print("FAILURES:", failures)
        return 0 if failures == 0 else 1
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    if "--check" in sys.argv:
        sys.exit(check())
    else:
        show()
