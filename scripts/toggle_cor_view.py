"""
toggle_cor_view.py

用 mesh 点云显示 CoR（不涉及 dynamics，不影响 evaluation manager）。

用法：
    import toggle_cor_view
    toggle_cor_view.toggle("optimizedSkinCluster1")
    toggle_cor_view.toggle()   # 用当前选中节点
"""

import maya.cmds as cmds
import maya.api.OpenMaya as om


def toggle(node_name=None):
    if node_name is None:
        sel = cmds.ls(selection=True)
        if not sel:
            cmds.warning("No node selected")
            return
        node_name = sel[0]

    if not cmds.objExists(node_name):
        cmds.warning("Node not found: " + node_name)
        return

    if cmds.nodeType(node_name) != "optimizedSkinCluster":
        cmds.warning("Not an optimizedSkinCluster: " + node_name)
        return

    display = cmds.getAttr(node_name + ".displayCoR")
    viewer_name = node_name + "_corViewer"

    if display:
        # 查询 CoR
        try:
            cor_flat = cmds.optimizedSkinClusterCoR(node=node_name)
        except Exception as e:
            cmds.warning("Query failed: " + str(e))
            return

        if not cor_flat:
            cmds.warning("No CoR data (cache not valid)")
            return

        # 删除旧 viewer
        if cmds.objExists(viewer_name):
            cmds.delete(viewer_name)
        if cmds.objExists(viewer_name + "Shape"):
            cmds.delete(viewer_name + "Shape")

        # 生成 mesh
        _create_mesh_points(viewer_name, cor_flat, radius=0.15)
        n = len(cor_flat) // 3
        print("CoR viewer created: %d points (%s)" % (n, viewer_name))

    else:
        if cmds.objExists(viewer_name):
            cmds.delete(viewer_name)
            print("CoR viewer deleted: " + viewer_name)
        else:
            print("CoR viewer already hidden")


def _create_mesh_points(mesh_name, flat_xyz, radius=0.15):
    """
    用 MFnMesh 批量创建小八面体点云。
    flat_xyz: [x0,y0,z0, x1,y1,z1, ...]
    radius: 八面体半径
    """
    n = len(flat_xyz) // 3
    if n == 0:
        return

    # 八面体：6 顶点、8 三角面
    r = radius
    oct_v = [
        ( r,  0,  0), (-r,  0,  0),
        ( 0,  r,  0), ( 0, -r,  0),
        ( 0,  0,  r), ( 0,  0, -r),
    ]
    oct_f = [
        (0, 2, 4), (2, 1, 4), (1, 3, 4), (3, 0, 4),
        (2, 0, 5), (1, 2, 5), (3, 1, 5), (0, 3, 5),
    ]

    # 预分配数组
    verts = om.MFloatPointArray()
    verts.setLength(n * 6)
    counts = om.MIntArray()
    counts.setLength(n * 8)
    connects = om.MIntArray()
    connects.setLength(n * 8 * 3)

    for i in range(n):
        cx = flat_xyz[3 * i]
        cy = flat_xyz[3 * i + 1]
        cz = flat_xyz[3 * i + 2]
        base_v = i * 6
        base_f = i * 8

        for j in range(6):
            dx, dy, dz = oct_v[j]
            verts[base_v + j] = om.MFloatPoint(cx + dx, cy + dy, cz + dz)

        for j in range(8):
            a, b, c = oct_f[j]
            counts[base_f + j] = 3
            conn_base = (base_f + j) * 3
            connects[conn_base + 0] = base_v + a
            connects[conn_base + 1] = base_v + b
            connects[conn_base + 2] = base_v + c

    # 创建 mesh
    mesh_fn = om.MFnMesh()
    mesh_obj = mesh_fn.create(verts, counts, connects)

    # Rename shape
    shape_dag = om.MDagPath.getAPathTo(mesh_obj)
    shape_dag.extendToShape()
    shape_fn = om.MFnDagNode(shape_dag)
    shape_fn.setName(mesh_name + "Shape")

    # Rename transform
    parent_obj = shape_fn.parent(0)
    if not parent_obj.isNull():
        parent_fn = om.MFnDagNode(parent_obj)
        parent_fn.setName(mesh_name)

    # 用 Lamina 材质（无光照，纯色）
    shader = cmds.shadingNode("lambert", asShader=True,
                              name=mesh_name + "_shader")
    sg = cmds.sets(renderable=True, noSurfaceShader=True, empty=True,
                   name=mesh_name + "_sg")
    cmds.connectAttr(shader + ".outColor", sg + ".surfaceShader", force=True)
    cmds.setAttr(shader + ".color", 1.0, 0.0, 0.0, type="double3")   # 红
    cmds.setAttr(shader + ".incandescence", 1.0, 0.0, 0.0, type="double3")
    cmds.sets(mesh_name, e=True, forceElement=sg)