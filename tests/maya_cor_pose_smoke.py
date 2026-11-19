"""Run smoke_test() in Maya 2024 after loading optimizedSkinningPlugin."""

import maya.api.OpenMaya as om
import maya.cmds as cmds


def smoke_test(check_ddm=False):
    """Check rest and rotated output through Maya with sparse weight and matrix indices."""
    nodes = []
    selection = cmds.ls(selection=True, long=True)
    try:
        mesh = cmds.polyPlane(name="corSmokeMesh#", subdivisionsX=1, subdivisionsY=1,
                              constructionHistory=False)[0]
        nodes.append(mesh)
        for name in ("corSmokeFixed#", "corSmokeDriver#"):
            nodes.append(cmds.createNode("joint", name=name, skipSelect=True))
        fixed, driver = nodes[1:]
        skin = cmds.deformer(mesh, type="optimizedSkinCluster")[0]
        nodes.append(skin)
        cmds.setAttr(skin + ".algorithmType", 0)
        cmds.setAttr(skin + ".corDebugLog", True)

        def matrix_values(path):
            values = cmds.getAttr(path)
            return values[0] if len(values) == 1 else values

        for index, joint in ((2, fixed), (9, driver)):
            cmds.connectAttr(joint + ".worldMatrix[0]", "{}.matrix[{}]".format(skin, index))
            bind = matrix_values(joint + ".worldInverseMatrix[0]")
            cmds.setAttr("{}.bindPreMatrix[{}]".format(skin, index), *bind, type="matrix")

        shape = cmds.listRelatives(mesh, shapes=True, noIntermediate=True, fullPath=True)[0]
        selection_list = om.MSelectionList()
        selection_list.add(shape)
        mesh_fn = om.MFnMesh(selection_list.getDagPath(0))
        rest = mesh_fn.getPoints(om.MSpace.kObject)
        for vertex, influence, weight in ((0, 2, 1.0), (2, 2, 0.5), (2, 9, 0.5), (3, 9, 1.0)):
            cmds.setAttr("{}.weightList[{}].weights[{}]".format(skin, vertex, influence), weight)

        def assert_point(actual, expected, label):
            error = (actual - expected).length()
            if error > 1e-5:
                raise AssertionError("{} error = {:.8g}".format(label, error))

        bound = mesh_fn.getPoints(om.MSpace.kObject)
        for vertex in range(len(rest)):
            assert_point(bound[vertex], rest[vertex], "bind vertex {}".format(vertex))

        cmds.setAttr(driver + ".rotateZ", 90)
        posed = mesh_fn.getPoints(om.MSpace.kObject)
        assert_point(posed[0], rest[0], "fixed influence")
        assert_point(posed[1], rest[1], "unweighted vertex")
        expected = rest[3] * om.MMatrix(matrix_values(driver + ".worldMatrix[0]"))
        assert_point(posed[3], expected, "rotating influence")
        assert (posed[3] - rest[3]).length() > 0.1
        assert (posed[2] - rest[2]).length() > 0.1

        if check_ddm:
            cmds.setAttr(skin + ".algorithmType", 6)
            cmds.setAttr(skin + ".ddmDebugLog", True)
            ddm = mesh_fn.getPoints(om.MSpace.kObject)
            assert_point(ddm[1], rest[1], "DDM unweighted vertex")
            assert (ddm[3] - rest[3]).length() > 0.1
            assert (ddm[2] - posed[2]).length() > 1e-3

            cmds.setAttr(skin + ".ddmIterations", 0)
            unsmoothed = mesh_fn.getPoints(om.MSpace.kObject)
            assert_point(unsmoothed[3], expected, "DDM zero-iteration rigid influence")
            cmds.setAttr(skin + ".ddmIterations", 20)

        cmds.setAttr(driver + ".rotateZ", 0)
        reset = mesh_fn.getPoints(om.MSpace.kObject)
        for vertex in range(len(rest)):
            assert_point(reset[vertex], rest[vertex], "reset vertex {}".format(vertex))
        print("[DDM v4] Maya smoke test passed" if check_ddm
              else "[CoR] Maya sparse weight/pose smoke test passed")
    finally:
        for node in reversed(nodes):
            if cmds.objExists(node):
                cmds.delete(node)
        if selection:
            cmds.select(selection)
        else:
            cmds.select(clear=True)
