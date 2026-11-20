"""Run smoke_test() in Maya 2024 after loading optimizedSkinningPlugin."""

import math

import maya.api.OpenMaya as om
import maya.cmds as cmds


def smoke_test():
    """Exercise the new curve mode, sparse influence IDs and live channel changes."""
    nodes = []
    selected = cmds.ls(selection=True, long=True)
    try:
        mesh = cmds.polyCylinder(name="mosSmokeMesh#", axis=(1, 0, 0),
                                 radius=1, height=8, subdivisionsX=24,
                                 subdivisionsY=24, constructionHistory=False)[0]
        nodes.append(mesh)
        root = cmds.createNode("joint", name="mosSmokeUpper#", skipSelect=True)
        nodes.append(root)
        elbow = cmds.createNode("joint", name="mosSmokeLower#", parent=root, skipSelect=True)
        cmds.setAttr(root + ".translateX", -4)
        cmds.setAttr(elbow + ".translateX", 4)

        shape = cmds.listRelatives(mesh, shapes=True, noIntermediate=True, fullPath=True)[0]
        selection = om.MSelectionList()
        selection.add(shape)
        mesh_fn = om.MFnMesh(selection.getDagPath(0))
        rest = mesh_fn.getPoints(om.MSpace.kObject)

        skin = cmds.deformer(mesh, type="optimizedSkinCluster")[0]
        nodes.append(skin)
        cmds.setAttr(skin + ".algorithmType", 3)
        cmds.setAttr(skin + ".mosCurveMode", 4)
        for index, joint in ((2, root), (9, elbow)):
            cmds.connectAttr(joint + ".worldMatrix[0]", f"{skin}.matrix[{index}]")
            values = cmds.getAttr(joint + ".worldInverseMatrix[0]")
            values = values[0] if len(values) == 1 else values
            cmds.setAttr(f"{skin}.bindPreMatrix[{index}]", *values, type="matrix")

        for index, point in enumerate(rest):
            child_weight = min(1.0, max(0.0, (point.x + 2.0) / 4.0))
            cmds.setAttr(f"{skin}.weightList[{index}].weights[2]", 1.0 - child_weight)
            cmds.setAttr(f"{skin}.weightList[{index}].weights[9]", child_weight)

        def points():
            return mesh_fn.getPoints(om.MSpace.kObject)

        def max_delta(a, b):
            return max((left - right).length() for left, right in zip(a, b))

        assert max_delta(points(), rest) < 1e-5, "bind must reconstruct the input mesh"

        for angle in (45, 90, 140):
            cmds.setAttr(elbow + ".rotateZ", angle)
            base = points()
            assert all(math.isfinite(v) for p in base for v in (p.x, p.y, p.z))
            cmds.setAttr(skin + ".slideInWeight_prev", 0.2)
            assert max_delta(points(), base) > 1e-3, "Slide should reshape the curve"
            cmds.setAttr(skin + ".slideInWeight_prev", 0)
            cmds.setAttr(skin + ".slideOrthoInWeight_prev", 0.2)
            assert max_delta(points(), base) > 1e-3, "Slide Ortho should reshape the curve"
            cmds.setAttr(skin + ".slideOrthoInWeight_prev", 0)
            assert max_delta(points(), base) < 1e-5, "reset must reproduce base MOS"

        cmds.setAttr(skin + ".mosInfluenceRadius", 0)
        lbs = points()
        cmds.setAttr(skin + ".mosInfluenceRadius", 1)
        assert max_delta(lbs, points()) > 1e-3, "curve sampling should differ from the affine LBS blend"
        cmds.setAttr(elbow + ".rotateZ", 0)
        assert max_delta(points(), rest) < 1e-5, "reset must reproduce bind"
        print("[MOS Curve Sampling] Maya smoke test passed")
    finally:
        for node in reversed(nodes):
            if cmds.objExists(node):
                cmds.delete(node)
        if selected:
            cmds.select(selected)
        else:
            cmds.select(clear=True)
