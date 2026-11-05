"""
test_cor_viewer.py

Build a simple scene (cylinder + 3 joints + optimizedSkinCluster),
create the CoR viewer node, and turn on displayCoR.

Run in Maya Script Editor:
    exec(open(r"D:/visualStudioProject/optimizedSkinningPlugin/scripts/test_cor_viewer.py", encoding="utf-8").read())
    test_cor_viewer()
"""

import maya.cmds as cmds


def test_cor_viewer():
    print("=" * 60)
    print("CoR Viewer Test")
    print("=" * 60)

    # ------------------------------------------------------------
    # 1. New scene
    # ------------------------------------------------------------
    cmds.file(new=True, force=True)
    print("[1] New scene")

    # ------------------------------------------------------------
    # 2. Cylinder mesh
    # ------------------------------------------------------------
    cyl = cmds.polyCylinder(r=0.5, h=10, sx=20, sy=80, ch=1)[0]
    mesh = cmds.listRelatives(cyl, s=True)[0]
    vtx_count = cmds.polyEvaluate(mesh, v=True)
    print("[2] Mesh: %s  (%d verts)" % (mesh, vtx_count))

    # ------------------------------------------------------------
    # 3. Joints
    # ------------------------------------------------------------
    cmds.select(clear=True)
    cmds.joint(p=(0, 5, 0), n="j_root")
    cmds.joint(p=(0, 0, 0), n="j_mid")
    cmds.joint(p=(0, -5, 0), n="j_tip")
    joints = ["j_root", "j_mid", "j_tip"]
    print("[3] Joints: %s" % joints)

    # ------------------------------------------------------------
    # 4. Create optimizedSkinCluster
    # ------------------------------------------------------------
    cmds.select(mesh)
    try:
        node = cmds.deformer(type="optimizedSkinCluster")[0]
    except Exception as e:
        print("!!! Failed to create optimizedSkinCluster: %s" % e)
        print("    Is the plugin loaded?")
        return
    print("[4] Node: %s" % node)

    # ------------------------------------------------------------
    # 5. Connect joints
    # ------------------------------------------------------------
    for i, j in enumerate(joints):
        if not cmds.objExists(j + ".lockInfluenceWeights"):
            cmds.addAttr(j, sn="liw", ln="lockInfluenceWeights", at="bool")
        cmds.connectAttr(j + ".liw",
                         node + ".lockWeights[%d]" % i, force=True)
        cmds.connectAttr(j + ".worldMatrix[0]",
                         node + ".matrix[%d]" % i, force=True)
        m = cmds.getAttr(j + ".wim")
        cmds.setAttr(node + ".bindPreMatrix[%d]" % i,
                     m[0], m[1], m[2], m[3],
                     m[4], m[5], m[6], m[7],
                     m[8], m[9], m[10], m[11],
                     m[12], m[13], m[14], m[15],
                     type="matrix")
    print("[5] Joints connected")

    # ------------------------------------------------------------
    # 6. Assign weights by Y
    # ------------------------------------------------------------
    ys = []
    for v in range(vtx_count):
        p = cmds.xform(mesh + ".vtx[%d]" % v, q=True, os=True, t=True)
        ys.append(p[1])
    y_min, y_max = min(ys), max(ys)
    y_rng = y_max - y_min or 1.0

    for v in range(vtx_count):
        for k in range(3):
            cmds.setAttr(node + ".weightList[%d].weights[%d]" % (v, k), 0.0)
        t = (ys[v] - y_min) / y_rng
        b = 0 if t > 0.66 else (2 if t < 0.33 else 1)
        cmds.setAttr(node + ".weightList[%d].weights[%d]" % (v, b), 1.0)
    print("[6] Weights assigned")

    # ------------------------------------------------------------
    # 7. Set algorithm = CoR
    # ------------------------------------------------------------
    cmds.setAttr(node + ".algorithmType", 0)
    print("[7] algorithmType = CoR (0)")

    # ------------------------------------------------------------
    # 8. Trigger CoR computation
    # ------------------------------------------------------------
    cmds.refresh(force=True)
    print("[8] CoR computed")

    # ------------------------------------------------------------
    # 9. Create viewer node
    # ------------------------------------------------------------
    viewer_name = node + "_corViewer"
    if cmds.objExists(viewer_name):
        cmds.delete(viewer_name)
    try:
        cmds.optimizedSkinClusterCoR(node=node, create=True)
    except Exception as e:
        print("!!! create viewer failed: %s" % e)
        return
    print("[9] Viewer created: %s" % viewer_name)

    # ------------------------------------------------------------
    # 10. Turn on displayCoR
    # ------------------------------------------------------------
    cmds.setAttr(node + ".displayCoR", 1)
    print("[10] displayCoR = 1")

    # ------------------------------------------------------------
    # 11. Frame the view
    # ------------------------------------------------------------
    cmds.select(mesh)
    cmds.viewFit()

    # ------------------------------------------------------------
    # 12. Report
    # ------------------------------------------------------------
    print("=" * 60)
    print("DONE")
    print("=" * 60)
    print("Check the viewport for red dots.")
    print("")
    print("If nothing shows up:")
    print("  - Try pointSize = 10:")
    print("      cmds.setAttr('%s.pointSize', 10)" % viewer_name)
    print("  - Verify CoR data:")
    print("      r = cmds.optimizedSkinClusterCoR(node='%s')" % node)
    print("      print(len(r) // 3)")
    print("  - Confirm viewport is VP2 (or try legacy):")
    print("      cmds.optionVar(sv=('vp2RenderingEngine', 'opengl'))")
    print("      # then restart Maya")


def pose_test():
    """Rotate the middle joint to see CoR follow the pose."""
    cmds.setAttr("j_mid.rotateZ", 45)
    cmds.refresh(force=True)
    print("Pose set: j_mid.rotateZ = 45")

    # If CoR viewer's point size too small, bump it up
    v = "optimizedSkinCluster1_corViewer"
    if cmds.objExists(v):
        cmds.setAttr(v + ".pointSize", 8)
        print("pointSize set to 8")


def turn_off():
    """Hide CoR."""
    cmds.setAttr("optimizedSkinCluster1.displayCoR", 0)
    print("displayCoR = 0")


def turn_on():
    """Show CoR."""
    cmds.setAttr("optimizedSkinCluster1.displayCoR", 1)
    print("displayCoR = 1")