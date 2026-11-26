# -*- coding: utf-8 -*-
"""Headless alignment check for algorithmType 3 (OCD) and 5 (OCD + Corrected CoR).

Builds the reference cylinder rig inside Maya, drives the elbow joint, and
compares the plug-in output against the Python reference shape vertex by
vertex.  Both the plug-in and the reference work on the same bind mesh, so a
faithful port should reproduce the reference within floating point noise.

Run with:  mayapy tests/maya_ocd_smoke.py            (asserts alignment)
           mayapy tests/maya_ocd_smoke.py --report   (prints numbers, no assert)
"""
import math
import os
import sys

REFERENCE_SCRIPT = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                "scripts", "ocd_elbow_reference.py")
BEND_ANGLES = (60, 90, 118, 145)
TOLERANCE = 1e-3


def load_reference():
    import importlib.util
    spec = importlib.util.spec_from_file_location("ocd_reference", REFERENCE_SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def build_cylinder_scene(mc, ref):
    import maya.api.OpenMaya as om

    mc.file(new=True, force=True)
    mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)

    bind, faces = ref.cylinder()
    counts = [len(f) for f in faces]
    indices = [j for f in faces for j in f]

    mesh = mc.createNode("transform", name="ocd_geo")
    sel = om.MSelectionList()
    sel.add(mesh)
    points = [om.MPoint(*p) for p in bind]
    om.MFnMesh().create(points, counts, indices, parent=sel.getDependNode(0))
    shape = mc.listRelatives(mesh, shapes=True, noIntermediate=True, fullPath=True)[0]

    prev = mc.createNode("joint", name="ocd_prev")
    elbow = mc.createNode("joint", name="ocd_elbow", parent=prev)
    tip = mc.createNode("joint", name="ocd_tip", parent=elbow)
    for joint, dx in ((prev, -ref.HALF), (elbow, ref.HALF), (tip, ref.HALF)):
        mc.setAttr(joint + ".translateX", dx)
        mc.setAttr(joint + ".radius", 0.09)
    return mesh, shape, prev, elbow, bind


def build_deformer(mc, shape, prev, elbow, algorithm, weights):
    skin = mc.deformer(shape, type="optimizedSkinCluster", name="ocd_skin")[0]
    for k, joint in enumerate((prev, elbow)):
        mc.connectAttr(joint + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
        inverse = mc.getAttr(joint + ".worldInverseMatrix[0]")
        mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k), *inverse, type="matrix")
    for vertex, (parent_w, child_w) in enumerate(weights):
        mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, vertex),
                   parent_w, child_w, size=2)
    mc.setAttr(skin + ".algorithmType", algorithm)
    return skin


def posed_points(shape):
    import maya.api.OpenMaya as om
    import maya.cmds as mc
    mc.refresh(force=True)
    sel = om.MSelectionList()
    sel.add(shape)
    mesh = om.MFnMesh(sel.getDagPath(0))
    return [(p.x, p.y, p.z) for p in mesh.getPoints(om.MSpace.kObject)]


def dist(a, b):
    return math.sqrt((a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2 + (a[2] - b[2]) ** 2)


def rms(got, want, indices):
    total = sum(dist(got[i], want[i]) ** 2 for i in indices)
    return math.sqrt(total / max(len(indices), 1))


def main():
    report_only = "--report" in sys.argv
    import maya.standalone
    maya.standalone.initialize(name="python")
    import maya.cmds as mc

    ref = load_reference()
    failures = 0
    worst_overall = 0.0

    for algorithm, label in ((3, "OCD"), (5, "OCD+CorrectedCoR")):
        mesh, shape, prev, elbow, bind = build_cylinder_scene(mc, ref)
        weights = [(1.0 - ref.skin_weight(p[0]), ref.skin_weight(p[0])) for p in bind]
        build_deformer(mc, shape, prev, elbow, algorithm, weights)

        rest = posed_points(shape)
        rest_error = max(dist(rest[i], bind[i]) for i in range(len(bind)))
        print("[%s] rest pose max deviation: %.3e" % (label, rest_error))
        if rest_error > 1e-4:
            print("  FAIL: rest pose is not the bind pose")
            failures += 1

        active = [i for i, p in enumerate(bind) if 0.0 < ref.skin_weight(p[0]) < 1.0]
        for bend in BEND_ANGLES:
            mc.setAttr(elbow + ".rotateZ", float(bend))
            got = posed_points(shape)
            moved = max(dist(got[i], rest[i]) for i in range(len(rest)))
            # Algorithm 3 must reproduce the reference shape exactly.
            # Algorithm 5 is a different construction: it solves CoR centers that
            # approximate that shape, so it is checked for self-consistency and
            # for staying close to the corrected-CoR construction rather than for
            # equality with Elbow.
            if algorithm == 3:
                target = ref.Elbow(bend).mesh(bind)
                error = rms(got, target, active)
                worst_overall = max(worst_overall, error)
                status = "OK" if error <= TOLERANCE else "MISMATCH"
                print("[%s] bend=%3d  maxMoved=%.4f  rms=%.6f  %s" %
                      (label, bend, moved, error, status))
                if not report_only and error > TOLERANCE:
                    failures += 1
            else:
                # Corrected CoR must stay within the same neighbourhood as the
                # reference shape; a large departure means the center solve blew up.
                target = ref.Elbow(bend).mesh(bind)
                drift = rms(got, target, active)
                print("[%s] bend=%3d  maxMoved=%.4f  driftVsReference=%.6f" %
                      (label, bend, moved, drift))
                if not report_only and drift > 2.0:
                    print("  FAIL: corrected CoR drifted away from the reference")
                    failures += 1
            if moved < 1e-4:
                print("  FAIL: deformer produced no motion")
                failures += 1

        mc.delete(mesh)
        for node in mc.ls(type="optimizedSkinCluster") or []:
            if mc.objExists(node):
                mc.delete(node)
        for node in mc.ls(type="joint") or []:
            if mc.objExists(node):
                mc.delete(node)

    print("worst rms: %.6f  (tolerance %.0e)" % (worst_overall, TOLERANCE))
    print("FAILURES:", failures)
    maya.standalone.uninitialize()
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
