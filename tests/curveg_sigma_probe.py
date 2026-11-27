# -*- coding: utf-8 -*-
"""确认 C 组的弧长项是否真的生效：改 sigma 看结果是否变化。

如果 C 对 sigma 完全不敏感，说明弧长项没接上（实现 bug）；
如果敏感，说明是这个夹具里弧长差太小（物理原因）。

    mayapy tests/curveg_sigma_probe.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import curveg_chain_ab as ab  # noqa: E402
import ocd_elbow_reference_latest as ref  # noqa: E402


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind, shape, skin, joints, flat = ab.build(mc, om)
        sides, rings = 48, 41
        for tag, root, tip, ang in (("A", joints[0][0], joints[0][1], +ab.BEND),
                                    ("B", joints[1][0], joints[1][1], -ab.BEND)):
            s = om.MSelectionList()
            s.add(tip)
            om.MFnTransform(s.getDagPath(0)).setRotation(
                om.MQuaternion(*ref.joint_quaternion(ang, 0, 0)),
                om.MSpace.kTransform)
        mc.setAttr(skin + ".algorithmType", 0)
        mc.setAttr(skin + ".corSigma", 0.1)
        mc.setAttr(skin + ".corCurveGate", 0.0)
        mc.refresh(force=True)
        PA = ab.read(mc, om, shape)

        print("C 组（归属 + 同链弧长）对 sigma 的敏感度：")
        print("%8s %14s %14s" % ("sigma", "max|dv|", "mean|dv|"))
        base = None
        for sigma in (0.02, 0.05, 0.1, 0.3, 1.0, 3.0):
            mc.setAttr(skin + ".corCurveGate", 1.0)
            mc.setAttr(skin + ".corCurveCrossOnly", False)
            mc.setAttr(skin + ".corCurveSigma", sigma)
            mc.refresh(force=True)
            P = ab.read(mc, om, shape)
            d = np.linalg.norm(P - PA, axis=1)
            print("%8.2f %14.6e %14.6e" % (sigma, d.max(), d.mean()))
            if base is None:
                base = P.copy()
            else:
                dd = np.linalg.norm(P - base, axis=1)
                if dd.max() > 1e-9:
                    print("         ^^ 与 sigma=0.02 的差 max %.3e" % dd.max())
                    base = P.copy()

        print("\nB 组（只衰减跨链）对 sigma 的敏感度（应当完全不敏感）：")
        for sigma in (0.02, 0.1, 3.0):
            mc.setAttr(skin + ".corCurveGate", 1.0)
            mc.setAttr(skin + ".corCurveCrossOnly", True)
            mc.setAttr(skin + ".corCurveSigma", sigma)
            mc.refresh(force=True)
            P = ab.read(mc, om, shape)
            d = np.linalg.norm(P - PA, axis=1)
            print("  sigma=%-5.2f max|dv| %.6e" % (sigma, d.max()))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
