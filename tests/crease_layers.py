# -*- coding: utf-8 -*-
r"""把三层折角分开量清楚，别混在一起。

层 1  形状折角 s=0        : hinge_section 内轨自带（单侧导数 176.10°）
层 2  求解器病态 |s|≈2.2  : A_i 弱区增益 0.00249
层 3  限幅撕裂 2.2..2.6   : 714 顶点被截到 16.0

做一个「只关掉层 1」的对照：给 hinge_section 加同样的 Hermite 圆角，
再看剩下的折角是多少。这样能确认层 2/3 的净贡献。

    mayapy tests/crease_layers.py
"""
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

BEND = 146.1265
SIDES, RINGS, HALF = 80, 201, 6.0
SKIN = "optimizedSkinCluster"


def smoothstep(t):
    t = max(0.0, min(1.0, t))
    return t * t * (3.0 - 2.0 * t)


def hinge_section_filleted(station, depth, radius, weights, angle, fillet=0.0):
    """hinge_section + 在 station=0 的 Hermite 圆角（与 C++ 的实现一致）。"""
    if fillet <= 0.0 or abs(station) >= fillet or angle <= 1e-8:
        return ref.hinge_section(np.atleast_1d(station), np.atleast_1d(depth),
                                 radius, weights, angle)[0]

    def raw(s):
        return ref.hinge_section(np.array([s]), np.array([depth]), radius,
                                 weights, angle)[0]

    w = fillet
    left, right = np.asarray(raw(-w), float), np.asarray(raw(w), float)
    h = min(1e-4, w * 1e-3)
    l0, l1 = np.asarray(raw(-w - h), float), np.asarray(raw(-w + h), float)
    r0, r1 = np.asarray(raw(w - h), float), np.asarray(raw(w + h), float)
    t0 = l1 - l0
    t1 = r1 - r0
    n0, n1 = np.linalg.norm(t0), np.linalg.norm(t1)
    chord = np.linalg.norm(right - left)
    if n0 < 1e-12 or n1 < 1e-12 or chord < 1e-12:
        return ref.hinge_section(np.atleast_1d(station),
                                 np.atleast_1d(depth), radius, weights, angle)[0]
    t0 /= n0
    t1 /= n1
    u = smoothstep((station + w) / (2.0 * w))
    h00 = 2 * u ** 3 - 3 * u ** 2 + 1
    h10 = u ** 3 - 2 * u ** 2 + u
    h01 = -2 * u ** 3 + 3 * u ** 2
    h11 = u ** 3 - u ** 2
    return h00 * left + h10 * (chord * t0) + h01 * right + h11 * (chord * t1)


def guide_with_fillet(bind, original, W, fillet):
    """复刻 hinge_displacement，但用带圆角的 hinge_section。"""
    station, y, z = bind.T
    plane = 0.0
    phase = 0.0
    depth = y * np.cos(phase - plane) - z * np.sin(phase - plane)
    lateral = y * np.sin(phase - plane) + z * np.cos(phase - plane)
    radius = float(np.max(np.hypot(y, z)))
    angle = math.radians(abs(BEND))
    ext = np.full_like(station, radius)
    inner = np.array([hinge_section_filleted(s, +e, radius, W[i:i + 1], angle, fillet)
                      for i, (s, e) in enumerate(zip(station, ext))])
    outer = np.array([hinge_section_filleted(s, -e, radius, W[i:i + 1], angle, fillet)
                      for i, (s, e) in enumerate(zip(station, ext))])
    neutral = (inner + outer) * 0.5
    gradient = (inner - outer) / (2.0 * radius)
    x = neutral[:, 0] + depth * gradient[:, 0]
    h = neutral[:, 1] + depth * gradient[:, 1]
    target = np.column_stack([x,
                              h * np.cos(plane) - lateral * np.sin(plane),
                              h * np.sin(plane) + lateral * np.cos(plane)])
    return target - original


def signed_turning(pts):
    u = pts[1:-1] - pts[:-2]
    v = pts[2:] - pts[1:-1]
    cross = u[:, 0] * v[:, 1] - u[:, 1] * v[:, 0]
    dot = u[:, 0] * v[:, 0] + u[:, 1] * v[:, 1]
    return np.degrees(np.arctan2(cross, dot))


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        def tube(sides, rings, half):
            pts, faces = [], []
            for row in range(rings):
                s = -half + row * (2 * half / (rings - 1))
                for j in range(sides):
                    a = 2 * math.pi * j / sides
                    pts.append((s, math.cos(a), math.sin(a)))
            for row in range(rings - 1):
                for j in range(sides):
                    p = row * sides + j
                    q = row * sides + (j + 1) % sides
                    faces.append((p, q, q + sides, p + sides))
            return pts, faces

        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        bind_list, faces = tube(SIDES, RINGS, HALF)
        bind = np.asarray(bind_list, float)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [j for f in faces for j in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="pv", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="el", parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type=SKIN, name="sk#")[0]
        for idx, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, idx))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, idx),
                       *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
        w1 = np.array([ref.skin_weight(p[0]) for p in bind])
        for i in range(len(bind)):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w1[i], w1[i], size=2)
        s2 = om.MSelectionList()
        s2.add(elbow)
        om.MFnTransform(s2.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
        sl = om.MSelectionList()
        sl.add(shape)
        mc.setAttr(skin + ".algorithmType", 0)
        mc.refresh(force=True)
        for _ in range(2):
            mc.dgeval(shape)
            mc.refresh(force=True)
        original = np.asarray([[p.x, p.y, p.z] for p in om.MFnMesh(
            sl.getDagPath(0)).getPoints(om.MSpace.kObject)])
        mc.delete(holder)

        W = np.column_stack([1.0 - w1, w1])
        quats = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(BEND, 0, 0)]
        edges = ref.local_edges(bind_list, faces)
        outer = np.asarray(ref.outer_participation(bind, W, BEND, 0.0, 0.0))
        stations = np.array([-HALF + r * (2 * HALF / (RINGS - 1))
                             for r in range(RINGS)])
        ring_idx = [r * SIDES for r in range(RINGS)]

        print("%-28s %12s %12s %12s" % ("配置", "s=0 转角", "|s|≈2.2 转角",
                                        "|delta|max"))
        # 只做无限幅版本，避免层 3 干扰
        for fillet in (0.0, 0.4, 0.8):
            guide = guide_with_fillet(bind, original, W, fillet)
            delta, _ = __import__("center_coupling").structural_centers_coupled(
                edges, original, W, quats, guide, outer, center_limit=None)
            A = ref.center_response(W, quats)
            final = (original + np.einsum("nij,nj->ni", A, delta))[ring_idx]
            turns = signed_turning(final)
            pos = stations[1:-1]
            mid = int(np.argmin(np.abs(pos - 0.0)))
            near = [i for i in range(len(pos)) if 2.0 < abs(pos[i]) < 2.4]
            t_mid = abs(turns[mid])
            t_22 = max(abs(turns[i]) for i in near) if near else float("nan")
            print("%-28s %12.2f %12.2f %12.3f"
                  % ("fillet=%.1f  无限幅" % fillet, t_mid, t_22,
                     np.linalg.norm(delta, axis=1).max()))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
