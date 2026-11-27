# -*- coding: utf-8 -*-
r"""按 cor_paper::vertexTransform 的确切语义复算混权点。

源码（cor_paper.cpp:184-202）：
    sum = Σ w_k q_k              （带反对偶符号）
    rotation = normalize(sum)
    posedCenter = Σ w_k · apply_transform(bindToPose[k], center)
    T(x) = rotation·x + (posedCenter - rotation·center)

注意：中心是**各自变换后按权重线性混合**，旋转是**四元数归一化混合**，
两者不是同一个线性组合。

    mayapy tests/alg0_mixed.py
"""
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402

HALF, BEND = 6.0, 146.1265


def quat_from_mat(R):
    """3x3 旋转矩阵 -> 四元数 (x,y,z,w)。"""
    m = np.asarray(R, float)
    tr = m[0, 0] + m[1, 1] + m[2, 2]
    if tr > 0:
        s = np.sqrt(tr + 1.0) * 2
        w = 0.25 * s
        x = (m[2, 1] - m[1, 2]) / s
        y = (m[0, 2] - m[2, 0]) / s
        z = (m[1, 0] - m[0, 1]) / s
    elif m[0, 0] > m[1, 1] and m[0, 0] > m[2, 2]:
        s = np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2]) * 2
        w = (m[2, 1] - m[1, 2]) / s
        x = 0.25 * s
        y = (m[0, 1] + m[1, 0]) / s
        z = (m[0, 2] + m[2, 0]) / s
    elif m[1, 1] > m[2, 2]:
        s = np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2]) * 2
        w = (m[0, 2] - m[2, 0]) / s
        x = (m[0, 1] + m[1, 0]) / s
        y = 0.25 * s
        z = (m[1, 2] + m[2, 1]) / s
    else:
        s = np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1]) * 2
        w = (m[1, 0] - m[0, 1]) / s
        x = (m[0, 2] + m[2, 0]) / s
        y = (m[1, 2] + m[2, 1]) / s
        z = 0.25 * s
    return np.array([x, y, z, w])


def quat_conj(q):
    return np.array([-q[0], -q[1], -q[2], q[3]])


def quat_rotate(q, v):
    """q·v·q*（标准列向量旋转）。

    Maya 用行向量约定（v·R），而 MTransformationMatrix 取出的四元数对应
    行向量矩阵的转置，所以对顶点求值时要用共轭。
    """
    x, y, z, w = q
    R = np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
    return R @ np.asarray(v, float)


def quat_rotate_row(q, v):
    """行向量版：v·R = (Rᵀ v)，等价于用共轭的四元数做列向量旋转。"""
    return quat_rotate(quat_conj(q), v)


def main():
    import maya.standalone
    maya.standalone.initialize(name="python")
    try:
        import maya.cmds as mc
        import maya.api.OpenMaya as om

        bind = np.load(os.path.join(HERE, "_aban_lo_bind.npy"))
        posed = np.load(os.path.join(HERE, "_aban_lo_posed.npy"))
        w1 = np.load(os.path.join(HERE, "_aban_lo_w1.npy"))
        faces = []
        with open(os.path.join(HERE, "_aban_lo_faces.txt")) as fh:
            for line in fh:
                faces.append(tuple(int(x) for x in line.split()))
        W = np.column_stack([1.0 - w1, w1])

        i_mix = int(np.argmin(np.abs(w1 - 0.5)))

        # 场景取矩阵
        mc.file(new=True, force=True)
        mc.loadPlugin("optimizedSkinningPlugin.mll", quiet=True)
        holder = mc.createNode("transform", name="h")
        geo = mc.createNode("transform", name="g", parent=holder)
        sel = om.MSelectionList()
        sel.add(geo)
        om.MFnMesh().create([om.MPoint(*p) for p in bind],
                            [len(f) for f in faces],
                            [k for f in faces for k in f],
                            parent=sel.getDependNode(0))
        shape = mc.listRelatives(geo, shapes=True, noIntermediate=True,
                                 fullPath=True)[0]
        prev = mc.createNode("joint", name="pv", parent=holder)
        mc.setAttr(prev + ".translateX", -HALF)
        elbow = mc.createNode("joint", name="el", parent=prev)
        mc.setAttr(elbow + ".translateX", HALF)
        skin = mc.deformer(shape, type="optimizedSkinCluster", name="sk")[0]
        for k, jnt in enumerate((prev, elbow)):
            mc.connectAttr(jnt + ".worldMatrix[0]", "%s.matrix[%d]" % (skin, k))
            mc.setAttr("%s.bindPreMatrix[%d]" % (skin, k),
                       *mc.getAttr(jnt + ".worldInverseMatrix[0]"), type="matrix")
        for i in range(len(bind)):
            mc.setAttr("%s.weightList[%d].weights[0:1]" % (skin, i),
                       1.0 - w1[i], w1[i], size=2)
        s = om.MSelectionList()
        s.add(elbow)
        om.MFnTransform(s.getDagPath(0)).setRotation(
            om.MQuaternion(*ref.joint_quaternion(BEND, 0, 0)), om.MSpace.kTransform)
        mc.setAttr(skin + ".algorithmType", 0)
        mc.refresh(force=True)

        def m2n(name):
            return np.asarray(mc.getAttr(name), float).reshape(4, 4)

        M0 = m2n("%s.bindPreMatrix[0]" % skin) @ m2n(prev + ".worldMatrix[0]")
        M1 = m2n("%s.bindPreMatrix[1]" % skin) @ m2n(elbow + ".worldMatrix[0]")

        # 用 Maya 自己的 MTransformationMatrix 取相对变换的四元数与平移，
        # 与源码 optimizedSkinCluster.cpp:2104-2107 完全一致：
        #   relative = bindInverse * pose
        #   q = MTransformationMatrix(relative).rotation()
        #   t = (relative[3][0], relative[3][1], relative[3][2])
        def rel_quat_t(rel_np):
            """与源码 2104-2107 一致：relative = bindInverse * pose，
            取 MTransformationMatrix(relative).rotation() 作为四元数，
            平移取相对矩阵的第 4 行前三个。"""
            mmat = om.MMatrix(rel_np.flatten().tolist())
            tm = om.MTransformationMatrix(mmat)
            eu = tm.rotation()                 # MEulerRotation
            q = eu.asQuaternion()              # 与 C++ 的隐式转换等效
            return (np.array([q.x, q.y, q.z, q.w]),
                    np.array([rel_np[3, 0], rel_np[3, 1], rel_np[3, 2]]))

        q0, t0 = rel_quat_t(M0)
        q1, t1 = rel_quat_t(M1)
        print("\nrelative 四元数：")
        print("  q0 = %s   t0 = %s" % (np.array2string(q0, precision=5),
                                       np.array2string(t0, precision=5)))
        print("  q1 = %s   t1 = %s" % (np.array2string(q1, precision=5),
                                       np.array2string(t1, precision=5)))

        # ---- computeCenters 复刻 ----
        tris = []
        for a, b, c, d in faces:
            tris.append((a, b, c))
            tris.append((a, c, d))
        tri = np.asarray(tris)
        n = len(bind)
        cen = bind[tri].mean(axis=1)
        wm = W[tri].mean(axis=1)
        e1 = bind[tri[:, 1]] - bind[tri[:, 0]]
        e2 = bind[tri[:, 2]] - bind[tri[:, 0]]
        area = 0.5 * np.linalg.norm(np.cross(e1, e2), axis=1)
        keep = area > 1e-18
        tri, cen, wm, area = tri[keep], cen[keep], wm[keep], area[keep]
        mom = np.zeros((n, 3))
        mass = np.zeros(n)
        m = area * wm[:, 0] * wm[:, 1]
        for b in range(3):
            v = tri[:, b]
            mass += np.bincount(v, weights=m, minlength=n)
            for c in range(3):
                mom[:, c] += np.bincount(v, weights=m * cen[:, c], minlength=n)
        centers = bind.copy()
        good = mass > 0
        centers[good] = mom[good] / mass[good, None]

        # ---- 按 vertexTransform 语义求值 ----
        for tag, c in (("CoR 中心", centers[i_mix]), ("bind 自身", bind[i_mix])):
            w = W[i_mix]
            # 四元数混合（带反对偶），与源码 190-200 行一致
            acc = np.zeros(4)
            for wk, q in ((w[0], q0), (w[1], q1)):
                sign = -1.0 if float(acc @ q) < 0 else 1.0
                acc += wk * sign * q
            acc /= np.linalg.norm(acc)
            # posedCenter = Σ w_k · apply_transform(bindToPose[k], c)
            posed_center = np.zeros(3)
            for wk, q, t in ((w[0], q0, t0), (w[1], q1, t1)):
                posed_center += wk * (quat_rotate_row(q, c) + t)
            shift = posed_center - quat_rotate_row(acc, c)
            p = quat_rotate_row(acc, bind[i_mix]) + shift
            print("%-10s 复算 (%+.5f, %+.5f, %+.5f)  差 %.4e"
                  % (tag, *p, np.linalg.norm(p - posed[i_mix])))
        print("插件 alg0 (%+.5f, %+.5f, %+.5f)" % tuple(posed[i_mix]))
        return 0
    finally:
        maya.standalone.uninitialize()


if __name__ == "__main__":
    sys.exit(main())
