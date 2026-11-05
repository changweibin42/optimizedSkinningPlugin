"""
test_incremental_cor.py

验证增量 CoR 和全量 CoR 的一致性。

用法（在 Maya Script Editor 里 source）：
    exec(open(r"D:/visualStudioProject/optimizedSkinningPlugin/test_incremental_cor.py").read())
    test_incremental_cor()
"""

import maya.cmds as cmds
import time


def _create_scene():
    """建一个简单场景：圆柱 + 3 关节 + optimizedSkinCluster"""
    cmds.file(new=True, force=True)

    # 建网格
    cyl = cmds.polyCylinder(r=0.5, h=10, sx=20, sy=80, ch=1)[0]
    mesh = cmds.listRelatives(cyl, s=True)[0]

    # 建骨骼
    cmds.select(clear=True)
    cmds.joint(p=(0, 5, 0), n="j_root")
    cmds.joint(p=(0, 0, 0), n="j_mid")
    cmds.joint(p=(0, -5, 0), n="j_tip")
    joints = ["j_root", "j_mid", "j_tip"]

    # 创建 optimizedSkinCluster
    cmds.select(mesh)
    node = cmds.deformer(type="optimizedSkinCluster")[0]
    print(f"Created node: {node}")

    # 绑骨骼
    for i, j in enumerate(joints):
        if not cmds.objExists(f"{j}.lockInfluenceWeights"):
            cmds.addAttr(j, sn="liw", ln="lockInfluenceWeights", at="bool")
        cmds.connectAttr(f"{j}.liw", f"{node}.lockWeights[{i}]", force=True)
        cmds.connectAttr(f"{j}.worldMatrix[0]", f"{node}.matrix[{i}]", force=True)
        m = cmds.getAttr(f"{j}.wim")
        cmds.setAttr(f"{node}.bindPreMatrix[{i}]",
                     m[0], m[1], m[2], m[3],
                     m[4], m[5], m[6], m[7],
                     m[8], m[9], m[10], m[11],
                     m[12], m[13], m[14], m[15],
                     type="matrix")

    # 按 Y 高度分配权重
    vtx_count = cmds.polyEvaluate(mesh, v=True)
    ys = []
    for v in range(vtx_count):
        p = cmds.xform(f"{mesh}.vtx[{v}]", q=True, os=True, t=True)
        ys.append(p[1])
    y_min, y_max = min(ys), max(ys)
    y_rng = y_max - y_min or 1.0

    for v in range(vtx_count):
        for k in range(3):
            cmds.setAttr(f"{node}.weightList[{v}].weights[{k}]", 0.0)
        t = (ys[v] - y_min) / y_rng
        if t > 0.66:
            b = 0
        elif t < 0.33:
            b = 2
        else:
            b = 1
        cmds.setAttr(f"{node}.weightList[{v}].weights[{b}]", 1.0)

    return mesh, node, joints


def _get_positions(mesh):
    """读所有顶点世界位置"""
    vtx_count = cmds.polyEvaluate(mesh, v=True)
    positions = []
    for v in range(vtx_count):
        p = cmds.xform(f"{mesh}.vtx[{v}]", q=True, ws=True, t=True)
        positions.append(p)
    return positions


def _diff_stats(a, b):
    """返回 max/mean diff"""
    n = len(a)
    if n == 0:
        return 0.0, 0.0
    max_d = 0.0
    sum_d = 0.0
    for i in range(n):
        dx = a[i][0] - b[i][0]
        dy = a[i][1] - b[i][1]
        dz = a[i][2] - b[i][2]
        d = (dx*dx + dy*dy + dz*dz) ** 0.5
        if d > max_d:
            max_d = d
        sum_d += d
    return max_d, sum_d / n


def test_incremental_cor():
    print("=" * 60)
    print("Incremental CoR Consistency Test")
    print("=" * 60)

    # 建场景
    mesh, node, joints = _create_scene()
    cmds.refresh(force=True)

    # ---- 1. 初始全量 ----
    print("\n[1] Initial full rebuild...")
    cmds.setAttr(f"{node}.forceFullCoR", 1)
    cmds.refresh(force=True)
    V_init = _get_positions(mesh)

    # ---- 2. 改一个顶点权重，触发增量 ----
    print("\n[2] Modify 1 vertex weight (incremental)...")
    cmds.setAttr(f"{node}.forceFullCoR", 0)
    # 找一个中部顶点
    vtx_count = cmds.polyEvaluate(mesh, v=True)
    v_test = vtx_count // 2
    # 把权重改成 (0.5, 0.5, 0)
    cmds.setAttr(f"{node}.weightList[{v_test}].weights[0]", 0.5)
    cmds.setAttr(f"{node}.weightList[{v_test}].weights[1]", 0.5)
    cmds.setAttr(f"{node}.weightList[{v_test}].weights[2]", 0.0)
    cmds.refresh(force=True)
    V_incr = _get_positions(mesh)

    # ---- 3. 强制全量重算 ----
    print("\n[3] Force full rebuild after weight change...")
    cmds.setAttr(f"{node}.forceFullCoR", 1)
    cmds.refresh(force=True)
    V_full = _get_positions(mesh)

    # ---- 4. 对比 ----
    print("\n[4] Compare:")
    max_d, mean_d = _diff_stats(V_incr, V_full)
    print(f"  max diff  = {max_d:.6f}")
    print(f"  mean diff = {mean_d:.6f}")

    # ---- 5. 阈值判断 ----
    print("\n[5] Verdict:")
    if max_d < 1e-4:
        print("  ✅ PASS: incremental ≈ full (max diff < 1e-4)")
    elif max_d < 1e-2:
        print("  ⚠️  WARN: small diff (1e-4 ~ 1e-2), likely fine visually")
    else:
        print(f"  ❌ FAIL: diff too large ({max_d:.4f})")

    # ---- 6. 清理 ----
    cmds.setAttr(f"{node}.forceFullCoR", 0)
    print("\nDone.")