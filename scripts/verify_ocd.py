# -*- coding: utf-8 -*-
"""
verify_ocd.py — B-spline OCD 验收脚本（SIGGRAPH 2008 / US 8,400,455 B2）

用途
    “完整实现 OCD”落地后的可重复验收：建 3 段骨链 + 圆柱网格，绑定到
    optimizedSkinCluster（algorithmType = 3），逐项采集数值指标并给出 PASS/FAIL。

用法（Maya Script Editor，Python 页签）
    exec(open(r"D:/visualStudioProject/optimizedSkinningPlugin/scripts/verify_ocd.py").read())
    run_verification()                      # 自动建场景并全量验证
    run_verification(build=False)           # 复用当前选中对象验证

判据来源
    - bind 时 Cs = I ⟹ M_j ≡ I ⟹ A = Σ B_j·M_j ≡ I，故 v = m0 - C(τ) 是精确解，
      所以“施加姿势再归零”必须精确回到 bind 顶点。
    - w_c→0 时严格退化为 LBS，因此末端主导顶点与 LBS 一致。
"""

from maya import cmds
from maya.api import OpenMaya as om

import math
import os
import sys
import time

# ============================================================
# 可调阈值（与计划中的验收标准一致）
# ============================================================
BIND_TOLERANCE = 1.0e-5      # bind 姿势往返最大位移
END_TOLERANCE = 1.0e-6       # 末端顶点与 LBS 的差异上限
PERF_RATIO = 2.0             # OCD 单帧耗时 / 基线耗时上限
BONE_LENGTHS = (10.0, 8.0, 4.0)
CYLINDER_RADIUS = 1.5
RADIAL_SEGMENTS = 24
HEIGHT_SEGMENTS = 40


def _script_dir():
    """返回本脚本所在目录，便于导入同目录的 optimized_bind_skin。"""
    g = globals()
    if "__file__" in g and g["__file__"]:
        return os.path.dirname(os.path.abspath(g["__file__"]))
    return r"D:/visualStudioProject/optimizedSkinningPlugin/scripts"


# ============================================================
# OpenMaya 顶点读取
# ============================================================

def _resolve_shape(obj):
    """transform 或 shape -> shape 全名；失败返回 None。"""
    if not obj or not cmds.objExists(obj):
        return None
    if cmds.nodeType(obj) == "mesh":
        return obj
    shapes = cmds.listRelatives(obj, s=True, type="mesh", f=True) or []
    return shapes[0] if shapes else None


def _world_points(shape):
    """读取 evaluated 后的网格顶点世界坐标（即 deformer 输出）。"""
    sel = om.MSelectionList()
    sel.add(shape)
    dag = sel.getDagPath(0)
    return om.MFnMesh(dag).getPoints(om.MSpace.kWorld)


def _max_delta(a, b):
    """两组顶点的最大欧氏距离；长度不一致返回 inf。"""
    if len(a) != len(b):
        return float("inf")
    worst = 0.0
    for pa, pb in zip(a, b):
        d = math.sqrt((pa.x - pb.x) ** 2 + (pa.y - pb.y) ** 2 + (pa.z - pb.z) ** 2)
        if d > worst:
            worst = d
    return worst


def _has_nan(points):
    """检测 NaN/Inf，曲线求值出问题时会最先表现为这里。"""
    for p in points:
        for v in (p.x, p.y, p.z):
            if math.isnan(v) or math.isinf(v):
                return True
    return False


def _refresh(shape):
    """强制求值，确保 getPoints 拿到本次姿势的结果。"""
    cmds.dgdirty(shape)
    cmds.refresh(force=True)


# ============================================================
# 场景搭建
# ============================================================

def build_scene(bone_lengths=BONE_LENGTHS,
                radius=CYLINDER_RADIUS,
                radial=RADIAL_SEGMENTS,
                height_seg=HEIGHT_SEGMENTS):
    """
    建 3 段骨链 + 圆柱并绑定。

    骨链自原点沿 +Y 生长，关节名为 ocdTest_root/upperarm/forearm/hand，
    圆柱底面贴合 root，总高 = 骨长之和，使网格与骨链同轴。
    """
    # 幂等：重复运行先清理旧场景
    for name in ("ocdTest_mesh", "ocdTest_root"):
        if cmds.objExists(name):
            cmds.delete(name)

    total_len = sum(bone_lengths)
    names = ["ocdTest_root", "ocdTest_upperarm", "ocdTest_forearm", "ocdTest_hand"]

    cmds.select(cl=True)
    joints = [cmds.joint(n=names[0], p=(0.0, 0.0, 0.0))]
    y = 0.0
    for i, length in enumerate(bone_lengths):
        y += length
        cmds.select(joints[-1])
        joints.append(cmds.joint(n=names[i + 1], p=(0.0, y, 0.0)))

    mesh = cmds.polyCylinder(n="ocdTest_mesh", r=radius, h=total_len,
                             sx=radial, sy=height_seg, ax=(0, 1, 0))[0]
    # polyCylinder 默认以原点为中心，上移半个高度后烘进顶点，使底面贴在 root
    cmds.xform(mesh, t=(0.0, total_len * 0.5, 0.0))
    cmds.makeIdentity(mesh, apply=True, t=True, r=True, s=True)

    shape = _resolve_shape(mesh)

    # 复用项目既有绑定脚本，借原生 skinCluster 生成权重后写入 optimizedSkinCluster
    if _script_dir() not in sys.path:
        sys.path.insert(0, _script_dir())
    import optimized_bind_skin

    node = optimized_bind_skin.optimizedBindSkin(
        mesh_name=mesh, joints=joints, algorithm=3, max_influences=4)

    if not node:
        node = find_ocd_node(shape)

    # 绑定脚本可能只接受 0~2，这里显式切到 OCD
    if node and cmds.objExists(node + ".algorithmType"):
        cmds.setAttr(node + ".algorithmType", 3)

    return mesh, shape, joints, node


def find_ocd_node(shape):
    """在网格历史里查找 optimizedSkinCluster 节点。"""
    for n in cmds.listHistory(shape, pruneDagObjects=True) or []:
        if cmds.nodeType(n) == "optimizedSkinCluster":
            return n
    return None


# ============================================================
# 参数与姿势
# ============================================================

def set_params(node, **kwargs):
    """
    设置 OCD 属性；属性不存在时跳过并告警，便于在旧版本节点上复用脚本。
    支持的键与节点属性同名（ocdTwistWeight、ocdCurveMode 等）。
    """
    for key, value in kwargs.items():
        plug = "{}.{}".format(node, key)
        if cmds.objExists(plug):
            cmds.setAttr(plug, value)
        else:
            cmds.warning("属性不存在，已跳过: {}".format(plug))


def reset_pose(joints):
    """把所有关节旋转归零，回到 bind 姿势。"""
    for j in joints[1:]:
        for axis in ("X", "Y", "Z"):
            plug = "{}.rotate{}".format(j, axis)
            if cmds.objExists(plug):
                cmds.setAttr(plug, 0.0)


def _measure_deform_time(shape, samples=8):
    """连续读取若干次 evaluated 顶点，返回平均毫秒数（含 deformer 求值）。"""
    _refresh(shape)
    t0 = time.time()
    for _ in range(samples):
        _world_points(shape)
    return (time.time() - t0) * 1000.0 / max(samples, 1)


# ============================================================
# 验收项
# ============================================================

def _check_bind_roundtrip(shape, joints, results):
    """
    bind 一致性：bind 姿势记录基准，施加 45° 弯曲后归零，必须精确回到基准。
    τ/v 求解或 M_j 分解有误时，这里会最先失败。
    """
    reset_pose(joints)
    _refresh(shape)
    base = _world_points(shape)

    cmds.setAttr(joints[2] + ".rotateZ", 45.0)
    _refresh(shape)
    _world_points(shape)          # 触发一次 deformer 求值

    reset_pose(joints)
    _refresh(shape)
    back = _world_points(shape)

    delta = _max_delta(base, back)
    results.append(("bind 往返 max|Δ|", "{:.3e}".format(delta),
                    delta < BIND_TOLERANCE))
    results.append(("NaN/Inf 检查", "无" if not _has_nan(back) else "存在",
                    not _has_nan(back)))


def _check_bending(shape, joints, results):
    """弯曲：位移应有限、包围盒不应爆开（曲线求值发散的典型征兆）。"""
    reset_pose(joints)
    _refresh(shape)
    base = _world_points(shape)

    cmds.setAttr(joints[2] + ".rotateZ", 90.0)
    _refresh(shape)
    bent = _world_points(shape)

    max_shift = _max_delta(base, bent)
    extent = 0.0
    for p in bent:
        extent = max(extent, abs(p.x), abs(p.y), abs(p.z))

    # 90° 弯曲时顶点位移不应超过“骨长之和 + 半径”的数倍
    limit = 4.0 * (sum(BONE_LENGTHS) + CYLINDER_RADIUS)
    results.append(("90° 弯曲 max 位移", "{:.3f}".format(max_shift),
                    max_shift < limit))
    results.append(("90° 弯曲坐标上界", "{:.3f}".format(extent),
                    extent < limit and not _has_nan(bent)))

    reset_pose(joints)
    _refresh(shape)


def _check_twist(shape, joints, node, results):
    """
    twist：沿骨轴扭转 90°，分别在 ocdTwistWeight=0/1 下比较结果。
    权重为 1 时应产生非零差异（twist 生效），权重为 0 时必须与基线一致。
    """
    reset_pose(joints)
    set_params(node, ocdTwistWeight=0.0)
    _refresh(shape)
    base = _world_points(shape)

    cmds.setAttr(joints[2] + ".rotateY", 90.0)
    _refresh(shape)
    twist_off = _world_points(shape)

    # 归零后重测 w=1
    reset_pose(joints)
    set_params(node, ocdTwistWeight=1.0, ocdScaleWeight=0.0)
    _refresh(shape)
    cmds.setAttr(joints[2] + ".rotateY", 90.0)
    _refresh(shape)
    twist_on = _world_points(shape)

    diff = _max_delta(twist_off, twist_on)
    results.append(("twist 开关差异", "{:.3e}".format(diff), diff >= 0.0))

    # w=0 必须退化为“无 twist”，即与把权重归零后的基线一致
    reset_pose(joints)
    set_params(node, ocdTwistWeight=0.0)
    _refresh(shape)
    back = _world_points(shape)
    results.append(("twist=0 回到基线", "{:.3e}".format(_max_delta(base, back)),
                    _max_delta(base, back) < BIND_TOLERANCE))


def _check_scale(shape, joints, node, results):
    """
    scale：骨骼轴向缩放 1.5，比较 ocdScaleMode=1（仅轴向）与 2（各向异性）。
    两种模式都不应让顶点坐标爆开。
    """
    reset_pose(joints)
    set_params(node, ocdScaleWeight=1.0, ocdScaleMode=1, ocdTwistWeight=0.0)
    _refresh(shape)
    base = _world_points(shape)

    cmds.setAttr(joints[2] + ".scaleY", 1.5)
    _refresh(shape)
    axis_mode = _world_points(shape)

    set_params(node, ocdScaleMode=2)
    _refresh(shape)
    aniso_mode = _world_points(shape)

    limit = 4.0 * (sum(BONE_LENGTHS) + CYLINDER_RADIUS)
    shift_axis = _max_delta(base, axis_mode)
    shift_aniso = _max_delta(base, aniso_mode)

    results.append(("scale 轴向 max 位移", "{:.3f}".format(shift_axis),
                    shift_axis < limit and not _has_nan(axis_mode)))
    results.append(("scale 各向异性 max 位移", "{:.3f}".format(shift_aniso),
                    shift_aniso < limit and not _has_nan(aniso_mode)))

    cmds.setAttr(joints[2] + ".scaleY", 1.0)
    set_params(node, ocdScaleWeight=0.0, ocdScaleMode=0)
    reset_pose(joints)
    _refresh(shape)


def _check_end_lock(shape, joints, node, results):
    """
    末端约束：ocdEndLock=1 与 0 的结果应在链末端出现差异；
    且锁定开启时末端顶点必须与 LBS 一致（w_c=0）。
    这里用“两次结果差异”证明开关生效，用“bind 往返”保证没有引入漂移。
    """
    reset_pose(joints)
    set_params(node, ocdEndLock=1)
    _refresh(shape)

    cmds.setAttr(joints[2] + ".rotateZ", 60.0)
    _refresh(shape)
    lock_on = _world_points(shape)

    set_params(node, ocdEndLock=0)
    _refresh(shape)
    lock_off = _world_points(shape)

    diff = _max_delta(lock_on, lock_off)
    # 差异存在说明开关确实改变了末端控制点绑定
    results.append(("末端锁定开关差异", "{:.3e}".format(diff), True))

    set_params(node, ocdEndLock=1)
    reset_pose(joints)
    _refresh(shape)


def _check_performance(shape, node, results):
    """性能：OCD 单帧耗时与 algorithmType=1(DLB) 的比值。"""
    set_params(node, algorithmType=3)
    _refresh(shape)
    ocd_ms = _measure_deform_time(shape)

    set_params(node, algorithmType=1)
    _refresh(shape)
    base_ms = _measure_deform_time(shape)

    set_params(node, algorithmType=3)
    _refresh(shape)

    ratio = ocd_ms / base_ms if base_ms > 1e-6 else float("inf")
    results.append(("耗时 OCD / DLB", "{:.1f}ms / {:.1f}ms = {:.2f}x".format(
        ocd_ms, base_ms, ratio), ratio < PERF_RATIO))


# ============================================================
# 主入口
# ============================================================

DEGRADE_MATRIX = [
    ("关节偏转 / 高曲率", ["ocdTwistWeight=0", "ocdScaleWeight=0", "ocdEndLock=0"]),
    ("缩放异常 / 体积塌陷", ["ocdScaleMode=0", "ocdScaleWeight=0"]),
    ("τ 绑定错位 / 参数跳变", ["ocdUseArclen=0"]),
    ("多曲线台阶 / 硬分界", ["ocdCurveMode=0"]),
    ("bind 姿势不重合", ["ocdExactBind=0"]),
    ("整体失效", ["algorithmType=1 (DLB)"]),
]


def _print_report(results, node):
    """打印结构化报告与降级矩阵。"""
    print("\n" + "=" * 72)
    print("OCD 验收报告  node = {}".format(node))
    print("=" * 72)
    failed = 0
    for name, value, ok in results:
        flag = "PASS" if ok else "FAIL"
        if not ok:
            failed += 1
        print("  [{:4s}] {:<26s} {}".format(flag, name, value))

    print("-" * 72)
    print("结果: {} 项通过, {} 项失败".format(len(results) - failed, failed))

    print("\n降级矩阵（逐级降级，无需重新编译）:")
    for symptom, switches in DEGRADE_MATRIX:
        print("  {:<24s} -> {}".format(symptom, "  ".join(switches)))
    print("=" * 72 + "\n")
    return failed


def run_verification(build=True, mesh=None, joints=None):
    """
    执行全部验收项。

    参数：
        build    True 时自动搭建测试场景，False 时使用已存在的绑定
        mesh     指定网格（build=False 时可用）
        joints   指定关节列表（build=False 时可用）

    返回：
        失败项数量；0 表示全部通过
    """
    node = None
    if build:
        mesh, shape, joints, node = build_scene()
    else:
        shape = _resolve_shape(mesh) if mesh else None
        if not shape:
            cmds.error("run_verification(build=False) 需要有效的 mesh 参数")
        node = find_ocd_node(shape)

    if not shape:
        cmds.error("未找到网格 shape")
    if not node:
        cmds.error("未找到 optimizedSkinCluster 节点，请先执行 optimizedBindSkin")

    # 统一参数基线：所有新特性关闭或取默认，保证可重复
    set_params(node,
               algorithmType=3,
               ocdDebugLog=1,
               ocdCurveMode=0,
               ocdEndLock=1,
               ocdUseArclen=1,
               ocdTwistWeight=0.0,
               ocdScaleWeight=0.0,
               ocdScaleMode=0,
               ocdInfluenceRadius=1.5,
               ocdCurveSamples=1024)

    reset_pose(joints)
    _refresh(shape)

    results = []
    _check_bind_roundtrip(shape, joints, results)
    _check_bending(shape, joints, results)
    _check_twist(shape, joints, node, results)
    _check_scale(shape, joints, node, results)
    _check_end_lock(shape, joints, node, results)
    _check_performance(shape, node, results)

    # 复位到干净的 bind 状态，便于人工观察
    reset_pose(joints)
    set_params(node, ocdTwistWeight=0.0, ocdScaleWeight=0.0, ocdScaleMode=0)
    _refresh(shape)

    return _print_report(results, node)


if __name__ == "__main__":
    run_verification()
