"""
optimized_bind_skin.py

类似 Maya 的 bindSkin，用 optimizedSkinCluster 蒙皮。

流程：
    1. 临时建原生 skinCluster（借用 Maya 内置高质量权重算法）
    2. 用 OpenMaya 一次性读全部权重
    3. 删掉临时 skinCluster
    4. 建 optimizedSkinCluster + 绑骨骼
    5. 把权重写进 A
    6. 设置算法类型

用法（Python）：
    import sys
    sys.path.insert(0, "D:/visualStudioProject/optimizedSkinningPlugin/scripts")
    import optimized_bind_skin

    # 显式传参
    nodeA = optimized_bind_skin.optimizedBindSkin(
        mesh_name="pCylinder1",
        joints=["j_root", "j_mid", "j_tip"],
        algorithm=0,        # 0=CoR, 1=DLB, 2=DQS
        max_influences=4,
        normalize=True,
    )

    # 或从选择推断（第一个 mesh + 其余 joints）
    nodeA = optimized_bind_skin.optimizedBindSkinFromSelection()

MEL 用法（需先 install_mel_commands）：
    optimizedBindSkinFromSelection();
"""

from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim
import time


# ============================================================
# 内部：OpenMaya 权重读/写
# ============================================================

def _resolve_shape(node):
    """transform 或 shape -> 全路径 shape"""
    if not cmds.objExists(node):
        return None
    if cmds.nodeType(node) == 'transform':
        shapes = cmds.listRelatives(node, s=True, f=True) or []
        if not shapes:
            return None
        return shapes[0]
    return node


def _get_transform(node):
    """shape 或 transform -> transform"""
    if cmds.nodeType(node) == 'transform':
        return node
    pars = cmds.listRelatives(node, p=True) or []
    return pars[0] if pars else node


def _read_native_skin_weights(skin_node, shape_name):
    """读原生 skinCluster 全部权重。
    返回 (weights_flat, num_inf, num_verts, influence_names)
    """
    sel = OpenMaya.MSelectionList()
    sel.add(skin_node)
    sel.add(shape_name)
    skin_dep = sel.getDependNode(0)
    shape_dag = sel.getDagPath(1)

    fn_skin = OpenMayaAnim.MFnSkinCluster(skin_dep)
    fn_mesh = OpenMaya.MFnMesh(shape_dag)
    num_verts = fn_mesh.numVertices

    # 影响骨骼
    inf_dags = fn_skin.influenceObjects()
    num_inf = len(inf_dags)
    influence_names = []
    for dag in inf_dags:
        full = dag.fullPathName()
        short = full.split('|')[-1]
        # 去掉命名空间
        short = short.split(':')[-1]
        influence_names.append(short)

    # 全部 influence 的 physical index
    indices = OpenMaya.MIntArray()
    for i in range(num_inf):
        indices.append(i)

    components = OpenMaya.MObject()   # 空 = 全部顶点

    # ★ getWeights 返回单数组
    weights_arr = fn_skin.getWeights(shape_dag, components, indices)

    return list(weights_arr), num_inf, num_verts, influence_names


def _write_weights(skin_node, shape_name, weights_flat, num_inf):
    """把权重写入 skinCluster（原生或 optimizedSkinCluster 都行）"""
    sel = OpenMaya.MSelectionList()
    sel.add(skin_node)
    sel.add(shape_name)
    skin_dep = sel.getDependNode(0)
    shape_dag = sel.getDagPath(1)

    fn_skin = OpenMayaAnim.MFnSkinCluster(skin_dep)

    indices = OpenMaya.MIntArray()
    for i in range(num_inf):
        indices.append(i)

    components = OpenMaya.MObject()
    weights_mda = OpenMaya.MDoubleArray(weights_flat)

    # ★ setWeights(dagPath, components, indices, weights, normalize)
    fn_skin.setWeights(shape_dag, components, indices, weights_mda, False)


# ============================================================
# 内部：参数检查
# ============================================================

def _prepare_mesh(mesh_name):
    """归一化 + 检查 mesh。返回 (mesh_transform, mesh_shape) 或 (None, None)"""
    if not cmds.objExists(mesh_name):
        cmds.warning(f"Mesh does not exist: {mesh_name}")
        return None, None

    if cmds.nodeType(mesh_name) == 'transform':
        shapes = cmds.listRelatives(mesh_name, s=True, type='mesh') or []
        if not shapes:
            cmds.warning(f"{mesh_name} has no mesh shape")
            return None, None
        return mesh_name, shapes[0]

    if cmds.nodeType(mesh_name) == 'mesh':
        pars = cmds.listRelatives(mesh_name, p=True) or []
        return (pars[0] if pars else mesh_name), mesh_name

    cmds.warning(f"{mesh_name} is not a mesh")
    return None, None


def _prepare_joints(joints):
    """过滤有效的 joint"""
    if isinstance(joints, str):
        joints = [joints]

    valid = []
    for j in joints:
        if not cmds.objExists(j):
            cmds.warning(f"Joint not found: {j}")
            continue
        if cmds.nodeType(j) != 'joint':
            cmds.warning(f"Not a joint: {j}")
            continue
        valid.append(j)
    return valid


def _find_skin_cluster(mesh_shape, node_type):
    """在 mesh 历史里找指定类型的 skin 节点"""
    hist = cmds.listHistory(mesh_shape, pruneDagObjects=True) or []
    for n in hist:
        if cmds.nodeType(n) == node_type:
            return n
    return None


def _ensure_liw(joints):
    """确保所有 joint 都有 lockInfluenceWeights 属性"""
    for j in joints:
        if not cmds.objExists(f"{j}.lockInfluenceWeights"):
            cmds.addAttr(j, sn="liw", ln="lockInfluenceWeights",
                         at="bool", dv=True)
            cmds.setAttr(f"{j}.lockInfluenceWeights", 1)


# ============================================================
# 主入口
# ============================================================

def optimizedBindSkin(mesh_name, joints,
                       algorithm=0,
                       max_influences=4,
                       normalize=True,
                       verbose=True):
    """
    用 optimizedSkinCluster 蒙皮（借用 Maya 内置权重算法）

    参数：
        mesh_name      mesh transform 或 shape 名
        joints         joint 名列表
        algorithm      0=CoR, 1=DLB, 2=DQS
        max_influences 最大影响骨骼数（1~4）
        normalize      是否归一化权重
        verbose        打印详细信息

    返回：
        optimizedSkinCluster 节点名（失败返回 None）
    """
    t_start = time.time()

    # ---------- 参数检查 ----------
    mesh_transform, mesh_shape = _prepare_mesh(mesh_name)
    if not mesh_transform:
        return None

    valid_joints = _prepare_joints(joints)
    if not valid_joints:
        cmds.warning("No valid joints")
        return None

    max_influences = max(1, min(4, int(max_influences)))

    if verbose:
        print(f"[bind] mesh = {mesh_transform}")
        print(f"[bind] joints = {valid_joints}")
        print(f"[bind] max_influences = {max_influences}, algorithm = {algorithm}")

    # ---------- 检查冲突 ----------
    existing_native = _find_skin_cluster(mesh_shape, 'skinCluster')
    if existing_native:
        cmds.warning(
            f"{mesh_transform} already has native skinCluster: {existing_native}. "
            f"Please delete it first.")
        return None

    existing_opt = _find_skin_cluster(mesh_shape, 'optimizedSkinCluster')
    if existing_opt:
        cmds.warning(
            f"{mesh_transform} already has optimizedSkinCluster: {existing_opt}")
        return existing_opt

    # ---------- 保存选择 ----------
    old_sel = cmds.ls(sl=True) or []

    # ---------- 步骤 1：临时建原生 skinCluster ----------
    if verbose:
        print("[bind] Step 1: temp native skinCluster for weight computation")

    _ensure_liw(valid_joints)

    temp_skin = None
    try:
        cmds.select(clear=True)
        for j in valid_joints:
            cmds.select(j, add=True)
        cmds.select(mesh_transform, add=True)

        result = cmds.skinCluster(
            tsb=True,
            toSelectedBones=True,
            maximumInfluences=max_influences,
            normalizeWeights=1 if normalize else 0,
            obeyMaxInfluences=True,
            removeUnusedInfluence=False,
            name="_optimizedBind_temp#")
        temp_skin = result[0] if isinstance(result, (list, tuple)) else result

        if not temp_skin or not cmds.objExists(temp_skin):
            cmds.warning("skinCluster command returned invalid node")
            cmds.select(old_sel, r=True)
            return None

        if verbose:
            print(f"[bind] temp skinCluster = {temp_skin}")
    except Exception as e:
        cmds.warning(f"Temp skinCluster failed: {e}")
        cmds.select(old_sel, r=True)
        return None

    # ---------- 步骤 2：读权重 ----------
    if verbose:
        print("[bind] Step 2: read weights via OpenMaya")

    try:
        weights_flat, num_inf, num_verts, influence_names = \
            _read_native_skin_weights(temp_skin, mesh_shape)
        if verbose:
            print(f"[bind] read {len(weights_flat)} values "
                  f"({num_verts} verts x {num_inf} inf)")
            print(f"[bind] influences = {influence_names}")
    except Exception as e:
        cmds.warning(f"Read weights failed: {e}")
        cmds.delete(temp_skin)
        cmds.select(old_sel, r=True)
        return None

    # ---------- 步骤 3：删掉临时 skinCluster ----------
    if verbose:
        print("[bind] Step 3: delete temp skinCluster")

    try:
        cmds.delete(temp_skin)
    except Exception as e:
        cmds.warning(f"Delete temp skinCluster failed: {e}")

    # ---------- 步骤 4：建 optimizedSkinCluster ----------
    if verbose:
        print("[bind] Step 4: create optimizedSkinCluster")

    try:
        cmds.select(mesh_shape, r=True)
        nodeA = cmds.deformer(type="optimizedSkinCluster")[0]
        if verbose:
            print(f"[bind] optimizedSkinCluster = {nodeA}")
    except Exception as e:
        cmds.warning(f"Create optimizedSkinCluster failed: {e}")
        cmds.select(old_sel, r=True)
        return None

    # ---------- 步骤 5：绑骨骼到 A ----------
    if verbose:
        print("[bind] Step 5: bind joints to A")

    # 骨骼按名称匹配：temp 的 influence_names 顺序 vs valid_joints 顺序
    # 用户通常按顺序给，直接按顺序连
    name_to_idx = {name: i for i, name in enumerate(influence_names)}

    # 用于写权重时的重排索引：dst_idx -> src_idx
    dst_to_src = []
    matched = 0
    for j in valid_joints:
        short_name = j.split('|')[-1].split(':')[-1]
        si = name_to_idx.get(short_name, -1)
        dst_to_src.append(si)
        if si >= 0:
            matched += 1

    if verbose:
        print(f"[bind] matched {matched}/{len(valid_joints)} influences by name")
        print(f"[bind] dst_to_src = {dst_to_src}")

    for i, j in enumerate(valid_joints):
        try:
            # lockWeights
            cmds.connectAttr(f"{j}.lockInfluenceWeights",
                             f"{nodeA}.lockWeights[{i}]", f=True)
            # matrix
            cmds.connectAttr(f"{j}.worldMatrix[0]",
                             f"{nodeA}.matrix[{i}]", f=True)
            # influenceColor（可选）
            if cmds.objExists(f"{j}.objectColorRGB"):
                try:
                    cmds.connectAttr(f"{j}.objectColorRGB",
                                     f"{nodeA}.influenceColor[{i}]", f=True)
                except Exception:
                    pass
            # bindPreMatrix
            m = cmds.getAttr(f"{j}.wim")
            cmds.setAttr(f"{nodeA}.bindPreMatrix[{i}]", *m, type="matrix")
        except Exception as e:
            cmds.warning(f"Bind joint {j} failed: {e}")

    # ---------- 步骤 6：写权重 ----------
    if verbose:
        print("[bind] Step 6: write weights to A")

    dst_ninf = len(valid_joints)

    # 重排 + 归一化
    if matched == len(valid_joints) and dst_ninf == num_inf:
        # 顺序一致，直接写
        dst_weights = weights_flat
    else:
        # 需要重排
        dst_weights = [0.0] * (num_verts * dst_ninf)
        for v in range(num_verts):
            src_base = v * num_inf
            dst_base = v * dst_ninf
            total = 0.0
            for j in range(dst_ninf):
                si = dst_to_src[j]
                if 0 <= si < num_inf:
                    val = weights_flat[src_base + si]
                else:
                    val = 0.0
                dst_weights[dst_base + j] = val
                total += val

            if normalize and total > 1e-9:
                inv = 1.0 / total
                for j in range(dst_ninf):
                    dst_weights[dst_base + j] *= inv

    try:
        _write_weights(nodeA, mesh_shape, dst_weights, dst_ninf)
        if verbose:
            print(f"[bind] wrote {len(dst_weights)} values")
    except Exception as e:
        cmds.warning(f"Write weights failed: {e}")

    # ---------- 步骤 7：设置算法 ----------
    try:
        cmds.setAttr(f"{nodeA}.algorithmType", int(algorithm))
    except Exception:
        pass

    # ---------- 收尾 ----------
    cmds.dgdirty(nodeA)
    cmds.select(old_sel, r=True)

    elapsed = time.time() - t_start
    if verbose:
        print(f"[bind] DONE in {elapsed:.2f}s. nodeA = {nodeA}")

    return nodeA


def optimizedBindSkinFromSelection(algorithm=0, max_influences=4,
                                     normalize=True):
    """
    从当前选择推断：
      · 第一个 mesh = 要蒙皮的模型
      · 其余 joints = 影响骨骼
    """
    sel = cmds.ls(sl=True) or []
    if len(sel) < 2:
        raise RuntimeError("请先选择 1 个 mesh + 至少 1 个 joint")

    mesh_name = None
    joints = []
    for s in sel:
        if cmds.nodeType(s) == 'transform':
            shapes = cmds.listRelatives(s, s=True, type='mesh') or []
            if shapes and mesh_name is None:
                mesh_name = s
                continue
        elif cmds.nodeType(s) == 'mesh':
            if mesh_name is None:
                mesh_name = s
                continue
        if cmds.nodeType(s) == 'joint':
            joints.append(s)

    if not mesh_name:
        raise RuntimeError("选择里没有 mesh")
    if not joints:
        raise RuntimeError("选择里没有 joint")

    return optimizedBindSkin(mesh_name, joints,
                              algorithm=algorithm,
                              max_influences=max_influences,
                              normalize=normalize)


# ============================================================
# MEL 封装
# ============================================================

def install_mel_commands():
    """安装 MEL 命令：
        optimizedBindSkinFromSelection()
        optimizedBindSkin(mesh, joints)
    """
    import maya.mel as mel
    code = r'''
global proc optimizedBindSkinFromSelection() {
    python("import sys; "
         + "sys.path.insert(0, 'D:/visualStudioProject/optimizedSkinningPlugin/scripts'); "
         + "import importlib, optimized_bind_skin; "
         + "importlib.reload(optimized_bind_skin); "
         + "optimized_bind_skin.optimizedBindSkinFromSelection()");
}

global proc optimizedBindSkin(string $mesh, string $joints[]) {
    string $jointStr = "";
    for ($i = 0; $i < size($joints); $i++) {
        if ($i > 0) $jointStr += ",";
        $jointStr += "'" + $joints[$i] + "'";
    }
    python("import sys; "
         + "sys.path.insert(0, 'D:/visualStudioProject/optimizedSkinningPlugin/scripts'); "
         + "import importlib, optimized_bind_skin; "
         + "importlib.reload(optimized_bind_skin); "
         + "optimized_bind_skin.optimizedBindSkin('"
         + $mesh + "', [" + $jointStr + "])");
}
'''
    mel.eval(code)
    print("[optimized_bind_skin] MEL 命令已安装")


# ============================================================
# 命令行入口
# ============================================================

if __name__ == "__main__":
    optimizedBindSkinFromSelection()