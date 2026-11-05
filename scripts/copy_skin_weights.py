"""
copy_skin_weights.py

从原生 skinCluster 拷贝权重到 optimizedSkinCluster。
用 OpenMaya 2.0 (MFnSkinCluster) 一次性读/写所有权重。
"""

from maya import cmds
from maya.api import OpenMaya, OpenMayaAnim
import time


# ============================================================
# 内部工具
# ============================================================

def _resolve_shape(node):
    if not cmds.objExists(node):
        raise RuntimeError(f"Object does not exist: {node}")
    if cmds.nodeType(node) == 'transform':
        shapes = cmds.listRelatives(node, s=True, f=True) or []
        if not shapes:
            raise RuntimeError(f"No shape under: {node}")
        return shapes[0]
    return node


def _find_in_history(shape, node_type):
    hist = cmds.listHistory(shape, pruneDagObjects=True) or []
    for n in hist:
        if cmds.nodeType(n) == node_type:
            return n
    return None


def _get_influence_names(skin_node):
    try:
        sz = cmds.getAttr(f"{skin_node}.matrix", size=True)
    except Exception:
        return []
    names = []
    for i in range(sz):
        conns = cmds.listConnections(f"{skin_node}.matrix[{i}]",
                                     s=True, d=False)
        names.append(conns[0] if conns else "")
    return names


# ============================================================
# 权重读/写
# ============================================================

def _read_skin_weights(skin_node, shape_node):
    """返回 (weights_flat, num_inf, num_verts)"""
    sel = OpenMaya.MSelectionList()
    sel.add(skin_node)
    sel.add(shape_node)

    skin_dep = sel.getDependNode(0)
    shape_dag = sel.getDagPath(1)

    fn_skin = OpenMayaAnim.MFnSkinCluster(skin_dep)
    fn_mesh = OpenMaya.MFnMesh(shape_dag)
    num_verts = fn_mesh.numVertices

    num_inf = len(fn_skin.influenceObjects())

    indices = OpenMaya.MIntArray()
    for i in range(num_inf):
        indices.append(i)

    components = OpenMaya.MObject()

    # ★ getWeights 只返回一个 MDoubleArray
    weights_arr = fn_skin.getWeights(shape_dag, components, indices)

    return list(weights_arr), int(num_inf), num_verts


def _write_skin_weights(skin_node, shape_node,
                        weights_flat, num_verts, num_inf):
    """把所有权重写入 skinNode"""
    sel = OpenMaya.MSelectionList()
    sel.add(skin_node)
    sel.add(shape_node)

    skin_dep = sel.getDependNode(0)
    shape_dag = sel.getDagPath(1)

    fn_skin = OpenMayaAnim.MFnSkinCluster(skin_dep)

    indices = OpenMaya.MIntArray()
    for i in range(num_inf):
        indices.append(i)

    components = OpenMaya.MObject()

    weights_mda = OpenMaya.MDoubleArray(weights_flat)

    # ★ setWeights(dagPath, components, indices, weights, normalize)
    fn_skin.setWeights(shape_dag, components, indices,
                       weights_mda, normalize=False)


# ============================================================
# 主入口
# ============================================================

def copySkinWeightsToOptimized(source, target,
                                normalize=True, verbose=True):
    t0 = time.time()

    src_shape = _resolve_shape(source)
    dst_shape = _resolve_shape(target)

    src_skin = _find_in_history(src_shape, 'skinCluster')
    dst_skin = _find_in_history(dst_shape, 'optimizedSkinCluster')

    if not src_skin:
        raise RuntimeError(f"No skinCluster on {source}")
    if not dst_skin:
        raise RuntimeError(f"No optimizedSkinCluster on {target}")

    if verbose:
        print(f"[copy] source skinCluster         = {src_skin}")
        print(f"[copy] target optimizedSkinCluster = {dst_skin}")

    src_nv = cmds.polyEvaluate(src_shape, v=True)
    dst_nv = cmds.polyEvaluate(dst_shape, v=True)
    if src_nv != dst_nv:
        raise RuntimeError(
            f"Vertex count mismatch: src={src_nv}, dst={dst_nv}")
    num_verts = src_nv

    src_inf = _get_influence_names(src_skin)
    dst_inf = _get_influence_names(dst_skin)
    src_ninf = len(src_inf)
    dst_ninf = len(dst_inf)

    if verbose:
        print(f"[copy] src influences ({src_ninf}): {src_inf}")
        print(f"[copy] dst influences ({dst_ninf}): {dst_inf}")

    src_by_name = {name: i for i, name in enumerate(src_inf) if name}
    dst_to_src = [src_by_name.get(name, -1) for name in dst_inf]
    matched = sum(1 for v in dst_to_src if v >= 0)

    if verbose:
        print(f"[copy] matched {matched}/{dst_ninf} influences by name")

    if matched == 0:
        raise RuntimeError(
            "No influence names matched between source and target")

    if verbose:
        print(f"[copy] reading {num_verts} verts x {src_ninf} inf ...")
    t1 = time.time()
    src_w, src_ninf_eff, _ = _read_skin_weights(src_skin, src_shape)
    if verbose:
        print(f"[copy] read {len(src_w)} values in {time.time()-t1:.2f}s")

    if verbose:
        print(f"[copy] building target weights ...")
    t2 = time.time()
    dst_w = [0.0] * (num_verts * dst_ninf)

    for v in range(num_verts):
        src_base = v * src_ninf_eff
        dst_base = v * dst_ninf
        total = 0.0
        for j in range(dst_ninf):
            si = dst_to_src[j]
            if 0 <= si < src_ninf_eff:
                val = src_w[src_base + si]
            else:
                val = 0.0
            dst_w[dst_base + j] = val
            total += val

        if normalize and total > 1e-9:
            inv = 1.0 / total
            for j in range(dst_ninf):
                dst_w[dst_base + j] *= inv

    if verbose:
        print(f"[copy] build done in {time.time()-t2:.2f}s")

    if verbose:
        print(f"[copy] writing {num_verts} verts x {dst_ninf} inf ...")
    t3 = time.time()
    _write_skin_weights(dst_skin, dst_shape, dst_w, num_verts, dst_ninf)
    if verbose:
        print(f"[copy] write done in {time.time()-t3:.2f}s")

    cmds.dgdirty(dst_skin)

    elapsed = time.time() - t0
    if verbose:
        print(f"[copy] total {elapsed:.2f}s")

    return {
        'source_skin': src_skin,
        'target_skin': dst_skin,
        'num_verts': num_verts,
        'src_influences': src_ninf,
        'dst_influences': dst_ninf,
        'matched': matched,
        'elapsed': elapsed,
    }


def copySkinWeightsToOptimizedFromSelection(normalize=True):
    sel = cmds.ls(sl=True, transforms=True)
    if len(sel) < 2:
        raise RuntimeError(
            "请先选择 2 个物体：source (带 skinCluster) + "
            "target (带 optimizedSkinCluster)")
    return copySkinWeightsToOptimized(sel[0], sel[1], normalize=normalize)


def install_mel_commands():
    import maya.mel as mel
    code = r'''
global proc copySkinWeightsToOptimized(string $src, string $dst) {
    python("import copy_skin_weights; "
         + "copy_skin_weights.copySkinWeightsToOptimized('"
         + $src + "', '" + $dst + "')");
}
global proc copySkinWeightsToOptimizedFromSelection() {
    python("import copy_skin_weights; "
         + "copy_skin_weights.copySkinWeightsToOptimizedFromSelection()");
}
'''
    mel.eval(code)
    print("[copy_skin_weights] MEL 命令已安装")


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        copySkinWeightsToOptimized(sys.argv[1], sys.argv[2])
    else:
        copySkinWeightsToOptimizedFromSelection()