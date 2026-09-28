# -*- coding: utf-8 -*-
"""
计算 optimizedSkinCluster 的 CoR（相似权重顶点组的质心）
用法：compute_coR("optimizedSkinCluster1", weight_epsilon=0.05)
"""
import maya.cmds as cmds


def compute_coR(node, weight_epsilon=0.05):
    # 1. 拿到 mesh shape
    conns = cmds.listConnections(f'{node}.input[0].inputGeometry',
                                  s=True, d=False)
    if not conns:
        print(f"错误：节点 {node} 没有连接几何体")
        return
    shape = conns[0]
    num_vtx = cmds.polyEvaluate(shape, vertex=True)
    print(f"[compute_coR] mesh={shape}, num_vtx={num_vtx}")

    # 2. 读每个顶点的权重
    weights = []
    for i in range(num_vtx):
        w_dict = {}
        try:
            indices = cmds.getAttr(f'{node}.weightList[{i}].weights',
                                    multiIndices=True) or []
            for j in indices:
                w = cmds.getAttr(f'{node}.weightList[{i}].weights[{j}]')
                if w > 1e-6:
                    w_dict[j] = w
        except Exception as e:
            print(f"读取 vtx{i} 权重失败: {e}")
        weights.append(w_dict)

    # 3. 读每个顶点在静止姿势下的世界坐标
    positions = []
    for i in range(num_vtx):
        p = cmds.xform(f'{shape}.vtx[{i}]', q=True, ws=True, t=True)
        positions.append(p)

    # 4. 分组：相似权重在同一组
    def similar(w1, w2):
        keys = set(w1.keys()) | set(w2.keys())
        for k in keys:
            if abs(w1.get(k, 0.0) - w2.get(k, 0.0)) > weight_epsilon:
                return False
        return True

    groups = []
    for i in range(num_vtx):
        found = False
        for ref_w, verts in groups:
            if similar(weights[i], ref_w):
                verts.append(i)
                found = True
                break
        if not found:
            groups.append((dict(weights[i]), [i]))

    # 5. 每组算质心
    cors = {}
    for ref_w, verts in groups:
        cx = sum(positions[v][0] for v in verts) / len(verts)
        cy = sum(positions[v][1] for v in verts) / len(verts)
        cz = sum(positions[v][2] for v in verts) / len(verts)
        for v in verts:
            cors[v] = (cx, cy, cz)

    print(f"[compute_coR] 分组完成：{len(groups)} 组")
    for gi, (ref_w, verts) in enumerate(groups):
        w_str = " ".join(f"j{k}={v:.2f}" for k, v in sorted(ref_w.items()))
        c = cors[verts[0]]
        print(f"  组{gi}: [{w_str}]  "
              f"CoR=({c[0]:.3f},{c[1]:.3f},{c[2]:.3f})  "
              f"顶点数={len(verts)}")

    # 6. 写入 coRList
    #    ★ 用 evalDeferred 或一次性块写入，避免触发多次重算
    print(f"[compute_coR] 开始写入 {num_vtx} 个 CoR...")
    for i in range(num_vtx):
        cx, cy, cz = cors[i]
        try:
            cmds.setAttr(f'{node}.coRList[{i}].coRX', cx)
            cmds.setAttr(f'{node}.coRList[{i}].coRY', cy)
            cmds.setAttr(f'{node}.coRList[{i}].coRZ', cz)
        except Exception as e:
            print(f"❌ 写入 coRList[{i}] 失败: {e}")
            return

    print(f"[compute_coR] ✓ 写入完成")