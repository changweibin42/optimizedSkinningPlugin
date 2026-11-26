# -*- coding: utf-8 -*-
r"""把当前场景里的 OCD（算法 5）用例导出成一份可重放的 JSON。

不依赖 numpy，可直接在 Maya 脚本编辑器（Python 页签）里跑：

    import sys
    sys.path.insert(0, r"D:\optimizedSkinningPlugin\<目录>\scripts")
    import export_ocd_case
    export_ocd_case.main()

或指定路径：

    export_ocd_case.main(r"D:\tmp\ocd_case.json")

导出内容（同一个当前姿态）：

* 绑定网格：rest 顶点、面、对象矩阵
* **每个顶点的稀疏权重**：存成 {influenceIndex: weight}，不是按位置摊平
* 骨骼：matrix[] 的实际顺序、名字、当前世界矩阵、bindPreMatrix，
  以及由 bindPreMatrix 的逆算出的 bind 位置
* 铰链：插件自报的 parent / child / helper（ocdHingeReport）
* 节点属性：算法、中心限额、扭转权重、补偿通道、注入的中心偏移
* **算法 0 / 3 / 5 三份顶点输出**
* 按铰链轴分环的对照表

三个曾经出过错的地方，这里都按正确语义处理：

1. weightList[i].weights 是 multi 属性的**稀疏**数组，multiIndices 就是
   influence 索引。只存非零项，且必须带索引，不能按出现顺序当列。
2. bindPreMatrix 已经是「绑定世界矩阵的逆」，bind 位置直接读它的平移
   （行向量约定下在 12..14），不需要再求逆。
3. 环宽取相邻 station 的**中位间隔**，不是最小间隔；浮点噪声否则会把
   一个环拆成很多假环。
"""
import json
import math
import os
import sys

SKIN_TYPE = "optimizedSkinCluster"

CHANNEL_ATTRS = (
    "ocdInnerFalloff", "ocdInnerShift", "ocdInnerTilt",
    "ocdOuterFalloff", "ocdOuterShift", "ocdOuterTilt",
    "ocdBulge", "ocdCrease", "ocdTwist", "ocdStretch", "ocdProfile",
)
SCALAR_ATTRS = (
    "algorithmType", "corCenterLimit", "ocdCenterLimit", "ocdTwistWeight",
    "ocdAvoidOverlap", "corDebugLog", "ocdHingeParent", "ocdHingeChild",
    "ocdHingeReport",
)


def _attr(node, name):
    try:
        if not mc.attributeQuery(name, node=node, exists=True):
            return None
        return mc.getAttr("%s.%s" % (node, name))
    except Exception:
        return None


def _short(name):
    return name.split("|")[-1]


def _default_path():
    name = "ocd_case.json"
    candidates = []
    scene = mc.file(query=True, sceneName=True) or ""
    if scene:
        candidates.append(os.path.dirname(scene))
    try:
        project = mc.workspace(query=True, rootDirectory=True) or ""
        if project:
            candidates.append(project)
    except Exception:
        pass
    candidates.append(os.path.expanduser("~"))
    import tempfile
    candidates.append(tempfile.gettempdir())
    for directory in candidates:
        if not directory:
            continue
        try:
            candidate = os.path.join(directory, name)
            with open(candidate, "a"):
                pass
            os.remove(candidate)
            return candidate
        except Exception:
            continue
    return name


def _find_skin(preferred_shape=None):
    skins = mc.ls(type=SKIN_TYPE) or []
    if not skins:
        raise RuntimeError(
            "当前场景里没有 %s 节点。\n"
            "如果插件显示名不同，请把它加到 SKIN_TYPE；"
            "或者先在 Outliner 里选中蒙皮节点再运行。" % SKIN_TYPE)
    if preferred_shape:
        related = mc.listConnections(preferred_shape + ".inMesh", source=True,
                                     destination=False, type=SKIN_TYPE) or []
        if not related:
            related = [node for node in mc.listHistory(preferred_shape) or []
                       if mc.nodeType(node) == SKIN_TYPE]
        if related:
            return related[0]
    if len(skins) > 1:
        raise RuntimeError("场景里有 %d 个 %s，请先选中目标网格再运行。"
                           % (len(skins), SKIN_TYPE))
    return skins[0]


def _influences(skin):
    """[(matrixIndex, node), ...]，按 matrix[] 的实际下标排序。"""
    out = []
    for index in mc.getAttr(skin + ".matrix", multiIndices=True) or []:
        plug = "%s.matrix[%d]" % (skin, index)
        sources = mc.listConnections(plug, source=True, destination=False) or []
        if sources:
            out.append((int(index), sources[0]))
    out.sort()
    return out


def _rest_geometry(skin):
    """蒙皮的输入几何 = rest 位姿的网格。"""
    candidates = []
    for source in ("inputGeometry", "input[0].inputGeometry"):
        try:
            candidates = mc.listConnections(
                "%s.%s" % (skin, source), source=True, destination=False,
                shapes=True) or []
        except Exception:
            candidates = []
        if candidates:
            break
    if not candidates:
        raise RuntimeError("找不到蒙皮的输入几何（rest 网格）。")
    rest = candidates[0]
    if mc.nodeType(rest) == "mesh":
        parents = mc.listRelatives(rest, parent=True, fullPath=True) or []
        return rest, (parents[0] if parents else rest)
    shapes = mc.listRelatives(rest, shapes=True, noIntermediate=True,
                              fullPath=True) or [rest]
    return shapes[0], rest


def _mesh_arrays(shape, om):
    selection = om.MSelectionList()
    selection.add(shape)
    mesh = om.MFnMesh(selection.getDagPath(0))
    points = [[float(p.x), float(p.y), float(p.z)]
              for p in mesh.getPoints(om.MSpace.kObject)]
    count = mesh.numPolygons() if callable(mesh.numPolygons) else mesh.numPolygons
    faces = []
    for polygon in range(count):
        vertices = mesh.getPolygonVertices(polygon)
        faces.append([int(v) for v in vertices])
    return points, faces


def _evaluate(shape, algorithm, skin, om):
    original = mc.getAttr(skin + ".algorithmType")
    try:
        mc.setAttr(skin + ".algorithmType", algorithm)
        mc.dgeval(shape)
        mc.refresh(force=True)
        points, _ = _mesh_arrays(shape, om)
        return points
    finally:
        mc.setAttr(skin + ".algorithmType", original)
        mc.dgeval(shape)
        mc.refresh(force=True)


def _sparse_weights(skin, vertex_count):
    """返回 (sparse, dense)。

    weightList[v].weights 是**稀疏** multi 属性：multiIndices 给出实际的
    influence 索引，值按这些索引排列。旧版把它按出现顺序摊平成列，结果是
    顶点 0 只有 influence 1 有权重时，把 1.0 记到了第 0 列、又补一个 0.0
    到第 1 列——列与骨骼完全错位。
    """
    sparse = []      # 每顶点 {index: weight}
    for vertex in range(vertex_count):
        entry = {}
        try:
            plug = "%s.weightList[%d].weights" % (skin, vertex)
            indices = mc.getAttr(plug, multiIndices=True) or []
            for index in indices:
                try:
                    value = mc.getAttr("%s[%d]" % (plug, index))
                except Exception:
                    continue
                if isinstance(value, (list, tuple)):
                    value = value[0] if value else 0.0
                value = float(value)
                if abs(value) > 0.0:
                    entry[int(index)] = value
        except Exception:
            entry = {}
        sparse.append(entry)
    return sparse


def _mat_inverse(m):
    """4x4 行主序矩阵求逆（Maya MMatrix 约定）。"""
    a = [[float(m[r * 4 + c]) for c in range(4)] for r in range(4)]
    # 增广矩阵
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(4)]
           for i, row in enumerate(a)]
    for col in range(4):
        pivot = max(range(col, 4), key=lambda r: abs(aug[r][col]))
        if abs(aug[pivot][col]) < 1e-12:
            raise ValueError("bindPreMatrix 奇异，无法求逆")
        aug[col], aug[pivot] = aug[pivot], aug[col]
        scale = aug[col][col]
        aug[col] = [v / scale for v in aug[col]]
        for row in range(4):
            if row == col:
                continue
            factor = aug[row][col]
            if factor != 0.0:
                aug[row] = [v - factor * w for v, w in zip(aug[row], aug[col])]
    return [aug[r][4 + c] for r in range(4) for c in range(4)]


def _bind_positions(bones):
    """由 bindPreMatrix 的逆求 bind 世界矩阵，再读它的位置。

    实测确认：bindPreMatrix == 绑定姿态的 worldInverseMatrix，所以它的
    平移是位置取负（shoulder 在 x=-6 时 bindPreMatrix 的平移是 +6）。
    直接读 bindPreMatrix 的平移会得到反号的坐标，必须先求逆。
    """
    for bone in bones:
        world = _mat_inverse(bone["bindPreMatrix"])
        bone["bindWorldMatrix"] = world
        bone["bindPosition"] = [world[12], world[13], world[14]]
        # bind 世界矩阵的第一列 = 该骨绑定姿态的局部 X 轴（行向量约定）。
        axis = [world[0], world[4], world[8]]
        length = math.sqrt(sum(v * v for v in axis))
        bone["bindAxis"] = ([v / length for v in axis] if length > 1e-9
                            else [1.0, 0.0, 0.0])
    return bones


def _hinge_frame(bones, report):
    parent = None
    if report:
        for bone in bones:
            if ("parent=" + bone["short"]) in report:
                parent = bone
                break
        if parent is None:
            for bone in bones:
                if bone["short"] in report:
                    parent = bone
                    break
    axis, origin = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if parent is not None:
        axis = list(parent["bindAxis"])
        origin = list(parent["bindPosition"])
    return axis, origin, (parent["short"] if parent else None)


def _ring_table(bind_points, axis, origin, cor_points, final_points,
                sparse, ring_hint=None):
    stations = [sum((point[i] - origin[i]) * axis[i] for i in range(3))
                for point in bind_points]
    if not stations:
        return []
    uniq = sorted({round(s, 5) for s in stations})
    gaps = [b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6]
    if not gaps:
        return []
    # 用**中位**间隔，不是最小间隔：浮点噪声否则会拆出几百个假环。
    gaps.sort()
    width = gaps[len(gaps) // 2]
    if width <= 0:
        return []
    buckets = {}
    for vertex, station in enumerate(stations):
        key = int(math.floor((station - uniq[0]) / width + 0.5))
        buckets.setdefault(key, []).append(vertex)

    def distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    table = []
    for key in sorted(buckets):
        members = buckets[key]
        total = [distance(final_points[v], bind_points[v]) for v in members]
        cor = [distance(cor_points[v], bind_points[v]) for v in members]
        extra = [distance(final_points[v], cor_points[v]) for v in members]
        # 有效权重：该环上出现过的 influence 权重范围，能直接看出单骨/混权。
        counts = {}
        for index in sparse[members[0]]:
            counts[index] = []
        for vertex in members:
            for index, value in sparse[vertex].items():
                counts.setdefault(index, []).append(value)
        spread = {}
        for index, values in counts.items():
            if values:
                spread[str(index)] = [min(values), max(values)]
        mean = lambda values: sum(values) / len(values)
        table.append({
            "ring": key,
            "station": mean([stations[v] for v in members]),
            "count": len(members),
            "meanTotalShift": mean(total),
            "maxTotalShift": max(total),
            "meanCorShift": mean(cor),
            "meanOcdExtra": mean(extra),
            "maxOcdExtra": max(extra),
            "weightSpread": spread,
            "mixed": int(sum(1 for v in members if len(sparse[v]) > 1)),
            "vertices": members,
        })
    for previous, current in zip(table, table[1:]):
        span = current["station"] - previous["station"]
        scale = 1.0 / span if abs(span) > 1e-9 else 0.0
        current["slopeTotal"] = (current["meanTotalShift"]
                                 - previous["meanTotalShift"]) * scale
        current["slopeCor"] = (current["meanCorShift"]
                               - previous["meanCorShift"]) * scale
        current["slopeOcd"] = (current["meanOcdExtra"]
                               - previous["meanOcdExtra"]) * scale
    for row in table:
        row.setdefault("slopeTotal", 0.0)
        row.setdefault("slopeCor", 0.0)
        row.setdefault("slopeOcd", 0.0)
    return table


def main(path=None):
    global mc
    import maya.cmds as cmds
    import maya.api.OpenMaya as om
    mc = cmds

    target = path or _default_path()

    preferred = None
    for item in mc.ls(selection=True, long=True) or []:
        if mc.nodeType(item) in ("mesh", "transform"):
            shapes = mc.listRelatives(item, shapes=True, noIntermediate=True,
                                      fullPath=True) or []
            preferred = shapes[0] if shapes else item
            break

    skin = _find_skin(preferred)
    deformed = None
    for node in mc.listConnections(skin + ".outputGeometry", source=False,
                                   destination=True, shapes=True) or []:
        deformed = node
        break
    if deformed is None:
        raise RuntimeError("找不到 %s 的输出网格。" % skin)

    rest_shape, rest_transform = _rest_geometry(skin)
    bind_points, faces = _mesh_arrays(rest_shape, om)
    count = len(bind_points)

    cor_points = _evaluate(deformed, 0, skin, om)
    ocd_points = _evaluate(deformed, 3, skin, om)
    final_points = _evaluate(deformed, 5, skin, om)
    if not (len(cor_points) == count == len(final_points)):
        raise RuntimeError("顶点数不一致：rest=%d cor=%d final=%d"
                           % (count, len(cor_points), len(final_points)))

    bones = []
    for index, node in _influences(skin):
        bones.append({
            "index": index,
            "name": node,
            "short": _short(node),
            "worldMatrix": [float(v) for v in
                            mc.getAttr("%s.worldMatrix[0]" % node)],
            "bindPreMatrix": [float(v) for v in
                              mc.getAttr("%s.bindPreMatrix[%d]" % (skin, index))],
        })
    _bind_positions(bones)

    attributes = {}
    for name in SCALAR_ATTRS:
        value = _attr(skin, name)
        if value is not None:
            attributes[name] = value
    channels = {}
    for name in CHANNEL_ATTRS:
        value = _attr(skin, name)
        if value is not None:
            channels[name] = value

    report = attributes.get("ocdHingeReport") or ""
    axis, origin, hinge_parent = _hinge_frame(bones, report)

    sparse = _sparse_weights(skin, count)
    rings = _ring_table(bind_points, axis, origin, cor_points, final_points,
                        sparse)

    payload = {
        "mayaVersion": mc.about(version=True),
        "scene": mc.file(query=True, sceneName=True) or "(untitled)",
        "skin": skin,
        "shapes": {"rest": rest_shape, "deformed": deformed},
        "mesh": {"vertexCount": count, "faces": faces,
                 "bindPoints": bind_points,
                 "worldMatrix": [float(v) for v in
                                 mc.getAttr(rest_transform + ".worldMatrix[0]")]},
        "bones": bones,
        "weightsSparse": [{str(k): v for k, v in row.items()} for row in sparse],
        "attributes": attributes,
        "channels": channels,
        "ocdCenterOffsets": _attr(skin, "ocdCenterOffsets"),
        "hinge": {"report": report, "axis": axis, "origin": origin,
                  "parent": hinge_parent},
        "points": {"cor": cor_points, "ocd": ocd_points, "final": final_points},
        "rings": rings,
    }

    with open(target, "w") as handle:
        json.dump(payload, handle)

    # ---- 控制台摘要 ----
    print("=" * 80)
    print("导出:", target, "(%.2f MB)" % (os.path.getsize(target) / 1048576.0))
    print("场景:", payload["scene"])
    print("网格: %d 顶点 / %d 面   蒙皮: %s" % (count, len(faces), skin))
    print("骨骼 (%d):" % len(bones))
    for bone in bones:
        print("  [%d] %-22s bind=%s" % (bone["index"], bone["short"],
                                        [round(v, 3) for v in bone["bindPosition"]]))
    print("铰链报告:", report or "(空：把 corDebugLog 设为 1，再刷新一次视图)")
    print("铰链轴/原点:", [round(v, 3) for v in axis],
          [round(v, 3) for v in origin])
    print("中心限额:", attributes.get("ocdCenterLimit", "n/a"),
          " 扭转权重:", attributes.get("ocdTwistWeight", "n/a"))

    # 稀疏权重的真实形状
    sizes = {}
    for row in sparse:
        sizes[len(row)] = sizes.get(len(row), 0) + 1
    print("每顶点有效 influence 数分布:", dict(sorted(sizes.items())))
    per_influence = {}
    for row in sparse:
        for index in row:
            per_influence[index] = per_influence.get(index, 0) + 1
    print("每个 influence 影响到的顶点数:", dict(sorted(per_influence.items())))

    print("-" * 80)
    print("%4s %8s %5s %8s %9s %9s %9s %6s"
          % ("环", "station", "顶点", "|cor-B|", "|fin-B|", "OCD增量", "混权数", "环宽"))
    for row in rings:
        print("%4d %8.3f %5d %8.4f %9.4f %9.4f %9d"
              % (row["ring"], row["station"], row["count"],
                 row["meanCorShift"], row["meanTotalShift"],
                 row["meanOcdExtra"], row["mixed"]))
    print("=" * 80)
    print("把这个 JSON 发给我。")
    return target


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
else:
    try:
        main()
    except Exception as _exc:  # noqa: BLE001
        import traceback
        traceback.print_exc()
        print("导出失败，请检查是否选中了带 OCD 蒙皮的网格：", _exc)
