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
* 每个顶点的完整权重列
* 骨骼：matrix[] 的实际顺序、名字、当前世界矩阵、bindPreMatrix
* 铰链：插件自报的 parent / child / helper（ocdHingeReport）
* 节点属性：算法、中心限额、扭转权重、11 个补偿通道、注入的中心偏移
* **算法 0 / 3 / 5 三份顶点输出**：区分「基础 CoR 已折」与「引导位移没实现」
* 按铰链轴分环的对照表：环距不均匀时归一化的斜率才是「开始弯的速度」

注意：绑定点取自蒙皮的**输入几何**（rest 位姿），不是当前变形结果。
"""
import json
import math
import os
import sys

SKIN_TYPE = "optimizedSkinCluster"

# 11 个补偿通道，导出全部属性便于核对。
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


def _default_path():
    """挑一个可写的位置存 JSON。

    依次尝试：场景所在目录、Maya 项目目录、用户主目录、系统临时目录。
    在受限环境里主目录可能不可写，所以不能只写死一个。
    """
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


def _attr(node, name):
    try:
        if not mc.attributeQuery(name, node=node, exists=True):
            return None
        return mc.getAttr("%s.%s" % (node, name))
    except Exception:
        return None


def _short(name):
    return name.split("|")[-1]


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
            # 选中网格时，蒙皮节点可能挂在它的历史里。
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
    # API 2.0 里 numPolygons 是属性；1.0 才是方法。
    count = mesh.numPolygons() if callable(mesh.numPolygons) else mesh.numPolygons
    faces = []
    for polygon in range(count):
        # API 2.0: getPolygonVertices(polygon) -> tuple
        vertices = mesh.getPolygonVertices(polygon)
        faces.append([int(v) for v in vertices])
    return points, faces


def _evaluate(shape, algorithm, skin, om):
    """把 algorithmType 临时改成目标值，取出该算法的对象空间顶点。"""
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


def _flatten(values):
    """Maya 有时把 multi 属性读成嵌套元组/列表，摊平成一维 float。"""
    out = []
    stack = [values] if not isinstance(values, (list, tuple)) else list(values)
    while stack:
        item = stack.pop(0)
        if isinstance(item, (list, tuple)):
            stack = list(item) + stack
        else:
            out.append(float(item))
    return out


def _weights(skin, vertex_count):
    rows = []
    for vertex in range(vertex_count):
        try:
            row = _flatten(
                mc.getAttr("%s.weightList[%d].weights" % (skin, vertex)))
        except Exception:
            row = []
        rows.append(row)
    width = max((len(row) for row in rows), default=0)
    return [row + [0.0] * (width - len(row)) for row in rows]


def _hinge_frame(bones, report):
    """铰链父骨的绑定 X 轴与位置（从 bindPreMatrix 的逆读取）。

    bindPreMatrix 是 bind 世界矩阵的逆，行主序，平移在 12..14。
    """
    parent = None
    if report:
        for bone in bones:
            if ("parent=" + _short(bone["name"])) in report.replace("|", ""):
                parent = bone
                break
        if parent is None:
            for bone in bones:
                if _short(bone["name"]) in report:
                    parent = bone
                    break
    if parent is None and bones:
        parent = bones[0]

    axis, origin = [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]
    if parent is not None:
        inverse = mc.getAttr("%s.bindPreMatrix[%d]" % (parent["_skin"],
                                                      parent["index"]))
        origin = [float(inverse[12]), float(inverse[13]), float(inverse[14])]
        # 逆矩阵的第一行 = 原始 bind 矩阵第一列（Maya 行向量约定下的 X 轴）。
        candidate = [float(inverse[0]), float(inverse[1]), float(inverse[2])]
        length = math.sqrt(sum(v * v for v in candidate))
        if length > 1e-9:
            axis = [v / length for v in candidate]
    return axis, origin


def _ring_table(bind_points, axis, origin, cor_points, final_points):
    """按铰链轴上的 station 分环，给出每环位移与归一化斜率。"""
    stations = [sum((point[i] - origin[i]) * axis[i] for i in range(3))
                for point in bind_points]
    if not stations:
        return []
    ordered = sorted(stations)
    gaps = [b - a for a, b in zip(ordered, ordered[1:]) if b - a > 1e-9]
    if not gaps:
        return []
    width = min(gaps) * 0.5
    buckets = {}
    for vertex, station in enumerate(stations):
        key = int(math.floor(station / width + 0.5))
        buckets.setdefault(key, []).append(vertex)

    def distance(a, b):
        return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))

    table = []
    for key in sorted(buckets):
        members = buckets[key]
        total = [distance(final_points[v], bind_points[v]) for v in members]
        cor = [distance(cor_points[v], bind_points[v]) for v in members]
        extra = [distance(final_points[v], cor_points[v]) for v in members]
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

    cor_points = _evaluate(deformed, 0, skin, om)
    ocd_points = _evaluate(deformed, 3, skin, om)
    final_points = _evaluate(deformed, 5, skin, om)
    if not (len(cor_points) == len(bind_points) == len(final_points)):
        raise RuntimeError("顶点数不一致：rest=%d cor=%d final=%d"
                           % (len(bind_points), len(cor_points), len(final_points)))

    bones = []
    for index, node in _influences(skin):
        entry = {"index": index, "name": node, "short": _short(node),
                 "_skin": skin}
        entry["worldMatrix"] = [float(v) for v in
                                mc.getAttr("%s.worldMatrix[0]" % node)]
        entry["bindPreMatrix"] = [float(v) for v in
                                  mc.getAttr("%s.bindPreMatrix[%d]" % (skin, index))]
        bones.append(entry)

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
    axis, origin = _hinge_frame(bones, report)
    for bone in bones:
        bone.pop("_skin", None)

    weights = _weights(skin, len(bind_points))
    rings = _ring_table(bind_points, axis, origin, cor_points, final_points)

    payload = {
        "mayaVersion": mc.about(version=True),
        "scene": mc.file(query=True, sceneName=True) or "(untitled)",
        "skin": skin,
        "shapes": {"rest": rest_shape, "deformed": deformed},
        "mesh": {"vertexCount": len(bind_points), "faces": faces,
                 "bindPoints": bind_points,
                 "worldMatrix": [float(v) for v in
                                 mc.getAttr(rest_transform + ".worldMatrix[0]")]},
        "bones": bones,
        "weights": weights,
        "attributes": attributes,
        "channels": channels,
        "ocdCenterOffsets": _attr(skin, "ocdCenterOffsets"),
        "hinge": {"report": report, "axis": axis, "origin": origin},
        "points": {"cor": cor_points, "ocd": ocd_points, "final": final_points},
        "rings": rings,
    }

    with open(target, "w") as handle:
        json.dump(payload, handle)

    print("=" * 78)
    print("导出:", target, "(%.2f MB)" % (os.path.getsize(target) / 1048576.0))
    print("场景:", payload["scene"])
    print("网格: %d 顶点 / %d 面   蒙皮: %s" % (len(bind_points), len(faces), skin))
    print("骨骼 (%d):" % len(bones))
    for bone in bones:
        print("  [%d] %s" % (bone["index"], bone["short"]))
    print("铰链报告:", report or "(空：把 corDebugLog 设为 1，再刷新一次视图)")
    print("中心限额:", attributes.get("ocdCenterLimit", "n/a"),
          " 扭转权重:", attributes.get("ocdTwistWeight", "n/a"))
    print("注入的中心偏移:", "有 %d 个数" % len(payload["ocdCenterOffsets"])
          if payload["ocdCenterOffsets"] else "无")
    print("-" * 78)
    print("%3s %9s %5s %9s %9s %9s %9s %9s"
          % ("环", "station", "顶点", "总位移", "CoR位移", "OCD增量",
             "斜率总", "斜率OCD"))
    for row in rings:
        print("%3d %9.4f %5d %9.4f %9.4f %9.4f %9.4f %9.4f"
              % (row["ring"], row["station"], row["count"],
                 row["meanTotalShift"], row["meanCorShift"],
                 row["meanOcdExtra"], row["slopeTotal"], row["slopeOcd"]))
    print("=" * 78)
    print("把这个 JSON 发给我。看「斜率总 / 斜率OCD」哪几环突变就是红圈。")
    return target


# 既支持 mayapy 命令行，也支持把整个文件内容粘进 Maya 脚本编辑器执行。
# 粘进去时 __name__ 不是 "__main__"，所以这里无条件尝试一次。
if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
else:
    try:
        main()
    except Exception as _exc:  # noqa: BLE001 - 让脚本编辑器里能看到原因
        import traceback
        traceback.print_exc()
        print("导出失败，请检查是否选中了带 OCD 蒙皮的网格：", _exc)
