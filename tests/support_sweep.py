"""support 扫描：找内侧中线转角、邻环新尖峰、外侧支撑代价的折中点。

只改 support，保留原厚度梯度与中心求解参数。
三档弯角：118 / 120 / 146.1265。

    python tests/support_sweep.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

HERE = r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests"
R = 1.0
SIDES = ref.SIDES
RINGS = 2 * ref.HALF_RINGS + 1
RING_STEP = 2.0 * ref.HALF / (RINGS - 1)


def hinge_section_sup(station, depth, radius, angle, support):
    station = np.atleast_1d(np.asarray(station, float))
    depth = np.atleast_1d(np.asarray(depth, float))
    distance = np.abs(station)
    u = np.minimum(distance / support, 1.0)
    raw = np.maximum(depth, 0.0) * np.tan(angle / 2)
    limit = 0.95 * min(support, 2.6 * radius)
    shift = raw / np.sqrt(1.0 + (raw / limit) ** 2) if limit > 1e-12 else np.zeros_like(raw)
    inner_travel = shift * (1.0 - u)
    release = 0.28 * math.sin(angle / 2) ** 2 * (4.0 * u * (1.0 - u)) ** 2
    outward = distance + inner_travel * (1.0 - release)
    height = depth.astype(float).copy()
    if angle > 1e-8:
        height = height + (shift / math.tan(angle / 2) - np.maximum(depth, 0.0)) \
            * (1.0 - u) * (1.0 - release)
    outer_radius = np.maximum(-depth, 0.0)
    half = angle / 2
    fillet_radius = outer_radius * (1.0 + 0.45 * math.cos(half))
    tangent_station = 0.45 * outer_radius * np.sin(half)
    arc_length = fillet_radius * half
    extra = arc_length - tangent_station
    side = (depth < 0).astype(float)
    gather = 0.85 * math.sin(half) * side ** 2
    opening = 0.32 * distance * math.sin(half) * side ** 2 * (1.0 - u) ** 5
    material = distance * (1.0 - gather * (1.0 - u) ** 2) + opening
    join = u ** 3 * (4.0 - 3.0 * u)
    progress = material + extra * join
    remaining = np.maximum(arc_length - progress, 0.0)
    phi = np.where(fillet_radius > 0, remaining / np.where(fillet_radius > 0, fillet_radius, 1), 0)
    arc_station = tangent_station - fillet_radius * np.sin(phi) + np.maximum(progress - arc_length, 0)
    arc_height = depth + 2 * fillet_radius * np.sin(phi / 2) ** 2
    outward = np.where(depth < 0, arc_station, outward)
    height = np.where(depth < 0, arc_height, height)
    x = np.where(station <= 0, -outward, np.cos(angle) * outward - np.sin(angle) * height)
    h = np.where(station <= 0, height, np.sin(angle) * outward + np.cos(angle) * height)
    return np.column_stack((x, h))


def build_guide(bend, stations, W, cor_l, bind_l, support):
    angle = np.radians(abs(bend))
    ext = np.full_like(stations, R)
    inner = hinge_section_sup(stations, ext, R, angle, support)
    outer = hinge_section_sup(stations, -ext, R, angle, support)
    neutral = (inner + outer) * 0.5
    grad = (inner - outer) / (2.0 * R)
    depth = bind_l[:, 1]
    lateral = bind_l[:, 2]
    x = neutral[:, 0] + depth * grad[:, 0]
    h = neutral[:, 1] + depth * grad[:, 1]
    return np.column_stack([x, h, lateral]) - cor_l


def three_pt(prof, row0):
    if row0 <= 0 or row0 >= RINGS - 1:
        return None
    a, b, c = prof[(row0 - 1) * SIDES], prof[row0 * SIDES], prof[(row0 + 1) * SIDES]
    v1, v2 = b - a, c - b
    cc = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(max(-1, min(1, cc))))


def main():
    d1 = json.load(open(HERE + r"\_la1.json"))
    bind = np.asarray(d1["bind"], float)
    faces = d1["faces"]
    W = np.asarray(d1["weights"], float)
    stations = bind[:, 0]
    bind_l = np.column_stack([bind[:, 0] / R, bind[:, 1] / R, bind[:, 2] / R])
    edges = ref.local_edges(bind_l.tolist(), faces)
    row0 = RINGS // 2

    sup_list = [4.064216, 3.6, 3.2, 3.0, 2.8, 2.6, 2.4, 2.2, 2.0]

    for bend_s, cor_list in d1["cor"].items():
        bend = float(bend_s)
        cor = np.asarray(cor_list, float)
        cor_l = np.column_stack([cor[:, 0] / R, cor[:, 1] / R, cor[:, 2] / R])
        q = ref.joint_quaternion(bend, 0.0, 0.0)
        rots = [(0.0, 0.0, 0.0, 1.0), q]
        orp = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))

        print("\n" + "=" * 92)
        print("bend=%.4f" % bend)
        print("=" * 92)
        print("  %7s %10s %10s %10s %10s %10s %10s"
              % ("support", "中线°", "±0.667°", "±1.333°", "±2.0°", "±2.667°", "外侧位移@-2"))

        for sup in sup_list:
            g = build_guide(bend, stations, W, cor_l, bind_l, sup)
            d = np.asarray(ref.structural_centers(edges, cor_l, W, rots, g, orp, 16.0))
            # 用 guide 轮廓本身衡量（最终网格需 Maya，这里先用引导）
            gi = cor_l + g
            mid = three_pt(gi, row0)
            r1 = abs(mid - three_pt(gi, row0 - 1)) if three_pt(gi, row0 - 1) else 0
            t067 = three_pt(gi, row0 - 1)
            t133 = three_pt(gi, row0 - 2)
            t200 = three_pt(gi, row0 - 3)
            t266 = three_pt(gi, row0 - 4)
            # 外侧位移 @ x=-2.0：用该环外侧极点在 guide 中的位移
            i_out = (row0 - 3) * SIDES + SIDES // 2
            disp_out = math.sqrt(sum(g[i_out][k] ** 2 for k in range(3)))
            print("  %7.3f %10.2f %10.2f %10.2f %10.2f %10.2f %10.4f"
                  % (sup, mid, t067, t133, t200, t266, disp_out))


if __name__ == "__main__":
    main()
