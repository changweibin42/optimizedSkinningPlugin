# -*- coding: utf-8 -*-
"""Dump reference values from ocd_elbow_reference.py for C++ port comparison."""
import json
import math
import sys

sys.path.insert(0, "tests")
import ocd_elbow_reference_new as ref

out = {}


def dump_elbow():
    rows = []
    for bend in (-150, -90, -35, 0, 35, 90, 118, 145, 150):
        for twist in (0, 35, -70):
            for direction in (0, 37, -80):
                for avoid in (False, True):
                    for values in (ref.ZERO, ref.VIDEO, ref.example_values(0, True)):
                        pose = ref.Elbow(bend, twist, direction, values, avoid)
                        for station in (-6.0, -5.0, -3.0, -1.0, -0.5, 0.0, 0.5, 1.0, 3.0, 5.0, 6.0):
                            for y, z in ((1.0, 0.0), (0.0, 1.0), (-1.0, 0.0), (0.5, -0.3), (0.0, 0.0)):
                                p = pose.point((station, y, z))
                                rows.append([bend, twist, direction, int(avoid), station, y, z,
                                             p[0], p[1], p[2], list(values)])
    out["elbow"] = rows


def hinge_case(bend, twist, direction, stations, depths, laterals, weights):
    """weights: list of (parent, child, helper)."""
    import numpy as np
    angle = abs(math.radians(bend))
    w = np.asarray(weights, dtype=float)
    mixed = np.count_nonzero(w > 0, axis=1) > 1
    distance = np.abs(np.asarray(stations, dtype=float))
    radius = 1.0
    reach = max(radius, distance[mixed].max()) if mixed.any() else radius
    rigid_edges = distance[(~mixed) & (distance > reach)]
    support = rigid_edges.min() if rigid_edges.size else reach + radius
    support = min(support, 2.6 * radius) + 1.6 * radius * math.sin(angle / 2) ** 2

    plane = math.radians(direction + (180 if bend < 0 else 0))
    phase = math.radians(twist) * w[:, 1]
    if w.shape[1] > 2:
        phase += math.radians(twist) * 0.35 * w[:, 2]
    s = np.asarray(stations, dtype=float)
    y = np.asarray(depths, dtype=float)
    z = np.asarray(laterals, dtype=float)
    depth = y * np.cos(phase - plane) - z * np.sin(phase - plane)
    lateral = y * np.sin(phase - plane) + z * np.cos(phase - plane)
    ext = np.full_like(s, radius)
    inner = ref.hinge_section(s, ext, radius, w, angle)
    outer = ref.hinge_section(s, -ext, radius, w, angle)
    neutral = (inner + outer) * 0.5
    gradient = (inner - outer) / (2 * radius)
    x, h = (neutral + depth[:, None] * gradient).T
    target = np.column_stack((x, h * math.cos(plane) - lateral * math.sin(plane),
                              h * math.sin(plane) + lateral * math.cos(plane)))
    return support, target.tolist()


def dump_hinge():
    rows = []
    stations = [-4.0, -2.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.5, 4.0]
    depths = [1.0, -1.0, 0.35, -0.6, 0.0, 1.0, -1.0, 0.2, -0.4]
    laterals = [0.0, 0.0, 0.7, -0.5, 0.0, 0.3, -0.2, 0.9, 0.0]
    for bend in (-145, -60, 0, 60, 118, 150):
        for twist in (0, 45):
            for direction in (0, -60):
                for weights in (
                    [(1.0, 0.0, 0.0)] * 9,
                    [(0.5, 0.5, 0.0)] * 9,
                    [(0.2, 0.6, 0.2), (0.9, 0.1, 0.0), (0.1, 0.9, 0.0),
                     (1.0, 0.0, 0.0), (0.5, 0.5, 0.0), (0.3, 0.7, 0.0),
                     (0.0, 1.0, 0.0), (0.6, 0.4, 0.0), (0.8, 0.2, 0.0)],
                ):
                    support, target = hinge_case(bend, twist, direction,
                                                 stations, depths, laterals, weights)
                    rows.append({
                        "bend": bend, "twist": twist, "direction": direction,
                        "stations": stations, "depths": depths, "laterals": laterals,
                        "weights": weights, "support": support, "target": target,
                    })
    out["hinge"] = rows


def dump_section():
    import numpy as np
    rows = []
    for bend in (-150, -90, 0, 45, 90, 118, 150):
        angle = abs(math.radians(bend))
        for depth in (-1.0, -0.25, 0.0, 0.25, 1.0):
            for station in (-6.0, -3.0, -1.0, 0.0, 1.0, 3.0, 6.0):
                w = np.asarray([[1.0, 0.0, 0.0]])
                p = ref.hinge_section(np.asarray([station]), np.asarray([depth]),
                                     1.0, w, angle)
                rows.append([bend, depth, station, float(p[0][0]), float(p[0][1])])
    out["section"] = rows


dump_elbow()
dump_hinge()
dump_section()
with open("ocd_reference_dump.json", "w", encoding="utf-8") as f:
    json.dump(out, f)
print("elbow rows:", len(out["elbow"]))
print("hinge cases:", len(out["hinge"]))
print("section rows:", len(out["section"]))
