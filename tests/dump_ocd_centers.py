# -*- coding: utf-8 -*-
"""Dump structural_centers / outer_participation reference values for C++ port."""
import json
import math
import sys

import numpy as np

sys.path.insert(0, "tests")
import ocd_elbow_reference_new as ref

HALF, HALF_RINGS, SIDES = ref.HALF, ref.HALF_RINGS, ref.SIDES


def quat_from_bend(bend, twist=0.0, direction=0.0):
    return ref.joint_quaternion(bend, twist, direction)


def build_grid():
    """Cylinder like the preview mesh: rings x sides + 2 poles."""
    points, faces = ref.cylinder()
    return points, faces


def cases():
    out = []
    bind, faces = build_grid()
    edges = ref.local_edges(bind, faces)
    stations = np.asarray(bind)[:, 0]
    for bend in (60, 90, 118, 145):
        for twist in (0, 35):
            for direction in (0, -60):
                w = np.array([[1.0 - ref.skin_weight(s), ref.skin_weight(s)]
                              for s in stations])
                rotations = [ref.joint_quaternion(0, 0, 0),
                             quat_from_bend(bend, twist, direction)]
                pose = ref.Elbow(bend, twist, direction, ref.ZERO, False)
                original = np.asarray(pose.mesh(bind))
                guide = ref.hinge_displacement(bind, original, w, bend, twist, direction)
                outer = ref.outer_participation(bind, w, bend, twist, direction)
                delta = ref.structural_centers(edges, original, w, rotations,
                                               guide, outer, 16.0)
                out.append({
                    "bend": bend, "twist": twist, "direction": direction,
                    "rotations": [list(map(float, q)) for q in rotations],
                    "weights": [[float(x) for x in row] for row in w],
                    "bind": [[float(x) for x in p] for p in bind],
                    "original": [[float(x) for x in p] for p in original],
                    "guide": [[float(x) for x in p] for p in guide],
                    "outer": [float(x) for x in outer],
                    "delta": [[float(x) for x in p] for p in delta],
                    "faces": [list(map(int, f)) for f in faces],
                })
    return out


data = {"cases": cases()}
with open("ocd_centers_dump.json", "w", encoding="utf-8") as f:
    json.dump(data, f)
print("cases:", len(data["cases"]))
print("verts:", len(data["cases"][0]["bind"]))
print("edges:", len(ref.local_edges(build_grid()[0], build_grid()[1])[0]))
