# -*- coding: utf-8 -*-
"""Generate the reference delta table used by tests/test_ocd_alg5.cpp.

Mirrors the reference preview's algorithm-5 pipeline, which is the only
definition of "OCD + Corrected CoR":

    original = Elbow(bend).mesh(bind)          # CoR output at zero offsets
    guide    = original + hinge_displacement(bind, original, weights, ...)
    outer    = outer_participation(bind, weights, ...)
    delta    = structural_centers(edges, original, weights, rotations, guide, outer)

The plug-in then evaluates CoR with (standard centres + delta).  Because CoR is
not an exact inverse of the centre parameterisation, the resulting shape only
approximates `guide`; the reference behaves the same way, so the observable
contract is the delta itself.

Usage:  python tests/dump_ocd_alg5.py [output.json]
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import ocd_elbow_reference_latest as ref  # noqa: E402


def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "ocd_alg5_reference.json")
    bind, faces = ref.cylinder()
    edges = ref.local_edges(bind, faces)
    cases = []
    for bend in (60, 90, 118, 145):
        weights = np.array([[1.0 - ref.skin_weight(p[0]), ref.skin_weight(p[0])] for p in bind])
        rotations = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(bend, 0, 0)]
        pose = ref.Elbow(bend)
        original = np.asarray(pose.mesh(bind))
        guide = original + ref.hinge_displacement(bind, original, weights, bend, 0, 0)
        outer = ref.outer_participation(bind, weights, bend, 0, 0)
        delta = ref.structural_centers(edges, original, weights, rotations, guide, outer, 16.0)
        cases.append({
            "bend": bend,
            "bind": bind,
            "faces": faces,
            "original": original.tolist(),
            "weights": weights.tolist(),
            "rotations": [list(map(float, q)) for q in rotations],
            "guide": guide.tolist(),
            "outer": outer.tolist(),
            "delta": delta.tolist(),
        })
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump({"cases": cases}, handle)
    print("wrote %s (%d cases, %d verts)" % (out_path, len(cases), len(bind)))


if __name__ == "__main__":
    main()
