# -*- coding: utf-8 -*-
"""逐级对拍的另一端：用 1285f9a 参考实现算出 guide 与 centerOffset。

权重不再自行推导，而是直接消费 `maya_ocd_alg5_dump.py --dump` 导出的
实际写入节点的权重矩阵（以及 boneCount、helper 旋转），保证两端用的是
同一套数据。

用法：
    python tests/ocd_alg5_truth.py tests/_alg5_original.json tests/_alg5_truth.json
"""
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import ocd_elbow_reference_latest as ref


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        return 2
    source = sys.argv[1]
    target = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "_alg5_truth.json")

    with open(source) as handle:
        payload = json.load(handle)

    bind = payload["bind"]
    faces = payload["faces"]
    weights = np.asarray(payload["weights"], dtype=float)
    bone_count = int(payload["boneCount"])
    if weights.shape[0] != len(bind):
        print("FAIL: 权重行数与顶点数不一致")
        return 1
    if weights.shape[1] != bone_count:
        print("FAIL: 权重列数与 boneCount 不一致")
        return 1

    edges = ref.local_edges(bind, faces)
    out = {"boneCount": bone_count,
           "weights": payload["weights"],
           "cases": {}}

    for key, case in payload["cases"].items():
        bend = case["bend"]
        original = np.asarray(case["original"])

        rotations = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(bend, 0, 0)]
        if bone_count >= 3:
            helper_rotation = case.get("helperRotation")
            if helper_rotation is None:
                print("FAIL: 三骨场景缺少 helperRotation")
                return 1
            rotations.append(tuple(helper_rotation))

        # 与参考预览相同的顺序：guide 是相对位移，outer 决定凸侧参与度。
        outer = ref.outer_participation(bind, weights, bend, 0, 0)
        guide = ref.hinge_displacement(bind, original, weights, bend, 0, 0)
        delta = ref.structural_centers(edges, original, weights, rotations,
                                       guide, outer, 16.0)

        out["cases"][key] = {
            "bend": bend,
            "helperRotation": case.get("helperRotation"),
            "guide": guide.tolist(),
            "delta": delta.tolist(),
        }
        print("bend=%5.1f  boneCount=%d  |guide|max=%8.4f  "
              "|delta|mean=%7.4f  |delta|max=%8.4f"
              % (bend, bone_count,
                 float(np.max(np.linalg.norm(guide, axis=1))),
                 float(np.mean(np.linalg.norm(delta, axis=1))),
                 float(np.max(np.linalg.norm(delta, axis=1)))))

    with open(target, "w") as handle:
        json.dump(out, handle)
    print("wrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
