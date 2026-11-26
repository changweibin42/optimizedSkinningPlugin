# -*- coding: utf-8 -*-
"""逐级对拍的另一端：用 1285f9a 参考实现算出 guide 与 centerOffset。

Maya 自带的 Python 没有 numpy，所以这一步必须在外部 numpy 环境里跑。
输入是 `tests/maya_ocd_alg5_dump.py --dump` 导出的 JSON，输出供
`--inject` 使用。

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
    edges = ref.local_edges(bind, faces)
    weights = np.array([[1.0 - ref.skin_weight(p[0]), ref.skin_weight(p[0])]
                        for p in bind])

    out = {"cases": {}}
    for key, case in payload["cases"].items():
        bend = case["bend"]
        original = np.asarray(case["original"])
        rotations = [ref.joint_quaternion(0, 0, 0), ref.joint_quaternion(bend, 0, 0)]

        # 与参考预览完全相同的顺序：guide 是相对位移，outer 决定凸侧参与度。
        outer = ref.outer_participation(bind, weights, bend, 0, 0)
        guide = ref.hinge_displacement(bind, original, weights, bend, 0, 0)
        delta = ref.structural_centers(edges, original, weights, rotations,
                                       guide, outer, 16.0)

        out["cases"][key] = {
            "bend": bend,
            "guide": guide.tolist(),
            "delta": delta.tolist(),
        }
        print("bend=%5.1f  |guide|max=%8.4f  |delta|mean=%7.4f  |delta|max=%8.4f"
              % (bend,
                 float(np.max(np.linalg.norm(guide, axis=1))),
                 float(np.mean(np.linalg.norm(delta, axis=1))),
                 float(np.max(np.linalg.norm(delta, axis=1)))))

    with open(target, "w") as handle:
        json.dump(out, handle)
    print("wrote", target)
    return 0


if __name__ == "__main__":
    sys.exit(main())
