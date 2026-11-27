"""按三次修正把内侧引导在中线接成一阶连续（v2：support 显式化）。

关键：hinge_section 的 support 依赖整条肢体的权重分布，
所以必须先算 support（用完整 458 点），再把它传给局部求值。

    python tests/smooth_inner_build.py 146.1265 118 120
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
RING_STEP = 2.0 * ref.HALF / (RINGS - 1)   # 2/3


def hinge_support(stations, W, radius, angle):
    """复刻 hinge_section 内部对 support 的推断。"""
    distance = np.abs(stations)
    mixed = np.count_nonzero(W > 0, axis=1) > 1
    reach = max(radius, distance[mixed].max()) if mixed.any() else radius
    rigid = distance[(~mixed) & (distance > reach)]
    support = rigid.min() if rigid.size else reach + radius
    support = min(support, 2.6 * radius) + 1.6 * radius * np.sin(angle / 2) ** 2
    return support


def section_at(st, depth, angle, weights_row, support, radius=R):
    """station 可以正负；内部把 support 当作已定常数。"""
    d = np.maximum(depth, 0.0) * np.tan(angle / 2)
    # 这里复刻 hinge_section 主体，support 直接用传入值
    return ref.hinge_section(st, np.full_like(st, np.asarray(depth).ravel()[0]
                                              if np.ndim(depth) else depth),
                             radius, weights_row, angle)


def main():
    bends = [float(x) for x in sys.argv[1:]] or [146.1265]

    d1 = json.load(open(HERE + r"\_la1.json"))
    bind = np.asarray(d1["bind"], float)
    faces = d1["faces"]
    W = np.asarray(d1["weights"], float)
    n = len(bind)
    stations = bind[:, 0]
    bind_l = np.column_stack([stations / R, bind[:, 1] / R, bind[:, 2] / R])
    edges = ref.local_edges(bind_l.tolist(), faces)

    results = {}
    for bend in bends:
        key = str(bend)
        if key not in d1["cor"]:
            key = [k for k in d1["cor"] if abs(float(k) - bend) < 1e-6][0]
        cor = np.asarray(d1["cor"][key], float)
        cor_l = np.column_stack([cor[:, 0] / R, cor[:, 1] / R, cor[:, 2] / R])
        angle = np.radians(abs(bend))

        sup = hinge_support(stations, W, R, angle)
        print("\n" + "=" * 78)
        print("bend=%.4f  support=%.6f" % (bend, sup))
        print("=" * 78)

        # 整条肢体求内外轮廓
        ext = np.full_like(stations, R)
        inner = ref.hinge_section(stations, ext, R, W, angle)
        outer = ref.hinge_section(stations, -ext, R, W, angle)

        # s=0 两侧切向：单点求值时用对应权重（取 x=0 那环）
        row0 = RINGS // 2
        wrow = W[row0 * SIDES:row0 * SIDES + 1]     # (1,2)

        def sec(st):
            # 单点 station，support 用整条的
            D = np.array([1.0])
            pts = []
            for s in st:
                # 内轮廓 depth=+R
                d = np.maximum(D, 0.0) * np.tan(angle / 2)
                # 直接调 ref.hinge_section 但把 weights 换成整条以保住 support
                # → 用整条 stations 里最接近 0 的权重行
                pts.append(ref.hinge_section(np.array([s]), D, R,
                                             np.vstack([wrow, wrow]), angle)[0])
            return np.asarray(pts)

        # 上面这样会因 weights 行数与 stations 不匹配出错，改为构造全量 st
        def sec_pt(s):
            st_all = np.concatenate([[s], stations])
            w_all = np.vstack([wrow, W])
            return ref.hinge_section(st_all, np.full_like(st_all, R), R,
                                     w_all, angle)[0]

        eps = 1e-6
        L = (sec_pt(-eps) - sec_pt(-2 * eps)) / eps
        Rr = (sec_pt(2 * eps) - sec_pt(eps)) / eps
        T = 0.5 * (L + Rr)
        dL, dR = L - T, Rr - T
        ang = math.degrees(math.acos(max(-1, min(1,
              np.dot(L, Rr) / (np.linalg.norm(L) * np.linalg.norm(Rr))))))
        print("  内侧切向 L=(%+.5f,%+.5f) R=(%+.5f,%+.5f)  夹角=%.2f°  |L-R|=%.5f"
              % (L[0], L[1], Rr[0], Rr[1], ang, np.linalg.norm(L - Rr)))

        def apply_corr(prof, h):
            out = prof.copy()
            mx = 0.0
            for i, s in enumerate(stations):
                if abs(s) >= h:
                    continue
                t = abs(s) / h
                delta = dL if s < 0 else dR
                corr = h * delta * (t * (1.0 - t) ** 2)
                mx = max(mx, math.hypot(corr[0], corr[1]))
                out[i] = out[i] + corr
            return out, mx

        for h in (RING_STEP, 2 * RING_STEP):
            _, mx = apply_corr(inner, h)
            pred = (4 * h / 27.0) * np.linalg.norm(L - Rr) / 2.0
            print("  h=%.4f  峰值位移实测=%.6f  公式=%.6f" % (h, mx, pred))

        h = 2 * RING_STEP
        inner_s, mx = apply_corr(inner, h)

        def three_pt(prof):
            a = prof[(row0 - 1) * SIDES]
            b = prof[row0 * SIDES]
            c = prof[(row0 + 1) * SIDES]
            v1, v2 = b - a, c - b
            cc = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
            return math.degrees(math.acos(max(-1, min(1, cc))))

        print("\n  3 点转角（s=-%.3f/0/+%.3f）:" % (RING_STEP, RING_STEP))
        print("    原内侧   %.2f°" % three_pt(inner))
        print("    平滑内侧 %.2f°" % three_pt(inner_s))

        def make_guide(ip, op):
            neutral = (ip + op) * 0.5
            grad = (ip - op) / (2.0 * R)
            depth = bind[:, 1]
            lat = bind[:, 2]
            tgt = np.column_stack([neutral[:, 0] + depth * grad[:, 0],
                                   neutral[:, 1] + depth * grad[:, 1],
                                   lat])
            return tgt - cor_l

        g_o = make_guide(inner, outer)
        g_s = make_guide(inner_s, outer)
        ref_g = np.asarray(ref.hinge_displacement(bind_l, cor_l, W, bend, 0.0, 0.0))
        print("\n  guide 与参考最大差: 原=%.3e" % np.abs(g_o - ref_g).max())

        q = ref.joint_quaternion(bend, 0.0, 0.0)
        rots = [(0.0, 0.0, 0.0, 1.0), q]
        orp = np.asarray(ref.outer_participation(bind_l, W, bend, 0.0, 0.0))

        d_o = np.asarray(ref.structural_centers(edges, cor_l, W, rots, g_o, orp, 16.0))
        d_s = np.asarray(ref.structural_centers(edges, cor_l, W, rots, g_s, orp, 16.0))
        print("  |delta| 原=%.4f  平滑=%.4f"
              % (np.linalg.norm(d_o, axis=1).mean(),
                 np.linalg.norm(d_s, axis=1).mean()))

        results[key] = {"deltaOrig": d_o.ravel().tolist(),
                        "deltaSmooth": d_s.ravel().tolist(),
                        "guideOrig": g_o.ravel().tolist(),
                        "guideSmooth": g_s.ravel().tolist()}

    json.dump(results, open(HERE + r"\_smooth_inner.json", "w"))
    print("\nwrote _smooth_inner.json")


if __name__ == "__main__":
    main()
