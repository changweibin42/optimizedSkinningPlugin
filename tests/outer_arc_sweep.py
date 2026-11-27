"""外轮廓参数扫描：让 guide 的外极点尽量贴合 CoR 外极点（=红线那条饱满轮廓）。

外侧（depth<0）走 hinge_section 的 arc 分支：
    fillet = outer_radius*(1 + F*cos(half))
    tangent_station = T*outer_radius*sin(half)
    arc_length = fillet*half
    extra = arc_length - tangent_station
    side = 1
    gather = G*sin(half)*side^2
    opening = O*distance*sin(half)*side^2*(1-u)^5
    material = distance*(1-gather*(1-u)^2) + opening
    join = u^3*(4-3u)
    progress = material + extra*join
    phi = max(arc_length-progress,0)/fillet
    arc_height = depth + 2*fillet*sin(phi/2)^2

外极点的 h 恰好等于 arc_height，所以这条曲线完全由 (F,T,G,O) 决定。

目标：min RMS( guide_outer_h  -  cor_outer_h )，在 x∈[-3,0]。

    python tests/outer_arc_sweep.py
"""
import json
import math
import sys

import numpy as np

sys.path.insert(0, r"C:\Users\changweibin\Documents\deepseek-harness\default-workspace\optimizedSkinningPlugin\tests")
import ocd_elbow_reference_latest as ref

D = r"D:\tmp\ocd_case3.json"
SIDES = 80
STEP = 0.12
R = 1.0


def arc_height(distance, u, angle, depth, support,
               F=0.45, T=0.45, G=0.85, O=0.32):
    outer_radius = max(-depth, 0.0)
    half = angle / 2.0
    fillet = outer_radius * (1.0 + F * math.cos(half))
    tangent = T * outer_radius * math.sin(half)
    arc_len = fillet * half
    extra = arc_len - tangent
    side = 1.0
    gather = G * math.sin(half) * side ** 2
    opening = O * distance * math.sin(half) * side ** 2 * (1.0 - u) ** 5
    material = distance * (1.0 - gather * (1.0 - u) ** 2) + opening
    join = u ** 3 * (4.0 - 3.0 * u)
    progress = material + extra * join
    remaining = max(arc_len - progress, 0.0)
    phi = remaining / fillet if fillet > 1e-12 else 0.0
    return depth + 2.0 * fillet * math.sin(phi / 2.0) ** 2


def neutral_h(distance, u, angle, depth, support, radius):
    """d>0 分支的 height，用于求 neutral。"""
    raw = max(depth, 0.0) * math.tan(angle / 2)
    limit = 0.95 * min(support, 2.6 * radius)
    shift = raw / math.sqrt(1 + (raw / limit) ** 2) if limit > 1e-12 else 0.0
    release = 0.28 * math.sin(angle / 2) ** 2 * (4 * u * (1 - u)) ** 2
    h = depth
    if angle > 1e-8:
        h += (shift / math.tan(angle / 2) - max(depth, 0.0)) * (1 - u) * (1 - release)
    return h


def main():
    d = json.load(open(D))
    bind = np.asarray(d["mesh"]["bindPoints"], float)
    cor = np.asarray(d["points"]["cor"], float)
    n = len(bind)

    child = [b for b in d["bones"] if b["short"] == "L_Ankle_jnt"][0]
    pivot = np.asarray(child["bindPosition"], float)
    axis = np.asarray(d["hinge"]["axis"], float)
    axis = axis / np.linalg.norm(axis)
    bY = np.array([0.0, 1.0, 0.0])
    bZ = np.cross(axis, bY)
    bZ /= np.linalg.norm(bZ)
    rel = bind - pivot
    st = rel @ axis
    rad = rel - np.outer(st, axis)
    RR = float(np.linalg.norm(rad, axis=1).max())
    bind_l = np.column_stack([st / RR, (rad / RR) @ bY, (rad / RR) @ bZ])

    def to_local(P):
        o = P - pivot
        return np.column_stack([(o @ axis) / RR, (o @ bY) / RR, (o @ bZ) / RR])

    cor_l = to_local(cor)
    W = np.zeros((n, len(d["bones"])))
    for i, row in enumerate(d["weightsSparse"]):
        for k, v in row.items():
            W[i, int(k)] = float(v)
    used = [j for j in range(len(d["bones"])) if W[:, j].max() > 0.0]
    W = W[:, used]

    m = np.asarray(child["worldMatrix"], float).reshape(4, 4)[:3, :3]
    bend = math.degrees(math.atan2(m[1, 0], m[0, 0]))
    angle = math.radians(abs(bend))

    # support（原版）
    distance_all = np.abs(bind_l[:, 0])
    mixed = np.count_nonzero(W > 0, axis=1) > 1
    reach = max(R, distance_all[mixed].max()) if mixed.any() else R
    rigid = distance_all[(~mixed) & (distance_all > reach)]
    support = (rigid.min() if rigid.size else reach + R)
    support = min(support, 2.6 * R) + 1.6 * R * math.sin(angle / 2) ** 2
    print("bend=%.4f  support=%.5f  R=%.6f" % (bend, support, RR))

    rings = np.round(bind[:, 0] / STEP).astype(int)
    # 目标：CoR 外极点（局部空间 y 最小）
    targets = []
    for rid in sorted(set(rings.tolist())):
        x0 = rid * STEP
        if not (-3.0 <= x0 <= 0.0):
            continue
        m2 = np.where(rings == rid)[0]
        # 外极点 = 该环 depth(=y) 最小
        i = m2[np.argmin(bind_l[m2, 1])]
        targets.append((x0, bind_l[i, 0], cor_l[i, 1]))
    print("目标环数 %d，x 从 %.2f 到 %.2f" % (len(targets), targets[0][0], targets[-1][0]))

    def guide_outer(depth_sign=-1.0, F=0.45, T=0.45, G=0.85, O=0.32):
        """返回每个目标环的 guide 外极点 h（局部空间）。"""
        out = []
        for x0, station, cor_y in targets:
            dist = abs(station)
            u = min(dist / support, 1.0)
            ah = arc_height(dist, u, angle, depth_sign, support, F, T, G, O)
            nh = neutral_h(dist, u, angle, depth_sign, support, R)
            # neutral 用 inner(depth=+1) 与 outer(depth=-1) 的平均
            ih = neutral_h(dist, u, angle, 1.0, support, R)
            neutral = (ih + ah) * 0.5
            grad = (ih - ah) / (2.0 * R)
            # depth = -1 的外极点
            h = neutral + depth_sign * grad
            out.append(h)
        return np.asarray(out)

    cor_y = np.asarray([t[2] for t in targets])

    print("\n基准（原版参数）:")
    base = guide_outer()
    print("  guide 外极点 vs CoR: RMS=%.5f  max=%.5f"
          % (np.sqrt(np.mean((base - cor_y) ** 2)),
             np.max(np.abs(base - cor_y))))

    print("\n扫描 G（gather 系数）:")
    print("  %8s %10s %10s %10s" % ("G", "RMS", "max|Δ|", "端点Δ"))
    best = None
    for G in (0.0, 0.1, 0.2, 0.3, 0.394, 0.5, 0.6, 0.7, 0.85, 1.0):
        g = guide_outer(G=G)
        rms = np.sqrt(np.mean((g - cor_y) ** 2))
        print("  %8.3f %10.5f %10.5f %+10.5f"
              % (G, rms, np.max(np.abs(g - cor_y)), g[-1] - cor_y[-1]))
        if best is None or rms < best[1]:
            best = (G, rms)

    print("\n扫描 F（fillet 系数）:")
    print("  %8s %10s %10s" % ("F", "RMS", "max|Δ|"))
    for F in (0.0, 0.2, 0.45, 0.7, 1.0, 1.3):
        g = guide_outer(F=F)
        rms = np.sqrt(np.mean((g - cor_y) ** 2))
        print("  %8.3f %10.5f %10.5f" % (F, rms, np.max(np.abs(g - cor_y))))

    print("\n扫描 T（tangent 系数）:")
    print("  %8s %10s %10s" % ("T", "RMS", "max|Δ|"))
    for T in (0.0, 0.2, 0.45, 0.7, 1.0):
        g = guide_outer(T=T)
        rms = np.sqrt(np.mean((g - cor_y) ** 2))
        print("  %8.3f %10.5f %10.5f" % (T, rms, np.max(np.abs(g - cor_y))))

    print("\n二维细扫 (G, F)：")
    print("  %8s %8s %10s" % ("G", "F", "RMS"))
    best2 = None
    for G in np.arange(0.0, 0.61, 0.05):
        for F in np.arange(0.1, 1.01, 0.1):
            g = guide_outer(G=float(G), F=float(F))
            rms = np.sqrt(np.mean((g - cor_y) ** 2))
            if best2 is None or rms < best2[2]:
                best2 = (float(G), float(F), rms)
    print("  最优: G=%.2f F=%.2f RMS=%.5f" % best2)

    # 最优参数下的逐环对比
    G0, F0, _ = best2
    g = guide_outer(G=G0, F=F0)
    print("\n最优 (G=%.2f, F=%.2f) 逐环对比：" % (G0, F0))
    print("  %8s %12s %12s %12s %12s"
          % ("x", "CoR外极点", "原guide", "新guide", "新-原"))
    orig = guide_outer()
    for (x0, _, cy), go, gn2 in zip(targets, orig, g):
        print("  %8.3f %12.5f %12.5f %12.5f %+12.5f"
              % (x0, cy, go, gn2, gn2 - go))


if __name__ == "__main__":
    main()
