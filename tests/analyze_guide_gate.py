# -*- coding: utf-8 -*-
r"""在已导出夹具上验证「引导位移被阶跃释放」这个假设，不改 C++。

思路：算法 5 的最终结果 = CoR（用修正后的中心重算）。
修正中心 = baseCenter + delta，所以 guide 与 delta 通过求解器耦合，
不能简单地把 final - cor 当成 guide。因此这里改用一个**可判定的代理量**：

  allowed(v) = |final(v) - cor(v)|     算法5 实际落下的位移量
  want(v)    = |ocd(v) - cor(v)|       算法3 的引导位移量（同一权重、同一姿态）

如果假设成立，应当看到：在 w1 很小的一段里 allowed 远小于 want（被扣押），
随后 allowed 在很短的 station 区间内追平甚至超过 want（被释放）。

第二部分是"门控实验"：把 guide 按 gate(w1) 缩放后重新驱动算法 5，
直接看折点强度与凹陷深度怎么变。为了不改 C++，这里用
ocdCenterOffsets 能注入的唯一东西——中心偏移——做不了这件事，
所以改用**合成**：在 cor 与 final 之间按 gate 重新插值，
得到"如果引导位移按 gate 释放"会是什么形状。这是形状预估，不是真实求解，
因此只用于判断方向，不作为最终结论。
"""
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
sys.path.insert(0, os.path.join(ROOT, "scripts"))
sys.path.insert(0, HERE)


def dist(a, b):
    return math.sqrt(sum((a[i] - b[i]) ** 2 for i in range(3)))


def smoothstep(edge0, edge1, x):
    if edge1 <= edge0:
        return 0.0 if x < edge1 else 1.0
    t = (x - edge0) / (edge1 - edge0)
    t = min(1.0, max(0.0, t))
    return t * t * (3.0 - 2.0 * t)


def load(source):
    payload = json.load(open(source))
    bind = payload["mesh"]["bindPoints"]
    axis = payload["hinge"]["axis"]
    origin = payload["hinge"]["origin"]
    sparse = payload["weightsSparse"]
    weight1 = [float(row.get("1", 0.0)) for row in sparse]
    stations = [sum((p[i] - origin[i]) * axis[i] for i in range(3)) for p in bind]
    return payload, bind, axis, origin, weight1, stations


def ring_table(bind, stations, weight1, cor, fin, width):
    buckets = {}
    for vertex, station in enumerate(stations):
        buckets.setdefault(int(round(station / width)), []).append(vertex)
    rows = []
    for key in sorted(buckets):
        members = buckets[key]
        allowed = [dist(fin[v], cor[v]) for v in members]
        radial = []
        for vertex in members:
            rb = math.hypot(bind[vertex][1], bind[vertex][2])
            rf = math.hypot(fin[vertex][1], fin[vertex][2])
            radial.append(rf - rb)
        rows.append({
            "station": sum(stations[v] for v in members) / len(members),
            "w1": sum(weight1[v] for v in members) / len(members),
            "w1max": max(weight1[v] for v in members),
            "allowed": sum(allowed) / len(allowed),
            "radial": sum(radial) / len(radial),
            "count": len(members),
        })
    return rows


def fold_score(rows, low, high):
    best, where = 0.0, 0.0
    for a, b in zip(rows, rows[1:]):
        if not (low <= b["station"] <= high):
            continue
        span = b["station"] - a["station"]
        if span <= 1e-9:
            continue
        slope = (b["allowed"] - a["allowed"]) / span
        if slope > best:
            best, where = slope, b["station"]
    return best, where


def deepest(rows):
    row = min(rows, key=lambda r: r["radial"])
    return row["station"], row["radial"]


def main():
    source = sys.argv[1] if len(sys.argv) > 1 else r"D:\tmp\ocd_case3.json"
    payload, bind, axis, origin, weight1, stations = load(source)
    cor = payload["points"]["cor"]
    ocd = payload["points"]["ocd"]
    fin = payload["points"]["final"]

    uniq = sorted({round(s, 5) for s in stations})
    gaps = sorted(b - a for a, b in zip(uniq, uniq[1:]) if b - a > 1e-6)
    width = gaps[len(gaps) // 2]

    # ---------- 1. 扣押 vs 释放 ----------
    print("== 1. 引导位移的「扣押 / 释放」（每顶点比对）==")
    print("%9s %7s %7s %10s %10s %9s"
          % ("station", "w1avg", "w1max", "want|ocd-cor|", "allowed|fin-cor|",
             "allowed/want"))
    rows = ring_table(bind, stations, weight1, cor, fin, width)
    want_ring = {}
    for vertex, station in enumerate(stations):
        want_ring.setdefault(int(round(station / width)), []).append(vertex)
    for row in rows:
        key = int(round(row["station"] / width))
        members = want_ring[key]
        want = sum(dist(ocd[v], cor[v]) for v in members) / len(members)
        row["want"] = want
        row["ratio"] = row["allowed"] / want if want > 1e-9 else 0.0
        if 6.5 <= row["station"] <= 11.5:
            print("%9.3f %7.4f %7.4f %10.4f %10.4f %9.3f"
                  % (row["station"], row["w1"], row["w1max"], want,
                     row["allowed"], row["ratio"]))

    print()
    print("  纯单骨区(w1max=0) allowed 最大值: %.6f"
          % max([r["allowed"] for r in rows if r["w1max"] == 0.0] or [0.0]))
    first = next((r for r in rows if r["w1max"] > 0.0), None)
    if first:
        print("  首个混权环: station=%.3f  w1max=%.4f  allowed=%.4f  want=%.4f"
              % (first["station"], first["w1max"], first["allowed"], first["want"]))

    score, where = fold_score(rows, 8.0, 11.0)
    print("  折点强度 = %.4f @ station %.3f" % (score, where))
    ds, dr = deepest(rows)
    print("  最凹 = station %.3f  径向 %.4f" % (ds, dr))

    # ---------- 2. 门控形状预估 ----------
    print("\n== 2. 门控形状预估（在 cor 与 final 之间按 gate 重插值）==")
    print("   注意：这是形状预估，不是真实求解，仅用于判断方向。")
    print("%10s %10s %10s %10s %10s"
          % ("gate上限", "折点强度", "折点station", "最凹station", "最凹深度"))

    results = []
    for gate_hi in (0.0, 0.002, 0.01, 0.05, 0.15, 0.35, 1.0):
        blended = []
        for vertex, point in enumerate(cor):
            g = smoothstep(0.0, gate_hi, weight1[vertex]) if gate_hi > 0 else 0.0
            blended.append(tuple(cor[vertex][i] + g * (fin[vertex][i]
                                                       - cor[vertex][i])
                                 for i in range(3)))
        sub = ring_table(bind, stations, weight1, cor, blended, width)
        s, w = fold_score(sub, 8.0, 11.0)
        ds2, dr2 = deepest(sub)
        results.append((gate_hi, s, w, ds2, dr2))
        print("%10.4f %10.4f %10.3f %10.3f %10.4f" % (gate_hi, s, w, ds2, dr2))

    print("\n== 3. 结论 ==")
    baseline = next(r for r in results if abs(r[0] - 0.0) < 1e-12)
    print("  gate=0 相当于完全丢掉引导位移，折点强度 %.4f" % baseline[1])
    print("  gate=1 相当于当前行为，折点强度 %.4f"
          % next(r for r in results if abs(r[0] - 1.0) < 1e-12)[1])
    best = min(results, key=lambda r: r[1])
    print("  折点最弱: gate上限=%.4f 强度 %.4f @ station %.3f"
          % (best[0], best[1], best[2]))
    return 0


if __name__ == "__main__":
    sys.exit(main())
