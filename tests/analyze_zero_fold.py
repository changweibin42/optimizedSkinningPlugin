"""解析：若 hinge_section 在 s=0 附近改用单一光滑函数，
三点转角是多少，s=0 的点需要移多少。

核心问题定位：
  station<=0 分支返回 (-outward, height)
  station>0  分支返回 (cos a*outward - sin a*height, sin a*outward + cos a*height)
  outward/height 是 |station| 的偶函数 → 位置连续，但映射不同。

实测内侧三点（146.13°, support=4.064）：
  s=-0.667  (-2.18960, 0.69225)
  s= 0      (-1.97394, 0.60111)
  s=+0.667  (-2.20379, 0.64564)
x 先增后减 → 折返。

    python tests/analyze_zero_fold.py
"""
import math

import numpy as np

R = 1.0
SUPPORT = 4.064216
ANG = math.radians(146.1265)


def section_raw(station, depth, support=SUPPORT, angle=ANG, radius=R):
    """复刻 hinge_section，返回 (x, h)。"""
    distance = abs(station)
    u = min(distance / support, 1.0)
    raw = max(depth, 0.0) * math.tan(angle / 2)
    limit = 0.95 * min(support, 2.6 * radius)
    shift = raw / math.sqrt(1 + (raw / limit) ** 2) if limit > 1e-12 else 0.0
    inner_travel = shift * (1.0 - u)
    release = 0.28 * math.sin(angle / 2) ** 2 * (4.0 * u * (1.0 - u)) ** 2
    outward = distance + inner_travel * (1.0 - release)
    height = depth
    if angle > 1e-8:
        height += (shift / math.tan(angle / 2) - max(depth, 0.0)) * (1.0 - u) * (1.0 - release)
    outer_radius = max(-depth, 0.0)
    half = angle / 2
    fillet = outer_radius * (1.0 + 0.45 * math.cos(half))
    tangent = 0.45 * outer_radius * math.sin(half)
    arc = fillet * half
    extra = arc - tangent
    side = 1.0 if depth < 0 else 0.0
    gather = 0.85 * math.sin(half) * side ** 2
    opening = 0.32 * distance * math.sin(half) * side ** 2 * (1.0 - u) ** 5
    material = distance * (1.0 - gather * (1.0 - u) ** 2) + opening
    join = u ** 3 * (4.0 - 3.0 * u)
    progress = material + extra * join
    remaining = max(arc - progress, 0.0)
    phi = remaining / fillet if fillet > 1e-12 else 0.0
    arc_station = tangent - fillet * math.sin(phi) + max(progress - arc, 0.0)
    arc_height = depth + 2 * fillet * math.sin(phi / 2) ** 2
    if depth < 0:
        outward, height = arc_station, arc_height
    ca, sa = math.cos(angle), math.sin(angle)
    if station <= 0:
        return -outward, height
    return ca * outward - sa * height, sa * outward + ca * height


def angle3(p1, p2, p3):
    v1 = np.array(p2) - np.array(p1)
    v2 = np.array(p3) - np.array(p2)
    c = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))
    return math.degrees(math.acos(max(-1, min(1, c))))


def main():
    print("=" * 78)
    print("146.1265°, support=%.6f, depth=+1（内侧）" % SUPPORT)
    print("=" * 78)

    st = [-0.6667, 0.0, 0.6667]
    pts = [section_raw(s, 1.0) for s in st]
    print("\n内侧三点（原版）:")
    for s, p in zip(st, pts):
        print("  s=%+7.4f  (%+9.5f, %+9.5f)" % (s, p[0], p[1]))
    print("  转角 = %.4f°" % angle3(*pts))

    xs = [p[0] for p in pts]
    print("\n  x 序列: %s" % ["%+.5f" % v for v in xs])
    print("  x 先增后减? %s" % ("是" if xs[1] > xs[0] and xs[1] > xs[2] else "否"))

    # 中点移到连接两端的线段上，需要移动多少
    a, c = np.array(pts[0]), np.array(pts[2])
    b = np.array(pts[1])
    d = c - a
    t = np.dot(b - a, d) / np.dot(d, d)
    proj = a + t * d
    dist = np.linalg.norm(b - proj)
    print("\n  中点 B 到线段 AC 的距离 = %.5f R" % dist)
    print("  （若把 B 移到 AC 上，三点共线，转角=0）")
    print("  需要把 B 从 (%+.5f,%+.5f) 移到 (%+.5f,%+.5f)"
          % (b[0], b[1], proj[0], proj[1]))
    print("  移动量 = (%.5f, %.5f)  模 = %.5f"
          % (proj[0] - b[0], proj[1] - b[1], np.linalg.norm(proj - b)))

    # ---- 若两分支的导数在 0 处接顺，位置会怎样 ----
    print("\n" + "=" * 78)
    print("若 s=0 附近改用单一光滑函数（不求导连续，而求位置单调）")
    print("=" * 78)
    eps = 1e-6
    lo = section_raw(-eps, 1.0)
    hi = section_raw(eps, 1.0)
    print("  s=-eps: (%+9.6f, %+9.6f)" % lo)
    print("  s=+eps: (%+9.6f, %+9.6f)" % hi)

    # 左支在 s=0 的 x = -outward(0)，右支 x = ca*outward(0) - sa*height(0)
    # 令 outward(0)=O, height(0)=H
    # 左 x=-O ; 右 x=ca*O - sa*H
    O = section_raw(0, 1.0)
    print("  s=0 exact: (%+9.6f, %+9.6f)" % O)
    outward0 = -O[0]
    height0 = O[1]
    ca, sa = math.cos(ANG), math.sin(ANG)
    x_left = -outward0
    x_right = ca * outward0 - sa * height0
    print("\n  outward0=%.6f  height0=%.6f" % (outward0, height0))
    print("  左支 x(0) = -outward0        = %+.6f" % x_left)
    print("  右支 x(0) = ca*O - sa*H      = %+.6f" % x_right)
    print("  两者相等？差 = %.3e" % (x_right - x_left))

    # 若把右支改成和左支同样的映射（即不分两支，都用 -outward, height）
    print("\n  若两分支都用 (-outward, height)：")
    pts_mono = []
    for s in st:
        d = abs(s)
        u = min(d / SUPPORT, 1.0)
        raw = 1.0 * math.tan(ANG / 2)
        limit = 0.95 * min(SUPPORT, 2.6 * R)
        shift = raw / math.sqrt(1 + (raw / limit) ** 2)
        it = shift * (1 - u)
        rel = 0.28 * math.sin(ANG / 2) ** 2 * (4 * u * (1 - u)) ** 2
        ow = d + it * (1 - rel)
        hh = 1.0 + (shift / math.tan(ANG / 2) - 1.0) * (1 - u) * (1 - rel)
        pts_mono.append((-ow, hh))
    for s, p in zip(st, pts_mono):
        print("    s=%+7.4f  (%+9.5f, %+9.5f)" % (s, p[0], p[1]))
    print("    转角 = %.4f°" % angle3(*pts_mono))

    # 右支旋转是刚体，转角不变？
    print("\n  注：右支是刚体旋转（绕原点转 angle），")
    print("      只改变 s>0 那一段的整体位置，不改变段内转角。")
    print("      所以 s>0 段内（0 到 +0.667）原本就是 0°（直线）。")

    # 逐点看 s>0 段内转角
    print("\n  s>0 段内三点（0, 0.333, 0.667）转角：")
    p = [section_raw(s, 1.0) for s in (0.0001, 0.3334, 0.6667)]
    print("    %.4f°" % angle3(*p))
    print("  s<0 段内三点（-0.667, -0.333, 0）转角：")
    p = [section_raw(s, 1.0) for s in (-0.6667, -0.3334, -0.0001)]
    print("    %.4f°" % angle3(*p))

    print("\n" + "=" * 78)
    print("结论")
    print("=" * 78)
    print("  折返完全来自两个分支在 s=0 的映射不同：")
    print("    左支把 outward 直接当 -x（不旋转）")
    print("    右支把 outward/height 旋转 angle")
    print("  两段各自内部都是直的（0°），只在 s=0 这一个点形成 168° 折角。")
    print("  因此这不是'导数不连续'的渐变问题，而是两个刚体段之间的夹角。")


if __name__ == "__main__":
    main()
