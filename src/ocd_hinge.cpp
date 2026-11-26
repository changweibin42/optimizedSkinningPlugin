#include "ocd_hinge.h"

#include <algorithm>
#include <cmath>
#include <limits>

namespace ocd_hinge {

using namespace mathcore;

namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double degToRad = pi / 180.0;
constexpr double radToDeg = 180.0 / pi;

inline double smoothStep(double t) {
    t = std::clamp(t, 0.0, 1.0);
    return t * t * (3.0 - 2.0 * t);
}

inline double smoothStepRange(double edge0, double edge1, double value) {
    if (edge1 <= edge0) return value >= edge1 ? 1.0 : 0.0;
    return smoothStep((value - edge0) / (edge1 - edge0));
}

// 参考实现的 fade(distance, reach) = 1 - smooth(distance/reach)
inline double fade(double distance, double reach) {
    return 1.0 - smoothStep(distance / reach);
}

inline double mix(double a, double b, double t) {
    return a + (b - a) * t;
}

// 参考实现的 skin_weight：smooth(0.5 + station/5.2)
inline double skinWeight(double station) {
    return smoothStep(0.5 + station / 5.2);
}

} // namespace

const double outerOffsets[halfRings + 1] = {
    0.3573783304, 0.5395253322, 0.4023315278, 0.2424765910, 0.1149518661,
    0.0305340904, 0.0257631388, 0.0152670452, 0.0047709516, 0.0
};

const double innerOffsets[halfRings + 1] = {
    -2.4071727709, -2.0916385862, -1.7084857196, -1.2864088566, -0.8637314900,
    -0.4817217350, -0.1748798316, -0.1295406160, -0.0453392156, -0.0
};

const double innerClearOffsets[halfRings + 1] = {
    -2.4071727709, -2.1190059439, -1.7430531180, -1.3217063759, -0.8973578005,
    -0.5123994744, -0.2092234804, -0.1549803559, -0.0542431246, -0.0
};

// ================================================================
// Elbow：OCD 主算法
// ================================================================
Elbow::Elbow(const Pose& pose) {
    const double bend = pose.bend;
    double direction = pose.direction;
    avoidOverlap_ = pose.avoidOverlap;

    // 与参考实现一致：负弯曲等价于把弯曲平面翻转 180 度。
    const double plane = direction * degToRad + (bend < 0.0 ? pi : 0.0);
    const double beta = std::abs(bend) * degToRad;
    cp_ = std::cos(plane);
    sp_ = std::sin(plane);
    cb_ = std::cos(beta);
    sb_ = std::sin(beta);
    h_ = beta * 0.5;
    ch_ = std::cos(h_);
    sh_ = std::sin(h_);
    bend_ = beta;

    fraction_ = std::abs(bend) / 150.0;
    const double compression = (1.0 - ch_) / (1.0 - std::cos(75.0 * degToRad));
    const double c3 = compression * compression * compression;
    tipY_ = 1.0 - 0.355 * c3;
    tipX_ = -tipY_ * std::tan(h_);
    innerReach_ = 3.15 + 1.90 * compression;
    tangentX_ = -0.22 * std::tan(h_);

    const double raw = 1.12 / ch_ - 0.12;
    const double excess = std::max(0.0, raw - 1.95);
    innerClearance_ = 1.0 / (1.0 + 3.0 * (excess + 0.60 * std::exp(-excess / 0.60) - 0.60));
    if (!avoidOverlap_) innerClearance_ = 1.0;
    twistRadians_ = pose.twist * degToRad;

    // 11 个补偿通道：按参考实现的 VIDEO 表归一化，全零则不做次级编辑。
    static const double video[11] = {0.33, 0.29, 0.19, 0.19, 0.21, 0.21,
        0.10, 0.10, 0.10, 0.10, 0.10};
    edited_ = false;
    for (int k = 0; k < 11; ++k) {
        values_[k] = pose.values[k] / video[k];
        if (std::abs(pose.values[k]) > 0.0) edited_ = true;
    }
    response_ = std::min(1.15, sh_ * sh_ / 0.75);
}

double Elbow::twistSkin(double station) const {
    return twistRadians_ * skinWeight(station);
}

Rail Elbow::rail(double station, const double offsets[halfRings + 1], int pole) const {
    const double u = std::abs(station);
    const int row = std::min(static_cast<int>(u / ringStep), halfRings - 1);
    const double t = std::min(1.0, u / ringStep - row);
    const double dx = mix(offsets[row], offsets[row + 1], t);
    double x = -u + fraction_ * dx;
    const double tip = pole > 0 ? tipX_ : tangentX_ + 1.22 * sh_;
    if (pole > 0) {
        // 以 ring zero 为锚点的凸空间衰减，两侧肢体的间隙在末端归零。
        const double fade = std::max(0.0, 1.0 - u / halfSpan);
        x = -u + tip * fade * fade;
    }
    else if (u < ringStep) {
        x += (1.0 - u / ringStep) * (tip - fraction_ * offsets[0]);
    }

    double y = 0.0;
    if (pole > 0) {
        const double a = std::clamp((tip - x) / (innerReach_ + tip), 0.0, 1.0);
        const double oneMinus = 1.0 - a;
        const double remainder = avoidOverlap_
            ? oneMinus * oneMinus * (1.0 + 0.87 * a)
            : oneMinus * oneMinus * oneMinus * oneMinus;
        y = 1.0 - (1.0 - tipY_) * remainder;
    }
    else {
        const double run = std::max(0.0, x - tangentX_);
        const double sq = std::max(0.0, 1.22 * 1.22 - run * run);
        y = -1.0 + 1.22 - std::sqrt(sq);
    }

    if (station > 0.0) return {-cb_ * x - sb_ * y, -sb_ * x + cb_ * y};
    return {x, y};
}

Section Elbow::section(double station) const {
    const Rail outer = rail(station, outerOffsets, -1);
    const Rail inner = rail(station, avoidOverlap_ ? innerClearOffsets : innerOffsets, 1);
    return {outer.x, outer.y, 0.5 * (inner.x - outer.x), 0.5 * (inner.y - outer.y)};
}

Vec3 Elbow::point(double station, double y, double z) const {
    const double tw = twistSkin(station);
    const double ct = std::cos(tw), st = std::sin(tw);
    const double cy = y * cp_ + z * sp_;
    const double cz = -y * sp_ + z * cp_;
    const double ry = cy * ct - cz * st;
    const double rz = cy * st + cz * ct;

    const Section sec = section(station);
    const double x0 = sec.ox + (1.0 + ry) * sec.rx;
    const double y0 = sec.oy + (1.0 + ry) * sec.ry;

    double px = x0;
    double py = y0;
    double pz = rz;

    // 补偿通道全部为零时参考实现直接返回基础截面（self.edited == False）。
    if (!edited_) {
        return {px, cp_ * py - sp_ * pz, sp_ * py + cp_ * pz};
    }

    const double u = std::abs(station);
    if (u >= reach || response_ <= 0.0) {
        return {px, cp_ * py - sp_ * pz, sp_ * py + cp_ * pz};
    }

    const double ci = values_[0], co = values_[1];
    const double ip = values_[2], inn = values_[3], op = values_[4], on = values_[5];
    const double nip = values_[6], nin = values_[7], nop = values_[8], non = values_[9];
    const double bulge = values_[10];

    const double wi = response_ * std::pow(std::max(ry, 0.0), 1.7);
    const double wo = response_ * std::pow(std::max(-ry, 0.0), 1.7);
    const double branch = station < 0.0 ? -1.0 : 1.0;
    const bool negative = station < 0.0;
    const double inner = negative ? ip : inn;
    const double outer = negative ? op : on;
    const double liftInner = negative ? nip : nin;
    const double liftOuter = negative ? nop : non;

    const double travel = branch * (inTravelGain_ * wi * inner * shoulder(u, inTravelPeak_, inTravelEnd_) -
        0.48 * wo * outer * shoulder(u, 2.05, 4.8));
    if (std::abs(travel) > 0.0) {
        const Section shifted = section(station + travel);
        px = x0 + shifted.ox + (1.0 + ry) * shifted.rx - (sec.ox + (1.0 + ry) * sec.rx);
        py = y0 + shifted.oy + (1.0 + ry) * shifted.ry - (sec.oy + (1.0 + ry) * sec.ry);
    }

    // 单侧 Slide Out 在顶点处贡献成对行程的一半。
    const double outerAtCrease = mix(op, on, smoothStep(0.5 + station / 1.6));
    const double shift = 0.62 * innerClearance_ * wi * ci * fade(u, 3.6) -
        0.392 * wo * co * fade(u, 3.55) -
        0.251 * wo * outerAtCrease * fade(u, 2.75);
    px -= sh_ * shift;
    py += ch_ * shift;

    const double lift = 0.186 * innerClearance_ * wi * liftInner * shoulder(u, 1.45, 4.8) +
        0.120 * wo * liftOuter * shoulder(u, 1.35, 4.8);
    if (std::abs(lift) > 0.0) {
        // 等环向坐标处材料子午线的切向。
        const Section left = section(station - 0.002);
        const Section right = section(station + 0.002);
        const double tx = right.ox - left.ox + (1.0 + ry) * (right.rx - left.rx);
        const double ty = right.oy - left.oy + (1.0 + ry) * (right.ry - left.ry);
        const double nx = -ry * ty;
        const double ny = ry * tx;
        const double nz = rz * (sec.ry * tx - sec.rx * ty);
        const double norm = std::sqrt(nx * nx + ny * ny + nz * nz);
        if (norm > 1e-12) {
            px += lift * nx / norm;
            py += lift * ny / norm;
            pz += lift * nz / norm;
        }
    }
    pz += 0.42 * bulge * response_ * fade(u, 3.65) * pz * pz * pz;

    // 防止组合后的肩部编辑越过折痕角平分线。
    if (avoidOverlap_ && u > 1e-8) {
        const double room = branch * (ch_ * x0 + sh_ * y0);
        const double gap = branch * (ch_ * px + sh_ * py);
        const double minimum = std::max(0.0, 0.12 * room);
        if (gap < minimum) {
            const double delta = branch * (minimum - gap);
            px += ch_ * delta;
            py += sh_ * delta;
        }
    }
    return {px, cp_ * py - sp_ * pz, sp_ * py + cp_ * pz};
}

double Elbow::shoulder(double distance, double peak, double end) const {
    if (distance < peak) return smoothStep(distance / peak);
    return 1.0 - smoothStep((distance - peak) / (end - peak));
}

// ================================================================
// hinge_section / hinge_displacement
// ================================================================
HingePoint hingeSection(double station, double depth, double radius,
    double weightParent, double weightChild, double weightHelper,
    double angle, double support) {
    (void)weightHelper;
    const double distance = std::abs(station);
    if (radius <= 1e-12) return {station, depth};

    const double half = angle * 0.5;
    const double sineHalf = std::sin(half);
    const double tangentHalf = std::tan(half);
    const double u = std::min(distance / support, 1.0);

    // 内侧偏移线在角平分线处相交，轴向行程由 support 限制。
    const double raw = std::max(depth, 0.0) * tangentHalf;
    const double limit = 0.95 * std::min(support, 2.6 * radius);
    const double shift = limit > 1e-12
        ? raw / std::sqrt(1.0 + (raw / limit) * (raw / limit))
        : 0.0;

    // 二次松弛让内侧纵向间隙在折痕处最小，同时站点映射的导数保持为正。
    const double innerTravel = shift * (1.0 - u);
    const double release = 0.28 * sineHalf * sineHalf *
        std::pow(4.0 * u * (1.0 - u), 2.0);
    double outward = distance + innerTravel * (1.0 - release);
    double height = depth;
    if (angle > 1e-8) {
        height += (shift / tangentHalf - std::max(depth, 0.0)) * (1.0 - u) * (1.0 - release);
    }

    // 与两条刚性偏移线相切的圆弧倒角，按弧长推进，额外长度在 support 处归零。
    const double outerRadius = std::max(-depth, 0.0);
    const double filletRadius = outerRadius * (1.0 + 0.45 * std::cos(half));
    const double tangentStation = 0.45 * outerRadius * sineHalf;
    const double arcLength = filletRadius * half;
    const double extra = arcLength - tangentStation;

    const double side = depth < 0.0 ? 1.0 : 0.0;
    const double gather = 0.85 * sineHalf * side * side;
    const double opening = 0.32 * distance * sineHalf * side * side *
        std::pow(1.0 - u, 5.0);
    const double material = distance * (1.0 - gather * (1.0 - u) * (1.0 - u)) + opening;

    const double join = u * u * u * (4.0 - 3.0 * u);
    const double progress = material + extra * join;
    const double remaining = std::max(arcLength - progress, 0.0);
    const double phi = filletRadius > 1e-12 ? remaining / filletRadius : 0.0;
    const double arcStation = tangentStation - filletRadius * std::sin(phi) +
        std::max(progress - arcLength, 0.0);
    const double arcHeight = depth + 2.0 * filletRadius * std::pow(std::sin(phi * 0.5), 2.0);

    if (depth < 0.0) {
        outward = arcStation;
        height = arcHeight;
    }

    const double cosA = std::cos(angle);
    const double sinA = std::sin(angle);
    if (station <= 0.0) return {-outward, height};
    return {cosA * outward - sinA * height, sinA * outward + cosA * height};
}

double hingeSupport(const std::vector<double>& stations,
    const std::vector<double>& distances,
    const std::vector<char>& mixed, double radius, double angle) {
    // 逐行对应参考实现：
    //   reach = max(radius, distance[mixed].max()) if mixed.any() else radius
    //   rigid_edges = distance[(~mixed) & (distance > reach)]
    //   support = rigid_edges.min() if rigid_edges.size else reach + radius
    double reachDistance = radius;
    bool anyMixed = false;
    for (size_t i = 0; i < mixed.size(); ++i) {
        if (!mixed[i]) continue;
        anyMixed = true;
        if (distances[i] > reachDistance) reachDistance = distances[i];
    }
    (void)anyMixed;

    double support = -1.0;
    for (size_t i = 0; i < mixed.size(); ++i) {
        if (mixed[i] || distances[i] <= reachDistance) continue;
        if (support < 0.0 || distances[i] < support) support = distances[i];
    }
    if (support < 0.0) support = reachDistance + radius;
    (void)stations;

    const double sineHalf = std::sin(angle * 0.5);
    support = std::min(support, 2.6 * radius) + 1.6 * radius * sineHalf * sineHalf;
    return support > 1e-12 ? support : radius;
}

std::vector<Vec3> hingeDisplacement(
    const std::vector<double>& stations,
    const std::vector<double>& depths,
    const std::vector<double>& laterals,
    const std::vector<Vec3>& current,
    const std::vector<HingeWeight>& weights,
    const HingeParams& params) {

    const size_t count = stations.size();
    std::vector<Vec3> out(count, Vec3{0.0, 0.0, 0.0});
    if (count == 0 || params.radius <= 1e-12) return out;

    const double plane = (params.direction + (params.bend < 0.0 ? 180.0 : 0.0)) * degToRad;
    const double cosPlane = std::cos(plane);
    const double sinPlane = std::sin(plane);
    const double angle = std::abs(params.bend) * degToRad;

    // 与参考实现一致：整条肢体用同一个 support，由权重混合区推断。
    std::vector<char> mixed(count, 0);
    std::vector<double> distances(count, 0.0);
    for (size_t i = 0; i < count; ++i) {
        const HingeWeight& w = weights[i];
        int positive = 0;
        if (w.parent > 0.0) ++positive;
        if (w.child > 0.0) ++positive;
        if (w.helper > 0.0) ++positive;
        mixed[i] = positive > 1 ? 1 : 0;
        distances[i] = std::abs(stations[i]);
    }
    const double support = hingeSupport(stations, distances, mixed, params.radius, angle);

    for (size_t i = 0; i < count; ++i) {
        const HingeWeight& w = weights[i];
        double phase = params.twist * degToRad * w.child;
        if (w.helper > 0.0) phase += params.twist * degToRad * 0.35 * w.helper;

        const double cosPhase = std::cos(phase - plane);
        const double sinPhase = std::sin(phase - plane);
        const double depth = depths[i] * cosPhase - laterals[i] * sinPhase;
        const double lateral = depths[i] * sinPhase + laterals[i] * cosPhase;

        const double ext = params.radius;
        const HingePoint inner = hingeSection(stations[i], ext, params.radius,
            w.parent, w.child, w.helper, angle, support);
        const HingePoint outer = hingeSection(stations[i], -ext, params.radius,
            w.parent, w.child, w.helper, angle, support);
        const double neutralX = (inner.x + outer.x) * 0.5;
        const double neutralH = (inner.h + outer.h) * 0.5;
        const double gradientX = (inner.x - outer.x) / (2.0 * params.radius);
        const double gradientH = (inner.h - outer.h) / (2.0 * params.radius);

        const double x = neutralX + depth * gradientX;
        const double h = neutralH + depth * gradientH;
        const Vec3 target{
            x,
            h * cosPlane - lateral * sinPlane,
            h * sinPlane + lateral * cosPlane
        };
        out[i] = v3_sub(target, current[i]);
    }
    return out;
}

} // namespace ocd_hinge
