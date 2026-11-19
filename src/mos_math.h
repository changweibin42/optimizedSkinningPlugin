#pragma once

#include "ocd_math.h"

#include <algorithm>
#include <array>
#include <cmath>

namespace mos_math {

using namespace mathcore;

struct Basis {
    int first = 0;
    int count = 0;
    std::array<double, 4> weights{};
};

// The nonzero cubic B-spline basis coefficients, evaluated only at bind time.
inline Basis sampleBasis(const std::vector<double>& knots, int degree, double t) {
    const int last = static_cast<int>(knots.size()) - degree - 2;
    if (degree < 1 || degree > 3 || last < degree) return {};
    t = std::clamp(t, knots[degree], knots[last + 1]);
    const auto end = knots.begin() + last + 1;
    const int span = static_cast<int>(
        std::upper_bound(knots.begin() + degree, end, t) - knots.begin()) - 1;
    Basis basis{span - degree, degree + 1, {1.0}};
    for (int level = 1; level <= degree; ++level) {
        double carry = 0.0;
        for (int index = 0; index < level; ++index) {
            const double left = t - knots[span - level + index + 1];
            const double right = knots[span + index + 1] - t;
            const double denominator = left + right;
            const double previous = denominator > 0.0 ? basis.weights[index] / denominator : 0.0;
            basis.weights[index] = carry + right * previous;
            carry = left * previous;
        }
        basis.weights[level] = carry;
    }
    return basis;
}

// DreamWorks US20130088497A1, Eq. 9. Each control has its own bind frame.
inline bool bindOffset(const Vec3& vertex, const std::vector<Vec3>& centers,
    const std::vector<ocd_math::Mat3>& frames, const Basis& basis, Vec3& offset) {
    if (basis.count == 0) return false;
    Vec3 center{};
    ocd_math::Mat3 matrix{};
    for (int k = 0; k < basis.count; ++k) {
        const int j = basis.first + k;
        const double w = basis.weights[k];
        center = v3_add(center, v3_scale(centers[j], w));
        for (int element = 0; element < 9; ++element) matrix.m[element] += w * frames[j].m[element];
    }
    ocd_math::Mat3 inverse;
    if (!matrix.inverse(inverse)) return false;
    offset = inverse.apply(v3_sub(vertex, center));
    return std::isfinite(offset.x) && std::isfinite(offset.y) && std::isfinite(offset.z);
}

struct Secondary {
    double bulgeInner = 0.0;
    double bulgeOuter = 0.0;
    double bulgeSide = 0.0;
    double slideInner = 0.0;
    double slideOuter = 0.0;
    double orthoInner = 0.0;
    double orthoOuter = 0.0;
    double centerInner = 0.0;
    double centerOuter = 0.0;
};

struct BendControl {
    Vec3 axis{};
    Vec3 inward{};
    Vec3 normal{};
    double response = 0.0;
    double length = 0.0;
    bool incoming = false;
};

inline Vec3 sampleControl(const Vec3& center, const ocd_math::Mat3& frame, const Vec3& offset,
    const BendControl& bend, const Secondary& settings) {
    const Vec3 mapped = frame.apply(offset);
    const Vec3 base = v3_add(center, mapped);
    if (bend.response <= 0.0) return base;
    const Vec3 radial = v3_sub(mapped, v3_scale(bend.axis, v3_dot(mapped, bend.axis)));
    const double radius = v3_len(radial);
    if (radius < 1e-9) return base;

    const Vec3 direction = v3_scale(radial, 1.0 / radius);
    const double facing = std::clamp(v3_dot(direction, bend.inward), -1.0, 1.0);
    const double inner = std::max(0.0, facing);
    const double outer = std::max(0.0, -facing);
    const double side = v3_dot(direction, bend.normal);
    const double bulge = settings.bulgeInner * inner * inner +
        settings.bulgeOuter * outer * outer + settings.bulgeSide * side * side;
    const double radialScale = std::max(-0.95, bulge * bend.response);
    const double travel = bend.length * bend.response;
    const double slide = (settings.slideInner * inner * inner - settings.slideOuter * outer * outer) *
        (bend.incoming ? travel : -travel);
    const double ortho = travel * (settings.orthoOuter * outer * outer - settings.orthoInner * inner * inner);
    const double centerSlide = travel * (settings.centerInner * inner * inner -
        settings.centerOuter * outer * outer);
    return v3_add(base, v3_add(v3_scale(direction, radialScale * radius + ortho),
        v3_add(v3_scale(bend.axis, slide), v3_scale(bend.inward, centerSlide))));
}

} // namespace mos_math
