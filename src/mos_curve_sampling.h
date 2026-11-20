#pragma once

#include "mos_math.h"

#include <optional>

namespace mos_curve_sampling {

using namespace mathcore;
using ocd_math::Mat3;

struct Joint {
    Vec3 position{};
    Quat rotation{0, 0, 0, 1};
};

struct Control {
    Vec3 position{};
    unsigned int joint = 0;
    std::array<std::optional<mos_math::Secondary>, 2> edits{};
};

struct Driver {
    unsigned int joint = 0;
    double slide = 0.0;
    double bulge = 0.0;
};

struct Guide {
    std::vector<Joint> joints;
    std::vector<Control> controls;
    std::vector<std::vector<Driver>> drivers;
    bool closed = false;
    double terminalLength = 0.0;

    std::vector<Vec3> axes;
    std::vector<Quat> frames;
    std::vector<Vec3> centers;
    std::vector<Mat3> bindFrames;
    std::vector<double> knots;
    std::vector<Vec3> samples;
};

struct Attachment {
    mos_math::Basis basis;
    Vec3 offset{};
    double parameter = 0.0;
    double distanceSquared = 0.0;
};

struct Bend {
    Vec3 inward{};
    double response = 0.0;
    double length = 0.0;
};

struct Pose {
    std::vector<Joint> joints;
    std::vector<Vec3> centers;
    std::vector<Vec3> axes;
    std::vector<Mat3> frames;
    std::vector<Bend> bends;
    bool secondary = false;
};

bool prepare(Guide& guide, int sampleCount);
bool bind(const Guide& guide, const Vec3& vertex, Attachment& attachment);
Pose makePose(const Guide& guide, const std::vector<Joint>& joints,
    const std::vector<Vec3>& scales, const std::vector<Vec3>& manualIn,
    double twistWeight, bool secondary);
Vec3 evaluate(const Guide& guide, const Pose& pose, const Attachment& attachment,
    const mos_math::Secondary& previous, const mos_math::Secondary& next);

} // namespace mos_curve_sampling
