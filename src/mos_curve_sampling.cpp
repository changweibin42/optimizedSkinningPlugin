#include "mos_curve_sampling.h"

#include <limits>

namespace mos_curve_sampling {
namespace {

constexpr double pi = 3.14159265358979323846;
constexpr double epsilon = 1e-10;

Vec3 perpendicular(const Vec3& v, const Vec3& axis) {
    return v3_sub(v, v3_scale(axis, v3_dot(v, axis)));
}

Quat transport(const Quat& frame, const Vec3& from, const Vec3& to) {
    if (v3_dot(from, to) < -1.0 + 1e-9)
        return q_mul(q_axis_angle(q_rotate_v3(frame, {0, 1, 0}), pi), frame);
    return q_mul(q_between(from, to), frame);
}

std::vector<Vec3> restAxes(const Guide& guide) {
    const auto count = guide.joints.size();
    std::vector<Vec3> axes(count);
    for (size_t i = 0; i < count; ++i) {
        if (!guide.closed && i + 1 == count) axes[i] = axes[i - 1];
        else axes[i] = v3_norm(v3_sub(guide.joints[(i + 1) % count].position,
            guide.joints[i].position));
    }
    return axes;
}

std::vector<Quat> transportedFrames(const std::vector<Vec3>& axes, const Quat& root,
    bool closed) {
    std::vector<Quat> frames(axes.size());
    frames[0] = transport(root, q_rotate_v3(root, {1, 0, 0}), axes.front());
    for (size_t i = 1; i < axes.size(); ++i)
        frames[i] = transport(frames[i - 1], axes[i - 1], axes[i]);

    if (closed) {
        // The duplicated first transform receives the frame from the last
        // transform. Distribute its roll difference before periodic sampling.
        const auto ghost = transport(frames.back(), axes.back(), axes.front());
        const auto endNormal = q_rotate_v3(ghost, {0, 1, 0});
        const auto startNormal = q_rotate_v3(frames.front(), {0, 1, 0});
        const double closure = std::atan2(v3_dot(axes.front(), v3_cross(endNormal, startNormal)),
            v3_dot(endNormal, startNormal));
        for (size_t i = 1; i < frames.size(); ++i)
            frames[i] = q_mul(q_axis_angle(axes[i], closure * i / frames.size()), frames[i]);
    }
    return frames;
}

double angle(const Vec3& before, const Vec3& after) {
    return std::atan2(v3_len(v3_cross(before, after)), v3_dot(before, after));
}

double smooth(double x) {
    x = std::clamp(x, 0.0, 1.0);
    return x * x * (3.0 - 2.0 * x);
}

double spanLength(const Guide& guide, size_t segment) {
    if (!guide.closed && segment + 1 == guide.joints.size()) return guide.terminalLength;
    const auto& a = guide.joints[segment].position;
    const auto& b = guide.joints[(segment + 1) % guide.joints.size()].position;
    return v3_len(v3_sub(b, a));
}

void generateControls(Guide& guide) {
    for (unsigned int i = 0; i < guide.joints.size(); ++i) {
        const auto start = guide.joints[i].position;
        const auto extent = v3_scale(guide.axes[i], spanLength(guide, i));
        if (guide.closed) {
            for (const double t : {0.0, 1.0 / 3.0, 2.0 / 3.0})
                guide.controls.push_back({v3_add(start, v3_scale(extent, t)), i});
        } else {
            const double first = i == 0 ? 0.0 : 0.08;
            const double last = i + 1 == guide.joints.size() ? 1.0 : 0.92;
            for (const double t : {first, 0.33, 0.67, last})
                guide.controls.push_back({v3_add(start, v3_scale(extent, t)), i});
        }
    }
}

Vec3 controlEdit(const Guide& guide, const Pose& pose, const Control& control,
    const Vec3& mapped, const std::vector<Driver>& drivers,
    const mos_math::Secondary& previous, const mos_math::Secondary& next) {
    const auto axis = pose.axes[control.joint];
    const auto radial = perpendicular(mapped, axis);
    const double radius = v3_len(radial);
    if (radius <= epsilon) return {};
    const auto outward = v3_scale(radial, 1.0 / radius);
    Vec3 edit{};
    for (const auto& driver : drivers) {
        const auto& bend = pose.bends[driver.joint];
        const auto inward = v3_norm(perpendicular(bend.inward, axis));
        if (bend.response <= epsilon || v3_len(inward) <= epsilon) continue;
        const auto sideways = v3_cross(axis, inward);
        const double facing = std::clamp(v3_dot(outward, inward), -1.0, 1.0);
        const double in = std::max(facing, 0.0);
        const double out = std::max(-facing, 0.0);
        const double lateral = v3_dot(outward, sideways);
        const bool isPrevious = guide.closed
            ? control.joint == (driver.joint + guide.joints.size() - 1) % guide.joints.size()
            : control.joint < driver.joint;
        const auto& authored = control.edits[isPrevious ? 0 : 1];
        const auto& settings = authored ? *authored : (isPrevious ? previous : next);
        const double slide = (settings.slideInner * in * in - settings.slideOuter * out * out) *
            (isPrevious ? 1.0 : -1.0);
        const double ortho = settings.orthoInner * in * in + settings.orthoOuter * out * out;
        const double distance = v3_len(v3_sub(control.position, guide.joints[driver.joint].position));
        const double central = 1.0 - smooth(distance / std::max(0.5 * bend.length, epsilon));
        const double center = central * (settings.centerInner * in * in -
            settings.centerOuter * out * out);
        const double bulge = settings.bulgeInner * in * in + settings.bulgeOuter * out * out;
        const auto bulgeEdit = v3_add(v3_scale(outward, bulge),
            v3_scale(sideways, settings.bulgeSide * lateral * std::abs(lateral)));
        const auto slideEdit = v3_add(v3_scale(axis, slide),
            v3_add(v3_scale(outward, ortho), v3_scale(inward, center)));
        edit = v3_add(edit, v3_scale(v3_add(v3_scale(bulgeEdit, radius * driver.bulge),
            v3_scale(slideEdit, bend.length * driver.slide)), bend.response));
    }
    return edit;
}

} // namespace

bool prepare(Guide& guide, int sampleCount) {
    const auto count = guide.joints.size();
    if (count < (guide.closed ? 3u : 2u)) return false;
    guide.axes = restAxes(guide);
    for (const auto& axis : guide.axes)
        if (v3_len(axis) < epsilon) return false;
    if (!guide.closed && guide.terminalLength <= epsilon)
        guide.terminalLength = spanLength(guide, count - 2);
    guide.frames = transportedFrames(guide.axes, guide.joints.front().rotation, guide.closed);
    if (guide.controls.empty()) generateControls(guide);
    const int controlCount = static_cast<int>(guide.controls.size());
    if (controlCount < 4) return false;

    if (guide.drivers.empty()) {
        guide.drivers.resize(controlCount);
        for (int i = 0; i < controlCount; ++i) {
            const auto& control = guide.controls[i];
            if (control.joint >= count) return false;
            const auto appendDriver = [&](unsigned int index, double length) {
                if (!guide.closed && index == 0) return;
                const double distance = v3_len(v3_sub(control.position, guide.joints[index].position));
                const double influence = 1.0 - smooth(distance / std::max(length, epsilon));
                if (influence > 0.0) guide.drivers[i].push_back({index, influence, influence});
            };
            const auto next = (control.joint + 1) % count;
            const auto length = spanLength(guide, control.joint);
            appendDriver(control.joint, length);
            if (guide.closed || next != 0) appendDriver(next, length);
        }
    }
    if (guide.drivers.size() != guide.controls.size()) return false;

    const int padded = controlCount + (guide.closed ? 3 : 0);
    guide.centers.resize(padded);
    guide.bindFrames.resize(padded);
    for (int i = 0; i < padded; ++i) {
        const auto index = i % controlCount;
        const auto& control = guide.controls[index];
        if (control.joint >= count) return false;
        for (const auto& driver : guide.drivers[index])
            if (driver.joint >= count) return false;
        guide.centers[i] = control.position;
        guide.bindFrames[i] = ocd_math::mat3_from_quat(guide.frames[control.joint]);
    }
    if (guide.closed) {
        guide.knots.resize(padded + 4);
        for (int i = 0; i < padded + 4; ++i)
            guide.knots[i] = static_cast<double>(i - 3) / controlCount;
    } else {
        guide.knots = ocd_math::make_clamped_knots(padded, 3);
    }
    sampleCount = std::max({64, sampleCount, controlCount * 8});
    guide.samples.resize(sampleCount + 1);
    for (int i = 0; i <= sampleCount; ++i)
        guide.samples[i] = ocd_math::bspline_eval(guide.centers, guide.knots, 3,
            static_cast<double>(i) / sampleCount);
    return true;
}

bool bind(const Guide& guide, const Vec3& vertex, Attachment& attachment) {
    const auto last = guide.samples.size() - 1;
    size_t best = 0;
    auto closest = std::numeric_limits<double>::infinity();
    for (size_t i = 0; i <= last; ++i) {
        const auto delta = v3_sub(vertex, guide.samples[i]);
        const auto distance = v3_dot(delta, delta);
        if (distance < closest) {
            best = i;
            closest = distance;
        }
    }
    double parameter = static_cast<double>(best) / last;
    const auto refined = ocd_math::refine_closest_t(guide.centers, guide.knots, 3, vertex, parameter);
    const auto delta = v3_sub(vertex, ocd_math::bspline_eval(guide.centers, guide.knots, 3, refined));
    if (v3_dot(delta, delta) < closest) {
        parameter = refined;
        closest = v3_dot(delta, delta);
    }
    const auto basis = mos_math::sampleBasis(guide.knots, 3, parameter);
    Vec3 offset{};
    if (!mos_math::bindOffset(vertex, guide.centers, guide.bindFrames, basis, offset)) return false;
    attachment = {basis, offset, parameter, closest};
    return true;
}

Pose makePose(const Guide& guide, const std::vector<Joint>& joints,
    const std::vector<Vec3>& scales, const std::vector<Vec3>& manualIn,
    double twistWeight, bool secondary) {
    Pose result;
    result.joints = joints;
    result.secondary = secondary;
    const auto count = joints.size();
    std::vector<Quat> delta(count);
    result.axes.resize(count);
    for (size_t i = 0; i < count; ++i) {
        delta[i] = q_mul(joints[i].rotation, q_inverse(guide.joints[i].rotation));
        result.axes[i] = q_rotate_v3(delta[i], guide.axes[i]);
    }
    const auto transported = transportedFrames(result.axes, joints.front().rotation, guide.closed);
    std::vector<Mat3> jointFrames(count);
    for (size_t i = 0; i < count; ++i) {
        auto full = q_mul(delta[i], guide.frames[i]);
        if (twistWeight < 1.0) {
            const auto from = q_rotate_v3(transported[i], {0, 1, 0});
            const auto to = q_rotate_v3(full, {0, 1, 0});
            const double twist = std::atan2(v3_dot(result.axes[i], v3_cross(from, to)), v3_dot(from, to));
            full = q_mul(q_axis_angle(result.axes[i], -(1.0 - twistWeight) * twist), full);
        }
        auto frame = ocd_math::mat3_from_quat(full);
        if (!scales.empty())
            frame = frame * ocd_math::mat3_anisotropic_scale(scales[i].x, scales[i].y, scales[i].z);
        jointFrames[i] = frame;
    }

    result.bends.resize(count);
    if (secondary) {
        for (size_t i = guide.closed ? 0 : 1; i < count; ++i) {
            const auto before = (i + count - 1) % count;
            const auto& incoming = result.axes[before];
            const auto& outgoing = result.axes[i];
            auto inward = v3_norm(v3_sub(outgoing, incoming));
            if (!manualIn.empty() && v3_len(manualIn[i]) > epsilon)
                inward = q_rotate_v3(joints[i].rotation, v3_norm(manualIn[i]));
            const double response = std::abs(angle(incoming, outgoing) -
                angle(guide.axes[before], guide.axes[i])) / pi;
            result.bends[i] = {inward, response,
                std::min(spanLength(guide, before), spanLength(guide, i))};
        }
    }

    result.centers.resize(guide.centers.size());
    result.frames.resize(guide.centers.size());
    for (size_t i = 0; i < result.centers.size(); ++i) {
        const auto joint = guide.controls[i % guide.controls.size()].joint;
        const auto local = v3_sub(guide.centers[i], guide.joints[joint].position);
        auto moved = q_rotate_v3(delta[joint], local);
        if (!scales.empty()) {
            const auto coordinate = q_rotate_v3(q_inverse(guide.frames[joint]), local);
            moved = jointFrames[joint].apply(coordinate);
        }
        result.centers[i] = v3_add(joints[joint].position, moved);
        result.frames[i] = jointFrames[joint];
    }
    return result;
}

Vec3 evaluate(const Guide& guide, const Pose& pose, const Attachment& attachment,
    const mos_math::Secondary& previous, const mos_math::Secondary& next) {
    Vec3 result{};
    const auto& basis = attachment.basis;
    for (int k = 0; k < basis.count; ++k) {
        const auto index = basis.first + k;
        const auto mapped = pose.frames[index].apply(attachment.offset);
        auto sample = v3_add(pose.centers[index], mapped);
        if (pose.secondary) {
            const auto unique = index % guide.controls.size();
            sample = v3_add(sample, controlEdit(guide, pose, guide.controls[unique], mapped,
                guide.drivers[unique], previous, next));
        }
        result = v3_add(result, v3_scale(sample, basis.weights[k]));
    }
    return result;
}

} // namespace mos_curve_sampling
