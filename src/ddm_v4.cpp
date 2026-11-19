#include "ddm_v4.h"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstddef>
#include <numeric>

namespace ddm_v4 {
namespace {

using Graph = std::vector<std::vector<unsigned int>>;
using Channel = std::vector<double>;

double dot(const Channel& lhs, const Channel& rhs) {
    return std::inner_product(lhs.begin(), lhs.end(), rhs.begin(), 0.0);
}

class Diffusion {
public:
    explicit Diffusion(const Graph& neighbors)
        : neighbors_(neighbors), inverseSqrtDegree_(neighbors.size(), 1.0) {
        for (size_t i = 0; i < neighbors_.size(); ++i) {
            if (!neighbors_[i].empty())
                inverseSqrtDegree_[i] = 1.0 / std::sqrt(static_cast<double>(neighbors_[i].size()));
        }
    }

    bool smooth(Channel& field, double step, int passes) const {
        if (passes == 0 || step == 0.0) return true;

        // Solve (I + step * (I - D^-1 A)) x = b via its symmetric
        // equivalent with y = sqrt(D) x; isolated vertices are identity rows.
        for (size_t i = 0; i < field.size(); ++i) field[i] /= inverseSqrtDegree_[i];
        Channel residual(field.size()), direction(field.size()), product(field.size());
        for (int pass = 0; pass < passes; ++pass) {
            const double rhsNormSq = dot(field, field);
            const double toleranceSq = rhsNormSq * 1e-24;
            if (!std::isfinite(rhsNormSq)) return false;
            apply(field, step, product);
            for (size_t i = 0; i < field.size(); ++i)
                direction[i] = residual[i] = field[i] - product[i];
            double normSq = dot(residual, residual);
            for (int iteration = 0; normSq > toleranceSq && iteration < 256; ++iteration) {
                apply(direction, step, product);
                const double denominator = dot(direction, product);
                if (denominator <= 0.0 || !std::isfinite(denominator)) return false;
                const double scale = normSq / denominator;
                for (size_t i = 0; i < field.size(); ++i) {
                    field[i] += scale * direction[i];
                    residual[i] -= scale * product[i];
                }
                const double next = dot(residual, residual);
                if (!std::isfinite(next)) return false;
                const double beta = next / normSq;
                for (size_t i = 0; i < field.size(); ++i)
                    direction[i] = residual[i] + beta * direction[i];
                normSq = next;
            }
            if (normSq > toleranceSq || !std::isfinite(normSq)) return false;
        }
        for (size_t i = 0; i < field.size(); ++i) field[i] *= inverseSqrtDegree_[i];
        return true;
    }

private:
    void apply(const Channel& field, double step, Channel& output) const {
        for (size_t i = 0; i < neighbors_.size(); ++i) {
            const auto& neighbors = neighbors_[i];
            double sum = 0.0;
            for (unsigned int j : neighbors)
                sum += field[j] * inverseSqrtDegree_[j];
            output[i] = neighbors.empty() ? field[i]
                : (1.0 + step) * field[i] - step * sum * inverseSqrtDegree_[i];
        }
    }

    const Graph& neighbors_;
    Channel inverseSqrtDegree_;
};

void normalize(cor_paper::Weights& weights) {
    double sum = 0.0;
    for (const auto& weight : weights) sum += weight.second;
    for (auto& weight : weights) weight.second /= sum;
}

} // namespace

bool Settings::operator==(const Settings& other) const {
    return iterations == other.iterations && lambda == other.lambda &&
        kappa == other.kappa && alpha == other.alpha;
}

bool precompute(const std::vector<mathcore::Vec3>& restPoints,
    const std::vector<cor_paper::Weights>& inputWeights, size_t influenceCount,
    const std::vector<Edge>& edges, const Settings& settings, BindData& output) {
    const size_t count = restPoints.size();
    if (inputWeights.size() != count || settings.iterations < 0 ||
        !std::isfinite(settings.lambda) || !std::isfinite(settings.kappa) ||
        !std::isfinite(settings.alpha) || settings.lambda < 0.0 || settings.kappa < 0.0 ||
        settings.alpha < 0.0 || settings.alpha > 1.0) return false;

    Graph graph(count);
    for (const auto& [a, b] : edges) {
        if (a >= count || b >= count) return false;
        if (a == b || inputWeights[a].empty() || inputWeights[b].empty()) continue;
        graph[a].push_back(b);
        graph[b].push_back(a);
    }
    for (auto& neighbors : graph) {
        std::sort(neighbors.begin(), neighbors.end());
        neighbors.erase(std::unique(neighbors.begin(), neighbors.end()), neighbors.end());
    }
    const Diffusion diffusion(graph);

    BindData result;
    result.centers = restPoints;
    // p = B u because the input skin weights sum to one. No per-bone
    // position moments or full Omega matrices are needed by v4.
    const auto origin = restPoints.empty() ? mathcore::Vec3{} : restPoints.front();
    Channel coordinates[3];
    for (auto& channel : coordinates) channel.reserve(count);
    for (const auto& point : restPoints) {
        coordinates[0].push_back(point.x - origin.x);
        coordinates[1].push_back(point.y - origin.y);
        coordinates[2].push_back(point.z - origin.z);
    }
    for (auto& channel : coordinates)
        if (!diffusion.smooth(channel, settings.lambda, settings.iterations)) return false;
    for (size_t i = 0; i < count; ++i) {
        if (!graph[i].empty() && settings.iterations > 0 && settings.lambda > 0.0)
            result.centers[i] = {coordinates[0][i] + origin.x, coordinates[1][i] + origin.y,
                coordinates[2][i] + origin.z};
    }

    using InputEntry = std::pair<size_t, double>;
    std::vector<std::vector<InputEntry>> inputColumns(influenceCount);
    for (size_t i = 0; i < count; ++i) {
        for (const auto& [bone, weight] : inputWeights[i]) {
            if (bone >= influenceCount || !std::isfinite(weight) || weight <= 0.0) return false;
            inputColumns[bone].emplace_back(i, weight);
        }
    }

    const bool split = settings.alpha != 0.0 && settings.lambda != settings.kappa &&
        settings.iterations != 0;
    std::vector<Channel> rotationColumns(influenceCount), translationColumns(split ? influenceCount : 0);
    std::atomic<bool> valid{true};
#pragma omp parallel for schedule(dynamic, 1)
    for (std::ptrdiff_t bone = 0; bone < static_cast<std::ptrdiff_t>(influenceCount); ++bone) {
        if (!valid.load(std::memory_order_relaxed) || inputColumns[bone].empty()) continue;
        Channel input(count, 0.0);
        for (const auto& [vertex, weight] : inputColumns[bone]) input[vertex] = weight;
        auto& rotation = rotationColumns[bone];
        rotation = input;
        const bool rotationOk = diffusion.smooth(rotation, settings.lambda, settings.iterations);
        const bool translationOk = !split || diffusion.smooth(input, settings.kappa, settings.iterations);
        if (!rotationOk || !translationOk) {
            valid.store(false, std::memory_order_relaxed);
            continue;
        }
        if (split) {
            // psi = B w; omega = (1-alpha) psi + alpha C w (Eq. 11).
            for (size_t i = 0; i < count; ++i)
                input[i] = (1.0 - settings.alpha) * rotation[i] + settings.alpha * input[i];
            translationColumns[bone] = std::move(input);
        }
    }
    if (!valid.load(std::memory_order_relaxed)) return false;

    result.rotationWeights.resize(count);
    if (split) result.translationWeights.resize(count);
    for (unsigned int bone = 0; bone < influenceCount; ++bone) {
        if (rotationColumns[bone].empty()) continue;
        for (size_t i = 0; i < count; ++i) {
            // Analytically nonnegative; clip negative solver roundoff.
            if (rotationColumns[bone][i] > 0.0)
                result.rotationWeights[i].emplace_back(bone, rotationColumns[bone][i]);
            if (split && translationColumns[bone][i] > 0.0)
                result.translationWeights[i].emplace_back(bone, translationColumns[bone][i]);
        }
    }
    for (size_t i = 0; i < count; ++i) {
        if (inputWeights[i].empty()) continue;
        if (result.rotationWeights[i].empty() || (split && result.translationWeights[i].empty()))
            return false;
        normalize(result.rotationWeights[i]);
        if (split) normalize(result.translationWeights[i]);
    }
    output = std::move(result);
    return true;
}

} // namespace ddm_v4
