#include "optimizedSkinCluster.h"

#include <maya/MDagPath.h>
#include <maya/MDoubleArray.h>
#include <maya/MFnDoubleArrayData.h>
#include <maya/MFnIntArrayData.h>
#include <maya/MFnMatrixData.h>
#include <maya/MFnMesh.h>
#include <maya/MGlobal.h>
#include <maya/MIntArray.h>
#include <maya/MPlugArray.h>
#include <maya/MQuaternion.h>
#include <maya/MTransformationMatrix.h>
#include <maya/MVector.h>

#include <climits>
#include <map>
#include <string>
#include <set>

namespace {

using namespace mathcore;
using namespace mos_curve_sampling;

struct Influence {
    MMatrix skin;
    Joint rest;
    Joint pose;
    MObject object;
};

Joint jointFromMatrix(const MMatrix& m) {
    const auto q = MTransformationMatrix(m).rotation();
    return {{m[3][0], m[3][1], m[3][2]}, {q.x, q.y, q.z, q.w}};
}

using Influences = std::map<unsigned int, Influence>;

std::vector<std::vector<unsigned int>> automaticGuides(const Influences& influences) {
    std::map<unsigned int, unsigned int> parent;
    std::set<unsigned int> hasChild;
    for (const auto& entry : influences) {
        if (!entry.second.object.hasFn(MFn::kJoint)) continue;
        MDagPath path;
        if (!MDagPath::getAPathTo(entry.second.object, path)) continue;
        while (path.length() > 1 && path.pop()) {
            const auto ancestor = path.node();
            const auto found = std::find_if(influences.begin(), influences.end(), [&](const auto& item) {
                return item.second.object == ancestor;
            });
            if (found == influences.end()) continue;
            parent.emplace(entry.first, found->first);
            hasChild.insert(found->first);
            break;
        }
    }

    std::vector<std::vector<unsigned int>> result;
    for (const auto& entry : parent) {
        if (hasChild.count(entry.first) != 0) continue;
        std::vector<unsigned int> chain{entry.first};
        auto p = parent.find(chain.back());
        while (p != parent.end()) {
            chain.push_back(p->second);
            p = parent.find(chain.back());
        }
        std::reverse(chain.begin(), chain.end());
        result.push_back(std::move(chain));
    }
    return result;
}

MIntArray readIntArray(MDataBlock& block, const MObject& attribute) {
    MStatus status;
    auto object = block.inputValue(attribute, &status).data();
    if (!status || object.isNull()) return {};
    return MFnIntArrayData(object).array();
}

MDoubleArray readDoubleArray(MDataBlock& block, const MObject& attribute) {
    MStatus status;
    auto object = block.inputValue(attribute, &status).data();
    if (!status || object.isNull()) return {};
    return MFnDoubleArrayData(object).array();
}

bool isIndex(double value, size_t limit) {
    return std::isfinite(value) && value >= 0.0 && value < limit && std::floor(value) == value;
}

bool finiteArray(const MDoubleArray& values) {
    for (unsigned int i = 0; i < values.length(); ++i)
        if (!std::isfinite(values[i])) return false;
    return true;
}

MStatus configurationError(const char* detail) {
    MGlobal::displayError(MString("[MOS Curve Sampling] ") + detail);
    return MS::kFailure;
}

} // namespace

MStatus OptimizedSkinCluster::deformMosCurveSampling(MDataBlock& block, MItGeometry& iter,
    const MMatrix& worldMat, unsigned int geometryIndex, const MosParams& params) {
    using namespace mos_curve_sampling;

    MStatus status;
    const double envelope = block.inputValue(MPxGeometryFilter::envelope, &status).asFloat();
    if (!status || envelope == 0.0) return status;

    auto poseMatrices = block.inputArrayValue(MPxSkinCluster::matrix, &status);
    if (!status) return status;
    auto bindMatrices = block.inputArrayValue(MPxSkinCluster::bindPreMatrix, &status);
    if (!status) return status;

    Influences influences;
    const unsigned int matrixCount = poseMatrices.elementCount();
    for (unsigned int i = 0; i < matrixCount; ++i) {
        if (!poseMatrices.jumpToArrayElement(i)) continue;
        const unsigned int id = poseMatrices.elementIndex();
        if (!bindMatrices.jumpToElement(id)) continue;
        auto bindObject = bindMatrices.inputValue(&status).data();
        if (!status || !bindObject.hasFn(MFn::kMatrixData)) continue;
        auto poseObject = poseMatrices.inputValue(&status).data();
        if (!status || !poseObject.hasFn(MFn::kMatrixData)) continue;

        const auto bindInverse = MFnMatrixData(bindObject).matrix();
        const auto bind = bindInverse.inverse();
        const auto pose = MFnMatrixData(poseObject).matrix();
        influences.emplace(id, Influence{bindInverse * pose,
            jointFromMatrix(bind), jointFromMatrix(pose), {}});
    }
    std::vector<unsigned int> influenceIds;
    for (const auto& entry : influences) influenceIds.push_back(entry.first);

    auto inputs = block.inputArrayValue(MPxSkinCluster::input, &status);
    if (!status) return status;
    status = inputs.jumpToElement(geometryIndex);
    if (!status) return status;
    const auto geometry = inputs.inputValue(&status).child(MPxSkinCluster::inputGeom).asMesh();
    if (!status || geometry.isNull()) return MS::kFailure;
    MFnMesh mesh(geometry);
    const auto vertexCount = mesh.numVertices(&status);
    if (!status) return status;
    rebuildVtxCache(block, vertexCount);

    auto& cache = m_mosSampling;
    if (!cache.valid || cache.geometryIndex != geometryIndex ||
        cache.weightVersion != m_currentWeightVersion || cache.sampleCount != params.curveSamples ||
        cache.vertices.size() != static_cast<size_t>(vertexCount) ||
        cache.influenceIds != influenceIds || !cache.geometryMatrix.isEquivalent(worldMat, 1e-12)) {
        MosSamplingCache next;
        next.geometryIndex = geometryIndex;
        next.weightVersion = m_currentWeightVersion;
        next.sampleCount = params.curveSamples;
        next.influenceIds = influenceIds;
        next.geometryMatrix = worldMat;

        MPointArray points;
        status = mesh.getPoints(points, MSpace::kObject);
        if (!status) return status;
        for (unsigned int i = 0; i < points.length(); ++i) points[i] *= worldMat;

        std::vector<std::vector<unsigned int>> chains;
        const auto authoredChains = readIntArray(block, aMosGuideInfluences);
        if (authoredChains.length() == 0) {
            MPlug matrixPlug(thisMObject(), MPxSkinCluster::matrix);
            for (auto& entry : influences) {
                MPlugArray connections;
                matrixPlug.elementByLogicalIndex(entry.first).connectedTo(connections, true, false);
                if (connections.length() != 0) entry.second.object = connections[0].node();
            }
            chains = automaticGuides(influences);
        } else {
            chains.emplace_back();
            for (unsigned int i = 0; i < authoredChains.length(); ++i) {
                const auto id = authoredChains[i];
                if (id == -1 && !chains.back().empty()) chains.emplace_back();
                else if (id < 0 || influences.count(id) == 0)
                    return configurationError("mosGuideInfluences contains an invalid influence index.");
                else chains.back().push_back(id);
            }
            if (chains.back().empty()) return configurationError("The last MOS guide is empty.");
        }
        for (auto& chain : chains) {
            MosSamplingGuide guide;
            guide.curve.closed = chain.size() > 1 && chain.front() == chain.back();
            if (guide.curve.closed) chain.pop_back();
            const std::set<unsigned int> unique(chain.begin(), chain.end());
            if (unique.size() != chain.size() || chain.size() < (guide.curve.closed ? 3u : 2u))
                return configurationError("A guide needs two distinct influences, or three for a closed loop.");
            guide.influences = std::move(chain);
            for (const auto id : guide.influences) guide.curve.joints.push_back(influences.at(id).rest);
            if (!guide.curve.closed) {
                const auto last = guide.curve.joints.back().position;
                const auto before = guide.curve.joints[guide.curve.joints.size() - 2].position;
                const auto axis = v3_norm(v3_sub(last, before));
                const auto tip = guide.influences.back();
                for (unsigned int i = 0; i < points.length(); ++i) {
                    const auto& weights = m_vtxCache.vertices[i];
                    const auto found = std::find_if(weights.begin(), weights.end(), [tip](const auto& w) {
                        return w.first == tip && w.second > 0.02f;
                    });
                    if (found != weights.end()) {
                        const Vec3 p{points[i].x, points[i].y, points[i].z};
                        guide.curve.terminalLength = std::max(guide.curve.terminalLength,
                            v3_dot(v3_sub(p, last), axis));
                    }
                }
            }
            next.guides.push_back(std::move(guide));
        }

        const auto controls = readDoubleArray(block, aMosGuideControls);
        const auto drivers = readDoubleArray(block, aMosControlDrivers);
        const auto values = readDoubleArray(block, aMosControlValues);
        const auto inVectors = readDoubleArray(block, aMosDriverInVectors);
        if (controls.length() % 5 != 0 || drivers.length() % 5 != 0 ||
            inVectors.length() % 4 != 0 || values.length() % 12 != 0 ||
            !finiteArray(values) || !finiteArray(controls) ||
            !finiteArray(drivers) || !finiteArray(inVectors))
            return configurationError("Expected complete, finite control, driver and in-vector records.");
        for (unsigned int i = 0; i < controls.length(); i += 5) {
            if (!isIndex(controls[i], next.guides.size()))
                return configurationError("A control refers to an invalid guide index.");
            auto& guide = next.guides[static_cast<size_t>(controls[i])].curve;
            if (!isIndex(controls[i + 1], guide.joints.size()))
                return configurationError("A control refers to an invalid joint slot.");
            guide.controls.push_back({{controls[i + 2], controls[i + 3], controls[i + 4]},
                static_cast<unsigned int>(controls[i + 1])});
        }
        for (auto& guide : next.guides) {
            if (!prepare(guide.curve, params.curveSamples) &&
                (authoredChains.length() != 0 || !guide.curve.controls.empty()))
                return configurationError("A guide has coincident joints or fewer than four controls.");
        }
        std::set<std::pair<size_t, size_t>> authoredControls;
        for (unsigned int i = 0; i < drivers.length(); i += 5) {
            if (!isIndex(drivers[i], next.guides.size()))
                return configurationError("A driver refers to an invalid guide index.");
            const auto guideIndex = static_cast<size_t>(drivers[i]);
            auto& guide = next.guides[guideIndex].curve;
            if (!isIndex(drivers[i + 1], guide.drivers.size()) ||
                !isIndex(drivers[i + 2], guide.joints.size()) ||
                drivers[i + 3] < 0.0 || drivers[i + 4] < 0.0)
                return configurationError("A driver has an invalid control, joint slot or weight.");
            const auto controlIndex = static_cast<size_t>(drivers[i + 1]);
            auto& list = guide.drivers[controlIndex];
            if (authoredControls.emplace(guideIndex, controlIndex).second) list.clear();
            list.push_back({static_cast<unsigned int>(drivers[i + 2]), drivers[i + 3], drivers[i + 4]});
        }
        for (const auto& control : authoredControls) {
            auto& list = next.guides[control.first].curve.drivers[control.second];
            double slide = 0.0, bulge = 0.0;
            for (const auto& driver : list) {
                slide += driver.slide;
                bulge += driver.bulge;
            }
            if (!std::isfinite(slide) || !std::isfinite(bulge))
                return configurationError("The combined driver weights are too large.");
            for (auto& driver : list) {
                if (slide > 0.0) driver.slide /= slide;
                if (bulge > 0.0) driver.bulge /= bulge;
            }
        }
        for (unsigned int i = 0; i < values.length(); i += 12) {
            if (!isIndex(values[i], next.guides.size()))
                return configurationError("Control values refer to an invalid guide.");
            auto& guide = next.guides[static_cast<size_t>(values[i])].curve;
            if (!isIndex(values[i + 1], guide.controls.size()) || !isIndex(values[i + 2], 2))
                return configurationError("Control values require a valid control slot and side (0 or 1).");
            auto& edit = guide.controls[static_cast<size_t>(values[i + 1])].edits[
                static_cast<size_t>(values[i + 2])];
            edit = mos_math::Secondary{values[i + 3], values[i + 4], values[i + 5],
                values[i + 6], values[i + 7], values[i + 8], values[i + 9],
                values[i + 10], values[i + 11]};
        }
        for (unsigned int i = 0; i < inVectors.length(); i += 4) {
            if (!isIndex(inVectors[i], static_cast<size_t>(UINT_MAX)) ||
                influences.count(static_cast<unsigned int>(inVectors[i])) == 0)
                return configurationError("An in-vector refers to an invalid influence index.");
            next.manualIn[static_cast<unsigned int>(inVectors[i])] =
                {inVectors[i + 1], inVectors[i + 2], inVectors[i + 3]};
        }

        next.vertices.resize(points.length());
        const auto& allWeights = m_vtxCache.vertices;
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < vertexCount; ++i) {
            const auto& weights = allWeights[i];
            if (weights.empty()) continue;
            const Vec3 p{points[i].x, points[i].y, points[i].z};
            auto& vertex = next.vertices[i];
            double total = 0.0;
            for (const auto& w : weights) total += w.second;
            for (size_t j = 0; j < next.guides.size(); ++j) {
                const auto& guide = next.guides[j];
                if (guide.curve.samples.empty()) continue;
                double support = 0.0;
                for (const auto& w : weights) {
                    if (std::find(guide.influences.begin(), guide.influences.end(), w.first) !=
                        guide.influences.end()) support += w.second;
                }
                if (support <= 0.0) continue;
                Attachment attachment;
                if (bind(guide.curve, p, attachment) && (vertex.guide < 0 ||
                    attachment.distanceSquared < vertex.attachment.distanceSquared)) {
                    vertex = {static_cast<int>(j), support / total, attachment};
                }
            }
        }
        if (params.debugLog) {
            const auto count = std::count_if(next.vertices.begin(), next.vertices.end(),
                [](const auto& vertex) { return vertex.guide >= 0; });
            const auto message = "[MOS Curve Sampling] Bound " + std::to_string(count) +
                " / " + std::to_string(vertexCount) + " vertices.";
            MGlobal::displayInfo(MString(message.c_str()));
        }
        next.valid = true;
        cache = std::move(next);
    }

    const mos_math::Secondary previous{params.inBulgeWeight, params.outBulgeWeight, params.sideBulgeWeight,
        params.slideInWeight_prev, params.slideOutWeight_prev, params.slideOrthoInWeight_prev,
        params.slideOrthoOutWeight_prev, params.centerSlideInWeight, params.centerSlideOutWeight};
    const mos_math::Secondary next{params.inBulgeWeight, params.outBulgeWeight, params.sideBulgeWeight,
        params.slideInWeight_next, params.slideOutWeight_next, params.slideOrthoInWeight_next,
        params.slideOrthoOutWeight_next, params.centerSlideInWeight, params.centerSlideOutWeight};
    const bool nodeSecondary = params.inBulgeWeight != 0.0 || params.outBulgeWeight != 0.0 ||
        params.sideBulgeWeight != 0.0 || params.centerSlideInWeight != 0.0 ||
        params.centerSlideOutWeight != 0.0 || params.slideInWeight_prev != 0.0 ||
        params.slideInWeight_next != 0.0 || params.slideOutWeight_prev != 0.0 ||
        params.slideOutWeight_next != 0.0 || params.slideOrthoInWeight_prev != 0.0 ||
        params.slideOrthoInWeight_next != 0.0 || params.slideOrthoOutWeight_prev != 0.0 ||
        params.slideOrthoOutWeight_next != 0.0;

    std::vector<Pose> poses(cache.guides.size());
    for (size_t i = 0; i < cache.guides.size(); ++i) {
        const auto& guide = cache.guides[i];
        if (guide.curve.samples.empty()) continue;
        std::vector<Joint> joints;
        std::vector<Vec3> manual;
        std::vector<Vec3> scales;
        for (size_t j = 0; j < guide.influences.size(); ++j) {
            const auto id = guide.influences[j];
            const auto& influence = influences.at(id);
            joints.push_back(influence.pose);
            const auto vector = cache.manualIn.find(id);
            manual.push_back(vector != cache.manualIn.end() ? vector->second :
                params.useManualInVector ? params.manualInVector : Vec3{});

            if (params.scaleMode != 0 && params.scaleWeight > 0.0) {
                const auto scaleAlong = [&](const Vec3& direction) {
                    const auto axis = q_rotate_v3(guide.curve.frames[j], direction);
                    const MVector transformed = MVector(axis.x, axis.y, axis.z) * influence.skin;
                    return std::pow(std::max(1e-9, transformed.length()), params.scaleWeight);
                };
                scales.push_back({scaleAlong({1, 0, 0}),
                    params.scaleMode == 2 ? scaleAlong({0, 1, 0}) : 1.0,
                    params.scaleMode == 2 ? scaleAlong({0, 0, 1}) : 1.0});
            }
        }
        const bool secondary = nodeSecondary || std::any_of(guide.curve.controls.begin(),
            guide.curve.controls.end(), [](const auto& control) {
                return control.edits[0].has_value() || control.edits[1].has_value();
            });
        poses[i] = makePose(guide.curve, joints, scales, manual, params.twistWeight, secondary);
    }

    MPointArray points;
    status = iter.allPositions(points);
    if (!status) return status;
    std::vector<unsigned int> indices;
    indices.reserve(points.length());
    for (; !iter.isDone(); iter.next()) indices.push_back(iter.index());
    if (indices.size() != points.length()) return MS::kFailure;

    const auto worldInverse = worldMat.inverse();
    #pragma omp parallel for schedule(static)
    for (int row = 0; row < static_cast<int>(points.length()); ++row) {
        const auto index = indices[row];
        if (index >= cache.vertices.size()) continue;
        const auto& vertex = cache.vertices[index];
        const MPoint worldPoint = points[row] * worldMat;
        const Vec3 original{worldPoint.x, worldPoint.y, worldPoint.z};
        const double mix = std::clamp(vertex.coverage * params.influenceRadius, 0.0, 1.0);
        Vec3 result{};
        if (mix > 0.0) {
            const auto& guide = cache.guides[vertex.guide];
            result = v3_scale(evaluate(guide.curve, poses[vertex.guide], vertex.attachment, previous, next), mix);
        }
        if (mix < 1.0) {
            Vec3 linear{};
            double total = 0.0;
            for (const auto& weight : m_vtxCache.vertices[index]) {
                const auto influence = influences.find(weight.first);
                if (influence == influences.end()) continue;
                const MPoint p = worldPoint * influence->second.skin;
                linear = v3_add(linear, v3_scale({p.x, p.y, p.z}, weight.second));
                total += weight.second;
            }
            const auto lbs = total > 0.0 ? v3_scale(linear, 1.0 / total) : original;
            result = v3_add(result, v3_scale(lbs, 1.0 - mix));
        }
        result = applyEnvelope(original, result, envelope);
        points[row] = MPoint(result.x, result.y, result.z) * worldInverse;
    }
    status = iter.reset();
    if (!status) return status;
    return iter.setAllPositions(points);
}
