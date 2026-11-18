#include "dlb_gpu.h"
#include "dlb_gpu_kernel.h"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <iterator>
#include <limits>
#include <mutex>

#include <clew/clew.h>
#include <maya/MFnDependencyNode.h>
#include <maya/MGlobal.h>
#include <maya/MOpenCLAutoPtr.h>
#include <maya/MPoint.h>

namespace {

struct Buffer {
    MAutoCLMem storage;
    size_t bytes = 0;

    bool reserve(cl_context context, size_t required, cl_mem_flags flags) {
        if (bytes >= required) return true;
        cl_int status;
        const cl_mem buffer = clCreateBuffer(context, flags, required, nullptr, &status);
        if (status != CL_SUCCESS) return false;
        storage.attach(buffer);
        bytes = required;
        return true;
    }
};

bool initializeOpenCL() {
#ifdef _WIN32
    static const bool available = clewInit("OpenCL.dll") == CLEW_SUCCESS;
#elif defined(__APPLE__)
    static const bool available =
        clewInit("/System/Library/Frameworks/OpenCL.framework/OpenCL") == CLEW_SUCCESS;
#else
    static const bool available = clewInit("libOpenCL.so.1") == CLEW_SUCCESS;
#endif
    return available;
}

bool upload(cl_command_queue queue, const Buffer& buffer, const void* data, size_t bytes) {
    return clEnqueueWriteBuffer(queue, buffer.storage.get(), CL_TRUE, 0, bytes,
        data, 0, nullptr, nullptr) == CL_SUCCESS;
}

} // namespace

struct DlbGpu::State {
    std::mutex mutex;
    cl_context context = nullptr;
    cl_command_queue queue = nullptr;
    MAutoCLKernel kernel;
    Buffer input, output, offsets, boneIds, weights, transforms;

    std::vector<float> inputPoints, resultPoints, dualQuaternions;
    unsigned long long cachedWeightVersion = 0;
    size_t cachedPointCount = 0;
    size_t cachedTransformCount = 0;
    bool weightsUploaded = false;
    std::optional<std::string> initializationError;

    ~State() {
        if (queue) clReleaseCommandQueue(queue);
        if (context) clReleaseContext(context);
    }

    std::string initialize() {
        if (!initializeOpenCL()) return "OpenCL loader unavailable";

        cl_uint platformCount = 0;
        if (clGetPlatformIDs(0, nullptr, &platformCount) != CL_SUCCESS || platformCount == 0)
            return "no OpenCL platform";

        std::vector<cl_platform_id> platforms(platformCount);
        if (clGetPlatformIDs(platformCount, platforms.data(), nullptr) != CL_SUCCESS)
            return "could not enumerate OpenCL platforms";

        std::string reason = "no usable OpenCL GPU";
        for (const auto platform : platforms) {
            cl_uint deviceCount = 0;
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 0, nullptr, &deviceCount)
                != CL_SUCCESS || deviceCount == 0) continue;

            std::vector<cl_device_id> devices(deviceCount);
            if (clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, deviceCount, devices.data(), nullptr)
                != CL_SUCCESS) continue;

            for (const auto device : devices) {
                reason = initializeDevice(device);
                if (reason.empty()) return {};
            }
        }
        return reason;
    }

    std::string initializeDevice(cl_device_id device) {
        cl_int status;
        const cl_context candidate = clCreateContext(nullptr, 1, &device, nullptr, nullptr, &status);
        if (status != CL_SUCCESS) return "could not create OpenCL GPU context";

        const char* source = dlbOpenCLSource;
        const size_t sourceSize = sizeof(dlbOpenCLSource) - 1;
        const cl_program program = clCreateProgramWithSource(candidate, 1, &source, &sourceSize, &status);
        if (status != CL_SUCCESS) {
            clReleaseContext(candidate);
            return "could not create OpenCL program";
        }

        status = clBuildProgram(program, 1, &device, "", nullptr, nullptr);
        if (status != CL_SUCCESS) {
            size_t logSize = 0;
            clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG, 0, nullptr, &logSize);
            std::vector<char> log(logSize + 1, '\0');
            if (logSize) clGetProgramBuildInfo(program, device, CL_PROGRAM_BUILD_LOG,
                logSize, log.data(), nullptr);
            clReleaseProgram(program);
            clReleaseContext(candidate);
            return std::string("OpenCL kernel build failed: ") + log.data();
        }

        MAutoCLKernel candidateKernel;
        candidateKernel.attach(clCreateKernel(program, "dlb_skin_cluster", &status));
        clReleaseProgram(program);
        if (status != CL_SUCCESS) {
            clReleaseContext(candidate);
            return "could not create DLB kernel";
        }

        const cl_command_queue candidateQueue = clCreateCommandQueue(candidate, device, 0, &status);
        if (status != CL_SUCCESS) {
            clReleaseContext(candidate);
            return "could not create OpenCL command queue";
        }
        context = candidate;
        queue = candidateQueue;
        kernel = std::move(candidateKernel);
        return {};
    }

    bool uploadWeights(const VertexWeights& vertexWeights, unsigned long long version,
        size_t transformCount) {
        if (weightsUploaded && cachedWeightVersion == version &&
            cachedPointCount == vertexWeights.size() && cachedTransformCount == transformCount)
            return true;

        weightsUploaded = false;
        std::vector<std::uint32_t> hostOffsets;
        std::vector<std::uint32_t> hostIds;
        std::vector<float> hostWeights;
        hostOffsets.reserve(vertexWeights.size() + 1);
        hostOffsets.push_back(0);
        for (const auto& vertex : vertexWeights) {
            for (const auto& influence : vertex) {
                if (influence.first >= transformCount ||
                    hostIds.size() >= (std::numeric_limits<std::uint32_t>::max)() ||
                    !std::isfinite(influence.second)) return false;
                hostIds.push_back(influence.first);
                hostWeights.push_back(influence.second);
            }
            hostOffsets.push_back(static_cast<std::uint32_t>(hostIds.size()));
        }

        // OpenCL buffers need a nonzero allocation even on an unweighted mesh.
        if (hostIds.empty()) {
            hostIds.push_back(0);
            hostWeights.push_back(0);
        }
        const size_t offsetBytes = hostOffsets.size() * sizeof(hostOffsets.front());
        const size_t idBytes = hostIds.size() * sizeof(hostIds.front());
        const size_t weightBytes = hostWeights.size() * sizeof(float);
        weightsUploaded =
            offsets.reserve(context, offsetBytes, CL_MEM_READ_ONLY) &&
            boneIds.reserve(context, idBytes, CL_MEM_READ_ONLY) &&
            weights.reserve(context, weightBytes, CL_MEM_READ_ONLY) &&
            upload(queue, offsets, hostOffsets.data(), offsetBytes) &&
            upload(queue, boneIds, hostIds.data(), idBytes) &&
            upload(queue, weights, hostWeights.data(), weightBytes);
        if (!weightsUploaded) return false;
        cachedWeightVersion = version;
        cachedPointCount = vertexWeights.size();
        cachedTransformCount = transformCount;
        return true;
    }

    bool launch(cl_uint vertexCount, float envelope) {
        const size_t pointBytes = inputPoints.size() * sizeof(float);
        const size_t transformBytes = dualQuaternions.size() * sizeof(float);
        if (!input.reserve(context, pointBytes, CL_MEM_READ_ONLY) ||
            !output.reserve(context, pointBytes, CL_MEM_WRITE_ONLY) ||
            !transforms.reserve(context, transformBytes, CL_MEM_READ_ONLY) ||
            !upload(queue, input, inputPoints.data(), pointBytes) ||
            !upload(queue, transforms, dualQuaternions.data(), transformBytes)) return false;

        const cl_mem args[] = {input.storage.get(), output.storage.get(), offsets.storage.get(),
            boneIds.storage.get(), weights.storage.get(), transforms.storage.get()};
        cl_uint i = 0;
        for (const cl_mem arg : args) {
            if (clSetKernelArg(kernel.get(), i++, sizeof(arg), &arg) != CL_SUCCESS) return false;
        }
        if (clSetKernelArg(kernel.get(), i++, sizeof(vertexCount), &vertexCount) != CL_SUCCESS ||
            clSetKernelArg(kernel.get(), i, sizeof(envelope), &envelope) != CL_SUCCESS) return false;

        const size_t global = vertexCount;
        return clEnqueueNDRangeKernel(queue, kernel.get(), 1, nullptr, &global, nullptr,
            0, nullptr, nullptr) == CL_SUCCESS &&
            clEnqueueReadBuffer(queue, output.storage.get(), CL_TRUE, 0, pointBytes,
                resultPoints.data(), 0, nullptr, nullptr) == CL_SUCCESS;
    }
};

DlbGpu::DlbGpu() : state_(std::make_unique<State>()) {}
DlbGpu::~DlbGpu() = default;

bool DlbGpu::deform(MPointArray& points, const VertexWeights& weights,
    unsigned long long weightVersion, const std::vector<mathcore::RigidTransform>& transforms,
    const MMatrix& worldMatrix, const MMatrix& worldInverse, double envelope,
    const MObject& node) {
    State& gpu = *state_;
    const std::lock_guard<std::mutex> lock(gpu.mutex);
    const auto report = [&](const std::string& reason) {
        if (reportedStatus_ == reason) return;
        reportedStatus_ = reason;
        const MString prefix = MString("[optimizedSkinningPlugin] ") + MFnDependencyNode(node).name() + ": ";
        if (reason.empty()) MGlobal::displayInfo(prefix + "DLB OpenCL GPU active");
        else MGlobal::displayWarning(prefix + "DLB using CPU (" + reason.c_str() + ")");
    };

    const unsigned int count = points.length();
    if (!count || count != weights.size() || transforms.empty() ||
        !std::isfinite(envelope) || envelope <= 1e-6) return false;

    if (!gpu.initializationError) gpu.initializationError = gpu.initialize();
    if (!gpu.initializationError->empty()) {
        report(*gpu.initializationError);
        return false;
    }
    if (!gpu.uploadWeights(weights, weightVersion, transforms.size())) {
        report("could not upload vertex weights");
        return false;
    }

    gpu.inputPoints.resize(size_t(count) * 3);
    gpu.resultPoints.resize(gpu.inputPoints.size());
    gpu.dualQuaternions.resize(transforms.size() * 8);
    for (size_t i = 0; i < transforms.size(); ++i) {
        const auto dq = mathcore::dual_quat_from_C(transforms[i]);
        const double components[] = {dq.real.x, dq.real.y, dq.real.z, dq.real.w,
            dq.dual.x, dq.dual.y, dq.dual.z, dq.dual.w};
        std::copy(std::begin(components), std::end(components), gpu.dualQuaternions.begin() + i * 8);
    }
    for (unsigned int i = 0; i < count; ++i) {
        const MPoint world = points[i] * worldMatrix;
        const size_t k = size_t(i) * 3;
        gpu.inputPoints[k] = static_cast<float>(world.x);
        gpu.inputPoints[k + 1] = static_cast<float>(world.y);
        gpu.inputPoints[k + 2] = static_cast<float>(world.z);
    }

    const float gpuEnvelope = envelope >= 1.0 - 1e-6 ? 1.0f : static_cast<float>(envelope);
    if (!gpu.launch(count, gpuEnvelope)) {
        report("OpenCL dispatch or readback failed");
        return false;
    }

    // Keep the CPU path's sign choice at ambiguous rotations and reject
    // unreadable results before modifying any Maya geometry.
    if (!std::all_of(gpu.resultPoints.begin(), gpu.resultPoints.end(),
        [](float value) { return std::isfinite(value); })) {
        report("GPU result requires CPU precision");
        return false;
    }

    for (unsigned int i = 0; i < count; ++i) {
        if (weights[i].empty()) continue;
        const size_t k = size_t(i) * 3;
        points[i] = MPoint(gpu.resultPoints[k], gpu.resultPoints[k + 1],
            gpu.resultPoints[k + 2]) * worldInverse;
    }
    report({});
    return true;
}
