// Algorithm 1 from Le and Hodgins (2016). The host preserves the order
// of the sparse influences and supplies positions in mesh-centered world space.
__kernel void cor_skin_cluster(
    __global const float* input,
    __global float* output,
    __global const uint* offsets,
    __global const uint* boneIds,
    __global const float* weights,
    __global const float4* transforms,
    const uint vertexCount,
    const float envelope,
    __global const float* centers)
{
    const size_t vertex = get_global_id(0);
    if (vertex >= vertexCount) return;

    const uint first = offsets[vertex];
    const uint last = offsets[vertex + 1];
    const float3 point = vload3(vertex, input);
    if (first == last) {
        vstore3(point, vertex, output);
        return;
    }

    const float3 center = vload3(vertex, centers);
    float3 posedCenter = (float3)(0.0f);
    float4 sum = (float4)(0.0f);
    for (uint i = first; i < last; ++i) {
        const uint bone = boneIds[i];
        const float4 rotation = transforms[2 * bone];
        const float3 translation = transforms[2 * bone + 1].xyz;
        const float alignment = dot(sum, rotation);
        // The CPU uses double precision for the sign test. Near a tie,
        // request the CPU path rather than let rounding change the rotation.
        if (i != first && fabs(alignment) <= 2e-6f * length(sum)) {
            vstore3((float3)(NAN), vertex, output);
            return;
        }
        const float weight = weights[i];
        sum += rotation * (alignment < 0.0f ? -weight : weight);
        posedCenter += (q_rotate(rotation, center) + translation) * weight;
    }

    const float4 rotation = q_norm4(sum);
    const float3 deformed = q_rotate(rotation, point - center) + posedCenter;
    vstore3(envelope * deformed + (1.0f - envelope) * point, vertex, output);
}
