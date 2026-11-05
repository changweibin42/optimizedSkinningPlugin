// ================================================================
// kernel_dqs.cl — DQS 补偿 (Sulema 2020 Algorithm 1)
// ================================================================

inline float4 q_mul(float4 a, float4 b) {
    return (float4)(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}
inline float4 q_conj(float4 q) { return (float4)(-q.x, -q.y, -q.z, q.w); }
inline float3 q_rotate(float4 q, float3 v) {
    float4 vq = (float4)(v.x, v.y, v.z, 0.0f);
    float4 r = q_mul(q_mul(q, vq), q_conj(q));
    return (float3)(r.x, r.y, r.z);
}
inline void dq_from_C(float4 Cq, float3 Ct, float4* qr, float4* qd) {
    float4 tq = (float4)(Ct.x, Ct.y, Ct.z, 0.0f);
    *qr = Cq;
    *qd = 0.5f * q_mul(tq, Cq);
}
inline float3 dlb_transform(float4 qr, float4 qd, float3 p) {
    float3 pr = q_rotate(qr, p);
    float4 tr = q_mul(qd, q_conj(qr));
    return pr + (float3)(2.0f*tr.x, 2.0f*tr.y, 2.0f*tr.z);
}
inline float pt_to_seg_dist(float3 p, float3 a, float3 b) {
    float3 ab = b - a;
    float ab_sq = dot(ab, ab);
    if (ab_sq < 1e-12f) return length(p - a);
    float t = clamp(dot(p - a, ab) / ab_sq, 0.0f, 1.0f);
    return length(p - (a + ab * t));
}

__kernel void dqs_deform(
    __global const float* inPos,
    __global float* outPos,
    __global const int* boneIds,
    __global const float* boneWts,
    __global const int* boneCounts,
    __global const float* restQ,
    __global const float* restT,
    __global const float* poseQ,
    __global const float* poseT,
    __global const float* boneDirs,
    __global const float* jointPos,
    __global const int* affectMap,
    const int numAffect,
    const int hasAffectMap,
    const int numBones,
    const float envelope,
    const float dqsStrength)
{
    int i = get_global_id(0);
    if (i >= numAffect) return;

    int idx = hasAffectMap ? affectMap[i] : i;
    float3 p = (float3)(inPos[idx*3+0], inPos[idx*3+1], inPos[idx*3+2]);
    int n = boneCounts[idx];

    if (n <= 0) {
        outPos[idx*3+0] = p.x;
        outPos[idx*3+1] = p.y;
        outPos[idx*3+2] = p.z;
        return;
    }

    // Step 1: DLB 基础变形
    int refIdx = boneIds[idx*4 + 0];
    for (int k = 1; k < n; ++k) {
        int b = boneIds[idx*4 + k];
        if (b < refIdx) refIdx = b;
    }
    float4 refQ = (float4)(
        poseQ[refIdx*4+0], poseQ[refIdx*4+1],
        poseQ[refIdx*4+2], poseQ[refIdx*4+3]);

    float4 qr_sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    float4 qd_sum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);

    for (int k = 0; k < n; ++k) {
        int b = boneIds[idx*4 + k];
        float w = boneWts[idx*4 + k];
        if (w < 1e-9f) continue;

        float4 restQb = (float4)(restQ[b*4+0], restQ[b*4+1], restQ[b*4+2], restQ[b*4+3]);
        float4 poseQb = (float4)(poseQ[b*4+0], poseQ[b*4+1], poseQ[b*4+2], poseQ[b*4+3]);
        float4 Cq = q_mul(poseQb, restQb);

        float3 restTb = (float3)(restT[b*3+0], restT[b*3+1], restT[b*3+2]);
        float3 poseTb = (float3)(poseT[b*3+0], poseT[b*3+1], poseT[b*3+2]);
        float3 Ct = poseTb + q_rotate(poseQb, restTb);

        float4 qr, qd;
        dq_from_C(Cq, Ct, &qr, &qd);

        float d = qr.x*refQ.x + qr.y*refQ.y + qr.z*refQ.z + qr.w*refQ.w;
        if (d < 0.0f) { qr = -qr; qd = -qd; }

        qr_sum += qr * w;
        qd_sum += qd * w;
    }

    float nrm = sqrt(qr_sum.x*qr_sum.x + qr_sum.y*qr_sum.y +
                     qr_sum.z*qr_sum.z + qr_sum.w*qr_sum.w);
    if (nrm < 1e-8f) {
        outPos[idx*3+0] = p.x;
        outPos[idx*3+1] = p.y;
        outPos[idx*3+2] = p.z;
        return;
    }

    float4 qr = qr_sum * (1.0f / nrm);
    float4 qd = qd_sum * (1.0f / nrm);
    float3 dqs_p = dlb_transform(qr, qd, p);

    // Step 2: 前三大权重骨骼
    int maxIdx = 0;
    for (int k = 1; k < n; ++k)
        if (boneWts[idx*4 + k] > boneWts[idx*4 + maxIdx]) maxIdx = k;

    int I1 = boneIds[idx*4 + maxIdx];
    float W1 = boneWts[idx*4 + maxIdx];

    int secondIdx = -1;
    for (int k = 0; k < n; ++k) {
        if (k == maxIdx) continue;
        if (secondIdx == -1 || boneWts[idx*4+k] > boneWts[idx*4+secondIdx])
            secondIdx = k;
    }
    if (secondIdx == -1) {
        float3 r = dqs_p * envelope + p * (1.0f - envelope);
        outPos[idx*3+0] = r.x; outPos[idx*3+1] = r.y; outPos[idx*3+2] = r.z;
        return;
    }
    int I2 = boneIds[idx*4 + secondIdx];
    float W2 = boneWts[idx*4 + secondIdx];

    int thirdIdx = -1;
    for (int k = 0; k < n; ++k) {
        if (k == maxIdx || k == secondIdx) continue;
        if (thirdIdx == -1 || boneWts[idx*4+k] > boneWts[idx*4+thirdIdx])
            thirdIdx = k;
    }
    float W3 = (thirdIdx != -1) ? boneWts[idx*4 + thirdIdx] : 0.0f;

    if (W1 + W2 < 1e-6f) {
        float3 r = dqs_p * envelope + p * (1.0f - envelope);
        outPos[idx*3+0] = r.x; outPos[idx*3+1] = r.y; outPos[idx*3+2] = r.z;
        return;
    }

    // Step 3: V_bisector
    float3 bd1 = (float3)(boneDirs[I1*3+0], boneDirs[I1*3+1], boneDirs[I1*3+2]);
    float3 bd2 = (float3)(boneDirs[I2*3+0], boneDirs[I2*3+1], boneDirs[I2*3+2]);
    float3 bisector = bd1 + bd2;
    float bl = length(bisector);
    if (bl < 1e-6f) {
        float3 r = dqs_p * envelope + p * (1.0f - envelope);
        outPos[idx*3+0] = r.x; outPos[idx*3+1] = r.y; outPos[idx*3+2] = r.z;
        return;
    }
    bisector = bisector / bl;
    if (I2 > I1) bisector = -bisector;

    // Step 4: RQ = RQ1 * RQ2^-1
    float4 q1 = (float4)(poseQ[I1*4+0], poseQ[I1*4+1], poseQ[I1*4+2], poseQ[I1*4+3]);
    float4 q2 = (float4)(poseQ[I2*4+0], poseQ[I2*4+1], poseQ[I2*4+2], poseQ[I2*4+3]);
    float4 q_rel = q_mul(q1, q_conj(q2));

    // Step 5: V_offset
    float3 axis = (float3)(q_rel.x, q_rel.y, q_rel.z);
    float ax_len = length(axis);
    float3 V_offset = bisector;
    if (ax_len > 1e-9f) {
        axis = axis / ax_len;
        V_offset = bisector - axis * dot(axis, bisector);
    }

    // Step 6: f(w)
    float w = W2 / (W1 + W2);
    float f = 2.2f*w - 8.1f*w*w + 7.4f*w*w*w;

    // Step 7: geo
    float cos_half = fabs(q_rel.w);
    if (cos_half > 1.0f) cos_half = 1.0f;
    float geo = 2.0f * sqrt(fmax(0.0f, 1.0f - cos_half));
    if (geo > 1.0f) geo = 1.0f;

    // Step 8: r = 到最近骨骼线段距离
    float r = 1e18f;
    for (int s = 0; s + 1 < numBones; ++s) {
        float3 a = (float3)(jointPos[s*3+0],     jointPos[s*3+1],     jointPos[s*3+2]);
        float3 b = (float3)(jointPos[(s+1)*3+0], jointPos[(s+1)*3+1], jointPos[(s+1)*3+2]);
        float d = pt_to_seg_dist(p, a, b);
        if (d < r) r = d;
    }
    if (r > 1e17f) r = 0.0f;

    // Step 9: k3
    float k3 = 1.0f;
    if (W3 > 1e-6f && (W2 + W3) > 1e-6f)
        k3 = 1.0f - W3 / (W2 + W3);

    // Step 10: 最终
    float l = f * r * geo * k3 * dqsStrength;
    float3 compensated = dqs_p + V_offset * l;
    float3 result = compensated * envelope + p * (1.0f - envelope);
    outPos[idx*3+0] = result.x;
    outPos[idx*3+1] = result.y;
    outPos[idx*3+2] = result.z;
}