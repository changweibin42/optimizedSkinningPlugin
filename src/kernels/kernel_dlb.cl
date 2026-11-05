// ================================================================
// kernel_dlb.cl — DLB (Kavan 2008 Algorithm 1)
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
inline float4 q_norm4(float4 q) {
    float n = sqrt(q.x*q.x + q.y*q.y + q.z*q.z + q.w*q.w);
    if (n < 1e-12f) return (float4)(0.0f, 0.0f, 0.0f, 1.0f);
    float inv = 1.0f / n;
    return (float4)(q.x*inv, q.y*inv, q.z*inv, q.w*inv);
}
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

__kernel void dlb_deform(
    __global const float* inPos,
    __global float* outPos,
    __global const int* boneIds,
    __global const float* boneWts,
    __global const int* boneCounts,
    __global const float* restQ,
    __global const float* restT,
    __global const float* poseQ,
    __global const float* poseT,
    __global const int* affectMap,
    const int numAffect,
    const int hasAffectMap,
    const float envelope)
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

    // 参考 = 最小索引骨骼
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

        float4 restQb = (float4)(
            restQ[b*4+0], restQ[b*4+1], restQ[b*4+2], restQ[b*4+3]);
        float4 poseQb = (float4)(
            poseQ[b*4+0], poseQ[b*4+1], poseQ[b*4+2], poseQ[b*4+3]);
        float4 Cq = q_mul(poseQb, restQb);

        float3 restTb = (float3)(restT[b*3+0], restT[b*3+1], restT[b*3+2]);
        float3 poseTb = (float3)(poseT[b*3+0], poseT[b*3+1], poseT[b*3+2]);
        float3 rotBT = q_rotate(poseQb, restTb);
        float3 Ct = poseTb + rotBT;

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
    float3 result = dlb_transform(qr, qd, p);
    result = result * envelope + p * (1.0f - envelope);

    outPos[idx*3+0] = result.x;
    outPos[idx*3+1] = result.y;
    outPos[idx*3+2] = result.z;
}