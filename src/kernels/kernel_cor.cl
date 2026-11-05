// ================================================================
// kernel_cor.cl
// CoR 蒙皮变形 GPU 内核（支持 affectMap）
// ================================================================

inline float4 q_mul(float4 a, float4 b) {
    return (float4)(
        a.w*b.x + a.x*b.w + a.y*b.z - a.z*b.y,
        a.w*b.y - a.x*b.z + a.y*b.w + a.z*b.x,
        a.w*b.z + a.x*b.y - a.y*b.x + a.z*b.w,
        a.w*b.w - a.x*b.x - a.y*b.y - a.z*b.z
    );
}

inline float4 q_conj(float4 q) {
    return (float4)(-q.x, -q.y, -q.z, q.w);
}

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

__kernel void cor_deform(
    __global const float* inPos,       // 0: full count * 3
    __global float* outPos,            // 1: full count * 3
    __global const float* cors,        // 2: full count * 3
    __global const int* boneIds,       // 3: full count * 4
    __global const float* boneWts,     // 4: full count * 4
    __global const int* boneCounts,    // 5: full count
    __global const float* restQ,       // 6: B*4
    __global const float* restT,       // 7: B*3
    __global const float* poseQ,       // 8: B*4
    __global const float* poseT,       // 9: B*3
    __global const int* affectMap,     // 10: affect count (可能为 NULL)
    const int numAffect,               // 11: kernel 执行次数 = affectCount
    const int hasAffectMap,            // 12: 0 或 1
    const float envelope)              // 13
{
    int i = get_global_id(0);
    if (i >= numAffect) return;

    // ★ 映射到真实顶点索引
    int idx = hasAffectMap ? affectMap[i] : i;

    float3 p   = (float3)(inPos[idx*3+0], inPos[idx*3+1], inPos[idx*3+2]);
    float3 cor = (float3)(cors[idx*3+0],  cors[idx*3+1],  cors[idx*3+2]);

    int n = boneCounts[idx];
    if (n <= 0) {
        outPos[idx*3+0] = p.x;
        outPos[idx*3+1] = p.y;
        outPos[idx*3+2] = p.z;
        return;
    }

    // 参考骨骼 = 最小索引
    int refIdx = boneIds[idx*4 + 0];
    for (int k = 1; k < n; ++k) {
        int b = boneIds[idx*4 + k];
        if (b < refIdx) refIdx = b;
    }
    float4 refQ = (float4)(
        poseQ[refIdx*4+0], poseQ[refIdx*4+1],
        poseQ[refIdx*4+2], poseQ[refIdx*4+3]);

    // QLERP
    float4 qSum = (float4)(0.0f, 0.0f, 0.0f, 0.0f);
    for (int k = 0; k < n; ++k) {
        int b = boneIds[idx*4 + k];
        float w = boneWts[idx*4 + k];
        if (w < 1e-9f) continue;
        float4 q = (float4)(
            poseQ[b*4+0], poseQ[b*4+1], poseQ[b*4+2], poseQ[b*4+3]);
        float d = q.x*refQ.x + q.y*refQ.y + q.z*refQ.z + q.w*refQ.w;
        if (d < 0.0f) q = -q;
        qSum += q * w;
    }
    float4 qBlend = q_norm4(qSum);

    // LBS on CoR
    float3 corRuntime = (float3)(0.0f, 0.0f, 0.0f);
    for (int k = 0; k < n; ++k) {
        int b = boneIds[idx*4 + k];
        float w = boneWts[idx*4 + k];
        if (w < 1e-9f) continue;

        float4 restQb = (float4)(
            restQ[b*4+0], restQ[b*4+1], restQ[b*4+2], restQ[b*4+3]);
        float4 poseQb = (float4)(
            poseQ[b*4+0], poseQ[b*4+1], poseQ[b*4+2], poseQ[b*4+3]);
        float4 cQ = q_mul(poseQb, restQb);

        float3 restTb = (float3)(restT[b*3+0], restT[b*3+1], restT[b*3+2]);
        float3 poseTb = (float3)(poseT[b*3+0], poseT[b*3+1], poseT[b*3+2]);

        float3 rotBT = q_rotate(poseQb, restTb);
        float3 cT = poseTb + rotBT;

        float3 rotCor = q_rotate(cQ, cor);
        float3 cCor = rotCor + cT;
        corRuntime += cCor * w;
    }

    float3 corRot = q_rotate(qBlend, cor);
    float3 tP = corRuntime - corRot;

    float3 pRot = q_rotate(qBlend, p);
    float3 pFinal = pRot + tP;

    float3 result = pFinal * envelope + p * (1.0f - envelope);

    outPos[idx*3+0] = result.x;
    outPos[idx*3+1] = result.y;
    outPos[idx*3+2] = result.z;
}