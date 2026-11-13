#include "common.h"

template<typename T>
kernel void kernel_soft_max(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint sgptg[[threads_per_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float * psrc0 =                (device const float *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const     T * pmask = src1 != src0 ? (device const T *    ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float * psrc2 = src2 != src0 ? (device const float *) (src2)                                                 : nullptr;
    device       float * pdst  =                (device       float *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    // ALiBi
    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float lmax = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        lmax = MAX(lmax, psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f));
    }

    // find the max value in the block
    float max_val = simd_max(lmax);
    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float lsum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        const float exp_psrc0 = exp((psrc0[i00]*args.scale + (pmask ? slope*pmask[i00] : 0.0f)) - max_val);
        lsum += exp_psrc0;
        pdst[i00] = exp_psrc0;
    }

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += tptg.x) {
        pdst[i00] *= inv_sum;
    }
}

template<typename T>
kernel void kernel_soft_max_4(
        constant ggml_metal_kargs_soft_max & args,
        device const  char * src0,
        device const  char * src1,
        device const  char * src2,
        device        char * dst,
        threadgroup  float * buf [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint  sgitg[[simdgroup_index_in_threadgroup]],
        uint  tiisg[[thread_index_in_simdgroup]],
        uint sgptg[[threads_per_simdgroup]],
        uint3  tptg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i01 = tgpig.x;

    const int32_t i13 = i03%args.ne13;
    const int32_t i12 = i02%args.ne12;
    const int32_t i11 = i01;

    device const float4 * psrc4 =                (device const float4 *) (src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    device const      T * pmask = src1 != src0 ? (device const T *     ) (src1 + i11*args.nb11 + i12*args.nb12 + i13*args.nb13) : nullptr;
    device const float *  psrc2 = src2 != src0 ? (device const float * ) (src2)                                                 : nullptr;
    device       float4 * pdst4 =                (device       float4 *) (dst  + i01*args.nb1  + i02*args.nb2  + i03*args.nb3);

    float slope = 1.0f;

    if (args.max_bias > 0.0f) {
        const int32_t h = i02;

        const float base = h < args.n_head_log2 ? args.m0 : args.m1;
        const int   exp  = h < args.n_head_log2 ? h + 1 : 2*(h - args.n_head_log2) + 1;

        slope = pow(base, exp);
    }

    // parallel max
    float4 lmax4 = psrc2 ? psrc2[i02] : -INFINITY;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        lmax4 = fmax(lmax4, psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f)));
    }

    const float lmax = MAX(MAX(lmax4[0], lmax4[1]), MAX(lmax4[2], lmax4[3]));

    float max_val = simd_max(lmax);
    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = -INFINITY;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = max_val;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        max_val = buf[tiisg];
        max_val = simd_max(max_val);
    }

    // parallel sum
    float4 lsum4 = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        const float4 exp_psrc4 = exp((psrc4[i00]*args.scale + (float4)((pmask ? slope*pmask[i00] : 0.0f))) - max_val);
        lsum4 += exp_psrc4;
        pdst4[i00] = exp_psrc4;
    }

    const float lsum = lsum4[0] + lsum4[1] + lsum4[2] + lsum4[3];

    // This barrier fixes a failing test
    // ref: https://github.com/ggml-org/ggml/pull/621#discussion_r1425156335
    threadgroup_barrier(mem_flags::mem_none);

    float sum = simd_sum(lsum);

    if (tptg.x > sgptg) {
        if (sgitg == 0) {
            buf[tiisg] = 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (tiisg == 0) {
            buf[sgitg] = sum;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        sum = buf[tiisg];
        sum = simd_sum(sum);
    }

    if (psrc2) {
        sum += exp(psrc2[i02] - max_val);
    }

    const float inv_sum = 1.0f/sum;

    for (int i00 = tpitg.x; i00 < args.ne00/4; i00 += tptg.x) {
        pdst4[i00] *= inv_sum;
    }
}

typedef decltype(kernel_soft_max<float>)    kernel_soft_max_t;
typedef decltype(kernel_soft_max_4<float4>) kernel_soft_max_4_t;

template [[host_name("kernel_soft_max_f16")]]   kernel kernel_soft_max_t   kernel_soft_max<half>;
template [[host_name("kernel_soft_max_f32")]]   kernel kernel_soft_max_t   kernel_soft_max<float>;
template [[host_name("kernel_soft_max_f16_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<half4>;
template [[host_name("kernel_soft_max_f32_4")]] kernel kernel_soft_max_4_t kernel_soft_max_4<float4>;

kernel void kernel_tosh_topk_moe_f32(
        constant ggml_metal_kargs_tosh_topk_moe & args,
        device const char * src_logits,
        device const char * src_bias,
        device       char * dst_weights,
        device       char * dst_ids,
        threadgroup float * sf [[threadgroup(0)]],
        threadgroup int   * si [[threadgroup(1)]],
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint  tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        uint3 ntgg [[threads_per_threadgroup]]) {
    const uint ntg = ntgg.x;
    const int row = tgpig.x;
    if (row >= args.nrows) {
        return;
    }

    const int ne = args.n_expert;
    const int nu = args.n_expert_used;

    device const float   * logits  = (device const float   *) src_logits  + (size_t) ne*row;
    device const float   * bias    = (device const float   *) src_bias;
    device       float   * weights = (device       float   *) dst_weights + (size_t) nu*row;
    device       int32_t * ids     = (device       int32_t *) dst_ids     + (size_t) ne*row;

    threadgroup float * wt  = sf;          // value that becomes the weight
    threadgroup float * sel = sf + ne;     // value the top-k selects on (wt + bias)
    threadgroup float * rv  = sf + 2*ne;   // reduction scratch, ntg entries

    for (int i = tiitg; i < ne; i += ntg) {
        wt[i] = logits[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (args.mode == 0) {
        float lmax = -INFINITY;
        for (int i = tiitg; i < ne; i += ntg) {
            lmax = max(lmax, wt[i]);
        }
        rv[tiitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] = max(rv[tiitg], rv[tiitg + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float vmax = rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = exp(wt[i] - vmax);
            wt[i] = v;
            lsum += v;
        }
        rv[tiitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] += rv[tiitg + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = 1.0f/rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < ne; i += ntg) {
            wt[i] *= inv;
        }
    } else if (args.mode == 1) {
        for (int i = tiitg; i < ne; i += ntg) {
            wt[i] = 1.0f/(1.0f + exp(-wt[i]));
        }
    } else if (args.mode == 2) {
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = wt[i];
            wt[i] = sqrt(v > 20.0f ? v : log(1.0f + exp(v)));
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // a NaN never compares greater, so the argmax below would return the same
    // expert every round; -FLT_MAX still loses to every real score
    for (int i = tiitg; i < ne; i += ntg) {
        float v = wt[i];
        if (v != v) {
            v = -FLT_MAX;
            wt[i] = v;
        }
        sel[i] = args.has_bias ? v + bias[i] : v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (int k = 0; k < nu; ++k) {
        float bv = -INFINITY;
        int   bi = ne;
        for (int i = tiitg; i < ne; i += ntg) {
            const float v = sel[i];
            if (v > bv || (v == bv && i < bi)) {
                bv = v;
                bi = i;
            }
        }
        rv[tiitg] = bv;
        si[tiitg] = bi;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                const float ov = rv[tiitg + s];
                const int   oi = si[tiitg + s];
                if (ov > rv[tiitg] || (ov == rv[tiitg] && oi < si[tiitg])) {
                    rv[tiitg] = ov;
                    si[tiitg] = oi;
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        if (tiitg == 0) {
            const int e = si[0];
            ids[k]     = e;
            weights[k] = wt[e];
            sel[e]     = -INFINITY;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    // the same arithmetic as sum_rows (one lane per element, float4 when it divides), clamp and div,
    // so the fused router gives the separate ops' weights bit for bit; the host keeps nu within a simdgroup
    if (args.with_norm != 0) {
        if (sgitg == 0) {
            float s;
            if (nu % 4 == 0) {
                const int i = 4*tiisg;
                float4 v = tiisg < nu/4 ? float4(weights[i], weights[i + 1], weights[i + 2], weights[i + 3]) : float4(0.0f);
                v = simd_sum(v);
                s = sum(v);
            } else {
                s = simd_sum(tiisg < nu ? weights[tiisg] : 0.0f);
            }
            if (tiisg == 0) {
                rv[0] = s;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        const float s = clamp(rv[0], args.clamp, INFINITY);
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] = weights[i] / s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    if (args.delayed_softmax != 0) {
        float lmax = -INFINITY;
        for (int i = tiitg; i < nu; i += ntg) {
            lmax = max(lmax, weights[i]);
        }
        rv[tiitg] = lmax;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] = max(rv[tiitg], rv[tiitg + s]);
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float vmax = rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);

        float lsum = 0.0f;
        for (int i = tiitg; i < nu; i += ntg) {
            const float v = exp(weights[i] - vmax);
            weights[i] = v;
            lsum += v;
        }
        rv[tiitg] = lsum;
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
        for (uint s = ntg/2; s > 0; s >>= 1) {
            if (tiitg < s) {
                rv[tiitg] += rv[tiitg + s];
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        const float inv = 1.0f/rv[0];
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] *= inv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device);
    }

    if (args.scale != 1.0f) {
        for (int i = tiitg; i < nu; i += ntg) {
            weights[i] *= args.scale;
        }
    }
}

[[host_name("kernel_soft_max_back")]]
kernel void kernel_soft_max_back(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float * dy = (device const float *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float * y  = (device const float *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float * dx = (device       float *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        sum += dy[i00] * y[i00];
    }

    sum = simd_sum(sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = shmem_f32[tiisg];
    sum = simd_sum(sum);

    const float scale = args.scale;

    for (int i00 = tpitg.x; i00 < args.ne00; i00 += ntg.x) {
        dx[i00] = (dy[i00] - sum) * y[i00] * scale;
    }
}

[[host_name("kernel_soft_max_back_4")]]
kernel void kernel_soft_max_back_4(
        constant ggml_metal_kargs_soft_max_back & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float * shmem_f32 [[threadgroup(0)]],
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    if (sgitg == 0) {
        shmem_f32[tiisg] = 0.0f;
    }

    const int i01 = tgpig.x;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;

    device const float4 * dy = (device const float4 *) (src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device const float4 * y  = (device const float4 *) (src1 + i03*args.nb13 + i02*args.nb12 + i01*args.nb11);
    device       float4 * dx = (device       float4 *) (dst  + i03*args.nb3  + i02*args.nb2  + i01*args.nb1);

    float sum = 0.0f;
    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        sum += dot(dy[i00], y[i00]);
    }

    sum = simd_sum(sum);

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiisg == 0) {
        shmem_f32[sgitg] = sum;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    sum = shmem_f32[tiisg];
    sum = simd_sum(sum);

    const float scale = args.scale;
    const float4 sum4 = float4(sum);

    for (int i00 = tpitg.x; i00 < args.ne00_4; i00 += ntg.x) {
        const float4 dy4 = dy[i00];
        const float4 y4  = y[i00];
        dx[i00] = (dy4 - sum4) * y4 * scale;
    }
}


// cross-entropy per row of logits against one label index, one threadgroup per row
static inline float ce_rows_reduce_max(float v, threadgroup float * shmem, ushort sgitg, ushort tiisg, ushort nsg) {
    v = simd_max(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = tiisg < nsg ? shmem[tiisg] : -INFINITY;
    return simd_max(v);
}

static inline float ce_rows_reduce_sum(float v, threadgroup float * shmem, ushort sgitg, ushort tiisg, ushort nsg) {
    v = simd_sum(v);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiisg == 0) {
        shmem[sgitg] = v;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    v = tiisg < nsg ? shmem[tiisg] : 0.0f;
    return simd_sum(v);
}

kernel void kernel_cross_entropy_rows_f32(
        constant ggml_metal_kargs_cross_entropy_rows & args,
        device const char * logits,
        device const char * labels,
        device       char * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const ushort nsg = ntg/32;
    device const float * x = (device const float *) (logits + tgpig*args.nb01);

    float m = -INFINITY;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        m = max(m, x[i]);
    }
    m = ce_rows_reduce_max(m, shmem, sgitg, tiisg, nsg);

    float s = 0.0f;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        s += exp(x[i] - m);
    }
    s = ce_rows_reduce_sum(s, shmem, sgitg, tiisg, nsg);

    if (tpitg == 0) {
        const int32_t label = ((device const int32_t *) labels)[tgpig];
        ((device float *) dst)[tgpig] = log(s) + m - x[label];
    }
}

kernel void kernel_cross_entropy_rows_back_f32(
        constant ggml_metal_kargs_cross_entropy_rows & args,
        device const char * grad,
        device const char * logits,
        device const char * labels,
        device       char * dst,
        threadgroup float * shmem [[threadgroup(0)]],
        uint   tgpig[[threadgroup_position_in_grid]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const ushort nsg = ntg/32;
    device const float * x = (device const float *) (logits + tgpig*args.nb01);
    device       float * d = (device       float *) (dst    + tgpig*args.nb1);

    float m = -INFINITY;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        m = max(m, x[i]);
    }
    m = ce_rows_reduce_max(m, shmem, sgitg, tiisg, nsg);

    float s = 0.0f;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        s += exp(x[i] - m);
    }
    s = ce_rows_reduce_sum(s, shmem, sgitg, tiisg, nsg);

    // d/dx of logsumexp(x) - x[label], times the row's incoming gradient: g*(softmax(x) - onehot(label))
    const float   g     = ((device const float   *) grad)[tgpig];
    const int32_t label = ((device const int32_t *) labels)[tgpig];
    const float   scale = g/s;
    for (int i = tpitg; i < args.ne00; i += ntg) {
        d[i] = exp(x[i] - m)*scale - (i == label ? g : 0.0f);
    }
}

kernel void kernel_count_equal_masked(
        constant ggml_metal_kargs_count_equal_masked & args,
        device const char * a,
        device const char * b,
        device const char * mask,
        device       char * dst,
        threadgroup int * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const ushort nsg = ntg/32;
    int count = 0;
    for (int i = tpitg; i < args.n; i += ntg) {
        count += ((device const float *) mask)[i] > 0.5f &&
                 ((device const int32_t *) a)[i] == ((device const int32_t *) b)[i];
    }
    count = simd_sum(count);
    if (tiisg == 0) {
        shmem[sgitg] = count;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (sgitg == 0) {
        count = tiisg < nsg ? shmem[tiisg] : 0;
        count = simd_sum(count);
        if (tiisg == 0) {
            ((device int64_t *) dst)[0] = count;
        }
    }
}


// flash attention backward in three passes (STATS, DQ, DKV) that recompute the softmax per row;
// one simdgroup per row with the head dimension split over its lanes
#define FA_BACK_NSG 8

static inline float fa_back_mask(constant ggml_metal_kargs_flash_attn_ext_back & args, device const char * mask,
                                 int iq1, int iq2, int iq3, int ic) {
    if (!args.has_mask) {
        return 0.0f;
    }
    device const half * mp = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    return (float) mp[ic];
}

template<int D, typename TK>
kernel void kernel_flash_attn_ext_back_stats(
        constant ggml_metal_kargs_flash_attn_ext_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * o,
        device const char * grad_o,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int NE = D/32;
    const int iq1 = tgpig.x*FA_BACK_NSG + sgitg;
    const int iq2 = tgpig.y;
    const int iq3 = tgpig.z;
    if (iq1 >= args.ne01) {
        return;
    }
    const int ik2 = iq2/(args.ne02/args.ne12);

    device const float * qr = (device const float *) (q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
    float qv[NE];
    for (int e = 0; e < NE; ++e) {
        qv[e] = qr[tiisg + 32*e];
    }

    float M = -INFINITY;
    float S = 0.0f;
    for (int ic = 0; ic < args.ne11; ++ic) {
        const float mv = fa_back_mask(args, mask, iq1, iq2, iq3, ic);
        if (mv == -INFINITY) {
            continue;
        }
        device const TK * kr = (device const TK *) (k + ic*args.nb11 + ik2*args.nb12 + iq3*args.nb13);
        float s = 0.0f;
        for (int e = 0; e < NE; ++e) {
            s += qv[e]*(float) kr[tiisg + 32*e];
        }
        s = simd_sum(s)*args.scale + mv;
        if (s > M) {
            S = S*exp(M - s) + 1.0f;
            M = s;
        } else {
            S += exp(s - M);
        }
    }

    device const float * orow = (device const float *) (o      + iq2*args.nbo1 + iq1*args.nbo2 + iq3*args.nbo3);
    device const float * grow = (device const float *) (grad_o + iq2*args.nbg1 + iq1*args.nbg2 + iq3*args.nbg3);
    float d = 0.0f;
    for (int e = 0; e < NE; ++e) {
        d += grow[tiisg + 32*e]*orow[tiisg + 32*e];
    }
    d = simd_sum(d);

    if (tiisg == 0) {
        device float * st = (device float *) (dst + iq1*args.nb1 + iq2*args.nb2 + iq3*args.nb3);
        st[0] = S == 0.0f ? -INFINITY : M + log(S);
        st[1] = d;
    }
}

template<int D, typename TK>
kernel void kernel_flash_attn_ext_back_dq(
        constant ggml_metal_kargs_flash_attn_ext_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * grad_o,
        device const char * stats,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int NE = D/32;
    const int iq1 = tgpig.x*FA_BACK_NSG + sgitg;
    const int iq2 = tgpig.y;
    const int iq3 = tgpig.z;
    if (iq1 >= args.ne01) {
        return;
    }
    const int ik2 = iq2/(args.ne02/args.ne12);

    device const float * qr = (device const float *) (q      + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);
    device const float * gr = (device const float *) (grad_o + iq2*args.nbg1 + iq1*args.nbg2 + iq3*args.nbg3);
    device const float * st = (device const float *) (stats  + iq1*args.nbs1 + iq2*args.nbs2 + iq3*args.nbs3);
    const float lse = st[0];
    const float dd  = st[1];

    float qv[NE], gv[NE], dq[NE];
    for (int e = 0; e < NE; ++e) {
        qv[e] = qr[tiisg + 32*e];
        gv[e] = gr[tiisg + 32*e];
        dq[e] = 0.0f;
    }

    for (int ic = 0; ic < args.ne11; ++ic) {
        const float mv = fa_back_mask(args, mask, iq1, iq2, iq3, ic);
        if (mv == -INFINITY) {
            continue;
        }
        device const TK * kr = (device const TK *) (k + ic*args.nb11 + ik2*args.nb12 + iq3*args.nb13);
        device const TK * vr = (device const TK *) (v + ic*args.nb21 + ik2*args.nb22 + iq3*args.nb23);
        float s  = 0.0f;
        float dp = 0.0f;
        float kv[NE];
        for (int e = 0; e < NE; ++e) {
            kv[e] = (float) kr[tiisg + 32*e];
            s  += qv[e]*kv[e];
            dp += gv[e]*(float) vr[tiisg + 32*e];
        }
        s  = simd_sum(s)*args.scale + mv;
        dp = simd_sum(dp);
        const float p  = exp(s - lse);
        const float ds = p*(dp - dd)*args.scale;
        for (int e = 0; e < NE; ++e) {
            dq[e] += ds*kv[e];
        }
    }

    device float * out = (device float *) (dst + iq1*args.nb1 + iq2*args.nb2 + iq3*args.nb3);
    for (int e = 0; e < NE; ++e) {
        out[tiisg + 32*e] = dq[e];
    }
}

template<int D, typename TK>
kernel void kernel_flash_attn_ext_back_dkv(
        constant ggml_metal_kargs_flash_attn_ext_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * mask,
        device const char * grad_o,
        device const char * stats,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constexpr int NE = D/32;
    const int ik1 = tgpig.x*FA_BACK_NSG + sgitg;
    const int ik2 = tgpig.y;
    const int ik3 = tgpig.z;
    if (ik1 >= args.ne11) {
        return;
    }
    const int rk2 = args.ne02/args.ne12;

    device const TK * kr = (device const TK *) (k + ik1*args.nb11 + ik2*args.nb12 + ik3*args.nb13);
    device const TK * vr = (device const TK *) (v + ik1*args.nb21 + ik2*args.nb22 + ik3*args.nb23);
    float kv[NE], vv[NE], dk[NE], dv[NE];
    for (int e = 0; e < NE; ++e) {
        kv[e] = (float) kr[tiisg + 32*e];
        vv[e] = (float) vr[tiisg + 32*e];
        dk[e] = 0.0f;
        dv[e] = 0.0f;
    }

    // every query head of this key/value head, every query row
    for (int iq2 = ik2*rk2; iq2 < (ik2 + 1)*rk2; ++iq2) {
        for (int iq1 = 0; iq1 < args.ne01; ++iq1) {
            const float mv = fa_back_mask(args, mask, iq1, iq2, ik3, ik1);
            if (mv == -INFINITY) {
                continue;
            }
            device const float * qr = (device const float *) (q      + iq1*args.nb01 + iq2*args.nb02 + ik3*args.nb03);
            device const float * gr = (device const float *) (grad_o + iq2*args.nbg1 + iq1*args.nbg2 + ik3*args.nbg3);
            device const float * st = (device const float *) (stats  + iq1*args.nbs1 + iq2*args.nbs2 + ik3*args.nbs3);
            float s  = 0.0f;
            float dp = 0.0f;
            float qv[NE], gv[NE];
            for (int e = 0; e < NE; ++e) {
                qv[e] = qr[tiisg + 32*e];
                gv[e] = gr[tiisg + 32*e];
                s  += qv[e]*kv[e];
                dp += gv[e]*vv[e];
            }
            s  = simd_sum(s)*args.scale + mv;
            dp = simd_sum(dp);
            const float p  = exp(s - st[0]);
            const float ds = p*(dp - st[1])*args.scale;
            for (int e = 0; e < NE; ++e) {
                dv[e] += p*gv[e];
                dk[e] += ds*qv[e];
            }
        }
    }

    device float * dkr = (device float *) (dst + ik1*args.nb1 + ik2*args.nb2 +  ik3             *args.nb3);
    device float * dvr = (device float *) (dst + ik1*args.nb1 + ik2*args.nb2 + (ik3 + args.ne13)*args.nb3);
    for (int e = 0; e < NE; ++e) {
        dkr[tiisg + 32*e] = dk[e];
        dvr[tiisg + 32*e] = dv[e];
    }
}

typedef decltype(kernel_flash_attn_ext_back_stats<128, float>) kernel_flash_attn_ext_back_stats_t;
typedef decltype(kernel_flash_attn_ext_back_dq<128, float>)    kernel_flash_attn_ext_back_dq_t;
typedef decltype(kernel_flash_attn_ext_back_dkv<128, float>)   kernel_flash_attn_ext_back_dkv_t;

#define FA_BACK_INST(d, nm, tk) \
template [[host_name("kernel_flash_attn_ext_back_stats_d" #d "_" nm)]] kernel kernel_flash_attn_ext_back_stats_t kernel_flash_attn_ext_back_stats<d, tk>; \
template [[host_name("kernel_flash_attn_ext_back_dq_d" #d "_" nm)]]    kernel kernel_flash_attn_ext_back_dq_t    kernel_flash_attn_ext_back_dq<d, tk>; \
template [[host_name("kernel_flash_attn_ext_back_dkv_d" #d "_" nm)]]   kernel kernel_flash_attn_ext_back_dkv_t   kernel_flash_attn_ext_back_dkv<d, tk>;
FA_BACK_INST( 64, "f32", float)
FA_BACK_INST(128, "f32", float)
FA_BACK_INST(256, "f32", float)
FA_BACK_INST( 64, "f16", half)
FA_BACK_INST(128, "f16", half)
FA_BACK_INST(256, "f16", half)
#undef FA_BACK_INST
