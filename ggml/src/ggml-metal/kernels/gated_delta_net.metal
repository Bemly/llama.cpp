#include "common.h"

constant short FC_gated_delta_net_ne20 [[function_constant(FC_GATED_DELTA_NET + 0)]];
constant short FC_gated_delta_net_ne30 [[function_constant(FC_GATED_DELTA_NET + 1)]];
constant short FC_gated_delta_net_K    [[function_constant(FC_GATED_DELTA_NET + 2)]];
// read the state from the recurrent cache row of each sequence, and write the new one back there,
// instead of going through a gathered copy and a copy back
constant bool  FC_gated_delta_net_rows [[function_constant(FC_GATED_DELTA_NET + 3)]];
constant bool  FC_gated_delta_net_fold [[function_constant(FC_GATED_DELTA_NET + 4)]];
// g and b arrive before their activations: b -> sigmoid(b), g -> a[h] * softplus(g + dt[h])
constant bool  FC_gated_delta_net_raw  [[function_constant(FC_GATED_DELTA_NET + 5)]];
// q and k arrive before their l2 norm
constant bool  FC_gated_delta_net_l2q  [[function_constant(FC_GATED_DELTA_NET + 6)]];
constant bool  FC_gated_delta_net_l2k  [[function_constant(FC_GATED_DELTA_NET + 7)]];
static inline float gdn_sum32(float v) {
    v += simd_shuffle_xor(v, 16);
    v += simd_shuffle_xor(v,  8);
    v += simd_shuffle_xor(v,  4);
    v += simd_shuffle_xor(v,  2);
    v += simd_shuffle_xor(v,  1);
    return v;
}


#if 1
template<short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device const char * rows,
        device       char * sdst,
        device const char * gdt,
        device const char * ga,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B (n_seqs)
    const uint i21 = tgpig.y; // H (head)
    const uint i20 = tgpig.x*NSG + ty; // row within S_v

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    // input state layout [S_v, S_v, H, n_seqs] (s0 only): per-seq stride is H*D.
    // state is stored transposed: M[i20][is] = S[is][i20], so row i20 is contiguous
    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float * s_ptr = FC_gated_delta_net_rows ?
        (device const float *) (s + ((device const int32_t *) rows)[i23]*args.s_nb1) + i21*S_v*S_v + i20*S_v :
        (device const float *) (s) + state_in_base;

    float ls[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        ls[j] = s_ptr[is];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    // snapshot slot mapping: slot 0 = most recent state, slot s = s tokens back.
    // When n_tokens < K, only slots 0..n_tokens-1 are written; older slots are caller-owned.

    // output state base offset: after attention scores
    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    // output state per-slot size: S_v * S_v * H * n_seqs
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    // per-(seq,head) offset within a slot
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    const float g_dt = FC_gated_delta_net_raw ? ((device const float *) gdt)[i21] : 0.0f;
    const float g_a  = FC_gated_delta_net_raw ? ((device const float *) ga)[i21]  : 0.0f;

    for (short t = 0; t < args.ne22; t++) {
        float s_k = 0.0f;

        // l2 norms of this token's q and k rows, reduced over the lanes that hold them
        float k_n = 1.0f;
        float q_n = 1.0f;
        if (FC_gated_delta_net_l2k) {
            float kk = 0.0f;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const float kv = k_ptr[tx*NSG + j];
                kk += kv*kv;
            }
            k_n = 1.0f/sqrt(gdn_sum32(kk) + args.eps_k);
        }
        if (FC_gated_delta_net_l2q) {
            float qq = 0.0f;
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const float qv = q_ptr[tx*NSG + j];
                qq += qv*qv;
            }
            q_n = 1.0f/sqrt(gdn_sum32(qq) + args.eps_q);
        }

        if (G == 1) {
            float g_val = g_ptr[0];
            if (FC_gated_delta_net_raw) {
                const float x = g_val + g_dt;
                g_val = g_a*select(log(1 + exp(x)), x, x > 20);
            }
            const float g_exp = exp(g_val);

            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= g_exp;

                s_k += ls[j]*k_ptr[is];
            }
        } else {
            // KDA
            FOR_UNROLL (short j = 0; j < NSG; j++) {
                const short is = tx*NSG + j;
                ls[j] *= exp(g_ptr[is]);

                s_k += ls[j]*k_ptr[is];
            }
        }

        s_k = gdn_sum32(s_k)*k_n;

        const float b_val = FC_gated_delta_net_raw ? 1 / (1 + exp(-b_ptr[0])) : b_ptr[0];
        const float d = (v_ptr[i20] - s_k)*b_val;
        const float d_k = d*k_n;

        float y = 0.0f;

        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            ls[j] += k_ptr[is]*d_k;

            y += ls[j]*q_ptr[is];
        }

        y = gdn_sum32(y)*q_n;

        if (tx == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float * dst_state = (device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + state_out_base;
                FOR_UNROLL (short j = 0; j < NSG; j++) {
                    const short is = tx*NSG + j;
                    dst_state[is] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float * dst_state = FC_gated_delta_net_fold ?
            (device float *) (sdst + i23*args.sd_nb1) + i21*S_v*S_v + i20*S_v :
            (device float *) (dst) + attn_size + state_out_base;
        FOR_UNROLL (short j = 0; j < NSG; j++) {
            const short is = tx*NSG + j;
            dst_state[is] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

// LPR lanes share one row of the transposed state, EPL = S_v/LPR values each, so the two dot
// products per token reduce over log2(LPR) lanes instead of 32 and every lane reads its slice of
// q and k as whole vectors. A threadgroup of 128 threads covers 128/LPR rows.
template<short LPR>
static inline float gdn_sum_lpr(float v) {
    FOR_UNROLL (short o = LPR/2; o > 0; o /= 2) {
        v += simd_shuffle_xor(v, o);
    }
    return v;
}

template<short LPR>
kernel void kernel_gated_delta_net_lpr(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device const char * rows,
        device       char * sdst,
        device const char * gdt,
        device const char * ga,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30
#define K   FC_gated_delta_net_K
    constexpr short EPL = 128/LPR; // S_v is 128 on this path
    constexpr short RPT = 128/LPR;

    const short sub = tiitg % LPR;
    const uint i23 = tgpig.z;
    const uint i21 = tgpig.y;
    const uint i20 = tgpig.x*RPT + tiitg/LPR;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    const uint state_in_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;
    device const float4 * s_ptr = (device const float4 *) (FC_gated_delta_net_rows ?
        (device const float *) (s + ((device const int32_t *) rows)[i23]*args.s_nb1) + i21*S_v*S_v + i20*S_v :
        (device const float *) (s) + state_in_base) + sub*(EPL/4);

    float4 ls[EPL/4];
    FOR_UNROLL (short j = 0; j < EPL/4; j++) {
        ls[j] = s_ptr[j];
    }

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    const uint attn_size = args.ne22 * args.ne21 * S_v * args.ne23;
    const uint state_size_per_snap = S_v * S_v * args.ne21 * args.ne23;
    const uint state_out_base = (i23*args.ne21 + i21)*S_v*S_v + i20*S_v;

    const float g_dt = FC_gated_delta_net_raw ? ((device const float *) gdt)[i21] : 0.0f;
    const float g_a  = FC_gated_delta_net_raw ? ((device const float *) ga)[i21]  : 0.0f;

    for (short t = 0; t < args.ne22; t++) {
        device const float4 * k4 = (device const float4 *) k_ptr + sub*(EPL/4);
        device const float4 * q4 = (device const float4 *) q_ptr + sub*(EPL/4);

        float4 kv[EPL/4];
        float4 qv[EPL/4];
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            kv[j] = k4[j];
            qv[j] = q4[j];
        }

        float k_n = 1.0f;
        float q_n = 1.0f;
        if (FC_gated_delta_net_l2k) {
            float kk = 0.0f;
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                kk += dot(kv[j], kv[j]);
            }
            k_n = 1.0f/sqrt(gdn_sum_lpr<LPR>(kk) + args.eps_k);
        }
        if (FC_gated_delta_net_l2q) {
            float qq = 0.0f;
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                qq += dot(qv[j], qv[j]);
            }
            q_n = 1.0f/sqrt(gdn_sum_lpr<LPR>(qq) + args.eps_q);
        }

        float s_k = 0.0f;
        if (G == 1) {
            float g_val = g_ptr[0];
            if (FC_gated_delta_net_raw) {
                const float x = g_val + g_dt;
                g_val = g_a*select(log(1 + exp(x)), x, x > 20);
            }
            const float g_exp = exp(g_val);
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                ls[j] *= g_exp;
                s_k += dot(ls[j], kv[j]);
            }
        } else {
            device const float4 * g4 = (device const float4 *) g_ptr + sub*(EPL/4);
            FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                ls[j] *= exp(g4[j]);
                s_k += dot(ls[j], kv[j]);
            }
        }

        s_k = gdn_sum_lpr<LPR>(s_k)*k_n;

        const float b_val = FC_gated_delta_net_raw ? 1 / (1 + exp(-b_ptr[0])) : b_ptr[0];
        const float d_k = (v_ptr[i20] - s_k)*b_val*k_n;

        float y = 0.0f;
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            ls[j] += kv[j]*d_k;
            y += dot(ls[j], qv[j]);
        }

        y = gdn_sum_lpr<LPR>(y)*q_n;

        if (sub == 0) {
            dst_attn[t*args.ne21*S_v] = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        if (K > 1) {
            const int target_slot = (int)args.ne22 - 1 - (int)t;
            if (target_slot >= 0 && target_slot < (int)K) {
                device float4 * ds4 = (device float4 *) ((device float *) (dst) + attn_size + (uint)target_slot * state_size_per_snap + state_out_base) + sub*(EPL/4);
                FOR_UNROLL (short j = 0; j < EPL/4; j++) {
                    ds4[j] = ls[j];
                }
            }
        }
    }

    if (K == 1) {
        device float4 * ds4 = (device float4 *) (FC_gated_delta_net_fold ?
            (device float *) (sdst + i23*args.sd_nb1) + i21*S_v*S_v + i20*S_v :
            (device float *) (dst) + attn_size + state_out_base) + sub*(EPL/4);
        FOR_UNROLL (short j = 0; j < EPL/4; j++) {
            ds4[j] = ls[j];
        }
    }

#undef S_v
#undef G
#undef K
}

typedef decltype(kernel_gated_delta_net_lpr<8>) kernel_gated_delta_net_lpr_t;

template [[host_name("kernel_gated_delta_net_f32_lpr4")]]  kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<4>;
template [[host_name("kernel_gated_delta_net_f32_lpr8")]]  kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<8>;
template [[host_name("kernel_gated_delta_net_f32_lpr16")]] kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<16>;
template [[host_name("kernel_gated_delta_net_f32_lpr32")]] kernel kernel_gated_delta_net_lpr_t kernel_gated_delta_net_lpr<32>;

typedef decltype(kernel_gated_delta_net_impl<4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<4>;

#else
// a simplified version of the above
// no performance improvement, so keep the above version for now

template<typename T, short NSG>
kernel void kernel_gated_delta_net_impl(
        constant ggml_metal_kargs_gated_delta_net & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * s,
        device       char * dst,
        device       char * dst_fuse,
        uint3 tgpig[[threadgroup_position_in_grid]],
        uint3 tpitg[[thread_position_in_threadgroup]],
        uint3   ntg[[threads_per_threadgroup]])  {
#define S_v FC_gated_delta_net_ne20
#define G   FC_gated_delta_net_ne30

    const uint tx = tpitg.x;
    const uint ty = tpitg.y;

    const uint i23 = tgpig.z; // B
    const uint i21 = tgpig.y; // H
    const uint i20 = tgpig.x*NSG + ty;

    const uint i01 = i21 % args.ne01;
    const uint i11 = i21 % args.ne11;

    const float scale = 1.0f / sqrt((float)S_v);

    device const float * s_ptr = (device const float *) (s) + (i23*args.ne21 + i21)*S_v*S_v + i20;

    float lsf[NSG];

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        lsf[j] = s_ptr[is*S_v];
    }

    thread T * ls = (thread T *) (lsf);

    device float * dst_attn = (device float *) (dst) + (i23*args.ne22*args.ne21 + i21)*S_v + i20;

    device const float * q_ptr = (device const float *) (q + i23*args.nb03 + i01*args.nb01);
    device const float * k_ptr = (device const float *) (k + i23*args.nb13 + i11*args.nb11);
    device const float * v_ptr = (device const float *) (v + i23*args.nb23 + i21*args.nb21);

    device const float * b_ptr  = (device const float *) (b) + (i23*args.ne22*args.ne21 + i21);
    device const float * g_ptr  = (device const float *) (g) + (i23*args.ne22*args.ne21 + i21)*G;

    for (short t = 0; t < args.ne22; t++) {
        device const T * qt_ptr = (device const T *) (q_ptr);
        device const T * kt_ptr = (device const T *) (k_ptr);
        device const T * gt_ptr = (device const T *) (g_ptr);

        if (G == 1) {
            *ls *= exp(g_ptr[0]);
        } else {
            // KDA
            *ls *= exp(gt_ptr[tx]);
        }

        const float s_k = simd_sum(dot(*ls, kt_ptr[tx]));

        const float d = (v_ptr[i20] - s_k)*b_ptr[0];

        *ls += kt_ptr[tx]*d;

        const float y = simd_sum(dot(*ls, qt_ptr[tx]));

        if (tx == 0) {
            *dst_attn = y*scale;
        }

        q_ptr += args.ns02;
        k_ptr += args.ns12;
        v_ptr += args.ns22;

        b_ptr += args.ne21;
        g_ptr += args.ne21*G;

        dst_attn += args.ne21*S_v;
    }

    // when fused with the cache cpy, write the snapshots straight into the cache buffer using
    // the slot stride; otherwise append them after the attn scores (nb_out == 0)
    const bool fused = args.nb_out > 0;
    const device float * state_out = fused ? (device float *)dst_fuse : (device float *)dst + args.ne23*args.ne22*args.ne21*S_v;
    const uint slot_stride = fused ? (uint)args.nb_out : S_v*S_v;

    device float * dst_state  = (device float *)state_out + (i23*args.ne21 + i21)*slot_stride + i20;
    device T     * dstt_state = (device T     *) (dst_state);

    FOR_UNROLL (short j = 0; j < NSG; j++) {
        const short is = tx*NSG + j;
        dst_state[is*S_v] = lsf[j];
    }

#undef S_v
#undef G
}

typedef decltype(kernel_gated_delta_net_impl<float4, 4>) kernel_gated_delta_net_t;

template [[host_name("kernel_gated_delta_net_f32_1")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float,  1>;
template [[host_name("kernel_gated_delta_net_f32_2")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float2, 2>;
template [[host_name("kernel_gated_delta_net_f32_4")]] kernel kernel_gated_delta_net_t kernel_gated_delta_net_impl<float4, 4>;
#endif

// Backward of the gated delta rule, chunked as in flash-linear-attention (chunk_gated_delta_rule_bwd): chunks of 64
// tokens, the WY form of each chunk, the state at every chunk start, then the gradients from matrix products. Head
// size 128, f32 throughout. The state h of a chunk is S[i][j] (i over k, j over v); the cache layout is its transpose.

#define GDN_BT  64
#define GDN_S   128
#define GDN_P32 33  // a block of 32 columns, padded
#define GDN_P64 65
#define GDN_P16 132 // a row block of 16 x 128, padded

struct gdn_back_scratch {
    device float * gc; // [B*H][Tp]      cumulative log gate inside the chunk
    device float * dg; // [2][B*H][Tp]   d gc from the two halves of k
    device float * ai; // [B*H][NT][64*64] inverse of I + L
    device float * w;  // [B*H][Tp][S]   w, then -dw
    device float * u;  // [B*H][Tp][S]   u, then v_new
    device float * dv; // [B*H][Tp][S]   intra-chunk dv, then the full du
    device float * hs; // [B*H][NT][S*S] state at each chunk start
    device float * dh; // [B*H][NT][S*S] gradient of the state at each chunk end
};

static inline gdn_back_scratch gdn_back_scratch_of(constant ggml_metal_kargs_gated_delta_net_back & a, device char * dst) {
    const ulong bh = (ulong) a.B*a.H;
    const ulong tp = (ulong) a.NT*GDN_BT;
    gdn_back_scratch s;
    s.gc = (device float *) (dst + a.offs);
    s.dg = s.gc + bh*tp;
    s.ai = s.dg + 2*bh*tp;
    s.w  = s.ai + bh*a.NT*GDN_BT*GDN_BT;
    s.u  = s.w  + bh*tp*GDN_S;
    s.dv = s.u  + bh*tp*GDN_S;
    s.hs = s.dv + bh*tp*GDN_S;
    s.dh = s.hs + bh*a.NT*GDN_S*GDN_S;
    return s;
}

// one thread loads 8 consecutive values of a 64 x 32 block (row tiitg/4)
static inline void gdn_ld8(threadgroup float * dst, device const float * src, bool ok, ushort tiitg) {
    const short r = tiitg >> 2;
    const short c = (tiitg & 3)*8;
    float4 a = 0.0f, b = 0.0f;
    if (ok) {
        a = ((device const float4 *) (src + c))[0];
        b = ((device const float4 *) (src + c))[1];
    }
    threadgroup float * d = dst + r*GDN_P32 + c;
    d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w;
    d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}

// one thread loads 8 consecutive values of a 16 x 128 block (row tiitg/16)
static inline void gdn_ld8_rows(threadgroup float * dst, device const float * src, bool ok, ushort tiitg) {
    const short r = tiitg >> 4;
    const short c = (tiitg & 15)*8;
    float4 a = 0.0f, b = 0.0f;
    if (ok) {
        a = ((device const float4 *) (src + c))[0];
        b = ((device const float4 *) (src + c))[1];
    }
    threadgroup float * d = dst + r*GDN_P16 + c;
    d[0] = a.x; d[1] = a.y; d[2] = a.z; d[3] = a.w;
    d[4] = b.x; d[5] = b.y; d[6] = b.z; d[7] = b.w;
}

// grid (NT, H, B): gc, L = strict_lower(beta_i k_i.k_j exp(gc_i - gc_j)), its inverse A = (I + L)^-1,
// w = A (beta exp(gc) k), u = A (beta v), and the intra-chunk dv = (scale k_i.q_j exp(gc_j - gc_i), i <= j) dO
kernel void kernel_gated_delta_net_back_prep(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const char * q,
        device const char * k,
        device const char * v,
        device const char * g,
        device const char * b,
        device const char * grad,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int c  = tgpig.x;
    const int h  = tgpig.y;
    const int ib = tgpig.z;
    const int hq = h % args.Hq;
    const int t0 = c*GDN_BT;
    const int nt = min(GDN_BT, args.T - t0);
    const ulong bh = (ulong) ib*args.H + h;
    const ulong tp = (ulong) args.NT*GDN_BT;

    const gdn_back_scratch sc = gdn_back_scratch_of(args, dst);

    threadgroup float * gcs = shmem;
    threadgroup float * bs  = gcs + 64;
    threadgroup float * tot = bs  + 64;
    threadgroup float * Am  = tot + 4;            // L, then A
    threadgroup float * Pm  = Am + 64*GDN_P64;
    threadgroup float * Xa  = Pm + 64*GDN_P64;
    threadgroup float * Xb  = Xa + 64*GDN_P32;

    {
        float gv = 0.0f;
        float bv = 0.0f;
        if (tiitg < nt) {
            gv = *(device const float *) (g + ib*args.nb33 + (t0 + tiitg)*args.nb32 + h*args.nb31);
            bv = *(device const float *) (b + ib*args.nb43 + (t0 + tiitg)*args.nb42 + h*args.nb41);
        }
        float ps = simd_prefix_inclusive_sum(gv);
        if (tiitg == 31) {
            tot[0] = ps;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 64) {
            ps += sgitg == 1 ? tot[0] : 0.0f;
            gcs[tiitg] = ps;
            bs[tiitg]  = bv;
            sc.gc[bh*tp + t0 + tiitg] = ps;
        }
    }

    const bool lok = (int) (tiitg >> 2) < nt;
    device const float * krow = (device const float *) (k + ib*args.nb13 + (t0 + (tiitg >> 2))*args.nb12 + hq*args.nb11);
    device const float * qrow = (device const float *) (q + ib*args.nb03 + (t0 + (tiitg >> 2))*args.nb02 + hq*args.nb01);
    device const float * vrow = (device const float *) (v + ib*args.nb23 + (t0 + (tiitg >> 2))*args.nb22 + h*args.nb21);
    device const float * orow = (device const float *) grad + (((ulong) ib*args.T + t0 + (tiitg >> 2))*args.H + h)*GDN_S;

    // k k^T and k q^T over the head dimension
    const short tr = tiitg/16;
    const short tc = tiitg%16;
    float kk[4][4];
    float kq[4][4];
    FOR_UNROLL (short a = 0; a < 4; ++a) {
        FOR_UNROLL (short e = 0; e < 4; ++e) {
            kk[a][e] = 0.0f;
            kq[a][e] = 0.0f;
        }
    }
    for (short cb = 0; cb < GDN_S/32; ++cb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gdn_ld8(Xa, krow + cb*32, lok, tiitg);
        gdn_ld8(Xb, qrow + cb*32, lok, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short x = 0; x < 32; ++x) {
            float ar[4], bk[4], bq[4];
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                ar[a] = Xa[(tr + 16*a)*GDN_P32 + x];
                bk[a] = Xa[(tc + 16*a)*GDN_P32 + x];
                bq[a] = Xb[(tc + 16*a)*GDN_P32 + x];
            }
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    kk[a][e] += ar[a]*bk[e];
                    kq[a][e] += ar[a]*bq[e];
                }
            }
        }
    }
    FOR_UNROLL (short a = 0; a < 4; ++a) {
        FOR_UNROLL (short e = 0; e < 4; ++e) {
            const short r = tr + 16*a;
            const short s = tc + 16*e;
            Am[r*GDN_P64 + s] = r >  s ? bs[r]*kk[a][e]*exp(gcs[r] - gcs[s]) : 0.0f;
            Pm[r*GDN_P64 + s] = r <= s ? args.scale*kq[a][e]*exp(gcs[s] - gcs[r]) : 0.0f;
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // (I + L)^-1 as in fla solve_tril: the four diagonal 16 x 16 blocks by forward substitution, then the blocks
    // below them from products of the blocks already known
    {
        const short d  = tiitg >> 6;
        const short j  = (tiitg >> 2) & 15;
        const short p  = tiitg & 3;
        const short o  = 16*d;
        for (short i = 1; i < 16; ++i) {
            float s = 0.0f;
            if (j < i) {
                for (short m = j + 1 + p; m < i; m += 4) {
                    s += Am[(o + i)*GDN_P64 + o + m]*Am[(o + m)*GDN_P64 + o + j];
                }
                s += p == 0 ? Am[(o + i)*GDN_P64 + o + j] : 0.0f;
            }
            s += simd_shuffle_xor(s, 1);
            s += simd_shuffle_xor(s, 2);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (j < i && p == 0) {
                Am[(o + i)*GDN_P64 + o + j] = -s;
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        if (tiitg < 64) {
            Am[tiitg*GDN_P64 + tiitg] = 1.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // block (I, J) of Am; X* are 16 x 16 temporaries in Xa/Xb
        #define GDN_AB(I, J, i, j) Am[(16*(I) + (i))*GDN_P64 + 16*(J) + (j)]
        threadgroup float * X21 = Xa;
        threadgroup float * X32 = Xa + 256;
        threadgroup float * X43 = Xa + 512;
        threadgroup float * X31 = Xa + 768;
        threadgroup float * X42 = Xa + 1024;
        threadgroup float * X41 = Xa + 1280;
        threadgroup float * Tm  = Xb;          // 3 x 256
        const short i = tiitg >> 4;
        const short jj = tiitg & 15;

        // T = L_{n+1,n} A_nn, then A_{n+1,n} = -A_{n+1,n+1} T
        FOR_UNROLL (short n = 0; n < 3; ++n) {
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(n + 1, n, i, m)*GDN_AB(n, n, m, jj);
            }
            Tm[n*256 + i*16 + jj] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short n = 0; n < 3; ++n) {
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(n + 1, n + 1, i, m)*Tm[n*256 + m*16 + jj];
            }
            (n == 0 ? X21 : n == 1 ? X32 : X43)[i*16 + jj] = -s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A_31 = -A_33 (L_31 A_11 + L_32 A_21), A_42 = -A_44 (L_42 A_22 + L_43 A_32)
        FOR_UNROLL (short n = 0; n < 2; ++n) {
            threadgroup const float * Xn = n == 0 ? X21 : X32;
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(n + 2, n, i, m)*GDN_AB(n, n, m, jj) + GDN_AB(n + 2, n + 1, i, m)*Xn[m*16 + jj];
            }
            Tm[n*256 + i*16 + jj] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short n = 0; n < 2; ++n) {
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(n + 2, n + 2, i, m)*Tm[n*256 + m*16 + jj];
            }
            (n == 0 ? X31 : X42)[i*16 + jj] = -s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // A_41 = -A_44 (L_41 A_11 + L_42 A_21 + L_43 A_31)
        {
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(3, 0, i, m)*GDN_AB(0, 0, m, jj) + GDN_AB(3, 1, i, m)*X21[m*16 + jj] + GDN_AB(3, 2, i, m)*X31[m*16 + jj];
            }
            Tm[i*16 + jj] = s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            float s = 0.0f;
            for (short m = 0; m < 16; ++m) {
                s += GDN_AB(3, 3, i, m)*Tm[m*16 + jj];
            }
            X41[i*16 + jj] = -s;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        GDN_AB(1, 0, i, jj) = X21[i*16 + jj];
        GDN_AB(2, 1, i, jj) = X32[i*16 + jj];
        GDN_AB(3, 2, i, jj) = X43[i*16 + jj];
        GDN_AB(2, 0, i, jj) = X31[i*16 + jj];
        GDN_AB(3, 1, i, jj) = X42[i*16 + jj];
        GDN_AB(3, 0, i, jj) = X41[i*16 + jj];
        #undef GDN_AB
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    {
        device float * ao = sc.ai + (bh*args.NT + c)*GDN_BT*GDN_BT;
        for (short e = tiitg; e < GDN_BT*GDN_BT; e += 256) {
            ao[e] = Am[(e/64)*GDN_P64 + e%64];
        }
    }

    // w, u and the intra-chunk dv, 32 columns at a time: rows wr and wr + 32, columns wc + 8*e
    const short wr = tiitg/8;
    const short wc = tiitg%8;
    for (short cb = 0; cb < GDN_S/32; ++cb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gdn_ld8(Xa, krow + cb*32, lok, tiitg);
        gdn_ld8(Xb, vrow + cb*32, lok, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const short r = tiitg >> 2;
            const float fk = bs[r]*exp(gcs[r]);
            FOR_UNROLL (short e = 0; e < 8; ++e) {
                Xa[r*GDN_P32 + (tiitg & 3)*8 + e] *= fk;
                Xb[r*GDN_P32 + (tiitg & 3)*8 + e] *= bs[r];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float aw[2][4] = { { 0.0f } };
        float au[2][4] = { { 0.0f } };
        for (short m = 0; m < 64; ++m) {
            const float a0 = Am[wr*GDN_P64 + m];
            const float a1 = Am[(wr + 32)*GDN_P64 + m];
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                const float xw = Xa[m*GDN_P32 + wc + 8*e];
                const float xu = Xb[m*GDN_P32 + wc + 8*e];
                aw[0][e] += a0*xw; aw[1][e] += a1*xw;
                au[0][e] += a0*xu; au[1][e] += a1*xu;
            }
        }
        FOR_UNROLL (short f = 0; f < 2; ++f) {
            const ulong row = (bh*tp + t0 + wr + 32*f)*GDN_S + cb*32 + wc;
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                sc.w[row + 8*e] = aw[f][e];
                sc.u[row + 8*e] = au[f][e];
            }
        }
    }
    for (short cb = 0; cb < GDN_S/32; ++cb) {
        threadgroup_barrier(mem_flags::mem_threadgroup);
        gdn_ld8(Xa, orow + cb*32, lok, tiitg);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float ad[2][4] = { { 0.0f } };
        for (short m = 0; m < 64; ++m) {
            const float p0 = Pm[wr*GDN_P64 + m];
            const float p1 = Pm[(wr + 32)*GDN_P64 + m];
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                const float x = Xa[m*GDN_P32 + wc + 8*e];
                ad[0][e] += p0*x; ad[1][e] += p1*x;
            }
        }
        FOR_UNROLL (short f = 0; f < 2; ++f) {
            const ulong row = (bh*tp + t0 + wr + 32*f)*GDN_S + cb*32 + wc;
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                sc.dv[row + 8*e] = ad[f][e];
            }
        }
    }
}

// grid (4, H, B), one block of 32 value columns each: the state at each chunk start, v_new = u - w h stored over u,
// h <- exp(gl) h + k^T (exp(gl - gc) v_new)
kernel void kernel_gated_delta_net_back_fwd_h(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const char * k,
        device const char * s,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const int j0 = tgpig.x*32;
    const int h  = tgpig.y;
    const int ib = tgpig.z;
    const int hq = h % args.Hq;
    const ulong bh = (ulong) ib*args.H + h;
    const ulong tp = (ulong) args.NT*GDN_BT;

    const gdn_back_scratch sc = gdn_back_scratch_of(args, dst);

    threadgroup float * Hb  = shmem;              // 128 x 33
    threadgroup float * Xb  = Hb + 128*GDN_P32;   // 64 x 33, or 16 x 132
    threadgroup float * Vd  = Xb + 64*GDN_P32;    // 64 x 33
    threadgroup float * gcs = Vd + 64*GDN_P32;

    // the state: rows ti + 32*a, columns tj + 8*e of this block; v_new: rows vr, vr + 32, columns tj + 8*e
    const short ti = tiitg/8;
    const short tj = tiitg%8;
    float hr[4][4];
    {
        device const float * s0 = (device const float *) (s + ib*args.nb53 + h*args.nb52);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                hr[a][e] = s0[(j0 + tj + 8*e)*GDN_S + ti + 32*a];
            }
        }
    }

    for (int c = 0; c < args.NT; ++c) {
        const int t0 = c*GDN_BT;
        const int nt = min(GDN_BT, args.T - t0);
        device float * hs = sc.hs + (bh*args.NT + c)*GDN_S*GDN_S;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                hs[(ti + 32*a)*GDN_S + j0 + tj + 8*e] = hr[a][e];
                Hb[(ti + 32*a)*GDN_P32 + tj + 8*e]    = hr[a][e];
            }
        }
        if (tiitg < 64) {
            gcs[tiitg] = sc.gc[bh*tp + t0 + tiitg];
        }

        float acc[2][4] = { { 0.0f } };
        for (short kb = 0; kb < GDN_S/32; ++kb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8(Xb, sc.w + (bh*tp + t0 + (tiitg >> 2))*GDN_S + kb*32, true, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short x = 0; x < 32; ++x) {
                const float w0 = Xb[ti*GDN_P32 + x];
                const float w1 = Xb[(ti + 32)*GDN_P32 + x];
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const float hv = Hb[(kb*32 + x)*GDN_P32 + tj + 8*e];
                    acc[0][e] += w0*hv;
                    acc[1][e] += w1*hv;
                }
            }
        }
        const float gl = gcs[GDN_BT - 1];
        FOR_UNROLL (short f = 0; f < 2; ++f) {
            const short r = ti + 32*f;
            const float dec = exp(gl - gcs[r]);
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                device float * up = sc.u + (bh*tp + t0 + r)*GDN_S + j0 + tj + 8*e;
                const float vn = *up - acc[f][e];
                *up = vn;
                Vd[r*GDN_P32 + tj + 8*e] = vn*dec;
            }
        }

        const float eg = exp(gl);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                hr[a][e] *= eg;
            }
        }
        device const float * krow = (device const float *) (k + ib*args.nb13 + hq*args.nb11);
        for (short rb = 0; rb < GDN_BT/16; ++rb) {
            const int r = rb*16 + (tiitg >> 4);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8_rows(Xb, (device const float *) ((device const char *) krow + (t0 + r)*args.nb12), r < nt, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short rr = 0; rr < 16; ++rr) {
                float kv[4], vv[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    kv[a] = Xb[rr*GDN_P16 + ti + 32*a];
                    vv[a] = Vd[(rb*16 + rr)*GDN_P32 + tj + 8*a];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        hr[a][e] += kv[a]*vv[e];
                    }
                }
            }
        }
    }
}

// grid (4, H, B), backward over the chunks: dh at each chunk end, du = dv + exp(gl - gc) k dh stored over dv,
// dh <- exp(gl) dh + scale (exp(gc) q)^T dO - w^T du; the gradient of the initial state at the end
kernel void kernel_gated_delta_net_back_dhu(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const char * q,
        device const char * k,
        device const char * grad,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const int j0 = tgpig.x*32;
    const int h  = tgpig.y;
    const int ib = tgpig.z;
    const int hq = h % args.Hq;
    const ulong bh = (ulong) ib*args.H + h;
    const ulong tp = (ulong) args.NT*GDN_BT;

    const gdn_back_scratch sc = gdn_back_scratch_of(args, dst);

    threadgroup float * Hb  = shmem;              // 128 x 33, then the q and w row blocks
    threadgroup float * Qb  = shmem;              // 16 x 132
    threadgroup float * Wb  = shmem + 16*GDN_P16; // 16 x 132
    threadgroup float * Kb  = Hb + 128*GDN_P32;   // 64 x 33
    threadgroup float * Dv  = Kb + 64*GDN_P32;    // 64 x 33
    threadgroup float * Do  = Dv + 64*GDN_P32;    // 64 x 33
    threadgroup float * gcs = Do + 64*GDN_P32;

    const short ti = tiitg/8;
    const short tj = tiitg%8;
    float dh[4][4];
    {
        device const float * dht = (device const float *) grad + (ulong) GDN_S*args.H*args.T*args.B + bh*GDN_S*GDN_S;
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                dh[a][e] = args.has_dht ? dht[(j0 + tj + 8*e)*GDN_S + ti + 32*a] : 0.0f;
            }
        }
    }

    device const float * kbase = (device const float *) (k + ib*args.nb13 + hq*args.nb11);
    device const float * qbase = (device const float *) (q + ib*args.nb03 + hq*args.nb01);

    for (int c = args.NT - 1; c >= 0; --c) {
        const int t0 = c*GDN_BT;
        const int nt = min(GDN_BT, args.T - t0);
        device float * dhs = sc.dh + (bh*args.NT + c)*GDN_S*GDN_S;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                dhs[(ti + 32*a)*GDN_S + j0 + tj + 8*e] = dh[a][e];
                Hb[(ti + 32*a)*GDN_P32 + tj + 8*e]     = dh[a][e];
            }
        }
        if (tiitg < 64) {
            gcs[tiitg] = sc.gc[bh*tp + t0 + tiitg];
        }

        float acc[2][4] = { { 0.0f } };
        for (short kb = 0; kb < GDN_S/32; ++kb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8(Kb, (device const float *) ((device const char *) kbase + (t0 + (tiitg >> 2))*args.nb12) + kb*32,
                    (int) (tiitg >> 2) < nt, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short x = 0; x < 32; ++x) {
                const float k0 = Kb[ti*GDN_P32 + x];
                const float k1 = Kb[(ti + 32)*GDN_P32 + x];
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const float hv = Hb[(kb*32 + x)*GDN_P32 + tj + 8*e];
                    acc[0][e] += k0*hv;
                    acc[1][e] += k1*hv;
                }
            }
        }
        const float gl = gcs[GDN_BT - 1];
        FOR_UNROLL (short f = 0; f < 2; ++f) {
            const short r = ti + 32*f;
            const float dec = exp(gl - gcs[r]);
            const float qs  = args.scale*exp(gcs[r]);
            device const float * orow = (device const float *) grad + (((ulong) ib*args.T + t0 + r)*args.H + h)*GDN_S + j0;
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                device float * dvp = sc.dv + (bh*tp + t0 + r)*GDN_S + j0 + tj + 8*e;
                const float d2 = *dvp + acc[f][e]*dec;
                *dvp = d2;
                Dv[r*GDN_P32 + tj + 8*e] = d2;
                Do[r*GDN_P32 + tj + 8*e] = r < nt ? qs*orow[tj + 8*e] : 0.0f;
            }
        }

        const float eg = exp(gl);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                dh[a][e] *= eg;
            }
        }
        for (short rb = 0; rb < GDN_BT/16; ++rb) {
            const int r = rb*16 + (tiitg >> 4);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8_rows(Qb, (device const float *) ((device const char *) qbase + (t0 + r)*args.nb02), r < nt, tiitg);
            gdn_ld8_rows(Wb, sc.w + (bh*tp + t0 + r)*GDN_S, true, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short rr = 0; rr < 16; ++rr) {
                float qv[4], wv[4], ov[4], dv[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    qv[a] = Qb[rr*GDN_P16 + ti + 32*a];
                    wv[a] = Wb[rr*GDN_P16 + ti + 32*a];
                    ov[a] = Do[(rb*16 + rr)*GDN_P32 + tj + 8*a];
                    dv[a] = Dv[(rb*16 + rr)*GDN_P32 + tj + 8*a];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        dh[a][e] += qv[a]*ov[e] - wv[a]*dv[e];
                    }
                }
            }
        }
    }

    device float * ds = (device float *) dst + 2*(ulong) GDN_S*args.Hq*args.T*args.B + (ulong) GDN_S*args.H*args.T*args.B
                      + 2*(ulong) args.H*args.T*args.B + bh*GDN_S*GDN_S;
    FOR_UNROLL (short a = 0; a < 4; ++a) {
        FOR_UNROLL (short e = 0; e < 4; ++e) {
            ds[(j0 + tj + 8*e)*GDN_S + ti + 32*a] = dh[a][e];
        }
    }
}

// grid (2, NT, Hq*B), one half of the k dimension each, the value heads of a q/k head in turn: dq, the dk of the
// state path, -dw over w, and d gc of this half (fla chunk_bwd_kernel_dqkwg); dq and dk summed over the value heads
kernel void kernel_gated_delta_net_back_dqkwg(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const char * q,
        device const char * k,
        device const char * grad,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int kb = tgpig.x;
    const int c  = tgpig.y;
    const int hq = tgpig.z % args.Hq;
    const int ib = tgpig.z / args.Hq;
    const int t0 = c*GDN_BT;
    const int nt = min(GDN_BT, args.T - t0);
    const int k0 = kb*64;
    const ulong tp = (ulong) args.NT*GDN_BT;
    const ulong nbh = (ulong) args.B*args.H;

    const gdn_back_scratch sc = gdn_back_scratch_of(args, dst);

    // over the value blocks: dO, v_new, du (64 x 33 each), h and dh (64 rows of k x 33)
    threadgroup float * Do  = shmem;
    threadgroup float * Vn  = Do + 64*GDN_P32;
    threadgroup float * D2  = Vn + 64*GDN_P32;
    threadgroup float * Hk  = D2 + 64*GDN_P32;
    threadgroup float * Dk  = Hk + 64*GDN_P32;
    // after them: q, k of this half and the masked dO v_new^T (64 x 65 each)
    threadgroup float * Qk  = shmem;
    threadgroup float * Kk  = Qk + 64*GDN_P64;
    threadgroup float * Ds  = Kk + 64*GDN_P64;
    threadgroup float * gcs = Ds + 64*GDN_P64;
    threadgroup float * red = gcs + 64;

    const short tr = tiitg/16;
    const short tc = tiitg%16;

    device const float * qbase = (device const float *) (q + ib*args.nb03 + hq*args.nb01);
    device const float * kbase = (device const float *) (k + ib*args.nb13 + hq*args.nb11);

    float qs[4][4] = { { 0.0f } };
    float ks[4][4] = { { 0.0f } };

    for (int h = hq; h < args.H; h += args.Hq) {
        const ulong bh = (ulong) ib*args.H + h;
        device const float * hs  = sc.hs + (bh*args.NT + c)*GDN_S*GDN_S + (ulong) k0*GDN_S;
        device const float * dhs = sc.dh + (bh*args.NT + c)*GDN_S*GDN_S + (ulong) k0*GDN_S;
        const short lr = tiitg >> 2;
        device const float * orow = (device const float *) grad + (((ulong) ib*args.T + t0 + lr)*args.H + h)*GDN_S;

        float ds[4][4] = { { 0.0f } };
        float dq[4][4] = { { 0.0f } };
        float dk[4][4] = { { 0.0f } };
        float dw[4][4] = { { 0.0f } };
        float dgl = 0.0f;

        for (short vb = 0; vb < GDN_S/32; ++vb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (vb == 0 && tiitg < 64) {
                gcs[tiitg] = sc.gc[bh*tp + t0 + tiitg];
            }
            gdn_ld8(Do, orow + vb*32, lr < nt, tiitg);
            gdn_ld8(Vn, sc.u  + (bh*tp + t0 + lr)*GDN_S + vb*32, true, tiitg);
            gdn_ld8(D2, sc.dv + (bh*tp + t0 + lr)*GDN_S + vb*32, true, tiitg);
            gdn_ld8(Hk, hs  + lr*GDN_S + vb*32, true, tiitg);
            gdn_ld8(Dk, dhs + lr*GDN_S + vb*32, true, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short x = 0; x < 32; ++x) {
                float o[4], n[4], d[4], ns[4], hk[4], dh[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    o[a]  = Do[(tr + 16*a)*GDN_P32 + x];
                    n[a]  = Vn[(tr + 16*a)*GDN_P32 + x];
                    d[a]  = D2[(tr + 16*a)*GDN_P32 + x];
                    ns[a] = Vn[(tc + 16*a)*GDN_P32 + x];
                    hk[a] = Hk[(tc + 16*a)*GDN_P32 + x];
                    dh[a] = Dk[(tc + 16*a)*GDN_P32 + x];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        ds[a][e] += o[a]*ns[e];
                        dq[a][e] += o[a]*hk[e];
                        dk[a][e] += n[a]*dh[e];
                        dw[a][e] += d[a]*hk[e];
                    }
                }
            }
            for (short e = tiitg; e < 64*32; e += 256) {
                dgl += Hk[(e/32)*GDN_P32 + e%32]*Dk[(e/32)*GDN_P32 + e%32];
            }
        }

        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                sc.w[(bh*tp + t0 + tr + 16*a)*GDN_S + k0 + tc + 16*e] = -dw[a][e];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            const short r  = tiitg >> 2;
            const short c0 = (tiitg & 3)*16;
            const bool ok = r < nt;
            device const float * qr = (device const float *) ((device const char *) qbase + (t0 + r)*args.nb02) + k0 + c0;
            device const float * kr = (device const float *) ((device const char *) kbase + (t0 + r)*args.nb12) + k0 + c0;
            FOR_UNROLL (short e = 0; e < 16; ++e) {
                Qk[r*GDN_P64 + c0 + e] = ok ? qr[e] : 0.0f;
                Kk[r*GDN_P64 + c0 + e] = ok ? kr[e] : 0.0f;
            }
        }
        dgl = simd_sum(dgl);
        if (tiisg == 0) {
            red[sgitg] = dgl;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        const float gl = gcs[GDN_BT - 1];
        float dg_last = 0.0f;
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            dg_last += red[i];
        }
        dg_last *= exp(gl);

        float part = 0.0f;
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            const short r = tr + 16*a;
            const float fq = exp(gcs[r])*args.scale;
            const float fk = exp(gl - gcs[r]);
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                const short s = tc + 16*e;
                dq[a][e] *= fq;
                dk[a][e] *= fk;
                part += dk[a][e]*Kk[r*GDN_P64 + s];
                Ds[r*GDN_P64 + s] = r >= s ? ds[a][e]*exp(gcs[r] - gcs[s])*args.scale : 0.0f;
            }
        }
        part = simd_sum(part);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiisg == 0) {
            red[sgitg] = part;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short i = 0; i < 8; ++i) {
            dg_last += red[i];
        }

        for (short x = 0; x < 64; ++x) {
            float dr[4], dc[4], kv[4], qv[4];
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                dr[a] = Ds[(tr + 16*a)*GDN_P64 + x];
                dc[a] = Ds[x*GDN_P64 + tr + 16*a];
                kv[a] = Kk[x*GDN_P64 + tc + 16*a];
                qv[a] = Qk[x*GDN_P64 + tc + 16*a];
            }
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    dq[a][e] += dr[a]*kv[e];
                    dk[a][e] += dc[a]*qv[e];
                }
            }
        }

        FOR_UNROLL (short a = 0; a < 4; ++a) {
            const short r = tr + 16*a;
            float dgr = 0.0f;
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                const short s = tc + 16*e;
                dgr += dq[a][e]*Qk[r*GDN_P64 + s] - dk[a][e]*Kk[r*GDN_P64 + s];
                qs[a][e] += dq[a][e];
                ks[a][e] += dk[a][e];
            }
            dgr += simd_shuffle_xor(dgr, 1);
            dgr += simd_shuffle_xor(dgr, 2);
            dgr += simd_shuffle_xor(dgr, 4);
            dgr += simd_shuffle_xor(dgr, 8);
            if (tc == 0 && r < nt) {
                sc.dg[(kb*nbh + bh)*tp + t0 + r] = dgr + (r == nt - 1 ? dg_last : 0.0f);
            }
        }
    }

    device float * dqo = (device float *) dst;
    device float * dko = dqo + (ulong) GDN_S*args.Hq*args.T*args.B;
    FOR_UNROLL (short a = 0; a < 4; ++a) {
        const short r = tr + 16*a;
        if (r < nt) {
            const ulong row = (((ulong) ib*args.T + t0 + r)*args.Hq + hq)*GDN_S + k0;
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                dqo[row + tc + 16*e] = qs[a][e];
                dko[row + tc + 16*e] = ks[a][e];
            }
        }
    }
}

// grid (NT, Hq*B), the value heads of a q/k head in turn: the backward of w = A (beta exp(gc) k), u = A (beta v)
// and A = (I + L)^-1 (fla prepare_wy_repr_bwd), then d gc summed and turned into d g by the reverse cumsum
kernel void kernel_gated_delta_net_back_wy(
        constant ggml_metal_kargs_gated_delta_net_back & args,
        device const char * k,
        device const char * v,
        device const char * b,
        device       char * dst,
        threadgroup  float * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int c  = tgpig.x;
    const int hq = tgpig.y % args.Hq;
    const int ib = tgpig.y / args.Hq;
    const int t0 = c*GDN_BT;
    const int nt = min(GDN_BT, args.T - t0);
    const ulong tp = (ulong) args.NT*GDN_BT;
    const ulong nbh = (ulong) args.B*args.H;

    const gdn_back_scratch sc = gdn_back_scratch_of(args, dst);

    threadgroup float * Ai  = shmem;              // 64 x 65
    threadgroup float * Dm  = Ai + 64*GDN_P64;    // 64 x 65
    threadgroup float * X1  = Dm + 64*GDN_P64;    // 64 x 33
    threadgroup float * X2  = X1 + 64*GDN_P32;    // 64 x 33
    threadgroup float * Tm  = X1;                 // 64 x 65 over X1 and X2
    threadgroup float * gcs = X2 + 64*GDN_P32;
    threadgroup float * bs  = gcs + 64;
    threadgroup float * fac = bs  + 64;
    threadgroup float * dgr = fac + 64;
    threadgroup float * tot = dgr + 64;

    const short wr = tiitg/8;   // 64 x 32 blocks: rows wr, wr + 32, columns wc + 8*e
    const short wc = tiitg%8;
    const short tr = tiitg/16;  // 64 x 64: rows tr + 16*a, columns tc + 16*e
    const short tc = tiitg%16;
    const short lr = tiitg >> 2;

    device const float * kbase = (device const float *) (k + ib*args.nb13 + hq*args.nb11);
    device const float * krow  = (device const float *) ((device const char *) kbase + (t0 + lr)*args.nb12);

    device float * dvo = (device float *) dst + 2*(ulong) GDN_S*args.Hq*args.T*args.B;
    device float * dgo = dvo + (ulong) GDN_S*args.H*args.T*args.B;
    device float * dbo = dgo + (ulong) args.H*args.T*args.B;

    float dka[2][16] = { { 0.0f } };

    for (int h = hq; h < args.H; h += args.Hq) {
        const ulong bh = (ulong) ib*args.H + h;

        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            device const float * ai = sc.ai + (bh*args.NT + c)*GDN_BT*GDN_BT;
            for (short e = tiitg; e < GDN_BT*GDN_BT; e += 256) {
                Ai[(e/64)*GDN_P64 + e%64] = ai[e];
            }
        }
        if (tiitg < 64) {
            const float gv = sc.gc[bh*tp + t0 + tiitg];
            const float bv = tiitg < nt ? *(device const float *) (b + ib*args.nb43 + (t0 + tiitg)*args.nb42 + h*args.nb41) : 0.0f;
            gcs[tiitg] = gv;
            bs[tiitg]  = bv;
            fac[tiitg] = bv*exp(gv);
        }

        float dA[4][4] = { { 0.0f } };
        float dbp[2] = { 0.0f, 0.0f };
        float dgp[2] = { 0.0f, 0.0f };

        // w = A (beta exp(gc) k): dA = dw (beta exp(gc) k)^T, d(beta exp(gc) k) = A^T dw
        FOR_UNROLL (short cb = 0; cb < GDN_S/32; ++cb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8(X1, krow + cb*32, lr < nt, tiitg);
            gdn_ld8(X2, sc.w + (bh*tp + t0 + lr)*GDN_S + cb*32, true, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short x = 0; x < 32; ++x) {
                float dwr[4], kr[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    dwr[a] = X2[(tr + 16*a)*GDN_P32 + x];
                    kr[a]  = X1[(tc + 16*a)*GDN_P32 + x];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        dA[a][e] += dwr[a]*kr[e];
                    }
                }
            }
            float t[2][4] = { { 0.0f } };
            for (short m = 0; m < 64; ++m) {
                const float a0 = Ai[m*GDN_P64 + wr];
                const float a1 = Ai[m*GDN_P64 + wr + 32];
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const float x = X2[m*GDN_P32 + wc + 8*e];
                    t[0][e] += a0*x;
                    t[1][e] += a1*x;
                }
            }
            FOR_UNROLL (short f = 0; f < 2; ++f) {
                const short r = wr + 32*f;
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const float kv = X1[r*GDN_P32 + wc + 8*e];
                    dka[f][cb*4 + e] += t[f][e]*fac[r];
                    dbp[f] += t[f][e]*kv*exp(gcs[r]);
                    dgp[f] += t[f][e]*kv*fac[r];
                }
            }
        }
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                dA[a][e] *= fac[tc + 16*e];
            }
        }

        // u = A (beta v): dA += du (beta v)^T, d(beta v) = A^T du
        {
            float dAv[4][4] = { { 0.0f } };
            device const float * vrow = (device const float *) (v + ib*args.nb23 + (t0 + lr)*args.nb22 + h*args.nb21);
            FOR_UNROLL (short cb = 0; cb < GDN_S/32; ++cb) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
                gdn_ld8(X1, vrow + cb*32, lr < nt, tiitg);
                gdn_ld8(X2, sc.dv + (bh*tp + t0 + lr)*GDN_S + cb*32, true, tiitg);
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (short x = 0; x < 32; ++x) {
                    float dur[4], vr[4];
                    FOR_UNROLL (short a = 0; a < 4; ++a) {
                        dur[a] = X2[(tr + 16*a)*GDN_P32 + x];
                        vr[a]  = X1[(tc + 16*a)*GDN_P32 + x];
                    }
                    FOR_UNROLL (short a = 0; a < 4; ++a) {
                        FOR_UNROLL (short e = 0; e < 4; ++e) {
                            dAv[a][e] += dur[a]*vr[e];
                        }
                    }
                }
                float t[2][4] = { { 0.0f } };
                for (short m = 0; m < 64; ++m) {
                    const float a0 = Ai[m*GDN_P64 + wr];
                    const float a1 = Ai[m*GDN_P64 + wr + 32];
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        const float x = X2[m*GDN_P32 + wc + 8*e];
                        t[0][e] += a0*x;
                        t[1][e] += a1*x;
                    }
                }
                FOR_UNROLL (short f = 0; f < 2; ++f) {
                    const short r = wr + 32*f;
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        dbp[f] += t[f][e]*X1[r*GDN_P32 + wc + 8*e];
                        if (r < nt) {
                            dvo[(((ulong) ib*args.T + t0 + r)*args.H + h)*GDN_S + cb*32 + wc + 8*e] = t[f][e]*bs[r];
                        }
                    }
                }
            }
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const short r = tr + 16*a;
                    const short s = tc + 16*e;
                    Dm[r*GDN_P64 + s] = r > s ? dA[a][e] + dAv[a][e]*bs[s] : 0.0f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // dL = -A^T dA A^T on the strictly lower part, times exp(gc_r - gc_s)
        {
            float t1[4][4] = { { 0.0f } };
            for (short x = 0; x < 64; ++x) {
                float dr[4], ar[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    dr[a] = Dm[(tr + 16*a)*GDN_P64 + x];
                    ar[a] = Ai[(tc + 16*a)*GDN_P64 + x];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        t1[a][e] += dr[a]*ar[e];
                    }
                }
            }
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    Tm[(tr + 16*a)*GDN_P64 + tc + 16*e] = t1[a][e];
                }
            }
            threadgroup_barrier(mem_flags::mem_threadgroup);
            float t2[4][4] = { { 0.0f } };
            for (short x = 0; x < 64; ++x) {
                float ac[4], tv[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    ac[a] = Ai[x*GDN_P64 + tr + 16*a];
                    tv[a] = Tm[x*GDN_P64 + tc + 16*a];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        t2[a][e] += ac[a]*tv[e];
                    }
                }
            }
            FOR_UNROLL (short a = 0; a < 4; ++a) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const short r = tr + 16*a;
                    const short s = tc + 16*e;
                    Dm[r*GDN_P64 + s] = r > s ? -t2[a][e]*exp(gcs[r] - gcs[s]) : 0.0f;
                }
            }
        }

        // L = beta_r k_r.k_s (decay folded above): dk += beta_r dL k + dL^T (beta k), dbeta += (dL k) . k
        float kk[4][4] = { { 0.0f } };
        FOR_UNROLL (short cb = 0; cb < GDN_S/32; ++cb) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
            gdn_ld8(X1, krow + cb*32, lr < nt, tiitg);
            threadgroup_barrier(mem_flags::mem_threadgroup);
            for (short x = 0; x < 32; ++x) {
                float kr[4], kc[4];
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    kr[a] = X1[(tr + 16*a)*GDN_P32 + x];
                    kc[a] = X1[(tc + 16*a)*GDN_P32 + x];
                }
                FOR_UNROLL (short a = 0; a < 4; ++a) {
                    FOR_UNROLL (short e = 0; e < 4; ++e) {
                        kk[a][e] += kr[a]*kc[e];
                    }
                }
            }
            float t[2][4]  = { { 0.0f } };
            float u2[2][4] = { { 0.0f } };
            for (short m = 0; m < 64; ++m) {
                const float d0 = Dm[wr*GDN_P64 + m];
                const float d1 = Dm[(wr + 32)*GDN_P64 + m];
                const float e0 = Dm[m*GDN_P64 + wr]*bs[m];
                const float e1 = Dm[m*GDN_P64 + wr + 32]*bs[m];
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    const float x = X1[m*GDN_P32 + wc + 8*e];
                    t[0][e]  += d0*x;
                    t[1][e]  += d1*x;
                    u2[0][e] += e0*x;
                    u2[1][e] += e1*x;
                }
            }
            FOR_UNROLL (short f = 0; f < 2; ++f) {
                const short r = wr + 32*f;
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    dbp[f] += t[f][e]*X1[r*GDN_P32 + wc + 8*e];
                    dka[f][cb*4 + e] += t[f][e]*bs[r] + u2[f][e];
                }
            }
        }

        // d gc: the A^T dw part, then rows minus columns of dL o (beta_r k_r.k_s), then what dqkwg left
        threadgroup_barrier(mem_flags::mem_threadgroup);
        FOR_UNROLL (short a = 0; a < 4; ++a) {
            FOR_UNROLL (short e = 0; e < 4; ++e) {
                const short r = tr + 16*a;
                const short s = tc + 16*e;
                Tm[r*GDN_P64 + s] = Dm[r*GDN_P64 + s]*kk[a][e]*bs[r];
            }
        }
        FOR_UNROLL (short f = 0; f < 2; ++f) {
            float x = dgp[f];
            x += simd_shuffle_xor(x, 1);
            x += simd_shuffle_xor(x, 2);
            x += simd_shuffle_xor(x, 4);
            float y = dbp[f];
            y += simd_shuffle_xor(y, 1);
            y += simd_shuffle_xor(y, 2);
            y += simd_shuffle_xor(y, 4);
            const short r = wr + 32*f;
            if (wc == 0) {
                dgr[r] = x;
                if (r < nt) {
                    dbo[((ulong) ib*args.T + t0 + r)*args.H + h] = y;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        float acc = 0.0f;
        const short rr = GDN_BT - 1 - tiitg;
        if (tiitg < 64 && rr < nt) {
            acc = dgr[rr] + sc.dg[bh*tp + t0 + rr] + sc.dg[(nbh + bh)*tp + t0 + rr];
            for (short s = 0; s < 64; ++s) {
                acc += Tm[rr*GDN_P64 + s] - Tm[s*GDN_P64 + rr];
            }
        }
        float ps = simd_prefix_inclusive_sum(acc);
        if (tiitg == 31) {
            tot[0] = ps;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (tiitg < 64 && rr < nt) {
            ps += sgitg == 1 ? tot[0] : 0.0f;
            dgo[((ulong) ib*args.T + t0 + rr)*args.H + h] = ps;
        }
    }

    device float * dko = (device float *) dst + (ulong) GDN_S*args.Hq*args.T*args.B;
    FOR_UNROLL (short f = 0; f < 2; ++f) {
        const short r = wr + 32*f;
        if (r < nt) {
            device float * p = dko + (((ulong) ib*args.T + t0 + r)*args.Hq + hq)*GDN_S + wc;
            FOR_UNROLL (short cb = 0; cb < 4; ++cb) {
                FOR_UNROLL (short e = 0; e < 4; ++e) {
                    p[cb*32 + 8*e] += dka[f][cb*4 + e];
                }
            }
        }
    }
}
