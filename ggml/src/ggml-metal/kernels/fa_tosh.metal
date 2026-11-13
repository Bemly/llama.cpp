#include "common.h"
#include "dequantize.h"

// AMD flash attention kernels; the upstream ones live in fa_*.metal
// vec kernel above miscompiles. Generalized over the K/V element type via the
// dequantize_*_t4 helpers; standard kernels cover their native dimensions and
// TurboQuant kernels cover padded head dimensions 128/256/384/512/640.

template<
    short DK, short NSG,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &),
    short NW = 32, short NWG = 1, short SUBWP = 0, short DVP = 0>
kernel void kernel_flash_attn_ext_vec_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        device const int  * idx   [[buffer(8)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DV  = DVP ? DVP : DK;   // MLA has DK != DV (K carries the rope tail, V is the bare latent)
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;   // float4 per head row (32 for DV=128, 64 for DV=256)
    // chunks per lane scale with the simd width; on wave64 with DK=128 the upper
    // lanes own no chunk (guarded below) but still join the simd_sum reductions.
    // heads narrower than the simd width would leave the upper lanes idle, so the
    // simdgroup splits into RPS sub-groups that each take their own cache row
    constexpr short SUBW = SUBWP ? SUBWP : (DV4 < NW ? DV4 : NW);
    constexpr short RPS  = NW/SUBW;
    constexpr short NCV  = (DV4 + SUBW - 1)/SUBW;
    constexpr short NCK  = (DK4 + SUBW - 1)/SUBW;
    constexpr short NG   = NSG*RPS;
    constexpr short GPL  = (NG + NW - 1)/NW;
    // NSG (simdgroups per threadgroup) is a template parameter, kept in sync with
    // the host dispatch. More simdgroups split the KV stream into shorter serial
    // loops, which is a large decode win at depth; it is chosen per head dim so
    // the sh_acc threadgroup buffer (NSG*DV4 float4) stays at 16 KiB: 32 for
    // DK=128, 16 for DK=256.

    // padded so the merge below, which reads down a column with the lane as the
    // simdgroup index, does not land every lane on the same bank
    constexpr short SAS = DV4 + 1;
    threadgroup float4 sh_acc[NG*SAS];
    threadgroup float  sh_m[NG];
    threadgroup float  sh_s[NG];

    const int iq1 = tgpig[0];              // query token
    const int iq2 = tgpig[1];              // query head
    const int iq3 = tgpig[2]/NWG;          // batch
    const short iwg = NWG == 1 ? 0 : tgpig[2]%NWG; // this workgroup's slice of the cache

    device const float4 * q4 = (device const float4 *) (q + iq1*args.nb01 + iq2*args.nb02 + iq3*args.nb03);

    // grouped-query attention: map query head -> kv head
    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // Without a mask ne32/ne33 are 0, so the index math would divide by zero.
    device const half * pm = (device const half *) mask;
    if (args.has_mask) {
        pm = (device const half *) (mask + iq1*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    // sparse: the index list is laid out per (row, head, batch), same as the mask
    // sparse when the host filled the index list; uniform across the threadgroup
    const bool has_sparse = args.n_kv_max_padded > 0;

    device const int * pidx = nullptr;
    if (has_sparse) {
        pidx = idx +
            ((int64_t)(iq3%args.ne33)*args.ne32 + (iq2%args.ne32))*args.ne31*args.n_kv_max_padded +
            (iq1%args.ne31)*args.n_kv_max_padded;
    }

    const short sub  = tiisg/SUBW;
    const short lch  = tiisg%SUBW;
    const short gidx = sgitg*RPS + sub;

    float4 myq[NCK];  // this lane's slice(s) of the query
    for (short c = 0; c < NCK; ++c) {
        const short ch = lch + c*SUBW;
        myq[c] = ch < DK4 ? q4[ch] : float4(0.0f);
    }

    float4 acc[NCV];  // this lane's slice(s) of the output
    for (short c = 0; c < NCV; ++c) {
        acc[c] = float4(0.0f);
    }
    float m = -FLT_MAX/2;
    float s = 0.0f;

    // sparse walks the compacted index list instead of the whole cache; a padded slot is -1
    const int n_kv = has_sparse ? args.n_kv_max_padded : args.ne11;

    for (int j = (iwg*NSG + sgitg)*RPS + sub; j < n_kv; j += NWG*NG) {
        const int i11 = has_sparse ? pidx[j] : j;

        // no early-out: the sub-groups of a simdgroup sit on different rows, so the
        // skip is not uniform and the shuffle reduction below would diverge
        const float mv = (has_sparse && i11 < 0) ? -MAXHALF
                       : (args.has_mask ? (float) pm[i11] : 0.0f);
        const bool  msk = mv <= -1e30f;
        if (RPS == 1 && msk) {
            continue;
        }

        // a padded slot keeps running through the body to hold the reduction uniform, and its
        // weight ends up zero, but the address is still formed: read row 0 instead of -1
        const int i11a = i11 < 0 ? 0 : i11;

        device const k_t * kb = (device const k_t *) (kp + (uint64_t) i11a*args.nb11);
        float pl = 0.0f;
        for (short c = 0; c < NCK; ++c) {
            const short ch = lch + c*SUBW;
            if (ch < DK4) {
                float4 kreg;
                deq_k(kb + ch/NKK, ch%NKK, kreg);
                pl += dot(myq[c], kreg);
            }
        }
        if (SUBW < NW) {
            for (ushort o = 1; o < SUBW; o <<= 1) {
                pl += simd_shuffle_xor(pl, o);
            }
        } else {
            pl = simd_sum(pl);
        }
        const float p = pl; // full DK-dim Q.K

        // the host has already divided the scale by the softcap
        const float ps   = p*args.scale;
        const float pp   = (args.logit_softcap != 0.0f ? args.logit_softcap*precise::tanh(ps) : ps) + mv;
        const float nm   = msk ? m : max(m, pp);
        const float corr = exp(m  - nm);
        const float w    = msk ? 0.0f : exp(pp - nm);

        device const v_t * vb = (device const v_t *) (vp + (uint64_t) i11a*args.nb21);
        for (short c = 0; c < NCV; ++c) {
            const short ch = lch + c*SUBW;
            if (ch < DV4) {
                float4 vreg;
                deq_v(vb + ch/NKV, ch%NKV, vreg);
                acc[c] = acc[c]*corr + w*vreg;
            }
        }
        s = s*corr + w;
        m = nm;
    }

    // publish this simdgroup's partials (acc, m, s) and merge in simdgroup 0
    for (short c = 0; c < NCV; ++c) {
        const short ch = lch + c*SUBW;
        if (ch < DV4) {
            sh_acc[gidx*SAS + ch] = acc[c];
        }
    }
    if (lch == 0) {
        sh_m[gidx] = m;
        sh_s[gidx] = s;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // lane g carries group g's (m, s), so the merge is two simd reductions and every
    // simdgroup takes a slice, instead of simdgroup 0 walking NSG entries alone
    float lm = -FLT_MAX/2;
    for (short r = 0; r < GPL; ++r) {
        const short g = tiisg*GPL + r;
        if (g < NG) {
            lm = max(lm, sh_m[g]);
        }
    }
    const float gm = simd_max(lm);

    float fr[GPL];
    float den = 0.0f;
    for (short r = 0; r < GPL; ++r) {
        const short g = tiisg*GPL + r;
        fr[r] = g < NG ? exp(sh_m[g] - gm) : 0.0f;
        den  += g < NG ? fr[r]*sh_s[g] : 0.0f;
    }
    den = simd_sum(den);

    float scorr = 1.0f;
    if (args.has_sinks) {
        const float sk = ((device const float *) sinks)[iq2];
        const float nm = max(gm, sk);
        scorr = exp(gm - nm);
        den   = den*scorr + exp(sk - nm);
    }

    const float inv = (den > 0.0f ? 1.0f/den : 0.0f)*scorr;

    const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq1*args.ne1;

    if (NWG == 1) {
        device float4 * dst4 = (device float4 *) dst;
        for (short i = sgitg; i < DV4; i += NSG) {
            float4 v = 0.0f;
            for (short r = 0; r < GPL; ++r) {
                const short g = tiisg*GPL + r;
                if (g < NG) {
                    v += fr[r]*sh_acc[g*SAS + i];
                }
            }
            v = simd_sum(v);
            if (tiisg == 0) {
                dst4[rid*DV4 + i] = v*inv;
            }
        }
    } else {
        // unnormalized partial in this slice's own max frame, plus (S, M), which is
        // the layout kernel_flash_attn_ext_vec_reduce expects
        const int64_t nrows = (int64_t) args.ne1*args.ne2*args.ne3;

        device float4 * htmp4 = (device float4 *) dst + rid*DV4*NWG;
        device float  * ss    = (device float  *) dst + nrows*DV4*4*NWG;

        for (short i = sgitg; i < DV4; i += NSG) {
            float4 v = 0.0f;
            for (short r = 0; r < GPL; ++r) {
                const short g = tiisg*GPL + r;
                if (g < NG) {
                    v += fr[r]*sh_acc[g*SAS + i];
                }
            }
            v = simd_sum(v);
            if (tiisg == 0) {
                htmp4[i*NWG + iwg] = v;
            }
        }
        if (sgitg == 0 && tiisg == 0) {
            ss[rid*(2*NWG) + 2*iwg + 0] = den;
            ss[rid*(2*NWG) + 2*iwg + 1] = gm;
        }
    }
}

// ===== TurboQuant store: centroids + quantize + set_rows kernels =====

typedef decltype(kernel_flash_attn_ext_vec_amd<128, 32, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_vec_amd_t;

// narrowing subw doubles the chunks per lane; the register pressure caps the pipeline,
// so pairs with q4_0 dequant take fewer simdgroups (see the PAIR macros below).
#define FA_AMD_INST_S(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 1, subw>;
#define FA_AMD_INST_WG(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 32, subw>;
// dk64 narrows to subw=8 like dk128; clean pairs keep n64=32, q4_0 in K or V drops to 24.
#define FA_AMD_PAIR_S(n64, s128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S( 64, n64, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(128, 16, s128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    /* the narrow heads need the split too, but only at depths the host gates apart */ \
    FA_AMD_INST_WG( 64, n64, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 16, s128, nm, kt, nkk, dkf, vt, nkv, dvf)
// q4_0 keys: dk64 and dk128 both take 24 simdgroups for the heavier key dequant
#define FA_AMD_PAIR_Q4(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S( 64, 24, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(128, 24, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG( 64, 24, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 24, 16, nm, kt, nkk, dkf, vt, nkv, dvf)

// symmetric pairs
FA_AMD_PAIR_S(32, 8,  "kf16_vf16",         half4,          1,  dequantize_f16_t4,      half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_S(32, 8,  "kq8_0_vq8_0",       block_q8_0,     8,  dequantize_q8_0_t4_h2,     block_q8_0,     8,  dequantize_q8_0_t4_h2)
FA_AMD_PAIR_Q4("kq4_0_vq4_0",       block_q4_0,     8,  dequantize_q4_0_t4,     block_q4_0,     8,  dequantize_q4_0_t4)
// asymmetric standard pairs (any mix of f16 / q8_0 / q4_0 for K and V), so every
// standard K/V combination runs on the GPU instead of falling back to the CPU.
FA_AMD_PAIR_S(32, 8,  "kq8_0_vf16",        block_q8_0,     8,  dequantize_q8_0_t4_h2,     half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_Q4("kq4_0_vf16",        block_q4_0,     8,  dequantize_q4_0_t4,     half4,          1,  dequantize_f16_t4)
FA_AMD_PAIR_S(32, 8,  "kf16_vq8_0",        half4,          1,  dequantize_f16_t4,      block_q8_0,     8,  dequantize_q8_0_t4_h2)
FA_AMD_PAIR_S(24, 8,  "kf16_vq4_0",        half4,          1,  dequantize_f16_t4,      block_q4_0,     8,  dequantize_q4_0_t4)
FA_AMD_PAIR_S(24, 8,  "kq8_0_vq4_0",       block_q8_0,     8,  dequantize_q8_0_t4_h2,     block_q4_0,     8,  dequantize_q4_0_t4)
FA_AMD_PAIR_Q4("kq4_0_vq8_0",       block_q4_0,     8,  dequantize_q4_0_t4,     block_q8_0,     8,  dequantize_q8_0_t4_h2)
// MLA (deepseek2): K is the latent plus the 64-wide rope tail (DK=576); V is the
// bare latent (DV=512). Same cache tensor, so K and V share one type.
#define FA_AMD_INST_ASYM(dk, dv, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 1, subw, dv>; \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 32, 32, subw, dv>;
FA_AMD_INST_ASYM(576, 512, 16, 32, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM(576, 512, 16, 32, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_INST_ASYM(320, 256, 16, 16, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM(320, 256, 16, 16, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_INST_ASYM
#define FA_AMD_INST_ASYM_W64(dk, dv, nsg, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 1, 64, dv>; \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_dv" #dv "_" nm "_w64_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 32, 64, dv>;
FA_AMD_INST_ASYM_W64(576, 512, 8, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM_W64(576, 512, 8, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_INST_ASYM_W64(320, 256, 8, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_INST_ASYM_W64(320, 256, 8, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_INST_ASYM_W64
// wave64 (AMD GCN/Vega): 64-lane simdgroups, so nsg is the host's base count for the
// head halved (host dispatches simd_width*nsg); a mismatch corrupts the merge. Host
// appends "_w64". Untested on real GCN/Vega, needs a hardware pass before release.
#define FA_AMD_INST_W64(dk, nsg, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64>;
#define FA_AMD_INST_W64_S(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 1, subw>;
#define FA_AMD_INST_W64_WG(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64_wg32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg, kt, nkk, dkf, vt, nkv, dvf, 64, 32, subw>;
#define FA_AMD_PAIR_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S( 64, 16,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(128,  8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(256,  8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(512,  8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG( 64, 16,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(128,  8,  8, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PAIR_W64("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)

// sh_acc needs about 33 KB on 64 lanes, over the 32 KB a GCN 1.0 threadgroup gets; these
// twins halve the simdgroups and the host falls back to them when the full one is refused
#define FA_AMD_INST_W64_LDS(dk, nsg, subw, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64_lds32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg/2, kt, nkk, dkf, vt, nkv, dvf, 64, 1, subw>; \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk" #dk "_" nm "_w64_wg32_lds32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<dk, nsg/2, kt, nkk, dkf, vt, nkv, dvf, 64, 32, subw>;
#define FA_AMD_PAIR_W64_LDS(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS( 64, 16,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(128,  8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(256,  8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(512,  8, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PAIR_W64_LDS("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64_LDS("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PAIR_W64_LDS("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64_LDS("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64_LDS("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64_LDS("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PAIR_W64_LDS("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64_LDS("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64_LDS("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_PAIR_W64_LDS

// DK=72 (vision towers: qwen3vl_merger and SigLIP are 1152/16). f16 only: 72 is
// not a multiple of the 32-element quant block, so quantized K/V cannot occur.
// DV4=18 is not a power of two, so the auto subw would break the shuffle reduction;
// pin subw=16 and let the ch < DV4 guards cover the top two chunks.
FA_AMD_INST_S   (72, 32, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S(72, 16, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// UNet diffusion heads (SD 1.5 runs 40 at full resolution, then 80 and 160). Same
// reasoning as dk72: DV4 is not a power of two, so subw is pinned and the guards
// cover the tail chunk. f16 only, since none of them is a whole quant block.
FA_AMD_INST_S   ( 40, 32,  8, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S( 40, 16,  8, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_S   ( 80, 32, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S( 80, 16, 16, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_S   (160, 16, 32, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_INST_W64_S(160,  8, 32, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PAIR_W64("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PAIR_W64("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PAIR_W64("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_PAIR_W64
// TurboQuant KV: DK is a multiple of 128 (the turbo block is 128 elements, each
// WHT-rotated independently). Dequant stays in the WHT-rotated domain; the graph
// un-rotates. Heads below 128 are zero-padded to 128 in the KV cache.
#define FA_AMD_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (128, 16, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(128, 16, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(256, 16, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (384, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_WG(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_S (640, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TURBO
// wave64 (GCN/Vega) parity; nsg=8. Untested on real hardware.
// The 32 KB twins skip dk384, which fits as is, and dk640 has no split kernel to pair with.
#define FA_AMD_TURBO_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(128, 8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(128, 8, 8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(384, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_WG(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_S(640, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(128, 8,  8, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(256, 8, 16, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_INST_W64_LDS(512, 8, 32, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_vec_amd_dk640_" nm "_w64_lds32")]] kernel flash_attn_ext_vec_amd_t \
    kernel_flash_attn_ext_vec_amd<640, 4, kt, nkk, dkf, vt, nkv, dvf, 64, 1, 32>;
FA_AMD_TURBO_W64("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO_W64("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO_W64("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TURBO_W64("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TURBO_W64("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TURBO_W64("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO_W64("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TURBO_W64("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TURBO_W64
#undef FA_AMD_INST_W64_LDS

#undef FA_AMD_INST_W64
#undef FA_AMD_INST_W64_S
#undef FA_AMD_INST_W64_WG
#undef FA_AMD_PAIR_S
#undef FA_AMD_PAIR_Q4
#undef FA_AMD_INST_S

// Prefill companion of the AMD vec kernel above: NQ simdgroups (one query each)
// share K/V tiles staged in threadgroup memory, so the cache is streamed from
// device memory once per NQ queries instead of once per query.
template<
    short DK, short NQ,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &),
    short NW = 32, short DVP = 0>
kernel void kernel_flash_attn_ext_tile_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiisg [[thread_index_in_simdgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    constexpr short DV  = DVP ? DVP : DK;
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NCK = (DK4 + NW - 1)/NW;
    constexpr short NCV = (DV4 + NW - 1)/NW;
    constexpr short TK  = 8192/(DK + DV); // KV rows per tile (sk + sv <= 16 KiB)

    threadgroup half4 sk[TK*DK4];
    threadgroup half4 sv[TK*DV4];
    threadgroup short sh_any;

    const int iq1 = tgpig.x*NQ + sgitg; // this simdgroup's query token
    const int iq2 = tgpig.y;            // query head
    const int iq3 = tgpig.z;            // batch

    // tail simdgroups past ne01 recompute the last query and skip the store;
    // they must stay for the cooperative loads and barriers
    const int iq1c = iq1 < args.ne01 ? iq1 : args.ne01 - 1;

    device const float4 * q4 = (device const float4 *) (q + iq1c*args.nb01 + iq2*args.nb02 + iq3*args.nb03);

    // grouped-query attention: map query head -> kv head
    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // Without a mask ne32/ne33 are 0, so the index math would divide by zero.
    device const half * pm = (device const half *) mask;
    if (args.has_mask) {
        pm = (device const half *) (mask + iq1c*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
    }

    float4 myq[NCK];
    for (short c = 0; c < NCK; ++c) {
        const short ch = tiisg + c*NW;
        myq[c] = ch < DK4 ? q4[ch] : float4(0.0f);
    }

    float4 acc[NCV];
    for (short c = 0; c < NCV; ++c) {
        acc[c] = float4(0.0f);
    }
    float m = -FLT_MAX/2;
    float s = 0.0f;

    const short tl = NW*sgitg + tiisg; // linear thread id for cooperative loads

    for (int j0 = 0; j0 < args.ne11; j0 += TK) {
        const short tn = (args.ne11 - j0 < TK) ? (short)(args.ne11 - j0) : TK;

        // skip tiles no query of this group attends to (causal top-right triangle)
        if (tl == 0) {
            sh_any = 0;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        {
            float mx = -FLT_MAX;
            for (short jj = tiisg; jj < tn; jj += NW) {
                mx = max(mx, args.has_mask ? (float) pm[j0 + jj] : 0.0f);
            }
            mx = simd_max(mx);
            if (tiisg == 0 && mx > -1e30f) {
                sh_any = 1;
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (sh_any == 0) {
            continue;
        }

        // cooperative dequant of the K/V tile rows into threadgroup memory;
        // the loops are separate because MLA has DK != DV
        for (short c = tl; c < tn*DK4; c += NW*NQ) {
            const short jj = c / DK4;
            const short ch = c % DK4;
            float4 t;
            device const k_t * kb = (device const k_t *) (kp + (uint64_t)(j0 + jj)*args.nb11);
            deq_k(kb + ch/NKK, ch%NKK, t);
            sk[c] = (half4) t;
        }
        for (short c = tl; c < tn*DV4; c += NW*NQ) {
            const short jj = c / DV4;
            const short ch = c % DV4;
            float4 t;
            device const v_t * vb = (device const v_t *) (vp + (uint64_t)(j0 + jj)*args.nb21);
            deq_v(vb + ch/NKV, ch%NKV, t);
            sv[c] = (half4) t;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short jj = 0; jj < tn; ++jj) {
            const float mv = args.has_mask ? (float) pm[j0 + jj] : 0.0f;
            if (mv <= -1e30f) {
                continue;
            }

            float pl = 0.0f;
            for (short c = 0; c < NCK; ++c) {
                const short ch = tiisg + c*NW;
                if (ch < DK4) {
                    pl += dot(myq[c], (float4) sk[jj*DK4 + ch]);
                }
            }
            const float p = simd_sum(pl); // full DK-dim Q.K

            const float ps   = p*args.scale;
            const float pp   = (args.logit_softcap != 0.0f ? args.logit_softcap*precise::tanh(ps) : ps) + mv;
            const float nm   = max(m, pp);
            const float corr = exp(m  - nm);
            const float w    = exp(pp - nm);

            for (short c = 0; c < NCV; ++c) {
                const short ch = tiisg + c*NW;
                if (ch < DV4) {
                    acc[c] = acc[c]*corr + w*(float4) sv[jj*DV4 + ch];
                }
            }
            s = s*corr + w;
            m = nm;
        }
        // the two barriers at the top of the next tile fence sk/sv reuse
    }

    // attention sink: one extra virtual token per head, weight only
    if (args.has_sinks) {
        const float sk_ = ((device const float *) sinks)[iq2];
        const float nm = max(m, sk_);
        const float corr = exp(m - nm);
        for (short c = 0; c < NCV; ++c) {
            acc[c] *= corr;
        }
        s = s*corr + exp(sk_ - nm);
    }

    if (iq1 >= args.ne01) {
        return;
    }

    const float inv = s > 0.0f ? 1.0f/s : 0.0f;
    const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq1*args.ne1;
    device float4 * dst4 = (device float4 *) dst;
    for (short c = 0; c < NCV; ++c) {
        const short ch = tiisg + c*NW;
        if (ch < DV4) {
            dst4[rid*DV4 + ch] = acc[c]*inv;
        }
    }
}

typedef decltype(kernel_flash_attn_ext_tile_amd<128, 16, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_tile_amd_t;

// standard K/V pairs only; turbo caches keep using the vec kernel for prefill
#define FA_AMD_TILE_INST(dk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 16, kt, nkk, dkf, vt, nkv, dvf>;
#define FA_AMD_TILE_PAIR(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST( 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(512, nm, kt, nkk, dkf, vt, nkv, dvf)

FA_AMD_TILE_PAIR("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kf32_vf32",   float4,     1, dequantize_f32_t4,  float4,     1, dequantize_f32_t4)
FA_AMD_TILE_PAIR("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_PAIR("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_PAIR("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
// MLA (deepseek2) keeps the 64-wide rope tail in K only.
#define FA_AMD_TILE_INST_ASYM(dk, dv, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 16, kt, nkk, dkf, vt, nkv, dvf, 32, dv>;
FA_AMD_TILE_INST_ASYM(576, 512, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM(576, 512, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_INST_ASYM(320, 256, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM(320, 256, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_TILE_INST_ASYM
// TurboQuant KV (prefill companion of the FA_AMD_TURBO vec kernels).
#define FA_AMD_TILE_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(384, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(512, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST(640, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TILE_TURBO
// wave64 variants: 64-lane simdgroups, 8 queries per threadgroup (512 threads).
#define FA_AMD_TILE_INST_W64(dk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_" nm "_w64")]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 8, kt, nkk, dkf, vt, nkv, dvf, 64>;
#define FA_AMD_TILE_PAIR_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64( 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(512, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_PAIR_W64("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_PAIR_W64("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_PAIR_W64("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)

// DK=72 vision towers; f16 only, see the vec kernel. TK = 4096/72 = 56 keeps
// sk+sv at 15.75 KiB.
FA_AMD_TILE_INST    (72, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_INST_W64(72, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_PAIR_W64("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_PAIR_W64("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
// MLA wave64 parity (Vega/GCN): eight 64-lane query groups.
#define FA_AMD_TILE_INST_ASYM_W64(dk, dv, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_tile_amd_dk" #dk "_dv" #dv "_" nm "_w64")]] kernel flash_attn_ext_tile_amd_t \
    kernel_flash_attn_ext_tile_amd<dk, 8, kt, nkk, dkf, vt, nkv, dvf, 64, dv>;
FA_AMD_TILE_INST_ASYM_W64(576, 512, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM_W64(576, 512, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_INST_ASYM_W64(320, 256, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_TILE_INST_ASYM_W64(320, 256, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_TILE_INST_ASYM_W64
// TurboQuant KV wave64 parity. Untested on real hardware.
#define FA_AMD_TILE_TURBO_W64(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(128, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(256, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(384, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(512, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_TILE_INST_W64(640, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO_W64("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO_W64("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_TILE_TURBO_W64("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_TILE_TURBO_W64("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_TILE_TURBO_W64("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_TILE_TURBO_W64
#undef FA_AMD_TILE_PAIR_W64
#undef FA_AMD_TILE_INST_W64
#undef FA_AMD_TILE_PAIR
#undef FA_AMD_TILE_INST

// Prefill attention for AMD: BQ queries stay resident while the cache streams in
// blocks of BK, so the score matrix never reaches device memory.
//
// Lanes split into groups of 16 so every softmax reduction stays inside one
// simdgroup on wave32 and wave64 alike. Row strides are padded off the 32-bank
// period to keep the 16 concurrent rows on 16 different banks.
template<
    short DK, short DV, short BQ, short BK, short NT,
    typename k_t, short NKK, void (*deq_k)(device const k_t *, short, thread float4 &),
    typename v_t, short NKV, void (*deq_v)(device const v_t *, short, thread float4 &)>
kernel void kernel_flash_attn_ext_pf_amd(
        constant ggml_metal_kargs_flash_attn_ext_amd & args [[buffer(0)]],
        device const char * q     [[buffer(1)]],
        device const char * k     [[buffer(2)]],
        device const char * v     [[buffer(3)]],
        device const char * mask  [[buffer(4)]],
        device const char * sinks [[buffer(5)]],
        device const char * pad   [[buffer(6)]],
        device       char * dst   [[buffer(7)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]]) {
    constexpr short DK4 = DK/4;
    constexpr short DV4 = DV/4;
    constexpr short NQG = NT/16;
    constexpr short NQR = BQ/NQG;
    constexpr short NKC = BK/16;
    constexpr short NDC = (DV + 15)/16;  // DK 72 (vision) leaves the top lanes idle
    // half2 run before flushing to float; it has to divide DK, and 576 (MLA) is not a multiple of 128
    constexpr short DKF = DK % 128 == 0 ? 128 : (DK % 64 == 0 ? 64 : DK);
    constexpr short SQS = DK + 4;
    constexpr short SVS = BK + 4;
    constexpr short SKV = BK*SQS > DV*SVS ? BK*SQS : DV*SVS;

    threadgroup half sq [BQ*SQS];
    threadgroup half skv[SKV];     // K rows first, then V transposed over the same memory
    threadgroup half sp [BQ*SVS];  // softmax weights

    const short kg = tiitg % 16;
    const short qg = tiitg / 16;

    const int iq1 = tgpig.x*BQ;
    const int iq2 = tgpig.y;
    const int iq3 = tgpig.z;

    const int ikv2 = iq2 / (args.ne02 / args.ne_12_2);
    const int ikv3 = iq3 / (args.ne03 / args.ne_12_3);

    device const char * kp = k + ikv2*args.nb12 + ikv3*args.nb13;
    device const char * vp = v + ikv2*args.nb22 + ikv3*args.nb23;

    // scale folded into Q so the score pass is a plain dot product
    for (short c = tiitg; c < BQ*DK4; c += NT) {
        const short jq = c / DK4;
        const short ch = c % DK4;
        float4 qv = 0.0f;
        if (iq1 + jq < args.ne01) {
            qv = ((device const float4 *) (q + (iq1 + jq)*args.nb01 + iq2*args.nb02 + iq3*args.nb03))[ch]*args.scale;
        }
        *(threadgroup half4 *) (sq + jq*SQS + 4*ch) = (half4) qv;
    }

    // the mask values a thread needs are exactly the ones it reduces, so they stay
    // in registers; staging them through LDS measured as costly as the Q*K pass
    device const half * pmr[NQR];
    if (args.has_mask) {
        for (short rr = 0; rr < NQR; ++rr) {
            const int jq = iq1 + qg + NQG*rr;
            const int gq = jq < args.ne01 ? jq : args.ne01 - 1;
            pmr[rr] = (device const half *)
                (mask + gq*args.nb31 + (iq2%args.ne32)*args.nb32 + (iq3%args.ne33)*args.nb33);
        }
    }

    float m   [NQR];
    float srun[NQR];
    float acc [NQR][NDC];
    for (short rr = 0; rr < NQR; ++rr) {
        m[rr]    = -FLT_MAX/2;
        srun[rr] = 0.0f;
        for (short dd = 0; dd < NDC; ++dd) {
            acc[rr][dd] = 0.0f;
        }
    }

    for (int j0 = 0; j0 < args.ne11; j0 += BK) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // mask and K read together so their loads overlap
        half  mv[NQR][NKC];
        float any = 1.0f;
        if (args.has_mask) {
            any = 0.0f;
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    const short col = kg + 16*cc;
                    const half x = j0 + col < args.ne11 ? pmr[rr][j0 + col] : (half) -MAXHALF;
                    mv[rr][cc] = x;
                    any = max(any, (float) x > -1e30f ? 1.0f : 0.0f);
                }
            }
        }

        for (short c = tiitg; c < BK*DK4; c += NT) {
            const short jj = c / DK4;
            const short ch = c % DK4;
            float4 tv = 0.0f;
            if (j0 + jj < args.ne11) {
                device const k_t * kb = (device const k_t *) (kp + (uint64_t)(j0 + jj)*args.nb11);
                deq_k(kb + ch/NKK, ch%NKK, tv);
            }
            *(threadgroup half4 *) (skv + jj*SQS + 4*ch) = (half4) tv;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        // skip is per query group; the loads and barriers below stay uniform
        for (ushort o = 1; o < 16; o <<= 1) {
            any = max(any, simd_shuffle_xor(any, o));
        }
        if (any > 0.0f) {

        float sc[NQR][NKC];
        for (short rr = 0; rr < NQR; ++rr) {
            for (short cc = 0; cc < NKC; ++cc) {
                sc[rr][cc] = 0.0f;
            }
        }
        for (short d0 = 0; d0 < DK; d0 += DKF) {
            half2 pacc[NQR][NKC];
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    pacc[rr][cc] = 0.0h;
                }
            }
            FOR_UNROLL (short d = 0; d < DKF; d += 2) {
                half2 av[NQR];
                half2 bv[NKC];
                for (short rr = 0; rr < NQR; ++rr) {
                    av[rr] = *(threadgroup const half2 *) (sq + (qg + NQG*rr)*SQS + d0 + d);
                }
                for (short cc = 0; cc < NKC; ++cc) {
                    bv[cc] = *(threadgroup const half2 *) (skv + (kg + 16*cc)*SQS + d0 + d);
                }
                for (short rr = 0; rr < NQR; ++rr) {
                    for (short cc = 0; cc < NKC; ++cc) {
                        pacc[rr][cc] = av[rr]*bv[cc] + pacc[rr][cc];
                    }
                }
            }
            for (short rr = 0; rr < NQR; ++rr) {
                for (short cc = 0; cc < NKC; ++cc) {
                    sc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                }
            }
        }

        for (short rr = 0; rr < NQR; ++rr) {
            const short row = qg + NQG*rr;
            float mx = -FLT_MAX;
            for (short cc = 0; cc < NKC; ++cc) {
                if (args.logit_softcap != 0.0f) {
                    sc[rr][cc] = args.logit_softcap*precise::tanh(sc[rr][cc]);
                }
                if (args.has_mask) {
                    sc[rr][cc] += (float) mv[rr][cc];
                } else if (j0 + kg + 16*cc >= args.ne11) {
                    sc[rr][cc] = -INFINITY;
                }
                mx = max(mx, sc[rr][cc]);
            }
            for (ushort o = 1; o < 16; o <<= 1) {
                mx = max(mx, simd_shuffle_xor(mx, o));
            }

            const float mn   = max(m[rr], mx);
            const float corr = exp(m[rr] - mn);

            float ls = 0.0f;
            for (short cc = 0; cc < NKC; ++cc) {
                const float w = exp(sc[rr][cc] - mn);
                sp[row*SVS + kg + 16*cc] = (half) w;
                ls += w;
            }
            for (ushort o = 1; o < 16; o <<= 1) {
                ls += simd_shuffle_xor(ls, o);
            }

            m[rr]    = mn;
            srun[rr] = srun[rr]*corr + ls;
            for (short dd = 0; dd < NDC; ++dd) {
                acc[rr][dd] *= corr;
            }
        }

        }

        // V lands transposed so the P*V pass reads two cache rows in one half2
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (short c = tiitg; c < BK*DV4; c += NT) {
            const short jj = c / DV4;
            const short ch = c % DV4;
            float4 tv = 0.0f;
            if (j0 + jj < args.ne11) {
                device const v_t * vb = (device const v_t *) (vp + (uint64_t)(j0 + jj)*args.nb21);
                deq_v(vb + ch/NKV, ch%NKV, tv);
            }
            for (short i = 0; i < 4; ++i) {
                skv[(4*ch + i)*SVS + jj] = (half) tv[i];
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (any == 0.0f) {
            continue;
        }

        for (short j = 0; j < BK; j += BK) {
            half2 pacc[NQR][NDC];
            for (short rr = 0; rr < NQR; ++rr) {
                for (short dd = 0; dd < NDC; ++dd) {
                    pacc[rr][dd] = 0.0h;
                }
            }
            FOR_UNROLL (short jj = 0; jj < BK; jj += 2) {
                half2 pv[NQR];
                half2 vv[NDC];
                for (short rr = 0; rr < NQR; ++rr) {
                    pv[rr] = *(threadgroup const half2 *) (sp + (qg + NQG*rr)*SVS + j + jj);
                }
                for (short dd = 0; dd < NDC; ++dd) {
                    vv[dd] = DV % 16 == 0 || kg + 16*dd < DV
                        ? *(threadgroup const half2 *) (skv + (kg + 16*dd)*SVS + j + jj) : 0.0h;
                }
                for (short rr = 0; rr < NQR; ++rr) {
                    for (short dd = 0; dd < NDC; ++dd) {
                        pacc[rr][dd] = pv[rr]*vv[dd] + pacc[rr][dd];
                    }
                }
            }
            for (short rr = 0; rr < NQR; ++rr) {
                for (short dd = 0; dd < NDC; ++dd) {
                    acc[rr][dd] += (float) pacc[rr][dd][0] + (float) pacc[rr][dd][1];
                }
            }
        }
    }

    if (args.has_sinks) {
        const float sv = ((device const float *) sinks)[iq2];
        for (short rr = 0; rr < NQR; ++rr) {
            const float mn   = max(m[rr], sv);
            const float corr = exp(m[rr] - mn);
            for (short dd = 0; dd < NDC; ++dd) {
                acc[rr][dd] *= corr;
            }
            srun[rr] = srun[rr]*corr + exp(sv - mn);
            m[rr]    = mn;
        }
    }

    device float * dst1 = (device float *) dst;
    for (short rr = 0; rr < NQR; ++rr) {
        const int iq = iq1 + qg + NQG*rr;
        if (iq >= args.ne01) {
            continue;
        }
        const float inv = srun[rr] > 0.0f ? 1.0f/srun[rr] : 0.0f;
        const int64_t rid = (int64_t) iq3*args.ne2*args.ne1 + iq2 + (int64_t) iq*args.ne1;
        for (short dd = 0; dd < NDC; ++dd) {
            if (DV % 16 == 0 || kg + 16*dd < DV) {
                dst1[rid*DV + kg + 16*dd] = acc[rr][dd]*inv;
            }
        }
    }
}

typedef decltype(kernel_flash_attn_ext_pf_amd<128, 128, 64, 64, 256, half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4>) flash_attn_ext_pf_amd_t;

// BQ falls as the head grows so the O accumulators stay near 32 registers per
// thread. Head 512 would leave one query row per thread, and stays on the tile kernel.
#define FA_AMD_PF_INST(dk, bq, bk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_pf_amd_dk" #dk "_" nm)]] kernel flash_attn_ext_pf_amd_t \
    kernel_flash_attn_ext_pf_amd<dk, dk, bq, bk, 256, kt, nkk, dkf, vt, nkv, dvf>;
// asymmetric head: MLA keeps K wider than V
#define FA_AMD_PF_INST_DV(dk, dv, bq, bk, nm, kt, nkk, dkf, vt, nkv, dvf) \
template [[host_name("kernel_flash_attn_ext_pf_amd_dk" #dk "_dv" #dv "_" nm)]] kernel flash_attn_ext_pf_amd_t \
    kernel_flash_attn_ext_pf_amd<dk, dv, bq, bk, 256, kt, nkk, dkf, vt, nkv, dvf>;
#define FA_AMD_PF_PAIR(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST( 64, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(128, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(256, 32, 64, nm, kt, nkk, dkf, vt, nkv, dvf)

FA_AMD_PF_PAIR("kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
// f32 K and V come from training, which reads the K and V of the step instead of a cache
FA_AMD_PF_PAIR("kf32_vf32",   float4,     1, dequantize_f32_t4,  float4,     1, dequantize_f32_t4)
FA_AMD_PF_PAIR("kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PF_PAIR("kq4_0_vq4_0", block_q4_0, 8, dequantize_q4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq8_0_vf16",  block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_PF_PAIR("kq4_0_vf16",  block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
FA_AMD_PF_PAIR("kf16_vq8_0",  half4,      1, dequantize_f16_t4,  block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PF_PAIR("kf16_vq4_0",  half4,      1, dequantize_f16_t4,  block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq8_0_vq4_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_PAIR("kq4_0_vq8_0", block_q4_0, 8, dequantize_q4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
// TurboQuant KV (large-batch prefill companion of the tile/vec turbo kernels).
// dk384/dk512/dk640 have no PF variant; the host routes them to the tile kernel.
#define FA_AMD_PF_TURBO(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(128, 64, 64, nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(256, 32, 64, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PF_TURBO("kq8_0_vturbo2", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kq8_0_vturbo3", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kq8_0_vturbo4", block_q8_0, 8, dequantize_q8_0_t4_h2, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo2", half4, 1, dequantize_f16_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo3", half4, 1, dequantize_f16_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kf16_vturbo4", half4, 1, dequantize_f16_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo2_vq8_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PF_TURBO("kturbo2_vf16", block_turbo2_0, 32, dequantize_turbo2_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo2", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo3", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo2_vturbo4", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo3_vq8_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PF_TURBO("kturbo3_vf16", block_turbo3_0, 32, dequantize_turbo3_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo2", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo3", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo3_vturbo4", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo4_vq8_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q8_0, 8, dequantize_q8_0_t4_h2)
FA_AMD_PF_TURBO("kturbo4_vf16", block_turbo4_0, 32, dequantize_turbo4_0_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo2", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo3", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kturbo4_vturbo4", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo2", block_q4_0, 8, dequantize_q4_0_t4, block_turbo2_0, 32, dequantize_turbo2_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo3", block_q4_0, 8, dequantize_q4_0_t4, block_turbo3_0, 32, dequantize_turbo3_0_t4)
FA_AMD_PF_TURBO("kq4_0_vturbo4", block_q4_0, 8, dequantize_q4_0_t4, block_turbo4_0, 32, dequantize_turbo4_0_t4)
FA_AMD_PF_TURBO("kturbo2_vq4_0", block_turbo2_0, 32, dequantize_turbo2_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_TURBO("kturbo3_vq4_0", block_turbo3_0, 32, dequantize_turbo3_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
FA_AMD_PF_TURBO("kturbo4_vq4_0", block_turbo4_0, 32, dequantize_turbo4_0_t4, block_q4_0, 8, dequantize_q4_0_t4)
#undef FA_AMD_PF_TURBO

// vision towers, f16 only like the vec and tile kernels
FA_AMD_PF_INST(72, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// UNet diffusion heads, f16 only like head 72
FA_AMD_PF_INST( 40, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST( 80, 64, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST(160, 32, 64, "kf16_vf16", half4, 1, dequantize_f16_t4, half4, 1, dequantize_f16_t4)
// head 512 has no PF variant and falls to the tile kernel, ~21% worse at depth; BK=32 keeps
// the shared memory at 54 KB, the same league as dk256
#define FA_AMD_PF_512(nm, kt, nkk, dkf, vt, nkv, dvf) \
    FA_AMD_PF_INST(512, 16, 32, nm, kt, nkk, dkf, vt, nkv, dvf)
FA_AMD_PF_512("kf16_vf16",  half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PF_512("kq8_0_vf16", block_q8_0, 8, dequantize_q8_0_t4_h2, half4,      1, dequantize_f16_t4)
FA_AMD_PF_512("kq4_0_vf16", block_q4_0, 8, dequantize_q4_0_t4, half4,      1, dequantize_f16_t4)
// MLA: K is 576 (512 latent + 64 rope) and V is 512; the tile kernel is ~21% slower at depth
FA_AMD_PF_INST_DV(576, 512, 16, 32, "kf16_vf16",  half4,      1, dequantize_f16_t4,  half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(576, 512, 16, 32, "kq8_0_vf16", block_q8_0, 8, dequantize_q8_0_t4_h2, half4, 1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(320, 256, 16, 32, "kf16_vf16",   half4,      1, dequantize_f16_t4,  half4,      1, dequantize_f16_t4)
FA_AMD_PF_INST_DV(320, 256, 16, 32, "kq8_0_vq8_0", block_q8_0, 8, dequantize_q8_0_t4_h2, block_q8_0, 8, dequantize_q8_0_t4_h2)
#undef FA_AMD_PF_512
#undef FA_AMD_PF_PAIR
#undef FA_AMD_PF_INST

// note: I think the s_t can be half instead of float, because the Q*K scaling is done before storing to shared mem
//       in the other (non-vec) kernel, we need s_t to also be float because we scale during the soft_max
//

constant short FC_lightning_indexer_amd_nh [[function_constant(FC_LIGHTNING_INDEXER + 0)]];

// Lightning indexer for GPUs without simdgroup matrices: one thread per key and query row, with
// the key tile and a few heads of the queries in threadgroup memory. It stays under 32 KB of it.
template<
    typename kd4x4_t,
    short nl_k,
    void (*deq_k)(device const kd4x4_t *, short, thread half4x4 &)>
kernel void kernel_lightning_indexer_amd(
        constant ggml_metal_kargs_lightning_indexer & args,
        device const char * q,
        device const char * k,
        device const char * w,
        device const char * m,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    constexpr short DK   = OP_LIGHTNING_INDEXER_DK;
    constexpr short DK4  = DK/4;
    constexpr short DK16 = DK/16;
    constexpr short NK   = OP_LIGHTNING_INDEXER_AMD_NK;
    constexpr short NB   = OP_LIGHTNING_INDEXER_AMD_NB;
    constexpr short NT   = NK*NB;
    constexpr short NHC  = 8;
    // a padded row puts the keys a simdgroup reads at the same step in different banks
    constexpr short SKS  = DK4 + 1;

    const short NH = FC_lightning_indexer_amd_nh;

    threadgroup half4 sk[NK*SKS];
    threadgroup half4 sq[NB*NHC*DK4];
    threadgroup float sw[NB*NHC];

    const int i_stream = tgpig.z;
    const int i_kv_0   = tgpig.x*NK;
    const int i_b_0    = tgpig.y*NB;

    for (short i = tiitg; i < NK*DK16; i += NT) {
        const short ik  = i/DK16;
        const short i16 = i%DK16;

        half4x4 tmp;
        if (i_kv_0 + ik < args.n_kv) {
            device const kd4x4_t * kr = (device const kd4x4_t *) (k + (uint64_t) (i_kv_0 + ik)*args.nbk2 + (uint64_t) i_stream*args.nbk3);
            deq_k(kr + i16/nl_k, i16%nl_k, tmp);
        } else {
            for (short j = 0; j < 4; ++j) {
                tmp[j] = half4(0.0h);
            }
        }
        for (short j = 0; j < 4; ++j) {
            sk[ik*SKS + i16*4 + j] = tmp[j];
        }
    }

    const short ik = tiitg%NK;
    const short ib = tiitg/NK;

    float score = 0.0f;

    for (short h0 = 0; h0 < NH; h0 += NHC) {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short i = tiitg; i < NB*NHC*DK4; i += NT) {
            const short b  = i/(NHC*DK4);
            const short h  = (i/DK4)%NHC;
            const short i4 = i%DK4;
            const int   bb = i_b_0 + b;

            half4 v = half4(0.0h);
            if (bb < args.n_batch && h0 + h < NH) {
                device const float4 * q4 = (device const float4 *) (q + (uint64_t) bb*args.nbq2 + (uint64_t) i_stream*args.nbq3 + (uint64_t) (h0 + h)*args.nbq1);
                v = half4(q4[i4]);
            }
            sq[i] = v;
        }
        for (short i = tiitg; i < NB*NHC; i += NT) {
            const short b  = i/NHC;
            const short h  = i%NHC;
            const int   bb = i_b_0 + b;

            sw[i] = bb < args.n_batch && h0 + h < NH
                ? ((device const float *) (w + (uint64_t) bb*args.nbw1 + (uint64_t) i_stream*args.nbw3))[h0 + h]
                : 0.0f;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const half4 * kr = sk + ik*SKS;
        for (short h = 0; h < NHC; ++h) {
            threadgroup const half4 * qr = sq + (ib*NHC + h)*DK4;

            float4 acc = 0.0f;
            for (short i4 = 0; i4 < DK4; ++i4) {
                acc += float4(kr[i4])*float4(qr[i4]);
            }
            score += max(acc[0] + acc[1] + acc[2] + acc[3], 0.0f)*sw[ib*NHC + h];
        }
    }

    const int i_kv    = i_kv_0 + ik;
    const int i_batch = i_b_0 + ib;
    if (i_kv < args.n_kv && i_batch < args.n_batch) {
        device const half  * pm = (device const half  *) (m   + (uint64_t) i_batch*args.nbm1 + (uint64_t) (i_stream % args.mask_ne3)*args.nbm3);
        device       float * pd = (device       float *) (dst + (uint64_t) i_batch*args.nb1  + (uint64_t) i_stream*args.nb3);

        pd[i_kv] = score + (float) pm[i_kv];
    }
}

typedef decltype(kernel_lightning_indexer_amd<half4x4, 1, dequantize_f16>) kernel_lightning_indexer_amd_t;

template [[host_name("kernel_lightning_indexer_amd_f32")]]  kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<float4x4,   1, dequantize_f32>;
template [[host_name("kernel_lightning_indexer_amd_f16")]]  kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<half4x4,    1, dequantize_f16>;
template [[host_name("kernel_lightning_indexer_amd_q4_0")]] kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<block_q4_0, 2, dequantize_q4_0>;
template [[host_name("kernel_lightning_indexer_amd_q4_1")]] kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<block_q4_1, 2, dequantize_q4_1>;
template [[host_name("kernel_lightning_indexer_amd_q5_0")]] kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<block_q5_0, 2, dequantize_q5_0>;
template [[host_name("kernel_lightning_indexer_amd_q5_1")]] kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<block_q5_1, 2, dequantize_q5_1>;
template [[host_name("kernel_lightning_indexer_amd_q8_0")]] kernel kernel_lightning_indexer_amd_t kernel_lightning_indexer_amd<block_q8_0, 2, dequantize_q8_0>;
