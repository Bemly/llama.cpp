#include "common.h"
#include "dequantize.h"

constant bool FC_mul_mm_bc_inp [[function_constant(FC_MUL_MM + 0)]];
constant bool FC_mul_mm_bc_out [[function_constant(FC_MUL_MM + 1)]];
constant short FC_mul_mm_ne12  [[function_constant(FC_MUL_MM + 2)]];
constant short FC_mul_mm_ne13  [[function_constant(FC_MUL_MM + 3)]];
constant short FC_mul_mm_r2    [[function_constant(FC_MUL_MM + 4)]];
constant short FC_mul_mm_r3    [[function_constant(FC_MUL_MM + 5)]];
constant int32_t FC_mul_mm_id_map0_ne02 [[function_constant(FC_MUL_MM + 7)]];
// ToshGEMM manual tiled path for GPUs without simdgroup_matrix (AMD RDNA/GCN)
constant bool  FC_mul_mm_manual [[function_constant(FC_MUL_MM + 8)]];
constant bool  FC_mul_mm_narrow [[function_constant(FC_MUL_MM + 9)]];
constant bool  FC_mul_mm_double_buffer [[function_constant(FC_MUL_MM + 10)]];
constant bool  FC_mul_mm_acc_f32 [[function_constant(FC_MUL_MM + 11)]];
constant bool  FC_mul_mm_uniform_scale [[function_constant(FC_MUL_MM + 12)]];
// A is read transposed (out_prod): output rows run along a row of src0 and K along its rows
constant bool  FC_mul_mm_ta_def [[function_constant(FC_MUL_MM + 14)]];
constant bool  FC_mul_mm_ta = is_function_constant_defined(FC_mul_mm_ta_def) && FC_mul_mm_ta_def;
// src1 columns arrive scaled by a power of two each, kept after dst, so a gradient fits the half tile
constant bool  FC_mul_mm_cs_def [[function_constant(FC_MUL_MM + 15)]];
constant bool  FC_mul_mm_cs = is_function_constant_defined(FC_mul_mm_cs_def) && FC_mul_mm_cs_def;
// map0 emits 64-entry tiles for the NLR 8 (_w) kernel_mul_mm_id; unset: 32, which every variant accepts
constant bool  FC_mul_mm_id_map0_w_def [[function_constant(FC_MUL_MM + 16)]];
constant bool  FC_mul_mm_id_map0_w = is_function_constant_defined(FC_mul_mm_id_map0_w_def) && FC_mul_mm_id_map0_w_def;

// each block_q contains 16*nl weights
#ifdef GGML_METAL_HAS_TENSOR
template<
    typename SA, typename SA_4x4, typename SA_8x8,
    typename SB, typename SB_2x4, typename SB_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread SA_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4, short NLR = 4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * srcA,
        device const char * srcB,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig [[threadgroup_position_in_grid]],
        ushort tiitg [[thread_index_in_threadgroup]],
        ushort sgitg [[simdgroup_index_in_threadgroup]]) {
    (void) sgitg;

    // Matrix dimensions: A(M,K) x B(K,N) -> C(M,N)
    const int K = args.ne00;
    const int M = args.ne0;
    const int N = args.ne1;

    // Batch dimension handling
    const int im = tgpig.z;
    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    // Batch offsets for srcA and srcB
    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;

    // Tile dimensions
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;
    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;

    // Tile offsets in output matrix
    const int ra = tgpig.y * NRA;
    const int rb = tgpig.x * NRB;

    // Threadgroup memory for dequantized A tile only
    threadgroup SA * sa = (threadgroup SA *)(shmem);

    // Work-item count for A loading
    constexpr int A_WORK_ITEMS = NRA * N_MM_NK;
    constexpr int NUM_THREADS = N_SIMDWIDTH * N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    // tA wraps threadgroup memory
    auto tA = tensor(sa, dextents<int32_t, 2>(N_MM_NK_TOTAL, NRA));

    // tB wraps device memory directly
    device T1 * ptrB = (device T1 *)(srcB + args.nb12*i12 + args.nb13*i13);
    const int strideB = args.nb11 / sizeof(T1);
    auto tB = tensor(ptrB, dextents<int32_t, 2>(K, N), array<int, 2>({1, strideB}));

    // Configure matmul operation
    // note: K is dynamic_extent (clamped to the valid range in PHASE 2), since a static
    //       N_MM_NK_TOTAL K tile would read src1 out of bounds when K % N_MM_NK_TOTAL != 0
    // ref: https://github.com/ggml-org/llama.cpp/pull/27064
    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(
            NRB, NRA, static_cast<int>(dynamic_extent), false, true, true,
            mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y>> mm;

    auto cT = mm.get_destination_cooperative_tensor<decltype(tB), decltype(tA), float>();

    // Accumulate partial results over K dimension
    for (int loop_k = 0; loop_k < K; loop_k += N_MM_NK_TOTAL) {
        // === PHASE 1: Dequantization of A into threadgroup memory ===
        for (int work = tiitg; work < A_WORK_ITEMS; work += NUM_THREADS) {
            const int row = work / N_MM_NK;
            const int k_chunk = work % N_MM_NK;
            const int k_pos = loop_k + k_chunk * 16;
            const short k_base = k_chunk * 16;

            // Bounds check: skip device read if row is out of matrix bounds
            if (ra + row < M) {
                if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
                    // Element-wise reads when K is not aligned (nb01 not aligned for half4x4/float4x4).
                    // MSL spec Table 2.5: half4x4 requires 8-byte alignment. When K is odd,
                    // nb01 = K*2 is not 8-byte aligned, so odd-row pointers are misaligned.
                    // Mirrors the legacy kernel's existing guard.
                    device const T0 * row_ptr = (device const T0 *)(srcA + args.nb01 * (ra + row) + offset0);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? (SA) row_ptr[k_pos + i] : (SA)0;
                    }
                } else {
                    const int block_idx = k_pos / (16 * nl);
                    const short il = (k_pos / 16) % nl;

                    device const block_q * row_ptr = (device const block_q *)(srcA + args.nb01 * (ra + row) + offset0);

                    SA_4x4 temp_a;
                    dequantize_func(row_ptr + block_idx, il, temp_a);

                    FOR_UNROLL (short i = 0; i < 16; i++) {
                        // Zero-pad A for K positions beyond valid range (handles partial K iterations)
                        sa[row * N_MM_NK_TOTAL + (k_base + i)] = (k_pos + i < K) ? temp_a[i/4][i%4] : (SA)0;
                    }
                }
            } else {
                // Zero-pad rows beyond matrix bounds
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    sa[row * N_MM_NK_TOTAL + (k_base + i)] = (SA)0;
                }
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        // === PHASE 2: Tensor matmul ===
        // Clamp the K extent of both operand tensors to the remaining valid K range so
        // the dynamic-K op never reads past the K extent of src1 (or the staged A tile).
        const int kExt = min(N_MM_NK_TOTAL, K - loop_k);

        auto tAv = tensor(sa, dextents<int32_t, 2>(kExt, NRA), array<int, 2>({1, N_MM_NK_TOTAL}));
        auto tBv = tensor(ptrB + loop_k + rb * strideB, dextents<int32_t, 2>(kExt, N - rb), array<int, 2>({1, strideB}));

        mm.run(tBv, tAv, cT);

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    // Store result tile to output matrix (with batch offset)
    // a slice is not used because its byte offset is int32 and wraps at 2 GiB
    // ref: https://github.com/ggml-org/llama.cpp/pull/28748
    const int tileM = min(NRA, M - ra);
    const int tileN = min(NRB, N - rb);

    device float * dstTile = (device float *)dst + (uint64_t) im * N * M + (uint64_t) rb * M + ra;

    auto tD = tensor(dstTile, dextents<int32_t, 2>(tileM, tileN), array<int, 2>({1, M}));
    cT.store(tD);
}

#else

template<
    typename S0, typename S0_4x4, typename S0_8x8,
    typename S1, typename S1_2x4, typename S1_8x8,
    typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &),
    typename T0, typename T0_4x4, typename T1, typename T1_2x4, short NLR = 4>
kernel void kernel_mul_mm(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        uint3  tgpg [[threadgroups_per_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    constexpr int NR0 = 64;

    // 64x64 over 8 simdgroups on the manual path: halves the re-read traffic per output
    const int NR1 = FC_mul_mm_manual ? (FC_mul_mm_narrow ? 32 : 64) : 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    // in bands of src1 tiles every src0 tile runs over one band before the next one starts, so
    // a src1 larger than the cache is read from the cache once per band, not from memory
    uint tx = tgpig.x;
    uint ty = tgpig.y;
    if (args.band > 0) {
        const uint lin = tgpig.y*tgpg.x + tgpig.x;
        const uint b0  = (lin / (args.band*tgpg.y))*args.band;
        const uint bw  = min((uint) args.band, tgpg.x - b0);
        const uint rem = lin - b0*tgpg.y;
        ty = rem / bw;
        tx = b0 + rem % bw;
    }

    const int im = tgpig.z;
    const int r0 = ty*NR0;
    const int r1 = tx*NR1;

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;
    const short nr1 = (args.ne1 - r1 < NR1) ? (args.ne1 - r1) : NR1;

    // a thread shouldn't load data outside of the matrix
    const short lr0 = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1  = ((short)tiitg/NL1)      < nr1 ? ((short)tiitg/NL1)      : nr1 - 1;
    const short lr1b = ((short)tiitg/NL1 + 32) < nr1 ? ((short)tiitg/NL1 + 32) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int i12 = im % FC_mul_mm_ne12;
    const int i13 = im / FC_mul_mm_ne12;

    const uint64_t offset0 = (i12/FC_mul_mm_r2)*args.nb02 + (i13/FC_mul_mm_r3)*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1)
        + args.nb10*iy);

    device const T1 * yb = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*(r1 + lr1b)
        + args.nb10*iy);

    device const float * cscale = (device const float *) dst + (uint64_t) args.ne0*args.ne1*FC_mul_mm_ne12*FC_mul_mm_ne13 + (uint64_t) im*args.ne1;
    const float cs  = FC_mul_mm_cs ? cscale[r1 + lr1]  : 1.0f;
    const float csb = FC_mul_mm_cs && NLR == 8 ? cscale[r1 + lr1b] : 1.0f;

    // Manual tiled path for GPUs without simdgroup matrix; simdgroup path otherwise.
    const short lane = tiitg % 32;
    const short lr = lane % 8; // row within an 8-row block of sa
    const short lc = lane / 8; // low 2 bits of the output column
    // Virtual 32-wide subgroup so the manual tile offsets are correct on wave64; == sgitg on wave32.
    const short vsg = tiitg / 32;

    // manual path: 4x4 register tile per lane, rows 8*rr + lr, cols lc + 4*cc
    // of the simdgroup's 32x16 output subtile
    float acc[NLR][4];
    S0_8x8 ma[4];
    S1_8x8 mb[2];
    simdgroup_float8x8 mc[8];

    if (FC_mul_mm_manual) {
        for (short i = 0; i < NLR; i++) { for (short j = 0; j < 4; j++) { acc[i][j] = 0.f; } }
    } else {
        for (short i = 0; i < 8; i++) {
            mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }
    }

    // NLR 8 runs 128 threads: all load A, each loads two columns of B
    const bool loadA = !FC_mul_mm_manual || NLR == 8 || tiitg < 128;

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
        // ToshGEMM ping-pongs two 8 KiB tiles.  While a fast simdgroup starts
        // filling the alternate tile, slower simdgroups can finish consuming
        // the current one; the barrier below then publishes the complete next
        // tile.  This removes the pre-load barrier from every K iteration.
        const short db = FC_mul_mm_double_buffer ? ((loop_k/NK) & 1) : 0;
        threadgroup S0 * sa = (threadgroup S0 *)(shmem + db*8192);
        threadgroup S1 * sb = (threadgroup S1 *)(shmem + db*8192 + 4096);

        // load data and store to threadgroup memory
        if (FC_mul_mm_ta) {
            // a thread dequantizes 16 consecutive outputs of one src0 row (one K)
            const short tk = tiitg >> 2;
            const short tg = tiitg & 3;
            const int   kr = loop_k + tk;
            const int   n0 = r0 + 16*tg;

            S0_4x4 temp_a;
            temp_a[0] = 0; temp_a[1] = 0; temp_a[2] = 0; temp_a[3] = 0;
            if (loadA && kr < args.ne00 && n0 < args.ne0) {
                device const block_q * xb = (device const block_q *) (src0 + offset0 + (uint64_t) kr*args.nb01) + n0/(16*nl);
                dequantize_func(xb, (n0/16) % nl, temp_a);
            }

            if (!FC_mul_mm_double_buffer) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            if (loadA) {
                const short sx = tk/8;
                const short ly = tk%8;
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    const short row = 16*tg + i;
                    *(sa + 64*(8*sx + row/8) + 8*(row%8) + ly) = temp_a[i/4][i%4];
                }
            }
        } else if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            if (!FC_mul_mm_double_buffer) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            // no need for dequantization
            for (short i = 0; loadA && i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // manual path stores row-major within the 8x8 block so (k, k+1)
                // pairs sit contiguous for its packed-half2 compute loop; the
                // simdgroup path keeps the layout simdgroup_load expects
                const short el = FC_mul_mm_manual ? (8*lx + ly) : (8*ly + lx);

                *(sa + 64*ib + el) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            if (loadA) {
                // q4_K/q5_K wide: il is il0 + 2*(t % 8) with t = loop_k/NK, so the scale pair's
                // j < 4 is (loop_k & 128) == 0, uniform per wave instead of derived per lane
                if (FC_mul_mm_uniform_scale && NLR == 8 &&
                    (is_same<block_q, block_q4_K>::value || is_same<block_q, block_q5_K>::value)) {
                    dequantize_lo(x, il, (loop_k & 128) == 0, temp_a);
                } else if (FC_mul_mm_manual && has_dequantize_words<block_q>()) {
                    dequantize_words(x, il, temp_a);
                } else {
                    dequantize_func(x, il, temp_a);
                }
            }

            if (!FC_mul_mm_double_buffer) {
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;

            if (FC_mul_mm_manual) {
                // its row-major 8x8 layout leaves each group of 4 contiguous, so the
                // 16 scalar stores collapse into 4 vector ones
                const short ib0 = 8*(2*il0    ) + sy;
                const short ib1 = 8*(2*il0 + 1) + sy;

                if (loadA) {
                    *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 0) = temp_a[0];
                    *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 4) = temp_a[1];
                    *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 0) = temp_a[2];
                    *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 4) = temp_a[3];
                }
            } else {
                FOR_UNROLL (short i = 0; loadA && i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short ly = i%8;
                    const short ib = 8*sx + sy;

                    *(sa + 64*ib + (8*ly + lx)) = temp_a[i/4][i%4];
                }
            }
        }

        {
            const short ncb = FC_mul_mm_manual ? 8 : 4;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            const short ib = ncb*sx + sy;

            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    *(sb + 64*ib + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) ((float) *((device T1 *) y + i)*cs) : 0;
                }
                if (NLR == 8) {
                    for (short i = 0; i < 8; ++i) {
                        *(sb + 64*(ib + 4) + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) ((float) *((device T1 *) yb + i)*csb) : 0;
                    }
                }
            } else if (FC_mul_mm_cs) {
                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(float2x4(*((device T1_2x4 *) y))*cs);
                if (NLR == 8) {
                    *(threadgroup S1_2x4 *)(sb + 64*(ib + 4) + 8*ly) = (S1_2x4)(float2x4(*((device T1_2x4 *) yb))*csb);
                }
            } else {
                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
                if (NLR == 8) {
                    *(threadgroup S1_2x4 *)(sb + 64*(ib + 4) + 8*ly) = (S1_2x4)(*((device T1_2x4 *) yb));
                }
            }
        }

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y  += NK;
        yb += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup const S0 * lsma = (sa + (NLR == 8 ? 0 : 4*64*(vsg%2)));
        threadgroup const S1 * lsmb = (sb + 2*64*(NLR == 8 ? vsg : vsg/2));

        if (FC_mul_mm_manual) {
            // packed rank-1 updates on (k, k+1) pairs, flushed to float once per NK. The
            // accumulator is half for weights; attention products overflow it, so those use float.
            if (FC_mul_mm_acc_f32) {
                vec<float,2> pacc[NLR][4];
                for (short rr = 0; rr < NLR; rr++) { for (short cc = 0; cc < 4; cc++) { pacc[rr][cc] = 0; } }

                for (short ik = 0; ik < NK/8; ik++) {
                    FOR_UNROLL (short k = 0; k < 8; k += 2) {
                        vec<S0,2> av[NLR];
                        vec<S1,2> bv[4];
                        for (short rr = 0; rr < NLR; rr++) {
                            av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                        }
                        for (short cc = 0; cc < 4; cc++) {
                            const short col = lc + 4*cc;
                            bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                        }
                        for (short rr = 0; rr < NLR; rr++) {
                            for (short cc = 0; cc < 4; cc++) {
                                // mul+add: no bfloat2 fma() overload; fast math contracts it anyway
                                pacc[rr][cc] = vec<float,2>(av[rr])*vec<float,2>(bv[cc]) + pacc[rr][cc];
                            }
                        }
                    }

                    lsma += 8*64;
                    lsmb += 8*64;
                }

                for (short rr = 0; rr < NLR; rr++) {
                    for (short cc = 0; cc < 4; cc++) {
                        acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                    }
                }
            } else {
                vec<S0,2> pacc[NLR][4];

                // the first block writes its product, so the accumulators need no zeroing pass
                FOR_UNROLL (short k = 0; k < 8; k += 2) {
                    vec<S0,2> av[NLR];
                    vec<S1,2> bv[4];
                    for (short rr = 0; rr < NLR; rr++) {
                        av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                    }
                    for (short cc = 0; cc < 4; cc++) {
                        const short col = lc + 4*cc;
                        bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                    }
                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < 4; cc++) {
                            const vec<S0,2> p = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]);
                            pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                        }
                    }
                }
                lsma += 8*64;
                lsmb += 8*64;

                for (short ik = 1; ik < NK/8; ik++) {
                    // bv is the small side (4 against NLR), so widening only it halves its reads
                    // for four extra registers instead of twelve. Only pays on the 8x4 tile:
                    // on the narrow one it measures 1.3 to 1.8% worse
                    if (NLR == 8) {
                        FOR_UNROLL (short k = 0; k < 8; k += 4) {
                            vec<S1,4> bv[4];
                            for (short cc = 0; cc < 4; cc++) {
                                const short col = lc + 4*cc;
                                bv[cc] = *(threadgroup const vec<S1,4> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                            }
                            FOR_UNROLL (short h = 0; h < 2; h++) {
                                vec<S0,2> av[NLR];
                                for (short rr = 0; rr < NLR; rr++) {
                                    av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k + 2*h);
                                }
                                for (short rr = 0; rr < NLR; rr++) {
                                    for (short cc = 0; cc < 4; cc++) {
                                        const vec<S0,2> b2 = h == 0 ? vec<S0,2>(bv[cc].xy) : vec<S0,2>(bv[cc].zw);
                                        pacc[rr][cc] = vec<S0,2>(av[rr])*b2 + pacc[rr][cc];
                                    }
                                }
                            }
                        }
                    } else {
                        FOR_UNROLL (short k = 0; k < 8; k += 2) {
                            vec<S0,2> av[NLR];
                            vec<S1,2> bv[4];
                            for (short rr = 0; rr < NLR; rr++) {
                                av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                            }
                            for (short cc = 0; cc < 4; cc++) {
                                const short col = lc + 4*cc;
                                bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                            }
                            for (short rr = 0; rr < NLR; rr++) {
                                for (short cc = 0; cc < 4; cc++) {
                                    // mul+add: no bfloat2 fma() overload; fast math contracts it anyway
                                    pacc[rr][cc] = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]) + pacc[rr][cc];
                                }
                            }
                        }
                    }

                    lsma += 8*64;
                    lsmb += 8*64;
                }

                for (short rr = 0; rr < NLR; rr++) {
                    for (short cc = 0; cc < 4; cc++) {
                        acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                    }
                }
            }
        } else {
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
    }

    if (FC_mul_mm_manual) {
        const int base_row = r0 + (NLR == 8 ? 0 : 32*(vsg & 1));
        const int base_col = r1 + 16*(NLR == 8 ? vsg : (vsg >> 1));
        device float * Dbase = (device float *) dst + im*(int64_t)args.ne1*args.ne0;
        for (short rr = 0; rr < NLR; rr++) {
            for (short cc = 0; cc < 4; cc++) {
                const int row = base_row + 8*rr + lr;
                const int col = base_col + lc + 4*cc;
                if (row < args.ne0 && col < args.ne1) {
                    Dbase[row + (int64_t)col*args.ne0] = FC_mul_mm_cs ? acc[rr][cc]/cscale[col] : acc[rr][cc];
                }
            }
        }
    } else if (!FC_mul_mm_bc_out || (r0 + NR0 <= args.ne0 && r1 + NR1 <= args.ne1)) {
        device float * C = (device float *) dst +
            (r0 + 32*(sgitg &  1)) +
            (int64_t) (r1 + 16*(sgitg >> 1))*args.ne0 + (int64_t) im*args.ne1*args.ne0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], C + 8*(i%4) + (int64_t) 8*args.ne0*(i/4), args.ne0, 0, false);
        }
    } else {
        threadgroup_barrier(mem_flags::mem_threadgroup);

        threadgroup float * temp_str = ((threadgroup float *) shmem) + 32*(sgitg&1) + (16*(sgitg >> 1))*NR0;

        for (short i = 0; i < 8; i++) {
            simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        if (sgitg == 0) {
            for (int j = tiitg; j < nr1; j += NR1) {
                device float  * D  = (device float  *) dst + r0 + (int64_t) (r1 + j)*args.ne0 + (int64_t) im*args.ne1*args.ne0;
                device float4 * D4 = (device float4 *) D;

                threadgroup float  * C  = temp_str + (j*NR0);
                threadgroup float4 * C4 = (threadgroup float4 *) C;

                int i = 0;
                for (; i < nr0/4; i++) {
                    *(D4 + i) = *(C4 + i);
                }

                i *= 4;
                for (; i < nr0; i++) {
                    *(D + i) = *(C + i);
                }
            }
        }
    }
}

#endif // GGML_METAL_HAS_TENSOR

template<short ne20> // n_expert_used
kernel void kernel_mul_mm_id_map0(
        constant ggml_metal_kargs_mul_mm_id_map0 & args,
        device  const char * src2,
        device        char * htpe,
        device        char * hids,
        threadgroup   char * shmem [[threadgroup(0)]],
        ushort tpitg[[thread_position_in_threadgroup]],
        ushort   ntg[[threads_per_threadgroup]]) {
    const short ide = tpitg; // expert id

    // per expert: entry count (pass 1), replaced by the base offset of the slice (pass 2)
    threadgroup uint32_t * cnt = (threadgroup uint32_t *) (shmem);
    threadgroup uint32_t * bas = cnt + FC_mul_mm_id_map0_ne02;

    // staging of the ids of ntg tokens
    threadgroup uint16_t * sids = (threadgroup uint16_t *) (bas + FC_mul_mm_id_map0_ne02);

    // pass 1: count the entries of this expert - every occurrence of this expert within the
    // ids is an entry (duplicate ids produce multiple entries for the same token)
    uint32_t n_all = 0;

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            // 0xFFFF marks a skipped slot, it never matches an expert since ne02 fits in a threadgroup
            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const int32_t id = src2_i32[i20];
                sids[tpitg*ne20 + i20] = id == -1 ? 0xFFFF : (uint16_t) id;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                n_all += sids_t[i20] == ide;
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    cnt[ide] = n_all;

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // exclusive prefix sum over the counts: the entries of this expert start at bas[ide] -
    // every slice is padded by one entry for the uniform dummy store of pass 2 - and emit the
    // entry tile descriptors of kernel_mul_mm_id: (expert, first entry, valid rows) per tile
    if (tpitg == 0) {
        const uint32_t NTILE = FC_mul_mm_id_map0_w ? 64 : 32; // must match NR1 of kernel_mul_mm_id

        uint32_t base = 0;
        uint32_t tile = 0;

        device uint32_t * tpe_u32 = (device uint32_t *) (htpe);

        for (int e = 0; e < FC_mul_mm_id_map0_ne02; ++e) {
            const uint32_t n = cnt[e];

            bas[e] = base;

            for (uint32_t k = 0; k < n; k += NTILE) {
                tpe_u32[1 + 3*tile    ] = e;
                tpe_u32[1 + 3*tile + 1] = base + k;
                tpe_u32[1 + 3*tile + 2] = n - k < NTILE ? n - k : NTILE;

                tile++;
            }

            base += n + 1;
        }

        tpe_u32[0] = tile;
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    // pass 2: write the entries of this expert into its slice of the compact list
    device int32_t * ids_i32 = (device int32_t *) (hids);

    uint32_t ient = bas[ide];

    for (int i21 = 0; i21 < args.ne21; i21 += ntg) { // n_tokens
        if (i21 + tpitg < args.ne21) {
            device const int32_t * src2_i32 = (device const int32_t *) (src2 + (i21 + tpitg)*args.nb21);

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                sids[tpitg*ne20 + i20] = src2_i32[i20];
            }
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short t = 0; t < ntg; t++) {
            if (i21 + t >= args.ne21) {
                break;
            }

            threadgroup const uint16_t * sids_t = sids + t*ne20;

            short first = -1;
            short n_ents = 0;

            FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                const short hit = sids_t[i20] == ide;

                first = hit && first < 0 ? i20 : first;
                n_ents += hit;
            }

            // one store per token for every thread, regardless of the match - conditional
            // stores of the divergent lanes are very slow - the dummy value of the tokens
            // without a match lands on the padding slot of this slice and is never read
            ids_i32[ient] = first >= 0 ? (i21 + t)*ne20 + first : 0;

            if (n_ents > 1) {
                // duplicate ids: the remaining matches of this token
                short j = 1;

                FOR_UNROLL (short i20 = 0; i20 < ne20; i20++) {
                    if (sids_t[i20] == ide && i20 != first) {
                        ids_i32[ient + j] = (i21 + t)*ne20 + i20;
                        j += 1;
                    }
                }
            }

            ient += n_ents;
        }

        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// kernel_mul_mm_id only writes the rows that an expert owns, so the skipped rows are zeroed here
kernel void kernel_mul_mm_id_zero(
        constant ggml_metal_kargs_mul_mm_id_zero & args,
        device const char * src2,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int i20 = tgpig.x; // slot
    const int i21 = tgpig.y; // token

    const int32_t id = ((device const int32_t *) (src2 + i21*args.nb21))[i20];

    if (id != -1) {
        return;
    }

    device float * dst_row = (device float *) (dst + i20*args.nb1 + i21*args.nb2);

    for (int i0 = tpitg.x; i0 < args.ne0; i0 += ntg.x) {
        dst_row[i0] = 0.0f;
    }
}

typedef decltype(kernel_mul_mm_id_map0<1>) kernel_mul_mm_id_map0_t;


// TODO: refactor into two seperate functions to avoid ifdef branching: with and without tensor support
template<typename S0, typename S0_4x4, typename S0_8x8, typename S1, typename S1_2x4, typename S1_8x8, typename block_q, short nl, void (*dequantize_func)(device const block_q *, short, thread S0_4x4 &), typename T0, typename T0_4x4, typename T1, typename T1_2x4, short NLR = 4>
kernel void kernel_mul_mm_id(
        constant ggml_metal_kargs_mul_mm_id & args,
        device const char * src0,
        device const char * src1,
        device const char * htpe,
        device const char * hids,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
#ifdef GGML_METAL_HAS_TENSOR
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
    threadgroup float * sc = (threadgroup float *)(shmem);
#else
    threadgroup S0 * sa = (threadgroup S0 *)(shmem);
    threadgroup S1 * sb = (threadgroup S1 *)(shmem + 4096);
#endif

    constexpr int NR0 = 64;
    constexpr int NR1 = NLR == 8 ? 64 : 32;

    constexpr int NK  = 32;
    constexpr int NL0 = NK/16;
    constexpr int NL1 = NK/8;

    const int tile = tgpig.x; // entry tile, described by the prepass
    const int r0   = tgpig.y*NR0;

    device const uint32_t * tpe_u32 = (device const uint32_t *) (htpe);
    device const int32_t  * ids_i32 = (device const int32_t  *) (hids);

    // the prepass emits one (expert, first entry, valid rows) descriptor per tile that has
    // work - the grid is sized by a static upper bound, so the tail early exits
    if (tile >= (int) tpe_u32[0]) {
        return;
    }

    const int im  = tpe_u32[1 + 3*tile];     // expert
    const int ie0 = tpe_u32[1 + 3*tile + 1]; // first entry of this tile in hids

    // number of entries in this tile, always <= NR1 (32-entry tiles leave the upper half of the
    // NLR 8 tile idle, see FC_mul_mm_id_map0_w)
    const short nr1 = (short) ((tpe_u32[1 + 3*tile + 2] < (uint32_t) NR1) ? tpe_u32[1 + 3*tile + 2] : NR1);

    // if this block is of 64x32 shape or smaller
    const short nr0 = (args.ne0 - r0 < NR0) ? (args.ne0 - r0) : NR0;

    // a thread shouldn't load data outside of the matrix
    const short lr0  = ((short)tiitg/NL0) < nr0 ? ((short)tiitg/NL0) : nr0 - 1; // 0 .. 63
    const short lr1  = ((short)tiitg/NL1)      < nr1 ? ((short)tiitg/NL1)      : nr1 - 1;
    const short lr1b = ((short)tiitg/NL1 + 32) < nr1 ? ((short)tiitg/NL1 + 32) : nr1 - 1;

    const short il0 = (tiitg % NL0);

    short il = il0;

    const int id  = ids_i32[ie0 + lr1];
    const int idb = NLR == 8 ? ids_i32[ie0 + lr1b] : 0;

    // int, not short: the AMD Metal compiler miscompiles device address math
    // built from short indices with ids-derived dynamic bases (reads zeros/junk
    // for every token with id >= 256, i.e. token index >= 128)
    const int i11 = (id % args.ne20) % args.ne11;
    const int i12 = (id / args.ne20);
    const int i13 = 0;

    const int i11b = (idb % args.ne20) % args.ne11;
    const int i12b = (idb / args.ne20);

    // Virtual 32-wide subgroup and lane; == sgitg/tiisg on wave32, keeps the
    // tile offsets, store and scatter wave-agnostic on wave64.
    const short lane = tiitg % 32;
    const short vsg  = tiitg / 32;

    const uint64_t offset0 = im*args.nb02 + i13*args.nb03;
    const short    offset1 = il0/nl;

    device const block_q * x = (device const block_q *)(src0 + args.nb01*(r0 + lr0) + offset0) + offset1;

    const short iy = 8*(tiitg % NL1);

    device const T1 * y = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12
        + args.nb11*i11
        + args.nb10*iy);

    device const T1 * yb = (device const T1 *)(src1
        + args.nb13*i13
        + args.nb12*i12b
        + args.nb11*i11b
        + args.nb10*iy);

    // skip the upper half of the token tile when the expert did not fill it
    constexpr short NR1H = NR1/2;

    const bool has_hi = nr1 > NR1H;

#ifndef GGML_METAL_HAS_TENSOR
    // Manual tiled path for GPUs without simdgroup matrix; simdgroup path otherwise.
    const short lr = lane % 8; // row within an 8-row block of sa
    const short lc = lane / 8; // low 2 bits of the output column

    // an expert narrower than the tile leaves whole simdgroups on padding columns; they still
    // load and hit the barriers, they just skip the products
    const bool sg_live = 16*(NLR == 8 ? vsg : vsg/2) < nr1;

    // manual path: NLRx4 register tile per lane, rows 8*rr + lr, cols lc + 4*cc
    // of the simdgroup's output subtile
    float acc[NLR][4];
    S0_8x8 ma[4];
    S1_8x8 mb[2];

    simdgroup_float8x8 mc[8];

    if (FC_mul_mm_manual) {
        for (short i = 0; i < NLR; i++) { for (short j = 0; j < 4; j++) { acc[i][j] = 0.f; } }
    } else {
        for (short i = 0; i < 8; i++){
            mc[i] = make_filled_simdgroup_matrix<float, 8>(0.f);
        }
    }

#else
    auto tA  = tensor<threadgroup S0, dextents<int32_t, 2>, tensor_inline>(sa, dextents<int32_t, 2>(NK, NR0));

    // sb is [NR1][NK] row-major
    auto tB0 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb,             dextents<int32_t, 2>(NK, NR1H));
    auto tB1 = tensor<threadgroup S1, dextents<int32_t, 2>, tensor_inline>(sb + NR1H*NK,   dextents<int32_t, 2>(NK, NR1H));

    mpp::tensor_ops::matmul2d<
        mpp::tensor_ops::matmul2d_descriptor(NR1H, NR0, NK, false, true, false, mpp::tensor_ops::matmul2d_descriptor::mode::multiply_accumulate),
        execution_simdgroups<4>> mm;

    auto cT0 = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB0), float>();
    auto cT1 = mm.template get_destination_cooperative_tensor<decltype(tA), decltype(tB1), float>();
#endif

    for (int loop_k = 0; loop_k < args.ne00; loop_k += NK) {
#ifndef GGML_METAL_HAS_TENSOR
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

              //const short lx = i%8;
              //const short ly = (tiitg/NL0)%8;
                const short lx = (tiitg/NL0)%8;
                const short ly = i%8;

                const short ib = 8*sx + sy;

                // manual path stores row-major within the 8x8 block so (k, k+1)
                // pairs sit contiguous for its packed-half2 compute loop
                const short el = FC_mul_mm_manual ? (8*lx + ly) : (8*ly + lx);

                *(sa + 64*ib + el) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : (S0) 0;
            }
        } else {
            S0_4x4 temp_a;
            if (FC_mul_mm_manual && has_dequantize_words<block_q>()) {
                dequantize_words(x, il, temp_a);
            } else {
                dequantize_func(x, il, temp_a);
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);

            const short sy = (tiitg/NL0)/8;
            const short lx = (tiitg/NL0)%8;

            if (FC_mul_mm_manual) {
                const short ib0 = 8*(2*il0    ) + sy;
                const short ib1 = 8*(2*il0 + 1) + sy;

                *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 0) = temp_a[0];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib0 + 8*lx + 4) = temp_a[1];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 0) = temp_a[2];
                *(threadgroup vec<S0, 4> *)(sa + 64*ib1 + 8*lx + 4) = temp_a[3];
            } else {
                FOR_UNROLL (short i = 0; i < 16; i++) {
                    const short sx = 2*il0 + i/8;
                    const short ly = i%8;
                    const short ib = 8*sx + sy;

                    *(sa + 64*ib + (8*ly + lx)) = temp_a[i/4][i%4];
                }
            }
        }

        {
            const short ncb = NLR == 8 ? 8 : 4;

            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;
            const short ly = (tiitg/NL1)%8;

            const short ib = ncb*sx + sy;

            const bool colb = NLR == 8 && (short)tiitg/NL1 + 32 < nr1;

            if (FC_mul_mm_bc_inp) {
                for (short i = 0; i < 8; ++i) {
                    *(sb + 64*ib + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
                }
                if (colb) {
                    for (short i = 0; i < 8; ++i) {
                        *(sb + 64*(ib + 4) + 8*ly + i) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) yb + i) : 0;
                    }
                }
            } else {
                *(threadgroup S1_2x4 *)(sb + 64*ib + 8*ly) = (S1_2x4)(*((device T1_2x4 *) y));
                if (colb) {
                    *(threadgroup S1_2x4 *)(sb + 64*(ib + 4) + 8*ly) = (S1_2x4)(*((device T1_2x4 *) yb));
                }
            }
        }
#else
        // load data and store to threadgroup memory
        if (is_same<T0_4x4, block_q>::value && FC_mul_mm_bc_inp) {
            threadgroup_barrier(mem_flags::mem_threadgroup);

            // no need for dequantization
            for (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = loop_k + 16*il + i < args.ne00 ? (S0) *((device T0 *) x + i) : 0;
            }
        } else {
            S0_4x4 temp_a;
            dequantize_func(x, il, temp_a);

            threadgroup_barrier(mem_flags::mem_threadgroup);

            FOR_UNROLL (short i = 0; i < 16; i++) {
                const short sx = 2*il0 + i/8;
                const short sy = (tiitg/NL0)/8;

                const short lx = i%8;
                const short ly = (tiitg/NL0)%8;
                //const short lx = (tiitg/NL0)%8;
                //const short ly = i%8;

                *(sa + NK*(8*sy + ly) + 8*sx + lx) = temp_a[i/4][i%4];
            }
        }

        if (FC_mul_mm_bc_inp) {
            for (short i = 0; i < 8; ++i) {
                const short sx = (tiitg%NL1);
                const short sy = (tiitg/NL1)/8;

                const short lx = i;
                const short ly = (tiitg/NL1)%8;
                //const short lx = (tiitg/NL1)%8;
                //const short ly = i;

                *(sb + NK*(8*sy + ly) + 8*sx + lx) = loop_k + iy + i < args.ne00 ? (S1) *((device T1 *) y + i) : 0;
            }
        } else {
            const short sx = (tiitg%NL1);
            const short sy = (tiitg/NL1)/8;

            //const short lx = i;
            const short ly = (tiitg/NL1)%8;
            //const short lx = (tiitg/NL1)%8;
            //const short ly = i;

            *(threadgroup S1_2x4 *)(sb + NK*(8*sy + ly) + 8*sx) = (S1_2x4)(*((device T1_2x4 *) y));
        }
#endif

        il = (il + 2 < nl) ? il + 2 : il % 2;
        x  = (il < 2) ? x + (2 + nl - 1)/nl : x;

        y  += NK;
        yb += NK;

        threadgroup_barrier(mem_flags::mem_threadgroup);

#ifndef GGML_METAL_HAS_TENSOR
        // load matrices from threadgroup memory and conduct outer products
        threadgroup const S0 * lsma = (sa + (NLR == 8 ? 0 : 4*64*(vsg%2)));
        threadgroup const S1 * lsmb = (sb + 2*64*(NLR == 8 ? vsg : vsg/2));

        constexpr short bstride = (NLR == 8 ? 8 : 4)*64;

        if (FC_mul_mm_manual && sg_live) {
            if (FC_mul_mm_acc_f32) {
                for (short ik = 0; ik < NK/8; ik++) {
                    // packed rank-1 updates on (k, k+1) pairs: half the instructions of the
                    // scalar loop; partials flush to float every 8-deep slice for precision
                    vec<float,2> pacc[NLR][4];

                    FOR_UNROLL (short k = 0; k < 8; k += 2) {
                        vec<S0,2> av[NLR];
                        vec<S1,2> bv[4];
                        for (short rr = 0; rr < NLR; rr++) {
                            av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                        }
                        for (short cc = 0; cc < 4; cc++) {
                            const short col = lc + 4*cc;
                            bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                        }
                        for (short rr = 0; rr < NLR; rr++) {
                            for (short cc = 0; cc < 4; cc++) {
                                // mul+add: no bfloat2 fma() overload; fast math contracts it anyway
                                const vec<float,2> p = vec<float,2>(av[rr])*vec<float,2>(bv[cc]);
                                pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                            }
                        }
                    }

                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < 4; cc++) {
                            acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                        }
                    }

                    lsma += 8*64;
                    lsmb += bstride;
                }
            } else {
                // pacc lives across the whole NK, as the dense kernel already does: flushing it
                // once per 8-deep slice cost four times the float conversions
                vec<S0,2> pacc[NLR][4];

                FOR_UNROLL (short k = 0; k < 8; k += 2) {
                    vec<S0,2> av[NLR];
                    vec<S1,2> bv[4];
                    for (short rr = 0; rr < NLR; rr++) {
                        av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                    }
                    for (short cc = 0; cc < 4; cc++) {
                        const short col = lc + 4*cc;
                        bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                    }
                    for (short rr = 0; rr < NLR; rr++) {
                        for (short cc = 0; cc < 4; cc++) {
                            const vec<S0,2> p = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]);
                            pacc[rr][cc] = k == 0 ? p : p + pacc[rr][cc];
                        }
                    }
                }
                lsma += 8*64;
                lsmb += bstride;

                for (short ik = 1; ik < NK/8; ik++) {
                    // widening only bv halves its reads for four extra registers; it pays on the
                    // 8x4 tile and measures worse on the 4x4 one
                    if (NLR == 8) {
                        FOR_UNROLL (short k = 0; k < 8; k += 4) {
                            vec<S1,4> bv[4];
                            for (short cc = 0; cc < 4; cc++) {
                                const short col = lc + 4*cc;
                                bv[cc] = *(threadgroup const vec<S1,4> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                            }
                            FOR_UNROLL (short h = 0; h < 2; h++) {
                                vec<S0,2> av[NLR];
                                for (short rr = 0; rr < NLR; rr++) {
                                    av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k + 2*h);
                                }
                                for (short rr = 0; rr < NLR; rr++) {
                                    for (short cc = 0; cc < 4; cc++) {
                                        const vec<S0,2> b2 = h == 0 ? vec<S0,2>(bv[cc].xy) : vec<S0,2>(bv[cc].zw);
                                        pacc[rr][cc] = vec<S0,2>(av[rr])*b2 + pacc[rr][cc];
                                    }
                                }
                            }
                        }
                    } else {
                        FOR_UNROLL (short k = 0; k < 8; k += 2) {
                            vec<S0,2> av[NLR];
                            vec<S1,2> bv[4];
                            for (short rr = 0; rr < NLR; rr++) {
                                av[rr] = *(threadgroup const vec<S0,2> *)(lsma + 64*rr + 8*lr + k);
                            }
                            for (short cc = 0; cc < 4; cc++) {
                                const short col = lc + 4*cc;
                                bv[cc] = *(threadgroup const vec<S1,2> *)(lsmb + 64*(col/8) + 8*(col%8) + k);
                            }
                            for (short rr = 0; rr < NLR; rr++) {
                                for (short cc = 0; cc < 4; cc++) {
                                    // mul+add: no bfloat2 fma() overload; fast math contracts it anyway
                                    pacc[rr][cc] = vec<S0,2>(av[rr])*vec<S0,2>(bv[cc]) + pacc[rr][cc];
                                }
                            }
                        }
                    }

                    lsma += 8*64;
                    lsmb += bstride;
                }

                for (short rr = 0; rr < NLR; rr++) {
                    for (short cc = 0; cc < 4; cc++) {
                        acc[rr][cc] += (float) pacc[rr][cc][0] + (float) pacc[rr][cc][1];
                    }
                }
            }
        } else if (!FC_mul_mm_manual) {
            FOR_UNROLL (short ik = 0; ik < NK/8; ik++) {
                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 4; i++) {
                    simdgroup_load(ma[i], lsma + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 2; i++) {
                    simdgroup_load(mb[i], lsmb + 64*i, 8, 0, false);
                }

                simdgroup_barrier(mem_flags::mem_none);

                FOR_UNROLL (short i = 0; i < 8; i++){
                    simdgroup_multiply_accumulate(mc[i], mb[i/4], ma[i%4], mc[i]);
                }

                lsma += 8*64;
                lsmb += 4*64;
            }
        }
#else
        auto sA  = tA.slice(0, 0);
        auto sB0 = tB0.slice(0, 0);

        mm.run(sB0, sA, cT0);

        if (has_hi) {
            auto sB1 = tB1.slice(0, 0);

            mm.run(sB1, sA, cT1);
        }
#endif
    }

    // block is smaller than the tile, we should avoid writing data outside of the matrix
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // the 8x4 tile's 64 columns do not fit the scratch, so it stages them in two halves
    for (short p = 0; p < (NLR == 8 ? 2 : 1); p++) {
        if (p > 0) {
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

#ifdef GGML_METAL_HAS_TENSOR
        auto tC0 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc,             dextents<int32_t, 2>(NR0, NR1H));
        cT0.store(tC0);

        if (has_hi) {
            auto tC1 = tensor<threadgroup float, dextents<int32_t, 2>, tensor_inline>(sc + NR1H*NR0, dextents<int32_t, 2>(NR0, NR1H));
            cT1.store(tC1);
        }
#else
        threadgroup float * temp_str = ((threadgroup float *) shmem) +
            (NLR == 8 ? (16*(vsg & 1))*NR0 : 32*(vsg & 1) + (16*(vsg >> 1))*NR0);

        if (FC_mul_mm_manual) {
            // mirror the simdgroup_store layout so the scatter loop below is unchanged
            if (sg_live && (NLR != 8 || (vsg >> 1) == p)) {
                for (short rr = 0; rr < NLR; rr++) {
                    for (short cc = 0; cc < 4; cc++) {
                        temp_str[(8*rr + lr) + (lc + 4*cc)*NR0] = acc[rr][cc];
                    }
                }
            }
        } else {
            for (short i = 0; i < 8; i++) {
                simdgroup_store(mc[i], temp_str + 8*(i%4) + 8*NR0*(i/4), NR0, 0, false);
            }
        }
#endif

        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (short j = vsg; j < 32 && j + 32*p < nr1; j += 4) {
            const int id = ids_i32[ie0 + 32*p + j];

            // int, not short: the AMD Metal compiler miscompiles device address math
            // built from short indices with ids-derived dynamic bases
            const int ide = id % args.ne20;
            const int idt = id / args.ne20;

            device float  * D  = (device float  *) dst + r0 + (int64_t) ide*args.ne0 + (int64_t) idt*args.ne1*args.ne0;
            device float4 * D4 = (device float4 *) D;

            threadgroup float  * C  = (threadgroup float  *) shmem + j*NR0;
            threadgroup float4 * C4 = (threadgroup float4 *) C;

            int i = lane;
            for (; i < nr0/4; i += 32) {
                *(D4 + i) = *(C4 + i);
            }

            i = (4*(nr0/4)) + lane;
            for (; i < nr0; i += 32) {
                *(D + i) = *(C + i);
            }
        }
    }
}

//
// matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_t;

#if defined(GGML_METAL_HAS_BF16)
#endif


//
// indirect matrix-matrix multiplication
//

typedef decltype(kernel_mul_mm_id<half, half4x4, simdgroup_half8x8, half, half2x4, simdgroup_half8x8, float4x4, 1, dequantize_f32, float, float4x4, float, float2x4>) mul_mm_id;

template [[host_name("kernel_mul_mm_id_map0_ne20_1" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<1>;
template [[host_name("kernel_mul_mm_id_map0_ne20_2" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<2>;
template [[host_name("kernel_mul_mm_id_map0_ne20_4" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<4>;
template [[host_name("kernel_mul_mm_id_map0_ne20_5" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<5>;
template [[host_name("kernel_mul_mm_id_map0_ne20_6" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<6>;
template [[host_name("kernel_mul_mm_id_map0_ne20_8" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<8>;
template [[host_name("kernel_mul_mm_id_map0_ne20_9" )]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<9>;
template [[host_name("kernel_mul_mm_id_map0_ne20_10")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<10>;
template [[host_name("kernel_mul_mm_id_map0_ne20_16")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<16>;
template [[host_name("kernel_mul_mm_id_map0_ne20_22")]] kernel kernel_mul_mm_id_map0_t kernel_mul_mm_id_map0<22>;
template [[host_name("kernel_mul_mm_f16_f32_w")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 8>;
template [[host_name("kernel_mul_mm_bf16e_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   bf16e4x4,      1,     dequantize_bf16e,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_0_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_1_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q8_0_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q2_K_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q3_K_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q5_K_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q6_K_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_0_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_1_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_mxfp4_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq4_nl_f32_w")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq4_xs_f32_w")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq3_s_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_s_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs,   QK_NL, dequantize_iq3_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_xs_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,   QK_NL, dequantize_iq2_xs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs,   QK_NL, dequantize_iq2_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq1_m_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_iq1_s_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_pq2_0_f32_w")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0,   8,     dequantize_pq2_0,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_ptq1_0_f32_w")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0,  8,     dequantize_ptq1_0,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_q4_K_f32_w")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_f32_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
// gradients: the f32 operand is staged as bfloat, which keeps the f32 range (a gradient past 65504 is inf in half)
template [[host_name("kernel_mul_mm_f32_f32_gb")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, float4x4, 1,  dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32_gb")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, half4x4,  1,  dequantize_f16,     half,   half4x4,   float, float2x4>;
template [[host_name("kernel_mul_mm_f16_f32_gb_w")]] kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, half4x4,  1,  dequantize_f16,     half,   half4x4,   float, float2x4, 8>;
// out_prod: the weight staged as half, the gradient as bfloat for its range (f32 takes kernel_mul_mm_f32_f32_gb)
template [[host_name("kernel_mul_mm_ta_q8_0_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   bfloat, bfloat2x4, simdgroup_bfloat8x8, block_q8_0, 2, dequantize_q8_0, float, float4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_ta_f16_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   bfloat, bfloat2x4, simdgroup_bfloat8x8, half4x4,    1, dequantize_f16,  half,  half4x4,  float, float2x4>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_bf16_f32")]]    kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
template [[host_name("kernel_mul_mm_bf16_bf16")]]   kernel mul_mm_t kernel_mul_mm<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, bfloat, bfloat2x4>;
#endif
template [[host_name("kernel_mul_mm_bf16e_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   bf16e4x4,      1,     dequantize_bf16e,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_pq2_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_ptq1_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_1_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q8_0_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q2_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q3_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q4_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q5_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_q6_K_f32")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f32")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f32")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f32")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_f32_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_f16_f16")]]     kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_q1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_pq2_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_ptq1_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_1_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q8_0_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_mxfp4_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q2_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q3_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q4_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q5_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_q6_K_f16")]]    kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_xxs_f16")]] kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq3_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq2_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_s_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq1_m_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_nl_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_iq4_xs_f16")]]  kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_tq2_0_f16")]]   kernel mul_mm_t kernel_mul_mm<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f32_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_bf16_f32")]]    kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, bfloat4x4,     1,     dequantize_bf16,    bfloat, bfloat4x4, float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_bf16e_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   bf16e4x4,      1,     dequantize_bf16e,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_pq2_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_ptq1_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_h")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f32")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f32")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  float, float2x4>;
template [[host_name("kernel_mul_mm_id_f16_f32_w")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_0_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_1_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_0_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_1_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q8_0_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_mxfp4_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q2_K_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q3_K_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q4_K_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q5_K_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_q6_K_f32_w")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq3_s_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_s_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs,   QK_NL, dequantize_iq3_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,   QK_NL, dequantize_iq2_xs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs,   QK_NL, dequantize_iq2_xxs,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq1_m_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq1_s_f32_w")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f32_w")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f32_w")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  float, float2x4, 8>;
template [[host_name("kernel_mul_mm_id_f32_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   float4x4,      1,     dequantize_f32,     float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_f16_f16")]]     kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   half4x4,       1,     dequantize_f16,     half,   half4x4,   half, half2x4>;
template [[host_name("kernel_mul_mm_id_q1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q1_0,    8,     dequantize_q1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_0,    4,     dequantize_q2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_pq2_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_pq2_0, 8,     dequantize_pq2_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_ptq1_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_ptq1_0, 8,     dequantize_ptq1_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_0,    2,     dequantize_q4_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_1,    2,     dequantize_q4_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_0,    2,     dequantize_q5_0,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_1_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_1,    2,     dequantize_q5_1,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q8_0_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q8_0,    2,     dequantize_q8_0,    float,  float4x4,  half, half2x4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<bfloat, bfloat4x4, simdgroup_bfloat8x8, bfloat, bfloat2x4, simdgroup_bfloat8x8, block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#else
template [[host_name("kernel_mul_mm_id_mxfp4_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_mxfp4,   2,     dequantize_mxfp4,   float,  float4x4,  half, half2x4>;
#endif
template [[host_name("kernel_mul_mm_id_q2_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q2_K,    QK_NL, dequantize_q2_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q3_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q3_K,    QK_NL, dequantize_q3_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q4_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q4_K,    QK_NL, dequantize_q4_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q5_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q5_K,    QK_NL, dequantize_q5_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_q6_K_f16")]]    kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_q6_K,    QK_NL, dequantize_q6_K,    float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xxs, QK_NL, dequantize_iq2_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_xs,  QK_NL, dequantize_iq2_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_xxs_f16")]] kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_xxs, QK_NL, dequantize_iq3_xxs, float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq3_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq3_s,   QK_NL, dequantize_iq3_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq2_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq2_s,   QK_NL, dequantize_iq2_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_s_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_s,   QK_NL, dequantize_iq1_s,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq1_m_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq1_m,   QK_NL, dequantize_iq1_m,   float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_nl_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_nl,  2,     dequantize_iq4_nl,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_iq4_xs_f16")]]  kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_iq4_xs,  QK_NL, dequantize_iq4_xs,  float,  float4x4,  half, half2x4>;
template [[host_name("kernel_mul_mm_id_tq2_0_f16")]]   kernel mul_mm_id kernel_mul_mm_id<half,   half4x4,   simdgroup_half8x8,   half,   half2x4,   simdgroup_half8x8,   block_tq2_0,   QK_NL, dequantize_tq2_0,   float,  float4x4,  half, half2x4>;

#if defined(GGML_METAL_HAS_BF16)
#endif


// dst[o][t] = sum_k src0[o][k] src1[t][k] for a short k (the LoRA B times A x), below the GEMM's k >= 64: each thread
// keeps its src0 row in registers and makes 8 outputs of it, with the rows along the threads so the stores coalesce
template<short KMAX>
kernel void kernel_mul_mm_lowk_f32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  float * sx [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    const int K  = args.ne00;
    const int o  = tgpig.x*64 + (tiitg & 63);
    const int t0 = tgpig.y*32;
    const int im = tgpig.z;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    // the 32 columns of src1, k-major: sx[k*33 + j]
    for (int e = tiitg; e < 32*KMAX; e += 256) {
        const int j = e / KMAX;
        const int k = e % KMAX;
        const int t = t0 + j;
        sx[k*33 + j] = t < args.ne1 && k < K ?
            *((device const float *) (src1 + i13*args.nb13 + i12*args.nb12 + (uint64_t) t*args.nb11) + k) : 0.0f;
    }

    float w[KMAX];
    if (o < args.ne0) {
        device const float * wr = (device const float *) (src0 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03 + (uint64_t) o*args.nb01);
        FOR_UNROLL (short k = 0; k < KMAX; ++k) {
            w[k] = k < K ? wr[k] : 0.0f;
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (o >= args.ne0) {
        return;
    }

    device float * d = (device float *) dst + (uint64_t) im*args.ne1*args.ne0 + o;
    const short jb = (tiitg >> 6)*8;
    FOR_UNROLL (short j = 0; j < 8; ++j) {
        const int t = t0 + jb + j;
        float acc = 0.0f;
        FOR_UNROLL (short k = 0; k < KMAX; ++k) {
            acc += w[k]*sx[k*33 + jb + j];
        }
        if (t < args.ne1) {
            d[(uint64_t) t*args.ne0] = acc;
        }
    }
}

typedef decltype(kernel_mul_mm_lowk_f32<16>) kernel_mul_mm_lowk_t;

template [[host_name("kernel_mul_mm_lowk_f32_16")]] kernel kernel_mul_mm_lowk_t kernel_mul_mm_lowk_f32<16>;
template [[host_name("kernel_mul_mm_lowk_f32_32")]] kernel kernel_mul_mm_lowk_t kernel_mul_mm_lowk_f32<32>;

// the scale of each src1 column for the half GEMM of out_prod: the power of two that brings its largest value into
// (0.5, 1], stored after dst
kernel void kernel_out_prod_colscale(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src1,
        device       char * dst,
        threadgroup  float * sm [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int j   = tgpig.x;
    const int im  = tgpig.y;
    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    device const float * col = (device const float *) (src1 + i13*args.nb13 + i12*args.nb12 + (uint64_t) j*args.nb11);

    float m = 0.0f;
    for (int k = tiitg; k < args.ne00; k += 256) {
        m = max(m, fabs(col[k]));
    }
    m = simd_max(m);
    if (tiisg == 0) {
        sm[sgitg] = m;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tiitg == 0) {
        float a = 0.0f;
        for (short i = 0; i < 8; ++i) {
            a = max(a, sm[i]);
        }
        const float e = a > 0.0f ? clamp(ceil(log2(a)), -120.0f, 120.0f) : 0.0f;
        // ne02 carries the number of output matrices here
        const uint64_t n_out = (uint64_t) args.ne0*args.ne1*args.ne02;
        ((device float *) dst)[n_out + (uint64_t) im*args.ne1 + j] = exp2(-e);
    }
}

// dst[m][t] = sum_k src0[m][k] src1[t][k] for few rows m and a long k (A x of a LoRA): a GEMM tile would stay mostly
// empty, so a threadgroup takes a few columns and splits k over its threads, which reduce at the end
template<short MMAX, short NC>
kernel void kernel_mul_mm_skinny_f32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  float * sred [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    const int K  = args.ne00;
    const int M  = args.ne0;
    const int t0 = tgpig.x*NC;
    const int im = tgpig.z;

    const int i12 = im % args.ne12;
    const int i13 = im / args.ne12;

    device const char * w0 = src0 + (i12/args.r2)*args.nb02 + (i13/args.r3)*args.nb03;
    device const char * x0 = src1 + i13*args.nb13 + i12*args.nb12;

    float acc[MMAX][NC];
    FOR_UNROLL (short m = 0; m < MMAX; ++m) {
        FOR_UNROLL (short c = 0; c < NC; ++c) {
            acc[m][c] = 0.0f;
        }
    }

    for (int k = tiitg; k < K; k += 256) {
        float xv[NC];
        FOR_UNROLL (short c = 0; c < NC; ++c) {
            xv[c] = t0 + c < args.ne1 ? ((device const float *) (x0 + (uint64_t) (t0 + c)*args.nb11))[k] : 0.0f;
        }
        FOR_UNROLL (short m = 0; m < MMAX; ++m) {
            const float w = m < M ? ((device const float *) (w0 + (uint64_t) m*args.nb01))[k] : 0.0f;
            FOR_UNROLL (short c = 0; c < NC; ++c) {
                acc[m][c] += w*xv[c];
            }
        }
    }

    FOR_UNROLL (short m = 0; m < MMAX; ++m) {
        FOR_UNROLL (short c = 0; c < NC; ++c) {
            const float s = simd_sum(acc[m][c]);
            if (tiisg == 0) {
                sred[sgitg*(MMAX*NC) + m*NC + c] = s;
            }
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (tiitg < MMAX*NC) {
        const short m = tiitg/NC;
        const short c = tiitg%NC;
        float s = 0.0f;
        FOR_UNROLL (short g = 0; g < 8; ++g) {
            s += sred[g*(MMAX*NC) + tiitg];
        }
        if (m < M && t0 + c < args.ne1) {
            ((device float *) dst)[(uint64_t) im*args.ne1*args.ne0 + (uint64_t) (t0 + c)*args.ne0 + m] = s;
        }
    }
}

typedef decltype(kernel_mul_mm_skinny_f32<16, 4>) kernel_mul_mm_skinny_t;

template [[host_name("kernel_mul_mm_skinny_f32_16_2")]] kernel kernel_mul_mm_skinny_t kernel_mul_mm_skinny_f32<16, 2>;
template [[host_name("kernel_mul_mm_skinny_f32_16_4")]] kernel kernel_mul_mm_skinny_t kernel_mul_mm_skinny_f32<16, 4>;
template [[host_name("kernel_mul_mm_skinny_f32_16_8")]] kernel kernel_mul_mm_skinny_t kernel_mul_mm_skinny_f32<16, 8>;

// out_prod(a, b)[i][j] = sum_k a[i][k] b[j][k] when one side (a if SMALL_A) has at most SMAX rows, as the gradient of a
// LoRA factor: a thread owns a row of the long side and keeps the short side's sums in registers, k split in four
template<short SMAX, bool SMALL_A>
kernel void kernel_out_prod_small_f32(
        constant ggml_metal_kargs_mul_mm & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  float * ss [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]]) {
    // args: ne00 = k, ne0/ne1 = rows of a/b, nb01/nb11 = k stride of a/b
    const int K  = args.ne00;
    const int ns = SMALL_A ? args.ne0 : args.ne1;
    const int nl = SMALL_A ? args.ne1 : args.ne0;
    const short r = tiitg & 63;
    const short q = tiitg >> 6;
    const int   l = tgpig.x*64 + r;

    device const char * sp = SMALL_A ? src0 : src1;
    device const char * lp = SMALL_A ? src1 : src0;
    const uint64_t snb = SMALL_A ? args.nb01 : args.nb11;
    const uint64_t lnb = SMALL_A ? args.nb11 : args.nb01;

    threadgroup float * sq = ss + q*32*SMAX;

    float acc[SMAX];
    FOR_UNROLL (short s = 0; s < SMAX; ++s) {
        acc[s] = 0.0f;
    }

    for (int k0 = 32*q; k0 < K + 96; k0 += 128) {
        // the short side's values at the 32 k of this group
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (int e = r; e < 32*SMAX; e += 64) {
            const int kk = e / SMAX;
            const int s  = e % SMAX;
            sq[e] = s < ns && k0 + kk < K ? ((device const float *) (sp + (uint64_t) (k0 + kk)*snb))[s] : 0.0f;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (l < nl && k0 < K) {
            const int kn = min(32, K - k0);
            for (int kk = 0; kk < kn; ++kk) {
                const float v = ((device const float *) (lp + (uint64_t) (k0 + kk)*lnb))[l];
                FOR_UNROLL (short s = 0; s < SMAX; ++s) {
                    acc[s] += v*sq[kk*SMAX + s];
                }
            }
        }
    }

    // the four groups' sums
    threadgroup_barrier(mem_flags::mem_threadgroup);
    FOR_UNROLL (short s = 0; s < SMAX; ++s) {
        ss[(q*SMAX + s)*64 + r] = acc[s];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    if (q == 0 && l < nl) {
        device float * d = (device float *) dst;
        FOR_UNROLL (short s = 0; s < SMAX; ++s) {
            if (s < ns) {
                const float v = ss[s*64 + r] + ss[(SMAX + s)*64 + r] + ss[(2*SMAX + s)*64 + r] + ss[(3*SMAX + s)*64 + r];
                // dst is [ne0, ne1]: (i, j) at j*ne0 + i
                if (SMALL_A) {
                    d[(uint64_t) l*args.ne0 + s] = v;
                } else {
                    d[(uint64_t) s*args.ne0 + l] = v;
                }
            }
        }
    }
}

typedef decltype(kernel_out_prod_small_f32<16, true>) kernel_out_prod_small_t;

template [[host_name("kernel_out_prod_small_f32_16_a")]] kernel kernel_out_prod_small_t kernel_out_prod_small_f32<16, true>;
template [[host_name("kernel_out_prod_small_f32_16_b")]] kernel kernel_out_prod_small_t kernel_out_prod_small_f32<16, false>;
template [[host_name("kernel_out_prod_small_f32_32_a")]] kernel kernel_out_prod_small_t kernel_out_prod_small_f32<32, true>;
template [[host_name("kernel_out_prod_small_f32_32_b")]] kernel kernel_out_prod_small_t kernel_out_prod_small_f32<32, false>;

// VALU blocked matmul for IQ3_S on AMD RDNA2 (no simdgroup_mm there, so a
// dense prefill would otherwise run as per-row mul_mv dispatches re-reading
// the weights once per row).
//
// C[N][M] = A[N][K] x B[K][M], A rows are IQ3_S blocks, B/D are F32.
// Tile 64(N) x 64(M), K step 32, 256 threads/TG, shared 18KB
// (8KB dequant A + 8KB B + 2KB grid). Gate requires K % 256 == 0.
struct mmq_amd_kargs {
    ggml_metal_kargs_mul_mm base;
    int32_t ne12;
    int32_t ne13;
    int32_t r2;
    int32_t r3;
};

kernel void kernel_mul_mm_iq3_s_f32_amd(
        constant mmq_amd_kargs & args_,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constant ggml_metal_kargs_mul_mm & args = args_.base;
    const int K = args.ne00;
    const int N = args.ne0;
    const int M = args.ne1;

    const int im = tgpig.z;
    const int i12 = im % args_.ne12;
    const int i13 = im / args_.ne12;
    const uint64_t offA = (i12/args_.r2)*args.nb02 + (i13/args_.r3)*args.nb03;
    device const char * srcB = src1 + args.nb12*i12 + args.nb13*i13;
    device float * dstB = (device float *)(dst + (uint64_t)im * (uint64_t)N * (uint64_t)M * sizeof(float));

    const int n0 = tgpig.x * 64;
    const int m0 = tgpig.y * 64;

    threadgroup float *    sA    = (threadgroup float *)shmem;
    threadgroup float *    sB    = (threadgroup float *)(shmem + 64*32*sizeof(float));
    threadgroup uint32_t * sgrid = (threadgroup uint32_t *)(shmem + 2*64*32*sizeof(float));

    for (int i = tiitg; i < 512; i += 256) {
        sgrid[i] = iq3s_grid[i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // 16 outputs per thread: 4 N x 4 M
    const int tn = tiitg % 16;
    const int tm = tiitg / 16;
    float acc[16];
    for (int i = 0; i < 16; ++i) {
        acc[i] = 0.f;
    }

    // dequant/load rows: 64 rows x 8 elems per thread
    const int  r = tiitg / 4;
    const int  q = tiitg % 4;

    for (int k0 = 0; k0 < K; k0 += 32) {
        // --- dequant A tile: row r, K-coords [k0+q*8, +8) ---
        {
            const int e0 = k0 + q*8;
            const int b   = e0 / 256;
            const int sub = (e0 % 256) / 32;
            const int l0  = 2*q;
            const int n = n0 + r;
            if (n < N) {
                device const block_iq3_s * xr =
                    (device const block_iq3_s *)(src0 + args.nb01*(uint64_t)n + offA) + b;
                device const half * dh = (device const half *)&xr->d;
                const float d = float(dh[0]);
                const uint8_t scb = xr->scales[sub/2];
                const float dsc = d * (1.f + 2.f * (float)((scb >> (4*(sub%2))) & 0xf));
                const uint8_t qh = xr->qh[sub];
                const uint8_t sbyte = xr->signs[4*sub + q];
                for (int jj = 0; jj < 2; ++jj) {
                    const int l = l0 + jj;
                    const uint8_t qs = xr->qs[8*sub + l];
                    // grid-select bit uses the l_iter index (q .. q, one per
                    // 8-elem half): bit (2*q+jj), not the qs index l = 2*q+jj
                    // scaled by 2.
                    const uint32_t g = sgrid[qs | (((qh >> (2*q + jj)) & 1) << 8)];
                    for (int j = 0; j < 4; ++j) {
                        const float gb = (float)((g >> (8*j)) & 0xff);
                        const float s = ((sbyte >> (j + 4*jj)) & 1) ? -1.f : 1.f;
                        sA[r*32 + q*8 + jj*4 + j] = dsc * gb * s;
                    }
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    sA[r*32 + q*8 + j] = 0.f;
                }
            }
        }
        // --- load B tile: sB[m][k], same thread decomposition ---
        {
            const int m = m0 + r;
            if (m < M) {
                device const char * Brow = srcB + args.nb11*(uint64_t)m;
                for (int j = 0; j < 8; ++j) {
                    sB[r*32 + q*8 + j] =
                        *(device const float *)(Brow + args.nb10*(uint64_t)(k0 + q*8 + j));
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    sB[r*32 + q*8 + j] = 0.f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 32; ++kk) {
            for (int a = 0; a < 4; ++a) {
                const float av0 = sA[(tn*4+a)*32 + kk];
                for (int b = 0; b < 4; ++b) {
                    acc[a*4+b] += av0 * sB[(tm*4+b)*32 + kk];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            const int n = n0 + tn*4 + a;
            const int m = m0 + tm*4 + b;
            if (n < N && m < M) {
                dstB[(uint64_t)n + (uint64_t)m * (uint64_t)N] = acc[a*4+b];
            }
        }
    }
}
// VALU blocked matmul for Q4_K on AMD RDNA2 (no simdgroup_mm there, so a
// dense prefill would otherwise run as per-row mul_mv dispatches re-reading
// the weights once per row).
//
// C[N][M] = A[N][K] x B[K][M], A rows are Q4_K blocks, B/D are F32.
// Tile 64(N) x 64(M), K step 32, 256 threads/TG, shared 16KB
// (8KB dequant A + 8KB B; no grid tables needed). Gate requires K % 256 == 0.

kernel void kernel_mul_mm_q4_K_f32_amd(
        constant mmq_amd_kargs & args_,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    constant ggml_metal_kargs_mul_mm & args = args_.base;
    const int K = args.ne00;
    const int N = args.ne0;
    const int M = args.ne1;

    const int im = tgpig.z;
    const int i12 = im % args_.ne12;
    const int i13 = im / args_.ne12;
    const uint64_t offA = (i12/args_.r2)*args.nb02 + (i13/args_.r3)*args.nb03;
    device const char * srcB = src1 + args.nb12*i12 + args.nb13*i13;
    device float * dstB = (device float *)(dst + (uint64_t)im * (uint64_t)N * (uint64_t)M * sizeof(float));

    const int n0 = tgpig.x * 64;
    const int m0 = tgpig.y * 64;

    threadgroup float *    sA    = (threadgroup float *)shmem;
    threadgroup float *    sB    = (threadgroup float *)(shmem + 64*32*sizeof(float));

    // 16 outputs per thread: 4 N x 4 M
    const int tn = tiitg % 16;
    const int tm = tiitg / 16;
    float acc[16];
    for (int i = 0; i < 16; ++i) {
        acc[i] = 0.f;
    }

    // dequant/load rows: 64 rows x 8 elems per thread
    const int  r = tiitg / 4;
    const int  q = tiitg % 4;

    for (int k0 = 0; k0 < K; k0 += 32) {
        // --- dequant A tile: row r, K-coords [k0+q*8, +8) ---
        {
            const int e0 = k0 + q*8;
            const int b   = e0 / 256;
            const int sub = (e0 % 256) / 32;
            const int l0  = 2*q; (void) l0;
            const int n = n0 + r;
            if (n < N) {
                device const block_q4_K * xr =
                    (device const block_q4_K *)(src0 + args.nb01*(uint64_t)n + offA) + b;
                device const half * dh = (device const half *)&xr->d;
                device const half * mh = (device const half *)&xr->dmin;
                const float d0 = float(dh[0]);
                const float m0 = float(mh[0]);
                // 8 elems at [e0, e0+8) stay inside one 32-half (e0 = k0+8q,
                // k0 % 32 == 0): pick its (d, m) scale pair once.
                const int e_in = (e0 % 256);
                const int j = e_in / 64;
                const int l = e_in % 64;
                const int is = 2*j + (l < 32 ? 0 : 1);
                uint8_t sc, m;
                if (is < 4) {
                    sc = xr->scales[is] & 63; m = xr->scales[is + 4] & 63;
                } else {
                    sc = (xr->scales[is+4] & 0xF) | ((xr->scales[is-4] >> 6) << 4);
                    m  = (xr->scales[is+4] >>  4) | ((xr->scales[is-0] >> 6) << 4);
                }
                const float dd = d0 * (float)sc;
                const float ml = m0 * (float)m;
                // Q4_K nibbles are NOT interleaved: low 32 elems are the low
                // nibbles of bytes 0..31, high 32 the high nibbles.
                const int P = (l < 32 ? l : l - 32);
                const int lowhalf = (l < 32) ? 1 : 0;
                for (int t = 0; t < 8; ++t) {
                    const uint8_t qb = xr->qs[32*j + P + t];
                    const uint8_t qv = lowhalf ? (qb & 0xF) : (qb >> 4);
                    sA[r*32 + q*8 + t] = dd * (float)qv - ml;
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    sA[r*32 + q*8 + j] = 0.f;
                }
            }
        }
        // --- load B tile: sB[m][k], same thread decomposition ---
        {
            const int m = m0 + r;
            if (m < M) {
                device const char * Brow = srcB + args.nb11*(uint64_t)m;
                for (int j = 0; j < 8; ++j) {
                    sB[r*32 + q*8 + j] =
                        *(device const float *)(Brow + args.nb10*(uint64_t)(k0 + q*8 + j));
                }
            } else {
                for (int j = 0; j < 8; ++j) {
                    sB[r*32 + q*8 + j] = 0.f;
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);

        for (int kk = 0; kk < 32; ++kk) {
            for (int a = 0; a < 4; ++a) {
                const float av0 = sA[(tn*4+a)*32 + kk];
                for (int b = 0; b < 4; ++b) {
                    acc[a*4+b] += av0 * sB[(tm*4+b)*32 + kk];
                }
            }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    for (int a = 0; a < 4; ++a) {
        for (int b = 0; b < 4; ++b) {
            const int n = n0 + tn*4 + a;
            const int m = m0 + tm*4 + b;
            if (n < N && m < M) {
                dstB[(uint64_t)n + (uint64_t)m * (uint64_t)N] = acc[a*4+b];
            }
        }
    }
}
