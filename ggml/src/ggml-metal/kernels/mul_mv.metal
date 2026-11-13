#include "common.h"
#include "dequantize.h"

constant bool  FC_mul_mv_align [[function_constant(FC_MUL_MV + 8)]];
// ids of a mixed expert layer mark host-run experts with -1
constant bool  FC_mul_mv_id_skip [[function_constant(FC_MUL_MV + 9)]];

// The 22-, 34-, 98-, 110- and 210-byte blocks leave every other one two bytes short of a word,
// and a dword read there is undefined on a card that requires natural alignment. The constant
// picks the pair of shorts for those cards, so every other one keeps the single read.
inline uint mv_ld_u32(device const void * p) {
    if (FC_mul_mv_align) {
        device const ushort * s = (device const ushort *) p;
        return (uint) s[0] | ((uint) s[1] << 16);
    }
    return *(device const uint *) p;
}

inline uint2 mv_ld_u32x2(device const void * p) {
    if (FC_mul_mv_align) {
        device const ushort * s = (device const ushort *) p;
        return uint2((uint) s[0] | ((uint) s[1] << 16), (uint) s[2] | ((uint) s[3] << 16));
    }
    return *(device const uint2 *) p;
}


constant short FC_mul_mv_nw [[function_constant(FC_MUL_MV + 5)]];
// Q1_0 dot product: dot = d * (2 * Σ(yl[i] where bit=1) - sumy)
inline float block_q_n_dot_y(device const block_q1_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + il / 8;
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];

    float acc = 0.0f;

    acc += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc += select(0.0f, yl[ 1], bool(b0 & 0x02));
    acc += select(0.0f, yl[ 2], bool(b0 & 0x04));
    acc += select(0.0f, yl[ 3], bool(b0 & 0x08));
    acc += select(0.0f, yl[ 4], bool(b0 & 0x10));
    acc += select(0.0f, yl[ 5], bool(b0 & 0x20));
    acc += select(0.0f, yl[ 6], bool(b0 & 0x40));
    acc += select(0.0f, yl[ 7], bool(b0 & 0x80));

    acc += select(0.0f, yl[ 8], bool(b1 & 0x01));
    acc += select(0.0f, yl[ 9], bool(b1 & 0x02));
    acc += select(0.0f, yl[10], bool(b1 & 0x04));
    acc += select(0.0f, yl[11], bool(b1 & 0x08));
    acc += select(0.0f, yl[12], bool(b1 & 0x10));
    acc += select(0.0f, yl[13], bool(b1 & 0x20));
    acc += select(0.0f, yl[14], bool(b1 & 0x40));
    acc += select(0.0f, yl[15], bool(b1 & 0x80));

    return qb_curr->d * (2.0f * acc - sumy);
}

// Q2_0 dot: d * (sum_lo(y) + 2*sum_hi(y) - sumy) via per-bit conditional adds
inline float block_q_n_dot_y(device const block_q2_0 * qb_curr, float sumy, thread float * yl, int il) {
    device const uint8_t * qs = qb_curr->qs + (il / 4);
    const uint8_t b0 = qs[0];
    const uint8_t b1 = qs[1];
    const uint8_t b2 = qs[2];
    const uint8_t b3 = qs[3];

    // Accumulate where low bit is set (bits 0,2,4,6 of each byte)
    float acc_lo = 0.0f;
    acc_lo += select(0.0f, yl[ 0], bool(b0 & 0x01));
    acc_lo += select(0.0f, yl[ 1], bool(b0 & 0x04));
    acc_lo += select(0.0f, yl[ 2], bool(b0 & 0x10));
    acc_lo += select(0.0f, yl[ 3], bool(b0 & 0x40));
    acc_lo += select(0.0f, yl[ 4], bool(b1 & 0x01));
    acc_lo += select(0.0f, yl[ 5], bool(b1 & 0x04));
    acc_lo += select(0.0f, yl[ 6], bool(b1 & 0x10));
    acc_lo += select(0.0f, yl[ 7], bool(b1 & 0x40));
    acc_lo += select(0.0f, yl[ 8], bool(b2 & 0x01));
    acc_lo += select(0.0f, yl[ 9], bool(b2 & 0x04));
    acc_lo += select(0.0f, yl[10], bool(b2 & 0x10));
    acc_lo += select(0.0f, yl[11], bool(b2 & 0x40));
    acc_lo += select(0.0f, yl[12], bool(b3 & 0x01));
    acc_lo += select(0.0f, yl[13], bool(b3 & 0x04));
    acc_lo += select(0.0f, yl[14], bool(b3 & 0x10));
    acc_lo += select(0.0f, yl[15], bool(b3 & 0x40));

    // Accumulate where high bit is set (bits 1,3,5,7 of each byte)
    float acc_hi = 0.0f;
    acc_hi += select(0.0f, yl[ 0], bool(b0 & 0x02));
    acc_hi += select(0.0f, yl[ 1], bool(b0 & 0x08));
    acc_hi += select(0.0f, yl[ 2], bool(b0 & 0x20));
    acc_hi += select(0.0f, yl[ 3], bool(b0 & 0x80));
    acc_hi += select(0.0f, yl[ 4], bool(b1 & 0x02));
    acc_hi += select(0.0f, yl[ 5], bool(b1 & 0x08));
    acc_hi += select(0.0f, yl[ 6], bool(b1 & 0x20));
    acc_hi += select(0.0f, yl[ 7], bool(b1 & 0x80));
    acc_hi += select(0.0f, yl[ 8], bool(b2 & 0x02));
    acc_hi += select(0.0f, yl[ 9], bool(b2 & 0x08));
    acc_hi += select(0.0f, yl[10], bool(b2 & 0x20));
    acc_hi += select(0.0f, yl[11], bool(b2 & 0x80));
    acc_hi += select(0.0f, yl[12], bool(b3 & 0x02));
    acc_hi += select(0.0f, yl[13], bool(b3 & 0x08));
    acc_hi += select(0.0f, yl[14], bool(b3 & 0x20));
    acc_hi += select(0.0f, yl[15], bool(b3 & 0x80));

    return qb_curr->d * (acc_lo + 2.0f * acc_hi - sumy);
}

// function for calculate inner product between half a q4_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < 8; i += 2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q4_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q4 quants begin (0 or QK4_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1] * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + 8] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + 9] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// function for calculate inner product between half a q5_0 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_0/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = (uint32_t) ((device const ushort *) qb_curr->qh)[0] |
                              ((uint32_t) ((device const ushort *) qb_curr->qh)[1] << 16);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

// function for calculate inner product between half a q5_1 block and 16 floats (yl), sumy is SUM(yl[i])
// il indicates where the q5 quants begin (0 or QK5_1/4)
// we assume that the yl's have been multiplied with the appropriate scale factor
// that corresponds to the missing bit shifts (1, 1/16, 1/256, 1/4096)
inline float block_q_n_dot_y(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = (uint32_t) ((device const ushort *) qb_curr->qh)[0] |
                              ((uint32_t) ((device const ushort *) qb_curr->qh)[1] << 16);

    for (int i = 0; i < 8; i+=2) {
        acc[0] += yl[i + 0] * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1] * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + 8] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + 9] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

// The lanes that share one block are a sweep front, so the dot products take the count as a
// template argument: HALF is how many of the block's 16 low quants this lane covers.
template<short HALF>
inline float block_q_n_dot_y_l(device const block_q4_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 1 + il/2);

    for (int i = 0; i < HALF; i += 2) {
        acc[0] += yl[i + 0]        * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1]        * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + HALF + 0] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + HALF + 1] * (qs[i / 2] & 0xF000);
    }

    return d * (sumy * -8.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

template<short HALF>
inline float block_q_n_dot_y_l(device const block_q4_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs = ((device const uint16_t *) qb_curr + 2 + il/2);

    for (int i = 0; i < HALF; i += 2) {
        acc[0] += yl[i + 0]        * (qs[i / 2] & 0x000F);
        acc[1] += yl[i + 1]        * (qs[i / 2] & 0x0F00);
        acc[2] += yl[i + HALF + 0] * (qs[i / 2] & 0x00F0);
        acc[3] += yl[i + HALF + 1] * (qs[i / 2] & 0xF000);
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}

template<short HALF>
inline float block_q_n_dot_y_l(device const block_q5_0 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 3 + il/2);
           const uint32_t   qh = (uint32_t) ((device const ushort *) qb_curr->qh)[0] |
                              ((uint32_t) ((device const ushort *) qb_curr->qh)[1] << 16);

    for (int i = 0; i < HALF; i += 2) {
        acc[0] += yl[i + 0]        * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1]        * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + HALF + 0] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + HALF + 1] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (sumy * -16.f + acc[0] + acc[1] + acc[2] + acc[3]);
}

template<short HALF>
inline float block_q_n_dot_y_l(device const block_q5_1 * qb_curr, float sumy, thread float * yl, int il) {
    float d = qb_curr->d;
    float m = qb_curr->m;

    float acc[4] = { 0.0f, 0.0f, 0.0f, 0.0f };

    device const uint16_t * qs =  ((device const uint16_t *)qb_curr + 4 + il/2);
           const uint32_t   qh = (uint32_t) ((device const ushort *) qb_curr->qh)[0] |
                              ((uint32_t) ((device const ushort *) qb_curr->qh)[1] << 16);

    for (int i = 0; i < HALF; i += 2) {
        acc[0] += yl[i + 0]        * ((qs[i / 2] & 0x000F) | ((qh >> (i+0+il        ) << 4 ) & 0x00010));
        acc[1] += yl[i + 1]        * ((qs[i / 2] & 0x0F00) | ((qh >> (i+1+il        ) << 12) & 0x01000));
        acc[2] += yl[i + HALF + 0] * ((qs[i / 2] & 0x00F0) | ((qh >> (i+0+il+QK5_0/2) << 8 ) & 0x00100));
        acc[3] += yl[i + HALF + 1] * ((qs[i / 2] & 0xF000) | ((qh >> (i+1+il+QK5_0/2) << 16) & 0x10000));
    }

    return d * (acc[0] + acc[1] + acc[2] + acc[3]) + sumy * m;
}


template<short NR0>
static inline void helper_mv_reduce_and_write(
        device float * dst_f32,
        float sumf[NR0],
        const int r0,
        const int ne01,
        ushort tiisg,
        ushort sgitg,
        threadgroup char * shmem) {
    const short NW = FC_mul_mv_nw;

    threadgroup float * shmem_f32[NR0];

    for (short row = 0; row < NR0; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR0 && r0 + row < ne01; ++row) {
        float tot = simd_sum(shmem_f32[row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = tot;
        }
    }
}

constant short FC_mul_mv_nsg   [[function_constant(FC_MUL_MV + 0)]];
constant short FC_mul_mv_nxpsg [[function_constant(FC_MUL_MV + 1)]];
constant short FC_mul_mv_ne12  [[function_constant(FC_MUL_MV + 2)]];
constant short FC_mul_mv_r2    [[function_constant(FC_MUL_MV + 3)]];
constant short FC_mul_mv_r3    [[function_constant(FC_MUL_MV + 4)]];
constant bool  FC_mul_mv_row4  [[function_constant(FC_MUL_MV + 6)]];
constant bool FC_mul_mv_row8 [[function_constant(FC_MUL_MV + 7)]];

template<typename block_q_type, short NR0, typename args_t>
void mul_vec_q_n_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    // 2 threads per block-half (the packing needs il=0/8); blocks-in-flight
    // scale with the simd width. wave32: NQ=16, identical to upstream.
    const short NW = FC_mul_mv_nw;
    const short NQ = NW/2;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
  //const int r0 =  tgpig.x*NR0;
    const int r1 =  tgpig.y;
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q_type * x = (device const block_q_type *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    //const int ib0 = sgitg*NQ + ix;
    const int ib0 = ix;

    float yl[16]; // src1 vector cache

    //device const float * yb = y + ix*QK4_0 + il;
    device const float * yb = y + ib0*QK4_0 + il;

    // each thread in a SIMD group deals with half a block.
    //for (int ib = ib0; ib < nb; ib += NSG*NQ) {
    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * NQ;
        //yb += NSG*NQ*QK4_0;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    //helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);

    for (int row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}

// a pair of columns per pass: each block's quants are read once for both, and the rows'
// pointers and the column loads stay as in the one-column kernel
template<typename block_q_type, short NR0, short NC>
void mul_vec_q_n_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW = FC_mul_mv_nw;
    const short NQ = NW/2;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int r1 =  tgpig.y*NC;
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    // a column past the end reads the last one and is not written
    device const float * yb[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yb[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix*QK4_0 + il;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib = ix; ib < nb; ib += NQ) {
        float yl[NC][16];
        float sy[NC];

        FOR_UNROLL (short c = 0; c < NC; c++) {
            float sumy[2] = { 0.f, 0.f };
            FOR_UNROLL (short i = 0; i < 8; i += 2) {
                sumy[0]  += yb[c][i +  0] + yb[c][i +  1];
                yl[c][i + 0] = yb[c][i +  0];
                yl[c][i + 1] = yb[c][i +  1]/256.f;

                sumy[1]  += yb[c][i + 16] + yb[c][i + 17];
                yl[c][i + 8] = yb[c][i + 16]/16.f;
                yl[c][i + 9] = yb[c][i + 17]/4096.f;
            }
            sy[c] = sumy[0] + sumy[1];
            yb[c] += QK4_0 * NQ;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            FOR_UNROLL (short c = 0; c < NC; c++) {
                sumf[row][c] += block_q_n_dot_y(ax[row] + ib, sy[c], yl[c], il);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + (r1 + c)*args.ne0;
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row][c]);
            if (tiisg == 0 && r0 + row < args.ne01) {
                dst_f32[r0 + row] = tot;
            }
        }
    }
}

#define QN_MV_NC(T, NR0) \
[[host_name("kernel_mul_mv_" #T "_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_##T##_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    mul_vec_q_n_f32_nc_impl<block_##T, NR0, 2>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

QN_MV_NC(q4_0, 2)
QN_MV_NC(q4_0, 4)
QN_MV_NC(q4_0, 8)
QN_MV_NC(q4_1, 2)
QN_MV_NC(q4_1, 4)
QN_MV_NC(q4_1, 8)
QN_MV_NC(q5_0, 2)
QN_MV_NC(q5_0, 4)
QN_MV_NC(q5_0, 8)
QN_MV_NC(q5_1, 2)
QN_MV_NC(q5_1, 4)
QN_MV_NC(q5_1, 8)

// Same matvec with the lanes per block as a template argument, so the sweep can move the
// front without a rebuild. LANES lanes cooperate on one block: LANES=2 is what ships.
template<typename block_q_type, short NR0, short LANES, typename args_t>
void mul_vec_q_n_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const short NW   = FC_mul_mv_nw;
    const short NQ   = NW/LANES;
    const short HALF = 16/LANES;

    // Below two the pair loop writes past yl: the block's quants come in pairs, so a lane
    // that covers one value has nowhere to put the second half.
    static_assert(16/LANES >= 2, "LANES over 8 leaves half a pair per lane");

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int r1 =  tgpig.y;
    const int im =  tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q_type * ax[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q_type *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = {0.f};

    const short ix = tiisg/LANES;
    const short il = (tiisg%LANES)*HALF;

    const int ib0 = ix;

    float yl[2*HALF];

    device const float * yb = y + ib0*QK4_0 + il;

    for (int ib = ib0; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < HALF; i += 2) {
            sumy[0]           += yb[i +  0] + yb[i +  1];
            yl[i + 0]          = yb[i +  0];
            yl[i + 1]          = yb[i +  1]/256.f;

            sumy[1]           += yb[i + 16] + yb[i + 17];
            yl[i + HALF + 0]   = yb[i + 16]/16.f;
            yl[i + HALF + 1]   = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumf[row] += block_q_n_dot_y_l<HALF>(ax[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * NQ;
    }

    device float * dst_f32 = (device float *) dst + im*args.ne0*args.ne1 + r1*args.ne0;

    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = tot;
        }
    }
}


template<int nr0, typename args_t>
void kernel_mul_mv_q1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q1_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q1_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    const short ix = (tiisg/8);
    const short il = (tiisg%8)*16;

    device const float * yb = y + ix*QK1_0 + il;

    for (int ib = ix; ib < nb; ib += NW/8) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK1_0 * (NW/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q1_0_f32")]]
kernel void kernel_mul_mv_q1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_q2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q2_0 *) ((device char *) src0 + offset0);
    }

    float yl[16];
    float sumf[nr0] = {0.f};

    // group 64: 4 sub-blocks of 16 weights per Q2_0 block
    const short ix = (tiisg/4);
    const short il = (tiisg%4)*16;

    device const float * yb = y + ix*QK2_0 + il;

    for (int ib = ix; ib < nb; ib += NW/4) {
        float sumy = 0.f;

        FOR_UNROLL (short i = 0; i < 16; i++) {
            yl[i] = yb[i];
            sumy += yb[i];
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            sumf[row] += block_q_n_dot_y(ax[row] + ib, sumy, yl, il);
        }

        yb += QK2_0 * (NW/4);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q2_0_f32")]]
kernel void kernel_mul_mv_q2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// PQ2_0: eight lanes per 128-weight block, sixteen weights per lane. Masking a word with
// 0x03030303 isolates one 2-bit field in each of its four bytes, so the weights reach the
// float pipe as byte conversions and the dot is d*(sum q*y - sum y).
template<int nr0, typename args_t>
void kernel_mul_mv_pq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK_PQ2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    // 32-bit offsets from src0: a weight tensor stays below 4 GiB
    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    }

    // yl[f] holds the activations of field f of the four bytes: elements f, f+4, f+8, f+12
    float4 yl[4];
    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short il = tiisg%8;

    device const float4 * yb = (device const float4 *) (y + ix*QK_PQ2_0 + 16*il);

    for (int ib = ix; ib < nb; ib += NW/8) {
        const float4 y0 = yb[0];
        const float4 y1 = yb[1];
        const float4 y2 = yb[2];
        const float4 y3 = yb[3];

        yl[0] = float4(y0[0], y1[0], y2[0], y3[0]);
        yl[1] = float4(y0[1], y1[1], y2[1], y3[1]);
        yl[2] = float4(y0[2], y1[2], y2[2], y3[2]);
        yl[3] = float4(y0[3], y1[3], y2[3], y3[3]);

        const float sumy = dot(y0 + y1 + y2 + y3, float4(1.0f));

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_pq2_0);
            const uint w = mv_ld_u32(src0 + o + 2 + 4*il);

            float acc = 0.f;
            FOR_UNROLL (short f = 0; f < 4; f++) {
                const float4 q = float4(as_type<uchar4>((w >> (2*f)) & 0x03030303u));
                acc += dot(q, yl[f]);
            }

            sumf[row] += (float) *(device const half *) (src0 + o) * (acc - sumy);
        }

        yb += QK_PQ2_0 * (NW/8) / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// PQ2_0 for 64 lanes, bound by instructions there: the weights enter as halves and meet the
// activation in half products with float accumulation, with no byte conversions
template<int nr0>
void kernel_mul_mv_pq2_0_f32_h_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK_PQ2_0;
    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * nr0;
    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;
    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;
    device const float * y = (device const float *) (src1 + offset1);

    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    }

    // yh[s] pairs elements s and s + 8 of the lane's sixteen, as the weight word pairs them
    half2 yh[8];
    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short il = tiisg%8;

    device const float4 * yb = (device const float4 *) (y + ix*QK_PQ2_0 + 16*il);

    for (int ib = ix; ib < nb; ib += NW/8) {
        const float4 y0 = yb[0];
        const float4 y1 = yb[1];
        const float4 y2 = yb[2];
        const float4 y3 = yb[3];
        yh[0] = half2(y0[0], y2[0]); yh[1] = half2(y0[1], y2[1]); yh[2] = half2(y0[2], y2[2]); yh[3] = half2(y0[3], y2[3]);
        yh[4] = half2(y1[0], y3[0]); yh[5] = half2(y1[1], y3[1]); yh[6] = half2(y1[2], y3[2]); yh[7] = half2(y1[3], y3[3]);

        // field s of each half sits at bit 2s; with the exponent of 2^(10-2s) the half reads
        // 2^(10-2s) + q exactly. Fields 5..7 overlap the exponent and are shifted down first.
        float bias = 0.f;
        FOR_UNROLL (short s = 0; s < 8; s++) {
            const float b = (float) (1 << (10 - 2*(s < 5 ? s : s - 5)));
            bias += (b + 1.0f)*((float) yh[s][0] + (float) yh[s][1]);
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_pq2_0);
            const uint w  = mv_ld_u32(src0 + o + 2 + 4*il);
            const uint wh = w >> 10;

            float acc = 0.f;
            FOR_UNROLL (short s = 0; s < 8; s++) {
                const short t = s < 5 ? s : s - 5;
                const uint  e = (uint) (25 - 2*t) << 10;
                const half2 q = as_type<half2>(((s < 5 ? w : wh) & (0x00030003u << (2*t))) | (e | (e << 16)));
                acc = fma((float) q[0], (float) yh[s][0], acc);
                acc = fma((float) q[1], (float) yh[s][1], acc);
            }

            sumf[row] += (float) *(device const half *) (src0 + o) * (acc - bias);
        }

        yb += QK_PQ2_0 * (NW/8) / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;
    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_pq2_0_f32_h")]]
kernel void kernel_mul_mv_pq2_0_f32_h(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_pq2_0_f32_h_impl<N_R0_PQ2_0_H>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_pq2_0_f32")]]
kernel void kernel_mul_mv_pq2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_pq2_0_f32_impl<N_R0_PQ2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// PQ2_0 gate and up projections of one token in a single pass, written as silu(gate)*up: both
// matrices share the staged activation and the two outputs never reach memory. HALF picks the
// products of the 64-lane mat-vec.
template<int nr0, bool HALF>
void kernel_mul_mv_pq2_0_swiglu_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK_PQ2_0;
    const int first_row = (tgpig.x * NSG + sgitg) * nr0;

    device const float * y = (device const float *) src1;

    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01);
    }

    float4 yl[4];
    half2  yh[8];
    float  sumg[nr0] = {0.f};
    float  sumu[nr0] = {0.f};

    const short ix = tiisg/8;
    const short il = tiisg%8;

    device const float4 * yb = (device const float4 *) (y + ix*QK_PQ2_0 + 16*il);

    for (int ib = ix; ib < nb; ib += NW/8) {
        const float4 y0 = yb[0];
        const float4 y1 = yb[1];
        const float4 y2 = yb[2];
        const float4 y3 = yb[3];

        float bias;
        if (HALF) {
            yh[0] = half2(y0[0], y2[0]); yh[1] = half2(y0[1], y2[1]); yh[2] = half2(y0[2], y2[2]); yh[3] = half2(y0[3], y2[3]);
            yh[4] = half2(y1[0], y3[0]); yh[5] = half2(y1[1], y3[1]); yh[6] = half2(y1[2], y3[2]); yh[7] = half2(y1[3], y3[3]);
            bias = 0.f;
            FOR_UNROLL (short s = 0; s < 8; s++) {
                const float b = (float) (1 << (10 - 2*(s < 5 ? s : s - 5)));
                bias += (b + 1.0f)*((float) yh[s][0] + (float) yh[s][1]);
            }
        } else {
            yl[0] = float4(y0[0], y1[0], y2[0], y3[0]);
            yl[1] = float4(y0[1], y1[1], y2[1], y3[1]);
            yl[2] = float4(y0[2], y1[2], y2[2], y3[2]);
            yl[3] = float4(y0[3], y1[3], y2[3], y3[3]);
            bias = dot(y0 + y1 + y2 + y3, float4(1.0f));
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_pq2_0);
            FOR_UNROLL (short m = 0; m < 2; m++) {
                device const char * src0 = m == 0 ? src0g : src0u;
                const uint w = mv_ld_u32(src0 + o + 2 + 4*il);

                float acc = 0.f;
                if (HALF) {
                    const uint wh = w >> 10;
                    FOR_UNROLL (short s = 0; s < 8; s++) {
                        const short t = s < 5 ? s : s - 5;
                        const uint  e = (uint) (25 - 2*t) << 10;
                        const half2 q = as_type<half2>(((s < 5 ? w : wh) & (0x00030003u << (2*t))) | (e | (e << 16)));
                        acc = fma((float) q[0], (float) yh[s][0], acc);
                        acc = fma((float) q[1], (float) yh[s][1], acc);
                    }
                } else {
                    FOR_UNROLL (short f = 0; f < 4; f++) {
                        acc += dot(float4(as_type<uchar4>((w >> (2*f)) & 0x03030303u)), yl[f]);
                    }
                }

                const float r = (float) *(device const half *) (src0 + o) * (acc - bias);
                if (m == 0) {
                    sumg[row] += r;
                } else {
                    sumu[row] += r;
                }
            }
        }

        yb += QK_PQ2_0 * (NW/8) / 4;
    }

    device float * dst_f32 = (device float *) dst;

    for (int row = 0; row < nr0; ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

[[host_name("kernel_mul_mv_pq2_0_swiglu_f32")]]
kernel void kernel_mul_mv_pq2_0_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_pq2_0_swiglu_impl<N_R0_PQ2_0, false>(args, src0g, src0u, src1, dst, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_pq2_0_swiglu_f32_h")]]
kernel void kernel_mul_mv_pq2_0_swiglu_f32_h(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_pq2_0_swiglu_impl<N_R0_PQ2_0_H, true>(args, src0g, src0u, src1, dst, tgpig, tiisg, sgitg);
}

// PQ2_0 for a few columns at once, as speculative verification batches them: every weight
// word is decoded once and reused by NC columns, so the weights stream once per NC tokens
template<int nr0, int NC>
void kernel_mul_mv_pq2_0_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    const int nb = args.ne00/QK_PQ2_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    }

    const short ix = tiisg/8;
    const short il = tiisg%8;

    // a column past the end reads the last one and is not written
    device const float4 * yb[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yb[c] = (device const float4 *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + (ix*QK_PQ2_0 + 16*il)/4;
    }

    float4 yl[NC][4];
    float  sumy[NC];
    float  sumf[nr0][NC];
    FOR_UNROLL (short row = 0; row < nr0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib = ix; ib < nb; ib += NW/8) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            const float4 y0 = yb[c][0];
            const float4 y1 = yb[c][1];
            const float4 y2 = yb[c][2];
            const float4 y3 = yb[c][3];

            yl[c][0] = float4(y0[0], y1[0], y2[0], y3[0]);
            yl[c][1] = float4(y0[1], y1[1], y2[1], y3[1]);
            yl[c][2] = float4(y0[2], y1[2], y2[2], y3[2]);
            yl[c][3] = float4(y0[3], y1[3], y2[3], y3[3]);

            sumy[c] = dot(y0 + y1 + y2 + y3, float4(1.0f));

            yb[c] += QK_PQ2_0 * (NW/8) / 4;
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_pq2_0);
            const uint w = mv_ld_u32(src0 + o + 2 + 4*il);
            const float d = *(device const half *) (src0 + o);

            float4 q[4];
            FOR_UNROLL (short f = 0; f < 4; f++) {
                q[f] = float4(as_type<uchar4>((w >> (2*f)) & 0x03030303u));
            }

            FOR_UNROLL (short c = 0; c < NC; c++) {
                const float acc = dot(q[0], yl[c][0]) + dot(q[1], yl[c][1]) + dot(q[2], yl[c][2]) + dot(q[3], yl[c][3]);
                sumf[row][c] += d * (acc - sumy[c]);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;

        for (int row = 0; row < nr0; ++row) {
            const float tot = simd_sum(sumf[row][c]);

            if (tiisg == 0 && first_row + row < args.ne01) {
                dst_f32[first_row + row] = tot;
            }
        }
    }
}

#define PQ2_0_MV_NC(NC) \
[[host_name("kernel_mul_mv_pq2_0_f32_c" #NC)]] \
kernel void kernel_mul_mv_pq2_0_f32_c##NC( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_pq2_0_f32_nc_impl<N_R0_PQ2_0_NC, NC>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

PQ2_0_MV_NC(2)
PQ2_0_MV_NC(3)
PQ2_0_MV_NC(4)

// PTQ1_0: eight lanes per block, and each lane owns whole bytes so every packed byte is read
// once: qs[2*il] and qs[2*il + 1] (five trits each), qs[16 + il] (five trits) and one trit of qh.
// A 256-entry table in threadgroup memory holds each byte value's trits as bytes, so a packed
// byte costs one table read and byte conversions instead of the base-3 arithmetic.
template<int nr0, typename args_t>
void kernel_mul_mv_ptq1_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    threadgroup uint2 * lut = (threadgroup uint2 *) shmem;
    for (short i = tiisg + NW*sgitg; i < 256; i += NW*NSG) {
        uint t[5];
        uint p = 1;
        FOR_UNROLL (short n = 0; n < 5; n++) {
            t[n] = (((i*p) & 0xFFu)*3u) >> 8;
            p *= 3;
        }
        lut[i] = uint2(t[0] | (t[1] << 8) | (t[2] << 16) | (t[3] << 24), t[4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.ne00/QK_PTQ1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 = r1*args.nb11 + (i12)*args.nb12 + (i13)*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    // 32-bit offsets from src0: a weight tensor stays below 4 GiB, and it keeps the address math
    // off the 64-bit carry chain
    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    }

    // per owned byte: trits 0..3 as a vector and trit 4 alone; then the qh activation
    float4 y4[3];
    float  y1[3];
    float  yh;
    float sumf[nr0] = {0.f};

    const short ix = tiisg/8;
    const short il = tiisg%8;

    // the qh trit this lane takes: byte il & 1, trit il >> 1
    const short sh = 8*(il >> 1);

    device const float * yb = y + ix*QK_PTQ1_0;

    for (int ib = ix; ib < nb; ib += NW/8) {
        FOR_UNROLL (short k = 0; k < 2; ++k) {
            const short m = 2*il + k;
            y4[k] = float4(yb[m], yb[16 + m], yb[32 + m], yb[48 + m]);
            y1[k] = yb[64 + m];
        }
        y4[2] = float4(yb[80 + il], yb[88 + il], yb[96 + il], yb[104 + il]);
        y1[2] = yb[112 + il];
        yh    = yb[120 + il];

        const float sumy = dot(y4[0] + y4[1] + y4[2], float4(1.0f)) + y1[0] + y1[1] + y1[2] + yh;

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_ptq1_0);

            const uint  q01 = *(device const ushort *) (src0 + o + 2*il);
            const uint2 l0 = lut[q01 & 0xFFu];
            const uint2 l1 = lut[q01 >> 8];
            const uint2 l2 = lut[(uchar) src0[o + 16 + il]];
            const uint  lh = lut[(uchar) src0[o + 24 + (il & 1)]].x;
            const float d  = *(device const half *) (src0 + o + 26);

            float acc = dot(float4(as_type<uchar4>(l0.x)), y4[0]) + (float) l0.y*y1[0];
            acc      += dot(float4(as_type<uchar4>(l1.x)), y4[1]) + (float) l1.y*y1[1];
            acc      += dot(float4(as_type<uchar4>(l2.x)), y4[2]) + (float) l2.y*y1[2];
            acc      += (float) ((lh >> sh) & 0xFFu)*yh;

            sumf[row] += d * (acc - sumy);
        }

        yb += QK_PTQ1_0 * (NW/8);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);

        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// PTQ1_0 for a few columns at once: each packed byte is looked up once and reused by NC columns
template<int nr0, int NC>
void kernel_mul_mv_ptq1_0_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;

    threadgroup uint2 * lut = (threadgroup uint2 *) shmem;
    for (short i = tiisg + NW*sgitg; i < 256; i += NW*NSG) {
        uint t[5];
        uint p = 1;
        FOR_UNROLL (short n = 0; n < 5; n++) {
            t[n] = (((i*p) & 0xFFu)*3u) >> 8;
            p *= 3;
        }
        lut[i] = uint2(t[0] | (t[1] << 8) | (t[2] << 16) | (t[3] << 24), t[4]);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int nb = args.ne00/QK_PTQ1_0;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    uint ro[nr0];
    for (int row = 0; row < nr0; ++row) {
        ro[row] = (uint) ((first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    }

    const short ix = tiisg/8;
    const short il = tiisg%8;
    const short sh = 8*(il >> 1);

    // a column past the end reads the last one and is not written
    device const float * yb[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yb[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix*QK_PTQ1_0;
    }

    float4 y4[NC][3];
    float  y1[NC][3];
    float  yh[NC];
    float  sumy[NC];
    float  sumf[nr0][NC];
    FOR_UNROLL (short row = 0; row < nr0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib = ix; ib < nb; ib += NW/8) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            device const float * y = yb[c];
            FOR_UNROLL (short k = 0; k < 2; ++k) {
                const short m = 2*il + k;
                y4[c][k] = float4(y[m], y[16 + m], y[32 + m], y[48 + m]);
                y1[c][k] = y[64 + m];
            }
            y4[c][2] = float4(y[80 + il], y[88 + il], y[96 + il], y[104 + il]);
            y1[c][2] = y[112 + il];
            yh[c]    = y[120 + il];

            sumy[c] = dot(y4[c][0] + y4[c][1] + y4[c][2], float4(1.0f)) + y1[c][0] + y1[c][1] + y1[c][2] + yh[c];

            yb[c] += QK_PTQ1_0 * (NW/8);
        }

        FOR_UNROLL (short row = 0; row < nr0; row++) {
            const uint o = ro[row] + ib*(uint) sizeof(block_ptq1_0);

            const uint  q01 = *(device const ushort *) (src0 + o + 2*il);
            const uint2 l0 = lut[q01 & 0xFFu];
            const uint2 l1 = lut[q01 >> 8];
            const uint2 l2 = lut[(uchar) src0[o + 16 + il]];
            const uint  lh = lut[(uchar) src0[o + 24 + (il & 1)]].x;
            const float d  = *(device const half *) (src0 + o + 26);

            const float4 t0 = float4(as_type<uchar4>(l0.x));
            const float4 t1 = float4(as_type<uchar4>(l1.x));
            const float4 t2 = float4(as_type<uchar4>(l2.x));
            const float3 t4 = float3(l0.y, l1.y, l2.y);
            const float  th = (float) ((lh >> sh) & 0xFFu);

            FOR_UNROLL (short c = 0; c < NC; c++) {
                const float acc = dot(t0, y4[c][0]) + dot(t1, y4[c][1]) + dot(t2, y4[c][2]) +
                                  dot(t4, float3(y1[c][0], y1[c][1], y1[c][2])) + th*yh[c];
                sumf[row][c] += d * (acc - sumy[c]);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;

        for (int row = 0; row < nr0; ++row) {
            const float tot = simd_sum(sumf[row][c]);

            if (tiisg == 0 && first_row + row < args.ne01) {
                dst_f32[first_row + row] = tot;
            }
        }
    }
}

#define PTQ1_0_MV_NC(NC) \
[[host_name("kernel_mul_mv_ptq1_0_f32_c" #NC)]] \
kernel void kernel_mul_mv_ptq1_0_f32_c##NC( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_ptq1_0_f32_nc_impl<N_R0_PTQ1_0_NC, NC>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \
}

PTQ1_0_MV_NC(2)
PTQ1_0_MV_NC(3)
PTQ1_0_MV_NC(4)

[[host_name("kernel_mul_mv_ptq1_0_f32")]]
kernel void kernel_mul_mv_ptq1_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ptq1_0_f32_impl<N_R0_PTQ1_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_ptq1_0_f32_r8")]]
kernel void kernel_mul_mv_ptq1_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_ptq1_0_f32_impl<N_R0_PTQ1_0_W64, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


kernel void kernel_mul_mv_q4_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
     mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_r8")]]
kernel void kernel_mul_mv_q5_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_1_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_r8")]]
kernel void kernel_mul_mv_q5_1_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<short NR0, typename args_t>
void kernel_mul_mv_q8_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = FC_mul_mv_nw;
    constexpr short NQ = 8;
    // Blocks processed per iteration scales with the SIMD width; lane partitioning
    // within a block stays tied to QK8_0 so wave64 reads the right quants.
    const short NBI = NW*NQ/QK8_0;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(QK8_0/NQ);
    const short il = tiisg%(QK8_0/NQ);

    const int ib0 = sgitg*NBI + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NBI) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            // the eight quants this lane wants are contiguous and the block is 34 bytes, so a
            // stride that keeps them word-aligned lets two loads replace eight
            if (FC_mul_mv_row8 && (NQ % 4) == 0) {
                FOR_UNROLL (short w = 0; w < NQ/4; ++w) {
                    const uint b = mv_ld_u32(qs + 4*w);
                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        sumq += (float)(int8_t)(b >> (8*i)) * yl[4*w + i];
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NBI*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

// a pair of columns per pass: each lane reads its eight quants once for both, and every
// simdgroup owns its rows so no threadgroup reduction is needed
template<short NR0, short NC>
void kernel_mul_mv_q8_0_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;
    constexpr short NQ = 8;
    const short NBI = NW*NQ/QK8_0;

    const int nb = args.ne00/QK8_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const int rr = min(r0 + row, (int) args.ne01 - 1);
        const uint64_t offset0 = rr*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    const short ix = tiisg/(QK8_0/NQ);
    const short il = tiisg%(QK8_0/NQ);

    // a column past the end reads the last one and is not written
    device const float * yb[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yb[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix*QK8_0 + il*NQ;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib = ix; ib < nb; ib += NBI) {
        float yl[NC][NQ];
        FOR_UNROLL (short c = 0; c < NC; c++) {
            FOR_UNROLL (short i = 0; i < NQ; ++i) {
                yl[c][i] = yb[c][i];
            }
            yb[c] += NBI*QK8_0;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;
            float q[NQ];
            FOR_UNROLL (short w = 0; w < NQ/4; ++w) {
                const uint b = mv_ld_u32(qs + 4*w);
                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    q[4*w + i] = (float)(int8_t)(b >> (8*i));
                }
            }
            const float d = ax[row][ib].d;
            FOR_UNROLL (short c = 0; c < NC; c++) {
                float sumq = 0.f;
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += q[i]*yl[c][i];
                }
                sumf[row][c] += sumq*d;
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;
        for (int row = 0; row < NR0; ++row) {
            const float tot = simd_sum(sumf[row][c]);
            if (tiisg == 0 && r0 + row < args.ne01) {
                dst_f32[r0 + row] = tot;
            }
        }
    }
}

#define Q8_0_MV_NC(NR0) \
[[host_name("kernel_mul_mv_q8_0_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_q8_0_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_q8_0_f32_nc_impl<NR0, 2>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

Q8_0_MV_NC(2)
Q8_0_MV_NC(4)

// three and four columns in one pass (64 lanes), for MTP verify windows
#define Q8_0_MV_NCN(NC, NR0) \
[[host_name("kernel_mul_mv_q8_0_f32_c" #NC "_r" #NR0)]] \
kernel void kernel_mul_mv_q8_0_f32_c##NC##_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_q8_0_f32_nc_impl<NR0, NC>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

Q8_0_MV_NCN(3, 2)
Q8_0_MV_NCN(3, 8)
Q8_0_MV_NCN(4, 2)
Q8_0_MV_NCN(4, 4)
Q8_0_MV_NC(8)

[[host_name("kernel_mul_mv_q8_0_f32")]]
kernel void kernel_mul_mv_q8_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// mat-vec kernel processing in chunks of float4
// chpb - chunks per quantization block
template<short r1ptg, typename q_t, short chpb, void (*deq_t4)(device const q_t *, short, thread float4 &) >
void kernel_mul_mv_ext_q4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 4; // chunks per thread

    // wave64: derive the row split from the real simdgroup width, not a fixed 32
    const short nypsg = (sgptg/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq = (i01 < args.ne01) ? (device const q_t *) (src0 + offset0) + tx/chpb : (device const q_t *) src0;

    device const float4 * y4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4 *) src1;
    }

    float sumf[r1ptg] = { [ 0 ... r1ptg - 1 ] = 0.0f };

    short cch = tx%chpb; // current chunk index

    for (int ich = tx; 4*ich < args.ne00; ich += chpt*nxpsg) {
        float4 lx[chpt];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
            deq_t4(xq, cch, lx[ch]);

            cch += nxpsg;
            if (cch >= chpb) {
                xq  += cch/chpb;
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                sumf[ir1] += dot(lx[ch], y4[ir1][ch*nxpsg]);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4[ir1] += chpt*nxpsg;
        }
    }

    // reduce only the threads in each row
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
        if (nxpsg >= 32) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1], 16);
        }
        if (nxpsg >= 16) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  8);
        }
        if (nxpsg >= 8) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  4);
        }
        if (nxpsg >= 4) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  2);
        }
        if (nxpsg >= 2) {
            sumf[ir1] += simd_shuffle_down(sumf[ir1],  1);
        }

        //sumf[ir1] = simd_sum(sumf[ir1]);
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            if (i01 < args.ne01) {
                dst_f32[i01] = sumf[ir1];
            }
        }
    }
}

// mat-vec kernel processing in chunks of float4x4
template<short r1ptg, short nr0, typename q_t, short chpb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &) >
void kernel_mul_mv_ext_q4x4_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpt = 1;

    const short nypsg = (sgptg/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = (tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty)*nr0;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq[nr0];

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        xq[ir0] = (i01 + ir0 < args.ne01) ? (device const q_t *) (src0 + offset0 + ir0*args.nb01) + tx/chpb : (device const q_t *) src0;
    }

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[nr0][r1ptg];

#pragma unroll(nr0)
    for (short ir0 = 0; ir0 < nr0; ++ir0) {
#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            sumf[ir0][ir1] = 0.0f;
        }
    }

    short cch = tx%chpb;

    for (int ich = tx; 16*ich < args.ne00; ich += chpt*nxpsg) {
        float4x4 lx[chpt][nr0];

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                deq_t4x4(xq[ir0], cch, lx[ch][ir0]);
            }

            cch += nxpsg;
            if (cch >= chpb) {
#pragma unroll(nr0)
                for (short ir0 = 0; ir0 < nr0; ++ir0) {
                    xq[ir0] += cch/chpb;
                }
                cch %= chpb;
            }
        }

#pragma unroll(chpt)
        for (short ch = 0; ch < chpt; ++ch) {
#pragma unroll(r1ptg)
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                const float4x4 ly = y4x4[ir1][ch*nxpsg];

#pragma unroll(nr0)
                for (short ir0 = 0; ir0 < nr0; ++ir0) {
                    sumf[ir0][ir1] +=
                        dot(lx[ch][ir0][0], ly[0]) +
                        dot(lx[ch][ir0][1], ly[1]) +
                        dot(lx[ch][ir0][2], ly[2]) +
                        dot(lx[ch][ir0][3], ly[3]);
                }
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += chpt*nxpsg;
        }
    }

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            if (nxpsg >= 32) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1], 16);
            }
            if (nxpsg >= 16) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  8);
            }
            if (nxpsg >= 8) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  4);
            }
            if (nxpsg >= 4) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  2);
            }
            if (nxpsg >= 2) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  1);
            }
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                if (i01 + ir0 < args.ne01) {
                    dst_f32[i01 + ir0] = sumf[ir0][ir1];
                }
            }
        }
    }
}

// dispatchers needed for compile-time nxpsg
// epb - elements per quantization block
template<short r1ptg, typename q_t, short epb, void (*deq_t4)(device const q_t *, short, thread float4 &)>
kernel void kernel_mul_mv_ext_q4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    kernel_mul_mv_ext_q4_f32_impl<r1ptg, q_t, epb/4, deq_t4>(args, src0, src1, dst, tgpig, tiisg, sgitg, sgptg);
}

template<short r1ptg, short nr0, typename q_t, short epb, void (*deq_t4x4)(device const q_t *, short, thread float4x4 &)>
kernel void kernel_mul_mv_ext_q4x4_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    kernel_mul_mv_ext_q4x4_f32_impl<r1ptg, nr0, q_t, epb/16, deq_t4x4>(args, src0, src1, dst, tgpig, tiisg, sgitg, sgptg);
}

typedef decltype(kernel_mul_mv_ext_q4_f32_disp  <2, block_q8_0, 32,  dequantize_q8_0_t4>) mul_mv_ext_q4_f32_t;
typedef decltype(kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q4_K, 256, dequantize_q4_K>) mul_mv_ext_q4x4_f32_t;



#if defined(GGML_METAL_HAS_BF16)
#endif















template<typename T0, typename T1, short NR0, typename args_t>
void kernel_mul_mv_t_t_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = FC_mul_mv_nw;
    constexpr short NB = 32;
    constexpr short NF = 8;
    // Blocks per iteration scales with the SIMD width; the per-block lane split (NB/NF)
    // stays fixed so wave64 partitions the 64 lanes correctly instead of overrunning.
    const short NBI = NW*NF/NB;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const T0 * x = (device const T0 *) (src0 + offset0);
    device const T1 * y = (device const T1 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0 * ax [NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const T0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NB/NF);
    const short il = tiisg%(NB/NF);

    const int ib0 = sgitg*NBI + ix;

    T1 yl[NF];

    device const T1 * yb = y + (ib0*NB + il*NF);

    for (int ib = ib0; ib < nb; ib += NSG*NBI) {
        for (short i = 0; i < NF; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T0 * xb = ax[row] + (ib*NB + il*NF);

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF; ++i) {
                sumq += xb[i] * yl[i];
            }

            sumf[row] += sumq;
        }

        yb += NSG*NBI*NB;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
      //case 1: kernel_mul_mv_t_t_impl<T0, T1, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_impl<T0, T1, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_impl<T0, T1, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_impl<T0, T1, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    }
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_disp<T0, T1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t<half, half>) mul_mv_t_t;

#if defined(GGML_METAL_HAS_BF16)
#endif

template<typename T0, typename T04, typename T1, typename T14, short NR0, typename args_t>
void kernel_mul_mv_t_t_4_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = FC_mul_mv_nw;
    constexpr short NB  = 32;
    constexpr short NF  = 16;
    constexpr short NF4 = NF/4;
    // Blocks per iteration scales with the SIMD width; the per-block lane split (NB/NF)
    // stays fixed so wave64 partitions the 64 lanes correctly instead of overrunning.
    const short NBI = NW*NF/NB;

    const int nb = args.ne00/NB;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const T1  * y  = (device const T1  *) (src1 + offset1);
    device const T14 * y4 = (device const T14 *) (src1 + offset1);

    // pointers to src0 rows
    device const T0  * ax [NR0];
    device const T04 * ax4[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax [row] = (device const T0  *) ((device char *) src0 + offset0);
        ax4[row] = (device const T04 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(NB/NF);
    const short il = tiisg%(NB/NF);

    const int ib0 = sgitg*NBI + ix;

    T14 yl4[NF4];

    device const T14 * yb4 = y4 + (ib0*NB + il*NF)/4;

    for (int ib = ib0; ib < nb; ib += NSG*NBI) {
        for (short i = 0; i < NF4; ++i) {
            yl4[i] = yb4[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const T04 * xb4 = ax4[row] + (ib*NB + il*NF)/4;

            float sumq = 0.f;
            FOR_UNROLL (short i = 0; i < NF4; ++i) {
                sumq += dot(float4(xb4[i]), float4(yl4[i]));
            }

            sumf[row] += sumq;
        }

        yb4 += NSG*NBI*NB/4;
    }

    for (int i = nb*NB + sgitg*NW + tiisg; i < args.ne00; i += NW*NSG) {
        for (short row = 0; row < NR0; row++) {
            sumf[row] += ax[row][i] * y[i];
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}

template<typename T0, typename T04, typename T1, typename T14, typename args_t>
void kernel_mul_mv_t_t_4_disp(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    switch (args.nr0) {
      //case 1: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 1, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
        case 2: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 2, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 3: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 3, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
      //case 4: kernel_mul_mv_t_t_4_impl<T0, T04, T1, T14, 4, args_t>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); break;
    };
}

template<typename T0, typename T04, typename T1, typename T14>
kernel void kernel_mul_mv_t_t_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_t_t_4_disp<T0, T04, T1, T14, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(kernel_mul_mv_t_t_4<half, half4, half, half4>) mul_mv_t_t_4;

#if defined(GGML_METAL_HAS_BF16)
#endif

template<typename T0, typename T1, typename args_t>
void kernel_mul_mv_t_t_short_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg) {
    const int r0 = tgpig.x*32 + tiisg;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    if (r0 >= args.ne01) {
        return;
    }

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const T0 * x = (device const T0 *) (src0 + offset0);

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    const uint64_t offset1 = r1*args.nb11 + (i12   )*args.nb12 + (i13   )*args.nb13;

    device const T1 * y = (device const T1 *) (src1 + offset1);

    float res = 0.0f;

    for (int i = 0; i < args.ne00; ++i) {
        res += (float) x[i] * (float) y[i];
    }

    dst_f32[(uint64_t)r1*args.ne0 + r0] = res;
}

template<typename T0, typename T1>
kernel void kernel_mul_mv_t_t_short(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    kernel_mul_mv_t_t_short_impl<T0, T1, constant ggml_metal_kargs_mul_mv &>(
        args,
        src0,
        src1,
        dst,
        tgpig,
        tiisg);
}

typedef decltype(kernel_mul_mv_t_t_short<half, half>) mul_mv_t_t_short_t;

#if defined(GGML_METAL_HAS_BF16)
#endif

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    // block groups by real simd width: 4 on wave32, 8 on wave64
    const short nbi = FC_mul_mv_nw/8;

    const short ix = tiisg/8;  // wave32: 0...3, wave64: 0...7
    const short it = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3
    const short is = (8*ir)/16;// 0 or 1

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    for (int ib = ix; ib < nb; ib += nbi) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+32]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+64]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+96]; sumy[3] += yl[i+24];
        }

        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + (8 * (int) iq + (int) is);
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + (16 * (int) iq + 4 * (int) ir);
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            // the four shorts this lane reads are contiguous, so an aligned stride turns
            // them into two words
            ushort qw[4];
            if (FC_mul_mv_row8) {
                const uint2 t = mv_ld_u32x2(qs);
                qw[0] = (ushort)(t.x); qw[1] = (ushort)(t.x >> 16);
                qw[2] = (ushort)(t.y); qw[3] = (ushort)(t.y >> 16);
            } else {
                qw[0] = qs[0]; qw[1] = qs[1]; qw[2] = qs[2]; qw[3] = qs[3];
            }
            for (int i = 0; i < 8; i += 2) {
                acc1[0] += yl[i+ 0] * (qw[i/2] & 0x0003);
                acc2[0] += yl[i+ 1] * (qw[i/2] & 0x0300);
                acc1[1] += yl[i+ 8] * (qw[i/2] & 0x000c);
                acc2[1] += yl[i+ 9] * (qw[i/2] & 0x0c00);
                acc1[2] += yl[i+16] * (qw[i/2] & 0x0030);
                acc2[2] += yl[i+17] * (qw[i/2] & 0x3000);
                acc1[3] += yl[i+24] * (qw[i/2] & 0x00c0);
                acc2[3] += yl[i+25] * (qw[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += nbi * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32")]]
kernel void kernel_mul_mv_q2_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float yl[32];

    // block groups by real simd width: 4 on wave32, 8 on wave64. tid stays 0..7 so
    // ip/il never overflow the mm[4] mask table
    const short ng  = FC_mul_mv_nw/8;
    const short bg  = tiisg/8;
    const short tid = tiisg%8;
    const short ip  = tid/4;          // 0 or 1
    const short il  = 2*((tid%4)/2);  // 0 or 2
    const short ir  = tid%2;
    const short l0  = 8*ir;

    // Possible masks for the high bit
    const ushort4 mm[4] = {{0x0001, 0x0100, 0x0002, 0x0200},  // ip = 0, il = 0
                           {0x0004, 0x0400, 0x0008, 0x0800},  // ip = 0, il = 2
                           {0x0010, 0x1000, 0x0020, 0x2000},  // ip = 1, il = 0
                           {0x0040, 0x4000, 0x0080, 0x8000}}; // ip = 1, il = 2

    // Possible masks for the low 2 bits
    const int4 qm[2] = {{0x0003, 0x0300, 0x000c, 0x0c00}, {0x0030, 0x3000, 0x00c0, 0xc000}};

    const ushort4 hm = mm[2*ip + il/2];

    const short shift = 2*il;

    const float v1 = il == 0 ? 4.f : 64.f;
    const float v2 = 4.f * v1;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    const short q_offset = 32*ip + l0;
    const short y_offset = 128*ip + 32*il + l0;

    device const float * y1 = yy + bg*QK_K + (int) y_offset;

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf1[nr0] = {0.f};
    float sumf2[nr0] = {0.f};

    for (int i = bg; i < nb; i += ng) {
        for (short l = 0; l < 8; ++l) {
            yl[l+ 0] = y1[l+ 0];
            yl[l+ 8] = y1[l+16];
            yl[l+16] = y1[l+32];
            yl[l+24] = y1[l+48];
        }

        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uint16_t * q = (device const uint16_t *)(x[i].qs + (int) q_offset);
        device const uint16_t * h = (device const uint16_t *)(x[i].hmask + (int) l0);
        device const uint16_t * a = (device const uint16_t *)(x[i].scales);
        device const half * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float)dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            // q and h are read four contiguous shorts at a time in each half, so an aligned
            // stride turns each group into two words
            ushort qa[4], qb[4], ha[4], hb[4];
            if (FC_mul_mv_row8) {
                const uint2 tq = mv_ld_u32x2(q);
                const uint2 tQ = mv_ld_u32x2(q + 8);
                const uint2 th = mv_ld_u32x2(h);
                const uint2 tH = mv_ld_u32x2(h + 8);
                qa[0]=(ushort)tq.x; qa[1]=(ushort)(tq.x>>16); qa[2]=(ushort)tq.y; qa[3]=(ushort)(tq.y>>16);
                qb[0]=(ushort)tQ.x; qb[1]=(ushort)(tQ.x>>16); qb[2]=(ushort)tQ.y; qb[3]=(ushort)(tQ.y>>16);
                ha[0]=(ushort)th.x; ha[1]=(ushort)(th.x>>16); ha[2]=(ushort)th.y; ha[3]=(ushort)(th.y>>16);
                hb[0]=(ushort)tH.x; hb[1]=(ushort)(tH.x>>16); hb[2]=(ushort)tH.y; hb[3]=(ushort)(tH.y>>16);
            } else {
                FOR_UNROLL (short t = 0; t < 4; ++t) { qa[t]=q[t]; qb[t]=q[t+8]; ha[t]=h[t]; hb[t]=h[t+8]; }
            }

            float s1 = 0, s2 = 0, s3 = 0, s4 = 0, s5 = 0, s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = qa[l/2];
                s1 += yl[l+0] * (qs & qm[il/2][0]);
                s2 += yl[l+1] * (qs & qm[il/2][1]);
                s3 += ((ha[l/2] & hm[0]) ? 0.f : yl[l+0]) + ((ha[l/2] & hm[1]) ? 0.f : yl[l+1]);
                s4 += yl[l+16] * (qs & qm[il/2][2]);
                s5 += yl[l+17] * (qs & qm[il/2][3]);
                s6 += ((ha[l/2] & hm[2]) ? 0.f : yl[l+16]) + ((ha[l/2] & hm[3]) ? 0.f : yl[l+17]);
            }
            float d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            float d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[0] - 32);
            sumf2[row] += d2 * (scales[2] - 32);

            s1 = s2 = s3 = s4 = s5 = s6 = 0;
            for (short l = 0; l < 8; l += 2) {
                const int32_t qs = qb[l/2];
                s1 += yl[l+8] * (qs & qm[il/2][0]);
                s2 += yl[l+9] * (qs & qm[il/2][1]);
                s3 += ((hb[l/2] & hm[0]) ? 0.f : yl[l+8]) + ((hb[l/2] & hm[1]) ? 0.f : yl[l+9]);
                s4 += yl[l+24] * (qs & qm[il/2][2]);
                s5 += yl[l+25] * (qs & qm[il/2][3]);
                s6 += ((hb[l/2] & hm[2]) ? 0.f : yl[l+24]) + ((hb[l/2] & hm[3]) ? 0.f : yl[l+25]);
            }
            d1 = d_all * (s1 + 1.f/256.f * s2 - s3*v1);
            d2 = d_all * (s4 + 1.f/256.f * s5 - s6*v2);
            sumf1[row] += d1 * (scales[1] - 32);
            sumf2[row] += d2 * (scales[3] - 32);

            q  += args.nb01/2;
            h  += args.nb01/2;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += ng * QK_K;
    }

    for (int row = 0; row < nr0; ++row) {
        const float sumf = (sumf1[row] + 0.25f * sumf2[row]) / (1 << shift);
        sumf1[row] = simd_sum(sumf);
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    if (tiisg == 0) {
        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            dst_f32[first_row + row] = sumf1[row];
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32")]]
kernel void kernel_mul_mv_q3_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t, short NSPLIT = 1>
void kernel_mul_mv_q4_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    // Number of 8-thread groups in the simdgroup: 4 on wave32, 8 on wave64. The block
    // stride below must match it so each super-block is summed exactly once.
    const short NG = FC_mul_mv_nw/8;
    // a row shorter than NG blocks leaves groups idle: with NSPLIT the groups split into
    // NSPLIT sets, each summing its own rows over GB groups
    const short GB = NG/NSPLIT;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix  = (tiisg/8)%GB;  // 0...GB-1
    const short sub = (tiisg/8)/GB;  // 0...NSPLIT-1
    const short it  = tiisg%8;  // 0...7
    const short iq = it/4;     // 0 or 1
    const short ir = it%4;     // 0...3

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = ((r0 * NSG + sgitg) * NSPLIT + sub) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[16];
    float yh[16];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += GB) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        // d, dmin and the scales are the block's first 16 bytes and the block is 144, so every
        // row lands 16-byte aligned and one load covers what was three shorts plus two halves
        device const uint4 * hp = (device const uint4 *) &x[ib];
        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + (16 * (int) iq + 4 * (int) ir);

        const ushort shs = 16*iq;

        for (short row = 0; row < nr0; row++) {
            const uint4  hdr = *hp;
            const half2  dm  = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc16[0] = s0 & kmask1;
            sc16[1] = s2 & kmask1;
            sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            const ushort4 qv1 = *(device const ushort4 *)(q1);
            const ushort4 qv2 = *(device const ushort4 *)(q1 + 32);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short i = 0; i < 4; ++i) {
                acc1[0] += yl[2*i + 0] * (qv1[i] & 0x000F);
                acc1[1] += yl[2*i + 1] * (qv1[i] & 0x0F00);
                acc1[2] += yl[2*i + 8] * (qv1[i] & 0x00F0);
                acc1[3] += yl[2*i + 9] * (qv1[i] & 0xF000);
                acc2[0] += yh[2*i + 0] * (qv2[i] & 0x000F);
                acc2[1] += yh[2*i + 1] * (qv2[i] & 0x0F00);
                acc2[2] += yh[2*i + 8] * (qv2[i] & 0x00F0);
                acc2[3] += yh[2*i + 9] * (qv2[i] & 0xF000);
            }

            sumf[row] += dm[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
        }

        y4 += GB * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all;
        if (NSPLIT == 1) {
            sum_all = simd_sum(sumf[row]);
        } else {
            sum_all = sumf[row];
            for (ushort o = 1; o < GB*8; o <<= 1) {
                sum_all += simd_shuffle_xor(sum_all, o);
            }
        }
        if (tiisg % (GB*8) == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// a few columns share one read of each block: the rows' quants and scales stay in registers
// and the columns stream past them
template<int NR0, int NC>
void kernel_mul_mv_q4_K_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NG  = FC_mul_mv_nw/8;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);

    // a column past the end reads the last one and is not written
    device const float * yc[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yc[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix * QK_K + 64 * iq + 8 * ir;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    const ushort shs = 16*iq;

    for (int ib = ix; ib < nb; ib += NG) {
        half2   dm [NR0];
        uint16_t sc[NR0][4];
        ushort4 qv1[NR0];
        ushort4 qv2[NR0];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_q4_K * xb = (device const block_q4_K *) ((device const char *) &x[ib] + row*args.nb01);
            const uint4 hdr = *(device const uint4 *) xb;
            dm[row] = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc[row][0] = s0 & kmask1;
            sc[row][1] = s2 & kmask1;
            sc[row][2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc[row][3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            device const uint16_t * q1 = (device const uint16_t *) xb->qs + (16 * (int) iq + 4 * (int) ir);
            qv1[row] = *(device const ushort4 *)(q1);
            qv2[row] = *(device const ushort4 *)(q1 + 32);
        }

        FOR_UNROLL (short c = 0; c < NC; c++) {
            float yl[16];
            float yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};

            device const float * y4 = yc[c];
            for (short i = 0; i < 8; ++i) {
                yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
                yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
                yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
                yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
            }
            yc[c] += NG * QK_K;

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                thread const uint8_t * sc8 = (thread const uint8_t *) sc[row];

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (qv1[row][i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (qv1[row][i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (qv1[row][i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (qv1[row][i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (qv2[row][i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (qv2[row][i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (qv2[row][i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (qv2[row][i] & 0xF000);
                }

                sumf[row][c] += dm[row][0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                              (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                              (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                              (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                dm[row][1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)(r1 + c)*args.ne0;

        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

#define Q4_K_MV_NC(NR0, NC) \
[[host_name("kernel_mul_mv_q4_K_f32_c" #NC "_r" #NR0)]] \
kernel void kernel_mul_mv_q4_K_f32_c##NC##_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_q4_K_f32_nc_impl<NR0, NC>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

Q4_K_MV_NC(4, 2)

[[host_name("kernel_mul_mv_q4_K_f32")]]
kernel void kernel_mul_mv_q4_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<N_R0_Q4_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    float yl[16], yh[16];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // 8 threads per block-group; scale the group count with the SIMD width so iq stays
    // in {0,1} on wave64 (NG = 4 on wave32, 8 on wave64).
    const short NG  = FC_mul_mv_nw/8;
    const short tid = tiisg%8;
    const short ix  = tiisg/8;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    // pre-shifting qh by this puts the four bits this lane wants at fixed positions
    // 0/1/4/5, so the masks below are constants and the selects become shifts
    const ushort hsh = 2*iq;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    // same header as q4_K: d, dmin and the 12 scales are the block's first 16 bytes, and 176
    // keeps every row 16-byte aligned, so one load covers what was two halves and three shorts
    const ushort shs = 16*iq;

    for (int i = ix; i < nb; i += NG) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const uint4  * hp = (device const uint4 *) &x[i];

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            const uint4  hdr = *hp;
            const half2  dm  = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc16[0] = s0 & kmask1;
            sc16[1] = s2 & kmask1;
            sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            float4 acc1 = {0.f};
            // qs and qh are 8-byte aligned here, so one load each instead of two
            const uint2 q1w = *(device const uint2 *)(q1);
            const uint2 q2w = *(device const uint2 *)(q2);
            const uint2 qhw = *(device const uint2 *)(qh);

            const uchar4 q1a = as_type<uchar4>(q1w.x), q1b = as_type<uchar4>(q1w.y);
            const uchar4 q2a = as_type<uchar4>(q2w.x), q2b = as_type<uchar4>(q2w.y);
            const uchar4 qha = as_type<uchar4>((qhw.x >> hsh) & 0x33333333u);
            const uchar4 qhb = as_type<uchar4>((qhw.y >> hsh) & 0x33333333u);

            FOR_UNROLL (short l = 0; l < 8; ++l) {
                const uint8_t h  = l < 4 ? qha[l] : qhb[l - 4];
                const uint8_t v1 = l < 4 ? q1a[l] : q1b[l - 4];
                const uint8_t v2 = l < 4 ? q2a[l] : q2b[l - 4];
                acc1[0] += yl[l+0] * (float) ((v1 & 0x0F) | ((h & 1) << 4));
                acc1[1] += yl[l+8] * (float) ((v1 >>   4) | ((h & 2) << 3));
                acc1[2] += yh[l+0] * (float) ((v2 & 0x0F) |  (h & 16));
                acc1[3] += yh[l+8] * (float) ((v2 >>   4) | ((h & 32) >> 1));
            }

            sumf[row] += dm[0] * (sc8[0] * acc1[0] +
                                  sc8[1] * acc1[1] +
                                  sc8[4] * acc1[2] +
                                  sc8[5] * acc1[3]) -
                         dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
        }

        y1 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

// a pair of columns per pass: the rows' quants and scales stay in registers and the columns
// stream past them, as in the q4_K variant
template<int NR0, int NC>
void kernel_mul_mv_q5_K_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short NG  = FC_mul_mv_nw/8;
    const short tid = tiisg%8;
    const short ix  = tiisg/8;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    const ushort hsh = 2*iq;
    const ushort shs = 16*iq;

    // a column past the end reads the last one and is not written
    device const float * y1[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        y1[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix*QK_K + y_offset;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int i = ix; i < nb; i += NG) {
        half2    dm [NR0];
        uint16_t sc [NR0][4];
        uint2    q1w[NR0];
        uint2    q2w[NR0];
        uint2    qhw[NR0];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_q5_K * xb = (device const block_q5_K *) ((device const char *) &x[i] + row*args.nb01);
            const uint4 hdr = *(device const uint4 *) xb;
            dm[row] = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc[row][0] = s0 & kmask1;
            sc[row][1] = s2 & kmask1;
            sc[row][2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc[row][3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            q1w[row] = *(device const uint2 *)(xb->qs + q_offset);
            q2w[row] = *(device const uint2 *)(xb->qs + q_offset + 64);
            qhw[row] = *(device const uint2 *)(xb->qh + l0);
        }

        FOR_UNROLL (short c = 0; c < NC; c++) {
            float yl[16], yh[16];
            float4 sumy = {0.f, 0.f, 0.f, 0.f};
            device const float * y2 = y1[c] + 128;
            for (short l = 0; l < 8; ++l) {
                yl[l+0] = y1[c][l+ 0]; sumy[0] += yl[l+0];
                yl[l+8] = y1[c][l+32]; sumy[1] += yl[l+8];
                yh[l+0] = y2[l+ 0];    sumy[2] += yh[l+0];
                yh[l+8] = y2[l+32];    sumy[3] += yh[l+8];
            }
            y1[c] += NG*QK_K;

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                thread const uint8_t * sc8 = (thread const uint8_t *) sc[row];

                const uchar4 q1a = as_type<uchar4>(q1w[row].x), q1b = as_type<uchar4>(q1w[row].y);
                const uchar4 q2a = as_type<uchar4>(q2w[row].x), q2b = as_type<uchar4>(q2w[row].y);
                const uchar4 qha = as_type<uchar4>((qhw[row].x >> hsh) & 0x33333333u);
                const uchar4 qhb = as_type<uchar4>((qhw[row].y >> hsh) & 0x33333333u);

                float4 acc1 = {0.f};
                FOR_UNROLL (short l = 0; l < 8; ++l) {
                    const uint8_t h  = l < 4 ? qha[l] : qhb[l - 4];
                    const uint8_t v1 = l < 4 ? q1a[l] : q1b[l - 4];
                    const uint8_t v2 = l < 4 ? q2a[l] : q2b[l - 4];
                    acc1[0] += yl[l+0] * (float) ((v1 & 0x0F) | ((h & 1) << 4));
                    acc1[1] += yl[l+8] * (float) ((v1 >>   4) | ((h & 2) << 3));
                    acc1[2] += yh[l+0] * (float) ((v2 & 0x0F) |  (h & 16));
                    acc1[3] += yh[l+8] * (float) ((v2 >>   4) | ((h & 32) >> 1));
                }

                sumf[row][c] += dm[row][0] * (sc8[0] * acc1[0] + sc8[1] * acc1[1] + sc8[4] * acc1[2] + sc8[5] * acc1[3]) -
                                dm[row][1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

#define Q5_K_MV_NC(NR0) \
[[host_name("kernel_mul_mv_q5_K_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_q5_K_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_q5_K_f32_nc_impl<NR0, 2>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

Q5_K_MV_NC(2)
Q5_K_MV_NC(4)
Q5_K_MV_NC(8)

[[host_name("kernel_mul_mv_q5_K_f32_c3_r4")]]
kernel void kernel_mul_mv_q5_K_f32_c3_r4(
        constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_nc_impl<4, 3>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32")]]
kernel void kernel_mul_mv_q5_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q6_K_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    float yl[16];

    // 16 threads per block-group; scale the group count with the SIMD width so ip stays
    // in {0,1} on wave64 (NG = 2 on wave32, 4 on wave64).
    const short NG  = FC_mul_mv_nw/16;
    const short tid = tiisg%16;
    const short ix  = tiisg/16;
    const short ip  = tid/8;         // 0 or 1
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += NG) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < 4; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            // the three arrays are read four consecutive bytes at a time. The block is 210 bytes,
            // so every other one sits two bytes off a word and a card that requires natural
            // alignment reads the wrong bytes there; two shorts land right on every block
            uint w1 = 0, w2 = 0, wh = 0;
            if (FC_mul_mv_row8) {
                w1 = mv_ld_u32(q1);
                w2 = mv_ld_u32(q2);
                wh = mv_ld_u32(qh);
            } else if (FC_mul_mv_row4) {
                device const ushort * s1 = (device const ushort *) q1;
                device const ushort * s2 = (device const ushort *) q2;
                device const ushort * sh = (device const ushort *) qh;

                w1 = (uint) s1[0] | ((uint) s1[1] << 16);
                w2 = (uint) s2[0] | ((uint) s2[1] << 16);
                wh = (uint) sh[0] | ((uint) sh[1] << 16);
            }

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                const uint8_t b1 = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(w1 >> (8*l)) : q1[l];
                const uint8_t b2 = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(w2 >> (8*l)) : q2[l];
                const uint8_t bh = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(wh >> (8*l)) : qh[l];

                sums[0] += yl[4*l + 0] * ((int8_t)((b1 & 0xF) | ((bh & kmask1) << 4)) - 32);
                sums[1] += yl[4*l + 1] * ((int8_t)((b2 & 0xF) | ((bh & kmask2) << 2)) - 32);
                sums[2] += yl[4*l + 2] * ((int8_t)((b1  >> 4) | ((bh & kmask3) << 0)) - 32);
                sums[3] += yl[4*l + 3] * ((int8_t)((b2  >> 4) | ((bh & kmask4) >> 2)) - 32);
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// a pair of columns per pass: the rows' quants and scales stay in registers and the columns
// stream past them
template<int NR0, int NC>
void kernel_mul_mv_q6_K_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);

    const short NG  = FC_mul_mv_nw/16;
    const short tid = tiisg%16;
    const short ix  = tiisg/16;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    // a column past the end reads the last one and is not written
    device const float * yy[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yy[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + y_offset;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int i = ix; i < nb; i += NG) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            float yl[16];
            device const float * y = yy[c] + i * QK_K;
            for (short l = 0; l < 4; ++l) {
                yl[4*l + 0] = y[l +  0];
                yl[4*l + 1] = y[l + 32];
                yl[4*l + 2] = y[l + 64];
                yl[4*l + 3] = y[l + 96];
            }

            device const uint8_t * q1 = x[i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = x[i].qh + q_offset_h;
            device const int8_t  * sc = x[i].scales + is;
            device const half    * dh = &x[i].d;

            FOR_UNROLL (short row = 0; row < NR0; row++) {
                device const ushort * s1 = (device const ushort *) q1;
                device const ushort * s2 = (device const ushort *) q2;
                device const ushort * sh = (device const ushort *) qh;
                const uint w1 = (uint) s1[0] | ((uint) s1[1] << 16);
                const uint w2 = (uint) s2[0] | ((uint) s2[1] << 16);
                const uint wh = (uint) sh[0] | ((uint) sh[1] << 16);

                float4 sums = {0.f, 0.f, 0.f, 0.f};
                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint8_t b1 = (uint8_t)(w1 >> (8*l));
                    const uint8_t b2 = (uint8_t)(w2 >> (8*l));
                    const uint8_t bh = (uint8_t)(wh >> (8*l));

                    sums[0] += yl[4*l + 0] * ((int8_t)((b1 & 0xF) | ((bh & kmask1) << 4)) - 32);
                    sums[1] += yl[4*l + 1] * ((int8_t)((b2 & 0xF) | ((bh & kmask2) << 2)) - 32);
                    sums[2] += yl[4*l + 2] * ((int8_t)((b1  >> 4) | ((bh & kmask3) << 0)) - 32);
                    sums[3] += yl[4*l + 3] * ((int8_t)((b2  >> 4) | ((bh & kmask4) >> 2)) - 32);
                }
                sumf[row][c] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

                q1 += args.nb01;
                q2 += args.nb01;
                qh += args.nb01;
                sc += args.nb01;
                dh += args.nb01/2;
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

#define Q6_K_MV_NC(NR0) \
[[host_name("kernel_mul_mv_q6_K_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_q6_K_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_q6_K_f32_nc_impl<NR0, 2>(args, src0, src1, dst, tgpig, tiisg, sgitg); \
}

Q6_K_MV_NC(2)
Q6_K_MV_NC(4)
Q6_K_MV_NC(8)

[[host_name("kernel_mul_mv_q6_K_f32_c3_r2")]]
kernel void kernel_mul_mv_q6_K_f32_c3_r2(
        constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst,
        uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_nc_impl<2, 3>(args, src0, src1, dst, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32")]]
kernel void kernel_mul_mv_q6_K_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<N_R0_Q6_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

// ======================= "True" 2-bit

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xxs * x = (device const block_iq2_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 256; i += gsz) svalues[i] = iq2xxs_grid[i];
        for (int i = gid; i < 128; i += gsz) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xxs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            device const uint8_t * aux8 = (device const uint8_t *)q2;
            const uint32_t aux32 = q2[2] | (q2[3] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum = 0;
            for (short l = 0; l < 4; ++l) {
                // one 32-bit read per half instead of eight byte reads; RDNA2 has no
                // native 64-bit shift, so two uint32 beat one uint64 here
                const threadgroup uint32_t * gp = (const threadgroup uint32_t *)(svalues + aux8[l]);
                const uint32_t ga = gp[0], gb = gp[1];
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 8; ++j) {
                    const uint32_t gw = j < 4 ? ga : gb;
                    sum += yl[8*l + j] * (float)((gw >> (8*(j & 3))) & 0xFF) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * sum;

            dh += args.nb01/2;
            q2 += args.nb01/2;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xxs_f32")]]
kernel void kernel_mul_mv_iq2_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_xs * x = (device const block_iq2_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 512);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 512; i += gsz) svalues[i] = iq2xs_grid[i];
        for (int i = gid; i < 128; i += gsz) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_xs * xr = x + ibl;
        device const uint16_t * q2 = xr->qs + 4 * ib;
        device const uint8_t  * sc = xr->scales + ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint8_t ls1 = sc[0] & 0xf;
            const uint8_t ls2 = sc[0] >>  4;
            const float d1 = db * (0.5f + ls1);
            const float d2 = db * (0.5f + ls2);

            float sum1 = 0, sum2 = 0;
            for (short l = 0; l < 2; ++l) {
                // one 32-bit read per half instead of eight byte reads; RDNA2 has no
                // native 64-bit shift, so two uint32 beat one uint64 here
                const threadgroup uint32_t * gp = (const threadgroup uint32_t *)(svalues + (q2[l] & 511));
                const uint32_t ga = gp[0], gb = gp[1];
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    const uint32_t gw = j < 4 ? ga : gb;
                    sum1 += yl[8*l + j] * (float)((gw >> (8*(j & 3))) & 0xFF) * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            for (short l = 2; l < 4; ++l) {
                const threadgroup uint8_t * grid = (const threadgroup uint8_t *)(svalues + (q2[l] & 511));
                const uint8_t signs = ssigns[(q2[l] >> 9)];
                for (short j = 0; j < 8; ++j) {
                    sum2 += yl[8*l + j] * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
                }
            }
            sumf[row] += d1 * sum1 + d2 * sum2;

            dh += args.nb01/2;
            q2 += args.nb01/2;
            sc += args.nb01;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_xs_f32")]]
kernel void kernel_mul_mv_iq2_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<N_R0_IQ2_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);
    device const float         * y = (device const float         *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 256; i += gsz) svalues[i] = iq3xxs_grid[i];
        for (int i = gid; i < 128; i += gsz) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.5f;
        }
    }
}

// a pair of columns per pass: each sub-block's grid values and signs are looked up once
template<int NR0, int NC>
void kernel_mul_mv_iq3_xxs_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_iq3_xxs * x = (device const block_iq3_xxs *) (src0 + offset0);

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *)(shmem);
    threadgroup uint8_t  * ssigns  = (threadgroup uint8_t  *)(svalues + 256);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 256; i += gsz) svalues[i] = iq3xxs_grid[i];
        for (int i = gid; i < 128; i += gsz) ssigns[i] = ksigns_iq2xs[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    // a column past the end reads the last one and is not written
    device const float * y4[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        y4[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + 32 * ix;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        float yl[NC][32];
        FOR_UNROLL (short c = 0; c < NC; c++) {
            for (short i = 0; i < 32; ++i) {
                yl[c][i] = y4[c][i];
            }
            y4[c] += 32 * FC_mul_mv_nw;
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_xxs * xr = x + ibl;
        device const uint8_t  * q3 = xr->qs + 8 * ib;
        device const uint16_t * gas = (device const uint16_t *)(xr->qs + QK_K/4) + 2 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; row++) {
            const float db = dh[0];
            const uint32_t aux32 = gas[0] | (gas[1] << 16);
            const float d = db * (0.5f + (aux32 >> 28));

            float sum[NC];
            FOR_UNROLL (short c = 0; c < NC; c++) {
                sum[c] = 0.f;
            }
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + q3[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + q3[2*l+1]);
                const uint8_t signs = ssigns[(aux32 >> 7*l) & 127];
                float g[8];
                for (short j = 0; j < 4; ++j) {
                    g[j + 0] = grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
                    g[j + 4] = grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
                }
                FOR_UNROLL (short c = 0; c < NC; c++) {
                    for (short j = 0; j < 8; ++j) {
                        sum[c] += yl[c][8*l + j] * g[j];
                    }
                }
            }
            FOR_UNROLL (short c = 0; c < NC; c++) {
                sumf[row][c] += d * sum[c];
            }

            dh  += args.nb01/2;
            q3  += args.nb01;
            gas += args.nb01/2;
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.5f;
            }
        }
    }
}

#define IQ3_XXS_MV_NC(NR0) \
[[host_name("kernel_mul_mv_iq3_xxs_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_iq3_xxs_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_iq3_xxs_f32_nc_impl<NR0, 2>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \
}

IQ3_XXS_MV_NC(4)
IQ3_XXS_MV_NC(8)

[[host_name("kernel_mul_mv_iq3_xxs_f32_c3_r8")]]
kernel void kernel_mul_mv_iq3_xxs_f32_c3_r8(
        constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq3_xxs_f32_nc_impl<8, 3>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32")]]
kernel void kernel_mul_mv_iq3_xxs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 512; i += gsz) svalues[i] = iq3s_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float2 sum = {0};
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                for (short j = 0; j < 4; ++j) {
                    sum[0] += yl[8*l + j + 0] * grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    sum[1] += yl[8*l + j + 4] * grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
            }
            sumf[row] += d * (sum[0] + sum[1]);

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

// a pair of columns per pass: each sub-block's grid values and signs are looked up once
template<int NR0, int NC>
void kernel_mul_mv_iq3_s_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + offset0);

    const int nb32 = nb * (QK_K / 32);

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 512; i += gsz) svalues[i] = iq3s_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int ix = tiisg;

    // a column past the end reads the last one and is not written
    device const float * y4[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        y4[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + 32 * ix;
    }

    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        float yl[NC][32];
        FOR_UNROLL (short c = 0; c < NC; c++) {
            for (short i = 0; i < 32; ++i) {
                yl[c][i] = y4[c][i];
            }
            y4[c] += 32 * FC_mul_mv_nw;
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 8 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half * dh = &xr->d;

        for (short row = 0; row < NR0; row++) {
            const float db = dh[0];
            const float d = db * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));

            float sum[NC];
            FOR_UNROLL (short c = 0; c < NC; c++) {
                sum[c] = 0.f;
            }
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(table1 + qs[2*l+0]);
                const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(table2 + qs[2*l+1]);
                float g[8];
                for (short j = 0; j < 4; ++j) {
                    g[j + 0] = grid1[j] * select(1, -1, signs[l] & kmask_iq2xs[j+0]);
                    g[j + 4] = grid2[j] * select(1, -1, signs[l] & kmask_iq2xs[j+4]);
                }
                FOR_UNROLL (short c = 0; c < NC; c++) {
                    for (short j = 0; j < 8; ++j) {
                        sum[c] += yl[c][8*l + j] * g[j];
                    }
                }
            }
            FOR_UNROLL (short c = 0; c < NC; c++) {
                sumf[row][c] += d * sum[c];
            }

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;
        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

#define IQ3_S_MV_NC(NR0) \
[[host_name("kernel_mul_mv_iq3_s_f32_c2_r" #NR0)]] \
kernel void kernel_mul_mv_iq3_s_f32_c2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_iq3_s_f32_nc_impl<NR0, 2>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \
}

IQ3_S_MV_NC(2)
IQ3_S_MV_NC(4)
IQ3_S_MV_NC(8)

[[host_name("kernel_mul_mv_iq3_s_f32_c3_r8")]]
kernel void kernel_mul_mv_iq3_s_f32_c3_r8(
        constant ggml_metal_kargs_mul_mv & args, device const char * src0, device const char * src1, device char * dst,
        threadgroup char * shmem [[threadgroup(0)]],
        uint3 tgpig[[threadgroup_position_in_grid]], ushort tiisg[[thread_index_in_simdgroup]], ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq3_s_f32_nc_impl<8, 3>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_s_f32")]]
kernel void kernel_mul_mv_iq3_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<N_R0_IQ3_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_iq3_s_f32_nr0_2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq3_s_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_iq3_s_f32_nr0_8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq3_s_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// RX6800: IQ3_S mat-vec over NC columns per threadgroup (spec-decode verify batches, ne11 = 2..8).
// The per-column kernel re-dequantizes the weights for every column, and decode on RDNA2 is
// dequant-bound, so each extra verify token costs ~0.7 of a decode step. Here each 32-weight
// chunk is dequantized once into registers and dotted against all NC columns.
// Same layout as kernel_mul_mv_iq3_s_f32_impl (non-split): grid table staged in threadgroup memory.
template<int nr0, short NC>
void kernel_mul_mv_iq3_s_f32_ncm_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb   = args.ne00/QK_K;
    const int nb32 = nb * (QK_K / 32);

    const int r0 = tgpig.x;
    const int c0 = tgpig.y*NC;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const short nc = MIN((int) NC, args.ne11 - c0);

    device const block_iq3_s * x = (device const block_iq3_s *) (src0 + first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
    device const char        * y = src1 + (uint64_t) c0*args.nb11 + i12*args.nb12 + i13*args.nb13;

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int nval = 8;
        const int pos  = (32*sgitg + tiisg)*nval;
        for (int i = 0; i < nval; ++i) svalues[pos + i] = iq3s_grid[pos + i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    float sumf[nr0][NC];
    for (short row = 0; row < nr0; ++row) {
        for (short c = 0; c < NC; ++c) {
            sumf[row][c] = 0.f;
        }
    }

    for (int ib32 = tiisg; ib32 < nb32; ib32 += 32) {
        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq3_s * xr = x + ibl;
        device const uint8_t * qs    = xr->qs + 8 * ib;
        device const uint8_t * qh    = xr->qh + ib;
        device const uint8_t * sc    = xr->scales + (ib/2);
        device const uint8_t * signs = xr->signs + 4 * ib;
        device const half    * dh    = &xr->d;

        // dequantize the 32 weights of this chunk once per row (d folded in)
        float4 w[nr0][8];
        for (short row = 0; row < nr0; ++row) {
            const float d = (float) dh[0] * (1 + 2*((sc[0] >> 4*(ib%2)) & 0xf));
            for (short l = 0; l < 4; ++l) {
                const threadgroup uint32_t * table1 = qh[0] & kmask_iq2xs[2*l+0] ? svalues + 256 : svalues;
                const threadgroup uint32_t * table2 = qh[0] & kmask_iq2xs[2*l+1] ? svalues + 256 : svalues;
                const float4 g1 = float4(as_type<uchar4>(table1[qs[2*l+0]]));
                const float4 g2 = float4(as_type<uchar4>(table2[qs[2*l+1]]));
                const uint8_t sg = signs[l];
                w[row][2*l+0] = d * g1 * float4(select(1.f, -1.f, (bool)(sg & 0x01)), select(1.f, -1.f, (bool)(sg & 0x02)),
                                                select(1.f, -1.f, (bool)(sg & 0x04)), select(1.f, -1.f, (bool)(sg & 0x08)));
                w[row][2*l+1] = d * g2 * float4(select(1.f, -1.f, (bool)(sg & 0x10)), select(1.f, -1.f, (bool)(sg & 0x20)),
                                                select(1.f, -1.f, (bool)(sg & 0x40)), select(1.f, -1.f, (bool)(sg & 0x80)));
            }
            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        // each column's 32 activations are loaded once and reused for every row
        for (short c = 0; c < NC; ++c) {
            if (c >= nc) {
                break;
            }
            device const float4 * y4 = (device const float4 *) (y + (uint64_t) c*args.nb11) + 8*ib32;
            float4 yv[8];
            for (short k = 0; k < 8; ++k) {
                yv[k] = y4[k];
            }
            for (short row = 0; row < nr0; ++row) {
                float acc = 0.f;
                for (short k = 0; k < 8; ++k) {
                    acc += dot(w[row][k], yv[k]);
                }
                sumf[row][c] += acc;
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1;

    for (short row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        for (short c = 0; c < NC; ++c) {
            if (c >= nc) {
                break;
            }
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[(uint64_t)(c0 + c)*args.ne0 + first_row + row] = sum_all;
            }
        }
    }
}

#define KERNEL_MUL_MV_IQ3_S_NC(NR0_, NC_) \
[[host_name("kernel_mul_mv_iq3_s_f32_r" #NR0_ "_nc" #NC_)]] \
kernel void kernel_mul_mv_iq3_s_f32_r##NR0_##_nc##NC_( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_iq3_s_f32_ncm_impl<NR0_, NC_>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \
}

KERNEL_MUL_MV_IQ3_S_NC(1, 2)
KERNEL_MUL_MV_IQ3_S_NC(1, 3)
KERNEL_MUL_MV_IQ3_S_NC(1, 4)
KERNEL_MUL_MV_IQ3_S_NC(1, 5)
KERNEL_MUL_MV_IQ3_S_NC(1, 6)
KERNEL_MUL_MV_IQ3_S_NC(1, 7)
KERNEL_MUL_MV_IQ3_S_NC(1, 8)
KERNEL_MUL_MV_IQ3_S_NC(2, 2)
KERNEL_MUL_MV_IQ3_S_NC(2, 3)
KERNEL_MUL_MV_IQ3_S_NC(2, 4)
KERNEL_MUL_MV_IQ3_S_NC(2, 5)
KERNEL_MUL_MV_IQ3_S_NC(2, 6)
KERNEL_MUL_MV_IQ3_S_NC(2, 7)
KERNEL_MUL_MV_IQ3_S_NC(2, 8)
KERNEL_MUL_MV_IQ3_S_NC(4, 2)
KERNEL_MUL_MV_IQ3_S_NC(4, 3)
KERNEL_MUL_MV_IQ3_S_NC(4, 4)
KERNEL_MUL_MV_IQ3_S_NC(4, 5)
KERNEL_MUL_MV_IQ3_S_NC(4, 6)
KERNEL_MUL_MV_IQ3_S_NC(4, 7)
KERNEL_MUL_MV_IQ3_S_NC(4, 8)

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq2_s * x = (device const block_iq2_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    // reading the 1024-entry grid straight out of constant memory costs 8 divergent loads per
    // inner step; staging it once per threadgroup is what iq3_xxs already does
    threadgroup uint64_t * svalues = (threadgroup uint64_t *)(shmem);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 1024; i += gsz) svalues[i] = iq2s_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq2_s * xr = x + ibl;
        device const uint8_t * qs = xr->qs + 4 * ib;
        device const uint8_t * qh = xr->qh + ib;
        device const uint8_t * sc = xr->scales + ib;
        device const uint8_t * signs = qs + QK_K/8;
        device const half * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            const float db = dh[0];
            const float d1 = db * (0.5f + (sc[0] & 0xf));
            const float d2 = db * (0.5f + (sc[0] >>  4));

            float2 sum = {0};
            for (short l = 0; l < 2; ++l) {
                //const threadgroup uint8_t * grid1 = (const threadgroup uint8_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                //const threadgroup uint8_t * grid2 = (const threadgroup uint8_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                // each grid entry is the eight bytes of this group, so one 64-bit read
                // replaces the eight byte reads the loop used to do
                // two 32-bit reads: RDNA2 has no native 64-bit shift, so a uint64 extract
                // is emulated and costs more than the byte reads it replaces
                const threadgroup uint32_t * p1 = (const threadgroup uint32_t *)(svalues + (qs[l+0] | ((qh[0] << (8-2*l)) & 0x300)));
                const threadgroup uint32_t * p2 = (const threadgroup uint32_t *)(svalues + (qs[l+2] | ((qh[0] << (4-2*l)) & 0x300)));
                const uint32_t g1a = p1[0], g1b = p1[1];
                const uint32_t g2a = p2[0], g2b = p2[1];
                const uint8_t s1 = signs[l+0];
                const uint8_t s2 = signs[l+2];
                for (short j = 0; j < 8; ++j) {
                    const uint32_t w1 = j < 4 ? g1a : g1b;
                    const uint32_t w2 = j < 4 ? g2a : g2b;
                    const short sh = 8*(j & 3);
                    sum[0] += yl[8*l + j +  0] * (float)((w1 >> sh) & 0xFF) * select(1, -1, s1 & kmask_iq2xs[j]);
                    sum[1] += yl[8*l + j + 16] * (float)((w2 >> sh) & 0xFF) * select(1, -1, s2 & kmask_iq2xs[j]);
                }
            }
            sumf[row] += d1 * sum[0] + d2 * sum[1];

            dh    += args.nb01/2;
            qs    += args.nb01;
            qh    += args.nb01;
            sc    += args.nb01;
            signs += args.nb01;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all * 0.25f;
        }
    }
}

[[host_name("kernel_mul_mv_iq2_s_f32")]]
kernel void kernel_mul_mv_iq2_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<N_R0_IQ2_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_s * x = (device const block_iq1_s *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    // the 2048-entry grid is read from constant memory on every inner step, four divergent
    // loads per row; staging it once per threadgroup is what iq2_s already does
    threadgroup uint32_t * sgrid = (threadgroup uint32_t *)(shmem);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < NGRID_IQ1S; i += gsz) sgrid[i] = iq1s_grid_gpu[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        float sumy = 0;
        for (short i = 0; i < 32; ++i) {
            yl[i] = y4[i];
            sumy += yl[i];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_s * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint16_t * qh = xr->qh + ib;
        device const half     * dh = &xr->d;

        for (short row = 0; row < nr0; row++) {
            threadgroup const uint8_t * grid1 = (threadgroup const uint8_t *)(sgrid + (qs[0] | ((qh[0] << 8) & 0x700)));
            threadgroup const uint8_t * grid2 = (threadgroup const uint8_t *)(sgrid + (qs[1] | ((qh[0] << 5) & 0x700)));
            threadgroup const uint8_t * grid3 = (threadgroup const uint8_t *)(sgrid + (qs[2] | ((qh[0] << 2) & 0x700)));
            threadgroup const uint8_t * grid4 = (threadgroup const uint8_t *)(sgrid + (qs[3] | ((qh[0] >> 1) & 0x700)));

            float sum = 0;
            for (short j = 0; j < 4; ++j) {
                sum += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                     + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4)
                     + yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                     + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            sumf[row] += (float)dh[0] * (sum + sumy * (qh[0] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA)) * (2*((qh[0] >> 12) & 7) + 1);

            dh += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01/2;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_s_f32")]]
kernel void kernel_mul_mv_iq1_s_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<N_R0_IQ1_S, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq1_m * x = (device const block_iq1_m *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    float yl[32];
    float sumf[nr0]={0.f};

    const int nb32 = nb * (QK_K / 32);

    // the 2048-entry grid is read from constant memory on every inner step, four divergent
    // loads per row; staging it once per threadgroup is what iq2_s already does
    threadgroup uint32_t * sgrid = (threadgroup uint32_t *)(shmem);
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < NGRID_IQ1S; i += gsz) sgrid[i] = iq1s_grid_gpu[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const short ix = tiisg;

    device const float * y4 = y + 32 * ix;

    iq1m_scale_t scale;

    for (int ib32 = ix; ib32 < nb32; ib32 += FC_mul_mv_nw) {
        float4 sumy = {0.f};
        for (short i = 0; i < 8; ++i) {
            yl[i+ 0] = y4[i+ 0]; sumy[0] += yl[i+ 0];
            yl[i+ 8] = y4[i+ 8]; sumy[1] += yl[i+ 8];
            yl[i+16] = y4[i+16]; sumy[2] += yl[i+16];
            yl[i+24] = y4[i+24]; sumy[3] += yl[i+24];
        }

        const int ibl = ib32 / (QK_K / 32);
        const int ib  = ib32 % (QK_K / 32);

        device const block_iq1_m * xr = x + ibl;
        device const uint8_t  * qs = xr->qs + 4 * ib;
        device const uint8_t  * qh = xr->qh + 2 * ib;
        device const uint16_t * sc = (device const uint16_t *)xr->scales;

        for (short row = 0; row < nr0; row++) {
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

            // each grid entry is one word; reading it whole costs one bank access instead of four
            const uint32_t g1 = sgrid[qs[0] | ((qh[0] << 8) & 0x700)];
            const uint32_t g2 = sgrid[qs[1] | ((qh[0] << 4) & 0x700)];
            const uint32_t g3 = sgrid[qs[2] | ((qh[1] << 8) & 0x700)];
            const uint32_t g4 = sgrid[qs[3] | ((qh[1] << 4) & 0x700)];
            thread const uint8_t * grid1 = (thread const uint8_t *)&g1;
            thread const uint8_t * grid2 = (thread const uint8_t *)&g2;
            thread const uint8_t * grid3 = (thread const uint8_t *)&g3;
            thread const uint8_t * grid4 = (thread const uint8_t *)&g4;

            float2 sum = {0.f};
            for (short j = 0; j < 4; ++j) {
                sum[0] += yl[j+ 0] * (grid1[j] & 0xf) + yl[j+ 4] * (grid1[j] >> 4)
                        + yl[j+ 8] * (grid2[j] & 0xf) + yl[j+12] * (grid2[j] >> 4);
                sum[1] += yl[j+16] * (grid3[j] & 0xf) + yl[j+20] * (grid3[j] >> 4)
                        + yl[j+24] * (grid4[j] & 0xf) + yl[j+28] * (grid4[j] >> 4);
            }
            const float delta1 = sumy[0] * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[1] * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
            const float delta2 = sumy[2] * (qh[1] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) + sumy[3] * (qh[1] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);

            sumf[row] += (float)scale.f16 * ((sum[0] + delta1) * (2*((sc[ib/2] >> (6*(ib%2)+0)) & 7) + 1) +
                                             (sum[1] + delta2) * (2*((sc[ib/2] >> (6*(ib%2)+3)) & 7) + 1));

            sc += args.nb01/2;
            qs += args.nb01;
            qh += args.nb01;
        }

        y4 += 32 * FC_mul_mv_nw;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq1_m_f32")]]
kernel void kernel_mul_mv_iq1_m_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<N_R0_IQ1_M, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_nl_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/2;
    const short ix = tiisg/2;  // 0...NG-1
    const short it = tiisg%2;  // 0 or 1

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK4_NL + it*8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 8*it);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = q4[0] | (q4[1] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = q4[2] | (q4[3] << 16);
            aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
            aux32[0] &= 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += NG * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_nl_f32")]]
kernel void kernel_mul_mv_iq4_nl_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<N_R0_IQ4_NL, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_r8")]]
kernel void kernel_mul_mv_iq4_nl_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// a few columns share one dequant of each block
template<int NR0, int NC>
void kernel_mul_mv_iq4_xs_f32_nc_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y*NC;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/16;
    const short ix = tiisg/16;
    const short it = tiisg%16;
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    // a column past the end reads the last one and is not written
    device const float * yb[NC];
    FOR_UNROLL (short c = 0; c < NC; c++) {
        const int col = min(r1 + c, (int) args.ne11 - 1);
        yb[c] = (device const float *) (src1 + col*args.nb11 + i12*args.nb12 + i13*args.nb13) + ix * QK_K + ib * 32 + il * 8;
    }

    float4 yl[NC][4];
    float sumf[NR0][NC];
    FOR_UNROLL (short row = 0; row < NR0; row++) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            sumf[row][c] = 0.f;
        }
    }

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += NG) {
        FOR_UNROLL (short c = 0; c < NC; c++) {
            device const float4 * y4 = (device const float4 *)yb[c];
            yl[c][0] = y4[0];
            yl[c][1] = y4[4];
            yl[c][2] = y4[1];
            yl[c][3] = y4[5];
            yb[c] += NG * QK_K;
        }

        FOR_UNROLL (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 qf[4];
            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf[0] = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf[1] = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf[2] = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf[3] = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            const float dl = (float)xb.d * ls;

            FOR_UNROLL (short c = 0; c < NC; c++) {
                const float4 acc = yl[c][0]*qf[0] + yl[c][1]*qf[1] + yl[c][2]*qf[2] + yl[c][3]*qf[3];
                sumf[row][c] += dl * (acc[0] + acc[1] + acc[2] + acc[3]);
            }
        }
    }

    FOR_UNROLL (short c = 0; c < NC; c++) {
        if (r1 + c >= args.ne11) {
            break;
        }
        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)(r1 + c)*args.ne0;

        for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row][c]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

#define IQ4_XS_MV_NC(NC) \
[[host_name("kernel_mul_mv_iq4_xs_f32_c" #NC)]] \
kernel void kernel_mul_mv_iq4_xs_f32_c##NC( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_iq4_xs_f32_nc_impl<N_R0_IQ4_XS, NC>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg); \
}

IQ4_XS_MV_NC(2)
IQ4_XS_MV_NC(3)

template<int NR0, typename args_t>
void kernel_mul_mv_iq4_xs_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;
    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_xs * x = (device const block_iq4_xs *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK_K;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/16;
    const short ix = tiisg/16;  // 0...NG-1
    const short it = tiisg%16;  // 0...15
    const short ib = it/2;
    const short il = it%2;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix * QK_K + ib * 32 + il * 8;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ibl = ix; ibl < nb && ibl < ns01; ibl += NG) {
        device const float4 * y4 = (device const float4 *)yb;
        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        for (short row = 0; row < NR0; ++row) {
            device const block_iq4_xs & xb = x[row*ns01 + ibl];
            device const uint32_t * q4 = (device const uint32_t *)(xb.qs + 16*ib + 8*il);

            float4 acc1 = {0.f}, acc2 = {0.f};

            aux32[0] = (q4[0]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[0] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[0] * qf1;
            acc2 += yl[1] * qf2;

            aux32[0] = (q4[1]     ) & 0x0f0f0f0f;
            aux32[1] = (q4[1] >> 4) & 0x0f0f0f0f;
            qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
            qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
            acc1 += yl[2] * qf1;
            acc2 += yl[3] * qf2;

            acc1 += acc2;

            const int ls = (((xb.scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((xb.scales_h >> 2*ib) & 3) << 4)) - 32;
            sumf[row] += (float)xb.d * ls * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_iq4_xs_f32")]]
kernel void kernel_mul_mv_iq4_xs_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<N_R0_IQ4_XS, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32_r8")]]
kernel void kernel_mul_mv_iq4_xs_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// decode the E2M1 value in the low nibble of n, scaled by 2^-14, without a lookup table: the 3 magnitude bits
// land in the fp16 exponent/mantissa (e=0 gives the 0.5 subnormal). same decode as MLX's fp4_e2m1
static inline half mxfp4_dq1(uint32_t n) {
    const half c = as_type<half>(ushort((n & 7) << 9));
    return (n & 8) ? -c : c;
}

// low / high nibbles of the 4 bytes of w, in byte order, scaled by 2^-14
static inline float4 mxfp4_lo4(uint32_t w) {
    return float4(mxfp4_dq1(w), mxfp4_dq1(w >> 8), mxfp4_dq1(w >> 16), mxfp4_dq1(w >> 24));
}

static inline float4 mxfp4_hi4(uint32_t w) {
    return float4(mxfp4_dq1(w >> 4), mxfp4_dq1(w >> 12), mxfp4_dq1(w >> 20), mxfp4_dq1(w >> 28));
}

template<int NR0, typename args_t>
void kernel_mul_mv_mxfp4_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    // block groups by real simd width: 16 on wave32, 32 on wave64
    const short NG = FC_mul_mv_nw/2;

    const short ix = tiisg/2;  // 0...NG-1
    const short it = tiisg%2;  // 0 or 1

    float4 yl[4];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            // blocks are 17 bytes, so the 8 quant bytes are only byte aligned: read the aligned words that cover them
            // and funnel-shift. the third word is read only when needed so no word past the block end is touched
            device const uint32_t * pw = (device const uint32_t *)((ulong) q2 & ~3ul);
            const uint sh = ((uint)(ulong) q2 & 3u)*8;
            const uint32_t a = pw[0];
            const uint32_t b = pw[1];
            const uint32_t c = pw[sh ? 2 : 1];
            const uint32_t w0 = (a >> sh) | ((b << 1) << (31 - sh));
            const uint32_t w1 = (b >> sh) | ((c << 1) << (31 - sh));

            float4 acc1 = yl[0]*mxfp4_lo4(w0);
            float4 acc2 = yl[1]*mxfp4_hi4(w0);
            float4 acc3 = yl[2]*mxfp4_lo4(w1);
            float4 acc4 = yl[3]*mxfp4_hi4(w1);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            // 16384 undoes the 2^-14 of the decode
            sumf[row] += (e8m0_to_fp32(xb.e)*16384.0f) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += NG * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_mxfp4_f32")]]
kernel void kernel_mul_mv_mxfp4_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_tq2_0_f32_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const float * y = (device const float *) (src1 + offset1);

    device const block_tq2_0 * ax[nr0];
    for (int row = 0; row < nr0; ++row) {
        const uint64_t offset0 = (first_row + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        ax[row] = (device const block_tq2_0 *) ((device char *) src0 + offset0);
    }

    float sumf[nr0] = {0.f};

    // 8 threads per block, NBLOCK blocks per pass, 2 halves per block per pass
    constexpr short NBLOCK = 4;

    constexpr short NB = N_SIMDWIDTH/NBLOCK; // threads per block

    const short blk = tiisg / NB;    // 0..NBLOCK-1, block handled by this thread
    const short htg = tiisg % NB;    // 0..NB-1, thread within block (0..7)

    // byte and y base offsets within the block (32 elements per thread, 4 per byte)
    device const float4 * yb4 = (device const float4 *)(y + 4*htg + blk*QK_K);

    // hoisted per-byte coefficients (from y) and total y-sum, shared across rows
    // ref: https://github.com/ggml-org/llama.cpp/pull/26980
    float4 coef[4];

    for (int ib = blk; ib < nb; ib += NBLOCK) {
        FOR_UNROLL (short h0 = 0; h0 < 2; ++h0) {
            const float4 y0 = yb4[ 0 + 32*h0];
            const float4 y1 = yb4[ 8 + 32*h0];
            const float4 y2 = yb4[16 + 32*h0];
            const float4 y3 = yb4[24 + 32*h0];

            float sumy = 0.f;
            FOR_UNROLL (short j = 0; j < 4; ++j) {
                coef[j] = float4(
                        y0[j],
                        y1[j] - 4.0f*y0[j],
                        y2[j] - 4.0f*y1[j],
                        y3[j] - 4.0f*y2[j]);

                sumy += (y0[j] + y1[j]) + (y2[j] + y3[j]);
            }

            FOR_UNROLL (short row = 0; row < nr0; ++row) {
                device const block_tq2_0 & xb = ax[row][ib];
                device const uchar * qs = xb.qs + 4*htg + 32*h0;

                float sum = -sumy;
                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    // express the 2-bit field shifts (v>>2, v>>4, v>>6) as float floor ops
                    const float v = (float)qs[j];

                    const float f0 = v;
                    const float f1 = floor(v*0.25f);    // v>>2
                    const float f2 = floor(v*0.0625);   // v>>4
                    const float f3 = floor(v*0.015625); // v>>6

                    sum += coef[j][0]*f0 + coef[j][1]*f1 + coef[j][2]*f2 + coef[j][3]*f3;
                }

                sumf[row] += xb.d * sum;
            }
        }

        yb4 += QK_K * NBLOCK / 4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0 && first_row + row < args.ne01) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_tq2_0_f32")]]
kernel void kernel_mul_mv_tq2_0_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<N_R0_TQ2_0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

//
// matrix-vector multiplication
//

typedef void (kernel_mul_mv_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg);

typedef void (kernel_mul_mv2_disp_t)(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg);

template<kernel_mul_mv_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, tgpig, tiisg);
}

template<kernel_mul_mv2_disp_t disp_fn>
void mmv_fn(
        ggml_metal_kargs_mul_mv args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiitg,
        ushort tiisg,
        ushort sgitg) {
    disp_fn(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

typedef decltype(mmv_fn<kernel_mul_mv_t_t_disp<half, half, ggml_metal_kargs_mul_mv>>) mul_mv_disp_fn_t;

template<mul_mv_disp_fn_t disp_fn>
kernel void kernel_mul_mv_id(
        constant ggml_metal_kargs_mul_mv_id & args,
        device const char * src0s,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiitg[[thread_index_in_threadgroup]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]],
        ushort3 ntg [[threads_per_threadgroup]]) {
    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    tgpig.z = 0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    const int64_t i11 = idx % args.ne11;
    const int64_t i12 = iid1;

    const int64_t i1 = idx;
    const int64_t i2 = i12;

    device char * dst_cur = dst + (i1*args.ne0 + i2*args.ne1*args.ne0)*sizeof(float);

    if (i02 == -1) {
        device float * dst_f32 = (device float *) dst_cur;
        const ushort nth = ntg.x*ntg.y*ntg.z;
        for (int i0 = tiitg; i0 < args.ne0; i0 += nth) {
            dst_f32[i0] = 0.0f;
        }
        return;
    }

    if (FC_mul_mv_id_skip && i02 < 0) {
        return;
    }

    device const char * src0_cur = src0s + i02*args.nb02;
    device const char * src1_cur = src1  + i11*args.nb11 + i12*args.nb12;

    ggml_metal_kargs_mul_mv args0 = {
        /*.ne00 =*/ args.ne00,
        /*.ne01 =*/ args.ne01,
        /*.ne02 =*/ 1, // args.ne02,
        /*.nb00 =*/ args.nb00,
        /*.nb01 =*/ args.nb01,
        /*.nb02 =*/ args.nb02,
        /*.nb03 =*/ args.nb02, // args.ne02 == 1
        /*.ne10 =*/ args.ne10,
        /*.ne11 =*/ 1, // args.ne11,
        /*.ne12 =*/ 1, // args.ne12,
        /*.nb10 =*/ args.nb10,
        /*.nb11 =*/ args.nb11,
        /*.nb12 =*/ args.nb12,
        /*.nb13 =*/ args.nb12, // ne12 == 1
        /*.ne0  =*/ args.ne0,
        /*.ne1  =*/ 1, // args.ne1,
        /*.nr0  =*/ args.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    disp_fn(
        args0,
        /* src0 */ src0_cur,
        /* src1 */ src1_cur,
        /* dst  */ dst_cur,
        shmem,
        tgpig,
        tiitg,
        tiisg,
        sgitg);
}

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>) kernel_mul_mv_id_t;

typedef decltype(kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>) kernel_mul_mv_id_4_t;

#if defined(GGML_METAL_HAS_BF16)
#endif
#if defined(GGML_METAL_HAS_BF16)
#endif

// byte of the 12-byte scale array held in three registers, no dynamic indexing
static inline uint sbyte(uint s0, uint s1, uint s2, short idx) {
    const uint w = idx < 4 ? s0 : (idx < 8 ? s1 : s2);
    return (w >> (8*(idx & 3))) & 0xFF;
}

static inline uchar2 scale_pair_shift(uint s0, uint s1, uint s2, short j, short k) {
    if (j < 4) {
        return uchar2{uchar(sbyte(s0,s1,s2, j+0+k) & 63), uchar(sbyte(s0,s1,s2, j+4+k) & 63)};
    }
    const uint a = sbyte(s0,s1,s2, j+4+k);
    return uchar2{uchar((a & 0xF) | ((sbyte(s0,s1,s2, j-4+k) & 0xc0) >> 2)),
                  uchar((a >>  4) | ((sbyte(s0,s1,s2, j-0+k) & 0xc0) >> 2))};
}

// same as above but keeping the quants raw: the scale is applied once per chunk on the
// accumulated products instead of dequantizing each value, which holds 4 registers per
// row instead of 16 and shortens the chain into the FMAs. The block header (d, dmin and
// the scale bytes) is read once per block, not once per chunk
template<short r1ptg, short nr0, bool SH>
void kernel_mul_mv_ext_q4_K_raw_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup float4x4 * sh,
        threadgroup float     * shsy,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpb  = 16;
    const short nypsg = (sgptg/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = (tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty)*nr0;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * xq[nr0];

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        xq[ir0] = (i01 + ir0 < args.ne01) ? (device const block_q4_K *) (src0 + offset0 + ir0*args.nb01) + tx/chpb : (device const block_q4_K *) src0;
    }

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[nr0][r1ptg];

#pragma unroll(nr0)
    for (short ir0 = 0; ir0 < nr0; ++ir0) {
#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            sumf[ir0][ir1] = 0.0f;
        }
    }

    short cch = tx%chpb;

    uint4 hdr[nr0];
    bool  need_hdr = true;

    for (int ich = tx; 16*ich < args.ne00; ich += nxpsg) {
        if (need_hdr) {
#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                hdr[ir0] = *(device const uint4 *) xq[ir0];
            }
            need_hdr = false;
        }

        if (SH) {
            // every row group of the threadgroup reads the same activation chunk, and the dmin
            // term needs its plain sum, which does not depend on the row: read and add it once
            threadgroup_barrier(mem_flags::mem_threadgroup);

            for (short j = sgitg*sgptg + tiisg; j < r1ptg*nxpsg; j += sgptg*NSG) {
                const float4x4 v = y4x4[j/nxpsg][(j%nxpsg) - tx];

                const float t = (v[0][0] + v[0][1] + v[0][2] + v[0][3]) +
                                (v[1][0] + v[1][1] + v[1][2] + v[1][3]) +
                                (v[2][0] + v[2][1] + v[2][2] + v[2][3]) +
                                (v[3][0] + v[3][1] + v[3][2] + v[3][3]);

                sh  [j] = v;
                shsy[j] = t;
            }

            threadgroup_barrier(mem_flags::mem_threadgroup);
        }

        const short il  = cch;
        const short ilm = il & 3;
        const int   qoff = (int) (il/4) * 32 + 16 * ((int) il & 1);
        const short is   = (il/4) * 2;

        const ushort mask0 = ilm < 2 ? 0x000F : 0x00F0;
        const ushort mask1 = mask0 << 8;

        uint4 qsv[nr0];
        float dl[nr0], ml[nr0];

#pragma unroll(nr0)
        for (short ir0 = 0; ir0 < nr0; ++ir0) {
            device const block_q4_K * xb = xq[ir0];

            qsv[ir0] = *(device const uint4 *)(xb->qs + qoff);

            const half2  dm = as_type<half2>(hdr[ir0].x);
            const uchar2 sc = scale_pair_shift(hdr[ir0].y, hdr[ir0].z, hdr[ir0].w, is, ilm/2);
            const float  d  = ilm < 2 ? (float) dm[0] : (float) (dm[0] / 16.h);

            dl [ir0] = d * sc[0];
            ml [ir0] = (float) dm[1] * sc[1];
        }

        // unpack every weight of the chunk once, reused across the r1ptg columns instead
        // of re-doing the mask and int-to-float per column
        half w0[nr0][8];
        half w1[nr0][8];

#pragma unroll(nr0)
        for (short ir0 = 0; ir0 < nr0; ++ir0) {
#pragma unroll(8)
            for (short i = 0; i < 8; ++i) {
                const ushort qsi = (i & 1) ? (ushort)(qsv[ir0][i/2] >> 16) : (ushort)(qsv[ir0][i/2] & 0xFFFF);
                w0[ir0][i] = (half)(qsi & mask0);
                w1[ir0][i] = (half)((qsi & mask1) >> 8);
            }
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            const float4x4 ly = SH ? sh[ir1*nxpsg + tx] : y4x4[ir1][0];

            float sy = SH ? shsy[ir1*nxpsg + tx] : 0.0f;
            float s0[nr0];
            float s1[nr0];

#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                s0[ir0] = 0.0f;
                s1[ir0] = 0.0f;
            }

#pragma unroll(8)
            for (short i = 0; i < 8; ++i) {
                const float y0 = ly[i/2][2*(i%2) + 0];
                const float y1 = ly[i/2][2*(i%2) + 1];

                if (!SH) {
                    sy += y0 + y1;
                }

#pragma unroll(nr0)
                for (short ir0 = 0; ir0 < nr0; ++ir0) {
                    s0[ir0] += y0 * (float) w0[ir0][i];
                    s1[ir0] += y1 * (float) w1[ir0][i];
                }
            }

#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                sumf[ir0][ir1] += dl[ir0]*s0[ir0] + dl[ir0]*s1[ir0] - ml[ir0]*sy;
            }
        }

        cch += nxpsg;
        if (cch >= chpb) {
#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                xq[ir0] += cch/chpb;
            }
            cch %= chpb;
            need_hdr = true;
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += nxpsg;
        }
    }

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            if (nxpsg >= 32) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1], 16);
            }
            if (nxpsg >= 16) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  8);
            }
            if (nxpsg >= 8) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  4);
            }
            if (nxpsg >= 4) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  2);
            }
            if (nxpsg >= 2) {
                sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  1);
            }
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                if (i01 + ir0 < args.ne01) {
                    dst_f32[i01 + ir0] = sumf[ir0][ir1];
                }
            }
        }
    }
}

template<short r1ptg, short nr0>
kernel void kernel_mul_mv_ext_q4_K_raw_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    kernel_mul_mv_ext_q4_K_raw_f32_impl<r1ptg, nr0, false>(args, src0, src1, dst, nullptr, nullptr, tgpig, tiisg, sgitg, sgptg);
}

template<short r1ptg, short nr0>
kernel void kernel_mul_mv_ext_q4_K_raw_lds_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    // sized for the widest row split the host can pick
#if __METAL_VERSION__ >= 300
    threadgroup float4x4 sh  [r1ptg*32];
#else
    threadgroup float4 shf[r1ptg*32*4];
    threadgroup float4x4 * sh = (threadgroup float4x4 *) shf;
#endif
    threadgroup float    shsy[r1ptg*32];
    kernel_mul_mv_ext_q4_K_raw_f32_impl<r1ptg, nr0, true>(args, src0, src1, dst, sh, shsy, tgpig, tiisg, sgitg, sgptg);
}

template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<3, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<3, 4>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<4, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<4, 4>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<5, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_lds_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_lds_f32_disp<5, 4>;

template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_2")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<2, 1>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<2, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<2, 4>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_3")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<3, 1>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<3, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<3, 4>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_4")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<4, 1>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<4, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<4, 4>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_5")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<5, 1>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<5, 2>;
template [[host_name("kernel_mul_mv_ext_q4_K_raw_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4_K_raw_f32_disp<5, 4>;

// ext matvec caching the block header once per block (nxpsg=8 does two chunks of the same
// 256-block). q5_K's 176-byte block is 16-aligned so its scale window is cached in a uint4;
// q2_K/q3_K blocks are not, so only d/dmin is cached and scales are read per chunk.

static inline void ext_load_hdr(device const block_q5_K * xb, thread uint4 & hdr, thread half2 & dm) {
    hdr = *(device const uint4 *) xb;      // dm in .x, scales[12] in .yzw
    dm  = as_type<half2>(hdr.x);
}

static inline void ext_deq_h(device const block_q5_K * xb, const thread uint4 & hdr, const thread half2 & dm, short il, thread float4x4 & reg) {
    const int qoff = 32 * (int) (il/4) + 16 * ((int) il & 1);
    const int hoff = 16 * ((int) il & 1);
    const uint4 qsv = *(device const uint4 *)(xb->qs + qoff);
    const uint4 qhv = *(device const uint4 *)(xb->qh + hoff);

    const short is  = (il/4) * 2;
    const ushort ul0 = 1 << (il/2);
    const ushort ul1 = ul0 << 8;
    const short ilm = il & 3;
    const uchar2 sc = scale_pair_shift(hdr.y, hdr.z, hdr.w, is, ilm/2);
    const float d   = ilm < 2 ? (float) dm[0] : (float) (dm[0] / 16.h);
    const float min = (float) dm[1];
    const float dl  = d * sc[0];
    const float ml  = min * sc[1];

    const ushort mask0 = ilm < 2 ? 0x000F : 0x00F0;
    const ushort mask1 = mask0 << 8;
    const float  dl2   = dl / 256.f;
    const float  dqh   = dl * (ilm < 2 ? 16.f : 256.f);

    for (int i = 0; i < 8; ++i) {
        const ushort qsi = (i & 1) ? (ushort)(qsv[i/2] >> 16) : (ushort)(qsv[i/2] & 0xFFFF);
        const ushort qhi = (i & 1) ? (ushort)(qhv[i/2] >> 16) : (ushort)(qhv[i/2] & 0xFFFF);
        reg[i/2][2*(i%2) + 0] = dl  * (qsi & mask0) + (qhi & ul0 ? dqh : 0.f) - ml;
        reg[i/2][2*(i%2) + 1] = dl2 * (qsi & mask1) + (qhi & ul1 ? dqh : 0.f) - ml;
    }
}

static inline void ext_load_hdr(device const block_q2_K * xb, thread uint4 & hdr, thread half2 & dm) {
    hdr = uint4(0);   // q2_K is 84 bytes, not 16-aligned: read scales per chunk
    dm  = half2(xb->d, xb->dmin);
}

static inline void ext_deq_h(device const block_q2_K * xb, const thread uint4 & hdr, const thread half2 & dm, short il, thread float4x4 & reg) {
    const float d   = (float) dm[0];
    const float min = (float) dm[1];
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    const uint8_t sc = xb->scales[il];

    const int qoff = 32*((int) il/8) + 16*((int) il & 1);
    device const uint16_t * qv = (device const uint16_t *)(q + qoff);
    il = (il/2) % 4;

    const half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    const float dl = d * (sc & 0xF) * coef;
    const float ml = min * (sc >> 4);

    const ushort mask1 = (ushort) mask << 8;
    const float  dl2   = dl / 256.f;

    for (int i = 0; i < 8; ++i) {
        reg[i/2][2*(i%2) + 0] = dl  * (qv[i] & mask)  - ml;
        reg[i/2][2*(i%2) + 1] = dl2 * (qv[i] & mask1) - ml;
    }
}

static inline void ext_load_hdr(device const block_q3_K * xb, thread uint4 & hdr, thread half2 & dm) {
    hdr = uint4(0);   // q3_K is 110 bytes, not 16-aligned: read scales per chunk
    dm.x = xb->d;
}

static inline void ext_deq_h(device const block_q3_K * xb, const thread uint4 & hdr, const thread half2 & dm, short il, thread float4x4 & reg) {
    const half d_all = dm[0];
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t  * scales = (device const int8_t *)xb->scales;

    const int qoff = 32 * ((int) il/8) + 16 * ((int) il & 1);
    const int hoff = 16 * ((int) il & 1);
    device const uint16_t * qv = (device const uint16_t *)(q + qoff);
    device const uint16_t * hv = (device const uint16_t *)(h + hoff);
    const ushort m0 = 1 << (il/2);
    const ushort m1 = m0 << 8;
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : ((il/4)>0 ? 12 : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8];
    uint16_t scale_1 = scales[8 + il%4];
    int16_t  dl_int = (il/4)&1 ? (scale_2&kmask2) | ((scale_1&kmask1) << 2)
                               : (scale_2&kmask2) | ((scale_1&kmask1) << 4);
    float dl = il<8 ? d_all * (dl_int - 32.f) : d_all * (dl_int / 16.f - 32.f);
    const float ml = 4.f * dl;

    il = (il/2) & 3;
    const half    coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    const uint8_t mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl *= coef;

    const ushort mask1 = (ushort)mask << 8;
    const float  dl2   = dl / 256.f;

    for (int i = 0; i < 8; ++i) {
        const ushort hi = hv[i];
        reg[i/2][2*(i%2) + 0] = dl  * (qv[i] & mask)  - (hi & m0 ? 0.f : ml);
        reg[i/2][2*(i%2) + 1] = dl2 * (qv[i] & mask1) - (hi & m1 ? 0.f : ml);
    }
}

template<short r1ptg, short nr0, typename q_t>
void kernel_mul_mv_ext_qK_hdr_f32_impl(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    const short NSG   = FC_mul_mv_nsg;
    const short nxpsg = FC_mul_mv_nxpsg;

    const short chpb  = 16;
    const short nypsg = (sgptg/nxpsg);

    const short tx = tiisg%nxpsg;
    const short ty = tiisg/nxpsg;

    const int i01 = (tgpig.x*(nypsg*NSG) + nypsg*sgitg + ty)*nr0;
    const int i11 = tgpig.y*r1ptg;
    const int i1m = tgpig.z;

    const int i12 = i1m%FC_mul_mv_ne12;
    const int i13 = i1m/FC_mul_mv_ne12;

    const uint64_t offset0 = i01*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const q_t * xq[nr0];

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        xq[ir0] = (i01 + ir0 < args.ne01) ? (device const q_t *) (src0 + offset0 + ir0*args.nb01) + tx/chpb : (device const q_t *) src0;
    }

    device const float4x4 * y4x4[r1ptg];

    for (int ir1 = 0; ir1 < r1ptg; ++ir1) {
        y4x4[ir1] = (i11 + ir1 < args.ne11) ? (device const float4x4 *) (src1 + offset1 + ir1*args.nb11) + tx : (device const float4x4 *) src1;
    }

    float sumf[nr0][r1ptg];

#pragma unroll(nr0)
    for (short ir0 = 0; ir0 < nr0; ++ir0) {
#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            sumf[ir0][ir1] = 0.0f;
        }
    }

    short cch = tx%chpb;

    uint4 hdr[nr0];
    half2 dm [nr0];
    bool  need_hdr = true;

    for (int ich = tx; 16*ich < args.ne00; ich += nxpsg) {
        if (need_hdr) {
#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                ext_load_hdr(xq[ir0], hdr[ir0], dm[ir0]);
            }
            need_hdr = false;
        }

        float4x4 lx[nr0];

#pragma unroll(nr0)
        for (short ir0 = 0; ir0 < nr0; ++ir0) {
            ext_deq_h(xq[ir0], hdr[ir0], dm[ir0], cch, lx[ir0]);
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            const float4x4 ly = y4x4[ir1][0];

#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                sumf[ir0][ir1] +=
                    dot(lx[ir0][0], ly[0]) +
                    dot(lx[ir0][1], ly[1]) +
                    dot(lx[ir0][2], ly[2]) +
                    dot(lx[ir0][3], ly[3]);
            }
        }

        cch += nxpsg;
        if (cch >= chpb) {
#pragma unroll(nr0)
            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                xq[ir0] += cch/chpb;
            }
            cch %= chpb;
            need_hdr = true;
        }

#pragma unroll(r1ptg)
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            y4x4[ir1] += nxpsg;
        }
    }

    for (short ir0 = 0; ir0 < nr0; ++ir0) {
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            if (nxpsg >= 32) { sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1], 16); }
            if (nxpsg >= 16) { sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  8); }
            if (nxpsg >=  8) { sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  4); }
            if (nxpsg >=  4) { sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  2); }
            if (nxpsg >=  2) { sumf[ir0][ir1] += simd_shuffle_down(sumf[ir0][ir1],  1); }
        }
    }

    if (tx == 0) {
        for (short ir1 = 0; ir1 < r1ptg && i11 + ir1 < args.ne11; ++ir1) {
            device float * dst_f32 = (device float *) dst + (uint64_t)i1m*args.ne0*args.ne1 + (uint64_t)(i11 + ir1)*args.ne0;

            for (short ir0 = 0; ir0 < nr0; ++ir0) {
                if (i01 + ir0 < args.ne01) {
                    dst_f32[i01 + ir0] = sumf[ir0][ir1];
                }
            }
        }
    }
}

template<short r1ptg, short nr0, typename q_t>
kernel void kernel_mul_mv_ext_qK_hdr_f32_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    kernel_mul_mv_ext_qK_hdr_f32_impl<r1ptg, nr0, q_t>(args, src0, src1, dst, tgpig, tiisg, sgitg, sgptg);
}

template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_2")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 1, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 2, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 4, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_3")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 1, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 2, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 4, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_4")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 1, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 2, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 4, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_5")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 1, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 2, block_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_hdr_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 4, block_q5_K>;


template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_2")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 1, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 2, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 4, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_3")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 1, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 2, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 4, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_4")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 1, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 2, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 4, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_5")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 1, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 2, block_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_hdr_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 4, block_q2_K>;

template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_2")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 1, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 2, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<2, 4, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_3")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 1, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 2, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<3, 4, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_4")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 1, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 2, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<4, 4, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_5")]]       kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 1, block_q3_K>;

[[host_name("kernel_mul_mv_q8_0_f32_r4")]]
kernel void kernel_mul_mv_q8_0_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q8_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_r8")]]
kernel void kernel_mul_mv_q8_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q8_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_r8")]]
kernel void kernel_mul_mv_q2_K_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q3_K_f32_r8")]]
kernel void kernel_mul_mv_q3_K_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_r4")]]
kernel void kernel_mul_mv_q4_K_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_r8")]]
kernel void kernel_mul_mv_q5_K_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_r4")]]
kernel void kernel_mul_mv_q6_K_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_qvk_q4_K_q6_K_q4_K_f32")]]
kernel void kernel_mul_mv_qvk_q4_K_q6_K_q4_K_f32(
        constant ggml_metal_kargs_mul_mv_qvk & args,
        device const char * wq,
        device const char * wv,
        device const char * wk,
        device const char * src,
        device       char * dst_q,
        device       char * dst_v,
        device       char * dst_k,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    uint3 local = tgpig;

    if (local.x < args.nwg_q) {
        kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
            args.q, wq, src, dst_q, nullptr, local, tiisg, sgitg);
        return;
    }

    local.x -= args.nwg_q;
    if (local.x < args.nwg_v) {
        kernel_mul_mv_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(
            args.v, wv, src, dst_v, nullptr, local, tiisg, sgitg);
        return;
    }

    local.x -= args.nwg_v;
    kernel_mul_mv_q4_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(
        args.k, wk, src, dst_k, nullptr, local, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32")]]
kernel void kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32(
        constant ggml_metal_kargs_mul_mv_qvk & args,
        device const char * wq,
        device const char * wv,
        device const char * wk,
        device const char * src,
        device       char * dst_q,
        device       char * dst_v,
        device       char * dst_k,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    uint3 local = tgpig;

    if (local.x < args.nwg_q) {
        mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(
            args.q, wq, src, dst_q, nullptr, local, tiisg, sgitg);
        return;
    }

    local.x -= args.nwg_q;
    if (local.x < args.nwg_v) {
        mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(
            args.v, wv, src, dst_v, nullptr, local, tiisg, sgitg);
        return;
    }

    local.x -= args.nwg_v;
    mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0, constant ggml_metal_kargs_mul_mv &>(
        args.k, wk, src, dst_k, nullptr, local, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_qvk_q2_K_q4_K_q2_K_f32")]]
kernel void kernel_mul_mv_qvk_q2_K_q4_K_q2_K_f32(
        constant ggml_metal_kargs_mul_mv_qvk & args,
        device const char * wq, device const char * wv, device const char * wk, device const char * src,
        device char * dst_q, device char * dst_v, device char * dst_k,
        uint3 tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    uint3 local = tgpig;
    if (local.x < args.nwg_q) {
        kernel_mul_mv_q2_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args.q, wq, src, dst_q, nullptr, local, tiisg, sgitg);
        return;
    }
    local.x -= args.nwg_q;
    if (local.x < args.nwg_v) {
        kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args.v, wv, src, dst_v, nullptr, local, tiisg, sgitg);
        return;
    }
    local.x -= args.nwg_v;
    kernel_mul_mv_q2_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args.k, wk, src, dst_k, nullptr, local, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32_r8")]]
kernel void kernel_mul_mv_iq3_xxs_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template [[host_name("kernel_mul_mv_ext_f32_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f32_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, float4,       4,  dequantize_f32_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_2")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_3")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_4")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, half4,        4,  dequantize_f16_t4>;
template [[host_name("kernel_mul_mv_ext_f16_f32_r1_5")]]    kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, half4,        4,  dequantize_f16_t4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, bfloat4,      4,  dequantize_bf16_t4>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, bfloat4,      4,  dequantize_bf16_t4>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, bfloat4,      4,  dequantize_bf16_t4>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_ext_bf16_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, bfloat4,      4,  dequantize_bf16_t4>;
#endif
template [[host_name("kernel_mul_mv_ext_bf16e_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, bf16e4,       4,  dequantize_bf16e_t4>;
template [[host_name("kernel_mul_mv_ext_bf16e_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, bf16e4,       4,  dequantize_bf16e_t4>;
template [[host_name("kernel_mul_mv_ext_bf16e_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, bf16e4,       4,  dequantize_bf16e_t4>;
template [[host_name("kernel_mul_mv_ext_bf16e_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, bf16e4,       4,  dequantize_bf16e_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q1_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q1_0,   128, dequantize_q1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_pq2_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_pq2_0,    128, dequantize_pq2_0_t4>;
template [[host_name("kernel_mul_mv_ext_ptq1_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_ptq1_0,    128, dequantize_ptq1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_pq2_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_pq2_0,    128, dequantize_pq2_0_t4>;
template [[host_name("kernel_mul_mv_ext_ptq1_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_ptq1_0,    128, dequantize_ptq1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_pq2_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_pq2_0,    128, dequantize_pq2_0_t4>;
template [[host_name("kernel_mul_mv_ext_ptq1_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_ptq1_0,    128, dequantize_ptq1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q2_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q2_0,    64, dequantize_q2_0_t4>;
template [[host_name("kernel_mul_mv_ext_pq2_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_pq2_0,    128, dequantize_pq2_0_t4>;
template [[host_name("kernel_mul_mv_ext_ptq1_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_ptq1_0,    128, dequantize_ptq1_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_0,   32, dequantize_q4_0_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q4_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q4_1,   32, dequantize_q4_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_0,   32, dequantize_q5_0_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q5_1_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q5_1,   32, dequantize_q5_1_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5")]]   kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_2_w64")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_q8_0,   32, dequantize_q8_0_t4_h2>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_3_w64")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_q8_0,   32, dequantize_q8_0_t4_h2>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_4_w64")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_q8_0,   32, dequantize_q8_0_t4_h2>;
template [[host_name("kernel_mul_mv_ext_q8_0_f32_r1_5_w64")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_q8_0,   32, dequantize_q8_0_t4_h2>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_2")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_3")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_4")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_mxfp4_f32_r1_5")]]  kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_mxfp4,  32, dequantize_mxfp4_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_2")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<2, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_3")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<3, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_4")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<4, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_iq4_nl_f32_r1_5")]] kernel mul_mv_ext_q4_f32_t kernel_mul_mv_ext_q4_f32_disp<5, block_iq4_nl, 32, dequantize_iq4_nl_t4>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 2, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q4_K_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 4, block_q4_K, 256, dequantize_q4_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 2, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q5_K_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 4, block_q5_K, 256, dequantize_q5_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 2, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_q6_K, 256, dequantize_q6_K>;

// RX6800: IQ types on the multi-column path too. Without it a 2..8-token batch (spec-decode verify)
// runs one mat-vec per column, re-dequantizing the weights each time; decode here is dequant-bound.
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq2_xxs, 256, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq2_xxs, 256, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq2_xxs, 256, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xxs_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq2_xxs, 256, dequantize_iq2_xxs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq2_xs, 256, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq2_xs, 256, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq2_xs, 256, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_xs_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq2_xs, 256, dequantize_iq2_xs>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq2_s, 256, dequantize_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq2_s, 256, dequantize_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq2_s, 256, dequantize_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq2_s_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq2_s, 256, dequantize_iq2_s>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq3_xxs, 256, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq3_xxs, 256, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq3_xxs, 256, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_xxs_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq3_xxs, 256, dequantize_iq3_xxs>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq3_s, 256, dequantize_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq3_s, 256, dequantize_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq3_s, 256, dequantize_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq3_s_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq3_s, 256, dequantize_iq3_s>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_iq4_xs, 256, dequantize_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_iq4_xs, 256, dequantize_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_iq4_xs, 256, dequantize_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_iq4_xs_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_iq4_xs, 256, dequantize_iq4_xs>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 2, block_q6_K, 256, dequantize_q6_K>;

// q6_K batched matvec for 64-lane simdgroups; the generic ext path leaves GCN latency-bound (~45 GB/s vs ~265 single-column).
// Same block layout as kernel_mul_mv_q6_K_f32_r4, plus one accumulator per src1 row so the weights are read once.
// With pf set, the next block's weight words are staged while the current block computes: HBM latency, not bandwidth, is the wall.
// pf pays where the register file has headroom (r1ptg 2) or where the plain form spills (r1ptg 5); measured on gfx906.
template<short r1ptg, short nr0, bool pf>
kernel void kernel_mul_mv_ext_q6_K_f32_w64_disp(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {

    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0  = tgpig.x;
    const int i11 = tgpig.y*r1ptg;
    const int im  = tgpig.z;

    const int first_row = (r0*NSG + sgitg)*nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x  = (device const block_q6_K *) (src0 + offset0);
    device const char       * yb = src1 + offset1;

    float sumf[r1ptg][nr0];

#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            sumf[ir1][row] = 0.0f;
        }
    }

    const short NG  = sgptg/16;
    const short tid = tiisg%16;
    const short ix  = tiisg/16;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    // weight words for the block in flight, one set per row
    uint pw1[nr0], pw2[nr0], pwh[nr0];

    const int i0 = ix < nb ? ix : 0;   // lanes past nb never loop; keep the prologue in bounds

    if (pf) {
#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            device const block_q6_K * xr = (first_row + row < args.ne01)
                ? (device const block_q6_K *) ((device const char *) x + row*args.nb01)
                : (device const block_q6_K *) src0;

            device const ushort * s1 = (device const ushort *) (xr[i0].ql + q_offset_l);
            device const ushort * s2 = (device const ushort *) (s1 + 16);
            device const ushort * sh = (device const ushort *) (xr[i0].qh + q_offset_h);

            pw1[row] = (uint) s1[0] | ((uint) s1[1] << 16);
            pw2[row] = (uint) s2[0] | ((uint) s2[1] << 16);
            pwh[row] = (uint) sh[0] | ((uint) sh[1] << 16);
        }
    }

    for (int i = ix; i < nb; i += NG) {
        // activation tiles for this block, hoisted out of the row loop and read as float4
        float4 y4[r1ptg][4];

#pragma unroll
        for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
            device const float4 * yc = (device const float4 *) (yb + (i11 + ir1 < args.ne11 ? ir1*args.nb11 : 0));
            device const float4 * y  = yc + i*(QK_K/4) + y_offset/4;

            FOR_UNROLL (short j = 0; j < 4; ++j) {
                y4[ir1][j] = y[j*8];
            }
        }

        // stage the next block's weight words before computing this one: the loads
        // then have a full iteration of compute to complete. The last iteration
        // restages the final block (L1 hit) instead of branching
        uint nw1[nr0], nw2[nr0], nwh[nr0];

        if (pf) {
            const int inext = i + NG < nb ? i + NG : nb - 1;

#pragma unroll
            for (short row = 0; row < nr0; ++row) {
                device const block_q6_K * xr = (first_row + row < args.ne01)
                    ? (device const block_q6_K *) ((device const char *) x + row*args.nb01)
                    : (device const block_q6_K *) src0;

                device const ushort * s1 = (device const ushort *) (xr[inext].ql + q_offset_l);
                device const ushort * s2 = (device const ushort *) (s1 + 16);
                device const ushort * sh = (device const ushort *) (xr[inext].qh + q_offset_h);

                nw1[row] = (uint) s1[0] | ((uint) s1[1] << 16);
                nw2[row] = (uint) s2[0] | ((uint) s2[1] << 16);
                nwh[row] = (uint) sh[0] | ((uint) sh[1] << 16);
            }
        }

#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            // rows past ne01 fall back to row zero; their sums are dropped at the store
            device const block_q6_K * xr = (first_row + row < args.ne01)
                ? (device const block_q6_K *) ((device const char *) x + row*args.nb01)
                : (device const block_q6_K *) src0;

            device const int8_t  * sc = xr[i].scales + is;
            device const half    * dh = &xr[i].d;

            uint w1, w2, wh;

            if (pf) {
                w1 = pw1[row];
                w2 = pw2[row];
                wh = pwh[row];
            } else {
                device const ushort * s1 = (device const ushort *) (xr[i].ql + q_offset_l);
                device const ushort * s2 = (device const ushort *) (s1 + 16);
                device const ushort * sh = (device const ushort *) (xr[i].qh + q_offset_h);

                w1 = (uint) s1[0] | ((uint) s1[1] << 16);
                w2 = (uint) s2[0] | ((uint) s2[1] << 16);
                wh = (uint) sh[0] | ((uint) sh[1] << 16);
            }

            // int-to-float via the exponent trick: full-rate ops, exact for v in [0, 63]
            float4 v4[4];

            FOR_UNROLL (short l = 0; l < 4; ++l) {
                const uint8_t b1 = (uint8_t)(w1 >> (8*l));
                const uint8_t b2 = (uint8_t)(w2 >> (8*l));
                const uint8_t bh = (uint8_t)(wh >> (8*l));

                v4[0][l] = as_type<float>(0x4B000000u | (uint)((b1 & 0xF) | ((bh & kmask1) << 4))) - 8388640.0f;
                v4[1][l] = as_type<float>(0x4B000000u | (uint)((b2 & 0xF) | ((bh & kmask2) << 2))) - 8388640.0f;
                v4[2][l] = as_type<float>(0x4B000000u | (uint)((b1  >> 4) | ((bh & kmask3) << 0))) - 8388640.0f;
                v4[3][l] = as_type<float>(0x4B000000u | (uint)((b2  >> 4) | ((bh & kmask4) >> 2))) - 8388640.0f;
            }

            const float4 dsc = (float) dh[0] * (float4) { (float) sc[0], (float) sc[2], (float) sc[4], (float) sc[6] };

#pragma unroll
            for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
                float4 sums = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    sums[j] = dot(y4[ir1][j], v4[j]);
                }

                sumf[ir1][row] += sums[0]*dsc[0] + sums[1]*dsc[1] + sums[2]*dsc[2] + sums[3]*dsc[3];
            }
        }

        if (pf) {
#pragma unroll
            for (short row = 0; row < nr0; ++row) {
                pw1[row] = nw1[row];
                pw2[row] = nw2[row];
                pwh[row] = nwh[row];
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)i11*args.ne0;

#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            const float sum_all = simd_sum(sumf[ir1][row]);
            if (tiisg == 0 && i11 + ir1 < args.ne11 && first_row + row < args.ne01) {
                dst_f32[ir1*args.ne0 + first_row + row] = sum_all;
            }
        }
    }
}

typedef decltype(kernel_mul_mv_ext_q6_K_f32_w64_disp<2, 4, true>) mul_mv_ext_q6_K_w64_t;

template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_2_w64")]] kernel mul_mv_ext_q6_K_w64_t kernel_mul_mv_ext_q6_K_f32_w64_disp<2, 4, true>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_3_w64")]] kernel mul_mv_ext_q6_K_w64_t kernel_mul_mv_ext_q6_K_f32_w64_disp<3, 4, false>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_4_w64")]] kernel mul_mv_ext_q6_K_w64_t kernel_mul_mv_ext_q6_K_f32_w64_disp<4, 4, false>;
template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5_w64")]] kernel mul_mv_ext_q6_K_w64_t kernel_mul_mv_ext_q6_K_f32_w64_disp<5, 4, true>;
// eight src1 rows in one weight pass: dequantized block values live across both
// four-column activation windows, so the register cost stays near the four-column
// kernel while the second group no longer re-reads the weights. Two rows per lane.
[[host_name("kernel_mul_mv_ext_q6_K_f32_r1_8_w64")]]
kernel void kernel_mul_mv_ext_q6_K_f32_w64x8(
        constant ggml_metal_kargs_mul_mv_ext & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort  tiisg[[thread_index_in_simdgroup]],
        ushort  sgitg[[simdgroup_index_in_threadgroup]],
        ushort  sgptg[[threads_per_simdgroup]]) {
    constexpr short nr0  = 2;
    constexpr short r1ptg = 8;
    constexpr short cch  = 4;   // columns per activation window

    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0  = tgpig.x;
    const int i11 = tgpig.y*r1ptg;
    const int im  = tgpig.z;

    const int first_row = (r0*NSG + sgitg)*nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        i11*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x  = (device const block_q6_K *) (src0 + offset0);
    device const char       * yb = src1 + offset1;

    float sumf[r1ptg][nr0];

#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            sumf[ir1][row] = 0.0f;
        }
    }

    const short NG  = sgptg/16;
    const short tid = tiisg%16;
    const short ix  = tiisg/16;
    const short ip  = tid/8;
    const short il  = tid%8;
    const short l0  = 4*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += NG) {
        // dequant the block for both rows before the activation windows load
        float4 v4r[nr0][4];
        float4 dscr[nr0];

#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            // rows past ne01 fall back to row zero; their sums are dropped at the store
            device const block_q6_K * xr = (first_row + row < args.ne01)
                ? (device const block_q6_K *) ((device const char *) x + row*args.nb01)
                : (device const block_q6_K *) src0;

            device const uint8_t * q1 = xr[i].ql + q_offset_l;
            device const uint8_t * q2 = q1 + 32;
            device const uint8_t * qh = xr[i].qh + q_offset_h;
            device const int8_t  * sc = xr[i].scales + is;
            device const half    * dh = &xr[i].d;

            device const ushort * s1 = (device const ushort *) q1;
            device const ushort * s2 = (device const ushort *) q2;
            device const ushort * sh = (device const ushort *) qh;

            const uint w1 = (uint) s1[0] | ((uint) s1[1] << 16);
            const uint w2 = (uint) s2[0] | ((uint) s2[1] << 16);
            const uint wh = (uint) sh[0] | ((uint) sh[1] << 16);

            // int-to-float via the exponent trick: full-rate ops, exact for v in [0, 63]
            FOR_UNROLL (short l = 0; l < 4; ++l) {
                const uint8_t b1 = (uint8_t)(w1 >> (8*l));
                const uint8_t b2 = (uint8_t)(w2 >> (8*l));
                const uint8_t bh = (uint8_t)(wh >> (8*l));

                v4r[row][0][l] = as_type<float>(0x4B000000u | (uint)((b1 & 0xF) | ((bh & kmask1) << 4))) - 8388640.0f;
                v4r[row][1][l] = as_type<float>(0x4B000000u | (uint)((b2 & 0xF) | ((bh & kmask2) << 2))) - 8388640.0f;
                v4r[row][2][l] = as_type<float>(0x4B000000u | (uint)((b1  >> 4) | ((bh & kmask3) << 0))) - 8388640.0f;
                v4r[row][3][l] = as_type<float>(0x4B000000u | (uint)((b2  >> 4) | ((bh & kmask4) >> 2))) - 8388640.0f;
            }

            dscr[row] = (float) dh[0] * (float4) { (float) sc[0], (float) sc[2], (float) sc[4], (float) sc[6] };
        }

#pragma unroll
        for (short c0 = 0; c0 < r1ptg; c0 += cch) {
            float4 y4[cch][4];

#pragma unroll
            for (short ir1 = 0; ir1 < cch; ++ir1) {
                device const float4 * yc = (device const float4 *) (yb + (i11 + c0 + ir1 < args.ne11 ? (uint64_t)(c0 + ir1)*args.nb11 : 0));
                device const float4 * y  = yc + i*(QK_K/4) + y_offset/4;

                FOR_UNROLL (short j = 0; j < 4; ++j) {
                    y4[ir1][j] = y[j*8];
                }
            }

#pragma unroll
            for (short row = 0; row < nr0; ++row) {
#pragma unroll
                for (short ir1 = 0; ir1 < cch; ++ir1) {
                    float4 sums = {0.f, 0.f, 0.f, 0.f};

                    FOR_UNROLL (short j = 0; j < 4; ++j) {
                        sums[j] = dot(y4[ir1][j], v4r[row][j]);
                    }

                    sumf[c0 + ir1][row] += sums[0]*dscr[row][0] + sums[1]*dscr[row][1] + sums[2]*dscr[row][2] + sums[3]*dscr[row][3];
                }
            }
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)i11*args.ne0;

#pragma unroll
    for (short ir1 = 0; ir1 < r1ptg; ++ir1) {
#pragma unroll
        for (short row = 0; row < nr0; ++row) {
            const float sum_all = simd_sum(sumf[ir1][row]);
            if (tiisg == 0 && i11 + ir1 < args.ne11 && first_row + row < args.ne01) {
                dst_f32[ir1*args.ne0 + first_row + row] = sum_all;
            }
        }
    }
}


template [[host_name("kernel_mul_mv_ext_q6_K_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 4, block_q6_K, 256, dequantize_q6_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 2, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q2_K_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 4, block_q2_K, 256, dequantize_q2_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 1, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_2_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<2, 4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 1, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_3_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<3, 4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 1, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_4_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<4, 4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 1, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 2, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_q4x4_f32_disp<5, 4, block_q3_K, 256, dequantize_q3_K>;
template [[host_name("kernel_mul_mv_f32_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<float, float>;
template [[host_name("kernel_mul_mv_f16_f32")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16")]]   kernel mul_mv_t_t kernel_mul_mv_t_t<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, float>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_bf16")]] kernel mul_mv_t_t kernel_mul_mv_t_t<bfloat, bfloat>;
#endif
template [[host_name("kernel_mul_mv_bf16e_f32")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<bf16e, float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_f32_bf16")]]  kernel mul_mv_t_t kernel_mul_mv_t_t<float,  bfloat>;
#endif
template [[host_name("kernel_mul_mv_f32_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float, float4, float, float4>;
template [[host_name("kernel_mul_mv_f16_f32_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  float, float4>;
template [[host_name("kernel_mul_mv_f16_f16_4")]]   kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<half,  half4,  half,  half4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, float,  float4>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_bf16_4")]] kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bfloat, bfloat4, bfloat, bfloat4>;
#endif
template [[host_name("kernel_mul_mv_bf16e_f32_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<bf16e, bf16e4, float,  float4>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_f32_bf16_4")]]  kernel mul_mv_t_t_4 kernel_mul_mv_t_t_4<float,  float4,  bfloat, bfloat4>;
#endif
template [[host_name("kernel_mul_mv_f32_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float, float>;
template [[host_name("kernel_mul_mv_f16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  float>;
template [[host_name("kernel_mul_mv_f16_f16_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<half,  half>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, float>;
#endif
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_bf16_bf16_short")]] kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bfloat, bfloat>;
#endif
template [[host_name("kernel_mul_mv_bf16e_f32_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<bf16e, float>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_f32_bf16_short")]]  kernel mul_mv_t_t_short_t kernel_mul_mv_t_t_short<float,  bfloat>;
#endif
template [[host_name("kernel_mul_mv_id_f32_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<float, float>>>;
template [[host_name("kernel_mul_mv_id_f16_f32")]]     kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<half,  float>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<bfloat, float>>>;
#endif
template [[host_name("kernel_mul_mv_id_bf16e_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_disp<bf16e, float>>>;
template [[host_name("kernel_mul_mv_id_f32_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<float, float4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_f16_f32_4")]]   kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<half,  half4,  float, float4>>>;
#if defined(GGML_METAL_HAS_BF16)
template [[host_name("kernel_mul_mv_id_bf16_f32_4")]]  kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<bfloat, bfloat4, float, float4>>>;
#endif
template [[host_name("kernel_mul_mv_id_bf16e_f32_4")]]  kernel kernel_mul_mv_id_4_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_t_t_4_disp<bf16e, bf16e4, float, float4>>>;
template [[host_name("kernel_mul_mv_id_q8_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q8_0_f32_impl<N_R0_Q8_0>>>;
template [[host_name("kernel_mul_mv_id_q1_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q1_0_f32_impl<N_R0_Q1_0>>>;
template [[host_name("kernel_mul_mv_id_q2_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_0_f32_impl<N_R0_Q2_0>>>;
template [[host_name("kernel_mul_mv_id_pq2_0_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_pq2_0_f32_impl<N_R0_PQ2_0>>>;
template [[host_name("kernel_mul_mv_id_ptq1_0_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_ptq1_0_f32_impl<N_R0_PTQ1_0>>>;

// Sweep entry points: lanes per block x rows per lane, for the 32-block family.

[[host_name("kernel_mul_mv_q4_0_f32_L2r1")]]
kernel void kernel_mul_mv_q4_0_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L2r2")]]
kernel void kernel_mul_mv_q4_0_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L2r4")]]
kernel void kernel_mul_mv_q4_0_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L2r8")]]
kernel void kernel_mul_mv_q4_0_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L4r1")]]
kernel void kernel_mul_mv_q4_0_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L4r2")]]
kernel void kernel_mul_mv_q4_0_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L4r4")]]
kernel void kernel_mul_mv_q4_0_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L4r8")]]
kernel void kernel_mul_mv_q4_0_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L8r1")]]
kernel void kernel_mul_mv_q4_0_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L8r2")]]
kernel void kernel_mul_mv_q4_0_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L8r4")]]
kernel void kernel_mul_mv_q4_0_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_0_f32_L8r8")]]
kernel void kernel_mul_mv_q4_0_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_0, 8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}





[[host_name("kernel_mul_mv_q4_1_f32_L2r1")]]
kernel void kernel_mul_mv_q4_1_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L2r2")]]
kernel void kernel_mul_mv_q4_1_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L2r4")]]
kernel void kernel_mul_mv_q4_1_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L2r8")]]
kernel void kernel_mul_mv_q4_1_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L4r1")]]
kernel void kernel_mul_mv_q4_1_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L4r2")]]
kernel void kernel_mul_mv_q4_1_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L4r4")]]
kernel void kernel_mul_mv_q4_1_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L4r8")]]
kernel void kernel_mul_mv_q4_1_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L8r1")]]
kernel void kernel_mul_mv_q4_1_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L8r2")]]
kernel void kernel_mul_mv_q4_1_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L8r4")]]
kernel void kernel_mul_mv_q4_1_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_1_f32_L8r8")]]
kernel void kernel_mul_mv_q4_1_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q4_1, 8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}





[[host_name("kernel_mul_mv_q5_0_f32_L2r1")]]
kernel void kernel_mul_mv_q5_0_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L2r2")]]
kernel void kernel_mul_mv_q5_0_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L2r4")]]
kernel void kernel_mul_mv_q5_0_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L2r8")]]
kernel void kernel_mul_mv_q5_0_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L4r1")]]
kernel void kernel_mul_mv_q5_0_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L4r2")]]
kernel void kernel_mul_mv_q5_0_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L4r4")]]
kernel void kernel_mul_mv_q5_0_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L4r8")]]
kernel void kernel_mul_mv_q5_0_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L8r1")]]
kernel void kernel_mul_mv_q5_0_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L8r2")]]
kernel void kernel_mul_mv_q5_0_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L8r4")]]
kernel void kernel_mul_mv_q5_0_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_0_f32_L8r8")]]
kernel void kernel_mul_mv_q5_0_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_0, 8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}





[[host_name("kernel_mul_mv_q5_1_f32_L2r1")]]
kernel void kernel_mul_mv_q5_1_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L2r2")]]
kernel void kernel_mul_mv_q5_1_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L2r4")]]
kernel void kernel_mul_mv_q5_1_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L2r8")]]
kernel void kernel_mul_mv_q5_1_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L4r1")]]
kernel void kernel_mul_mv_q5_1_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L4r2")]]
kernel void kernel_mul_mv_q5_1_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L4r4")]]
kernel void kernel_mul_mv_q5_1_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L4r8")]]
kernel void kernel_mul_mv_q5_1_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L8r1")]]
kernel void kernel_mul_mv_q5_1_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L8r2")]]
kernel void kernel_mul_mv_q5_1_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L8r4")]]
kernel void kernel_mul_mv_q5_1_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_1_f32_L8r8")]]
kernel void kernel_mul_mv_q5_1_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_l_impl<block_q5_1, 8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}






// Sweep entry points: rows per lane, one per type.

[[host_name("kernel_mul_mv_q8_0_f32_r1")]]
kernel void kernel_mul_mv_q8_0_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_r2")]]
kernel void kernel_mul_mv_q8_0_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_r1")]]
kernel void kernel_mul_mv_q2_K_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_r2")]]
kernel void kernel_mul_mv_q2_K_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_r4")]]
kernel void kernel_mul_mv_q2_K_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q2_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q3_K_f32_r1")]]
kernel void kernel_mul_mv_q3_K_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q3_K_f32_r2")]]
kernel void kernel_mul_mv_q3_K_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q3_K_f32_r4")]]
kernel void kernel_mul_mv_q3_K_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q3_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_r1")]]
kernel void kernel_mul_mv_q4_K_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_r2")]]
kernel void kernel_mul_mv_q4_K_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_r8")]]
kernel void kernel_mul_mv_q4_K_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q4_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_r1")]]
kernel void kernel_mul_mv_q5_K_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_r2")]]
kernel void kernel_mul_mv_q5_K_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_r4")]]
kernel void kernel_mul_mv_q5_K_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q5_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_r1")]]
kernel void kernel_mul_mv_q6_K_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_r2")]]
kernel void kernel_mul_mv_q6_K_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_r8")]]
kernel void kernel_mul_mv_q6_K_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_q6_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xxs_f32_r1")]]
kernel void kernel_mul_mv_iq2_xxs_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xxs_f32_r2")]]
kernel void kernel_mul_mv_iq2_xxs_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xxs_f32_r4")]]
kernel void kernel_mul_mv_iq2_xxs_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xxs_f32_r8")]]
kernel void kernel_mul_mv_iq2_xxs_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq2_xxs_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xs_f32_r1")]]
kernel void kernel_mul_mv_iq2_xs_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xs_f32_r2")]]
kernel void kernel_mul_mv_iq2_xs_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xs_f32_r4")]]
kernel void kernel_mul_mv_iq2_xs_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_xs_f32_r8")]]
kernel void kernel_mul_mv_iq2_xs_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_xs_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32_r1")]]
kernel void kernel_mul_mv_iq3_xxs_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32_r2")]]
kernel void kernel_mul_mv_iq3_xxs_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_xxs_f32_r4")]]
kernel void kernel_mul_mv_iq3_xxs_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_xxs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_s_f32_r1")]]
kernel void kernel_mul_mv_iq3_s_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_s_f32_r2")]]
kernel void kernel_mul_mv_iq3_s_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_s_f32_r4")]]
kernel void kernel_mul_mv_iq3_s_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq3_s_f32_r8")]]
kernel void kernel_mul_mv_iq3_s_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq3_s_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_s_f32_r1")]]
kernel void kernel_mul_mv_iq2_s_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_s_f32_r2")]]
kernel void kernel_mul_mv_iq2_s_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_s_f32_r4")]]
kernel void kernel_mul_mv_iq2_s_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq2_s_f32_r8")]]
kernel void kernel_mul_mv_iq2_s_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq2_s_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_s_f32_r1")]]
kernel void kernel_mul_mv_iq1_s_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_s_f32_r2")]]
kernel void kernel_mul_mv_iq1_s_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_s_f32_r4")]]
kernel void kernel_mul_mv_iq1_s_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_s_f32_r8")]]
kernel void kernel_mul_mv_iq1_s_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_s_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32_r1")]]
kernel void kernel_mul_mv_iq1_m_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32_r2")]]
kernel void kernel_mul_mv_iq1_m_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32_r4")]]
kernel void kernel_mul_mv_iq1_m_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq1_m_f32_r8")]]
kernel void kernel_mul_mv_iq1_m_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq1_m_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_r1")]]
kernel void kernel_mul_mv_iq4_nl_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_r2")]]
kernel void kernel_mul_mv_iq4_nl_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_r4")]]
kernel void kernel_mul_mv_iq4_nl_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_nl_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32_r1")]]
kernel void kernel_mul_mv_iq4_xs_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32_r2")]]
kernel void kernel_mul_mv_iq4_xs_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_xs_f32_r4")]]
kernel void kernel_mul_mv_iq4_xs_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_iq4_xs_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_r1")]]
kernel void kernel_mul_mv_mxfp4_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_r2")]]
kernel void kernel_mul_mv_mxfp4_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_r4")]]
kernel void kernel_mul_mv_mxfp4_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_r8")]]
kernel void kernel_mul_mv_mxfp4_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_mxfp4_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_tq2_0_f32_r1")]]
kernel void kernel_mul_mv_tq2_0_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_tq2_0_f32_r2")]]
kernel void kernel_mul_mv_tq2_0_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_tq2_0_f32_r4")]]
kernel void kernel_mul_mv_tq2_0_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_tq2_0_f32_r8")]]
kernel void kernel_mul_mv_tq2_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {

    kernel_mul_mv_tq2_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q1_0_f32_r1")]]
kernel void kernel_mul_mv_q1_0_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q1_0_f32_r2")]]
kernel void kernel_mul_mv_q1_0_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q1_0_f32_r4")]]
kernel void kernel_mul_mv_q1_0_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q1_0_f32_r8")]]
kernel void kernel_mul_mv_q1_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q1_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_0_f32_r1")]]
kernel void kernel_mul_mv_q2_0_f32_r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<1, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_0_f32_r2")]]
kernel void kernel_mul_mv_q2_0_f32_r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_0_f32_r4")]]
kernel void kernel_mul_mv_q2_0_f32_r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_0_f32_r8")]]
kernel void kernel_mul_mv_q2_0_f32_r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_0_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}



// q4_K with the lanes per block as a template argument. The quant loads go two wide so one
// body serves every count, so the shipped four-wide kernel is the control, not a cell here.
template<int nr0, short LANES, typename args_t>
void kernel_mul_mv_q4_K_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {

    static_assert(64/LANES >= 4,     "over 16 lanes a quarter has fewer than four values");
    static_assert((64/LANES) % 4 == 0, "the pair loop needs the quarter to divide by four");

    constexpr short H = 64/LANES;

    const short NSG = FC_mul_mv_nsg;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short NG = FC_mul_mv_nw/LANES;
    const short ix = tiisg/LANES;
    const short it = tiisg%LANES;
    const short iq = it/(LANES/2);
    const short ir = it%(LANES/2);

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q4_K * x = (device const block_q4_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    float yl[2*H];
    float yh[2*H];

    float sumf[nr0]={0.f};

    device const float * y4 = y + ix * QK_K + 64 * iq + H * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += NG) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < H; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+H] = y4[i+ 32]; sumy[1] += yl[i+H];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+H] = y4[i+160]; sumy[3] += yh[i+H];
        }

        device const uint4 * hp = (device const uint4 *) &x[ib];
        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + (16 * (int) iq + (H/2) * (int) ir);

        const ushort shs = 16*iq;

        for (short row = 0; row < nr0; row++) {
            const uint4  hdr = *hp;
            const half2  dm  = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc16[0] = s0 & kmask1;
            sc16[1] = s2 & kmask1;
            sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};

            FOR_UNROLL (short c = 0; c < H/2; c += 2) {
                const ushort2 qv1 = *(device const ushort2 *)(q1 + c);
                const ushort2 qv2 = *(device const ushort2 *)(q1 + c + 32);

                FOR_UNROLL (short k = 0; k < 2; ++k) {
                    const short i = c + k;
                    acc1[0] += yl[2*i + 0]     * (qv1[k] & 0x000F);
                    acc1[1] += yl[2*i + 1]     * (qv1[k] & 0x0F00);
                    acc1[2] += yl[2*i + H + 0] * (qv1[k] & 0x00F0);
                    acc1[3] += yl[2*i + H + 1] * (qv1[k] & 0xF000);
                    acc2[0] += yh[2*i + 0]     * (qv2[k] & 0x000F);
                    acc2[1] += yh[2*i + 1]     * (qv2[k] & 0x0F00);
                    acc2[2] += yh[2*i + H + 0] * (qv2[k] & 0x00F0);
                    acc2[3] += yh[2*i + H + 1] * (qv2[k] & 0xF000);
                }
            }

            sumf[row] += dm[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                  (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                  (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                  (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                         dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01/2;
            hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
        }

        y4 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (int64_t)im*args.ne0*args.ne1 + (int64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}


[[host_name("kernel_mul_mv_q4_K_f32_L4r1")]]
kernel void kernel_mul_mv_q4_K_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L4r2")]]
kernel void kernel_mul_mv_q4_K_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L4r4")]]
kernel void kernel_mul_mv_q4_K_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L4r8")]]
kernel void kernel_mul_mv_q4_K_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L8r1")]]
kernel void kernel_mul_mv_q4_K_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L8r2")]]
kernel void kernel_mul_mv_q4_K_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L8r4")]]
kernel void kernel_mul_mv_q4_K_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L8r8")]]
kernel void kernel_mul_mv_q4_K_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L16r1")]]
kernel void kernel_mul_mv_q4_K_f32_L16r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<1, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L16r2")]]
kernel void kernel_mul_mv_q4_K_f32_L16r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<2, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L16r4")]]
kernel void kernel_mul_mv_q4_K_f32_L16r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<4, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q4_K_f32_L16r8")]]
kernel void kernel_mul_mv_q4_K_f32_L16r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_l_impl<8, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// q8_0 with the quants per lane as a template argument: LANES lanes cover one block,
// so NQ is what each of them reads. The shipped kernel is four lanes to a block.
template<short NR0, short LANES, typename args_t>
void kernel_mul_mv_q8_0_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const short NW = FC_mul_mv_nw;
    static_assert(QK8_0 % LANES == 0, "the lanes have to divide the block");
    constexpr short NQ = QK8_0/LANES;
    // Blocks processed per iteration scales with the SIMD width; lane partitioning
    // within a block stays tied to QK8_0 so wave64 reads the right quants.
    const short NBI = NW*NQ/QK8_0;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

  //const uint64_t offset0 = r0*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 = r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

  //device const block_q8_0 * x = (device const block_q8_0 *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    // pointers to src0 rows
    device const block_q8_0 * ax[NR0];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (r0 + row)*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;

        ax[row] = (device const block_q8_0 *) ((device char *) src0 + offset0);
    }

    float sumf[NR0] = { 0.f };

    const short ix = tiisg/(QK8_0/NQ);
    const short il = tiisg%(QK8_0/NQ);

    const int ib0 = sgitg*NBI + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    // each thread in a SIMD group deals with NQ quants at a time
    for (int ib = ib0; ib < nb; ib += NSG*NBI) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR0; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            // the eight quants this lane wants are contiguous and the block is 34 bytes, so a
            // stride that keeps them word-aligned lets two loads replace eight
            if (FC_mul_mv_row8 && (NQ % 4) == 0) {
                FOR_UNROLL (short w = 0; w < NQ/4; ++w) {
                    const uint b = mv_ld_u32(qs + 4*w);
                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        sumq += (float)(int8_t)(b >> (8*i)) * yl[4*w + i];
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NBI*QK8_0;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    helper_mv_reduce_and_write<NR0>(dst_f32, sumf, r0, args.ne01, tiisg, sgitg, shmem);
}


[[host_name("kernel_mul_mv_q8_0_f32_L2r1")]]
kernel void kernel_mul_mv_q8_0_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L2r2")]]
kernel void kernel_mul_mv_q8_0_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L2r4")]]
kernel void kernel_mul_mv_q8_0_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L2r8")]]
kernel void kernel_mul_mv_q8_0_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L4r1")]]
kernel void kernel_mul_mv_q8_0_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L4r2")]]
kernel void kernel_mul_mv_q8_0_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L4r4")]]
kernel void kernel_mul_mv_q8_0_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L4r8")]]
kernel void kernel_mul_mv_q8_0_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L8r1")]]
kernel void kernel_mul_mv_q8_0_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L8r2")]]
kernel void kernel_mul_mv_q8_0_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L8r4")]]
kernel void kernel_mul_mv_q8_0_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L8r8")]]
kernel void kernel_mul_mv_q8_0_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L16r1")]]
kernel void kernel_mul_mv_q8_0_f32_L16r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<1, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L16r2")]]
kernel void kernel_mul_mv_q8_0_f32_L16r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<2, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L16r4")]]
kernel void kernel_mul_mv_q8_0_f32_L16r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<4, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q8_0_f32_L16r8")]]
kernel void kernel_mul_mv_q8_0_f32_L16r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_l_impl<8, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// q2_K with the lanes per block as a template argument. Q is how many values a lane covers
// in each of the block's four quarters; the shipped kernel is eight lanes to a block.
template<int nr0, short LANES, typename args_t>
void kernel_mul_mv_q2_K_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    static_assert(64/LANES >= 4, "over 16 lanes a quarter has fewer than four values");
    static_assert((64/LANES) % 2 == 0, "the pair loop needs an even quarter");
    constexpr short Q = 64/LANES;

    float yl[4*Q];
    float sumf[nr0]={0.f};

    // block groups by real simd width: 4 on wave32, 8 on wave64
    const short nbi = FC_mul_mv_nw/LANES;

    const short ix = tiisg/LANES;
    const short it = tiisg%LANES;
    const short iq = it/(LANES/2);
    const short ir = it%(LANES/2);
    const short is = (Q*ir)/16;

    device const float * y4 = y + ix * QK_K + 128 * iq + Q * ir;

    for (int ib = ix; ib < nb; ib += nbi) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short i = 0; i < Q; ++i) {
            yl[i+  0] = y4[i+ 0]; sumy[0] += yl[i+  0];
            yl[i+  Q] = y4[i+32]; sumy[1] += yl[i+  Q];
            yl[i+2*Q] = y4[i+64]; sumy[2] += yl[i+2*Q];
            yl[i+3*Q] = y4[i+96]; sumy[3] += yl[i+3*Q];
        }

        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uint8_t  * sc = (device const uint8_t  *)x[ib].scales + (8 * (int) iq + (int) is);
        device const uint16_t * qs = (device const uint16_t *)x[ib].qs + (16 * (int) iq + (Q/2) * (int) ir);
        device const half     * dh = &x[ib].d;

        for (short row = 0; row < nr0; row++) {
            float4 acc1 = {0.f, 0.f, 0.f, 0.f};
            float4 acc2 = {0.f, 0.f, 0.f, 0.f};
            // the four shorts this lane reads are contiguous, so an aligned stride turns
            // them into two words
            // the two-word read only lines up when a lane covers exactly four shorts
            ushort qw[Q/2];
            if (FC_mul_mv_row8 && Q == 8) {
                FOR_UNROLL (short j = 0; j < Q/2; j += 4) {
                    const uint2 t = mv_ld_u32x2(qs + j);
                    qw[j+0] = (ushort)(t.x); qw[j+1] = (ushort)(t.x >> 16);
                    qw[j+2] = (ushort)(t.y); qw[j+3] = (ushort)(t.y >> 16);
                }
            } else {
                FOR_UNROLL (short j = 0; j < Q/2; ++j) { qw[j] = qs[j]; }
            }
            for (int i = 0; i < Q; i += 2) {
                acc1[0] += yl[i+    0] * (qw[i/2] & 0x0003);
                acc2[0] += yl[i+    1] * (qw[i/2] & 0x0300);
                acc1[1] += yl[i+  Q  ] * (qw[i/2] & 0x000c);
                acc2[1] += yl[i+  Q+1] * (qw[i/2] & 0x0c00);
                acc1[2] += yl[i+2*Q  ] * (qw[i/2] & 0x0030);
                acc2[2] += yl[i+2*Q+1] * (qw[i/2] & 0x3000);
                acc1[3] += yl[i+3*Q  ] * (qw[i/2] & 0x00c0);
                acc2[3] += yl[i+3*Q+1] * (qw[i/2] & 0xc000);
            }
            float dall = dh[0];
            float dmin = dh[1] * 1.f/16.f;
            sumf[row] += dall * ((acc1[0] + 1.f/256.f * acc2[0]) * (sc[0] & 0xF) * 1.f/ 1.f +
                                 (acc1[1] + 1.f/256.f * acc2[1]) * (sc[2] & 0xF) * 1.f/ 4.f +
                                 (acc1[2] + 1.f/256.f * acc2[2]) * (sc[4] & 0xF) * 1.f/16.f +
                                 (acc1[3] + 1.f/256.f * acc2[3]) * (sc[6] & 0xF) * 1.f/64.f) -
                         dmin * (sumy[0] * (sc[0] & 0xF0) + sumy[1] * (sc[2] & 0xF0) + sumy[2] * (sc[4] & 0xF0) + sumy[3] * (sc[6] & 0xF0));

            qs += args.nb01/2;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += nbi * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}


[[host_name("kernel_mul_mv_q2_K_f32_L4r1")]]
kernel void kernel_mul_mv_q2_K_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L4r2")]]
kernel void kernel_mul_mv_q2_K_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L4r4")]]
kernel void kernel_mul_mv_q2_K_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L4r8")]]
kernel void kernel_mul_mv_q2_K_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L8r1")]]
kernel void kernel_mul_mv_q2_K_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L8r2")]]
kernel void kernel_mul_mv_q2_K_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L8r4")]]
kernel void kernel_mul_mv_q2_K_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L8r8")]]
kernel void kernel_mul_mv_q2_K_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L16r1")]]
kernel void kernel_mul_mv_q2_K_f32_L16r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<1, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L16r2")]]
kernel void kernel_mul_mv_q2_K_f32_L16r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<2, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L16r4")]]
kernel void kernel_mul_mv_q2_K_f32_L16r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<4, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q2_K_f32_L16r8")]]
kernel void kernel_mul_mv_q2_K_f32_L16r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_l_impl<8, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// q5_K with the lanes per block as a template argument. E is how many values a lane covers
// in each quarter; the quant reads go four bytes at a time so one body serves every count.
template<int nr0, short LANES, typename args_t>
void kernel_mul_mv_q5_K_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0]={0.f};

    static_assert(64/LANES >= 4,       "over 16 lanes a quarter has fewer than four values");
    static_assert((64/LANES) % 4 == 0, "the chunked read needs the quarter to divide by four");
    constexpr short E = 64/LANES;

    float yl[2*E], yh[2*E];

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    // 8 threads per block-group; scale the group count with the SIMD width so iq stays
    // in {0,1} on wave64 (NG = 4 on wave32, 8 on wave64).
    const short NG  = FC_mul_mv_nw/LANES;
    const short tid = tiisg%LANES;
    const short ix  = tiisg/LANES;
    const short iq  = tid/(LANES/2);
    const short ir  = tid%(LANES/2);

    const short l0 = E*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    // pre-shifting qh by this puts the four bits this lane wants at fixed positions
    // 0/1/4/5, so the masks below are constants and the selects become shifts
    const ushort hsh = 2*iq;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    // same header as q4_K: d, dmin and the 12 scales are the block's first 16 bytes, and 176
    // keeps every row 16-byte aligned, so one load covers what was two halves and three shorts
    const ushort shs = 16*iq;

    for (int i = ix; i < nb; i += NG) {
        device const uint8_t * q1 = x[i].qs + q_offset;
        device const uint8_t * qh = x[i].qh + l0;
        device const uint4  * hp = (device const uint4 *) &x[i];

        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < E; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+E] = y1[l+32]; sumy[1] += yl[l+E];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+E] = y2[l+32]; sumy[3] += yh[l+E];
        }

        for (short row = 0; row < nr0; ++row) {
            device const uint8_t * q2 = q1 + 64;

            const uint4  hdr = *hp;
            const half2  dm  = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc16[0] = s0 & kmask1;
            sc16[1] = s2 & kmask1;
            sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            float4 acc1 = {0.f};

            // four bytes at a time: l0 divides by four for every count, while the shipped
            // eight-byte load only lines up when a lane covers eight
            FOR_UNROLL (short c = 0; c < E/4; ++c) {
                const uchar4 q1c = as_type<uchar4>(*(device const uint *)(q1 + 4*c));
                const uchar4 q2c = as_type<uchar4>(*(device const uint *)(q2 + 4*c));
                const uchar4 qhc = as_type<uchar4>((*(device const uint *)(qh + 4*c) >> hsh) & 0x33333333u);

                FOR_UNROLL (short k = 0; k < 4; ++k) {
                    const short l = 4*c + k;
                    const uint8_t h  = qhc[k];
                    const uint8_t v1 = q1c[k];
                    const uint8_t v2 = q2c[k];
                    acc1[0] += yl[l+0] * (float) ((v1 & 0x0F) | ((h & 1) << 4));
                    acc1[1] += yl[l+E] * (float) ((v1 >>   4) | ((h & 2) << 3));
                    acc1[2] += yh[l+0] * (float) ((v2 & 0x0F) |  (h & 16));
                    acc1[3] += yh[l+E] * (float) ((v2 >>   4) | ((h & 32) >> 1));
                }
            }

            sumf[row] += dm[0] * (sc8[0] * acc1[0] +
                                  sc8[1] * acc1[1] +
                                  sc8[4] * acc1[2] +
                                  sc8[5] * acc1[3]) -
                         dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 += args.nb01;
            qh += args.nb01;
            hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
        }

        y1 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}


[[host_name("kernel_mul_mv_q5_K_f32_L4r1")]]
kernel void kernel_mul_mv_q5_K_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L4r2")]]
kernel void kernel_mul_mv_q5_K_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L4r4")]]
kernel void kernel_mul_mv_q5_K_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L4r8")]]
kernel void kernel_mul_mv_q5_K_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L8r1")]]
kernel void kernel_mul_mv_q5_K_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L8r2")]]
kernel void kernel_mul_mv_q5_K_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L8r4")]]
kernel void kernel_mul_mv_q5_K_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L8r8")]]
kernel void kernel_mul_mv_q5_K_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L16r1")]]
kernel void kernel_mul_mv_q5_K_f32_L16r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<1, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L16r2")]]
kernel void kernel_mul_mv_q5_K_f32_L16r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<2, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L16r4")]]
kernel void kernel_mul_mv_q5_K_f32_L16r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<4, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q5_K_f32_L16r8")]]
kernel void kernel_mul_mv_q5_K_f32_L16r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_l_impl<8, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// q6_K with the lanes per block as a template argument. V is how many consecutive values a
// lane covers; the shipped kernel is sixteen lanes to a block, so V is four there.
template<int nr0, short LANES, typename args_t>
void kernel_mul_mv_q6_K_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    constexpr uint8_t kmask1 = 0x03;
    constexpr uint8_t kmask2 = 0x0C;
    constexpr uint8_t kmask3 = 0x30;
    constexpr uint8_t kmask4 = 0xC0;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q6_K * x = (device const block_q6_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    float sumf[nr0] = { 0.f };

    static_assert(64/LANES >= 4,       "over 16 lanes a lane covers fewer than four values");
    static_assert((64/LANES) % 4 == 0, "the word read needs the run to divide by four");
    constexpr short V = 64/LANES;

    float yl[4*V];

    // 16 threads per block-group; scale the group count with the SIMD width so ip stays
    // in {0,1} on wave64 (NG = 2 on wave32, 4 on wave64).
    const short NG  = FC_mul_mv_nw/LANES;
    const short tid = tiisg%LANES;
    const short ix  = tiisg/LANES;
    const short ip  = tid/(LANES/2);
    const short il  = tid%(LANES/2);
    const short l0  = V*il;
    const short is  = 8*ip + l0/16;

    const short y_offset   = 128*ip + l0;
    const short q_offset_l =  64*ip + l0;
    const short q_offset_h =  32*ip + l0;

    for (int i = ix; i < nb; i += NG) {
        device const uint8_t * q1 = x[i].ql + q_offset_l;
        device const uint8_t * q2 = q1 + 32;
        device const uint8_t * qh = x[i].qh + q_offset_h;
        device const int8_t  * sc = x[i].scales + is;
        device const half    * dh = &x[i].d;

        device const float * y = yy + i * QK_K + y_offset;

        for (short l = 0; l < V; ++l) {
            yl[4*l + 0] = y[l +  0];
            yl[4*l + 1] = y[l + 32];
            yl[4*l + 2] = y[l + 64];
            yl[4*l + 3] = y[l + 96];
        }

        for (short row = 0; row < nr0; ++row) {
            float4 sums = {0.f, 0.f, 0.f, 0.f};

            // the three arrays are read four consecutive bytes at a time. The block is 210 bytes,
            // so every other one sits two bytes off a word and a card that requires natural
            // alignment reads the wrong bytes there; two shorts land right on every block
            FOR_UNROLL (short c = 0; c < V/4; ++c) {
                uint w1 = 0, w2 = 0, wh = 0;
                if (FC_mul_mv_row8) {
                    w1 = mv_ld_u32(q1 + 4*c);
                    w2 = mv_ld_u32(q2 + 4*c);
                    wh = mv_ld_u32(qh + 4*c);
                } else if (FC_mul_mv_row4) {
                    device const ushort * s1 = (device const ushort *) (q1 + 4*c);
                    device const ushort * s2 = (device const ushort *) (q2 + 4*c);
                    device const ushort * sh = (device const ushort *) (qh + 4*c);

                    w1 = (uint) s1[0] | ((uint) s1[1] << 16);
                    w2 = (uint) s2[0] | ((uint) s2[1] << 16);
                    wh = (uint) sh[0] | ((uint) sh[1] << 16);
                }

                FOR_UNROLL (short k = 0; k < 4; ++k) {
                    const short l = 4*c + k;
                    const uint8_t b1 = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(w1 >> (8*k)) : q1[l];
                    const uint8_t b2 = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(w2 >> (8*k)) : q2[l];
                    const uint8_t bh = (FC_mul_mv_row4 || FC_mul_mv_row8) ? (uint8_t)(wh >> (8*k)) : qh[l];

                    sums[0] += yl[4*l + 0] * ((int8_t)((b1 & 0xF) | ((bh & kmask1) << 4)) - 32);
                    sums[1] += yl[4*l + 1] * ((int8_t)((b2 & 0xF) | ((bh & kmask2) << 2)) - 32);
                    sums[2] += yl[4*l + 2] * ((int8_t)((b1  >> 4) | ((bh & kmask3) << 0)) - 32);
                    sums[3] += yl[4*l + 3] * ((int8_t)((b2  >> 4) | ((bh & kmask4) >> 2)) - 32);
                }
            }

            sumf[row] += dh[0] * (sums[0] * sc[0] + sums[1] * sc[2] + sums[2] * sc[4] + sums[3] * sc[6]);

            q1 += args.nb01;
            q2 += args.nb01;
            qh += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}


[[host_name("kernel_mul_mv_q6_K_f32_L4r1")]]
kernel void kernel_mul_mv_q6_K_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L4r2")]]
kernel void kernel_mul_mv_q6_K_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L4r4")]]
kernel void kernel_mul_mv_q6_K_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L4r8")]]
kernel void kernel_mul_mv_q6_K_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L8r1")]]
kernel void kernel_mul_mv_q6_K_f32_L8r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<1, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L8r2")]]
kernel void kernel_mul_mv_q6_K_f32_L8r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<2, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L8r4")]]
kernel void kernel_mul_mv_q6_K_f32_L8r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<4, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L8r8")]]
kernel void kernel_mul_mv_q6_K_f32_L8r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<8, 8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L16r1")]]
kernel void kernel_mul_mv_q6_K_f32_L16r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<1, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L16r2")]]
kernel void kernel_mul_mv_q6_K_f32_L16r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<2, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L16r4")]]
kernel void kernel_mul_mv_q6_K_f32_L16r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<4, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_q6_K_f32_L16r8")]]
kernel void kernel_mul_mv_q6_K_f32_L16r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_l_impl<8, 16, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// iq4_nl with the lanes per block as a template argument. The body works in pairs of eight
// values, so P is how many pairs a lane takes; the shipped kernel is two lanes, two pairs.
template<int NR0, short LANES, typename args_t>
void kernel_mul_mv_iq4_nl_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_iq4_nl * x = (device const block_iq4_nl *) (src0 + offset0);
    device const float        * y = (device const float        *) (src1 + offset1);

    const int nb   = args.ne00/QK4_NL;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/LANES;
    const short ix = tiisg/LANES;
    const short it = tiisg%LANES;

    shmem_f32[tiisg] = kvalues_iq4nl_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    static_assert(32/LANES >= 8,       "a lane has to cover at least one pair");
    static_assert((32/LANES) % 8 == 0, "the pair loop needs the run to divide by eight");
    constexpr short P = (32/LANES)/8;

    float4 yl[2*P];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK4_NL + it*4*P;

    uint32_t aux32[2];
    thread const uint8_t * q8 = (thread const uint8_t *)aux32;

    float4 qf1, qf2;

    // [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *)yb;
        FOR_UNROLL (short j = 0; j < P; ++j) {
            yl[2*j + 0] = y4[j    ];
            yl[2*j + 1] = y4[j + 4];
        }

        for (short row = 0; row < NR0; row++) {
            device const block_iq4_nl & xb = x[row*ns01 + ib];
            device const uint16_t * q4 = (device const uint16_t *)(xb.qs + 4*it*P);

            float4 acc1 = {0.f}, acc2 = {0.f};

            FOR_UNROLL (short j = 0; j < P; ++j) {
                aux32[0] = q4[2*j] | (q4[2*j + 1] << 16);
                aux32[1] = (aux32[0] >> 4) & 0x0f0f0f0f;
                aux32[0] &= 0x0f0f0f0f;
                qf1 = {shmem_f32[q8[0]], shmem_f32[q8[1]], shmem_f32[q8[2]], shmem_f32[q8[3]]};
                qf2 = {shmem_f32[q8[4]], shmem_f32[q8[5]], shmem_f32[q8[6]], shmem_f32[q8[7]]};
                acc1 += yl[2*j + 0] * qf1;
                acc2 += yl[2*j + 1] * qf2;
            }

            acc1 += acc2;

            sumf[row] += (float)xb.d * (acc1[0] + acc1[1] + acc1[2] + acc1[3]);
        }

        yb += NG * QK4_NL;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}


[[host_name("kernel_mul_mv_iq4_nl_f32_L2r1")]]
kernel void kernel_mul_mv_iq4_nl_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L2r2")]]
kernel void kernel_mul_mv_iq4_nl_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L2r4")]]
kernel void kernel_mul_mv_iq4_nl_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L2r8")]]
kernel void kernel_mul_mv_iq4_nl_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L4r1")]]
kernel void kernel_mul_mv_iq4_nl_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L4r2")]]
kernel void kernel_mul_mv_iq4_nl_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L4r4")]]
kernel void kernel_mul_mv_iq4_nl_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_iq4_nl_f32_L4r8")]]
kernel void kernel_mul_mv_iq4_nl_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_iq4_nl_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}


// mxfp4 with the lanes per block as a template argument. The body works in pairs of eight
// values, so P is how many pairs a lane takes; the shipped kernel is two lanes, two pairs.
template<int NR0, short LANES, typename args_t>
void kernel_mul_mv_mxfp4_f32_l_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * NR0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_mxfp4 * x = (device const block_mxfp4 *) (src0 + offset0);
    device const float       * y = (device const float       *) (src1 + offset1);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00; // this can be larger than nb for permuted src0 tensors

    // block groups by real simd width: 16 on wave32, 32 on wave64
    const short NG = FC_mul_mv_nw/LANES;

    const short ix = tiisg/LANES;
    const short it = tiisg%LANES;

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    static_assert(32/LANES >= 8,       "a lane has to cover at least one pair");
    static_assert((32/LANES) % 8 == 0, "the pair loop needs the run to divide by eight");
    constexpr short P = (32/LANES)/8;

    float4 yl[2*P];
    float sumf[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*4*P;

    // note: just the check `ib < nb` is enough, but adding the redundant `&& ib < ns01` check makes the kernel a bit faster
    //       no idea why that is - needs some deeper investigation [TAG_MUL_MV_WEIRD]
    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *) yb;

        FOR_UNROLL (short j = 0; j < P; ++j) {
            yl[2*j + 0] = y4[j    ];
            yl[2*j + 1] = y4[j + 4];
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            device const block_mxfp4 & xb = x[row*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 4*it*P);

            float4 acc1 = {0.f}, acc2 = {0.f};
            FOR_UNROLL (short j = 0; j < P; ++j) {
                device const uint8_t * qj = q2 + 4*j;
                acc1 += yl[2*j + 0]*float4(shmem_f32[qj[0] &  0x0F], shmem_f32[qj[1] &  0x0F], shmem_f32[qj[2] &  0x0F], shmem_f32[qj[3] &  0x0F]);
                acc2 += yl[2*j + 1]*float4(shmem_f32[qj[0] >> 4   ], shmem_f32[qj[1] >> 4   ], shmem_f32[qj[2] >> 4   ], shmem_f32[qj[3] >> 4   ]);
            }

            acc1 = acc1 + acc2;

            sumf[row] += e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
        }

        yb += NG * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}


[[host_name("kernel_mul_mv_mxfp4_f32_L2r1")]]
kernel void kernel_mul_mv_mxfp4_f32_L2r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<1, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L2r2")]]
kernel void kernel_mul_mv_mxfp4_f32_L2r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<2, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L2r4")]]
kernel void kernel_mul_mv_mxfp4_f32_L2r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<4, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L2r8")]]
kernel void kernel_mul_mv_mxfp4_f32_L2r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<8, 2, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L4r1")]]
kernel void kernel_mul_mv_mxfp4_f32_L4r1(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<1, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L4r2")]]
kernel void kernel_mul_mv_mxfp4_f32_L4r2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<2, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L4r4")]]
kernel void kernel_mul_mv_mxfp4_f32_L4r4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<4, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_f32_L4r8")]]
kernel void kernel_mul_mv_mxfp4_f32_L4r8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_mxfp4_f32_l_impl<8, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

template [[host_name("kernel_mul_mv_id_q4_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_0, N_R0_Q4_0>>>;
template [[host_name("kernel_mul_mv_id_q4_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q4_1, N_R0_Q4_1>>>;
template [[host_name("kernel_mul_mv_id_q5_0_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_0, N_R0_Q5_0>>>;
template [[host_name("kernel_mul_mv_id_q5_1_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_impl<block_q5_1, N_R0_Q5_1>>>;
template [[host_name("kernel_mul_mv_id_mxfp4_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_mxfp4_f32_impl<N_R0_MXFP4>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_impl   <N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_impl   <N_R0_Q3_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K>>>;
template [[host_name("kernel_mul_mv_id_q4_K_f32_s2")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q4_K_f32_impl   <N_R0_Q4_K, ggml_metal_kargs_mul_mv, 2>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32_r8")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_impl   <8>>>;
template [[host_name("kernel_mul_mv_id_q6_K_f32")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q6_K_f32_impl   <N_R0_Q6_K>>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_impl  <N_R0_IQ1_S>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_impl  <N_R0_IQ1_M>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_impl <N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_impl  <N_R0_IQ3_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_impl  <N_R0_IQ2_S>>>;
template [[host_name("kernel_mul_mv_id_iq4_nl_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_nl_f32_impl <N_R0_IQ4_NL>>>;
template [[host_name("kernel_mul_mv_id_iq4_xs_f32")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq4_xs_f32_impl <N_R0_IQ4_XS>>>;
template [[host_name("kernel_mul_mv_id_tq2_0_f32")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_tq2_0_f32_impl  <N_R0_TQ2_0>>>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_5_nr0_2")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 2, block_q3_K>;
template [[host_name("kernel_mul_mv_ext_q3_K_hdr_f32_r1_5_nr0_4")]] kernel mul_mv_ext_q4x4_f32_t kernel_mul_mv_ext_qK_hdr_f32_disp<5, 4, block_q3_K>;

// gpt-oss expert gate and up mat-vecs, their per-expert biases and the clamped swiglu in one
// dispatch; each simdgroup takes half the standalone rows from each matrix, with the same sums
[[host_name("kernel_mul_mv_id_mxfp4_swiglu_oai_f32")]]
kernel void kernel_mul_mv_id_mxfp4_swiglu_oai_f32(
        constant ggml_metal_kargs_mul_mv_id_swiglu_oai & argsg,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        device const char * ids,
        device const char * biasg,
        device const char * biasu,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]]) {
    constant ggml_metal_kargs_mul_mv_id & args = argsg.mv;

    const short NR0 = N_R0_MXFP4/2;

    const int iid1 = tgpig.z/args.nei0;
    const int idx  = tgpig.z%args.nei0;

    const int32_t i02 = ((device const int32_t *) (ids + iid1*args.nbi1))[idx];

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int first_row = tgpig.x*NR0;

    const uint64_t offset0 = i02*args.nb02 + first_row*args.nb01;

    device const block_mxfp4 * xg = (device const block_mxfp4 *) (src0g + offset0);
    device const block_mxfp4 * xu = (device const block_mxfp4 *) (src0u + offset0);
    device const float       * y  = (device const float       *) (src1 + (idx % args.ne11)*args.nb11 + iid1*args.nb12);

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/2;

    const short ix = tiisg/2;
    const short it = tiisg%2;

    float4 yl[4];
    float sumg[NR0]={0.f};
    float sumu[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < 2*NR0; row++) {
            device const block_mxfp4 & xb = row < NR0 ? xg[row*ns01 + ib] : xu[(row - NR0)*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            const float s = e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            if (row < NR0) {
                sumg[row] += s;
            } else {
                sumu[row - NR0] += s;
            }
        }

        yb += NG * QK_MXFP4;
    }

    device const float * bg = (device const float *) (biasg + i02*argsg.nbg1);
    device const float * bu = (device const float *) (biasu + i02*argsg.nbu1);

    device float * dst_f32 = (device float *) dst + (uint64_t)iid1*args.ne1*args.ne0 + (uint64_t)idx*args.ne0;

    for (int row = 0; row < NR0 && first_row + row < args.ne0; ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);
        if (tiisg == 0) {
            float x0 = g + bg[first_row + row];
            float x1 = u + bu[first_row + row];

            x0 = min(x0, argsg.limit);
            x1 = max(min(x1, argsg.limit), -argsg.limit);

            float out_glu = x0 / (1.0f + exp(-x0 * argsg.alpha));
            out_glu = out_glu * (1.0f + x1);

            dst_f32[first_row + row] = out_glu;
        }
    }
}

// gate and up mat-vecs of one token and their swiglu in one dispatch; each simdgroup takes half the
// standalone rows from each matrix, so as many groups stay in flight as the two dispatches had
template<short nr0>
void kernel_mul_mv_q4_K_swiglu_f32_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NG  = FC_mul_mv_nw/8;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * nr0;

    device const block_q4_K * xs[2] = {
        (device const block_q4_K *) (src0g + (uint64_t) first_row*args.nb01),
        (device const block_q4_K *) (src0u + (uint64_t) first_row*args.nb01),
    };
    device const float * y = (device const float *) src1;

    float yl[16];
    float yh[16];

    float sums[2][nr0] = {{0.f}};

    device const float * y4 = y + ix * QK_K + 64 * iq + 8 * ir;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    for (int ib = ix; ib < nb; ib += NG) {
        float4 sumy = {0.f, 0.f, 0.f, 0.f};

        for (short i = 0; i < 8; ++i) {
            yl[i+0] = y4[i+  0]; sumy[0] += yl[i+0];
            yl[i+8] = y4[i+ 32]; sumy[1] += yl[i+8];
            yh[i+0] = y4[i+128]; sumy[2] += yh[i+0];
            yh[i+8] = y4[i+160]; sumy[3] += yh[i+8];
        }

        const ushort shs = 16*iq;

        for (short m = 0; m < 2; ++m) {
            device const block_q4_K * x = xs[m];
            device const uint4 * hp = (device const uint4 *) &x[ib];
            // widen to int before the pointer math: short offsets into a device base miscompile on AMD
            device const uint16_t * q1 = (device const uint16_t *)x[ib].qs + (16 * (int) iq + 4 * (int) ir);

            for (short row = 0; row < nr0; row++) {
                const uint4  hdr = *hp;
                const half2  dm  = as_type<half2>(hdr.x);

                const uint16_t s0 = (uint16_t) (hdr.y >> shs);
                const uint16_t s2 = (uint16_t) (hdr.z >> shs);
                const uint16_t s4 = (uint16_t) (hdr.w >> shs);

                sc16[0] = s0 & kmask1;
                sc16[1] = s2 & kmask1;
                sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
                sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

                const ushort4 qv1 = *(device const ushort4 *)(q1);
                const ushort4 qv2 = *(device const ushort4 *)(q1 + 32);

                float4 acc1 = {0.f, 0.f, 0.f, 0.f};
                float4 acc2 = {0.f, 0.f, 0.f, 0.f};

                FOR_UNROLL (short i = 0; i < 4; ++i) {
                    acc1[0] += yl[2*i + 0] * (qv1[i] & 0x000F);
                    acc1[1] += yl[2*i + 1] * (qv1[i] & 0x0F00);
                    acc1[2] += yl[2*i + 8] * (qv1[i] & 0x00F0);
                    acc1[3] += yl[2*i + 9] * (qv1[i] & 0xF000);
                    acc2[0] += yh[2*i + 0] * (qv2[i] & 0x000F);
                    acc2[1] += yh[2*i + 1] * (qv2[i] & 0x0F00);
                    acc2[2] += yh[2*i + 8] * (qv2[i] & 0x00F0);
                    acc2[3] += yh[2*i + 9] * (qv2[i] & 0xF000);
                }

                sums[m][row] += dm[0] * ((acc1[0] + 1.f/256.f * acc1[1]) * sc8[0] +
                                         (acc1[2] + 1.f/256.f * acc1[3]) * sc8[1] * 1.f/16.f +
                                         (acc2[0] + 1.f/256.f * acc2[1]) * sc8[4] +
                                         (acc2[2] + 1.f/256.f * acc2[3]) * sc8[5] * 1.f/16.f) -
                                dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.nb01/2;
                hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
            }
        }

        y4 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float g = simd_sum(sums[0][row]);
        const float u = simd_sum(sums[1][row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

[[host_name("kernel_mul_mv_q4_K_swiglu_f32")]]
kernel void kernel_mul_mv_q4_K_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_swiglu_f32_impl<N_R0_Q4_K/2>(args, src0g, src0u, src1, dst, tgpig, tiisg, sgitg);
}

template<short NR0>
void kernel_mul_mv_q4_0_swiglu_f32_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;
    const short NQ  = NW/2;

    const int nb = args.ne00/QK4_0;

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;

    device const float * y = (device const float *) src1;

    device const block_q4_0 * ag[NR0];
    device const block_q4_0 * au[NR0];
    FOR_UNROLL (int row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (uint64_t) (r0 + row)*args.nb01;
        ag[row] = (device const block_q4_0 *) (src0g + offset0);
        au[row] = (device const block_q4_0 *) (src0u + offset0);
    }

    float sumg[NR0] = {0.f};
    float sumu[NR0] = {0.f};

    const short ix = (tiisg/(NW/NQ));
    const short il = (tiisg%(NW/NQ))*8;

    float yl[16];

    device const float * yb = y + ix*QK4_0 + il;

    for (int ib = ix; ib < nb; ib += NQ) {
        float sumy[2] = { 0.f, 0.f };

        FOR_UNROLL (short i = 0; i < 8; i += 2) {
            sumy[0]  += yb[i +  0] + yb[i +  1];
            yl[i + 0] = yb[i +  0];
            yl[i + 1] = yb[i +  1]/256.f;

            sumy[1]  += yb[i + 16] + yb[i + 17];
            yl[i + 8] = yb[i + 16]/16.f;
            yl[i + 9] = yb[i + 17]/4096.f;
        }

        FOR_UNROLL (short row = 0; row < NR0; row++) {
            sumg[row] += block_q_n_dot_y(ag[row] + ib, sumy[0] + sumy[1], yl, il);
            sumu[row] += block_q_n_dot_y(au[row] + ib, sumy[0] + sumy[1], yl, il);
        }

        yb += QK4_0 * NQ;
    }

    device float * dst_f32 = (device float *) dst;

    for (int row = 0; row < NR0; ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

[[host_name("kernel_mul_mv_q4_0_swiglu_f32")]]
kernel void kernel_mul_mv_q4_0_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_0_swiglu_f32_impl<N_R0_Q4_0/2>(args, src0g, src0u, src1, dst, tgpig, tiisg, sgitg);
}

[[host_name("kernel_mul_mv_mxfp4_swiglu_f32")]]
kernel void kernel_mul_mv_mxfp4_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    const short NSG = FC_mul_mv_nsg;
    const short NR0 = N_R0_MXFP4/2;

    threadgroup float * shmem_f32 = (threadgroup float *) shmem;

    shmem_f32[tiisg] = kvalues_mxfp4_f[tiisg%16];
    threadgroup_barrier(mem_flags::mem_threadgroup);

    const int r0 = (tgpig.x*NSG + sgitg)*NR0;

    device const block_mxfp4 * xg = (device const block_mxfp4 *) (src0g + (uint64_t) r0*args.nb01);
    device const block_mxfp4 * xu = (device const block_mxfp4 *) (src0u + (uint64_t) r0*args.nb01);
    device const float       * y  = (device const float       *) src1;

    const int nb   = args.ne00/QK_MXFP4;
    const int ns01 = args.nb01/args.nb00;

    const short NG = FC_mul_mv_nw/2;

    const short ix = tiisg/2;
    const short it = tiisg%2;

    float4 yl[4];
    float sumg[NR0]={0.f};
    float sumu[NR0]={0.f};

    device const float * yb = y + ix*QK_MXFP4 + it*8;

    for (int ib = ix; ib < nb && ib < ns01; ib += NG) {
        device const float4 * y4 = (device const float4 *) yb;

        yl[0] = y4[0];
        yl[1] = y4[4];
        yl[2] = y4[1];
        yl[3] = y4[5];

        FOR_UNROLL (short row = 0; row < 2*NR0; row++) {
            device const block_mxfp4 & xb = row < NR0 ? xg[row*ns01 + ib] : xu[(row - NR0)*ns01 + ib];
            device const uint8_t     * q2 = (device const uint8_t *)(xb.qs + 8*it);

            float4 acc1 = yl[0]*float4(shmem_f32[q2[0] &  0x0F], shmem_f32[q2[1] &  0x0F], shmem_f32[q2[2] &  0x0F], shmem_f32[q2[3] &  0x0F]);
            float4 acc2 = yl[1]*float4(shmem_f32[q2[0] >> 4   ], shmem_f32[q2[1] >> 4   ], shmem_f32[q2[2] >> 4   ], shmem_f32[q2[3] >> 4   ]);
            float4 acc3 = yl[2]*float4(shmem_f32[q2[4] &  0x0F], shmem_f32[q2[5] &  0x0F], shmem_f32[q2[6] &  0x0F], shmem_f32[q2[7] &  0x0F]);
            float4 acc4 = yl[3]*float4(shmem_f32[q2[4] >> 4   ], shmem_f32[q2[5] >> 4   ], shmem_f32[q2[6] >> 4   ], shmem_f32[q2[7] >> 4   ]);

            acc1 = (acc1 + acc3) + (acc2 + acc4);

            const float s = e8m0_to_fp32(xb.e) * ((acc1[0] + acc1[1]) + (acc1[2] + acc1[3]));
            if (row < NR0) {
                sumg[row] += s;
            } else {
                sumu[row - NR0] += s;
            }
        }

        yb += NG * QK_MXFP4;
    }

    device float * dst_f32 = (device float *) dst;

    for (int row = 0; row < NR0; ++row) {
        const float g = simd_sum(sumg[row]);
        const float u = simd_sum(sumu[row]);

        if (tiisg == 0 && r0 + row < args.ne01) {
            dst_f32[r0 + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

template<short nr0>
void kernel_mul_mv_q5_K_swiglu_f32_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NG  = FC_mul_mv_nw/8;

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short tid = tiisg%8;
    const short ix  = tiisg/8;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;
    const ushort hsh = 2*iq;
    const ushort shs = 16*iq;

    const int nb = args.ne00/QK_K;

    const int first_row = (tgpig.x * NSG + sgitg) * nr0;

    device const block_q5_K * xs[2] = {
        (device const block_q5_K *) (src0g + (uint64_t) first_row*args.nb01),
        (device const block_q5_K *) (src0u + (uint64_t) first_row*args.nb01),
    };

    float sums[2][nr0] = {{0.f}};

    float yl[16], yh[16];

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = (device const float *) src1 + ix*QK_K + y_offset;

    for (int i = ix; i < nb; i += NG) {
        device const float * y2 = y1 + 128;
        float4 sumy = {0.f, 0.f, 0.f, 0.f};
        for (short l = 0; l < 8; ++l) {
            yl[l+0] = y1[l+ 0]; sumy[0] += yl[l+0];
            yl[l+8] = y1[l+32]; sumy[1] += yl[l+8];
            yh[l+0] = y2[l+ 0]; sumy[2] += yh[l+0];
            yh[l+8] = y2[l+32]; sumy[3] += yh[l+8];
        }

        for (short m = 0; m < 2; ++m) {
            device const uint8_t * q1 = xs[m][i].qs + q_offset;
            device const uint8_t * qh = xs[m][i].qh + l0;
            device const uint4  * hp = (device const uint4 *) &xs[m][i];

            for (short row = 0; row < nr0; ++row) {
                device const uint8_t * q2 = q1 + 64;

                const uint4  hdr = *hp;
                const half2  dm  = as_type<half2>(hdr.x);

                const uint16_t s0 = (uint16_t) (hdr.y >> shs);
                const uint16_t s2 = (uint16_t) (hdr.z >> shs);
                const uint16_t s4 = (uint16_t) (hdr.w >> shs);

                sc16[0] = s0 & kmask1;
                sc16[1] = s2 & kmask1;
                sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
                sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

                float4 acc1 = {0.f};
                const uint2 q1w = *(device const uint2 *)(q1);
                const uint2 q2w = *(device const uint2 *)(q2);
                const uint2 qhw = *(device const uint2 *)(qh);

                const uchar4 q1a = as_type<uchar4>(q1w.x), q1b = as_type<uchar4>(q1w.y);
                const uchar4 q2a = as_type<uchar4>(q2w.x), q2b = as_type<uchar4>(q2w.y);
                const uchar4 qha = as_type<uchar4>((qhw.x >> hsh) & 0x33333333u);
                const uchar4 qhb = as_type<uchar4>((qhw.y >> hsh) & 0x33333333u);

                FOR_UNROLL (short l = 0; l < 8; ++l) {
                    const uint8_t h  = l < 4 ? qha[l] : qhb[l - 4];
                    const uint8_t v1 = l < 4 ? q1a[l] : q1b[l - 4];
                    const uint8_t v2 = l < 4 ? q2a[l] : q2b[l - 4];
                    acc1[0] += yl[l+0] * (float) ((v1 & 0x0F) | ((h & 1) << 4));
                    acc1[1] += yl[l+8] * (float) ((v1 >>   4) | ((h & 2) << 3));
                    acc1[2] += yh[l+0] * (float) ((v2 & 0x0F) |  (h & 16));
                    acc1[3] += yh[l+8] * (float) ((v2 >>   4) | ((h & 32) >> 1));
                }

                sums[m][row] += dm[0] * (sc8[0] * acc1[0] +
                                         sc8[1] * acc1[1] +
                                         sc8[4] * acc1[2] +
                                         sc8[5] * acc1[3]) -
                                dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

                q1 += args.nb01;
                qh += args.nb01;
                hp  = (device const uint4 *) ((device const char *) hp + args.nb01);
            }
        }

        y1 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float g = simd_sum(sums[0][row]);
        const float u = simd_sum(sums[1][row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_swiglu_f32")]]
kernel void kernel_mul_mv_q5_K_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_swiglu_f32_impl<N_R0_Q5_K/2>(args, src0g, src0u, src1, dst, tgpig, tiisg, sgitg);
}

// q8_0 splits every row across the simdgroups, so gate and up go through the same threadgroup
// reduction as the standalone rows, gate rows first
template<short NR0>
void kernel_mul_mv_q8_0_swiglu_f32_impl(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;
    const short NW  = FC_mul_mv_nw;
    constexpr short NQ = 8;
    const short NBI = NW*NQ/QK8_0;
    constexpr short NR = 2*NR0;

    const int nb = args.ne00/QK8_0;

    const int r0 = tgpig.x*NR0;

    device const float * y = (device const float *) src1;

    device const block_q8_0 * ax[NR];
    FOR_UNROLL (short row = 0; row < NR0; ++row) {
        const uint64_t offset0 = (uint64_t) (r0 + row)*args.nb01;
        ax[row]       = (device const block_q8_0 *) (src0g + offset0);
        ax[NR0 + row] = (device const block_q8_0 *) (src0u + offset0);
    }

    float sumf[NR] = { 0.f };

    const short ix = tiisg/(QK8_0/NQ);
    const short il = tiisg%(QK8_0/NQ);

    const int ib0 = sgitg*NBI + ix;

    float yl[NQ];

    device const float * yb = y + ib0*QK8_0 + il*NQ;

    for (int ib = ib0; ib < nb; ib += NSG*NBI) {
        for (short i = 0; i < NQ; ++i) {
            yl[i] = yb[i];
        }

        for (short row = 0; row < NR; row++) {
            device const int8_t * qs = ax[row][ib].qs + il*NQ;

            float sumq = 0.f;
            if (FC_mul_mv_row8 && (NQ % 4) == 0) {
                FOR_UNROLL (short w = 0; w < NQ/4; ++w) {
                    const uint b = mv_ld_u32(qs + 4*w);
                    FOR_UNROLL (short i = 0; i < 4; ++i) {
                        sumq += (float)(int8_t)(b >> (8*i)) * yl[4*w + i];
                    }
                }
            } else {
                FOR_UNROLL (short i = 0; i < NQ; ++i) {
                    sumq += qs[i] * yl[i];
                }
            }

            sumf[row] += sumq*ax[row][ib].d;
        }

        yb += NSG*NBI*QK8_0;
    }

    threadgroup float * shmem_f32[NR];

    for (short row = 0; row < NR; ++row) {
        shmem_f32[row] = (threadgroup float *) shmem + NW*row;

        if (sgitg == 0) {
            shmem_f32[row][tiisg] = 0.0f;
        }

        sumf[row] = simd_sum(sumf[row]);
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (short row = 0; row < NR; ++row) {
        if (tiisg == 0) {
            shmem_f32[row][sgitg] = sumf[row];
        }
    }

    threadgroup_barrier(mem_flags::mem_threadgroup);

    device float * dst_f32 = (device float *) dst;

    for (short row = 0; row < NR0 && r0 + row < args.ne01; ++row) {
        const float g = simd_sum(shmem_f32[row][tiisg]);
        const float u = simd_sum(shmem_f32[NR0 + row][tiisg]);

        if (tiisg == 0 && sgitg == 0) {
            dst_f32[r0 + row] = g / (1.0f + exp(-g)) * u;
        }
    }
}

[[host_name("kernel_mul_mv_q8_0_swiglu_f32")]]
kernel void kernel_mul_mv_q8_0_swiglu_f32(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0g,
        device const char * src0u,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_swiglu_f32_impl<N_R0_Q8_0/2>(args, src0g, src0u, src1, dst, shmem, tgpig, tiisg, sgitg);
}

// low-bit decode with the standalone geometry and fewer operations per weight: grid entries are
// read as one word, and as grid values are positive a sign is or-ed into the float

inline float4 mv_sign4(float4 v, uint s) {
    const uint4 m = uint4(s << 31, s << 30, s << 29, s << 28) & 0x80000000u;
    return as_type<float4>(as_type<uint4>(v) | m);
}

inline float mv_fma4(float acc, float4 a, float4 b) {
    acc = fma(a.x, b.x, acc);
    acc = fma(a.y, b.y, acc);
    acc = fma(a.z, b.z, acc);
    return fma(a.w, b.w, acc);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_xxs_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 256; i += gsz) svalues[i] = iq3xxs_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq3_xxs);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, then 64 grid indices, then 4 bytes of scale and sign indices per 32 weights
                device const ushort * q3 = (device const ushort *) (xr + 2 + 8*ib);
                device const ushort * gs = (device const ushort *) (xr + 2 + QK_K/4 + 4*ib);

                const uint  aux32 = (uint) gs[0] | ((uint) gs[1] << 16);
                const float d     = (float) *(device const half *) xr * (0.5f + (aux32 >> 28));

                float sum = 0.f;
                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const ushort qq = q3[l];
                    const float4 v1 = float4(as_type<uchar4>(svalues[qq & 0xFF]));
                    const float4 v2 = float4(as_type<uchar4>(svalues[qq >> 8]));
                    // the eighth sign is the parity of the seven stored ones
                    const uint s7 = (aux32 >> 7*l) & 127;
                    const uint s  = s7 | ((popcount(s7) & 1) << 7);
                    sum = mv_fma4(sum, mv_sign4(v1, s),      yl[2*l + 0]);
                    sum = mv_fma4(sum, mv_sign4(v2, s >> 4), yl[2*l + 1]);
                }
                sumf[row] += d * sum;
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.5f;
            }
        }
    }
}

// a threadgroup stages its grid once and then walks row blocks ntg apart, so the host can launch
// fewer of them than there are blocks
#define MV_V2_GRID_KERNEL(T, NR0, NAME) \
[[host_name(NAME)]] \
kernel void kernel_mul_mv_##T##_f32_v2_r##NR0( \
        constant ggml_metal_kargs_mul_mv & args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem [[threadgroup(0)]], \
        uint3  tgpig[[threadgroup_position_in_grid]], \
        uint3  tgpg[[threadgroups_per_grid]], \
        ushort tiisg[[thread_index_in_simdgroup]], \
        ushort sgitg[[simdgroup_index_in_threadgroup]]) { \
    kernel_mul_mv_##T##_f32_v2_impl<NR0, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tgpg.x, tiisg, sgitg); \
}

MV_V2_GRID_KERNEL(iq3_xxs, N_R0_IQ3_XXS, "kernel_mul_mv_iq3_xxs_f32_v2")
MV_V2_GRID_KERNEL(iq3_xxs, 2, "kernel_mul_mv_iq3_xxs_f32_v2_r2")
MV_V2_GRID_KERNEL(iq3_xxs, 8, "kernel_mul_mv_iq3_xxs_f32_v2_r8")

template<int nr0, typename args_t>
void kernel_mul_mv_iq3_s_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 512; i += gsz) svalues[i] = iq3s_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq3_s);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, qs[64], qh[8], signs[32], scales[4]
                device const ushort * qs = (device const ushort *) (xr + 2 + 8*ib);
                device const ushort * sg = (device const ushort *) (xr + 2 + QK_K/4 + QK_K/32 + 4*ib);

                const uint  qh  = (uint) *(device const uchar *) (xr + 2 + QK_K/4 + ib);
                const uint  sgw = (uint) sg[0] | ((uint) sg[1] << 16);
                const uint  sc  = (uint) *(device const uchar *) (xr + 2 + QK_K/4 + QK_K/32 + QK_K/8 + ib/2);
                const float d   = (float) *(device const half *) xr * (1 + 2*((sc >> 4*(ib%2)) & 0xf));

                float sum = 0.f;
                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint qq = qs[l];
                    const uint i1 = (qq & 0xFF) | ((qh << (8 - 2*l)) & 0x100);
                    const uint i2 = (qq >> 8)   | ((qh << (7 - 2*l)) & 0x100);
                    const uint s  = sgw >> 8*l;
                    sum = mv_fma4(sum, mv_sign4(float4(as_type<uchar4>(svalues[i1])), s),      yl[2*l + 0]);
                    sum = mv_fma4(sum, mv_sign4(float4(as_type<uchar4>(svalues[i2])), s >> 4), yl[2*l + 1]);
                }
                sumf[row] += d * sum;
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq3_s, N_R0_IQ3_S, "kernel_mul_mv_iq3_s_f32_v2")
MV_V2_GRID_KERNEL(iq3_s, 2, "kernel_mul_mv_iq3_s_f32_v2_r2")
MV_V2_GRID_KERNEL(iq3_s, 8, "kernel_mul_mv_iq3_s_f32_v2_r8")

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_s_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    // each grid entry is eight bytes, kept as two words
    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        threadgroup uint64_t * s64 = (threadgroup uint64_t *) shmem;
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 1024; i += gsz) s64[i] = iq2s_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq2_s);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, 32 grid indices, 32 sign bytes, qh[8], scales[8]
                device const ushort * qs = (device const ushort *) (xr + 2 + 4*ib);
                device const ushort * sg = (device const ushort *) (xr + 2 + QK_K/8 + 4*ib);

                const uint qi  = (uint) qs[0] | ((uint) qs[1] << 16);
                const uint sgw = (uint) sg[0] | ((uint) sg[1] << 16);
                const uint qh  = (uint) *(device const uchar *) (xr + 2 + QK_K/4 + ib);
                const uint sc  = (uint) *(device const uchar *) (xr + 2 + QK_K/4 + QK_K/32 + ib);

                const float db = (float) *(device const half *) xr;

                float sum[2] = {0.f, 0.f};
                FOR_UNROLL (short k = 0; k < 4; ++k) {
                    const uint idx = ((qi >> 8*k) & 0xFF) | (((qh >> 2*k) & 3) << 8);
                    const uint s   = sgw >> 8*k;
                    sum[k/2] = mv_fma4(sum[k/2], mv_sign4(float4(as_type<uchar4>(svalues[2*idx + 0])), s),      yl[2*k + 0]);
                    sum[k/2] = mv_fma4(sum[k/2], mv_sign4(float4(as_type<uchar4>(svalues[2*idx + 1])), s >> 4), yl[2*k + 1]);
                }
                sumf[row] += db * ((0.5f + (sc & 0xf)) * sum[0] + (0.5f + (sc >> 4)) * sum[1]);
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq2_s, N_R0_IQ2_S, "kernel_mul_mv_iq2_s_f32_v2")
MV_V2_GRID_KERNEL(iq2_s, 2, "kernel_mul_mv_iq2_s_f32_v2_r2")
MV_V2_GRID_KERNEL(iq2_s, 8, "kernel_mul_mv_iq2_s_f32_v2_r8")

// four 3-bit weights from one word of low fields and one of high bits, as 0..7
inline float4 mv_q3w(uint q, uint h, short sq, short sh) {
    return float4(as_type<uchar4>(((q >> sq) & 0x03030303u) | (((h >> sh) & 0x01010101u) << 2)));
}

template<int nr0, typename args_t>
void kernel_mul_mv_q3_K_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q3_K * x = (device const block_q3_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    // same lane split as the standalone kernel: a lane takes the fields j = il and il + 1 of
    // eight weights in each half of 16
    const short ng  = FC_mul_mv_nw/8;
    const short bg  = tiisg/8;
    const short tid = tiisg%8;
    const short ip  = tid/4;
    const short il  = 2*((tid%4)/2);
    const short ir  = tid%2;
    const short l0  = 8*ir;

    const short sq = 2*il;
    const short sh = 4*ip + il;

    const uint16_t s_shift1 = 4*ip;
    const uint16_t s_shift2 = s_shift1 + il;

    device const float * y1 = yy + bg*QK_K + (int) (128*ip + 32*il + l0);

    uint32_t scales32, aux32;
    thread uint16_t * scales16 = (thread uint16_t *)&scales32;
    thread const int8_t * scales = (thread const int8_t *)&scales32;

    float sumf[nr0] = {0.f};

    for (int i = bg; i < nb; i += ng) {
        // runs of eight in the order (j, half) = (il, 0), (il, 1), (il + 1, 0), (il + 1, 1),
        // which is also the order of this lane's four scales
        float4 yl[8];
        float  sy[4];
        FOR_UNROLL (short g = 0; g < 4; ++g) {
            device const float4 * yg = (device const float4 *) (y1 + 16*g);
            yl[2*g + 0] = yg[0];
            yl[2*g + 1] = yg[1];
            const float4 t = yl[2*g + 0] + yl[2*g + 1];
            sy[g] = t.x + t.y + t.z + t.w;
        }

        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uchar    * q  = x[i].qs + (int) (32*ip + l0);
        device const uchar    * h  = x[i].hmask + (int) l0;
        device const uint16_t * a  = (device const uint16_t *) (x[i].scales);
        device const half     * dh = &x[i].d;

        for (short row = 0; row < nr0; ++row) {
            const float d_all = (float) dh[0];

            scales16[0] = a[4];
            scales16[1] = a[5];
            aux32 = ((scales32 >> s_shift2) << 4) & 0x30303030;
            scales16[0] = a[il+0];
            scales16[1] = a[il+1];
            scales32 = ((scales32 >> s_shift1) & 0x0f0f0f0f) | aux32;

            const uint q0 = mv_ld_u32(q +  0), q1 = mv_ld_u32(q +  4);
            const uint q2 = mv_ld_u32(q + 16), q3 = mv_ld_u32(q + 20);
            const uint h0 = mv_ld_u32(h +  0), h1 = mv_ld_u32(h +  4);
            const uint h2 = mv_ld_u32(h + 16), h3 = mv_ld_u32(h + 20);

            float acc[4] = {0.f, 0.f, 0.f, 0.f};
            acc[0] = mv_fma4(acc[0], mv_q3w(q0, h0, sq,     sh    ), yl[0]);
            acc[0] = mv_fma4(acc[0], mv_q3w(q1, h1, sq,     sh    ), yl[1]);
            acc[1] = mv_fma4(acc[1], mv_q3w(q2, h2, sq,     sh    ), yl[2]);
            acc[1] = mv_fma4(acc[1], mv_q3w(q3, h3, sq,     sh    ), yl[3]);
            acc[2] = mv_fma4(acc[2], mv_q3w(q0, h0, sq + 2, sh + 1), yl[4]);
            acc[2] = mv_fma4(acc[2], mv_q3w(q1, h1, sq + 2, sh + 1), yl[5]);
            acc[3] = mv_fma4(acc[3], mv_q3w(q2, h2, sq + 2, sh + 1), yl[6]);
            acc[3] = mv_fma4(acc[3], mv_q3w(q3, h3, sq + 2, sh + 1), yl[7]);

            // a weight is its 0..7 value minus 4
            float s = 0.f;
            FOR_UNROLL (short g = 0; g < 4; ++g) {
                s = fma(acc[g] - 4.f*sy[g], (float) (scales[g] - 32), s);
            }
            sumf[row] += d_all * s;

            q  += args.nb01;
            h  += args.nb01;
            a  += args.nb01/2;
            dh += args.nb01/2;
        }

        y1 += ng * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q3_K_f32_v2")]]
kernel void kernel_mul_mv_q3_K_f32_v2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q3_K_f32_v2_impl<N_R0_Q3_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q2_K_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q2_K * x = (device const block_q2_K *) (src0 + offset0);
    device const float      * y = (device const float      *) (src1 + offset1);

    const short nbi = FC_mul_mv_nw/8;

    const short ix = tiisg/8;
    const short it = tiisg%8;
    const short iq = it/4;
    const short ir = it%4;
    const short is = (8*ir)/16;

    device const float * y4 = y + ix * QK_K + 128 * iq + 8 * ir;

    float sumf[nr0] = {0.f};

    for (int ib = ix; ib < nb; ib += nbi) {
        // run k holds the weights of the field shifted by 2k, which is also scale k
        float4 yl[8];
        float  sy[4];
        FOR_UNROLL (short k = 0; k < 4; ++k) {
            device const float4 * yk = (device const float4 *) (y4 + 32*k);
            yl[2*k + 0] = yk[0];
            yl[2*k + 1] = yk[1];
            const float4 t = yl[2*k + 0] + yl[2*k + 1];
            sy[k] = t.x + t.y + t.z + t.w;
        }

        // widen to int before the pointer math: short offsets into a device base miscompile on AMD
        device const uchar * q  = x[ib].qs + (int) (32*iq + 8*ir);
        device const uchar * sc = x[ib].scales + (int) (8*iq + is);
        device const half  * dh = &x[ib].d;

        for (short row = 0; row < nr0; ++row) {
            const uint w0 = mv_ld_u32(q + 0);
            const uint w1 = mv_ld_u32(q + 4);

            float acc[4];
            FOR_UNROLL (short k = 0; k < 4; ++k) {
                acc[k] = mv_fma4(0.f,    float4(as_type<uchar4>((w0 >> 2*k) & 0x03030303u)), yl[2*k + 0]);
                acc[k] = mv_fma4(acc[k], float4(as_type<uchar4>((w1 >> 2*k) & 0x03030303u)), yl[2*k + 1]);
            }

            const float dall = (float) dh[0];
            const float dmin = (float) dh[1];

            float sd = 0.f, sm = 0.f;
            FOR_UNROLL (short k = 0; k < 4; ++k) {
                const uint s = sc[2*k];
                sd = fma(acc[k], (float) (s & 0xF), sd);
                sm = fma(sy[k],  (float) (s >> 4),  sm);
            }
            sumf[row] += dall * sd - dmin * sm;

            q  += args.nb01;
            sc += args.nb01;
            dh += args.nb01/2;
        }

        y4 += nbi * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float sum_all = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = sum_all;
        }
    }
}

[[host_name("kernel_mul_mv_q2_K_f32_v2")]]
kernel void kernel_mul_mv_q2_K_f32_v2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q2_K_f32_v2_impl<N_R0_Q2_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_q5_K_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    const int r0 = tgpig.x;
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const int first_row = (r0 * NSG + sgitg) * nr0;

    const uint i12 = im%FC_mul_mv_ne12;
    const uint i13 = im/FC_mul_mv_ne12;

    const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
    const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

    device const block_q5_K * x = (device const block_q5_K *) (src0 + offset0);
    device const float     * yy = (device const float      *) (src1 + offset1);

    constexpr uint16_t kmask1 = 0x3f3f;
    constexpr uint16_t kmask2 = 0x0f0f;
    constexpr uint16_t kmask3 = 0xc0c0;

    const short NG  = FC_mul_mv_nw/8;
    const short tid = tiisg%8;
    const short ix  = tiisg/8;
    const short iq  = tid/4;
    const short ir  = tid%4;

    const short l0 = 8*ir;
    const short q_offset = 32*iq + l0;
    const short y_offset = 64*iq + l0;

    // pre-shifted by 2*iq, the fifth bits of this lane's four runs sit at bits 0, 1, 4 and 5
    const ushort hsh = 2*iq;
    const ushort shs = 16*iq;

    uint16_t sc16[4];
    thread const uint8_t * sc8 = (thread const uint8_t *)sc16;

    device const float * y1 = yy + ix*QK_K + y_offset;

    float sumf[nr0] = {0.f};

    for (int i = ix; i < nb; i += NG) {
        // runs: sub-blocks 2iq and 2iq + 1 of the first half, then the same of the second
        float4 yl[8];
        FOR_UNROLL (short k = 0; k < 4; ++k) {
            device const float4 * yk = (device const float4 *) (y1 + (k/2)*128 + (k%2)*32);
            yl[2*k + 0] = yk[0];
            yl[2*k + 1] = yk[1];
        }
        float4 sumy;
        FOR_UNROLL (short k = 0; k < 4; ++k) {
            const float4 t = yl[2*k + 0] + yl[2*k + 1];
            sumy[k] = t.x + t.y + t.z + t.w;
        }

        device const uint2 * q1 = (device const uint2 *) (x[i].qs + q_offset);
        device const uint2 * qh = (device const uint2 *) (x[i].qh + l0);
        device const uint4 * hp = (device const uint4 *) &x[i];

        for (short row = 0; row < nr0; ++row) {
            const uint4  hdr = *hp;
            const half2  dm  = as_type<half2>(hdr.x);

            const uint16_t s0 = (uint16_t) (hdr.y >> shs);
            const uint16_t s2 = (uint16_t) (hdr.z >> shs);
            const uint16_t s4 = (uint16_t) (hdr.w >> shs);

            sc16[0] = s0 & kmask1;
            sc16[1] = s2 & kmask1;
            sc16[2] = ((s4 >> 0) & kmask2) | ((s0 & kmask3) >> 2);
            sc16[3] = ((s4 >> 4) & kmask2) | ((s2 & kmask3) >> 2);

            const uint2 qa = q1[0];
            const uint2 qb = q1[8];
            const uint2 hw = qh[0] >> hsh;

            float4 acc = {0.f, 0.f, 0.f, 0.f};
            FOR_UNROLL (short w = 0; w < 2; ++w) {
                acc[0] = mv_fma4(acc[0], float4(as_type<uchar4>(( qa[w]       & 0x0F0F0F0Fu) | ((hw[w] << 4) & 0x10101010u))), yl[0 + w]);
                acc[1] = mv_fma4(acc[1], float4(as_type<uchar4>(((qa[w] >> 4) & 0x0F0F0F0Fu) | ((hw[w] << 3) & 0x10101010u))), yl[2 + w]);
                acc[2] = mv_fma4(acc[2], float4(as_type<uchar4>(( qb[w]       & 0x0F0F0F0Fu) | ( hw[w]       & 0x10101010u))), yl[4 + w]);
                acc[3] = mv_fma4(acc[3], float4(as_type<uchar4>(((qb[w] >> 4) & 0x0F0F0F0Fu) | ((hw[w] >> 1) & 0x10101010u))), yl[6 + w]);
            }

            sumf[row] += (float) dm[0] * (sc8[0] * acc[0] + sc8[1] * acc[1] + sc8[4] * acc[2] + sc8[5] * acc[3]) -
                         (float) dm[1] * (sumy[0] * sc8[2] + sumy[1] * sc8[3] + sumy[2] * sc8[6] + sumy[3] * sc8[7]);

            q1 = (device const uint2 *) ((device const char *) q1 + args.nb01);
            qh = (device const uint2 *) ((device const char *) qh + args.nb01);
            hp = (device const uint4 *) ((device const char *) hp + args.nb01);
        }

        y1 += NG * QK_K;
    }

    device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

    for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
        const float tot = simd_sum(sumf[row]);
        if (tiisg == 0) {
            dst_f32[first_row + row] = tot;
        }
    }
}

[[host_name("kernel_mul_mv_q5_K_f32_v2")]]
kernel void kernel_mul_mv_q5_K_f32_v2(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q5_K_f32_v2_impl<N_R0_Q5_K, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xxs_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        threadgroup uint64_t * s64 = (threadgroup uint64_t *) shmem;
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 256; i += gsz) s64[i] = iq2xxs_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq2_xxs);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, then per 32 weights four grid indices and a word of sign indices and scale
                device const ushort * q2 = (device const ushort *) (xr + 2 + 8*ib);

                const uint  idx   = (uint) q2[0] | ((uint) q2[1] << 16);
                const uint  aux32 = (uint) q2[2] | ((uint) q2[3] << 16);
                const float d     = (float) *(device const half *) xr * (0.5f + (aux32 >> 28));

                float sum = 0.f;
                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint g  = (idx >> 8*l) & 0xFF;
                    const uint s7 = (aux32 >> 7*l) & 127;
                    const uint s  = s7 | ((popcount(s7) & 1) << 7);
                    sum = mv_fma4(sum, mv_sign4(float4(as_type<uchar4>(svalues[2*g + 0])), s),      yl[2*l + 0]);
                    sum = mv_fma4(sum, mv_sign4(float4(as_type<uchar4>(svalues[2*g + 1])), s >> 4), yl[2*l + 1]);
                }
                sumf[row] += d * sum;
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq2_xxs, N_R0_IQ2_XXS, "kernel_mul_mv_iq2_xxs_f32_v2")
MV_V2_GRID_KERNEL(iq2_xxs, 2, "kernel_mul_mv_iq2_xxs_f32_v2_r2")
MV_V2_GRID_KERNEL(iq2_xxs, 8, "kernel_mul_mv_iq2_xxs_f32_v2_r8")

template<int nr0, typename args_t>
void kernel_mul_mv_iq2_xs_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
    {
        threadgroup uint64_t * s64 = (threadgroup uint64_t *) shmem;
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < 512; i += gsz) s64[i] = iq2xs_grid[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq2_xs);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, then per 32 weights four shorts of 9-bit grid index and 7-bit sign index
                device const ushort * q2 = (device const ushort *) (xr + 2 + 8*ib);

                const uint  sc = (uint) *(device const uchar *) (xr + 2 + QK_K/4 + ib);
                const float db = (float) *(device const half *) xr;

                float sum[2] = {0.f, 0.f};
                FOR_UNROLL (short l = 0; l < 4; ++l) {
                    const uint q  = q2[l];
                    const uint g  = q & 511;
                    const uint s7 = q >> 9;
                    const uint s  = s7 | ((popcount(s7) & 1) << 7);
                    sum[l/2] = mv_fma4(sum[l/2], mv_sign4(float4(as_type<uchar4>(svalues[2*g + 0])), s),      yl[2*l + 0]);
                    sum[l/2] = mv_fma4(sum[l/2], mv_sign4(float4(as_type<uchar4>(svalues[2*g + 1])), s >> 4), yl[2*l + 1]);
                }
                sumf[row] += db * ((0.5f + (sc & 0xf)) * sum[0] + (0.5f + (sc >> 4)) * sum[1]);
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all * 0.25f;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq2_xs, N_R0_IQ2_XS, "kernel_mul_mv_iq2_xs_f32_v2")
MV_V2_GRID_KERNEL(iq2_xs, 2, "kernel_mul_mv_iq2_xs_f32_v2_r2")
MV_V2_GRID_KERNEL(iq2_xs, 8, "kernel_mul_mv_iq2_xs_f32_v2_r8")

// the iq1 grids hold eight values as nibbles: byte j is weight j and its high nibble weight j + 4
template<int nr0, typename args_t>
void kernel_mul_mv_iq1_s_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * sgrid = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < NGRID_IQ1S; i += gsz) sgrid[i] = iq1s_grid_gpu[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            float4 ys = 0.f;
            FOR_UNROLL (short i = 0; i < 8; ++i) {
                yl[i] = y4[i];
                ys += yl[i];
            }
            const float sumy = ys.x + ys.y + ys.z + ys.w;

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq1_s);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // d, qs[32] with the low 8 bits of the indices, qh[8]: their high 3 bits, scale, delta sign
                device const ushort * qs = (device const ushort *) (xr + 2 + 4*ib);

                const uint qi = (uint) qs[0] | ((uint) qs[1] << 16);
                const uint qh = *(device const ushort *) (xr + 2 + QK_K/8 + 2*ib);

                float sum = 0.f;
                FOR_UNROLL (short k = 0; k < 4; ++k) {
                    const uint g = sgrid[((qi >> 8*k) & 0xFF) | (((qh >> 3*k) & 7) << 8)];
                    sum = mv_fma4(sum, float4(as_type<uchar4>( g       & 0x0F0F0F0Fu)), yl[2*k + 0]);
                    sum = mv_fma4(sum, float4(as_type<uchar4>((g >> 4) & 0x0F0F0F0Fu)), yl[2*k + 1]);
                }
                const float delta = qh & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
                sumf[row] += (float) *(device const half *) xr * fma(sumy, delta, sum) * (2*((qh >> 12) & 7) + 1);
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq1_s, N_R0_IQ1_S, "kernel_mul_mv_iq1_s_f32_v2")
MV_V2_GRID_KERNEL(iq1_s, 2, "kernel_mul_mv_iq1_s_f32_v2_r2")
MV_V2_GRID_KERNEL(iq1_s, 8, "kernel_mul_mv_iq1_s_f32_v2_r8")

template<int nr0, typename args_t>
void kernel_mul_mv_iq1_m_f32_v2_impl(
        args_t args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem,
        uint3  tgpig,
        uint   ntg,
        ushort tiisg,
        ushort sgitg) {
    const short NSG = FC_mul_mv_nsg;

    const int nb = args.ne00/QK_K;

    threadgroup uint32_t * sgrid = (threadgroup uint32_t *) shmem;
    {
        const int gid = FC_mul_mv_nw*sgitg + tiisg;
        const int gsz = FC_mul_mv_nw*FC_mul_mv_nsg;
        for (int i = gid; i < NGRID_IQ1S; i += gsz) sgrid[i] = iq1s_grid_gpu[i];
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const int nrb = (args.ne01 + NSG*nr0 - 1)/(NSG*nr0);

    for (int r0 = tgpig.x; r0 < nrb; r0 += ntg) {
        const int r1 = tgpig.y;
        const int im = tgpig.z;

        const int first_row = (r0 * NSG + sgitg) * nr0;

        const uint i12 = im%FC_mul_mv_ne12;
        const uint i13 = im/FC_mul_mv_ne12;

        const uint64_t offset0 = first_row*args.nb01 + (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03;
        const uint64_t offset1 =        r1*args.nb11 + (i12        )*args.nb12 + (i13        )*args.nb13;

        device const char  * x = src0 + offset0;
        device const float * y = (device const float *) (src1 + offset1);

        float sumf[nr0] = {0.f};

        const int nb32 = nb * (QK_K/32);

        for (int ib32 = tiisg; ib32 < nb32; ib32 += FC_mul_mv_nw) {
            device const float4 * y4 = (device const float4 *) (y + 32*ib32);
            float4 yl[8];
            float  sy[4];
            FOR_UNROLL (short k = 0; k < 4; ++k) {
                yl[2*k + 0] = y4[2*k + 0];
                yl[2*k + 1] = y4[2*k + 1];
                const float4 t = yl[2*k + 0] + yl[2*k + 1];
                sy[k] = t.x + t.y + t.z + t.w;
            }

            const int ibl = ib32 / (QK_K/32);
            const int ib  = ib32 % (QK_K/32);

            device const char * xb = x + (uint64_t) ibl*sizeof(block_iq1_m);

            for (short row = 0; row < nr0; ++row) {
                device const char * xr = xb + (uint64_t) row*args.nb01;

                // qs[32], qh[16], scales[8]; the block scale is the top nibble of each scales short
                const uint   qi = mv_ld_u32(xr + 4*ib);
                const uint   qh = *(device const ushort *) (xr + QK_K/8 + 2*ib);
                const uint2  sw = *(device const uint2 *) (xr + QK_K/8 + QK_K/16);

                iq1m_scale_t scale;
                scale.u16 = ((sw.x >> 12) & 0x000f) | ((sw.x >> 24) & 0x00f0) | ((sw.y >> 4) & 0x0f00) | ((sw.y >> 16) & 0xf000);

                const uint sb = (((ib/2) < 2 ? sw.x : sw.y) >> (16*((ib/2) & 1))) >> (6*(ib%2));

                float sum[2] = {0.f, 0.f};
                FOR_UNROLL (short k = 0; k < 4; ++k) {
                    const uint g = sgrid[((qi >> 8*k) & 0xFF) | (((qh >> 4*k) & 7) << 8)];
                    sum[k/2] = mv_fma4(sum[k/2], float4(as_type<uchar4>( g       & 0x0F0F0F0Fu)), yl[2*k + 0]);
                    sum[k/2] = mv_fma4(sum[k/2], float4(as_type<uchar4>((g >> 4) & 0x0F0F0F0Fu)), yl[2*k + 1]);
                }
                float delta[2];
                FOR_UNROLL (short h = 0; h < 2; ++h) {
                    const uint q = qh >> 8*h;
                    delta[h] = sy[2*h + 0] * (q & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA) +
                               sy[2*h + 1] * (q & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
                }
                sumf[row] += (float) scale.f16 * ((sum[0] + delta[0]) * (2*((sb >> 0) & 7) + 1) +
                                                  (sum[1] + delta[1]) * (2*((sb >> 3) & 7) + 1));
            }
        }

        device float * dst_f32 = (device float *) dst + (uint64_t)im*args.ne0*args.ne1 + (uint64_t)r1*args.ne0;

        for (int row = 0; row < nr0 && first_row + row < args.ne0; ++row) {
            const float sum_all = simd_sum(sumf[row]);
            if (tiisg == 0) {
                dst_f32[first_row + row] = sum_all;
            }
        }
    }
}

MV_V2_GRID_KERNEL(iq1_m, N_R0_IQ1_M, "kernel_mul_mv_iq1_m_f32_v2")
MV_V2_GRID_KERNEL(iq1_m, 2, "kernel_mul_mv_iq1_m_f32_v2_r2")
MV_V2_GRID_KERNEL(iq1_m, 8, "kernel_mul_mv_iq1_m_f32_v2_r8")

// the expert mat-vec runs one row block per threadgroup, so the grid kernels' walk takes one step
#define MV_V2_ID_ADAPTER(T) \
template<int nr0, typename args_t> \
void kernel_mul_mv_##T##_f32_v2_id_impl( \
        args_t args, \
        device const char * src0, \
        device const char * src1, \
        device       char * dst, \
        threadgroup  char * shmem, \
        uint3  tgpig, \
        ushort tiisg, \
        ushort sgitg) { \
    kernel_mul_mv_##T##_f32_v2_impl<nr0, args_t>(args, src0, src1, dst, shmem, tgpig, 1u << 30, tiisg, sgitg); \
}

MV_V2_ID_ADAPTER(iq3_xxs)
MV_V2_ID_ADAPTER(iq3_s)
MV_V2_ID_ADAPTER(iq2_s)
MV_V2_ID_ADAPTER(iq2_xxs)
MV_V2_ID_ADAPTER(iq2_xs)
MV_V2_ID_ADAPTER(iq1_s)
MV_V2_ID_ADAPTER(iq1_m)

template [[host_name("kernel_mul_mv_id_q5_1_f32_L2r2")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<mul_vec_q_n_f32_l_impl<block_q5_1, 2, 2>>>;
template [[host_name("kernel_mul_mv_id_q5_K_f32_v2")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q5_K_f32_v2_impl<N_R0_Q5_K>>>;
template [[host_name("kernel_mul_mv_id_q3_K_f32_v2")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q3_K_f32_v2_impl<N_R0_Q3_K>>>;
template [[host_name("kernel_mul_mv_id_q2_K_f32_v2")]]    kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_q2_K_f32_v2_impl<N_R0_Q2_K>>>;
template [[host_name("kernel_mul_mv_id_iq3_xxs_f32_v2")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_xxs_f32_v2_id_impl<N_R0_IQ3_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq3_s_f32_v2")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq3_s_f32_v2_id_impl<N_R0_IQ3_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_s_f32_v2")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_s_f32_v2_id_impl<N_R0_IQ2_S>>>;
template [[host_name("kernel_mul_mv_id_iq2_xxs_f32_v2")]] kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xxs_f32_v2_id_impl<N_R0_IQ2_XXS>>>;
template [[host_name("kernel_mul_mv_id_iq2_xs_f32_v2")]]  kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq2_xs_f32_v2_id_impl<N_R0_IQ2_XS>>>;
template [[host_name("kernel_mul_mv_id_iq1_s_f32_v2")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_s_f32_v2_id_impl<N_R0_IQ1_S>>>;
template [[host_name("kernel_mul_mv_id_iq1_m_f32_v2")]]   kernel kernel_mul_mv_id_t kernel_mul_mv_id<mmv_fn<kernel_mul_mv_iq1_m_f32_v2_id_impl<N_R0_IQ1_M>>>;

// ── RX6800 phase variants (ported from Basten7/iRon-Llama-RC2 layer 2/5) ──
// PP likes larger nr0, decode stays at the small default; device.cpp picks the
// variant by phase via GGML_METAL_{PP,DECODE}_*_NR0 env knobs.

kernel void kernel_mul_mv_q4_K_f32_nr0_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_K_f32_nr0_8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q4_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q6_K_f32_nr0_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q6_K_f32_nr0_8(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q6_K_f32_impl<8, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, nullptr, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q8_0_f32_nr0_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    kernel_mul_mv_q8_0_f32_impl<4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q4_0_f32_nr0_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q4_0, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}

kernel void kernel_mul_mv_q5_0_f32_nr0_4(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    mul_vec_q_n_f32_impl<block_q5_0, 4, constant ggml_metal_kargs_mul_mv &>(args, src0, src1, dst, shmem, tgpig, tiisg, sgitg);
}



// MMQ-style IQ3_S matvec (Vulkan mul_mat_vec_iq3_s port): one TG per output
// row, 256 threads, TPB=16 threads per K-block, vec4 y loads, FMA chains,
// simd+shared reduce. Decode path (ne11==1); opt-in via dispatch suffix.
// Grid tables staged once per TG (2KB).
kernel void kernel_mul_mv_iq3_s_f32_mmq(
        constant ggml_metal_kargs_mul_mv & args,
        device const char * src0,
        device const char * src1,
        device       char * dst,
        threadgroup  char * shmem [[threadgroup(0)]],
        uint3  tgpig[[threadgroup_position_in_grid]],
        ushort tiisg[[thread_index_in_simdgroup]],
        ushort sgitg[[simdgroup_index_in_threadgroup]]) {
    // One TG per output row (wr), 256 threads. Each thread owns one 16-elem
    // aligned chunk (16c is 32-aligned within subgroup s=c/2 for even c and
    // offset 16 for odd c; the 16 elems never cross a 32-elem boundary).
    // Reduce via simd + shared. Decode path (ne11==1).
    // NOTE dispatch grid: x = tokens, y = weight rows (see encode site).
    const int r1 = tgpig.y;
    const int im = tgpig.z;

    const uint i12 = im % FC_mul_mv_ne12;
    const uint i13 = im / FC_mul_mv_ne12;

    // One TG covers 8 weight rows (dispatch packs nr0*nsg rows per TG);
    // all 256 threads collaborate on each row in turn.
    for (int rr = 0; rr < 8; ++rr) {
        const int wr = tgpig.x*8 + rr;
        if (wr >= args.ne01) {
            break;
        }
            device const block_iq3_s * xbase =
                (device const block_iq3_s *)(src0 + (uint64_t)wr*args.nb01 +
                    (i12/FC_mul_mv_r2)*args.nb02 + (i13/FC_mul_mv_r3)*args.nb03);
            device const float * y =
                (device const float *)(src1 + (uint64_t)r1*args.nb11 + i12*args.nb12 + i13*args.nb13);

            const int nb = args.ne00 / 256;

            threadgroup uint32_t * svalues = (threadgroup uint32_t *) shmem;
            {
                const uint tid0 = sgitg * 32u + tiisg;
                for (int k = tid0*2; k < 512; k += 512) {
                    svalues[k+0] = iq3s_grid[k+0];
                    svalues[k+1] = iq3s_grid[k+1];
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }

            const uint tid = sgitg * 32u + tiisg;
            const uint c = tid % 16;   // 16-elem chunk index within block
            const uint ix = tid / 16;  // block group (16 in flight)

            float temp = 0.f;
            for (int bi = ix; bi < nb; bi += 16) {
                device const block_iq3_s * xr = xbase + bi;
                device const half * dh = (device const half *)&xr->d;
                const float d = float(dh[0]);
                // chunk elems [16c, 16c+16): subgroup s, scale nibble, qh byte
                const uint s = c / 2;
                const uint h = c % 2;
                const uint8_t scb = xr->scales[s/2];
                const float dscale = d * (1.f + 2.f * (float)((scb >> (4*(s%2))) & 0xf));
                const uint8_t qh = xr->qh[s];
                float bsum = 0.f;
                for (uint li = 0; li < 4; ++li) {
                    const uint l = 4*h + li;   // qs index 0..7 in subgroup
                    const uint8_t qs0 = xr->qs[8*s + l];
                    const uint sbyte = xr->signs[4*s + l/2];
                    const uint32_t g0 = svalues[qs0 | (((qh >> l) & 1) << 8)];
                    const uint sh = (l & 1) ? 4 : 0;
                    for (uint j = 0; j < 4; ++j) {
                        float yv = y[bi*256 + 32*s + 16*h + 4*li + j];
                        float gv = (float)((g0 >> (8*j)) & 0xff);
                        float sv = ((sbyte >> (sh + j)) & 1u) ? -1.f : 1.f;
                        bsum += yv * gv * sv;
                    }
                }
                temp += dscale * bsum;
            }

            threadgroup float * sht = (threadgroup float *)(shmem + 2048);
            sht[tid] = temp;
            threadgroup_barrier(mem_flags::mem_threadgroup);
            if (tid == 0) {
                float sum = 0.f;
                for (int k = 0; k < 256; ++k) {
                    sum += sht[k];
                }
                device float * dst_f32 = (device float *)dst + im*args.ne0*args.ne1 + r1*args.ne0;
                dst_f32[wr] = sum;
            }

    }
}

