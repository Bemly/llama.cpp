#pragma once

#include "common.h"

#define GGML_COMMON_DECL_METAL
#define GGML_COMMON_IMPL_METAL
#if defined(GGML_METAL_EMBED_LIBRARY)
__embed_ggml-common.h__
#else
#include "ggml-common.h"
#endif

#define QK_NL 16 // shared by mul_mm and get_rows_q instantiations

// NOTE: this is not dequantizing - we are simply fitting the template
template <typename type4x4>
void dequantize_f32(device const float4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f32_t4(device const float4 * src, short il, thread type4 & reg) {
    reg = (type4)(*src);
}

template <typename type4x4>
void dequantize_bf16e(device const bf16e4x4 * src, short il, thread type4x4 & reg) {
    FOR_UNROLL (short i = 0; i < 4; ++i) {
        const float4 f = as_type<float4>(uint4(src->b[i]) << 16);
        reg[i][0] = f.x;
        reg[i][1] = f.y;
        reg[i][2] = f.z;
        reg[i][3] = f.w;
    }
}

template <typename type4>
void dequantize_bf16e_t4(device const bf16e4 * src, short il, thread type4 & reg) {
    reg = (type4) as_type<float4>(uint4(src->b) << 16);
}

template <typename type4x4>
void dequantize_f16(device const half4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_f16_t4(device const half4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}

#if defined(GGML_METAL_HAS_BF16)
template <typename type4x4>
void dequantize_bf16(device const bfloat4x4 * src, short il, thread type4x4 & reg) {
    reg = (type4x4)(*src);
}

template <typename type4>
void dequantize_bf16_t4(device const bfloat4 * src, short il, thread type4 & reg) {
    reg = (type4)(*(src));
}
#endif

template <typename type4x4>
void dequantize_q1_0(device const block_q1_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;
    const float neg_d = -d;

    const int byte_offset = il * 2;  // il*16 bits = il*2 bytes
    const uint8_t b0 = qs[byte_offset];
    const uint8_t b1 = qs[byte_offset + 1];

    float4x4 reg_f;

    reg_f[0][0] = select(neg_d, d, bool(b0 & 0x01));
    reg_f[0][1] = select(neg_d, d, bool(b0 & 0x02));
    reg_f[0][2] = select(neg_d, d, bool(b0 & 0x04));
    reg_f[0][3] = select(neg_d, d, bool(b0 & 0x08));
    reg_f[1][0] = select(neg_d, d, bool(b0 & 0x10));
    reg_f[1][1] = select(neg_d, d, bool(b0 & 0x20));
    reg_f[1][2] = select(neg_d, d, bool(b0 & 0x40));
    reg_f[1][3] = select(neg_d, d, bool(b0 & 0x80));

    reg_f[2][0] = select(neg_d, d, bool(b1 & 0x01));
    reg_f[2][1] = select(neg_d, d, bool(b1 & 0x02));
    reg_f[2][2] = select(neg_d, d, bool(b1 & 0x04));
    reg_f[2][3] = select(neg_d, d, bool(b1 & 0x08));
    reg_f[3][0] = select(neg_d, d, bool(b1 & 0x10));
    reg_f[3][1] = select(neg_d, d, bool(b1 & 0x20));
    reg_f[3][2] = select(neg_d, d, bool(b1 & 0x40));
    reg_f[3][3] = select(neg_d, d, bool(b1 & 0x80));

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q1_0_t4(device const block_q1_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const float neg_d = -d;
    const int base = il * 4;
    const uint8_t byte = xb->qs[base / 8];
    const int s = base % 8;

    float4 reg_f;
    reg_f[0] = select(neg_d, d, bool((byte >> (s    )) & 1));
    reg_f[1] = select(neg_d, d, bool((byte >> (s + 1)) & 1));
    reg_f[2] = select(neg_d, d, bool((byte >> (s + 2)) & 1));
    reg_f[3] = select(neg_d, d, bool((byte >> (s + 3)) & 1));

    reg = (type4) reg_f;
}

template <typename type4x4>
void dequantize_q2_0(device const block_q2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    const int byte_offset = il * 4;  // il*16 elements = il*4 bytes (4 elements per byte)
    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const uint8_t b = qs[byte_offset + i];
        reg_f[i][0] = ((float)((b >> 0) & 3) - 1.0f) * d;
        reg_f[i][1] = ((float)((b >> 2) & 3) - 1.0f) * d;
        reg_f[i][2] = ((float)((b >> 4) & 3) - 1.0f) * d;
        reg_f[i][3] = ((float)((b >> 6) & 3) - 1.0f) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q2_0_t4(device const block_q2_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;
    const uint8_t b = xb->qs[il];

    float4 reg_f;
    reg_f[0] = ((float)((b >> 0) & 3) - 1.0f) * d;
    reg_f[1] = ((float)((b >> 2) & 3) - 1.0f) * d;
    reg_f[2] = ((float)((b >> 4) & 3) - 1.0f) * d;
    reg_f[3] = ((float)((b >> 6) & 3) - 1.0f) * d;

    reg = (type4) reg_f;
}

// PQ2_0: each 2-bit field is OR-ed into the mantissa of 1024.0h, so the half minus 1025 is the
// signed weight; a 16-bit lane pair covers elements s and s + 8 of the 16 in one operation
template <typename type4x4>
void dequantize_pq2_0(device const block_pq2_0 * xb, short il, thread type4x4 & reg) {
    device const ushort * qs = (device const ushort *) xb->qs + 2*il;
    const uint w = uint(qs[0]) | (uint(qs[1]) << 16);
    const half d = xb->d;

    half4x4 r;
    FOR_UNROLL (short s = 0; s < 8; ++s) {
        const half2 h = (as_type<half2>(((w >> (2*s)) & 0x00030003u) | 0x64006400u) - half2(1025.0h))*d;
        r[s/4][s%4]     = h[0];
        r[s/4 + 2][s%4] = h[1];
    }
    reg = (type4x4) r;
}

template <typename type4>
void dequantize_pq2_0_t4(device const block_pq2_0 * xb, short il, thread type4 & reg) {
    const uint b = xb->qs[il];
    const float d = xb->d;

    reg = (type4) ((float4(b & 3u, (b >> 2) & 3u, (b >> 4) & 3u, b >> 6) - 1.0f)*d);
}

// element e of a PTQ1_0 block: 16 bytes carry trits 0..4 of elements 0..79, the next 8 bytes
// trits 0..4 of elements 80..119, and qh four trits each of elements 120..127
inline float ptq1_0_q(device const block_ptq1_0 * xb, int e) {
    const float pow3f[5] = {1.0f, 3.0f, 9.0f, 27.0f, 81.0f};
    float b;
    int n;
    if (e < 80) {
        b = xb->qs[e & 15];                n = e >> 4;
    } else if (e < 120) {
        b = xb->qs[16 + ((e - 80) & 7)];   n = (e - 80) >> 3;
    } else {
        b = xb->qh[(e - 120) & 1];         n = (e - 120) >> 1;
    }
    const float c0 = pow3f[n]*(1.0f/256.0f);
    return floor(3.0f*c0*b) - 3.0f*floor(c0*b);
}

// trit n of the two bytes held in the low halves of a pair of 16-bit lanes, as TQ1_0 packs them,
// ((b * 3^n) mod 256) * 3 >> 8, returned as the half pair of trit - 1
inline half2 ptq1_0_trit2(ushort2 b, ushort p) {
    const ushort2 t = (((b*p) & ushort2(0xFF))*ushort2(3)) >> 8;
    return as_type<half2>(t | ushort2(0x6400)) - half2(1025.0h);
}

// the sixteen elements of a call are regular in il: il 0..4 are trit il of qs[0..15]; il 5 and 6
// trits 2(il-5) and 2(il-5)+1 of qs[16..23]; il 7 trit 4 of those, then the four trits of each qh
// byte. Written as eight byte pairs with a power of three each, so lanes on different il run the
// same code instead of diverging.
template <typename type4x4>
void dequantize_ptq1_0(device const block_ptq1_0 * xb, short il, thread type4x4 & reg) {
    const half d = xb->d;
    device const ushort * q16 = (device const ushort *) xb->qs; // qs then qh, 13 byte pairs

    const bool lo = il < 5;
    const bool md = il < 7;
    const ushort p0 = il == 0 ? 1 : il == 1 ? 3 : il == 2 ? 9 : il == 3 ? 27 : 81; // 3^il below 5
    const ushort pm = il == 5 ? 1 : 9;                                              // 3^(2(il-5))

    half4x4 r;
    FOR_UNROLL (short j = 0; j < 8; j++) {
        const short  src = lo ? j : md ? 8 + (j & 3) : (j < 4 ? 8 + j : 12);
        const ushort hp  = j < 4 ? 81 : (j == 4 ? 1 : j == 5 ? 3 : j == 6 ? 9 : 27);
        const ushort p   = lo ? p0 : md ? (j < 4 ? pm : 3*pm) : hp;

        const ushort w = q16[src];
        const half2  t = ptq1_0_trit2(ushort2(w & 0xFF, w >> 8), p)*d;
        r[(2*j)/4][(2*j)%4]         = t[0];
        r[(2*j + 1)/4][(2*j + 1)%4] = t[1];
    }

    reg = (type4x4) r;
}

template <typename type4>
void dequantize_ptq1_0_t4(device const block_ptq1_0 * xb, short il, thread type4 & reg) {
    const float d = xb->d;

    float4 r;
    FOR_UNROLL (short k = 0; k < 4; ++k) {
        r[k] = (ptq1_0_q(xb, 4*il + k) - 1.0f)*d;
    }
    reg = (type4) r;
}

template <typename type4x4>
void dequantize_q4_0(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    // packed: the block is 18 bytes, so qs is only 2-byte aligned
    device const packed_ushort4 * qs = (device const packed_ushort4 *)((device const uint16_t *)xb + 1);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    const ushort4 q0 = ushort4(qs[0]);
    const ushort4 q1 = ushort4(qs[1]);

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const ushort qsi = i < 4 ? q0[i] : q1[i - 4];
        reg_f[i/2][2*(i%2) + 0] = d1 * (qsi & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qsi & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_0_t4(device const block_q4_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 1);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float md = -8.h * xb->d;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + md;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + md;
    }
}



template <typename type4x4>
void dequantize_q4_1(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    // packed: the block is 20 bytes, so qs is only 4-byte aligned
    device const packed_ushort4 * qs = (device const packed_ushort4 *)((device const uint16_t *)xb + 2);
    const float d1 = il ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    const ushort4 q0 = ushort4(qs[0]);
    const ushort4 q1 = ushort4(qs[1]);

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const ushort qsi = i < 4 ? q0[i] : q1[i - 4];
        reg_f[i/2][2*(i%2) + 0] = ((qsi & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qsi & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q4_1_t4(device const block_q4_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 2);
    const float d1 = (il/4) ? (xb->d / 16.h) : xb->d;
    const float d2 = d1 / 256.f;
    const float  m = xb->m;
    const ushort mask0 = (il/4) ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    for (int i = 0; i < 2; i++) {
        reg[2*i + 0] = d1 * (qs[2*(il%4) + i] & mask0) + m;
        reg[2*i + 1] = d2 * (qs[2*(il%4) + i] & mask1) + m;
    }
}

template <typename type4x4>
void dequantize_q5_0(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + md;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_0_t4(device const block_q5_0 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 3);
    const float d = xb->d;
    const float md = -16.h * xb->d;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + md;
        reg[2*ii + 1] = d * x1 + md;
    }
}

template <typename type4x4>
void dequantize_q5_1(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = il ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = il ? 4 : 0;

    const int gh_mv = il ? 12 : 0;
    const int gh_bk = il ?  0 : 4;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg_f[i/2][2*(i%2) + 0] = d * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d * x1 + m;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q5_1_t4(device const block_q5_1 * xb, short il, thread type4 & reg) {
    device const uint16_t * qs = ((device const uint16_t *)xb + 4);
    const float d = xb->d;
    const float m = xb->m;
    const ushort mask = (il/4) ? 0x00F0 : 0x000F;

    const uint32_t qh = *((device const uint32_t *)xb->qh);

    const int x_mv = (il/4) ? 4 : 0;

    const int gh_mv = (il/4) ? 12 : 0;
    const int gh_bk = (il/4) ?  0 : 4;

    for (int ii = 0; ii < 2; ii++) {
        int i = 2*(il%4) + ii;

        // extract the 5-th bits for x0 and x1
        const uint8_t xh_0 = ((qh >> (gh_mv + 2*i  )) << gh_bk) & 0x10;
        const uint8_t xh_1 = ((qh >> (gh_mv + 2*i+1)) << gh_bk) & 0x10;

        // combine the 4-bits from qs with the 5th bit
        const int32_t x0 = ((((qs[i]     ) & mask) >> x_mv) | xh_0);
        const int32_t x1 = ((((qs[i] >> 8) & mask) >> x_mv) | xh_1);

        reg[2*ii + 0] = d * x0 + m;
        reg[2*ii + 1] = d * x1 + m;
    }
}

template <typename type4x4>
void dequantize_q8_0(device const block_q8_0 *xb, short il, thread type4x4 & reg) {
    // 4 packed loads instead of 16 byte loads; packed_char4 because qs is only
    // 2-byte aligned inside the 34-byte block. int offset: AMD miscompiles short here
    device const packed_char4 * qs = (device const packed_char4 *)((device const int8_t *)xb->qs + 16*(int)il);
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const char4 q = qs[i];
        reg_f[i] = float4(q.x, q.y, q.z, q.w) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_q8_0_t4(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const packed_char4 * qs = (device const packed_char4 *) xb->qs;
    const float d = xb->d;

    reg = (type4) (float4(qs[il]) * d);
}

// the 34-byte block leaves every other quad off a 4-byte boundary, so a packed char4 load
// becomes four byte loads; the quads always sit on 2 bytes, so this reads two halves.
// Faster in attention on both widths and in the 64-lane mat-vec, slower in the 32-lane one
template <typename type4>
void dequantize_q8_0_t4_h2(device const block_q8_0 *xb, short il, thread type4 & reg) {
    device const ushort * qs = (device const ushort *) xb->qs + 2*il;
    const char2 lo = as_type<char2>(qs[0]);
    const char2 hi = as_type<char2>(qs[1]);
    const float d = xb->d;

    reg = (type4) (float4(lo.x, lo.y, hi.x, hi.y) * d);
}

template <typename type4x4>
void dequantize_mxfp4(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const uint8_t shr = il >= 1 ? 4 : 0;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        reg_f[i][0] = d * kvalues_mxfp4_f[(q2[4*i + 0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(q2[4*i + 1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(q2[4*i + 2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(q2[4*i + 3] >> shr) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_mxfp4_t4(device const block_mxfp4 * xb, short il, thread type4 & reg) {
    device const uint8_t * q2 = (device const uint8_t *)xb->qs;

    const float d = e8m0_to_fp32(xb->e);
    const short il4 = il%4;

    const uint8_t shr = il >= 4 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q2[4*il4 + 0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q2[4*il4 + 1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q2[4*il4 + 2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q2[4*il4 + 3] >> shr) & 0x0F];
}

// NVFP4: 64 weights per block split into four 16-weight sub-blocks, each with its own UE4M3 scale.
// the 16 weights of a sub-block are the low nibbles of its 8 bytes followed by the high nibbles
template <typename type4x4>
void dequantize_nvfp4(device const block_nvfp4 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * q = (device const uint8_t *)xb->qs + 8*il;

    const float d = ue4m3_to_fp32(xb->d[il]);

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        device const uint8_t * qb = q + 4*(i % 2);
        const uint8_t shr = i < 2 ? 0 : 4;

        reg_f[i][0] = d * kvalues_mxfp4_f[(qb[0] >> shr) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(qb[1] >> shr) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(qb[2] >> shr) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(qb[3] >> shr) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

template <typename type4>
void dequantize_nvfp4_t4(device const block_nvfp4 * xb, short il, thread type4 & reg) {
    const short s   = il/4; // sub-block
    const short il4 = il%4; // 4-weight chunk within the sub-block

    device const uint8_t * q = (device const uint8_t *)xb->qs + 8*s + 4*(il4 % 2);

    const float d = ue4m3_to_fp32(xb->d[s]);
    const uint8_t shr = il4 >= 2 ? 4 : 0;

    reg[0] = d * kvalues_mxfp4_f[(q[0] >> shr) & 0x0F];
    reg[1] = d * kvalues_mxfp4_f[(q[1] >> shr) & 0x0F];
    reg[2] = d * kvalues_mxfp4_f[(q[2] >> shr) & 0x0F];
    reg[3] = d * kvalues_mxfp4_f[(q[3] >> shr) & 0x0F];
}

template <typename type4x4>
void dequantize_q2_K(device const block_q2_K *xb, short il, thread type4x4 & reg) {
    const float d = xb->d;
    const float min = xb->dmin;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    float dl, ml;
    uint8_t sc = xb->scales[il];

    // 16-bit loads, high byte folded into the scale (see dequantize_q4_K).
    // int offset: AMD miscompiles short device address math.
    const int qoff = 32*((int) il/8) + 16*((int) il & 1);
    device const uint16_t * qv = (device const uint16_t *)(q + qoff);
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);

    const ushort mask1 = (ushort) mask << 8;
    const float  dl2   = dl / 256.f;

    float4x4 reg_f;

    for (int i = 0; i < 8; ++i) {
        reg_f[i/2][2*(i%2) + 0] = dl  * (qv[i] & mask)  - ml;
        reg_f[i/2][2*(i%2) + 1] = dl2 * (qv[i] & mask1) - ml;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q3_K(device const block_q3_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint8_t * q = (device const uint8_t *)xb->qs;
    device const uint8_t * h = (device const uint8_t *)xb->hmask;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    // Same 16-bit loads as dequantize_q4_K, over qs and hmask.
    const int qoff = 32 * ((int) il/8) + 16 * ((int) il & 1);
    const int hoff = 16 * ((int) il & 1);
    device const uint16_t * qv = (device const uint16_t *)(q + qoff);
    device const uint16_t * hv = (device const uint16_t *)(h + hoff);
    const ushort m0 = 1 << (il/2);
    const ushort m1 = m0 << 8;
    uint16_t kmask1 = (il/4)>1 ? ((il/4)>2 ? 192 : 48) : \
                                 ((il/4)>0 ? 12  : 3);
    uint16_t kmask2 = il/8 ? 0xF0 : 0x0F;
    uint16_t scale_2 = scales[il%8], scale_1 = scales[8 + il%4];
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

    float4x4 reg_f;

    for (int i = 0; i < 8; ++i) {
        const ushort hi = hv[i];
        reg_f[i/2][2*(i%2) + 0] = dl  * (qv[i] & mask)  - (hi & m0 ? 0.f : ml);
        reg_f[i/2][2*(i%2) + 1] = dl2 * (qv[i] & mask1) - (hi & m1 ? 0.f : ml);
    }

    reg = (type4x4) reg_f;
}

// uniform: j < 4 comes in as j_lo, known for the whole wave
template <bool uniform>
static inline uchar2 get_scale_min_k4_just2(int j, int k, device const uchar * q, bool j_lo) {
    return (uniform ? j_lo : j < 4) ? uchar2{uchar(q[j+0+k] & 63), uchar(q[j+4+k] & 63)}
                 : uchar2{uchar((q[j+4+k] & 0xF) | ((q[j-4+k] & 0xc0) >> 2)), uchar((q[j+4+k] >> 4) | ((q[j-0+k] & 0xc0) >> 2))};
}

template <bool uniform>
static inline uchar2 get_scale_min_k4_just2_w(int j, int k, uint w0, uint w1, uint w2, bool j_lo) {
    const int  a  = j + k;
    const uint sh = 8*(a & 3);

    const uint lo = (a < 4 ? w0 : w1) >> sh; // byte a
    const uint hi = (a < 4 ? w1 : w2) >> sh; // byte a+4

    if (uniform ? j_lo : j < 4) {
        return uchar2{uchar(lo & 63), uchar(hi & 63)};
    }

    const uint pv = w0 >> sh;                // byte a-4, only reached when j >= 4
    return uchar2{uchar((hi & 0xF) | ((pv & 0xc0) >> 2)), uchar(((hi & 0xFF) >> 4) | ((lo & 0xc0) >> 2))};
}

template <bool uniform, typename type4x4>
void dequantize_q4_K_impl(device const block_q4_K * xb, short il, bool lo, thread type4x4 & reg) {
    // high nibble picked by a shifted mask and folded into the scale (as in dequantize_q4_0).
    // int offsets: AMD miscompiles short device address math.
    // 128-bit load: the block is 144 bytes (9*16) so the window lands 16-byte aligned
    const int qoff = (int) (il/4) * 32 + 16 * ((int) il & 1);
    const uint4 qsv = *(device const uint4 *)(xb->qs + qoff);

    // d, dmin and the 12 scale bytes are the block's first 16 bytes, and the block is 16-byte
    // aligned, so one load replaces the two halves and the three separate scale bytes
    const uint4 hdr = *(device const uint4 *) xb;
    const half2 dm  = as_type<half2>(hdr.x);

    short is = (il/4) * 2;
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2_w<uniform>(is, il/2, hdr.y, hdr.z, hdr.w, lo);
    const float d   = il < 2 ? (float) dm[0] : (float) (dm[0] / 16.h);
    const float min = dm[1];
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask0 = il < 2 ? 0x000F : 0x00F0;
    const ushort mask1 = mask0 << 8;
    const float dl2 = dl / 256.f;

    // write straight into reg, the float4x4 staging was redundant
    using value_t = typename value_type<type4x4>::type;
    for (int i = 0; i < 8; ++i) {
        const ushort qsi = (i & 1) ? (ushort)(qsv[i/2] >> 16) : (ushort)(qsv[i/2] & 0xFFFF);
        reg[i/2][2*(i%2) + 0] = (value_t) (dl  * (qsi & mask0) - ml);
        reg[i/2][2*(i%2) + 1] = (value_t) (dl2 * (qsi & mask1) - ml);
    }
}

template <typename type4x4>
void dequantize_q4_K(device const block_q4_K * xb, short il, thread type4x4 & reg) {
    dequantize_q4_K_impl<false>(xb, il, false, reg);
}

template <bool uniform, typename type4x4>
void dequantize_q5_K_impl(device const block_q5_K *xb, short il, bool lo, thread type4x4 & reg) {
    // 128-bit loads: the block is 176 bytes (11*16) and both windows land 16-byte aligned
    const int qoff = 32 * (int) (il/4) + 16 * ((int) il & 1);
    const int hoff = 16 * ((int) il & 1);
    const uint4 qsv = *(device const uint4 *)(xb->qs + qoff);
    const uint4 qhv = *(device const uint4 *)(xb->qh + hoff);

    short is = (il/4) * 2;
    const ushort ul0 = 1 << (il/2);
    const ushort ul1 = ul0 << 8;
    il = il & 3;
    const uchar2 sc = get_scale_min_k4_just2<uniform>(is, il/2, xb->scales, lo);
    const float d = il < 2 ? xb->d : xb->d / 16.f;
    const float min = xb->dmin;
    const float dl = d * sc[0];
    const float ml = min * sc[1];

    const ushort mask0 = il < 2 ? 0x000F : 0x00F0;
    const ushort mask1 = mask0 << 8;
    const float dl2 = dl / 256.f;
    const float dqh = dl * (il < 2 ? 16.f : 256.f);

    float4x4 reg_f;

    for (int i = 0; i < 8; ++i) {
        const ushort qsi = (i & 1) ? (ushort)(qsv[i/2] >> 16) : (ushort)(qsv[i/2] & 0xFFFF);
        const ushort qhi = (i & 1) ? (ushort)(qhv[i/2] >> 16) : (ushort)(qhv[i/2] & 0xFFFF);
        reg_f[i/2][2*(i%2) + 0] = dl  * (qsi & mask0) + (qhi & ul0 ? dqh : 0.f) - ml;
        reg_f[i/2][2*(i%2) + 1] = dl2 * (qsi & mask1) + (qhi & ul1 ? dqh : 0.f) - ml;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q5_K(device const block_q5_K *xb, short il, thread type4x4 & reg) {
    dequantize_q5_K_impl<false>(xb, il, false, reg);
}

// for kernels that know j < 4 per wave; a no-op for every other block type
template <typename block_q, typename type4x4>
void dequantize_lo(device const block_q * xb, short il, bool lo, thread type4x4 & reg) {}

template <typename type4x4>
void dequantize_lo(device const block_q4_K * xb, short il, bool lo, thread type4x4 & reg) {
    dequantize_q4_K_impl<true>(xb, il, lo, reg);
}

template <typename type4x4>
void dequantize_lo(device const block_q5_K * xb, short il, bool lo, thread type4x4 & reg) {
    dequantize_q5_K_impl<true>(xb, il, lo, reg);
}

// Word-sized reads for the manual GEMM path, where narrow device loads are what the dequant
// pays for. q8_0 and q4_0 blocks sit on 2 bytes (q4_0 on 4 every other block), q4_1 and q2_K
// on 4. q3_K and q6_K lose with the alignment branch, so they keep their own dequant.
template <typename block_q>
constexpr bool has_dequantize_words() {
    return is_same<block_q, block_q8_0>::value || is_same<block_q, block_q4_0>::value ||
           is_same<block_q, block_q4_1>::value || is_same<block_q, block_q2_K>::value ||
           is_same<block_q, block_q5_0>::value || is_same<block_q, block_q5_1>::value ||
           is_same<block_q, block_mxfp4>::value;
}

// 16 bytes on a 4-byte boundary
static inline uint4 load_words16(device const uint16_t * p) {
    device const uint * q = (device const uint *) p;
    return uint4(q[0], q[1], q[2], q[3]);
}

static inline ushort word_half(uint4 w, int i) {
    return (i & 1) ? ushort(w[i/2] >> 16) : ushort(w[i/2] & 0xFFFF);
}

template <typename block_q, typename type4x4>
void dequantize_words(device const block_q * xb, short il, thread type4x4 & reg) {}

template <typename type4x4>
void dequantize_words(device const block_q8_0 * xb, short il, thread type4x4 & reg) {
    // a packed char4 still becomes four byte loads off a 4-byte boundary
    device const ushort * qs = (device const ushort *) xb->qs + 8*(int)il;
    const float d = xb->d;

    float4x4 reg_f;

    for (int i = 0; i < 4; i++) {
        const char2 lo = as_type<char2>(qs[2*i + 0]);
        const char2 hi = as_type<char2>(qs[2*i + 1]);
        reg_f[i] = float4(lo.x, lo.y, hi.x, hi.y) * d;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_words(device const block_q4_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q16 = (device const uint16_t *) xb;
    half dh;
    uint4 w;
    if ((ulong) xb & 2) {
        device const uint * q = (device const uint *)(q16 + 1);
        dh = as_type<half>(q16[0]);
        w  = uint4(q[0], q[1], q[2], q[3]);
    } else {
        // d and the first two quants share a word; the last two quants are read on their own
        device const uint * q = (device const uint *) q16;
        const uint a0 = q[0], a1 = q[1], a2 = q[2], a3 = q[3];
        const uint h  = q16[8];
        dh = as_type<half>(ushort(a0 & 0xFFFF));
        w  = uint4((a0 >> 16) | (a1 << 16), (a1 >> 16) | (a2 << 16), (a2 >> 16) | (a3 << 16), (a3 >> 16) | (h << 16));
    }

    const float d1 = il ? (dh / 16.h) : dh;
    const float d2 = d1 / 256.f;
    const float md = -8.h * dh;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const ushort qsi = word_half(w, i);
        reg_f[i/2][2*(i%2) + 0] = d1 * (qsi & mask0) + md;
        reg_f[i/2][2*(i%2) + 1] = d2 * (qsi & mask1) + md;
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_words(device const block_q4_1 * xb, short il, thread type4x4 & reg) {
    device const uint32_t * q32 = (device const uint32_t *) xb;
    const half2 dm = as_type<half2>(q32[0]);
    const uint4 w  = uint4(q32[1], q32[2], q32[3], q32[4]);

    const float d1 = il ? (dm.x / 16.h) : dm.x;
    const float d2 = d1 / 256.f;
    const float  m = dm.y;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const ushort qsi = word_half(w, i);
        reg_f[i/2][2*(i%2) + 0] = ((qsi & mask0) * d1) + m;
        reg_f[i/2][2*(i%2) + 1] = ((qsi & mask1) * d2) + m;
    }

    reg = (type4x4) reg_f;
}

// The fifth bit goes into the integer next to its nibble, so each weight is still one exact
// product and one add, as in dequantize_q5_0, with no per-weight shifts of the nibble.
template <typename type4x4>
static inline void dequantize_q5_words(uint4 w, uint qh, float d, float m, short il, thread type4x4 & reg) {
    const float d1 = il ? d/16.f : d;
    const float d2 = d1/256.f;
    const ushort mask0 = il ? 0x00F0 : 0x000F;
    const ushort mask1 = mask0 << 8;
    const uint hq = qh >> (16*il);
    const short s0 = 4 + 4*il;

    float4x4 reg_f;

    for (int i = 0; i < 8; i++) {
        const uint u  = word_half(w, i);
        const uint x0 = (u & mask0) | (((hq >> (2*i    )) & 1) << s0);
        const uint x1 = (u & mask1) | (((hq >> (2*i + 1)) & 1) << (s0 + 8));
        reg_f[i/2][2*(i%2) + 0] = d1 * x0 + m;
        reg_f[i/2][2*(i%2) + 1] = d2 * x1 + m;
    }

    reg = (type4x4) reg_f;
}

// 22-byte blocks: every other one starts 2 past a 4-byte boundary, which puts qh and qs on 4
template <typename type4x4>
void dequantize_words(device const block_q5_0 * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q16 = (device const uint16_t *) xb;
    float d;
    uint qh;
    uint4 w;
    if ((ulong) xb & 2) {
        device const uint * q = (device const uint *)(q16 + 1);
        d  = as_type<half>(q16[0]);
        qh = q[0];
        w  = uint4(q[1], q[2], q[3], q[4]);
    } else {
        // d shares a word with half of qh, and qh with the first two quants
        device const uint * q = (device const uint *) q16;
        d  = as_type<half>(ushort(q[0] & 0xFFFF));
        qh = (q[0] >> 16) | (q[1] << 16);
        w  = uint4((q[1] >> 16) | (q[2] << 16), (q[2] >> 16) | (q[3] << 16), (q[3] >> 16) | (q[4] << 16), (q[4] >> 16) | (uint(q16[10]) << 16));
    }
    dequantize_q5_words(w, qh, d, -16.f*d, il, reg);
}

// 24-byte blocks: always on 4 bytes
template <typename type4x4>
void dequantize_words(device const block_q5_1 * xb, short il, thread type4x4 & reg) {
    device const uint * q = (device const uint *) xb;
    const half2 dm = as_type<half2>(q[0]);
    dequantize_q5_words(uint4(q[2], q[3], q[4], q[5]), q[1], dm.x, dm.y, il, reg);
}

// 84-byte blocks: always on 4 bytes
template <typename type4x4>
void dequantize_words(device const block_q2_K * xb, short il, thread type4x4 & reg) {
    const half2 dm = as_type<half2>(*((device const uint *) &xb->d));
    const float d = dm.x;
    const float min = dm.y;
    const uint8_t sc = xb->scales[il];

    const int qoff = 32*((int) il/8) + 16*((int) il & 1);
    const uint4 qw = load_words16((device const uint16_t *)(xb->qs + qoff));
    il = (il/2)%4;

    half  coef = il>1 ? (il>2 ? 1/64.h : 1/16.h) : (il>0 ? 1/4.h : 1.h);
    uchar mask = il>1 ? (il>2 ? 192    : 48)     : (il>0 ? 12    : 3);
    const float dl = d * (sc & 0xF) * coef, ml = min * (sc >> 4);

    const ushort mask1 = (ushort) mask << 8;
    const float  dl2   = dl / 256.f;

    float4x4 reg_f;

    for (int i = 0; i < 8; ++i) {
        const ushort qv = word_half(qw, i);
        reg_f[i/2][2*(i%2) + 0] = dl  * (qv & mask)  - ml;
        reg_f[i/2][2*(i%2) + 1] = dl2 * (qv & mask1) - ml;
    }

    reg = (type4x4) reg_f;
}

// 17-byte blocks start on any byte, so five words from the boundary below cover one. The fifth
// can reach 3 bytes past the block, which stays inside the tensor's padded allocation.
template <typename type4x4>
void dequantize_words(device const block_mxfp4 * xb, short il, thread type4x4 & reg) {
    const uint s = (uint) ((ulong) xb & 3);
    device const uint * q = (device const uint *) ((device const uint8_t *) xb - s);
    const uint q0 = q[0], q1 = q[1], q2 = q[2], q3 = q[3], q4 = q[4];

    // 64-bit shifts, since s == 3 moves a whole word
    const uint sh = 8*s + 8;
    const uint4 w = uint4(uint(as_type<ulong>(uint2(q0, q1)) >> sh), uint(as_type<ulong>(uint2(q1, q2)) >> sh),
                          uint(as_type<ulong>(uint2(q2, q3)) >> sh), uint(as_type<ulong>(uint2(q3, q4)) >> sh));

    const float d = e8m0_to_fp32(uint8_t(q0 >> 8*s));
    const uint shr = 4*il;

    float4x4 reg_f;

    for (int i = 0; i < 4; ++i) {
        const uint x = w[i] >> shr;
        reg_f[i][0] = d * kvalues_mxfp4_f[(x      ) & 0x0F];
        reg_f[i][1] = d * kvalues_mxfp4_f[(x >>  8) & 0x0F];
        reg_f[i][2] = d * kvalues_mxfp4_f[(x >> 16) & 0x0F];
        reg_f[i][3] = d * kvalues_mxfp4_f[(x >> 24) & 0x0F];
    }

    reg = (type4x4) reg_f;
}

template <typename type4x4>
void dequantize_q6_K(device const block_q6_K *xb, short il, thread type4x4 & reg) {
    const half d_all = xb->d;
    device const uint16_t * ql = (device const uint16_t *)xb->ql;
    device const uint16_t * qh = (device const uint16_t *)xb->qh;
    device const int8_t * scales = (device const int8_t *)xb->scales;

    ql = ql + 32*(il/8) + 16*((il/2)&1) + 8*(il&1);
    qh = qh + 16*(il/8) + 8*(il&1);
    float sc = scales[(il%2) + 2 * ((il/2))];
    il = (il/2) & 3;

    const uint32_t kmask1 = il>1 ? (il>2 ? 0xC0C0C0C0 : 0x30303030) : (il>0 ? 0x0C0C0C0C : 0x03030303);
    const uint32_t kmask2 = il>1 ? 0xF0F0F0F0                       : 0x0F0F0F0F;
    const float ml = d_all * sc * 32.f;
    const float dl0 = d_all * sc;
    const float dl1 = dl0 / 256.f;
    const float dl2 = dl0 / (256.f * 256.f);
    const float dl3 = dl0 / (256.f * 256.f * 256.f);
    const uint8_t shr_h = il>2 ? 2 : 0;
    const uint8_t shl_h = il>1 ? 0 : (il>0 ? 2 : 4);
    const uint8_t shr_l = il>1 ? 4 : 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t  low = (ql[2*i] | (uint32_t)(ql[2*i+1] << 16)) & kmask2;
        const uint32_t high = (qh[2*i] | (uint32_t)(qh[2*i+1] << 16)) & kmask1;
        const uint32_t q = ((high << shl_h) >> shr_h) | (low >> shr_l);
        reg[i][0] = (typename value_type<type4x4>::type) (dl0 *  ((half)(q & 0xFF))       - ml);
        reg[i][1] = (typename value_type<type4x4>::type) (dl1 * ((float)(q & 0xFF00))     - ml);
        reg[i][2] = (typename value_type<type4x4>::type) (dl2 * ((float)(q & 0xFF0000))   - ml);
        reg[i][3] = (typename value_type<type4x4>::type) (dl3 * ((float)(q & 0xFF000000)) - ml);
    }
}

template <typename type4x4>
void dequantize_iq2_xxs(device const block_iq2_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    // each block of 32 needs 2 uint32_t's for the quants & scale, so 4 uint16_t's.
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const uint32_t aux32_g = q2[0] | (q2[1] << 16);
    const uint32_t aux32_s = q2[2] | (q2[3] << 16);
    thread const uint8_t * aux8 = (thread const uint8_t *)&aux32_g;
    const float dl = d * (0.5f + (aux32_s >> 28)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+0]);
    uint8_t signs = ksigns_iq2xs[(aux32_s >> 14*il) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
    grid = (constant uint8_t *)(iq2xxs_grid + aux8[2*il+1]);
    signs = ksigns_iq2xs[(aux32_s >> (14*il+7)) & 127];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq2_xs(device const block_iq2_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint16_t * q2 = xb->qs + 4*ib32;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+0] & 511));
    uint8_t signs = ksigns_iq2xs[q2[2*il+0] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
    grid = (constant uint8_t *)(iq2xs_grid + (q2[2*il+1] & 511));
    signs = ksigns_iq2xs[q2[2*il+1] >> 9];
    for (int i = 0; i < 8; ++i) {
        reg[2+i/4][i%4] = (typename value_type<type4x4>::type) (dl * grid[i] * (signs & kmask_iq2xs[i] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq3_xxs(device const block_iq3_xxs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * q3 = xb->qs + 8*ib32;
    device const uint16_t * gas = (device const uint16_t *)(xb->qs + QK_K/4) + 2*ib32;
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float dl = d * (0.5f + (aux32 >> 28)) * 0.5f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+0]);
    constant uint8_t * grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+1]);
    uint8_t signs = ksigns_iq2xs[(aux32 >> 14*il) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f));
        reg[1][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f));
    }
    grid1 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+2]);
    grid2 = (constant uint8_t *)(iq3xxs_grid + q3[4*il+3]);
    signs = ksigns_iq2xs[(aux32 >> (14*il+7)) & 127];
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * (signs & kmask_iq2xs[i+0] ? -1.f : 1.f));
        reg[3][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * (signs & kmask_iq2xs[i+4] ? -1.f : 1.f));
    }
}

template <typename type4x4>
void dequantize_iq3_s(device const block_iq3_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 8*ib32;
    device const uint8_t * signs = xb->signs + 4*ib32 + 2*il;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (1 + 2*((xb->scales[ib32/2] >> 4*(ib32%2)) & 0xf));
    constant uint8_t * grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+0] | ((qh << 8) & 256)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+1] | ((qh << 7) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i+0]));
        reg[1][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[0] & kmask_iq2xs[i+4]));
    }
    grid1 = (constant uint8_t *)(iq3s_grid + (qs[4*il+2] | ((qh << 6) & 256)));
    grid2 = (constant uint8_t *)(iq3s_grid + (qs[4*il+3] | ((qh << 5) & 256)));
    for (int i = 0; i < 4; ++i) {
        reg[2][i] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[1] & kmask_iq2xs[i+0]));
        reg[3][i] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i+4]));
    }
}

template <typename type4x4>
void dequantize_iq2_s(device const block_iq2_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const float d = xb->d;
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * signs = qs + QK_K/8;
    const uint8_t qh = xb->qh[ib32] >> 4*il;
    const float dl = d * (0.5f + ((xb->scales[ib32] >> 4*il) & 0xf)) * 0.25f;
    constant uint8_t * grid1 = (constant uint8_t *)(iq2s_grid + (qs[0] | ((qh << 8) & 0x300)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq2s_grid + (qs[1] | ((qh << 6) & 0x300)));
    for (int i = 0; i < 8; ++i) {
        reg[i/4+0][i%4] = (typename value_type<type4x4>::type) (dl * grid1[i] * select(1, -1, signs[0] & kmask_iq2xs[i]));
        reg[i/4+2][i%4] = (typename value_type<type4x4>::type) (dl * grid2[i] * select(1, -1, signs[1] & kmask_iq2xs[i]));
    }
}

template <typename type4x4>
void dequantize_iq1_s(device const block_iq1_s * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    const float d = xb->d;
    device const uint8_t  * qs = xb->qs + 4*ib32 + 2*il;
    device const uint16_t * qh = xb->qh;
    const float dl = d * (2*((qh[ib32] >> 12) & 7) + 1);
    const float ml = dl * (qh[ib32] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA);
    const uint16_t h = qh[ib32] >> 6*il;
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((h << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((h << 5) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] & 0xf) + ml);
        reg[1][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] >>  4) + ml);
        reg[2][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] & 0xf) + ml);
        reg[3][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] >>  4) + ml);
    }
}

template <typename type4x4>
void dequantize_iq1_m(device const block_iq1_m * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    device const uint16_t * sc = (device const uint16_t *)xb->scales;

    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    const float d = scale.f16;

    device const uint8_t * qs = xb->qs + 4*ib32 + 2*il;
    device const uint8_t * qh = xb->qh + 2*ib32 + il;

    const float dl  = d * (2*((sc[ib32/2] >> (6*(ib32%2)+3*il)) & 7) + 1);
    const float ml1 = dl * (qh[0] & 0x08 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    const float ml2 = dl * (qh[0] & 0x80 ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA);
    constant uint8_t * grid1 = (constant uint8_t *)(iq1s_grid_gpu + (qs[0] | ((qh[0] << 8) & 0x700)));
    constant uint8_t * grid2 = (constant uint8_t *)(iq1s_grid_gpu + (qs[1] | ((qh[0] << 4) & 0x700)));
    for (int i = 0; i < 4; ++i) {
        reg[0][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] & 0xf) + ml1);
        reg[1][i] = (typename value_type<type4x4>::type) (dl * (grid1[i] >>  4) + ml1);
        reg[2][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] & 0xf) + ml2);
        reg[3][i] = (typename value_type<type4x4>::type) (dl * (grid2[i] >>  4) + ml2);
    }
}

template <typename type4x4>
void dequantize_iq4_nl(device const block_iq4_nl * xb, short il, thread type4x4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = ((q4[2*i] | (q4[2*i+1] << 16)) >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[0]]);
        reg[i][1] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[1]]);
        reg[i][2] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[2]]);
        reg[i][3] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[3]]);
    }
}

template <typename type4>
void dequantize_iq4_nl_t4(device const block_iq4_nl * xb, short il, thread type4 & reg) {
    device const uint16_t * q4 = (device const uint16_t *)xb->qs;
    const float d = xb->d;
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    aux32 = ((q4[2*(il%4)] | (q4[2*(il%4)+1] << 16)) >> 4*(il/4)) & 0x0f0f0f0f;
    reg[0] = d * kvalues_iq4nl_f[q8[0]];
    reg[1] = d * kvalues_iq4nl_f[q8[1]];
    reg[2] = d * kvalues_iq4nl_f[q8[2]];
    reg[3] = d * kvalues_iq4nl_f[q8[3]];
}

template <typename type4x4>
void dequantize_iq4_xs(device const block_iq4_xs * xb, short il, thread type4x4 & reg) {
    // il is 0...15 for QK_K = 256 => index of block of 32 is il/2
    const int ib32 = il/2;
    il = il%2;
    // il = 0 or 1. il = 0 processes the first 16 quants in a block of 32, il = 1 the second 16
    device const uint32_t * q4 = (device const uint32_t *)xb->qs + 4*ib32;
    const int ls = ((xb->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((xb->scales_h >> 2*ib32) & 3) << 4);
    const float d = (float)xb->d * (ls - 32);
    uint32_t aux32;
    thread const uint8_t * q8 = (thread const uint8_t *)&aux32;
    for (int i = 0; i < 4; ++i) {
        aux32 = (q4[i] >> 4*il) & 0x0f0f0f0f;
        reg[i][0] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[0]]);
        reg[i][1] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[1]]);
        reg[i][2] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[2]]);
        reg[i][3] = (typename value_type<type4x4>::type) (d * kvalues_iq4nl_f[q8[3]]);
    }
}

template <typename type4x4>
void dequantize_tq2_0(device const block_tq2_0 * xb, short il, thread type4x4 & reg) {
    device const uint8_t * qs = xb->qs;
    const float d = xb->d;

    float4x4 reg_f;

    // 2 bits per element, 4 elements per byte, 128 elements per 32-byte group
    const short base = il * 16;
    for (int k = 0; k < 16; k++) {
        const int i = base + k;
        const int byte = ((i >> 7) & 1) * 32 + (i & 31);
        const int l = (i >> 5) & 3;
        reg_f[k/4][k%4] = d * (float)(((qs[byte] >> (2*l)) & 3) - 1);
    }

    reg = (type4x4) reg_f;
}

constant float turbo_centroids_2bit[4] = { -0.133462f, -0.039994f, 0.039994f, 0.133462f };
// 3-bit centroids for d=128
constant float turbo_centroids_3bit[8] = {
    -0.190685f, -0.117832f, -0.065717f, -0.021460f,
     0.021460f,  0.065717f,  0.117832f,  0.190685f
};
// Midpoints for 2-bit nearest centroid lookup
constant float turbo_mid_2bit[3] = { -0.086728f, 0.0f, 0.086728f };
// Midpoints for 3-bit
constant float turbo_mid_3bit[7] = { -0.154259f, -0.091775f, -0.043589f, 0.0f, 0.043589f, 0.091775f, 0.154259f };

// 4-bit PolarQuant centroids (16 levels) — optimal for N(0, 1/sqrt(128))
constant float turbo_centroids_4bit[16] = {
    -0.173926f, -0.117195f, -0.089527f, -0.068756f,
    -0.051262f, -0.035597f, -0.020989f, -0.006938f,
     0.006938f,  0.020989f,  0.035597f,  0.051262f,
     0.068756f,  0.089527f,  0.117195f,  0.173926f
};
constant float turbo_mid_4bit[15] = {
    -0.145560f, -0.103361f, -0.079142f, -0.060009f,
    -0.043430f, -0.028293f, -0.013963f,  0.000000f,
     0.013963f,  0.028293f,  0.043430f,  0.060009f,
     0.079142f,  0.103361f,  0.145560f
};

// Half-precision 4-bit centroid LUT for vec path
constant half turbo_centroids_4bit_h[16] = {
    -0.173926h, -0.117195h, -0.089527h, -0.068756h,
    -0.051262h, -0.035597h, -0.020989h, -0.006938h,
     0.006938h,  0.020989h,  0.035597h,  0.051262h,
     0.068756h,  0.089527h,  0.117195h,  0.173926h
};

// magnitudes of the 4-bit centroids, positive half only: they are symmetric, so index 8-15 map
// to mag[idx & 7] and 0-7 to mag[7 - (idx & 7)] with the sign flipped
constant half turbo_mag_4bit_h[8] = {
    0.006938h, 0.020989h, 0.035597h, 0.051262h,
    0.068756h, 0.089527h, 0.117195h, 0.173926h
};

// Half-precision 2-bit centroid LUT for vec path
constant half turbo_centroids_2bit_h[4] = {
    -0.133462h, -0.039994h, 0.039994h, 0.133462h
};

// Quantize 32 elements into one block_turbo2_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
void quantize_turbo2_0(device const float * src, device block_turbo2_0 & dst) {
#pragma METAL fp math_mode(safe)
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO2; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    for (int j = 0; j < QK_TURBO2 / 4; j++) dst.qs[j] = 0;

    for (int j = 0; j < QK_TURBO2; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_2bit[0]) idx = 0;
        else if (val < turbo_mid_2bit[1]) idx = 1;
        else if (val < turbo_mid_2bit[2]) idx = 2;
        else                              idx = 3;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
    }
}

// Quantize 32 elements into one block_turbo3_0 (NO rotation — rotation happens
// at the 128-element group level in kernel_set_rows_turbo)
void quantize_turbo3_0(device const float * src, device block_turbo3_0 & dst) {
#pragma METAL fp math_mode(safe)
    // Compute norm for this 32-element sub-block
    float norm_sq = 0.0f;
    for (int j = 0; j < QK_TURBO3; j++) norm_sq += src[j] * src[j];
    float norm = sqrt(norm_sq);
    float inv_norm = norm > 1e-10f ? 1.0f / norm : 0.0f;
    dst.norm = half(norm);

    // Quantize to 3-bit centroids
    for (int j = 0; j < QK_TURBO3 / 4; j++) dst.qs[j] = 0;
    for (int j = 0; j < QK_TURBO3 / 8; j++) dst.signs[j] = 0;

    for (int j = 0; j < QK_TURBO3; j++) {
        float val = src[j] * inv_norm;
        uint8_t idx;
        if      (val < turbo_mid_3bit[0]) idx = 0;
        else if (val < turbo_mid_3bit[1]) idx = 1;
        else if (val < turbo_mid_3bit[2]) idx = 2;
        else if (val < turbo_mid_3bit[3]) idx = 3;
        else if (val < turbo_mid_3bit[4]) idx = 4;
        else if (val < turbo_mid_3bit[5]) idx = 5;
        else if (val < turbo_mid_3bit[6]) idx = 6;
        else                              idx = 7;

        dst.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
        if (idx & 0x4) {
            dst.signs[j / 8] |= (1 << (j % 8));
        }
    }
}

void quantize_turbo4_0(device const float * src, device block_turbo4_0 & dst) {
#pragma METAL fp math_mode(safe)
    // 4-bit PolarQuant: normalize → rotate → quantize to 16 centroids → nibble pack
    float norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) norm_sq += src[j] * src[j];
    float grp_norm = sqrt(norm_sq);
    float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

    float x[128];
    for (int j = 0; j < 128; j++) x[j] = src[j] * inv_norm;
    turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

    for (int j = 0; j < QK_TURBO4 / 2; j++) dst.qs[j] = 0;

    float recon_norm_sq = 0.0f;
    for (int j = 0; j < 128; j++) {
        float val = x[j];
        uint8_t idx;
        if      (val < turbo_mid_4bit[ 0]) idx = 0;
        else if (val < turbo_mid_4bit[ 1]) idx = 1;
        else if (val < turbo_mid_4bit[ 2]) idx = 2;
        else if (val < turbo_mid_4bit[ 3]) idx = 3;
        else if (val < turbo_mid_4bit[ 4]) idx = 4;
        else if (val < turbo_mid_4bit[ 5]) idx = 5;
        else if (val < turbo_mid_4bit[ 6]) idx = 6;
        else if (val < turbo_mid_4bit[ 7]) idx = 7;
        else if (val < turbo_mid_4bit[ 8]) idx = 8;
        else if (val < turbo_mid_4bit[ 9]) idx = 9;
        else if (val < turbo_mid_4bit[10]) idx = 10;
        else if (val < turbo_mid_4bit[11]) idx = 11;
        else if (val < turbo_mid_4bit[12]) idx = 12;
        else if (val < turbo_mid_4bit[13]) idx = 13;
        else if (val < turbo_mid_4bit[14]) idx = 14;
        else                               idx = 15;

        dst.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);
        recon_norm_sq += turbo_centroids_4bit[idx] * turbo_centroids_4bit[idx];
    }

    dst.rnorm = half(0.0f);
    float recon_norm = sqrt(recon_norm_sq);
    dst.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
}

// TurboQuant FA dequant (t4): decode 4 elements of a 128-value block in the
// WHT-rotated domain (no inverse WHT — the graph un-rotates the attention output).
// il selects the 4-element chunk (0..31) within the block.
template <typename type4>
void dequantize_turbo3_0_t4(device const block_turbo3_0 * xb, short il, thread type4 & reg) {
    const float norm = (float) xb->norm;
    // the 4-element chunk shares one qs byte (il) and one signs nibble (il>>1); load once
    const uint8_t qb = xb->qs[il];
    const uint8_t sb = xb->signs[il >> 1];
    const short so = (il & 1) * 4;
    for (short j = 0; j < 4; j++) {
        uint8_t low2 = (qb >> (j * 2)) & 0x3;
        uint8_t hi1  = (sb >> (so + j)) & 0x1;
        reg[j] = turbo_centroids_3bit[low2 | (hi1 << 2)] * norm;
    }
}

template <typename type4>
void dequantize_turbo2_0_t4(device const block_turbo2_0 * xb, short il, thread type4 & reg) {
    const float norm = (float) xb->norm;
    const uint8_t qb = xb->qs[il];  // 4 x 2-bit indices in one byte
    for (short j = 0; j < 4; j++) {
        uint8_t idx = (qb >> (j * 2)) & 0x3;
        reg[j] = turbo_centroids_2bit[idx] * norm;
    }
}

template <typename type4>
void dequantize_turbo4_0_t4(device const block_turbo4_0 * xb, short il, thread type4 & reg) {
    const float norm = (float) xb->norm;
    const uint8_t b0 = xb->qs[il*2];      // elements 0,1 (nibbles)
    const uint8_t b1 = xb->qs[il*2 + 1];  // elements 2,3
    reg[0] = turbo_centroids_4bit[ b0       & 0xF] * norm;
    reg[1] = turbo_centroids_4bit[(b0 >> 4) & 0xF] * norm;
    reg[2] = turbo_centroids_4bit[ b1       & 0xF] * norm;
    reg[3] = turbo_centroids_4bit[(b1 >> 4) & 0xF] * norm;
}

