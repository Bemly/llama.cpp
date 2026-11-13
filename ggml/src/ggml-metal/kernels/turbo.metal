#include "common.h"
#include "dequantize.h"
#include "quantize.h"

kernel void kernel_turbo_wht(
        constant ggml_metal_kargs_turbo_wht & args,
        device const float * src [[buffer(1)]],
        device       float * dst [[buffer(2)]],
        uint tgpig [[threadgroup_position_in_grid]],
        uint tiitg [[thread_index_in_threadgroup]],
        uint ntg   [[threads_per_threadgroup]]) {
    // Each thread handles one 128-element group
    const int64_t group_idx = tgpig * ntg + tiitg;
    const int64_t n_groups = args.n_elements / 128;
    if (group_idx >= n_groups) return;

    const device float * in = src + group_idx * 128;
    device float * out = dst + group_idx * 128;

    // Load into half4 vectors for fast butterfly
    half4 v[32];
    const bool is_inverse = (args.direction == 1);

    // pick the sign table once: the ternary inside the loop made every element address both
    constant half4 * sfirst = is_inverse ? turbo_wht_signs2_h4 : turbo_wht_signs1_h4;
    constant half4 * slast  = is_inverse ? turbo_wht_signs1_h4 : turbo_wht_signs2_h4;

    // Apply first signs (s1 for fwd, s2 for inv)
    device const packed_float4 * in4 = (device const packed_float4 *) in;
    for (int i = 0; i < 32; i++) {
        v[i] = half4(float4(in4[i])) * sfirst[i];
    }

    // WHT butterfly (7 stages, vectorized half4)
    // h=1: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.y, a.x - a.y, a.z + a.w, a.z - a.w);
    }
    // h=2: within each half4
    for (int i = 0; i < 32; i++) {
        half4 a = v[i];
        v[i] = half4(a.x + a.z, a.y + a.w, a.x - a.z, a.y - a.w);
    }
    // h=4..64: between half4 vectors
    for (int h = 4; h < 128; h *= 2) {
        int vec_stride = h / 4;
        for (int i = 0; i < 32; i++) {
            int group_pos = i % (2 * vec_stride);
            if (group_pos < vec_stride) {
                int partner = i + vec_stride;
                half4 a = v[i], b = v[partner];
                v[i]       = a + b;
                v[partner] = a - b;
            }
        }
    }

    // Apply second signs + normalize, write output as fp32
    const half4 inv_sqrt = half4(0.08838834764831845h);
    for (int i = 0; i < 32; i++) {
        half4 s = slast[i];
        float4 f = float4(v[i] * inv_sqrt * s);
        *(device packed_float4 *)(out + i*4) = packed_float4(f);
    }
}

// TurboQuant requantize (f32 -> turbo) for the KV shift: WHT-rotate + PolarQuant the
// full 128-element group, byte-identical to kernel_set_rows_turbo. One block per thread.
kernel void kernel_cpy_f32_turbo3(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK_TURBO3;

    device block_turbo3_0 * dst_data = (device block_turbo3_0 *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK_TURBO3)*args.nb00);

        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j]*src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = src[j]*inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        device block_turbo3_0 & blk = dst_data[i00];
        for (int j = 0; j < QK_TURBO3/4; j++) blk.qs[j] = 0;
        for (int j = 0; j < QK_TURBO3/8; j++) blk.signs[j] = 0;

        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            float rv = x[j];
            uint8_t idx;
            if      (rv < turbo_mid_3bit[0]) idx = 0;
            else if (rv < turbo_mid_3bit[1]) idx = 1;
            else if (rv < turbo_mid_3bit[2]) idx = 2;
            else if (rv < turbo_mid_3bit[3]) idx = 3;
            else if (rv < turbo_mid_3bit[4]) idx = 4;
            else if (rv < turbo_mid_3bit[5]) idx = 5;
            else if (rv < turbo_mid_3bit[6]) idx = 6;
            else                              idx = 7;
            blk.qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            if (idx & 0x4) blk.signs[j/8] |= (1 << (j%8));
            float c = turbo_centroids_3bit[idx];
            recon_norm_sq += c*c;
        }
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm/recon_norm : grp_norm);
        break;
    }
}

kernel void kernel_cpy_f32_turbo2(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK_TURBO2;

    device block_turbo2_0 * dst_data = (device block_turbo2_0 *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK_TURBO2)*args.nb00);

        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j]*src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = src[j]*inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        device block_turbo2_0 & blk = dst_data[i00];
        for (int j = 0; j < QK_TURBO2/4; j++) blk.qs[j] = 0;

        float recon_norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) {
            float rv = x[j];
            uint8_t idx;
            if      (rv < turbo_mid_2bit[0]) idx = 0;
            else if (rv < turbo_mid_2bit[1]) idx = 1;
            else if (rv < turbo_mid_2bit[2]) idx = 2;
            else                              idx = 3;
            blk.qs[j/4] |= (idx & 0x3) << ((j%4)*2);
            float c = turbo_centroids_2bit[idx];
            recon_norm_sq += c*c;
        }
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm/recon_norm : grp_norm);
        break;
    }
}

kernel void kernel_cpy_f32_turbo4(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0)/QK_TURBO4;

    device block_turbo4_0 * dst_data = (device block_turbo4_0 *)(dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0);

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const float * src = (device const float *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01 + (i00*QK_TURBO4)*args.nb00);

        float norm_sq = 0.0f;
        for (int j = 0; j < 128; j++) norm_sq += src[j]*src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = src[j]*inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        device block_turbo4_0 & blk = dst_data[i00];
        for (int j = 0; j < QK_TURBO4/2; j++) blk.qs[j] = 0;

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
            blk.qs[j/2] |= (idx & 0xF) << ((j%2)*4);
            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c*c;
        }
        blk.rnorm = half(0.0f);
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm/recon_norm : grp_norm);
        break;
    }
}

// TurboQuant dequantize (turbo -> f32) for the KV shift: PolarQuant decode + inverse WHT,
// producing the pre-turbo domain the RoPE shift expects. One 128-element group per thread
// (op_cpy sets nk0 = ne00/128 for a turbo source).
kernel void kernel_cpy_turbo3_f32(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_turbo3_0 * src_data = (device const block_turbo3_0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device char * dst_row = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0;

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const block_turbo3_0 & blk = src_data[i00];
        const float nrm = (float) blk.norm;

        float x[128];
        for (int j = 0; j < 128; j++) {
            uint8_t low2 = (blk.qs[j>>2] >> ((j&3)*2)) & 0x3;
            uint8_t hi1  = (blk.signs[j>>3] >> (j&7)) & 0x1;
            x[j] = turbo_centroids_3bit[low2 | (hi1<<2)] * nrm;
        }
        turbo_rotate_inverse(x, turbo_wht_signs1, turbo_wht_signs2);

        device float * out = (device float *)(dst_row + (i00*QK_TURBO3)*args.nb0);
        for (int j = 0; j < 128; j++) out[j] = x[j];
        break;
    }
}

kernel void kernel_cpy_turbo2_f32(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_turbo2_0 * src_data = (device const block_turbo2_0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device char * dst_row = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0;

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const block_turbo2_0 & blk = src_data[i00];
        const float nrm = (float) blk.norm;

        float x[128];
        for (int j = 0; j < 128; j++) {
            uint8_t idx = (blk.qs[j>>2] >> ((j&3)*2)) & 0x3;
            x[j] = turbo_centroids_2bit[idx] * nrm;
        }
        turbo_rotate_inverse(x, turbo_wht_signs1, turbo_wht_signs2);

        device float * out = (device float *)(dst_row + (i00*QK_TURBO2)*args.nb0);
        for (int j = 0; j < 128; j++) out[j] = x[j];
        break;
    }
}

kernel void kernel_cpy_turbo4_f32(
        constant ggml_metal_kargs_cpy & args,
        device const char * src0,
        device       char * dst,
        uint3   tgpig[[threadgroup_position_in_grid]],
        ushort3 tpitg[[thread_position_in_threadgroup]],
        ushort3   ntg[[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig[2];
    const int32_t i02 = tgpig[1];
    const int32_t i01 = ntg[1] == 1 ? tgpig[0]%args.ne01 : tgpig[0]*ntg[1] + tpitg.y;
    const int32_t iw0 = ntg[1] == 1 ? tgpig[0]/args.ne01 : 0;
    if (i01 >= args.ne01) return;

    const int64_t n = i03*args.ne02*args.ne01*args.ne00 + i02*args.ne01*args.ne00 + i01*args.ne00;
    const int32_t i3 = n/(args.ne2*args.ne1*args.ne0);
    const int32_t i2 = (n - i3*args.ne2*args.ne1*args.ne0)/(args.ne1*args.ne0);
    const int32_t i1 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0)/args.ne0;
    const int32_t i0 = (n - i3*args.ne2*args.ne1*args.ne0 - i2*args.ne1*args.ne0 - i1*args.ne0);

    device const block_turbo4_0 * src_data = (device const block_turbo4_0 *)(src0 + i03*args.nb03 + i02*args.nb02 + i01*args.nb01);
    device char * dst_row = dst + i3*args.nb3 + i2*args.nb2 + i1*args.nb1 + i0*args.nb0;

    for (int32_t i00 = iw0*ntg[0] + tpitg.x; i00 < args.nk0;) {
        device const block_turbo4_0 & blk = src_data[i00];
        const float nrm = (float) blk.norm;

        float x[128];
        for (int j = 0; j < 128; j++) {
            uint8_t idx = (blk.qs[j>>1] >> ((j&1)*4)) & 0xF;
            x[j] = turbo_centroids_4bit[idx] * nrm;
        }
        turbo_rotate_inverse(x, turbo_wht_signs1, turbo_wht_signs2);

        device float * out = (device float *)(dst_row + (i00*QK_TURBO4)*args.nb0);
        for (int j = 0; j < 128; j++) out[j] = x[j];
        break;
    }
}

// Serial TurboQuant SET_ROWS reference kernels process one 128-value block at a time.
template<typename TI, typename block_q, int QK, void (*quantize_func)(device const float *, device block_q &)>
kernel void kernel_set_rows_turbo(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_q * dst_row = (      device block_q *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float   * src_row = (const device float   *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    const int blocks_per_group = QK_TURBO3_GROUP / QK;
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO3_GROUP * grp;

        // Normalize and rotate the full 128-element group
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO3_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // All blocks in a transform group store the same corrected norm.
        // Norm correction (ported from @spiritbuun's CUDA implementation):
        // Accumulate ||centroid_vector||^2 across all 128 elements, then store
        // grp_norm / ||centroid_vector|| instead of raw grp_norm. This makes
        // dequantized vectors have the exact original L2 norm at zero decode cost.
        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_q & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK;

            for (int j = 0; j < QK / 4; j++) blk.qs[j] = 0;
            for (int j = 0; j < QK / 8; j++) blk.signs[j] = 0;

            // Quantize rotated values to 3-bit centroids
            for (int j = 0; j < QK; j++) {
                float rv = x[off + j];  // rotated, normalized value
                uint8_t idx;
                if      (rv < turbo_mid_3bit[0]) idx = 0;
                else if (rv < turbo_mid_3bit[1]) idx = 1;
                else if (rv < turbo_mid_3bit[2]) idx = 2;
                else if (rv < turbo_mid_3bit[3]) idx = 3;
                else if (rv < turbo_mid_3bit[4]) idx = 4;
                else if (rv < turbo_mid_3bit[5]) idx = 5;
                else if (rv < turbo_mid_3bit[6]) idx = 6;
                else                              idx = 7;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);
                if (idx & 0x4) blk.signs[j / 8] |= (1 << (j % 8));

                // Accumulate centroid reconstruction norm for norm correction
                float c = turbo_centroids_3bit[idx];
                recon_norm_sq += c * c;
            }
        }

        // Norm correction: store corrected norm so dequant(x) has exact original L2 norm.
        // Zero decode cost — dequant already multiplies by stored norm.
        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant2 SET_ROWS kernel — 2-bit PolarQuant, 4 centroids, no signs byte.
// Same 128-element group WHT rotation as turbo3, but simpler quantization.
template<typename TI>
kernel void kernel_set_rows_turbo2(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo2_0 * dst_row = (      device block_turbo2_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float           *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Process in groups of 4 blocks (128 elements) for rotation
    const int blocks_per_group = QK_TURBO2_GROUP / QK_TURBO2;  // 128/32 = 4
    const int n_groups = args.nk0 / blocks_per_group;

    for (int grp = tiitg%tptg.x; grp < n_groups; grp += tptg.x) {
        const device float * grp_src = src_row + QK_TURBO2_GROUP * grp;

        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO2_GROUP; j++) norm_sq += grp_src[j] * grp_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        for (int j = 0; j < 128; j++) x[j] = grp_src[j] * inv_norm;
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        float recon_norm_sq = 0.0f;

        for (int b = 0; b < blocks_per_group; b++) {
            device block_turbo2_0 & blk = dst_row[grp * blocks_per_group + b];
            const int off = b * QK_TURBO2;

            for (int j = 0; j < QK_TURBO2 / 4; j++) blk.qs[j] = 0;

            for (int j = 0; j < QK_TURBO2; j++) {
                float rv = x[off + j];
                uint8_t idx;
                if      (rv < turbo_mid_2bit[0]) idx = 0;
                else if (rv < turbo_mid_2bit[1]) idx = 1;
                else if (rv < turbo_mid_2bit[2]) idx = 2;
                else                              idx = 3;

                blk.qs[j / 4] |= (idx & 0x3) << ((j % 4) * 2);

                float c = turbo_centroids_2bit[idx];
                recon_norm_sq += c * c;
            }
        }

        float recon_norm = sqrt(recon_norm_sq);
        float corrected_norm = (recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm;
        for (int b = 0; b < blocks_per_group; b++) {
            dst_row[grp * blocks_per_group + b].norm = half(corrected_norm);
        }
    }
}

// TurboQuant4 SET_ROWS kernel — processes 128 elements per block.
// Turbo4 uses one 128-element block with packed 4-bit indices.
template<typename TI>
kernel void kernel_set_rows_turbo4(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiitg[[thread_index_in_threadgroup]],
        uint3                tptg [[threads_per_threadgroup]]) {
    const int32_t i03 = tgpig.z;
    const int32_t i02 = tgpig.y;
    const int32_t i12 = i03%args.ne12;
    const int32_t i11 = i02%args.ne11;
    const int32_t i01 = tgpig.x*tptg.y + tiitg/tptg.x;
    if (i01 >= args.ne01) return;

    const int32_t i10 = i01;
    const TI      i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

          device block_turbo4_0 * dst_row = (      device block_turbo4_0 *) ((      device char *) dst  +  i1*args.nb1  + i02*args.nb2  + i03*args.nb3);
    const device float           * src_row = (const device float         *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);

    // Each block is one 128-element group (nk0 = ne0 / QK_TURBO4)
    const int n_blocks = args.nk0;

    for (int blk_idx = tiitg%tptg.x; blk_idx < n_blocks; blk_idx += tptg.x) {
        const device float * blk_src = src_row + QK_TURBO4 * blk_idx;
        device block_turbo4_0 & blk = dst_row[blk_idx];

        // Step 1: Compute norm + normalize
        float norm_sq = 0.0f;
        for (int j = 0; j < QK_TURBO4; j++) norm_sq += blk_src[j] * blk_src[j];
        float grp_norm = sqrt(norm_sq);
        float inv_norm = grp_norm > 1e-10f ? 1.0f / grp_norm : 0.0f;

        float x[128];
        float normalized[128];
        for (int j = 0; j < 128; j++) {
            normalized[j] = blk_src[j] * inv_norm;
            x[j] = normalized[j];
        }

        // Step 2: WHT rotate in-place
        turbo_rotate_forward(x, turbo_wht_signs1, turbo_wht_signs2);

        // Step 3: 4-bit PolarQuant — nibble packing (2 indices per byte)
        for (int j = 0; j < QK_TURBO4 / 2; j++) blk.qs[j] = 0;
        // qs[64] covers full nibble range — no signs field in 4-bit struct

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

            // 4-bit nibble pack: 2 elements per byte
            blk.qs[j / 2] |= (idx & 0xF) << ((j % 2) * 4);

            float c = turbo_centroids_4bit[idx];
            recon_norm_sq += c * c;
        }

        blk.rnorm = half(0.0f);  // reserved field, unused in 4-bit mode

        // Norm correction
        float recon_norm = sqrt(recon_norm_sq);
        blk.norm = half((recon_norm > 1e-10f) ? grp_norm / recon_norm : grp_norm);
    }
}


// Cooperative store: one 32-lane simdgroup per 128-element group (block == group).
// The serial kernel ran one thread per group, so decode (few groups) left the GPU
// idle; here the 32 lanes share the norm reduction, the WHT butterfly and the quant.
// Host dispatches grid.x = nk0*ne01 (flattened group+row), 32 threads per threadgroup.
template<typename TI>
kernel void kernel_set_rows_turbo3_coop(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiisg[[thread_index_in_simdgroup]]) {
    const int nk0  = args.nk0;
    const int flat = tgpig.x;
    const int grp  = flat % nk0;
    const int i01  = flat / nk0;
    if (i01 >= args.ne01) return;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int i12 = i03 % args.ne12;
    const int i11 = i02 % args.ne11;
    const int i10 = i01;
    const TI  i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

    device block_turbo3_0 * dst_row = (device block_turbo3_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const device float4 * gsrc = (const device float4 *) (src_row + 128*grp);

    const uint l = tiisg;          // 0..31, owns elements l*4 .. l*4+3
    float4 v = gsrc[l];

    const float grp_norm = sqrt(simd_sum(dot(v, v)));
    v *= grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

    // forward rotation: signs1 -> FWHT-128 -> signs2 (matches turbo_rotate_forward)
    v *= (float4) turbo_wht_signs1_h4[l];
    v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
    for (uint d = 1; d <= 16; d <<= 1) {
        float4 p = simd_shuffle_xor(v, d);
        v = (l & d) == 0 ? v + p : p - v;
    }
    v *= 0.08838834764831845f;   // 1/sqrt(128)
    v *= (float4) turbo_wht_signs2_h4[l];

    uint8_t qbyte = 0, snib = 0;
    float rns = 0.0f;
    for (short k = 0; k < 4; k++) {
        const float val = v[k];
        uint8_t idx;
        if      (val < turbo_mid_3bit[0]) idx = 0;
        else if (val < turbo_mid_3bit[1]) idx = 1;
        else if (val < turbo_mid_3bit[2]) idx = 2;
        else if (val < turbo_mid_3bit[3]) idx = 3;
        else if (val < turbo_mid_3bit[4]) idx = 4;
        else if (val < turbo_mid_3bit[5]) idx = 5;
        else if (val < turbo_mid_3bit[6]) idx = 6;
        else                              idx = 7;
        qbyte |= (idx & 0x3) << (k * 2);
        if (idx & 0x4) snib |= (1 << k);
        const float c = turbo_centroids_3bit[idx];
        rns += c * c;
    }
    const float recon_norm = sqrt(simd_sum(rns));
    const float corrected  = recon_norm > 1e-10f ? grp_norm/recon_norm : grp_norm;

    device block_turbo3_0 & blk = dst_row[grp];
    blk.qs[l] = qbyte;                                    // one qs byte per lane (unique)
    // two lanes share a signs byte (low nibble = even lane, high nibble = odd); merge
    const ushort contrib = ushort(snib) << ((l & 1) * 4);
    const ushort merged  = contrib | simd_shuffle_xor(contrib, 1);
    if ((l & 1) == 0) blk.signs[l >> 1] = uint8_t(merged);
    if (l == 0)       blk.norm = half(corrected);
}

// turbo4: 16 centroids, nibble-packed (2 per byte), no signs. Lane l writes qs[2l], qs[2l+1].
template<typename TI>
kernel void kernel_set_rows_turbo4_coop(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiisg[[thread_index_in_simdgroup]]) {
    const int nk0  = args.nk0;
    const int flat = tgpig.x;
    const int grp  = flat % nk0;
    const int i01  = flat / nk0;
    if (i01 >= args.ne01) return;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int i12 = i03 % args.ne12;
    const int i11 = i02 % args.ne11;
    const int i10 = i01;
    const TI  i1  = ((const device TI *) ((const device char *) src1 + i10*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

    device block_turbo4_0 * dst_row = (device block_turbo4_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const device float4 * gsrc = (const device float4 *) (src_row + 128*grp);

    const uint l = tiisg;
    float4 v = gsrc[l];

    const float grp_norm = sqrt(simd_sum(dot(v, v)));
    v *= grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;

    v *= (float4) turbo_wht_signs1_h4[l];
    v = float4(v.x + v.y, v.x - v.y, v.z + v.w, v.z - v.w);
    v = float4(v.x + v.z, v.y + v.w, v.x - v.z, v.y - v.w);
    for (uint d = 1; d <= 16; d <<= 1) {
        float4 p = simd_shuffle_xor(v, d);
        v = (l & d) == 0 ? v + p : p - v;
    }
    v *= 0.08838834764831845f;
    v *= (float4) turbo_wht_signs2_h4[l];

    uint8_t nib[4];
    float rns = 0.0f;
    for (short k = 0; k < 4; k++) {
        const float val = v[k];
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
        nib[k] = idx;
        const float c = turbo_centroids_4bit[idx];
        rns += c * c;
    }
    const float recon_norm = sqrt(simd_sum(rns));

    device block_turbo4_0 & blk = dst_row[grp];
    blk.qs[l*2]     = nib[0] | (nib[1] << 4);
    blk.qs[l*2 + 1] = nib[2] | (nib[3] << 4);
    if (l == 0) {
        blk.rnorm = half(0.0f);
        blk.norm  = half(recon_norm > 1e-10f ? grp_norm/recon_norm : grp_norm);
    }
}

// Wave64 store: one lane owns two values and all 64 lanes cover one 128-value block.
template<typename TI>
kernel void kernel_set_rows_turbo3_coop_w64(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiisg[[thread_index_in_simdgroup]]) {
    const int nk0  = args.nk0;
    const int flat = tgpig.x;
    const int grp  = flat % nk0;
    const int i01  = flat / nk0;
    if (i01 >= args.ne01) return;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int i12 = i03 % args.ne12;
    const int i11 = i02 % args.ne11;
    const TI i1 = ((const device TI *) ((const device char *) src1 + i01*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

    device block_turbo3_0 * dst_row = (device block_turbo3_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const uint l = tiisg;
    float2 v = *((const device float2 *) (src_row + 128*grp) + l);

    const float grp_norm = sqrt(simd_sum(dot(v, v)));
    v *= grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;
    v *= float2(turbo_wht_signs1[2*l], turbo_wht_signs1[2*l + 1]);
    v = float2(v.x + v.y, v.x - v.y);
    for (uint d = 1; d <= 32; d <<= 1) {
        const float2 p = simd_shuffle_xor(v, d);
        v = (l & d) == 0 ? v + p : p - v;
    }
    v *= 0.08838834764831845f;
    v *= float2(turbo_wht_signs2[2*l], turbo_wht_signs2[2*l + 1]);

    uint8_t idx[2];
    float rns = 0.0f;
    for (short k = 0; k < 2; ++k) {
        const float val = v[k];
        if      (val < turbo_mid_3bit[0]) idx[k] = 0;
        else if (val < turbo_mid_3bit[1]) idx[k] = 1;
        else if (val < turbo_mid_3bit[2]) idx[k] = 2;
        else if (val < turbo_mid_3bit[3]) idx[k] = 3;
        else if (val < turbo_mid_3bit[4]) idx[k] = 4;
        else if (val < turbo_mid_3bit[5]) idx[k] = 5;
        else if (val < turbo_mid_3bit[6]) idx[k] = 6;
        else                              idx[k] = 7;
        const float c = turbo_centroids_3bit[idx[k]];
        rns += c*c;
    }

    device block_turbo3_0 & blk = dst_row[grp];
    const ushort qpart = ushort((idx[0] & 3) | ((idx[1] & 3) << 2)) << ((l & 1)*4);
    const ushort qbyte = qpart | simd_shuffle_xor(qpart, 1);
    if ((l & 1) == 0) blk.qs[l >> 1] = uint8_t(qbyte);

    ushort spart = ushort(((idx[0] >> 2) & 1) | (((idx[1] >> 2) & 1) << 1)) << ((l & 3)*2);
    spart |= simd_shuffle_xor(spart, 1);
    spart |= simd_shuffle_xor(spart, 2);
    if ((l & 3) == 0) blk.signs[l >> 2] = uint8_t(spart);

    const float recon_norm = sqrt(simd_sum(rns));
    if (l == 0) blk.norm = half(recon_norm > 1e-10f ? grp_norm/recon_norm : grp_norm);
}

template<typename TI>
kernel void kernel_set_rows_turbo4_coop_w64(
        constant ggml_metal_kargs_set_rows & args,
        device const  void * src0,
        device const  void * src1,
        device       float * dst,
        uint3                tgpig[[threadgroup_position_in_grid]],
        uint                 tiisg[[thread_index_in_simdgroup]]) {
    const int nk0  = args.nk0;
    const int flat = tgpig.x;
    const int grp  = flat % nk0;
    const int i01  = flat / nk0;
    if (i01 >= args.ne01) return;
    const int i02 = tgpig.y;
    const int i03 = tgpig.z;
    const int i12 = i03 % args.ne12;
    const int i11 = i02 % args.ne11;
    const TI i1 = ((const device TI *) ((const device char *) src1 + i01*args.nb10 + i11*args.nb11 + i12*args.nb12))[0];

    device block_turbo4_0 * dst_row = (device block_turbo4_0 *) ((device char *) dst + i1*args.nb1 + i02*args.nb2 + i03*args.nb3);
    const device float * src_row = (const device float *) ((const device char *) src0 + i01*args.nb01 + i02*args.nb02 + i03*args.nb03);
    const uint l = tiisg;
    float2 v = *((const device float2 *) (src_row + 128*grp) + l);

    const float grp_norm = sqrt(simd_sum(dot(v, v)));
    v *= grp_norm > 1e-10f ? 1.0f/grp_norm : 0.0f;
    v *= float2(turbo_wht_signs1[2*l], turbo_wht_signs1[2*l + 1]);
    v = float2(v.x + v.y, v.x - v.y);
    for (uint d = 1; d <= 32; d <<= 1) {
        const float2 p = simd_shuffle_xor(v, d);
        v = (l & d) == 0 ? v + p : p - v;
    }
    v *= 0.08838834764831845f;
    v *= float2(turbo_wht_signs2[2*l], turbo_wht_signs2[2*l + 1]);

    uint8_t idx[2];
    float rns = 0.0f;
    for (short k = 0; k < 2; ++k) {
        const float val = v[k];
        if      (val < turbo_mid_4bit[ 0]) idx[k] = 0;
        else if (val < turbo_mid_4bit[ 1]) idx[k] = 1;
        else if (val < turbo_mid_4bit[ 2]) idx[k] = 2;
        else if (val < turbo_mid_4bit[ 3]) idx[k] = 3;
        else if (val < turbo_mid_4bit[ 4]) idx[k] = 4;
        else if (val < turbo_mid_4bit[ 5]) idx[k] = 5;
        else if (val < turbo_mid_4bit[ 6]) idx[k] = 6;
        else if (val < turbo_mid_4bit[ 7]) idx[k] = 7;
        else if (val < turbo_mid_4bit[ 8]) idx[k] = 8;
        else if (val < turbo_mid_4bit[ 9]) idx[k] = 9;
        else if (val < turbo_mid_4bit[10]) idx[k] = 10;
        else if (val < turbo_mid_4bit[11]) idx[k] = 11;
        else if (val < turbo_mid_4bit[12]) idx[k] = 12;
        else if (val < turbo_mid_4bit[13]) idx[k] = 13;
        else if (val < turbo_mid_4bit[14]) idx[k] = 14;
        else                               idx[k] = 15;
        const float c = turbo_centroids_4bit[idx[k]];
        rns += c*c;
    }

    device block_turbo4_0 & blk = dst_row[grp];
    blk.qs[l] = idx[0] | (idx[1] << 4);
    const float recon_norm = sqrt(simd_sum(rns));
    if (l == 0) {
        blk.rnorm = half(0.0f);
        blk.norm = half(recon_norm > 1e-10f ? grp_norm/recon_norm : grp_norm);
    }
}

// TurboQuant3 set_rows instantiations (128-element block == rotation group)
typedef decltype(kernel_set_rows_turbo3_coop<int64_t>) set_rows_turbo3_t;

template [[host_name("kernel_set_rows_turbo3_i64")]] kernel set_rows_turbo3_t kernel_set_rows_turbo3_coop<int64_t>;
template [[host_name("kernel_set_rows_turbo3_i32")]] kernel set_rows_turbo3_t kernel_set_rows_turbo3_coop<int32_t>;

typedef decltype(kernel_set_rows_turbo3_coop_w64<int64_t>) set_rows_turbo3_w64_t;

template [[host_name("kernel_set_rows_turbo3_i64_w64")]] kernel set_rows_turbo3_w64_t kernel_set_rows_turbo3_coop_w64<int64_t>;
template [[host_name("kernel_set_rows_turbo3_i32_w64")]] kernel set_rows_turbo3_w64_t kernel_set_rows_turbo3_coop_w64<int32_t>;

// TurboQuant2 set_rows instantiations (dedicated kernel, 4x32-element blocks, no signs)
typedef decltype(kernel_set_rows_turbo2<int64_t>) set_rows_turbo2_t;

template [[host_name("kernel_set_rows_turbo2_i64")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int64_t>;
template [[host_name("kernel_set_rows_turbo2_i32")]] kernel set_rows_turbo2_t kernel_set_rows_turbo2<int32_t>;

// TurboQuant4 set_rows instantiations (128-element block == rotation group)
typedef decltype(kernel_set_rows_turbo4_coop<int64_t>) set_rows_turbo4_t;

template [[host_name("kernel_set_rows_turbo4_i64")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4_coop<int64_t>;
template [[host_name("kernel_set_rows_turbo4_i32")]] kernel set_rows_turbo4_t kernel_set_rows_turbo4_coop<int32_t>;

typedef decltype(kernel_set_rows_turbo4_coop_w64<int64_t>) set_rows_turbo4_w64_t;

template [[host_name("kernel_set_rows_turbo4_i64_w64")]] kernel set_rows_turbo4_w64_t kernel_set_rows_turbo4_coop_w64<int64_t>;
template [[host_name("kernel_set_rows_turbo4_i32_w64")]] kernel set_rows_turbo4_w64_t kernel_set_rows_turbo4_coop_w64<int32_t>;
