#pragma once

#include "ggml.h"

#ifdef __cplusplus
extern "C" {
#endif

struct ggml_metal_buffer_id {
    void * metal; // id<MTLBuffer>
    size_t offs;
};

typedef struct ggml_metal_device * ggml_metal_device_t;

//
// MTLFunctionConstantValues wrapper
//

typedef struct ggml_metal_cv * ggml_metal_cv_t;

ggml_metal_cv_t ggml_metal_cv_init(void);
void ggml_metal_cv_free(ggml_metal_cv_t cv);

void ggml_metal_cv_set_int16(ggml_metal_cv_t cv, int16_t value, int32_t idx);
void ggml_metal_cv_set_int32(ggml_metal_cv_t cv, int32_t value, int32_t idx);
void ggml_metal_cv_set_bool (ggml_metal_cv_t cv, bool    value, int32_t idx);

//
// MTLComputePipelineState wrapper
//

typedef struct ggml_metal_pipeline * ggml_metal_pipeline_t;

ggml_metal_pipeline_t ggml_metal_pipeline_init(void);
void ggml_metal_pipeline_free(ggml_metal_pipeline_t pipeline);

// a collection of pipelines
typedef struct ggml_metal_pipelines * ggml_metal_pipelines_t;

ggml_metal_pipelines_t ggml_metal_pipelines_init(void);
void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls);

void                  ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline);
ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name);

struct ggml_metal_pipeline_with_params {
    ggml_metal_pipeline_t pipeline;

    int nsg;

    int nr0;
    int nr1;

    size_t smem;

    bool c4;
    bool cnt;

    // lanes per simdgroup the dispatch asks for; 0 keeps the 32-lane default
    int sgw;

    // most threadgroups a dispatch launches, for kernels that walk row blocks; 0 is one per block
    int ntg;
};

int ggml_metal_pipeline_max_theads_per_threadgroup(struct ggml_metal_pipeline_with_params pipeline);

//
// MTLCommandBuffer wrapper
//

typedef void * ggml_metal_cmd_buf_t;

//
// MTLComputeCommandEncoder wrapper
//

typedef struct ggml_metal_encoder * ggml_metal_encoder_t;

ggml_metal_encoder_t ggml_metal_encoder_init(ggml_metal_cmd_buf_t cmd_buf_raw, bool concurrent);
void ggml_metal_encoder_free(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_debug_group_push(ggml_metal_encoder_t encoder, const char * name);
void ggml_metal_encoder_debug_group_pop (ggml_metal_encoder_t encoder);

void ggml_metal_encoder_set_pipeline(ggml_metal_encoder_t encoder, struct ggml_metal_pipeline_with_params pipeline);

// (a + b) + c over float4, compiled at runtime; the TP2 boundary's residual add
struct ggml_metal_pipeline_with_params ggml_metal_device_pipeline_add3(ggml_metal_device_t dev);
// the same sum followed by an RMS norm and its weight, one threadgroup per row
struct ggml_metal_pipeline_with_params ggml_metal_device_pipeline_add3_rms_mul(ggml_metal_device_t dev);
// true when t still lacks the peer partial of a TP2 boundary and the caller is its residual add
bool ggml_metal_fb_residual_take(ggml_metal_device_t dev, const struct ggml_tensor * t, struct ggml_metal_buffer_id * peer);
// the add whose first source is t folds a TP2 partial, so no fusion may take it in
bool ggml_metal_fb_residual_armed(ggml_metal_device_t dev, const struct ggml_tensor * t);

void ggml_metal_encoder_set_bytes (ggml_metal_encoder_t encoder, void * data, size_t size, int idx);
void ggml_metal_encoder_set_buffer(ggml_metal_encoder_t encoder, struct ggml_metal_buffer_id buffer, int idx);

void ggml_metal_encoder_set_threadgroup_memory_size(ggml_metal_encoder_t encoder, size_t size, int idx);

void ggml_metal_encoder_dispatch_threadgroups(ggml_metal_encoder_t encoder, int tg0, int tg1, int tg2, int tptg0, int tptg1, int tptg2);

// ends the encoder, signals or waits on `ev` at `value` in the command buffer, and reopens it
void ggml_metal_encoder_event(ggml_metal_encoder_t encoder, ggml_metal_cmd_buf_t cmd_buf, void * ev, uint64_t value, bool signal);

void ggml_metal_encoder_memory_barrier(ggml_metal_encoder_t encoder);

void ggml_metal_encoder_end_encoding(ggml_metal_encoder_t encoder);

//
// MTLLibrary wrapper
//

typedef struct ggml_metal_library * ggml_metal_library_t;

ggml_metal_library_t ggml_metal_library_init            (ggml_metal_device_t dev);
ggml_metal_library_t ggml_metal_library_init_from_source(ggml_metal_device_t dev, const char * source, bool verbose);

void ggml_metal_library_free(ggml_metal_library_t lib);

ggml_metal_device_t ggml_metal_library_get_device(ggml_metal_library_t lib);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline    (ggml_metal_library_t lib, const char * name);
struct ggml_metal_pipeline_with_params ggml_metal_library_compile_pipeline(ggml_metal_library_t lib, const char * base, const char * name, ggml_metal_cv_t cv);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base              (ggml_metal_library_t lib, enum ggml_op op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy               (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst, bool contig);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_flat          (ggml_metal_library_t lib, enum ggml_type tsrc, enum ggml_type tdst);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op, enum ggml_op_pool op_pool);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows          (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op, bool wave64);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat            (ggml_metal_library_t lib, enum ggml_type tsrc);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_mul          (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_softplus_mul      (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max          (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max_back     (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rms_norm_back     (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat_back       (ggml_metal_library_t lib);
// a kernel without function constants, compiled under its own name
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_by_name           (ggml_metal_library_t lib, const char * name);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_mask        (ggml_metal_library_t lib, const struct ggml_tensor * op, bool fill, bool has_score, bool has_blocks, bool has_extra, bool vec);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer_amd(ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv          (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t nc, bool use_silu);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched  (ggml_metal_library_t lib, const struct ggml_tensor * op, int ssm_conv_bs, int32_t nc, bool use_silu);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_decode   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool silu);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan          (ggml_metal_library_t lib, const struct ggml_tensor * op, bool tail);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma  (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool rows, bool fold, bool raw, bool l2q, bool l2k);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_turbo_wht         (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri         (ggml_metal_library_t lib, const struct ggml_tensor * op);
// RX6800: largest ne11 the IQ3_S multi-column mat-vec takes before the blocked mm (GGML_METAL_IQ3S_NC_MAX, default 512)
int ggml_metal_iq3s_nc_max(void);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net_back(ggml_metal_library_t lib, const char * stage);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_ta(ggml_metal_library_t lib, const struct ggml_tensor * op, bool cs);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nxpsg, int r1ptg, int nr0);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma        (ggml_metal_library_t lib, const struct ggml_tensor * op, int nsg, int nt, int rt, bool add);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto   (ggml_metal_library_t lib, const struct ggml_tensor * op, bool add);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv            (ggml_metal_library_t lib, const struct ggml_tensor * op);

// rows per simdgroup of the few-column mat-vec for this type, lane width and column count,
// 0 if the batch does not take it
int ggml_metal_mv_nc2_rows(enum ggml_type type, int simd_width, int ne11, int ne01, int ne00);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0    (ggml_metal_library_t lib, int ne02, int ne20);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_zero    (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_part       (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax_final      (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tosh_topk_moe     (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge     (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht              (ggml_metal_library_t lib, int n, enum ggml_type tsrc, bool w64);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_pq2_swiglu      (ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_dense_swiglu    (ggml_metal_library_t lib, const struct ggml_tensor * w);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id_mxfp4_swiglu_oai(ggml_metal_library_t lib);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht_fused        (ggml_metal_library_t lib, int n, int hd, int rep, int nk, bool w64);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k             (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix       (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge       (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe          (ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm, int32_t gating, bool pbias);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce        (ggml_metal_library_t lib, int32_t n_expert_used);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin               (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one           (ggml_metal_library_t lib, enum ggml_op op);
// scalar f32-accumulator bin pipeline with an explicit src1 type (f16 allreduce transport):
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one_src1      (ggml_metal_library_t lib, enum ggml_op op, enum ggml_type tsrc1);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm              (ggml_metal_library_t lib, const struct ggml_tensor * op, int32_t n_fuse);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d         (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake             (ggml_metal_library_t lib, enum ggml_type type);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_mm        (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_mm_reduce (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw        (ggml_metal_library_t lib, const struct ggml_tensor * op, bool tiled);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale           (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad               (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll              (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw    (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd      (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset            (ggml_metal_library_t lib, const struct ggml_tensor * op);
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal       (ggml_metal_library_t lib, const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_rows(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        bool    has_sparse,
        bool    has_rows,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t dv,
        int32_t nwg);

// ── RX6800 FA-RDNA2 自研 kernel pipeline ──
struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_amd_vec(
        ggml_metal_library_t lib,
        int32_t dk,
        int32_t nbc,
        bool    has_mask);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_amd_tile(
        ggml_metal_library_t lib,
        int32_t dk,
        int32_t nbc,
        bool    has_mask);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_amd_pf(
        ggml_metal_library_t lib,
        int32_t dk,
        bool    has_mask);

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_amd_reduce(
        ggml_metal_library_t lib,
        int32_t dk);

// MTLResidencySet wrapper

typedef void * ggml_metal_rset_t;

// a collection of residency sets (non-owning)
typedef struct ggml_metal_rsets * ggml_metal_rsets_t;

ggml_metal_rsets_t ggml_metal_rsets_init(ggml_metal_device_t dev);
void ggml_metal_rsets_free(ggml_metal_rsets_t rsets);

//
// device
//

enum ggml_metal_device_id {
    GGML_METAL_DEVICE_GENERIC = 0,

    GGML_METAL_DEVICE_M1,
    GGML_METAL_DEVICE_M1_PRO,
    GGML_METAL_DEVICE_M1_MAX,
    GGML_METAL_DEVICE_M1_ULTRA,
    GGML_METAL_DEVICE_M2,
    GGML_METAL_DEVICE_M2_PRO,
    GGML_METAL_DEVICE_M2_MAX,
    GGML_METAL_DEVICE_M2_ULTRA,
    GGML_METAL_DEVICE_M3,
    GGML_METAL_DEVICE_M3_PRO,
    GGML_METAL_DEVICE_M3_MAX,
    GGML_METAL_DEVICE_M3_ULTRA,
    GGML_METAL_DEVICE_M4,
    GGML_METAL_DEVICE_M4_PRO,
    GGML_METAL_DEVICE_M4_MAX,
    GGML_METAL_DEVICE_M5,
    GGML_METAL_DEVICE_M5_PRO,
    GGML_METAL_DEVICE_M5_MAX,
    GGML_METAL_DEVICE_M5_ULTRA,
    GGML_METAL_DEVICE_A18_PRO,
};

const char * ggml_metal_device_id_token(enum ggml_metal_device_id id);

struct ggml_metal_device_props {
    int device;
    int device_phys;
    int device_virt;
    char name[128];
    char desc[128];

    size_t max_buffer_size;
    size_t max_working_set_size;
    size_t max_theadgroup_memory_size;

    bool has_simdgroup_reduction;
    bool has_simdgroup_mm;
    bool use_mm_manual;
    bool has_unified_memory;
    // wave64 decode (experimental, opt-in): run the parameterized f16/q8_0 mat-vec
    // kernels on the GPU for GCN/Vega instead of falling back to the CPU.
    bool wave64_decode;

    // Hardware SIMD-group width (threadExecutionWidth), probed at init.
    // Apple Silicon and AMD RDNA report 32; AMD GCN/Vega report 64. The Metal
    // kernels assume 32, so a non-32 width signals a wave64 device.
    int simd_width;

    // GCN before Vega reads a dword off a word boundary as the aligned one, so the mat-vec
    // kernels take a pair of shorts there instead
    bool needs_aligned_loads;

    bool has_bfloat;
    bool has_tensor;
    bool use_residency_sets;
    bool use_shared_buffers;

    enum ggml_metal_device_id device_id;
    int gpu_family;

    int op_offload_min_batch_size;
};

typedef struct ggml_metal_event * ggml_metal_event_t;

void ggml_metal_event_encode_signal(ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);
void ggml_metal_event_encode_wait  (ggml_metal_event_t ev, ggml_metal_cmd_buf_t cmd_buf);

ggml_metal_device_t ggml_metal_device_init(int device, int n_devices);
void ggml_metal_device_free(ggml_metal_device_t dev);

ggml_metal_device_t ggml_metal_device_get(int device, int n_devices);

void * ggml_metal_device_get_obj  (ggml_metal_device_t dev); // id<MTLDevice>
void * ggml_metal_device_get_queue(ggml_metal_device_t dev); // id<MTLCommandQueue>

// per-context queue: the first context gets the shared device queue, later
// ones an owned queue so their uploads can overlap compute
void * ggml_metal_device_acquire_queue(ggml_metal_device_t dev, bool * owned); // id<MTLCommandQueue>
void   ggml_metal_device_release_queue(ggml_metal_device_t dev, void * queue_raw, bool owned);

// cached no-copy wrap of the host pages containing [data, data + size);
// returns the id<MTLBuffer> (borrowed) and the blit offset, NULL on failure
void * ggml_metal_device_wrap_host(ggml_metal_device_t dev, const void * data, size_t size, size_t * offs);

ggml_metal_library_t ggml_metal_device_get_library(ggml_metal_device_t dev);

void ggml_metal_device_rsets_add(ggml_metal_device_t dev, ggml_metal_rset_t rset);
void ggml_metal_device_rsets_rm (ggml_metal_device_t dev, ggml_metal_rset_t rset);

void ggml_metal_device_rsets_keep_alive(ggml_metal_device_t dev);

ggml_metal_event_t ggml_metal_device_event_init(ggml_metal_device_t dev);
void ggml_metal_device_event_free(ggml_metal_device_t dev, ggml_metal_event_t ev);
void ggml_metal_device_event_synchronize(ggml_metal_device_t dev, ggml_metal_event_t ev);

void ggml_metal_device_get_memory(ggml_metal_device_t dev, size_t * free, size_t * total);
bool ggml_metal_device_supports_op(ggml_metal_device_t dev, const struct ggml_tensor * op);

const struct ggml_metal_device_props * ggml_metal_device_get_props(ggml_metal_device_t dev);

#ifdef TOSH_ENABLE_DYNAMIC_MOE
void *                      ggml_metal_device_dmoe_event(ggml_metal_device_t dev);
struct ggml_metal_buffer_id ggml_metal_device_dmoe_stage(ggml_metal_device_t dev, size_t layer_stride);
struct ggml_metal_buffer_id ggml_metal_device_dmoe_tables(ggml_metal_device_t dev);
struct ggml_metal_buffer_id ggml_metal_device_dmoe_obs_ring(ggml_metal_device_t dev);
struct ggml_metal_buffer_id ggml_metal_device_dmoe_wide_stage(ggml_metal_device_t dev, size_t bytes);
size_t ggml_metal_private_bytes(void);
void * ggml_metal_device_dmoe_reach_event(ggml_metal_device_t dev);
#endif

struct ggml_metal_fusion_info;

// the device-owned fusion debugging context (NULL unless fusion debugging is enabled)
struct ggml_metal_fusion_info * ggml_metal_device_get_fusion_info(ggml_metal_device_t dev);

//
// device buffers
//

typedef struct ggml_metal_buffer * ggml_metal_buffer_t;

ggml_metal_buffer_t ggml_metal_buffer_init(ggml_metal_device_t dev, size_t size, bool shared);
ggml_metal_buffer_t ggml_metal_buffer_map (ggml_metal_device_t dev, void * ptr, size_t size, size_t max_tensor_size);

void   ggml_metal_buffer_free     (ggml_metal_buffer_t buf);
void * ggml_metal_buffer_get_base (ggml_metal_buffer_t buf);
bool   ggml_metal_buffer_is_shared(ggml_metal_buffer_t buf);

void   ggml_metal_buffer_memset_tensor(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, uint8_t value, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor   (ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor_2d(ggml_metal_buffer_t buf, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size,
                                       size_t n_copies, size_t stride_tensor, size_t stride_data);
void   ggml_metal_buffer_get_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);
void   ggml_metal_buffer_set_tensor_n (ggml_metal_buffer_t buf, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors);
void   ggml_metal_buffer_get_tensor_n (ggml_metal_buffer_t buf, const struct ggml_tensor * const * tensors,       void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors);

// small host<->device transfers through the device's persistent staging buffer
// (synchronous); false when the size exceeds the stage, so callers fall back
bool   ggml_metal_device_stage_set(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_dst, const void * data, size_t size);
bool   ggml_metal_device_stage_set_2d(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_dst, const void * data, size_t size,
                                      size_t n_copies, size_t stride_tensor, size_t stride_data);
bool   ggml_metal_device_stage_get(ggml_metal_device_t dev, struct ggml_metal_buffer_id bid_src, void * data, size_t size);

// TOSH_MOE_PROFILE: per-tensor-name transfer accounting; no-op when disabled
void   ggml_metal_prof_note(const char * name, size_t size, bool d2h);

// batch small synchronous reads into one device round-trip
void   ggml_metal_device_read_batch_begin(ggml_metal_device_t dev);
void   ggml_metal_device_read_batch_end  (ggml_metal_device_t dev);

// wait out the in-flight staged uploads (stage_set defers its waits)
void   ggml_metal_device_upload_drain(ggml_metal_device_t dev);

bool   ggml_metal_buffer_cpy_tensor   (ggml_metal_buffer_t buf, const struct ggml_tensor * src, struct ggml_tensor * dst);
bool   ggml_metal_buffer_cpy_tensor_n (ggml_metal_buffer_t buf, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors);
void   ggml_metal_buffer_clear        (ggml_metal_buffer_t buf, uint8_t value);

// finds the Metal buffer that contains the tensor data on the GPU device
// the assumption is that there is 1-to-1 mapping between the host and device memory buffers, so we can find the
// Metal buffer based on the host memory pointer
//
struct ggml_metal_buffer_id ggml_metal_buffer_get_id(ggml_metal_buffer_t buf, const struct ggml_tensor * t);

// KVMem Metal blit: synchronous batched device-to-device copies for the
// KVMem layout fast path (discrete GPUs have no CPU-visible device memory,
// so the adapter cannot memcpy device rows on the host). All buffers in one
// call must belong to the same MTLDevice as the queue. void* values are
// id<MTLBuffer> / id<MTLCommandQueue>. Returns false (loud) on any failure.
//
void * ggml_metal_queue_for_buf(void * buf_ctx);
bool   ggml_metal_blit_batched(void * queue,
                               const void * const * src_buf, const size_t * src_off,
                               void * const * dst_buf, const size_t * dst_off,
                               const size_t * nbytes, int n);
void * ggml_metal_scratch_alloc(void * queue, size_t nbytes); // StorageModePrivate, +1 retain
void   ggml_metal_scratch_free(void * buf);
// Shared staging for harvest D2H (CPU-visible after blit + wait).
void * ggml_metal_staging_alloc(void * queue, size_t nbytes); // StorageModeShared, +1 retain
const void * ggml_metal_staging_bytes(const void * buf); // [buf contents]

#ifdef __cplusplus
}
#endif
