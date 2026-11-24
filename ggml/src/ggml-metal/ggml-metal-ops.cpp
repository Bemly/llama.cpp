#include "ggml-metal-ops.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-dmoe.h"
#endif

#include "ggml.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-common.h"
#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"
#include "ggml-metal-tuning.h"

#include <mutex>
#include <cassert>
#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <cmath>
#include <unordered_map>
#include <unordered_set>
#include <string>

// threadgroups needed to fill the GPU
// TODO: dedup
static constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

static ggml_metal_buffer_id ggml_metal_get_buffer_id(const ggml_tensor * t) {
    if (!t) {
        return { nullptr, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t) buffer->context;

    return ggml_metal_buffer_get_id(ctx, t);
}

static ggml_metal_buffer_id ggml_metal_get_host_buffer_id(
        ggml_metal_device_t dev, const ggml_tensor * t) {
    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    // The canonical CPU buffer type intentionally has no device pointer in ggml. Detect the
    // property we actually need instead: whether the storage is host-addressable.
    if (!ggml_backend_buffer_is_host(buffer)) {
        return ggml_metal_get_buffer_id(t);
    }

    // Wrap the whole CPU allocation once. The expert tensors are slices of this allocation;
    // mapping them independently duplicates address ranges and eventually exhausts Metal's
    // resource table on large models.
    void * const base = ggml_backend_buffer_get_base(buffer);
    const size_t size = ggml_backend_buffer_get_size(buffer);
    const size_t off  = (const char *) t->data - (const char *) base;

    size_t offs = 0;
    void * metal = ggml_metal_device_wrap_host(dev, base, size, &offs);
    if (metal == nullptr) {
        // A driver can reject a single allocation larger than maxBufferLength. Keep a bounded
        // compatibility fallback for smaller tensors; never mistake this for a VRAM budget.
        metal = ggml_metal_device_wrap_host(dev, t->data, ggml_nbytes(t), &offs);
        GGML_ASSERT(metal != nullptr && "failed to wrap the Dynamic MoE CPU bank for Metal");
        return { metal, offs };
    }

    return { metal, offs + off };
}

struct ggml_metal_op {
    ggml_metal_op(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int  idx_start,
        int  idx_end,
        bool use_concurrency,
        bool use_capture,
        int  debug_graph) {
        this->dev             = dev;
        this->lib             = ggml_metal_device_get_library(dev);
        this->cmd_buf         = cmd_buf;
        this->enc             = ggml_metal_encoder_init(cmd_buf, use_concurrency);
        this->mem_ranges      = ggml_mem_ranges_init(debug_graph);
        this->finfo           = finfo;
        this->idx_start       = idx_start;
        this->idx_end         = idx_end;
        this->use_concurrency = use_concurrency;
        this->use_capture     = use_capture;
        this->debug_graph     = debug_graph;
        this->gf              = gf;

        idxs.reserve(gf->n_nodes);

        // filter empty nodes
        // TODO: this can be removed when the allocator starts filtering them earlier
        //       https://github.com/ggml-org/llama.cpp/pull/16130#issuecomment-3327905830
        for (int i = idx_start; i < idx_end; i++) {
            if (!ggml_op_is_empty(gf->nodes[i]->op) && !ggml_is_empty(gf->nodes[i])) {
                idxs.push_back(i);
            }
        }
    }

    ~ggml_metal_op() {
        ggml_metal_encoder_end_encoding(this->enc);
        ggml_metal_encoder_free(this->enc);
        ggml_mem_ranges_free(this->mem_ranges);
    }

    int n_nodes() const {
        return idxs.size();
    }

    ggml_tensor * node(int i) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_graph_node(gf, idxs[i]);
    }

    const ggml_cgraph * graph() const {
        return gf;
    }

    int graph_index(int i) const {
        return idxs[i];
    }

    bool has_n_uses(int i, int n) const {
        assert(i >= 0 && i < (int) idxs.size());
        return ggml_node_has_n_uses(gf, idxs[i], n);
    }

    // fixed op sequence starting at i0, for the fusions that are not in the table
    bool can_fuse(int i0, const ggml_op * ops, int n_ops) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        if (i0 + n_ops > n_nodes()) {
            return false;
        }

        return ggml_can_fuse_ext(gf, idxs.data() + i0, ops, n_ops);
    }

    // consult the fusion table for the longest pattern starting at i0
    // returns the matching pattern (nullptr if no fusion) and sets *n_out to the number of nodes
    const ggml_metal_fusion * can_fuse(int i0, enum ggml_metal_fusion_mode mode, int * n_out) const {
        assert(use_fusion());
        assert(i0 >= 0 && i0 < n_nodes());

        return ggml_metal_fusion_next(gf, idxs.data(), (int) idxs.size(), i0, mode, n_out);
    }

    // whether to attempt fusion; the toggle lives in the shared fusion debugging context owned
    // by the device (initialized from GGML_METAL_FUSION_DISABLE, overridable by the test)
    bool use_fusion() const {
        return ggml_metal_fusion_info_enabled(finfo);
    }

    // record that a fusion fired, indexed by the matching table entry
    void count_fusions(const ggml_metal_fusion * fusion) const {
        ggml_metal_fusion_info_count_fusion(finfo, fusion);
    }

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;
    ggml_metal_cmd_buf_t cmd_buf;
    ggml_metal_encoder_t enc;
    ggml_mem_ranges_t    mem_ranges;

    // shared fusion debugging context
    ggml_metal_fusion_info * finfo;

    bool use_concurrency;
    bool use_capture;

    int debug_graph;

private:
    ggml_cgraph * gf;

    int idx_start;
    int idx_end;

    // non-empty node indices
    std::vector<int> idxs;
};

ggml_metal_op_t ggml_metal_op_init(
        ggml_metal_device_t dev,
        ggml_metal_cmd_buf_t cmd_buf,
        ggml_cgraph * gf,
        ggml_metal_fusion_info * finfo,
        int idx_start,
        int idx_end,
        bool use_concurrency,
        bool use_capture,
        int debug_graph) {
    ggml_metal_op_t res = new ggml_metal_op(
        dev,
        cmd_buf,
        gf,
        finfo,
        idx_start,
        idx_end,
        use_concurrency,
        use_capture,
        debug_graph);

    return res;
}

void ggml_metal_op_free(ggml_metal_op_t ctx) {
    delete ctx;
}

int ggml_metal_op_n_nodes(ggml_metal_op_t ctx) {
    return ctx->n_nodes();
}

static bool ggml_metal_op_concurrency_reset(ggml_metal_op_t ctx) {
    if (!ctx->mem_ranges) {
        return true;
    }

    ggml_metal_encoder_memory_barrier(ctx->enc);

    ggml_mem_ranges_reset(ctx->mem_ranges);

    return true;
}

static bool ggml_metal_op_concurrency_check(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return false;
    }

    return ggml_mem_ranges_check(ctx->mem_ranges, node);
}

static bool ggml_metal_op_concurrency_add(ggml_metal_op_t ctx, const ggml_tensor * node) {
    if (!ctx->mem_ranges) {
        return true;
    }

    return ggml_mem_ranges_add(ctx->mem_ranges, node);
}

static int32_t ggml_metal_use_count(const ggml_cgraph * gf, const ggml_tensor * t);
static const ggml_tensor * ggml_metal_unwrap_reshape(const ggml_cgraph * gf, const ggml_tensor * t);
static int ggml_metal_op_rms_norm_fwht(ggml_metal_op_t ctx, int idx);
static bool ggml_metal_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b);
static bool ggml_metal_gdn_state_fusion_on(ggml_metal_op_t ctx);
static bool ggml_metal_gdn_state_skip(ggml_metal_op_t ctx, int idx);

static int ggml_metal_op_encode_impl(ggml_metal_op_t ctx, int idx) {
    struct ggml_tensor * node = ctx->node(idx);

    //GGML_LOG_INFO("%s: encoding node %3d, op = %8s\n", __func__, idx, ggml_op_name(node->op));

    if (ggml_is_empty(node)) {
        return 1;
    }

    switch (node->op) {
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_PERMUTE:
            {
                // noop -> next node
                if (ctx->debug_graph > 0) {
                    GGML_LOG_DEBUG("%s: node[%5d] - %-12s %s\n", __func__, idx, ggml_op_name(node->op), "(noop)");
                }
            } return 1;
        default:
            {
            } break;
    }

    // A tensor split hands a device an empty slice when the weight has fewer whole
    // quantization blocks than there are devices. The op then contributes nothing, but
    // its result still enters the sum across devices, so write zeros rather than abort.
    if (!ggml_is_empty(node)) {
        bool empty_src = false;
        for (int i = 0; i < GGML_MAX_SRC; i++) {
            if (node->src[i] && ggml_is_empty(node->src[i])) {
                empty_src = true;
                break;
            }
        }
        if (empty_src && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID)) {
            if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                return 1;
            }
            ggml_metal_pipeline_with_params pz = ggml_metal_library_get_pipeline(ctx->lib, "kernel_zero_bytes");
            if (!pz.pipeline) {
                ggml_metal_cv_t cv = ggml_metal_cv_init();
                pz = ggml_metal_library_compile_pipeline(ctx->lib, "kernel_zero_bytes", "kernel_zero_bytes", cv);
                ggml_metal_cv_free(cv);
            }
            if (pz.pipeline) {
                const uint64_t n = ggml_nbytes(node);
                const int nth = std::min<int>(256, ggml_metal_pipeline_max_theads_per_threadgroup(pz));
                ggml_metal_encoder_set_pipeline(ctx->enc, pz);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(node), 0);
                ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &n, sizeof(n), 1);
                ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (int) ((n + nth - 1)/nth), 1, 1, nth, 1, 1);
                return 1;
            }
        }
    }

    if (!ggml_metal_device_supports_op(ctx->dev, node)) {
        GGML_LOG_ERROR("%s: error: unsupported op '%s'\n", __func__, ggml_op_desc(node));
        GGML_ABORT("unsupported op");
    }

    if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
        return 1;
    }

    int n_fuse = 1;

    // check if the current node can run concurrently with other nodes before it
    // the condition is that:
    //  - the current node cannot write to any previous src or dst ranges
    //  - the current node cannot read from any previous dst ranges
    //
    // if the condition is not satisfied, we put a memory barrier and clear all ranges
    // otherwise, we add the new ranges to the encoding context and process the node concurrently
    //
    {
        bool is_concurrent = ggml_metal_op_concurrency_check(ctx, node);

        if (is_concurrent && ctx->use_fusion()) {
            int n_fuse = 1;
            const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
            if (fusion) {
                // fused kernels write to the last node of the group, not necessarily to the first node's dst
                is_concurrent = ggml_mem_ranges_check(ctx->mem_ranges, ctx->node(idx + n_fuse - 1));
            }
        }

        if (!is_concurrent) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        if (ctx->debug_graph > 0) {
            GGML_LOG_DEBUG("%s: node[%5d] - %-12s %-12s %s\n", __func__, idx, ggml_op_name(node->op), ggml_get_name(node), is_concurrent ? "(concurrent)" : "");
        }
        if (ctx->debug_graph > 1) {
            GGML_TENSOR_LOCALS( int64_t, ne0, node->src[0], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb0, node->src[0], nb);
            GGML_TENSOR_LOCALS( int64_t, ne1, node->src[1], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb1, node->src[1], nb);
            GGML_TENSOR_LOCALS( int64_t, ne2, node->src[2], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb2, node->src[2], nb);
            GGML_TENSOR_LOCALS( int64_t, ne3, node->src[3], ne);
            GGML_TENSOR_LOCALS(uint64_t, nb3, node->src[3], nb);
            GGML_TENSOR_LOCALS( int64_t, ne,  node,         ne);
            GGML_TENSOR_LOCALS(uint64_t, nb,  node,         nb);

            if (node->src[0]) {
                GGML_LOG_DEBUG("%s: src0 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[0]->type), ne00, ne01, ne02, ne03, nb00, nb01, nb02, nb03,
                        ggml_is_contiguous(node->src[0]), node->src[0]->name);
            }
            if (node->src[1]) {
                GGML_LOG_DEBUG("%s: src1 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[1]->type), ne10, ne11, ne12, ne13, nb10, nb11, nb12, nb13,
                        ggml_is_contiguous(node->src[1]), node->src[1]->name);
            }
            if (node->src[2]) {
                GGML_LOG_DEBUG("%s: src2 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[2]->type), ne20, ne21, ne22, ne23, nb20, nb21, nb22, nb23,
                        ggml_is_contiguous(node->src[2]), node->src[2]->name);
            }
            if (node->src[3]) {
                GGML_LOG_DEBUG("%s: src3 - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], %d, %s\n", __func__, ggml_type_name(node->src[3]->type), ne30, ne31, ne32, ne33, nb30, nb31, nb32, nb33,
                        ggml_is_contiguous(node->src[3]), node->src[3]->name);
            }
            if (node) {
                GGML_LOG_DEBUG("%s: node - %4s [%5lld, %5lld, %5lld, %5lld] [%5lld, %5lld, %5lld, %5lld], 1, %s\n", __func__, ggml_type_name(node->type), ne0, ne1, ne2, ne3, nb0, nb1, nb2, nb3,
                        node->name);
            }
        }
    }

    if ((node->op == GGML_OP_GET_ROWS || node->op == GGML_OP_CPY) &&
        ggml_metal_gdn_state_fusion_on(ctx) && ggml_metal_gdn_state_skip(ctx, idx)) {
        return 1;
    }

    if ((node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_ADD) && ctx->use_fusion() && !ctx->use_concurrency) {
        const int n = ggml_metal_op_rms_norm_fwht(ctx, idx);
        if (n > 0) {
            return n;
        }
    }

    const int n_fwht = ctx->use_fusion() && (node->op == GGML_OP_MUL || node->op == GGML_OP_CONT) ?
        ggml_metal_op_fwht_fused(ctx, idx) : 0;

    if (n_fwht > 0) {
        n_fuse = n_fwht;
    } else
    switch (node->op) {
        case GGML_OP_CONCAT:
            {
                n_fuse = ggml_metal_op_concat(ctx, idx);
            } break;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            {
                n_fuse = ggml_metal_op_bin(ctx, idx);
            } break;
        case GGML_OP_ADD_ID:
            {
                n_fuse = ggml_metal_op_add_id(ctx, idx);
            } break;
        case GGML_OP_REPEAT:
            {
                n_fuse = ggml_metal_op_repeat(ctx, idx);
            } break;
        case GGML_OP_ACC:
            {
                n_fuse = ggml_metal_op_acc(ctx, idx);
            } break;
        case GGML_OP_SCALE:
        case GGML_OP_FILL:
        case GGML_OP_CLAMP:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_LOG:
        case GGML_OP_UNARY:
            {
                n_fuse = ggml_metal_op_tosh_gate_add(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_unary(ctx, idx);
                }
            } break;
        case GGML_OP_SILU_BACK:
            {
                n_fuse = ggml_metal_op_silu_back(ctx, idx);
            } break;
        case GGML_OP_GLU:
            {
                n_fuse = ggml_metal_op_glu(ctx, idx);
            } break;
        case GGML_OP_SUM:
            {
                n_fuse = ggml_metal_op_sum(ctx, idx);
            } break;
        case GGML_OP_SUM_ROWS:
        case GGML_OP_MEAN:
            {
                n_fuse = ggml_metal_op_sum_rows(ctx, idx);
            } break;
        case GGML_OP_CUMSUM:
            {
                n_fuse = ggml_metal_op_cumsum(ctx, idx);
            } break;
        case GGML_OP_LIGHTNING_INDEXER:
            {
                n_fuse = ggml_metal_op_lightning_indexer(ctx, idx);
            } break;
        case GGML_OP_TOP_K_MASK:
            {
                n_fuse = ggml_metal_op_top_k_mask(ctx, idx);
            } break;
        case GGML_OP_DSV4_HC_COMB:
        case GGML_OP_DSV4_HC_PRE:
        case GGML_OP_DSV4_HC_POST:
            {
                n_fuse = ggml_metal_op_dsv4_hc(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX:
            {
                n_fuse = ggml_metal_op_tosh_topk_moe(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_soft_max(ctx, idx);
                }
            } break;
        case GGML_OP_SSM_CONV:
            {
                n_fuse = ggml_metal_op_ssm_conv(ctx, idx);
            } break;
        case GGML_OP_SSM_SCAN:
            {
                n_fuse = ggml_metal_op_ssm_scan(ctx, idx);
            } break;
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_RWKV_WKV7:
            {
                n_fuse = ggml_metal_op_rwkv(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_GATED_DELTA_NET_RECORD:
            {
                n_fuse = ggml_metal_op_gated_delta_net(ctx, idx);
            } break;
        case GGML_OP_GATED_DELTA_NET_BACK:
            {
                n_fuse = ggml_metal_op_gated_delta_net_back(ctx, idx);
            } break;
        case GGML_OP_OUT_PROD:
            {
                n_fuse = ggml_metal_op_out_prod(ctx, idx);
            } break;
        case GGML_OP_TURBO_WHT:
            {
                n_fuse = ggml_metal_op_turbo_wht(ctx, idx);
            } break;
        case GGML_OP_SOLVE_TRI:
            {
                n_fuse = ggml_metal_op_solve_tri(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT:
            {
                n_fuse = ggml_metal_op_mul_mat(ctx, idx);
            } break;
        case GGML_OP_MUL_MAT_ID:
            {
                n_fuse = ggml_metal_op_mul_mat_id(ctx, idx);
            } break;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        case GGML_OP_TOSH_DMOE_PUBLISH:
            {
                n_fuse = ggml_metal_op_tosh_dmoe_publish(ctx, idx);
            } break;
        case GGML_OP_TOSH_DMOE_JOIN:
            {
                n_fuse = ggml_metal_op_tosh_dmoe_join(ctx, idx);
            } break;
        case GGML_OP_TOSH_DMOE_OBSERVE:
            {
                n_fuse = ggml_metal_op_tosh_dmoe_observe(ctx, idx);
            } break;
#endif
        case GGML_OP_GET_ROWS:
            {
                n_fuse = ggml_metal_op_get_rows(ctx, idx);
            } break;
        case GGML_OP_SET_ROWS:
            {
                n_fuse = ggml_metal_op_set_rows(ctx, idx);
            } break;
        case GGML_OP_DIAG:
            {
                n_fuse = ggml_metal_op_diag(ctx, idx);
            } break;
        case GGML_OP_L2_NORM:
            {
                n_fuse = ggml_metal_op_l2_norm(ctx, idx);
            } break;
        case GGML_OP_GROUP_NORM:
            {
                n_fuse = ggml_metal_op_group_norm(ctx, idx);
            } break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
            {
                n_fuse = ggml_metal_op_norm(ctx, idx);
            } break;
        case GGML_OP_RMS_NORM_BACK:
            {
                n_fuse = ggml_metal_op_rms_norm_back(ctx, idx);
            } break;
        case GGML_OP_SOFT_MAX_BACK:
            {
                n_fuse = ggml_metal_op_soft_max_back(ctx, idx);
            } break;
        case GGML_OP_REPEAT_BACK:
            {
                n_fuse = ggml_metal_op_repeat_back(ctx, idx);
            } break;
        case GGML_OP_CROSS_ENTROPY_ROWS:
        case GGML_OP_CROSS_ENTROPY_ROWS_BACK:
            {
                n_fuse = ggml_metal_op_cross_entropy_rows(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL_MASKED:
            {
                n_fuse = ggml_metal_op_count_equal_masked(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT_BACK:
            {
                n_fuse = ggml_metal_op_flash_attn_ext_back(ctx, idx);
            } break;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            {
                n_fuse = ggml_metal_op_rope(ctx, idx);
            } break;
        case GGML_OP_IM2COL:
            {
                n_fuse = ggml_metal_op_im2col(ctx, idx);
            } break;
        case GGML_OP_CONV_2D:
            {
                n_fuse = ggml_metal_op_conv_2d(ctx, idx);
            } break;
        case GGML_OP_CONV_2D_DW:
            {
                n_fuse = ggml_metal_op_conv_2d_dw(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                n_fuse = ggml_metal_op_conv_transpose_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            {
                n_fuse = ggml_metal_op_conv_transpose_2d(ctx, idx);
            } break;
        case GGML_OP_COL2IM_1D:
            {
                n_fuse = ggml_metal_op_col2im_1d(ctx, idx);
            } break;
        case GGML_OP_CONV_3D:
            {
                n_fuse = ggml_metal_op_conv_3d(ctx, idx);
            } break;
        case GGML_OP_UPSCALE:
            {
                n_fuse = ggml_metal_op_upscale(ctx, idx);
            } break;
        case GGML_OP_PAD:
            {
                n_fuse = ggml_metal_op_pad(ctx, idx);
            } break;
        case GGML_OP_PAD_REFLECT_1D:
            {
                n_fuse = ggml_metal_op_pad_reflect_1d(ctx, idx);
            } break;
        case GGML_OP_ROLL:
            {
                n_fuse = ggml_metal_op_roll(ctx, idx);
            } break;
        case GGML_OP_ARANGE:
            {
                n_fuse = ggml_metal_op_arange(ctx, idx);
            } break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            {
                n_fuse = ggml_metal_op_timestep_embedding(ctx, idx);
            } break;
        case GGML_OP_ARGSORT:
            {
                n_fuse = ggml_metal_op_tosh_topk_moe(ctx, idx);
                if (n_fuse == 0) {
                    n_fuse = ggml_metal_op_argsort(ctx, idx);
                }
            } break;
        case GGML_OP_TOP_K:
            {
                n_fuse = ggml_metal_op_top_k(ctx, idx);
            } break;
        case GGML_OP_TRI:
            {
                n_fuse = ggml_metal_op_tri(ctx, idx);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                n_fuse = ggml_metal_op_flash_attn_ext(ctx, idx);
            } break;
        case GGML_OP_SET:
            {
                n_fuse = ggml_metal_op_set(ctx, idx);
            } break;
        case GGML_OP_DUP:
        case GGML_OP_CPY:
        case GGML_OP_CONT:
            {
                n_fuse = ggml_metal_op_cpy(ctx, idx);
            } break;
        case GGML_OP_POOL_1D:
            {
                n_fuse = ggml_metal_op_pool_1d(ctx, idx);
            } break;
        case GGML_OP_POOL_2D:
            {
                n_fuse = ggml_metal_op_pool_2d(ctx, idx);
            } break;
        case GGML_OP_ARGMAX:
            {
                n_fuse = ggml_metal_op_argmax(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_ADAMW:
            {
                n_fuse = ggml_metal_op_opt_step_adamw(ctx, idx);
            } break;
        case GGML_OP_OPT_STEP_SGD:
            {
                n_fuse = ggml_metal_op_opt_step_sgd(ctx, idx);
            } break;
        case GGML_OP_COUNT_EQUAL:
            {
                n_fuse = ggml_metal_op_count_equal(ctx, idx);
            } break;
        default:
            {
                GGML_LOG_ERROR("%s: error: node %3d, op = %8s not implemented\n", __func__, idx, ggml_op_name(node->op));
                GGML_ABORT("fatal error");
            }
    }

    if (ctx->debug_graph > 0) {
        if (n_fuse > 1) {
            GGML_LOG_DEBUG("%s:               fuse %d ops\n", __func__, n_fuse);
        }
    }

    // update the mem ranges in the encoding context
    for (int i = 0; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_add(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
        }
    }

    return n_fuse;
}


// TOSH_OPPROF=1: CPU time spent encoding each op kind, to compare dispatch cost across engines
#include <time.h>
static uint64_t g_opprof_ns[GGML_OP_COUNT];
static uint64_t g_opprof_n [GGML_OP_COUNT];
static bool tosh_opprof_on(void) {
    static int v = -1;
    if (v < 0) { v = TOSH_ENV("TOSH_OPPROF") != nullptr; }
    return v;
}
static void tosh_opprof_dump(void) {
    if (!tosh_opprof_on()) return;
    for (int i = 0; i < GGML_OP_COUNT; i++) {
        if (g_opprof_n[i] == 0) continue;
        fprintf(stderr, "TOSH_OPPROF %-22s n=%-7llu total_us=%-10.1f us/op=%.3f\n",
                ggml_op_name((enum ggml_op) i),
                (unsigned long long) g_opprof_n[i],
                g_opprof_ns[i]/1e3, (double) g_opprof_ns[i]/g_opprof_n[i]/1e3);
    }
}

int ggml_metal_op_encode(ggml_metal_op_t ctx, int idx) {
    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_push(ctx->enc, ggml_op_desc(ctx->node(idx)));
    }

    static bool reg = false;
    if (!reg && tosh_opprof_on()) { reg = true; atexit(tosh_opprof_dump); }
    const enum ggml_op prof_op = ctx->node(idx)->op;
    const uint64_t prof_t0 = tosh_opprof_on() ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

    int res = ggml_metal_op_encode_impl(ctx, idx);

    if (prof_t0) {
        g_opprof_ns[prof_op] += clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - prof_t0;
        g_opprof_n [prof_op] += 1;
    }
    if (idx + res > ctx->n_nodes()) {
        GGML_ABORT("fusion error: nodes spanning multiple encoders have been fused. this indicates a bug in the fusion logic %s",
                "https://github.com/ggml-org/llama.cpp/pull/14849");
    }

    if (ctx->use_capture) {
        ggml_metal_encoder_debug_group_pop(ctx->enc);
    }

    return res;
}

int ggml_metal_op_concat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t dim = ((const int32_t *) op->op_params)[0];

    // a short state ahead of a transposed f32 view (the conv input of the recurrent models)
    const ggml_metal_device_props * props_cat = ggml_metal_device_get_props(ctx->dev);
    if (props_cat->simd_width == 32 && !props_cat->has_unified_memory && dim == 0 &&
        op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
        nb0 == sizeof(float) && nb11 == sizeof(float) && nb10 > nb11 && ne00 <= 64 && ne10 >= 32 && ne1 >= 32 &&
        TOSH_ENV("TOSH_CONCAT_TRANSPOSE_DISABLE") == nullptr) {
        ggml_metal_kargs_concat args = {
            /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
            /*.ne10 =*/ ne10, /*.ne11 =*/ ne11, /*.ne12 =*/ ne12, /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10, /*.nb11 =*/ nb11, /*.nb12 =*/ nb12, /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
            /*.dim  =*/ dim,
            /*.nc0  =*/ 1,
        };
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, "kernel_concat_t0_f32");
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_concat_t0_f32", "kernel_concat_t0_f32", nullptr);
        }
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*33*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + 31)/32, (ne1 + 31)/32, ne2*ne3, 32, 8, 1);
        return 1;
    }

    const bool is_q = ggml_is_quantized(op->type);

    // for quantized types, concat is done at the block level (nb0 == type_size == block size)
    int32_t ne00_arg = ne00;
    int32_t ne10_arg = ne10;
    int32_t ne0_arg  = ne0;
    if (is_q) {
        const int32_t blck = ggml_blck_size(op->type);
        GGML_ASSERT(ne00 % blck == 0);
        GGML_ASSERT(ne10 % blck == 0);
        GGML_ASSERT(ne0  % blck == 0);
        ne00_arg = ne00/blck;
        ne10_arg = ne10/blck;
        ne0_arg  = ne0/blck;
    }

    int nth = std::min(256, ne0_arg);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;
    if (nth < 256) {
        nrptg = std::min((256 + nth - 1) / nth, ne1);
        if (nrptg * nth > 256) {
            nrptg = 256 / nth;
        }
    }

    const int nw0 = (ne1 + nrptg - 1) / nrptg;

    // split long rows across threadgroups when there are too few rows to fill the GPU
    const int64_t n_rows = (int64_t) nw0*ne2*ne3;

    int nc0 = 1;
    if (nrptg == 1) {
        nc0 = (int) std::max<int64_t>(1, std::min<int64_t>((ne0_arg + nth - 1)/nth, (GGML_METAL_MIN_THREADGROUPS + n_rows - 1)/n_rows));
    }

    ggml_metal_kargs_concat args = {
        /*.ne00 =*/ ne00_arg,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10_arg,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0_arg,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.dim  =*/ dim,
        /*.nc0  =*/ nc0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_concat(lib, op->type);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*nc0, ne2, ne3, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_repeat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // TODO: optimize for degenerate cases such as ggml_nelements(op->src[0]) == 1 and others

    auto pipeline = ggml_metal_library_get_pipeline_repeat(lib, op->type);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_acc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        // the cpy kernel copies one value per thread, so a long row takes several threadgroups
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);
        const int nw0 = (ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ pnb1,
        /*.nb02 =*/ pnb2,
        /*.nb03 =*/ pnb3,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
        /*.offs =*/ offs,
        /*.o1   =*/ { 0 },
    };

    auto pipeline = ggml_metal_library_get_pipeline_bin_one(lib, GGML_OP_ADD);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    int nth = 1;

    while (2*nth < args.ne0 && nth < nth_max) {
        nth *= 2;
    }

    ggml_metal_encoder_dispatch_threadgroups(enc, ne11, ne12, ne13, nth, 1, 1);

    return 1;
}

// sigmoid -> mul -> add [-> add]: a gated shared expert joining the routed sum and the residual. Three
// small dispatches per MoE layer become one; unified memory keeps the separate ones, unmeasured there.
int ggml_metal_op_tosh_gate_add(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = TOSH_ENV("TOSH_GATE_ADD_FUSION_DISABLE") != nullptr;
    if (disabled || !ctx->use_fusion() || idx + 2 >= ctx->n_nodes() ||
        ggml_metal_device_get_props(ctx->dev)->has_unified_memory) {
        return 0;
    }

    ggml_tensor * sig = ctx->node(idx);
    ggml_tensor * mul = ctx->node(idx + 1);
    ggml_tensor * add = ctx->node(idx + 2);
    if (sig->op != GGML_OP_UNARY || ggml_get_unary_op(sig) != GGML_UNARY_OP_SIGMOID ||
        mul->op != GGML_OP_MUL || add->op != GGML_OP_ADD) {
        return 0;
    }

    ggml_tensor * g = sig->src[0];
    ggml_tensor * x = mul->src[0] == sig ? mul->src[1] : mul->src[1] == sig ? mul->src[0] : nullptr;
    ggml_tensor * y = add->src[0] == mul ? add->src[1] : add->src[1] == mul ? add->src[0] : nullptr;
    if (!x || !y) {
        return 0;
    }
    if (ggml_metal_fb_residual_armed(ctx->dev, add->src[0])) {
        return 0;
    }

    int n_fuse = 3;
    ggml_tensor * dst = add;
    ggml_tensor * r   = nullptr;
    if (idx + 3 < ctx->n_nodes() && ctx->node(idx + 3)->op == GGML_OP_ADD && ctx->has_n_uses(idx + 2, 1) &&
        !ggml_metal_fb_residual_armed(ctx->dev, ctx->node(idx + 3)->src[0])) {
        ggml_tensor * add2 = ctx->node(idx + 3);
        ggml_tensor * res  = add2->src[0] == add ? add2->src[1] : add2->src[1] == add ? add2->src[0] : nullptr;
        if (res && res != add && ggml_are_same_shape(res, add2) && ggml_are_same_shape(add2, add) &&
            res->type == GGML_TYPE_F32 && ggml_is_contiguous(res)) {
            n_fuse = 4;
            dst    = add2;
            r      = res;
        }
    }

    auto f32c = [](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && ggml_is_contiguous(t) && (uintptr_t) t->data % 16 == 0;
    };
    // the inputs the kernel streams may share dst's memory only in place; g is read across rows
    auto apart = [dst](const ggml_tensor * t) { return t->data == dst->data || !ggml_metal_ranges_overlap(t, dst); };

    const bool ok =
        f32c(g) && f32c(sig) && f32c(x) && f32c(mul) && f32c(y) && f32c(add) && f32c(dst) &&
        g->ne[0] == 1 && ggml_are_same_shape(g, sig) &&
        ggml_are_same_shape(x, mul) && ggml_are_same_shape(y, add) && ggml_are_same_shape(mul, add) &&
        g->ne[1] == x->ne[1] && g->ne[2] == x->ne[2] && g->ne[3] == x->ne[3] &&
        x->ne[0] % 4 == 0 &&
        ctx->has_n_uses(idx, 1) && ctx->has_n_uses(idx + 1, 1) &&
        !ggml_metal_ranges_overlap(g, dst) && apart(x) && apart(y) && (!r || apart(r));
    if (!ok) {
        return 0;
    }

    for (int i = 1; i < n_fuse; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);
            break;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline(ctx->lib, "tosh_gate_add_f32");
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_gate_add_f32", "tosh_gate_add_f32", nullptr);
    }
    if (!pipeline.pipeline) {
        return 0;
    }

    struct {
        int32_t ne0_4;
        int32_t n_4;
        int32_t has_res;
    } args = {
        (int32_t) (x->ne[0]/4),
        (int32_t) (ggml_nelements(dst)/4),
        r ? 1 : 0,
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(g),       1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),       2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(y),       3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(r ? r : y), 4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     5);

    const int nth = 256;
    ggml_metal_encoder_dispatch_threadgroups(enc, (args.n_4 + nth - 1)/nth, 1, 1, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_unary(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // the sigmoid and sqrt(softplus) TOPK_MOE fusions anchor at the gating unary node
    if (op->op == GGML_OP_UNARY && ctx->use_fusion() &&
        (ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS || ggml_get_unary_op(op) == GGML_UNARY_OP_SIGMOID)) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            return ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_unary args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.slope =*/ 0.0,
        /*.scale =*/ 0.0,
        /*.bias  =*/ 0.0,
        /*.val   =*/ 0.0,
        /*.min   =*/ 0.0,
        /*.max   =*/ 0.0,
        /*.pre   =*/ 1.0,
        /*.post  =*/ 1.0,
    };

    // SCALE + SILU/SIGMOID [+ SCALE] in one pass
    int n_fuse = 1;
    const ggml_tensor * act = op;
    if (op->op == GGML_OP_SCALE && ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SCALE_UNARY) {
            n_fuse = n;
            ctx->count_fusions(fusion);
            act       = ctx->node(idx + 1);
            args.pre  = ggml_get_op_params_f32(op, 0);
            args.post = n_fuse == 3 ? ggml_get_op_params_f32(ctx->node(idx + 2), 0) : 1.0f;
            bid_dst   = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));
            for (int i = 1; i < n_fuse; ++i) {
                if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                    ggml_metal_op_concurrency_reset(ctx);
                    break;
                }
            }
        }
    }

    if (op->op == GGML_OP_LEAKY_RELU) {
        args.slope = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_SCALE) {
        args.scale = ggml_get_op_params_f32(op, 0);
        args.bias  = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_FILL) {
        args.val = ggml_get_op_params_f32(op, 0);
    }

    if (op->op == GGML_OP_CLAMP) {
        args.min = ggml_get_op_params_f32(op, 0);
        args.max = ggml_get_op_params_f32(op, 1);
    }

    if (op->op == GGML_OP_UNARY && ggml_get_unary_op(op) == GGML_UNARY_OP_XIELU) {
        args.slope = ggml_get_op_params_f32(op, 1); // alpha_n
        args.scale = ggml_get_op_params_f32(op, 2); // alpha_p
        args.bias  = ggml_get_op_params_f32(op, 3); // beta
        args.val   = ggml_get_op_params_f32(op, 4); // eps
    }

    auto pipeline = ggml_metal_library_get_pipeline_unary(lib, act);

    // SILU/SOFTPLUS + MUL fusion: emit one fused kernel for both nodes.
    // The MUL reads the unary output on either side; always bind
    // (unary_input, partner) so the kernel computes f(src0)*src1.
    // (n_fuse is already declared above, for the SCALE + UNARY fusion.)
    if (use_fusion && op->op == GGML_OP_UNARY &&
        (ggml_get_unary_op(op) == GGML_UNARY_OP_SILU ||
         ggml_get_unary_op(op) == GGML_UNARY_OP_SOFTPLUS)) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        const ggml_metal_fusion_id fid = fusion ? ggml_metal_fusion_get_id(fusion) : GGML_METAL_FUSION_NONE;
        if (fid == GGML_METAL_FUSION_SILU_MUL || fid == GGML_METAL_FUSION_SOFTPLUS_MUL) {
            const ggml_tensor * mul = ctx->node(idx + 1);
            // The silu output may sit on either MUL side; always bind
            // (silu_input, partner) so the kernel computes silu(src0)*src1.
            const ggml_tensor * partner =
                (mul->src[0] == op) ? mul->src[1] : mul->src[0];
            ggml_metal_kargs_glu args_f = {
                /*.ne00 =*/ ne00,
                /*.nb01 =*/ nb01,
                /*.ne10 =*/ ne00,
                /*.nb11 =*/ nb01,
                /*.ne0  =*/ ne0,
                /*.nb1  =*/ nb1,
                /*.i00  =*/ 0,
                /*.i10  =*/ 0,
                /*.alpha=*/ 0.0f,
                /*.limit=*/ 0.0f,
            };
            auto pipeline_f = fid == GGML_METAL_FUSION_SOFTPLUS_MUL
                ? ggml_metal_library_get_pipeline_softplus_mul(lib)
                : ggml_metal_library_get_pipeline_silu_mul(lib);
            ggml_metal_encoder_set_pipeline(enc, pipeline_f);
            ggml_metal_encoder_set_bytes   (enc, &args_f, sizeof(args_f), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(partner), 2);
            ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mul),     3);
            const int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_f),
                                                    ne00 / 2));
            const int64_t nrows = (int64_t) ne01 * ne02 * ne03;
            ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);
            n_fuse = 2;
            ctx->count_fusions(fusion);
            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: SILU + MUL\n", __func__);
            }
            if (!ggml_metal_op_concurrency_check(ctx, mul)) {
                ggml_metal_op_concurrency_reset(ctx);
            }
            return n_fuse;
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    if (pipeline.cnt) {
        const int n = pipeline.c4 ? ggml_nelements(op)/4 : ggml_nelements(op);

        const int nth = MIN(256, n);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        const int nth = MIN(args.ne00, nth_max);
        const int nk0 = (args.ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_glu(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    if (op->src[1]) {
        GGML_ASSERT(ggml_are_same_shape(op->src[0], op->src[1]));
    }

    auto pipeline = ggml_metal_library_get_pipeline_glu(lib, op);

    const int32_t swp = ggml_get_op_params_i32(op, 1);
    const float alpha = ggml_get_op_params_f32(op, 2);
    const float limit = ggml_get_op_params_f32(op, 3);

    const int32_t i00 = swp ? ne0 : 0;
    const int32_t i10 = swp ? 0 : ne0;

    ggml_metal_kargs_glu args = {
        /*.ne00 =*/ ne00,
        /*.nb01 =*/ nb01,
        /*.ne10 =*/ op->src[1] ? ne10 : ne00,
        /*.nb11 =*/ op->src[1] ? nb11 : nb01,
        /*.ne0  =*/ ne0,
        /*.nb1  =*/ nb1,
        /*.i00  =*/ op->src[1] ? 0 : i00,
        /*.i10  =*/ op->src[1] ? 0 : i10,
        /*.alpha=*/ alpha,
        /*.limit=*/ limit
    };

    const int64_t nrows = ggml_nrows(op->src[0]);

    // split each row over several threadgroups: with a single row (decode) one threadgroup would run on one GPU core
    const int32_t nth = std::max(1, std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), std::min(256, ne0)));
    const int32_t nk  = (ne0 + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk, nrows, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op  = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const uint64_t n = (uint64_t) ggml_nelements(op->src[0]);

    ggml_metal_kargs_sum args = {
        /*.np =*/ n,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum(lib, op);

    const int nw = ggml_metal_device_get_props(ctx->dev)->simd_width;

    int nth = nw;

    while (nth < (int) n && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) n);

    const int nsg = (nth + nw - 1) / nw;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, nsg * sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_sum_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_sum_rows args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_sum_rows(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < args.ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (int) args.ne00);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cumsum(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline_blk = ggml_metal_library_get_pipeline_cumsum_blk(lib, op);

    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_blk)) {
        nth *= 2;
    }

    GGML_ASSERT(ne00 <= nth*nth);

    const int64_t net0 = (ne00 + nth - 1) / nth;
    const int64_t net1 = ne01;
    const int64_t net2 = ne02;
    const int64_t net3 = ne03;

    const uint64_t nbt0 = sizeof(float);
    const uint64_t nbt1 = net0*nbt0;
    const uint64_t nbt2 = net1*nbt1;
    const uint64_t nbt3 = net2*nbt2;

    // one float per lane of the real simd width (64 on wave64)
    const size_t smem = GGML_PAD(ggml_metal_device_get_props(ctx->dev)->simd_width*sizeof(float), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    {
        ggml_metal_kargs_cumsum_blk args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.net0 =*/ net0,
            /*.net1 =*/ net1,
            /*.net2 =*/ net2,
            /*.net3 =*/ net3,
            /*.nbt0 =*/ nbt0,
            /*.nbt1 =*/ nbt1,
            /*.nbt2 =*/ nbt2,
            /*.nbt3 =*/ nbt3,
            /*.outb =*/ ne00 > nth,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
    }

    if (ne00 > nth) {
        ggml_metal_op_concurrency_reset(ctx);

        {
            ggml_metal_kargs_cumsum_blk args = {
                /*.ne00 =*/ net0,
                /*.ne01 =*/ net1,
                /*.ne02 =*/ net2,
                /*.ne03 =*/ net3,
                /*.nb00 =*/ nbt0,
                /*.nb01 =*/ nbt1,
                /*.nb02 =*/ nbt2,
                /*.nb03 =*/ nbt3,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
                /*.outb =*/ false,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_blk);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, net1, net2, net3, nth, 1, 1);
        }

        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline_add = ggml_metal_library_get_pipeline_cumsum_add(lib, op);

            ggml_metal_kargs_cumsum_add args = {
                /*.ne00 =*/ ne00,
                /*.ne01 =*/ ne01,
                /*.ne02 =*/ ne02,
                /*.ne03 =*/ ne03,
                /*.nb00 =*/ nb00,
                /*.nb01 =*/ nb01,
                /*.nb02 =*/ nb02,
                /*.nb03 =*/ nb03,
                /*.net0 =*/ net0,
                /*.net1 =*/ net1,
                /*.net2 =*/ net2,
                /*.net3 =*/ net3,
                /*.nbt0 =*/ nbt0,
                /*.nbt1 =*/ nbt1,
                /*.nbt2 =*/ nbt2,
                /*.nbt3 =*/ nbt3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline_add);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, net0*ne01, ne02, ne03, nth, 1, 1);
        }
    }

    return 1;
}

int ggml_metal_op_get_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_get_rows(lib, op->src[0]->type);

    ggml_metal_kargs_get_rows args = {
        /*.ne00t =*/ ggml_is_quantized(op->src[0]->type) ? ne00/16 : ne00,
        /*.ne00  =*/ ne00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    const int nth = std::min(args.ne00t, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const int nw0 = (args.ne00t + nth - 1)/nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne10, ne11, ne12, nth, 1, 1);

    return 1;
}

int ggml_metal_op_set_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int simd_width = ggml_metal_device_get_props(ctx->dev)->simd_width;
    const bool turbo_coop = op->type == GGML_TYPE_TURBO3_0 || op->type == GGML_TYPE_TURBO4_0;
    auto pipeline = ggml_metal_library_get_pipeline_set_rows(lib, op, turbo_coop && simd_width == 64);

    const int32_t nk0 = ne0/ggml_blck_size(op->type);

    int nth = 32; // SIMD width

    while (nth < nk0 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    int nrptg = 1;
    if (nth > nk0) {
        nrptg = (nth + nk0 - 1)/nk0;
        nth   = nk0;

        if (nrptg*nth > ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nrptg--;
        }
    }

    nth = std::min(nth, nk0);

    ggml_metal_kargs_set_rows args = {
        /*.nk0  =*/ nk0,
        /*.ne01 =*/ ne01,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // The turbo3/4 store is cooperative: one 32-lane simdgroup per 128-element group,
    // dispatched over a flattened (group, row) grid so decode fills the GPU.
    if (turbo_coop) {
        GGML_ASSERT((simd_width == 32 || simd_width == 64) && "TurboQuant cooperative store requires wave32 or wave64");
        ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne01, ne02, ne03, simd_width, 1, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);
    }

    return 1;
}

int ggml_metal_op_diag(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(int32_t,  ne, op, ne);
    GGML_TENSOR_LOCALS(uint64_t, nb, op, nb);

    ggml_metal_kargs_diag args = {
        /*.ne00 =*/ne00,
        /*.ne01 =*/ne01,
        /*.ne02 =*/ne02,
        /*.ne03 =*/ne03,
        /*.nb00 =*/nb00,
        /*.nb01 =*/nb01,
        /*.nb02 =*/nb02,
        /*.nb03 =*/nb03,
        /*.ne0  =*/ne0,
        /*.ne1  =*/ne1,
        /*.ne2  =*/ne2,
        /*.ne3  =*/ne3,
        /*.nb0  =*/nb0,
        /*.nb1  =*/nb1,
        /*.nb2  =*/nb2,
        /*.nb3  =*/nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_diag(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, 32, 1, 1);

    return 1;
}

int ggml_metal_op_lightning_indexer(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * w = op->src[2];
    const ggml_tensor * m = op->src[3];

    GGML_ASSERT(q->type == GGML_TYPE_F32);
    GGML_ASSERT(k->type == GGML_TYPE_F32  ||
                k->type == GGML_TYPE_F16  ||
                k->type == GGML_TYPE_BF16 ||
                k->type == GGML_TYPE_Q4_0 ||
                k->type == GGML_TYPE_Q4_1 ||
                k->type == GGML_TYPE_Q5_0 ||
                k->type == GGML_TYPE_Q5_1 ||
                k->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(w->type == GGML_TYPE_F32);
    GGML_ASSERT(m->type == GGML_TYPE_F16);
    GGML_ASSERT(op->type == GGML_TYPE_F32);

    GGML_ASSERT(q->ne[0] == OP_LIGHTNING_INDEXER_DK);

    ggml_metal_kargs_lightning_indexer args = {
        /*.n_kv      =*/ (int32_t) k->ne[2],
        /*.n_batch   =*/ (int32_t) q->ne[2],
        /*.mask_ne3  =*/ (int32_t) m->ne[3],
        /*.nb1       =*/ op->nb[1],
        /*.nb3       =*/ op->nb[3],
        /*.nbq1      =*/ q->nb[1],
        /*.nbq2      =*/ q->nb[2],
        /*.nbq3      =*/ q->nb[3],
        /*.nbk2      =*/ k->nb[2],
        /*.nbk3      =*/ k->nb[3],
        /*.nbw1      =*/ w->nb[1],
        /*.nbw3      =*/ w->nb[3],
        /*.nbm1      =*/ m->nb[1],
        /*.nbm3      =*/ m->nb[3],
    };

    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(q),  1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(k),  2);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(w),  3);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(m),  4);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 5);

    // without simdgroup matrices (AMD) a scalar kernel keeps the indexer on the GPU
    if (!ggml_metal_device_get_props(ctx->dev)->has_simdgroup_mm) {
        const int nk = OP_LIGHTNING_INDEXER_AMD_NK;
        const int nb = OP_LIGHTNING_INDEXER_AMD_NB;

        auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer_amd(ctx->lib, op);
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc,
                (k->ne[2] + nk - 1)/nk,
                (q->ne[2] + nb - 1)/nb,
                q->ne[3], nk*nb, 1, 1);

        return 1;
    }

    const int nsg   = OP_LIGHTNING_INDEXER_NSG;
    const int nkptg = OP_LIGHTNING_INDEXER_NKPSG*nsg;
    const int nbptg = OP_LIGHTNING_INDEXER_NBPTG;

    auto pipeline = ggml_metal_library_get_pipeline_lightning_indexer(ctx->lib, op);
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc,
            (k->ne[2] + nkptg - 1)/nkptg,
            (q->ne[2] + nbptg - 1)/nbptg,
            q->ne[3], 32, nsg, 1);

    return 1;
}

int ggml_metal_op_top_k_mask(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mask   = op->src[0];
    const ggml_tensor * top_k  = op->src[1];
    const ggml_tensor * score  = op->src[2];
    const ggml_tensor * blocks = op->src[3];
    const ggml_tensor * extra  = op->src[4];

    GGML_ASSERT(op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);
    GGML_ASSERT(mask->type == op->type);
    GGML_ASSERT(top_k->type == GGML_TYPE_I32);
    GGML_ASSERT(!score  || score->type  == GGML_TYPE_F32);
    GGML_ASSERT(!blocks || blocks->type == GGML_TYPE_I32);
    GGML_ASSERT(!extra  || extra->type  == GGML_TYPE_I32);

    const int32_t n_kv    = mask->ne[0];
    const int32_t n_top   = (int32_t) top_k->ne[0];
    const int32_t n_pick  = blocks ? blocks->ne[1] : score ? score->ne[0] : n_kv;
    const int32_t n_blk   = blocks ? blocks->ne[0] : 0;
    const int32_t n_extra = extra  ? extra->ne[0]  : 0;

    const ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(op);

    // the vector fill stores 4 elements at a time
    const int32_t ts      = ggml_type_size(op->type);
    const int64_t n_elems = ggml_nelements(op);

    const bool has_score  = score  != NULL;
    const bool has_blocks = blocks != NULL;
    const bool has_extra  = extra  != NULL;

    const bool vec = n_elems % 4 == 0 && bid.offs % (4*ts) == 0;

    ggml_metal_kargs_top_k_mask args = {
        /*.n_kv    =*/ n_kv,
        /*.m1      =*/ (int32_t) mask->ne[1],
        /*.m2      =*/ (int32_t) mask->ne[2],
        /*.m3      =*/ (int32_t) mask->ne[3],
        /*.n_top   =*/ n_top,
        /*.n_pick  =*/ n_pick,
        /*.n_blk   =*/ n_blk,
        /*.n_extra =*/ n_extra,
        /*.n_elems =*/ n_elems,
        /*.nb1     =*/ op->nb[1],
        /*.nb2     =*/ op->nb[2],
        /*.nb3     =*/ op->nb[3],
        /*.nbm1    =*/ mask->nb[1],
        /*.nbm2    =*/ mask->nb[2],
        /*.nbm3    =*/ mask->nb[3],
        /*.nbk0    =*/ top_k->nb[0],
        /*.nbk1    =*/ top_k->nb[1],
        /*.nbk2    =*/ top_k->nb[2],
        /*.nbk3    =*/ top_k->nb[3],
        /*.nbs0    =*/ score  ? score->nb[0]  : 0,
        /*.nbs1    =*/ score  ? score->nb[1]  : 0,
        /*.nbs2    =*/ score  ? score->nb[2]  : 0,
        /*.nbs3    =*/ score  ? score->nb[3]  : 0,
        /*.nbb0    =*/ blocks ? blocks->nb[0] : 0,
        /*.nbb1    =*/ blocks ? blocks->nb[1] : 0,
        /*.nbe0    =*/ extra  ? extra->nb[0]  : 0,
        /*.nbe1    =*/ extra  ? extra->nb[1]  : 0,
        /*.nbe2    =*/ extra  ? extra->nb[2]  : 0,
        /*.nbe3    =*/ extra  ? extra->nb[3]  : 0,
    };

    // the fill covers the whole dst, the scatter overwrites the kept cells with the mask values
    {
        auto pipeline_fill = ggml_metal_library_get_pipeline_top_k_mask(lib, op, true, false, false, false, vec);

        const int64_t n_fill  = n_elems/(vec ? 4 : 1);
        const int32_t nth_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_fill);

        int nth = 32; // SIMD width

        while (nth < nth_max && nth < n_fill) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline_fill);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid, 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n_fill + nth - 1)/nth, 1, 1, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    {
        auto pipeline = ggml_metal_library_get_pipeline_top_k_mask(lib, op, false, has_score, has_blocks, has_extra, false);

        const int32_t n_max   = n_top + n_extra;
        const int32_t nth_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

        int nth = 32; // SIMD width

        while (nth < nth_max && nth < n_max) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mask),  1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(top_k), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(score  ? score  : mask), 3);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(blocks ? blocks : mask), 4);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(extra  ? extra  : mask), 5);
        ggml_metal_encoder_set_buffer(enc, bid, 6);

        ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2], op->ne[3], nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_dsv4_hc(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_encoder_t enc = ctx->enc;
    auto pipeline = ggml_metal_library_get_pipeline_dsv4_hc(ctx->lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            {
                const ggml_tensor * mixes = op->src[0];
                const ggml_tensor * scale = op->src[1];
                const ggml_tensor * base  = op->src[2];

                GGML_ASSERT(mixes->type == GGML_TYPE_F32);
                GGML_ASSERT(scale->type == GGML_TYPE_F32);
                GGML_ASSERT(base->type  == GGML_TYPE_F32);
                GGML_ASSERT(op->type    == GGML_TYPE_F32);
                GGML_ASSERT(mixes->ne[0] == 24);
                GGML_ASSERT(op->ne[0] == 4 && op->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_comb args = {
                    /*.n_tokens =*/ (int32_t) mixes->ne[1],
                    /*.n_iter   =*/ ggml_get_op_params_i32(op, 1),
                    /*.nb_m0    =*/ mixes->nb[0],
                    /*.nb_m1    =*/ mixes->nb[1],
                    /*.nb_s0    =*/ scale->nb[0],
                    /*.nb_b0    =*/ base->nb[0],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                    /*.eps      =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(mixes), 1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(scale), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(base),  3);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),    4);

                // One SIMDgroup owns one 4x4 Sinkhorn matrix. Packing up to four
                // independent tokens per threadgroup keeps both decode and prompt
                // dispatches compact without any threadgroup-memory synchronization.
                const int nsg = std::min(4, args.n_tokens);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (args.n_tokens + nsg - 1)/nsg, 1, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_PRE:
            {
                const ggml_tensor * x       = op->src[0];
                const ggml_tensor * weights = op->src[1];

                GGML_ASSERT(x->type       == GGML_TYPE_F32);
                GGML_ASSERT(weights->type == GGML_TYPE_F32);
                GGML_ASSERT(op->type      == GGML_TYPE_F32);

                ggml_metal_kargs_dsv4_hc_pre args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[2],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_x2    =*/ x->nb[2],
                    /*.nb_w0    =*/ weights->nb[0],
                    /*.nb_w1    =*/ weights->nb[1],
                    /*.nb_w2    =*/ weights->nb[2],
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.scale    =*/ ggml_get_op_params_f32(op, 0),
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),       1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(weights), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),      3);

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        case GGML_OP_DSV4_HC_POST:
            {
                const ggml_tensor * x        = op->src[0];
                const ggml_tensor * residual = op->src[1];
                const ggml_tensor * post     = op->src[2];
                const ggml_tensor * comb     = op->src[3];

                GGML_ASSERT(x->type        == GGML_TYPE_F32);
                GGML_ASSERT(residual->type == GGML_TYPE_F32);
                GGML_ASSERT(post->type     == GGML_TYPE_F32);
                GGML_ASSERT(op->type       == GGML_TYPE_F32);
                GGML_ASSERT(residual->ne[1] == 4);

                ggml_metal_kargs_dsv4_hc_post args = {
                    /*.n_embd   =*/ (int32_t) x->ne[0],
                    /*.n_tokens =*/ (int32_t) x->ne[1],
                    /*.nb_x0    =*/ x->nb[0],
                    /*.nb_x1    =*/ x->nb[1],
                    /*.nb_r0    =*/ residual->nb[0],
                    /*.nb_r1    =*/ residual->nb[1],
                    /*.nb_r2    =*/ residual->nb[2],
                    /*.nb_p0    =*/ post->nb[0],
                    /*.nb_p1    =*/ post->nb[1],
                    /*.nb_c0    =*/ comb ? comb->nb[0] : 0,
                    /*.nb_c1    =*/ comb ? comb->nb[1] : 0,
                    /*.nb_c2    =*/ comb ? comb->nb[2] : 0,
                    /*.nb_d0    =*/ op->nb[0],
                    /*.nb_d1    =*/ op->nb[1],
                    /*.nb_d2    =*/ op->nb[2],
                };

                ggml_metal_encoder_set_bytes (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(x),        1);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(residual), 2);
                ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(post),     3);
                if (comb) {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(comb), 4);
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   5);
                } else {
                    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op),   4);
                }

                const int n_tiles = (args.n_embd + 31)/32;
                const int nsg = std::min(4, n_tiles);
                ggml_metal_encoder_dispatch_threadgroups(
                        enc, (n_tiles + nsg - 1)/nsg, args.n_tokens, 1, 32, nsg, 1);
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    return 1;
}

int ggml_metal_op_soft_max(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_TOPK_MOE) {
            return ggml_metal_op_topk_moe(ctx, idx);
        }
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale;
    float max_bias;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    // softmax

    ggml_metal_kargs_soft_max args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne11        =*/ ne11,
        /*.ne12        =*/ ne12,
        /*.ne13        =*/ ne13,
        /*.nb11        =*/ nb11,
        /*.nb12        =*/ nb12,
        /*.nb13        =*/ nb13,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.scale       =*/ scale,
        /*.max_bias    =*/ max_bias,
        /*.m0          =*/ m0,
        /*.m1          =*/ m1,
        /*.n_head_log2 =*/ n_head_log2,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max(lib, op);

    int nth = 32; // SIMD width

    if (ne00%4 == 0) {
        while (nth < ne00/4 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
            nth *= 2;
        }
    }

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    if (op->src[1]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 2);
    }
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_ssm_conv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int n_fuse = 1;
    bool use_silu = false;

    if (ctx->use_fusion()) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SSM_CONV_SILU) {
            n_fuse = n;
            use_silu = true;

            ctx->count_fusions(fusion);
        }
    }

    ggml_metal_kargs_ssm_conv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
    };

    const ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(n_fuse > 1 ? ctx->node(idx + n_fuse - 1) : op);

    // prefill by tiles of 64 channels x 32 tokens, both sides coalesced
    const ggml_metal_device_props * props_ssm = ggml_metal_device_get_props(ctx->dev);
    if (props_ssm->simd_width == 32 && !props_ssm->has_unified_memory && ne1 > 1 && ne10 <= 8 && nb00 == sizeof(float) && nb0 == sizeof(float) && TOSH_ENV("TOSH_SSM_CONV_TILED_DISABLE") == nullptr) {
        char name[128];
        snprintf(name, sizeof(name), "kernel_ssm_conv_f32_f32_tiled_nc=%d_silu=%d", (int) ne10, use_silu ? 1 : 0);
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
        if (!pipeline.pipeline) {
            ggml_metal_cv_t cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int16(cv, 1,              FC_SSM_CONV + 0);
            ggml_metal_cv_set_bool (cv, use_silu,       FC_SSM_CONV + 1);
            ggml_metal_cv_set_int32(cv, (int32_t) ne10, FC_SSM_CONV + 2);
            pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_ssm_conv_f32_f32_tiled", name, cv);
            ggml_metal_cv_free(cv);
        }
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, GGML_PAD(64*(32 + ne10 - 1)*sizeof(float), 16), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + 63)/64, (ne1 + 31)/32, ne02, 256, 1, 1);
        return n_fuse;
    }

    // Use batched kernel for prefill (ne1 > 1) to reduce threadgroup dispatch overhead
    const bool use_batched = (ne1 > 1);

    if (use_batched) {
        // Determine the smallest power of 2 that's >= ne1, but <= 256
        int BATCH_SIZE;
        if      (ne1 > 128) BATCH_SIZE = 256;
        else if (ne1 > 64 ) BATCH_SIZE = 128;
        else if (ne1 > 32 ) BATCH_SIZE = 64;
        else if (ne1 > 16 ) BATCH_SIZE = 32;
        else if (ne1 > 8  ) BATCH_SIZE = 16;
        else if (ne1 > 4  ) BATCH_SIZE = 8;
        else                BATCH_SIZE = 2;

        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_batched(lib, op, BATCH_SIZE, (int32_t) ne10, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        // Dispatch: ne01 rows, ceil(ne1/BATCH_SIZE) token batches, ne02 sequences
        // Each threadgroup has BATCH_SIZE threads, each handling one token
        const int n_token_batches = (ne1 + BATCH_SIZE - 1) / BATCH_SIZE;
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, n_token_batches, ne02, BATCH_SIZE, 1, 1);
    } else {
        // decode packs rows across a threadgroup; token/seq stay grid dims
        auto pipeline = ggml_metal_library_get_pipeline_ssm_conv_decode(lib, op, use_silu);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst, 3);

        const int nth  = 64;
        const int ntgx = (ne01 + nth - 1) / nth;
        ggml_metal_encoder_dispatch_threadgroups(enc, ntgx, ne1, ne02, nth, 1, 1);
    }

    if (n_fuse > 1 && ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: SSM_CONV + UNARY\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_ssm_scan(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne4, op->src[4], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb4, op->src[4], nb);
    GGML_TENSOR_LOCALS( int32_t, ne5, op->src[5], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb5, op->src[5], nb);
    GGML_TENSOR_LOCALS( int32_t, ne6, op->src[6], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb6, op->src[6], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const ggml_tensor * src3 = op->src[3];
    const ggml_tensor * src4 = op->src[4];
    const ggml_tensor * src5 = op->src[5];
    const ggml_tensor * src6 = op->src[6];

    GGML_ASSERT(src3);
    GGML_ASSERT(src4);
    GGML_ASSERT(src5);
    GGML_ASSERT(src6);

    const int64_t d_state      = ne00;
    const int64_t d_inner      = ne01;
    const int64_t n_head       = ne02;
    const int64_t n_group      = ne41;
    const int64_t n_seq_tokens = ne12;
    const int64_t n_seqs       = ne13;
    const int64_t K            = ggml_get_op_params_i32(op, 0);

    GGML_ASSERT(K >= 1);
    GGML_ASSERT(ggml_nelements(op->src[1]) + K*d_state*d_inner*n_head*n_seqs == ggml_nelements(op));

    ggml_metal_kargs_ssm_scan args = {
        /*.d_state      =*/ d_state,
        /*.d_inner      =*/ d_inner,
        /*.n_head       =*/ n_head,
        /*.n_group      =*/ n_group,
        /*.n_seq_tokens =*/ n_seq_tokens,
        /*.n_seq_tokens_total =*/ n_seq_tokens,
        /*.token_offset =*/ 0,
        /*.n_seqs       =*/ n_seqs,
        /*.K            =*/ K,
        /*.s_off        =*/ ggml_nelements(op->src[1]) * sizeof(float),
        /*.nb00         =*/ nb00,
        /*.nb01         =*/ nb01,
        /*.nb02         =*/ nb02,
        /*.nb03         =*/ nb03,
        /*.nb10         =*/ nb10,
        /*.nb11         =*/ nb11,
        /*.nb12         =*/ nb12,
        /*.ns12         =*/ nb12/nb10,
        /*.nb13         =*/ nb13,
        /*.nb20         =*/ nb20,
        /*.nb21         =*/ nb21,
        /*.ns21         =*/ nb21/nb20,
        /*.nb22         =*/ nb22,
        /*.ne30         =*/ ne30,
        /*.nb31         =*/ nb31,
        /*.nb41         =*/ nb41,
        /*.nb42         =*/ nb42,
        /*.ns42         =*/ nb42/nb40,
        /*.nb43         =*/ nb43,
        /*.nb51         =*/ nb51,
        /*.nb52         =*/ nb52,
        /*.ns52         =*/ nb52/nb50,
        /*.nb53         =*/ nb53,
        /*.nb0          =*/ nb0,
    };

    constexpr int64_t CHUNK = OP_SSM_SCAN_SSD_CS;

    const int64_t snap_reserve = K > 1 ? K : 0; // tokens reserved for sequential kernel rollback snapshots
    const int64_t mma_tokens = ((n_seq_tokens - snap_reserve) / CHUNK) * CHUNK; // largest multiple of CHUNK that leaves snap_reserve for the tail
    const bool use_mma =
        mma_tokens > 0 &&
        ne30 == 1 && // checks that A tensor is set to scalar decay per head (A shape {1, n_head})
        props_dev->has_simdgroup_mm && // hardware check for M1 or newer
        d_state % 8 == 0 && // d_state must be multiple of 8 to align with simdgroup_float 8x8 tiles
        d_inner == OP_SSM_SCAN_SSD_HD; // mma kernel is specialized for the Mamba-2 head dim; this checks it

    const auto dispatch = [&](ggml_metal_pipeline_with_params pipeline, int64_t nth, int64_t n_tg_x) {
        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(pipeline.smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), 4);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), 5);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), 6);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), 7);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         8);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, n_tg_x, n_head, n_seqs, nth, 1, 1);
    };

    if (!use_mma) {
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, false), d_state, d_inner);
        return 1;
    }

    args.n_seq_tokens = mma_tokens;
    dispatch(
        ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(lib, op),
        OP_SSM_SCAN_SSD_NSG*32,
        1);

    if (mma_tokens < n_seq_tokens) {
        ggml_metal_op_concurrency_reset(ctx);

        args.n_seq_tokens = n_seq_tokens - mma_tokens;
        args.token_offset = mma_tokens;
        dispatch(ggml_metal_library_get_pipeline_ssm_scan(lib, op, true), d_state, d_inner);
    }

    return 1;
}

int ggml_metal_op_rwkv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t B = op->op == GGML_OP_RWKV_WKV6 ? op->src[5]->ne[1] : op->src[6]->ne[1];
    const int64_t T = op->src[0]->ne[2];
    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    auto pipeline = ggml_metal_library_get_pipeline_rwkv(lib, op);

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[5]), ida++);
    if (op->op == GGML_OP_RWKV_WKV7) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[6]), ida++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &B, sizeof(B), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &T, sizeof(T), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &C, sizeof(C), ida++);
    ggml_metal_encoder_set_bytes   (enc, (void *) &H, sizeof(H), ida++);

    ggml_metal_encoder_dispatch_threadgroups(enc, B * H, 1, 1, C/H, 1, 1);

    return 1;
}

static bool ggml_metal_gdn_single_use(const ggml_cgraph * gf, const ggml_tensor * t) {
    return t && ggml_metal_use_count(gf, t) == 1 && !(t->flags & GGML_TENSOR_FLAG_OUTPUT) &&
           t->type == GGML_TYPE_F32 && ggml_is_contiguous(t);
}

// the l2 norm of q or k, scale(rms_norm(x, eps/n), 1/sqrt(n)), done by the kernel
struct ggml_metal_gdn_l2 {
    const ggml_tensor * scale = nullptr, * norm = nullptr, * x = nullptr;
    float eps = 0.0f;
};

static bool ggml_metal_gdn_l2_src(const ggml_cgraph * gf, const ggml_tensor * t, ggml_metal_gdn_l2 & r) {
    // a prompt keeps its separate norm: inside the kernel it would sit on the serial token loop
    if (t->ne[2] != 1) {
        return false;
    }
    r.scale = ggml_metal_unwrap_reshape(gf, t);
    if (!r.scale || r.scale->op != GGML_OP_SCALE || !ggml_metal_gdn_single_use(gf, r.scale) ||
        ggml_get_op_params_f32(r.scale, 1) != 0.0f) {
        return false;
    }
    r.norm = r.scale->src[0];
    if (r.norm->op != GGML_OP_RMS_NORM || !ggml_metal_gdn_single_use(gf, r.norm)) {
        return false;
    }
    r.x = r.norm->src[0];
    const float n = (float) r.x->ne[0];
    if (r.x->type != GGML_TYPE_F32 || r.x->nb[0] != sizeof(float) || !ggml_are_same_shape(r.x, t) ||
        fabsf(ggml_get_op_params_f32(r.scale, 0) - 1.0f/sqrtf(n)) > 1e-7f) {
        return false;
    }
    r.eps = ggml_get_op_params_f32(r.norm, 0)*n;
    return true;
}

// A single-sequence gated_delta_net can read its state from the recurrent cache row and write
// the new state back into the cache, so the gather before it and the copy after it are skipped.
// The three nodes decide with the same predicates, so a skipped node is always covered.
static const ggml_tensor * ggml_metal_gdn_rows_src(const ggml_cgraph * gf, const ggml_tensor * gdn) {
    const ggml_tensor * st = gdn->src[5];
    if (st->ne[3] != 1 || ggml_get_op_params_i32(gdn, 0) != 1) {
        return nullptr;
    }
    const ggml_tensor * gr = st;
    while (gr && gr->op == GGML_OP_RESHAPE) {
        if (ggml_metal_use_count(gf, gr) != 1) {
            return nullptr;
        }
        gr = gr->src[0];
    }
    if (!gr || gr->op != GGML_OP_GET_ROWS || ggml_metal_use_count(gf, gr) != 1 ||
        (gr->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        gr->type != GGML_TYPE_F32 || gr->src[0]->type != GGML_TYPE_F32 || gr->src[1]->type != GGML_TYPE_I32 ||
        gr->src[0]->nb[0] != sizeof(float) || gr->src[0]->ne[0] != ggml_nelements(st) ||
        gr->ne[1] != 1 || gr->src[1]->ne[0] != 1) {
        return nullptr;
    }
    return gr;
}

static const ggml_tensor * ggml_metal_gdn_fold_dst(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * gdn = ctx->node(idx);
    if (gdn->src[5]->ne[3] != 1 || ggml_get_op_params_i32(gdn, 0) != 1) {
        return nullptr;
    }
    const size_t attn_size = (size_t) ggml_nelements(gdn->src[2])*sizeof(float);
    const int64_t n_state  = ggml_nelements(gdn->src[5]);
    for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 16); ++i) {
        const ggml_tensor * c = ctx->node(i);
        if (c->op != GGML_OP_CPY) {
            continue;
        }
        const ggml_tensor * a = c->src[0];
        if (a->view_src == gdn && a->view_offs == attn_size && a->type == GGML_TYPE_F32 &&
            ggml_nelements(a) == n_state && ggml_is_contiguous(a) &&
            c->type == GGML_TYPE_F32 && ggml_is_contiguous(c) &&
            ggml_metal_use_count(ctx->graph(), a) == 1) {
            return c;
        }
    }
    return nullptr;
}

static bool ggml_metal_gdn_state_fusion_on(ggml_metal_op_t ctx) {
    static const bool disabled = TOSH_ENV("TOSH_GDN_STATE_FUSION_DISABLE") != nullptr;
    return ctx->use_fusion() && !ctx->use_concurrency && !disabled;
}

// true when node idx is a gather or a copy that a nearby gated_delta_net takes over
static bool ggml_metal_gdn_state_skip(ggml_metal_op_t ctx, int idx) {
    const ggml_tensor * node = ctx->node(idx);
    if (node->op == GGML_OP_GET_ROWS) {
        for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 32); ++i) {
            const ggml_tensor * g = ctx->node(i);
            if (g->op == GGML_OP_GATED_DELTA_NET && ggml_metal_gdn_rows_src(ctx->graph(), g) == node) {
                return true;
            }
        }
    } else if (node->op == GGML_OP_CPY) {
        for (int i = idx - 1; i >= std::max(0, idx - 16); --i) {
            if (ctx->node(i)->op == GGML_OP_GATED_DELTA_NET) {
                return ggml_metal_gdn_fold_dst(ctx, i) == node;
            }
        }
    } else if (node->op == GGML_OP_RMS_NORM || node->op == GGML_OP_SCALE) {
        for (int i = idx + 1; i < std::min(ctx->n_nodes(), idx + 40); ++i) {
            const ggml_tensor * g = ctx->node(i);
            if (g->op != GGML_OP_GATED_DELTA_NET) {
                continue;
            }
            ggml_metal_gdn_l2 lq, lk;
            if (((ggml_metal_gdn_l2_src(ctx->graph(), g->src[0], lq) && (node == lq.scale || node == lq.norm)) ||
                (ggml_metal_gdn_l2_src(ctx->graph(), g->src[1], lk) && (node == lk.scale || node == lk.norm)))) {
                return true;
            }
            return false;
        }
    }
    return false;
}

// out_prod(a, b) = a b^T, which the backward of a matrix product takes with b the transposed gradient: the GEMM
// reads a transposed and the gradient as its usual K-contiguous columns, so neither is copied
int ggml_metal_op_out_prod(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * a = op->src[0];
    const ggml_tensor * b = op->src[1];

    // one side of at most 32 rows (the gradient of a LoRA factor): outer products over k, nothing transposed
    if (a->type == GGML_TYPE_F32 && b->type == GGML_TYPE_F32 && b->nb[0] == sizeof(float) &&
        (a->ne[0] <= 32 || b->ne[0] <= 32)) {
        const bool    small_a = a->ne[0] <= b->ne[0];
        const int64_t ns      = small_a ? a->ne[0] : b->ne[0];
        const int64_t nl      = small_a ? b->ne[0] : a->ne[0];
        const int     smax    = ns <= 16 ? 16 : 32;

        char name[64];
        snprintf(name, sizeof(name), "kernel_out_prod_small_f32_%d_%s", smax, small_a ? "a" : "b");
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
        }

        ggml_metal_kargs_mul_mm sargs = {
            /*.ne00 =*/ (int32_t) a->ne[1],
            /*.ne02 =*/ 1,
            /*.nb01 =*/ a->nb[1],
            /*.nb02 =*/ a->nb[2],
            /*.nb03 =*/ a->nb[3],
            /*.ne12 =*/ 1,
            /*.nb10 =*/ b->nb[0],
            /*.nb11 =*/ b->nb[1],
            /*.nb12 =*/ b->nb[2],
            /*.nb13 =*/ b->nb[3],
            /*.ne0  =*/ (int32_t) op->ne[0],
            /*.ne1  =*/ (int32_t) op->ne[1],
            /*.r2   =*/ 1,
            /*.r3   =*/ 1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &sargs, sizeof(sargs), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),  1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(b),  2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 4*64*smax*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (nl + 63)/64, 1, 1, 256, 1, 1);

        return 1;
    }

    ggml_metal_kargs_mul_mm args = {
        /*.ne00 =*/ (int32_t) a->ne[1],
        /*.ne02 =*/ (int32_t) a->ne[2],
        /*.nb01 =*/ a->nb[1],
        /*.nb02 =*/ a->nb[2],
        /*.nb03 =*/ a->nb[3],
        /*.ne12 =*/ (int32_t) op->ne[2],
        /*.nb10 =*/ b->nb[1],
        /*.nb11 =*/ b->nb[0],
        /*.nb12 =*/ b->nb[2],
        /*.nb13 =*/ b->nb[3],
        /*.ne0  =*/ (int32_t) op->ne[0],
        /*.ne1  =*/ (int32_t) op->ne[1],
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    // a quantized or f16 weight takes the half tile, with each gradient column scaled into it first
    const bool cs = (a->type == GGML_TYPE_Q8_0 || a->type == GGML_TYPE_F16) && TOSH_ENV("TOSH_OUT_PROD_CS_DISABLE") == nullptr;
    if (cs) {
        ggml_metal_kargs_mul_mm sargs = args;
        sargs.ne02 = (int32_t) (op->ne[2]*op->ne[3]);

        ggml_metal_pipeline_with_params ps = ggml_metal_library_get_pipeline(lib, "kernel_out_prod_colscale");
        if (!ps.pipeline) {
            ps = ggml_metal_library_compile_pipeline(lib, "kernel_out_prod_colscale", "kernel_out_prod_colscale", nullptr);
        }
        ggml_metal_encoder_set_pipeline(enc, ps);
        ggml_metal_encoder_set_bytes   (enc, &sargs, sizeof(sargs), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(b),  1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 32, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, op->ne[1], op->ne[2]*op->ne[3], 1, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mm_ta(lib, op, cs);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(b),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (op->ne[1] + pipeline.nr1 - 1)/pipeline.nr1,
            (op->ne[0] + pipeline.nr0 - 1)/pipeline.nr0, op->ne[2]*op->ne[3], 32, pipeline.nsg, 1);

    return 1;
}

// the scratch of the chunked backward, after dst: gc and two halves of d gc per token, the inverse of each chunk,
// w, u and du per token, and the state and its gradient at each chunk
size_t ggml_metal_op_out_prod_extra(const ggml_tensor * op) {
    assert(op->op == GGML_OP_OUT_PROD);

    return op->ne[1]*op->ne[2]*op->ne[3]*sizeof(float);
}

size_t ggml_metal_op_gated_delta_net_back_extra(const ggml_tensor * op) {
    assert(op->op == GGML_OP_GATED_DELTA_NET_BACK);

    const int64_t S  = op->src[2]->ne[0];
    const int64_t BH = op->src[2]->ne[1]*op->src[2]->ne[3];
    const int64_t NT = (op->src[2]->ne[2] + 63)/64;
    const int64_t TP = NT*64;

    const int64_t n = BH*TP*3 + BH*NT*64*64 + 3*BH*TP*S + 2*BH*NT*S*S;

    return GGML_PAD(ggml_nbytes(op), 256) - ggml_nbytes(op) + n*sizeof(float);
}

int ggml_metal_op_gated_delta_net_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * q = op->src[0];
    const ggml_tensor * k = op->src[1];
    const ggml_tensor * v = op->src[2];
    const ggml_tensor * g = op->src[3];
    const ggml_tensor * b = op->src[4];
    const ggml_tensor * s = op->src[5];
    const ggml_tensor * d = op->src[6];

    const int64_t S = v->ne[0];
    const int32_t H = v->ne[1];
    const int32_t T = v->ne[2];
    const int32_t B = v->ne[3];
    const int32_t NT = (T + 63)/64;

    ggml_metal_kargs_gated_delta_net_back args = {
        /*.H       =*/ H,
        /*.Hq      =*/ (int32_t) q->ne[1],
        /*.T       =*/ T,
        /*.B       =*/ B,
        /*.NT      =*/ NT,
        /*.has_dht =*/ ggml_nelements(d) == S*H*T*B + S*S*H*B,
        /*.nb01    =*/ q->nb[1], q->nb[2], q->nb[3],
        /*.nb11    =*/ k->nb[1], k->nb[2], k->nb[3],
        /*.nb21    =*/ v->nb[1], v->nb[2], v->nb[3],
        /*.nb31    =*/ g->nb[1], g->nb[2], g->nb[3],
        /*.nb41    =*/ b->nb[1], b->nb[2], b->nb[3],
        /*.nb52    =*/ s->nb[2], s->nb[3],
        /*.offs    =*/ GGML_PAD(ggml_nbytes(op), 256),
        /*.scale   =*/ 1.0f/sqrtf((float) S),
    };

    struct stage {
        const char * name;
        int nx, ny, nz;
        size_t smem;
        const ggml_tensor * bufs[7];
    };

    const stage stages[] = {
        { "prep",  NT, H,  B,                    50704, { q, k, v, g, b, d, op } },
        { "fwd_h", 4,  H,  B,                    34048, { k, s, op } },
        { "dhu",   4,  H,  B,                    42496, { q, k, d, op } },
        { "dqkwg", 2,  NT, (int) q->ne[1]*B,     50208, { q, k, d, op } },
        { "wy",    NT, (int) q->ne[1]*B, 1,      51216, { k, v, b, op } },
    };

    for (const stage & st : stages) {
        auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net_back(lib, st.name);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        for (int i = 0; i < 7 && st.bufs[i]; ++i) {
            ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(st.bufs[i]), i + 1);
        }
        ggml_metal_encoder_set_threadgroup_memory_size(enc, st.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, st.nx, st.ny, st.nz, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    return 1;
}

int ggml_metal_op_gated_delta_net(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();
    const int  debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    const bool fusion = ggml_metal_gdn_state_fusion_on(ctx);
    const ggml_tensor * rows_src = fusion ? ggml_metal_gdn_rows_src(ctx->graph(), op) : nullptr;
    const ggml_tensor * fold_dst = fusion ? ggml_metal_gdn_fold_dst(ctx, idx) : nullptr;

    ggml_metal_gdn_l2 l2q, l2k;
    const bool use_raw = ggml_get_op_params_i32(op, 1) != 0;
    const bool use_l2q = fusion && ggml_metal_gdn_l2_src(ctx->graph(), op->src[0], l2q);
    const bool use_l2k = fusion && ggml_metal_gdn_l2_src(ctx->graph(), op->src[1], l2k);

    auto pipeline = ggml_metal_library_get_pipeline_gated_delta_net(lib, op, rows_src != nullptr, fold_dst != nullptr,
            use_raw, use_l2q, use_l2k);

    const ggml_tensor * t_q = use_l2q ? l2q.x : op->src[0];
    const ggml_tensor * t_k = use_l2k ? l2k.x : op->src[1];

    GGML_TENSOR_LOCALS( int32_t, ne0, t_q,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, t_q,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, t_k,        ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, t_k,        nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    int ida = 0;

    ggml_metal_kargs_gated_delta_net args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne20 =*/ ne20,
        /*.ne21 =*/ ne21,
        /*.ne22 =*/ ne22,
        /*.ne23 =*/ ne23,
        /*.nb20 =*/ nb20,
        /*.nb21 =*/ nb21,
        /*.nb22 =*/ nb22,
        /*.nb23 =*/ nb23,
        /*.ns02 =*/ (int32_t) (nb02/sizeof(float)),
        /*.ns12 =*/ (int32_t) (nb12/sizeof(float)),
        /*.ns22 =*/ (int32_t) (nb22/sizeof(float)),
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s_nb1  =*/ rows_src ? rows_src->src[0]->nb[1] : 0,
        /*.sd_nb1 =*/ fold_dst ? fold_dst->nb[1] : 0,
        /*.eps_q  =*/ use_l2q ? l2q.eps : 0.0f,
        /*.eps_k  =*/ use_l2k ? l2k.eps : 0.0f,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args),                  ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(t_q),        ida++); // q
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(t_k),        ida++); // k
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++); // v
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++); // gate
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++); // beta
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rows_src ? rows_src->src[0] : op->src[5]), ida++); // state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         ida++); // dst
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(rows_src ? rows_src->src[1] : op), ida++); // state rows
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(fold_dst ? fold_dst : op),          ida++); // new state
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(use_raw ? op->src[7] : op), ida++); // dt bias
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(use_raw ? op->src[8] : op), ida++); // a

    const int nsg = pipeline.nsg;

    if (nsg < 0) {
        // one flat group of 128 threads covers 128/lpr rows
        const int lpr = -nsg;
        ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/(128/lpr), op->src[2]->ne[1], op->src[2]->ne[3], 128, 1, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, op->src[2]->ne[0]/nsg, op->src[2]->ne[1], op->src[2]->ne[3], 32, nsg, 1);
    }

    return 1;
}

int ggml_metal_op_turbo_wht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int direction;
    memcpy(&direction, op->op_params, sizeof(int));

    const int64_t n_elements = ggml_nelements(op->src[0]);
    const int64_t n_groups = n_elements / 128;

    auto pipeline = ggml_metal_library_get_pipeline_turbo_wht(lib);

    ggml_metal_kargs_turbo_wht args = {
        /*.n_elements =*/ n_elements,
        /*.direction  =*/ direction,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int threads_per_tg = 256;
    const int n_threadgroups = (n_groups + threads_per_tg - 1) / threads_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_threadgroups, 1, 1, threads_per_tg, 1, 1);

    return 1;
}

int ggml_metal_op_solve_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_solve_tri args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_solve_tri(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    const int nsg = pipeline.nsg;
    const int nw  = ggml_metal_device_get_props(ctx->dev)->simd_width;

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne10 + nsg - 1)/nsg, ne02, ne03, nw, nsg, 1);

    return 1;
}

int ggml_metal_op_set(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const size_t pnb1 = ((const int32_t *) op->op_params)[0];
    const size_t pnb2 = ((const int32_t *) op->op_params)[1];
    const size_t pnb3 = ((const int32_t *) op->op_params)[2];
    const size_t offs = ((const int32_t *) op->op_params)[3];

    const bool inplace = (bool) ((const int32_t *) op->op_params)[4];

    if (!inplace) {
        // run a separate kernel to cpy src->dst
        // not sure how to avoid this
        // TODO: make a simpler cpy_bytes kernel

        //const id<MTLComputePipelineState> pipeline = ctx->pipelines[GGML_METAL_PIPELINE_TYPE_CPY_F32_F32].obj;
        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.ne2  =*/ ne2,
            /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,
            /*.nb1  =*/ nb1,
            /*.nb2  =*/ nb2,
            /*.nb3  =*/ nb3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

        // the cpy kernel copies one value per thread, so a long row takes several threadgroups
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);
        const int nw0 = (ne00 + nth - 1)/nth;

        ggml_metal_encoder_dispatch_threadgroups(enc, nw0*ne01, ne02, ne03, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);
    }

    auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[1]->type, op->type, false);

    GGML_ASSERT(ne10 % ggml_blck_size(op->src[1]->type) == 0);

    int64_t nk0 = ne10;
    if (ggml_is_quantized(op->src[1]->type)) {
        nk0 = ne10/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne10/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne11, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[1]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne10,
        /*.ne01 =*/ ne11,
        /*.ne02 =*/ ne12,
        /*.ne03 =*/ ne13,
        /*.nb00 =*/ nb10,
        /*.nb01 =*/ nb11,
        /*.nb02 =*/ nb12,
        /*.nb03 =*/ nb13,
        /*.ne0  =*/ ne10,
        /*.ne1  =*/ ne11,
        /*.ne2  =*/ ne12,
        /*.ne3  =*/ ne13,
        /*.nb0  =*/ ggml_element_size(op),
        /*.nb1  =*/ pnb1,
        /*.nb2  =*/ pnb2,
        /*.nb3  =*/ pnb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    bid_dst.offs += offs;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne11 + nrptg - 1)/nrptg, ne12, ne13, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_cpy(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool contig = ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1 &&
                        ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op) &&
                        TOSH_ENV("TOSH_CPY_CONTIG_DISABLE") == nullptr;

    if (contig) {
        const int64_t n = ggml_nelements(op);

        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ n,
            /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
        };

        auto pipeline = ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, true);

        const int nth = std::min<int>(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (n + nth - 1)/nth, 1, 1, nth, 1, 1);

        return 1;
    }

    // a transposed f32 view: tiles through threadgroup memory so both sides coalesce
    const ggml_metal_device_props * props_cpy = ggml_metal_device_get_props(ctx->dev);
    if (props_cpy->simd_width == 32 && !props_cpy->has_unified_memory &&
        op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous(op) &&
        nb01 == sizeof(float) && nb00 > nb01 && ne00 >= 32 && ne01 >= 32 &&
        TOSH_ENV("TOSH_CPY_TRANSPOSE_DISABLE") == nullptr) {
        ggml_metal_kargs_cpy args = {
            /*.nk0  =*/ ne00,
            /*.ne00 =*/ ne00, /*.ne01 =*/ ne01, /*.ne02 =*/ ne02, /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00, /*.nb01 =*/ nb01, /*.nb02 =*/ nb02, /*.nb03 =*/ nb03,
            /*.ne0  =*/ ne0,  /*.ne1  =*/ ne1,  /*.ne2  =*/ ne2,  /*.ne3  =*/ ne3,
            /*.nb0  =*/ nb0,  /*.nb1  =*/ nb1,  /*.nb2  =*/ nb2,  /*.nb3  =*/ nb3,
        };
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, "kernel_cpy_transpose_f32");
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, "kernel_cpy_transpose_f32", "kernel_cpy_transpose_f32", nullptr);
        }
        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*33*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (ne00 + 31)/32, (ne01 + 31)/32, ne02*ne03, 32, 8, 1);
        return 1;
    }

    const auto f32_f16 = [](ggml_type t) { return t == GGML_TYPE_F32 || t == GGML_TYPE_F16; };
    const bool flat = props_cpy->simd_width == 32 && !props_cpy->has_unified_memory &&
                      f32_f16(op->src[0]->type) && f32_f16(op->type) && ggml_is_contiguous(op) &&
                      (ne00 != ne0 || ne01 != ne1 || ne02 != ne2) && TOSH_ENV("TOSH_CPY_FLAT_DISABLE") == nullptr;

    auto pipeline = flat ? ggml_metal_library_get_pipeline_cpy_flat(lib, op->src[0]->type, op->type)
                         : ggml_metal_library_get_pipeline_cpy(lib, op->src[0]->type, op->type, false);

    GGML_ASSERT(ne00 % ggml_blck_size(op->src[0]->type) == 0);

    const auto is_turbo = [](ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };

    int64_t nk0 = ne00;
    if (is_turbo(op->src[0]->type)) {
        // turbo dequant does the inverse WHT over the whole 128-element group per thread
        nk0 = ne00/ggml_blck_size(op->src[0]->type);
    } else if (ggml_is_quantized(op->src[0]->type)) {
        nk0 = ne00/16;
    } else if (ggml_is_quantized(op->type)) {
        nk0 = ne00/ggml_blck_size(op->type);
    }

    int nth = std::min<int>(nk0*ne01, 256);

    // when rows are small, we can batch them together in a single threadgroup
    int nrptg = 1;

    // TODO: relax this constraint in the future
    if (ggml_blck_size(op->src[0]->type) == 1 && ggml_blck_size(op->type) == 1) {
        if (nth > nk0) {
            nrptg = (nth + nk0 - 1)/nk0;
            nth   = nk0;

            if (nrptg*nth > 256) {
                nrptg--;
            }
        }
    }

    nth = std::min<int>(nth, nk0);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ nk0,
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    const int nw0 = nrptg == 1 ? (nk0 + nth - 1)/nth : 1;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nw0*(ne01 + nrptg - 1)/nrptg, ne02, ne03, nth, nrptg, 1);

    return 1;
}

int ggml_metal_op_pool_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t s0 = opts[2];
    const int32_t p0 = opts[3];

    const int64_t IW = op->src[0]->ne[0];
    const int64_t OW = op->ne[0];

    const int64_t np = ggml_nelements(op);

    ggml_metal_kargs_pool_1d args_pool_1d = {
        /* .k0 = */  k0,
        /* .s0 = */  s0,
        /* .p0 = */  p0,
        /* .IW = */  IW,
        /* .OW = */  OW,
        /* .np = */  np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_1d(lib, op, op_pool);

    const int nth = std::min(256, std::max(1, k0));
    const int ntg = (int) np;

    const size_t smem = op_pool == GGML_OP_POOL_AVG ?
        2*32*sizeof(float) :
        32*sizeof(float);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_1d, sizeof(args_pool_1d),  0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_fwht(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_tensor * src1 = op->src[1];

    const int64_t n = src1->ne[0];
    const int64_t nrows = ggml_nrows(src1);

    ggml_metal_kargs_fwht args = {
        /*.nrows = */ (int32_t) nrows,
        /*.n_blk = */ 0,
    };

    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    auto pipeline = ggml_metal_library_get_pipeline_fwht(lib, n, src1->type, simd_size == 64);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 1);
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 2);
    // buffer 3 is never read when n_blk == 0; bind src so the slot is always valid
    ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(src1), 3);

    const int th_max = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);

    if (n >= GGML_METAL_FWHT_TG_MIN_N) {
        GGML_ASSERT(th_max >= GGML_METAL_FWHT_TG_NT);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, GGML_METAL_FWHT_TG_NT, 1, 1);

        return 1;
    }

    int sg_per_tg = 2;
    sg_per_tg = std::min(sg_per_tg, th_max/simd_size);
    sg_per_tg = std::max(sg_per_tg, 1);

    const int64_t n_tg = (nrows + sg_per_tg - 1) / sg_per_tg;
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tg, 1, 1, simd_size*sg_per_tg, 1, 1);

    return 1;
}

static int32_t ggml_metal_use_count(const ggml_cgraph * gf, const ggml_tensor * t) {
    const size_t pos = ggml_hash_find(&gf->visited_hash_set, t);
    if (!ggml_bitset_get(gf->visited_hash_set.used, pos)) {
        return 0;
    }
    return gf->use_counts[pos];
}

// the tensor under a chain of single-use reshapes, or nullptr if any of them is read elsewhere
static const ggml_tensor * ggml_metal_unwrap_reshape(const ggml_cgraph * gf, const ggml_tensor * t) {
    while (t && t->op == GGML_OP_RESHAPE) {
        if (ggml_metal_use_count(gf, t) != 1) {
            return nullptr;
        }
        t = t->src[0];
    }
    return t;
}

// rms_norm -> mul by the norm weight -> the Hadamard transform of a folded weight's input, all
// adjacent: one kernel writes the transform, and the norm too when other nodes read it
static int ggml_metal_op_rms_norm_fwht(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = TOSH_ENV("TOSH_NORM_FWHT_FUSION_DISABLE") != nullptr;
    if (disabled || idx + 3 >= ctx->n_nodes()) {
        return 0;
    }
    const ggml_cgraph * gf = ctx->graph();

    // an add before the norm (the residual) is taken in as well
    const ggml_tensor * add = ctx->node(idx)->op == GGML_OP_ADD ? ctx->node(idx) : nullptr;
    const int i0 = add ? idx + 1 : idx;
    if (i0 + 3 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * nrm = ctx->node(i0);
    const ggml_tensor * mw  = ctx->node(i0 + 1);
    const ggml_tensor * ms  = ctx->node(i0 + 2);
    const ggml_tensor * mm  = ctx->node(i0 + 3);

    if (nrm->op != GGML_OP_RMS_NORM || mw->op != GGML_OP_MUL || ms->op != GGML_OP_MUL || mm->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[1]->ne[0] != 1024) {
        return 0;
    }
    const ggml_tensor * x = nrm->src[0];
    const int64_t ne00 = x->ne[0];
    const ggml_tensor * w     = mw->src[0] == nrm ? mw->src[1] : mw->src[0];
    const ggml_tensor * signs = ms->src[1];

    const bool ok =
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) && ne00 % 1024 == 0 && ne00/1024 <= 8 &&
        x->ne[2] == 1 && x->ne[3] == 1 && ggml_are_same_shape(nrm, x) &&
        (mw->src[0] == nrm || mw->src[1] == nrm) && ggml_are_same_shape(mw, nrm) && mw->type == GGML_TYPE_F32 &&
        w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == ne00 &&
        ms->src[0] == mw && ggml_are_same_shape(ms, mw) && ms->type == GGML_TYPE_F32 &&
        signs->type == GGML_TYPE_F32 && ggml_is_contiguous(signs) && ggml_nelements(signs) == ne00 &&
        ggml_is_contiguous(mw) && ggml_is_contiguous(mm) && mm->type == GGML_TYPE_F32 &&
        ggml_metal_use_count(gf, nrm) == 1 && ggml_metal_use_count(gf, ms) == 1 &&
        !(nrm->flags & GGML_TENSOR_FLAG_OUTPUT) && !(ms->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        ggml_metal_unwrap_reshape(gf, mm->src[1]) == ms &&
        !ggml_metal_ranges_overlap(mm, mw) && !ggml_metal_ranges_overlap(mm, w) && !ggml_metal_ranges_overlap(mm, signs);
    if (!ok) {
        return 0;
    }
    const ggml_tensor * a1 = nullptr;
    if (add) {
        const ggml_tensor * a0 = add->src[0];
        a1 = add->src[1];
        if (x != add || a0->type != GGML_TYPE_F32 || a1->type != GGML_TYPE_F32 ||
            !ggml_are_same_shape(a0, add) || !ggml_are_same_shape(a1, add) ||
            a0->nb[0] != sizeof(float) || a1->nb[0] != sizeof(float) || add->nb[0] != sizeof(float) ||
            ggml_metal_ranges_overlap(mm, a0) || ggml_metal_ranges_overlap(mm, a1) || ggml_metal_ranges_overlap(mm, add) ||
            ggml_metal_ranges_overlap(mw, a0) || ggml_metal_ranges_overlap(mw, a1)) {
            return 0;
        }
        x = a0;
    }

    const bool write_norm = ggml_metal_use_count(gf, mw) > 1 || (mw->flags & GGML_TENSOR_FLAG_OUTPUT);
    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    const char * base = simd_size == 64 ? "kernel_rms_norm_fwht_f32_w64" : "kernel_rms_norm_fwht_f32";
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, base);
    if (!pipeline.pipeline) {
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, base, base, nullptr);
    }

    ggml_metal_kargs_rms_norm_fwht args = {
        /*.ne00       =*/ (int32_t) ne00,
        /*.eps        =*/ ggml_get_op_params_f32(nrm, 0),
        /*.nb01       =*/ x->nb[1],
        /*.nb1n       =*/ mw->nb[1],
        /*.nb1h       =*/ (uint64_t) ne00*sizeof(float),
        /*.write_norm =*/ write_norm,
        /*.add        =*/ add != nullptr,
        /*.nb11       =*/ a1 ? a1->nb[1] : 0,
        /*.nb1a       =*/ add ? add->nb[1] : 0,
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(w),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(signs), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mw),    4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),    5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a1 ? a1 : x),   6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add ? add : mw), 7);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (64 + ne00)*sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, x->ne[1], 1, 1, 128*(ne00/1024), 1, 1);

    return add ? 5 : 4;
}

// [CONT of a permuted view] -> MUL by a sign vector -> MUL_MAT with the Hadamard hint, as the
// graph builds the activation transform of a Hadamard-folded weight: one kernel instead of three
int ggml_metal_op_fwht_fused(ggml_metal_op_t ctx, int idx) {
    const ggml_cgraph * gf = ctx->graph();

    const ggml_tensor * cont = nullptr;
    int i = idx;
    if (ctx->node(i)->op == GGML_OP_CONT) {
        cont = ctx->node(i);
        i++;
    }
    if (i + 1 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * mul = ctx->node(i);
    const ggml_tensor * mm  = ctx->node(i + 1);

    if (mul->op != GGML_OP_MUL || mm->op != GGML_OP_MUL_MAT ||
        ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }

    const ggml_tensor * x     = mul->src[0];
    const ggml_tensor * signs = mul->src[1];
    const int64_t n = mm->src[1]->ne[0];

    const bool shapes_ok =
        mul->type == GGML_TYPE_F32 && x->type == GGML_TYPE_F32 && signs->type == GGML_TYPE_F32 &&
        mm->type == GGML_TYPE_F32 && ggml_are_same_shape(mul, x) &&
        ggml_is_contiguous(signs) && ggml_nelements(signs) == x->ne[0] && signs->ne[0] == x->ne[0] &&
        x->ne[0] % n == 0 && (n == 512 || n == 1024) &&
        ggml_is_contiguous(mm) && ggml_are_same_shape(mm, mm->src[1]);
    if (!shapes_ok) {
        return 0;
    }

    // every intermediate must be read only by the next step, since none of them gets written
    if (ggml_metal_use_count(gf, mul) != 1 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        ggml_metal_unwrap_reshape(gf, mm->src[1]) != mul) {
        return 0;
    }

    // with the copy, only the regrouping [hd, nk, rep] -> [hd, rep, nk] of a contiguous tensor
    const ggml_tensor * src = x;
    int hd = 0, rep = 0, nk = 0;
    if (cont) {
        const ggml_tensor * p = cont->src[0];
        if (ggml_metal_use_count(gf, cont) != 1 || (cont->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_metal_unwrap_reshape(gf, x) != cont || cont->type != GGML_TYPE_F32 ||
            p->op != GGML_OP_PERMUTE || p->type != GGML_TYPE_F32 || !ggml_is_contiguous(p->src[0])) {
            return 0;
        }
        hd  = (int) p->ne[0];
        rep = (int) p->ne[1];
        nk  = (int) p->ne[2];
        const size_t es = sizeof(float);
        const int64_t W = (int64_t) hd*rep*nk;
        if (p->nb[0] != es || p->nb[1] != es*hd*nk || p->nb[2] != es*hd || p->nb[3] != es*W ||
            x->ne[0] != W || W % n != 0 || hd > INT16_MAX || rep > INT16_MAX || nk > INT16_MAX) {
            return 0;
        }
        src = p;
    } else if (!ggml_is_contiguous(x)) {
        return 0;
    }

    const int64_t nrows = ggml_nelements(mm) / n;

    ggml_metal_kargs_fwht_fused args = {
        /*.nrows =*/ (int32_t) nrows,
        /*.nsign =*/ (int32_t) signs->ne[0],
    };

    const int simd_size = ggml_metal_device_get_props(ctx->dev)->simd_width;

    auto pipeline = ggml_metal_library_get_pipeline_fwht_fused(ctx->lib, (int) n, hd, rep, nk, simd_size == 64);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(src),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(signs), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mm),    3);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, n*sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 128, 1, 1);

    return i + 2 - idx;
}

int ggml_metal_op_pool_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t * opts = op->op_params;
    ggml_op_pool op_pool = (ggml_op_pool) opts[0];

    const int32_t k0 = opts[1];
    const int32_t k1 = opts[2];
    const int32_t s0 = opts[3];
    const int32_t s1 = opts[4];
    const int32_t p0 = opts[5];
    const int32_t p1 = opts[6];

    const int64_t IH = op->src[0]->ne[1];
    const int64_t IW = op->src[0]->ne[0];

    const int64_t N  = op->ne[3];
    const int64_t OC = op->ne[2];
    const int64_t OH = op->ne[1];
    const int64_t OW = op->ne[0];

    const int64_t np = N * OC * OH * OW;

    ggml_metal_kargs_pool_2d args_pool_2d = {
        /* .k0 = */ k0,
        /* .k1 = */ k1,
        /* .s0 = */ s0,
        /* .s1 = */ s1,
        /* .p0 = */ p0,
        /* .p1 = */ p1,
        /* .IH = */ IH,
        /* .IW = */ IW,
        /* .OH = */ OH,
        /* .OW = */ OW,
        /* .np = */ np
    };

    auto pipeline = ggml_metal_library_get_pipeline_pool_2d(lib, op, op_pool);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), (int) np);
    const int ntg = (np + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args_pool_2d, sizeof(args_pool_2d), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// the number of nodes from idx on that the fusion table fuses as pattern id into one dispatch, 1 if it fuses none
static int ggml_metal_op_try_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_fusion_id id) {
    if (!ctx->use_fusion()) {
        return 1;
    }

    int n = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
    if (fusion == nullptr || ggml_metal_fusion_get_id(fusion) != id) {
        return 1;
    }

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s to %s, %d nodes\n", __func__, ggml_op_name(ctx->node(idx)->op), ggml_op_name(ctx->node(idx + n - 1)->op), n);
    }

    return n;
}

static int ggml_metal_op_mul_mat_mma(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    // the MMA store adds the residual when the table fuses the ADD after this mat-mul
    const int  n_fuse   = ggml_metal_op_try_fusion(ctx, idx, GGML_METAL_FUSION_MUL_MAT_ADD);
    const bool fuse_add = n_fuse > 1;

    const ggml_tensor * dst = op;
    const ggml_tensor * res = dst;

    if (fuse_add) {
        dst = ctx->node(idx + n_fuse - 1);
        // the residual is the other operand of the ADD, by identity: it can itself be a MUL_MAT output
        res = dst->src[0] == op ? dst->src[1] : dst->src[0];
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma_auto(lib, op, fuse_add);

    ggml_metal_kargs_mul_mv_ext args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne10  =*/ ne10,
        /*.ne11  =*/ ne11,
        /*.ne12  =*/ ne12,
        /*.nb10  =*/ nb10,
        /*.nb11  =*/ nb11,
        /*.nb12  =*/ nb12,
        /*.nb13  =*/ nb13,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.r2    =*/ (int16_t) (ne12/ne02),
        /*.r3    =*/ (int16_t) (ne13/ne03),
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),        3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(res),        4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    const int rows0 = pipeline.nr0;
    const int rows1 = pipeline.nr1;

    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + rows0 - 1)/rows0, (ne11 + rows1 - 1)/rows1, ne12*ne13, 32, pipeline.nsg, 1);

    return n_fuse;
}

// the generic mat-vec kernel, or its 2-row Q4_0 variant if nc
static int ggml_metal_op_mul_mat_mv(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv(lib, op);

    const int nr0 = pipeline.nr0;
    const int nr1 = pipeline.nr1;
    const int nsg = pipeline.nsg;

    const size_t smem = pipeline.smem;

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nr0  =*/ nr0,
        /*.r2   =*/ r2,
        /*.r3   =*/ r3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    // wave64 decode dispatches a full 64-wide threadgroup so the kernel's lane
    // partitioning and reduction match the hardware simdgroup; 32 everywhere else.
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    int nth = 32;
    if (props_dev->wave64_decode &&
        (op->src[0]->type == GGML_TYPE_F16 ||
         op->src[0]->type == GGML_TYPE_F32 ||
         op->src[0]->type == GGML_TYPE_BF16 ||
         op->src[0]->type == GGML_TYPE_Q8_0 ||
         op->src[0]->type == GGML_TYPE_Q4_K ||
         op->src[0]->type == GGML_TYPE_Q5_K ||
         op->src[0]->type == GGML_TYPE_Q6_K ||
         op->src[0]->type == GGML_TYPE_Q2_K ||
         op->src[0]->type == GGML_TYPE_Q3_K ||
         op->src[0]->type == GGML_TYPE_MXFP4 ||
         op->src[0]->type == GGML_TYPE_IQ2_XXS ||
         op->src[0]->type == GGML_TYPE_IQ2_XS ||
         op->src[0]->type == GGML_TYPE_IQ3_XXS ||
         op->src[0]->type == GGML_TYPE_IQ3_S ||
         op->src[0]->type == GGML_TYPE_IQ2_S ||
         op->src[0]->type == GGML_TYPE_IQ1_S ||
         op->src[0]->type == GGML_TYPE_IQ1_M ||
         op->src[0]->type == GGML_TYPE_IQ4_NL ||
         op->src[0]->type == GGML_TYPE_IQ4_XS ||
         op->src[0]->type == GGML_TYPE_Q4_0 ||
         op->src[0]->type == GGML_TYPE_Q4_1 ||
         op->src[0]->type == GGML_TYPE_Q5_0 ||
         op->src[0]->type == GGML_TYPE_Q5_1 ||
         op->src[0]->type == GGML_TYPE_Q1_0 ||
         op->src[0]->type == GGML_TYPE_Q2_0 ||
         op->src[0]->type == GGML_TYPE_PQ2_0 ||
         op->src[0]->type == GGML_TYPE_PTQ1_0)) {
        nth = props_dev->simd_width;
    }

    if (op->src[0]->type == GGML_TYPE_F32 ||
        op->src[0]->type == GGML_TYPE_F16 ||
        op->src[0]->type == GGML_TYPE_BF16 ||
        op->src[0]->type == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + nr0 - 1)/(nr0)), ((ne11 + nr1 - 1)/nr1), ne12*ne13, nth, nsg, 1);
    } else {
        int ntg0 = (ne01 + nr0*nsg - 1)/(nr0*nsg);
        if (pipeline.ntg > 0) {
            ntg0 = MIN(ntg0, pipeline.ntg);
        }
        ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ((ne11 + nr1 - 1)/nr1), ne12*ne13, nth, nsg, 1);
    }

    return 1;
}

// RX6800: IQ types on mul_mv_ext for 2..8 columns (spec-decode verify batches).
// Opt-in (GGML_METAL_EXT_IQ=1): measured slower than the per-column mat-vec on RDNA2
// (Hauhau IQ3_M pp2 20.75 vs 24.23 t/s), the generic dequant reads the grid from constant memory.
// GGML_METAL_EXT_IQ_MIN sets the smallest ne11 (default 2).
static bool ggml_metal_op_mul_mat_ext_iq(ggml_type type) {
    static int cached = -1;
    if (cached < 0) {
        const char * val = getenv("GGML_METAL_EXT_IQ");
        cached = (val && val[0] && val[0] != '0') ? 1 : 0;
    }
    if (cached == 0) {
        return false;
    }
    switch (type) {
        case GGML_TYPE_IQ2_XXS:
        case GGML_TYPE_IQ2_XS:
        case GGML_TYPE_IQ2_S:
        case GGML_TYPE_IQ3_XXS:
        case GGML_TYPE_IQ3_S:
        case GGML_TYPE_IQ4_XS:
            return true;
        default:
            return false;
    }
}

static int ggml_metal_op_mul_mat_ext_iq_min(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * val = getenv("GGML_METAL_EXT_IQ_MIN");
        const int n = (val && val[0]) ? atoi(val) : 2;
        cached = n < 2 ? 2 : (n > 9 ? 9 : n);
    }
    return cached;
}

// ne11 the IQ3_S multi-column mat-vec keeps away from the blocked mm (0 when it is switched off)
static int ggml_metal_iq3s_nc_max_eff(void) {
    const char * e = getenv("GGML_METAL_IQ3S_NC");
    return (e && e[0] == '0') ? 0 : ggml_metal_iq3s_nc_max();
}

static int ggml_metal_op_mul_mat_qvk(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = TOSH_ENV("TOSH_QKV_FUSION_DISABLE") != nullptr;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);
    const bool width_ok = props_dev->simd_width == 32 ||
                          (props_dev->simd_width == 64 && props_dev->wave64_decode);
    if (disabled || !ctx->use_fusion() || !width_ok || !props_dev->use_mm_manual || idx + 2 >= ctx->n_nodes()) {
        return 0;
    }

    // Identify the three consecutive MUL_MAT nodes by weight-name suffix, not
    // position: llama.cpp's build_qkv() emits Q,K,V today, but no permutation
    // is guaranteed, so the match must not depend on the emission order.
    // Ends-with (not substring) so names like "*.attn_q.weight.extra" fail closed.
    auto has_suffix = [](const char * name, const char * suffix) {
        const size_t n = strlen(name), s = strlen(suffix);
        return n >= s && strcmp(name + n - s, suffix) == 0;
    };
    ggml_tensor * q = nullptr;
    ggml_tensor * v = nullptr;
    ggml_tensor * k = nullptr;
    for (int i = 0; i < 3; i++) {
        ggml_tensor * n = ctx->node(idx + i);
        if (n->op != GGML_OP_MUL_MAT) {
            return 0;
        }
        const char * wname = ggml_get_name(n->src[0]);
        if (has_suffix(wname, ".attn_q.weight")) {
            q = n;
        } else if (has_suffix(wname, ".attn_v.weight")) {
            v = n;
        } else if (has_suffix(wname, ".attn_k.weight")) {
            k = n;
        }
    }
    if (!q || !v || !k) {
        return 0;
    }

    // Avoid false-positive graph fusion: these must be the three independent
    // decode projections over exactly the same activation. The quant pattern is
    // selected separately below so every segment keeps its GGUF representation.
    // Single-token, single-plane decode only: the pipeline constants hardcode
    // ne12 = r2 = r3 = 1, so batched planes fall back to the unfused path (q, v,
    // k share src[1], checked below, so guarding one covers all three).
    if ((q->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (v->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        (k->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 ||
        q->src[1] != v->src[1] || q->src[1] != k->src[1] ||
        q->src[1]->type != GGML_TYPE_F32 ||
        q->src[1]->ne[1] != 1 || v->src[1]->ne[1] != 1 || k->src[1]->ne[1] != 1 ||
        q->src[1]->ne[2] != 1 || q->src[1]->ne[3] != 1 ||
        q->src[0]->ne[0] != q->src[1]->ne[0] ||
        v->src[0]->ne[0] != v->src[1]->ne[0] ||
        k->src[0]->ne[0] != k->src[1]->ne[0] ||
        q->src[0]->ne[2] != 1 || q->src[0]->ne[3] != 1 ||
        v->src[0]->ne[2] != 1 || v->src[0]->ne[3] != 1 ||
        k->src[0]->ne[2] != 1 || k->src[0]->ne[3] != 1) {
        return 0;
    }

    struct qvk_pattern {
        const char * kernel;
        uint32_t nsg;
        uint32_t nr_q;
        uint32_t nr_v;
        uint32_t nr_k;
    } pattern = {};

    const ggml_type tq = q->src[0]->type;
    const ggml_type tv = v->src[0]->type;
    const ggml_type tk = k->src[0]->type;

    const bool qvk_w64 = props_dev->simd_width == 64 && TOSH_ENV("TOSH_QVK_W64_DISABLE") == nullptr;

    if (tq == GGML_TYPE_Q4_0 && tv == GGML_TYPE_Q4_0 && tk == GGML_TYPE_Q4_0) {
        // the 64-lane twin carries the same decomposition as the standalone matvec, so the
        // fusion keeps that win instead of trading it for the saved dispatches
        pattern = qvk_w64 ? decltype(pattern){ "kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32_w64", 1, 4, 4, 4 }
                          : decltype(pattern){ "kernel_mul_mv_qvk_q4_0_q4_0_q4_0_f32",     1, 4, 4, 4 };
    } else if (tq == GGML_TYPE_Q2_K && tv == GGML_TYPE_Q4_K && tk == GGML_TYPE_Q2_K) {
        pattern = { "kernel_mul_mv_qvk_q2_K_q4_K_q2_K_f32", 2, 4, 2, 4 };
    } else if (tq == GGML_TYPE_Q4_K && tv == GGML_TYPE_Q6_K && tk == GGML_TYPE_Q4_K) {
        pattern = { "kernel_mul_mv_qvk_q4_K_q6_K_q4_K_f32", 2, 2, 2, 4 };
    } else {
        return 0;
    }

    auto make_args = [](const ggml_tensor * op, int32_t nr0) {
        const ggml_tensor * w = op->src[0];
        const ggml_tensor * x = op->src[1];

        GGML_ASSERT(x->ne[2] % w->ne[2] == 0);
        GGML_ASSERT(x->ne[3] % w->ne[3] == 0);

        return ggml_metal_kargs_mul_mv {
            /*.ne00 =*/ (int32_t) w->ne[0],
            /*.ne01 =*/ (int32_t) w->ne[1],
            /*.ne02 =*/ (int32_t) w->ne[2],
            /*.nb00 =*/ w->nb[0],
            /*.nb01 =*/ w->nb[1],
            /*.nb02 =*/ w->nb[2],
            /*.nb03 =*/ w->nb[3],
            /*.ne10 =*/ (int32_t) x->ne[0],
            /*.ne11 =*/ (int32_t) x->ne[1],
            /*.ne12 =*/ (int32_t) x->ne[2],
            /*.nb10 =*/ x->nb[0],
            /*.nb11 =*/ x->nb[1],
            /*.nb12 =*/ x->nb[2],
            /*.nb13 =*/ x->nb[3],
            /*.ne0  =*/ (int32_t) op->ne[0],
            /*.ne1  =*/ (int32_t) op->ne[1],
            /*.nr0  =*/ nr0,
            /*.r2   =*/ (int16_t) (x->ne[2]/w->ne[2]),
            /*.r3   =*/ (int16_t) (x->ne[3]/w->ne[3]),
        };
    };

    ggml_metal_kargs_mul_mv_qvk args = {
        /*.q     =*/ make_args(q, pattern.nr_q),
        /*.v     =*/ make_args(v, pattern.nr_v),
        /*.k     =*/ make_args(k, pattern.nr_k),
        /*.nwg_q =*/ (uint32_t) ((q->src[0]->ne[1] + pattern.nr_q*pattern.nsg - 1)/(pattern.nr_q*pattern.nsg)),
        /*.nwg_v =*/ (uint32_t) ((v->src[0]->ne[1] + pattern.nr_v*pattern.nsg - 1)/(pattern.nr_v*pattern.nsg)),
    };
    const uint32_t nwg_k = (uint32_t) ((k->src[0]->ne[1] + pattern.nr_k*pattern.nsg - 1)/(pattern.nr_k*pattern.nsg));

    char name[256];
    const int16_t nw = (int16_t) props_dev->simd_width;
    snprintf(name, sizeof(name), "%s_nsg=%u_ne12=1_r2=1_r3=1_nw=%d", pattern.kernel, pattern.nsg, nw);
    ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(ctx->lib, name);
    if (!pipeline.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, pattern.nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, nw,  FC_MUL_MV + 5);
        pipeline = ggml_metal_library_compile_pipeline(ctx->lib, pattern.kernel, name, cv);
        ggml_metal_cv_free(cv);
    }

    if (!pipeline.pipeline || ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < nw*(int) pattern.nsg) {
        return 0;
    }

    ggml_metal_encoder_set_pipeline(ctx->enc, pipeline);
    ggml_metal_encoder_set_bytes   (ctx->enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q->src[0]), 1);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(v->src[0]), 2);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(k->src[0]), 3);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q->src[1]), 4);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(q),         5);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(v),         6);
    ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(k),         7);
    ggml_metal_encoder_dispatch_threadgroups(ctx->enc, args.nwg_q + args.nwg_v + nwg_k, 1, 1, nw, pattern.nsg, 1);

    return 3;
}

// K-quants asked for four columns while every other type batches from two, so at two or three
// sequences each column paid its own full pass over the weights
static int kquant_ext_floor() {
    static const int v = []() {
        const char * s = TOSH_ENV("TOSH_EXT_KQ_FLOOR");
        return s ? atoi(s) : 2;
    }();
    return v;
}

// on 64 lanes the batched kernel loses to one mat-vec pass per column for these types: at every
// width for the table lookups and q2_K, and at two columns for the rest
static bool ext_loses_w64(ggml_type t, int ne11) {
    static const bool disabled = TOSH_ENV("TOSH_EXT_W64_SKIP_DISABLE") != nullptr;
    if (disabled) {
        return false;
    }
    switch (t) {
        case GGML_TYPE_IQ4_NL:
        case GGML_TYPE_MXFP4:
        case GGML_TYPE_Q2_K: return true;
        case GGML_TYPE_Q3_K:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1: return ne11 == 2;
        default: return false;
    }
}

// nine and ten columns only beat the tile with five per group, which needs the two-row variant
// and a cheap dequant: q5_K measures flat there and q6_K loses
static bool ext_wide_group(ggml_type tsrc0) {
    return tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q2_K || tsrc0 == GGML_TYPE_Q3_K;
}

// above eight columns the matmul tile takes over, but on 64 lanes it costs the same from nine
// columns to thirty-two, so three column groups of the batched kernel are still cheaper
static int ext_ceil(int simd_width, ggml_type tsrc0) {
    static const int v = []() {
        const char * s = TOSH_ENV("TOSH_EXT_MAX");
        return s ? atoi(s) : 0;
    }();

    if (v) {
        return v;
    }

    if (simd_width == 64) {
        return 12;
    }

    return ext_wide_group(tsrc0) ? 10 : 8;
}

static bool ggml_metal_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// gate and up mat-vecs of one token (PQ2_0, or the dense types on 32 lanes) followed by their swiglu: one
// kernel writes the swiglu output. The three nodes are adjacent, so writing that output early only
// needs it apart from the activation the kernel reads.
static int ggml_metal_op_mul_mat_swiglu(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = TOSH_ENV("TOSH_SWIGLU_FUSION_DISABLE") != nullptr;
    if (disabled || !ctx->use_fusion() || idx + 2 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_tensor * a   = ctx->node(idx);
    const ggml_tensor * b   = ctx->node(idx + 1);
    const ggml_tensor * glu = ctx->node(idx + 2);

    if (a->op != GGML_OP_MUL_MAT || b->op != GGML_OP_MUL_MAT || glu->op != GGML_OP_GLU ||
        ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU || ggml_get_op_params_i32(glu, 1) != 0 || !glu->src[1]) {
        return 0;
    }

    const ggml_tensor * gate = glu->src[0];
    const ggml_tensor * up   = glu->src[1];
    if (!((gate == a && up == b) || (gate == b && up == a))) {
        return 0;
    }

    const ggml_cgraph * gf = ctx->graph();
    const ggml_tensor * x  = gate->src[1];

    // the dense kernels copy the 32-lane standalone mat-vecs (q4_K only where those run two groups of
    // two rows); measured on discrete cards only, so unified memory keeps the separate ones
    static const bool dense_disabled = TOSH_ENV("TOSH_DENSE_SWIGLU_FUSION_DISABLE") != nullptr;
    const ggml_type wt = gate->src[0]->type;
    const ggml_metal_device_props * props = ggml_metal_device_get_props(ctx->dev);
    const bool dense = !dense_disabled && props->simd_width == 32 && !props->has_unified_memory &&
        (wt == GGML_TYPE_Q4_0 || wt == GGML_TYPE_Q5_K || wt == GGML_TYPE_Q8_0 || wt == GGML_TYPE_MXFP4 ||
         (wt == GGML_TYPE_Q4_K && gate->src[0]->ne[0] > 2048 && gate->src[0]->ne[1] > 2048));

    const bool ok =
        (wt == GGML_TYPE_PQ2_0 || dense) && up->src[0]->type == wt &&
        ggml_are_same_shape(gate->src[0], up->src[0]) && up->src[1] == x &&
        ggml_get_op_params_i32(gate, 1) == 0 && ggml_get_op_params_i32(up, 1) == 0 &&
        gate->src[0]->ne[2] == 1 && gate->src[0]->ne[3] == 1 &&
        ggml_is_contiguous(gate->src[0]) && ggml_is_contiguous(up->src[0]) &&
        x->type == GGML_TYPE_F32 && ggml_is_contiguous(x) && x->ne[1] == 1 && x->ne[2] == 1 && x->ne[3] == 1 &&
        (wt != GGML_TYPE_PQ2_0 || gate->src[0]->ne[0] % 128 == 0) &&
        glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && ggml_nelements(glu) == gate->ne[0] &&
        ggml_metal_use_count(gf, gate) == 1 && ggml_metal_use_count(gf, up) == 1 &&
        !(gate->flags & GGML_TENSOR_FLAG_OUTPUT) && !(up->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        !ggml_metal_ranges_overlap(glu, x);
    if (!ok) {
        return 0;
    }

    auto pipeline = wt == GGML_TYPE_PQ2_0 ? ggml_metal_library_get_pipeline_mul_mv_pq2_swiglu(ctx->lib)
                                          : ggml_metal_library_get_pipeline_mul_mv_dense_swiglu(ctx->lib, gate->src[0]);

    const ggml_tensor * w = gate->src[0];

    ggml_metal_kargs_mul_mv args = {
        /*.ne00 =*/ (int32_t) w->ne[0],
        /*.ne01 =*/ (int32_t) w->ne[1],
        /*.ne02 =*/ 1,
        /*.nb00 =*/ w->nb[0],
        /*.nb01 =*/ w->nb[1],
        /*.nb02 =*/ w->nb[2],
        /*.nb03 =*/ w->nb[3],
        /*.ne10 =*/ (int32_t) x->ne[0],
        /*.ne11 =*/ 1,
        /*.ne12 =*/ 1,
        /*.nb10 =*/ x->nb[0],
        /*.nb11 =*/ x->nb[1],
        /*.nb12 =*/ x->nb[2],
        /*.nb13 =*/ x->nb[3],
        /*.ne0  =*/ (int32_t) gate->ne[0],
        /*.ne1  =*/ 1,
        /*.nr0  =*/ pipeline.nr0,
        /*.r2   =*/ 1,
        /*.r3   =*/ 1,
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gate->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(up->src[0]),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),            3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(glu),          4);

    const int nr0 = pipeline.nr0;
    const int nsg = pipeline.nsg;
    if (wt == GGML_TYPE_Q8_0) {
        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (w->ne[1] + nr0 - 1)/nr0, 1, 1, pipeline.nr1, nsg, 1);
    } else {
        if (pipeline.smem) {
            ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
        }
        ggml_metal_encoder_dispatch_threadgroups(enc, (w->ne[1] + nr0*nsg - 1)/(nr0*nsg), 1, 1, pipeline.nr1, nsg, 1);
    }

    return 3;
}

int ggml_metal_op_mul_mat(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    if (ggml_metal_op_mul_mat_use_fwht(op, props_dev->max_theadgroup_memory_size)) {
        return ggml_metal_op_fwht(ctx, idx);
    }

    if (props_dev->has_simdgroup_mm && ggml_metal_op_mul_mat_use_mma(op)) {
        return ggml_metal_op_mul_mat_mma(ctx, idx);
    }

    if (const int n_fuse = ggml_metal_op_mul_mat_qvk(ctx, idx)) {
        return n_fuse;
    }

    if (const int n_fuse = ggml_metal_op_mul_mat_swiglu(ctx, idx)) {
        return n_fuse;
    }

    // few rows over a long k: the GEMM tile would leave most rows empty and too few threadgroups busy
    if (props_dev->simd_width == 32 && !props_dev->has_unified_memory &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
        op->src[0]->ne[1] <= 16 && op->src[0]->ne[0] >= 256 &&
        op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(float) &&
        TOSH_ENV("TOSH_MM_SKINNY_DISABLE") == nullptr) {
        static const int nc = []() { const char * e = TOSH_ENV("TOSH_MM_SKINNY_NC"); return e ? atoi(e) : 2; }();
        char sname[64];
        snprintf(sname, sizeof(sname), "kernel_mul_mm_skinny_f32_16_%d", nc);
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, sname);
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, sname, sname, nullptr);
        }

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ (int32_t) op->src[0]->ne[0],
            /*.ne02 =*/ (int32_t) op->src[0]->ne[2],
            /*.nb01 =*/ op->src[0]->nb[1],
            /*.nb02 =*/ op->src[0]->nb[2],
            /*.nb03 =*/ op->src[0]->nb[3],
            /*.ne12 =*/ (int32_t) op->src[1]->ne[2],
            /*.nb10 =*/ op->src[1]->nb[0],
            /*.nb11 =*/ op->src[1]->nb[1],
            /*.nb12 =*/ op->src[1]->nb[2],
            /*.nb13 =*/ op->src[1]->nb[3],
            /*.ne0  =*/ (int32_t) op->ne[0],
            /*.ne1  =*/ (int32_t) op->ne[1],
            /*.r2   =*/ (int16_t) (op->src[1]->ne[2]/op->src[0]->ne[2]),
            /*.r3   =*/ (int16_t) (op->src[1]->ne[3]/op->src[0]->ne[3]),
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 8*16*nc*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (op->ne[1] + nc - 1)/nc, 1, op->ne[2]*op->ne[3], 256, 1, 1);

        return 1;
    }

    // a short k with many columns: a low-rank factor, which neither the GEMM (k >= 64) nor the vector kernels serve
    if (props_dev->simd_width == 32 && !props_dev->has_unified_memory &&
        op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
        op->src[0]->ne[0] <= 32 && op->src[1]->ne[1] > 8 && op->src[0]->ne[1] >= 64 &&
        op->src[0]->nb[0] == sizeof(float) && op->src[1]->nb[0] == sizeof(float) &&
        TOSH_ENV("TOSH_MM_LOWK_DISABLE") == nullptr) {
        const int16_t kmax = op->src[0]->ne[0] <= 16 ? 16 : 32;
        char name[64];
        snprintf(name, sizeof(name), "kernel_mul_mm_lowk_f32_%d", kmax);
        ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline(lib, name);
        if (!pipeline.pipeline) {
            pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
        }

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ (int32_t) op->src[0]->ne[0],
            /*.ne02 =*/ (int32_t) op->src[0]->ne[2],
            /*.nb01 =*/ op->src[0]->nb[1],
            /*.nb02 =*/ op->src[0]->nb[2],
            /*.nb03 =*/ op->src[0]->nb[3],
            /*.ne12 =*/ (int32_t) op->src[1]->ne[2],
            /*.nb10 =*/ op->src[1]->nb[0],
            /*.nb11 =*/ op->src[1]->nb[1],
            /*.nb12 =*/ op->src[1]->nb[2],
            /*.nb13 =*/ op->src[1]->nb[3],
            /*.ne0  =*/ (int32_t) op->ne[0],
            /*.ne1  =*/ (int32_t) op->ne[1],
            /*.r2   =*/ (int16_t) (op->src[1]->ne[2]/op->src[0]->ne[2]),
            /*.r3   =*/ (int16_t) (op->src[1]->ne[3]/op->src[0]->ne[3]),
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 33*kmax*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, (op->ne[0] + 63)/64, (op->ne[1] + 31)/32, op->ne[2]*op->ne[3], 256, 1, 1);

        return 1;
    }

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 == ne10);

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    const int16_t r2 = ne12/ne02;
    const int16_t r3 = ne13/ne03;

    // RDNA2 VALU blocked mm (kernels/mmq_amd.metal) for Q4_K/IQ3_S prefill. The register-tiled
    // manual kernel_mul_mm is 1.4-3x faster on the same types, so this only runs when that path
    // is disabled; GGML_METAL_MMQ_AMD=1 forces it, =0 turns it off.
    {
        static int mmq_cached = -1;
        if (mmq_cached < 0) {
            const char * val = getenv("GGML_METAL_MMQ_AMD");
            mmq_cached = (val && val[0]) ? (val[0] != '0' ? 1 : 0) : (props_dev->use_mm_manual ? 0 : 1);
        }
        const bool mmq_gate =
            mmq_cached == 1 &&
            ((op->src[0]->type == GGML_TYPE_IQ3_S && ne11 > ggml_metal_iq3s_nc_max_eff()) ||
             op->src[0]->type == GGML_TYPE_Q4_K) &&
            op->src[1]->type == GGML_TYPE_F32 &&
            op->type == GGML_TYPE_F32 &&
            ne00 % 256 == 0 &&
            !ggml_is_transposed(op->src[0]) &&
            !ggml_is_transposed(op->src[1]) &&
            ne11 > 8;
        if (mmq_gate) {
            struct mmq_amd_args {
                ggml_metal_kargs_mul_mm base;
                int32_t ne12;
                int32_t ne13;
                int32_t r2;
                int32_t r3;
            } args = {
                {
                    /*.ne00 =*/ ne00,
                    /*.ne02 =*/ ne02,
                    /*.nb01 =*/ nb01,
                    /*.nb02 =*/ nb02,
                    /*.nb03 =*/ nb03,
                    /*.ne12 =*/ ne12,
                    /*.nb10 =*/ nb10,
                    /*.nb11 =*/ nb11,
                    /*.nb12 =*/ nb12,
                    /*.nb13 =*/ nb13,
                    /*.ne0  =*/ ne0,
                    /*.ne1  =*/ ne1,
                    /*.r2   =*/ r2,
                    /*.r3   =*/ r3,
                    /*.band =*/ 0,
                },
                ne12, ne13, r2, r3,
            };
            char mmq_name[256];
            snprintf(mmq_name, sizeof(mmq_name), "kernel_mul_mm_%s_f32_amd",
                     ggml_type_name(op->src[0]->type));
            auto pipeline = ggml_metal_library_get_pipeline(lib, mmq_name);
            if (!pipeline.pipeline) {
                pipeline = ggml_metal_library_compile_pipeline(
                    lib, mmq_name, mmq_name, nullptr);
            }
            if (pipeline.pipeline) {
                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
                ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);
                ggml_metal_encoder_set_threadgroup_memory_size(enc, (64*32 + 64*32)*sizeof(float) + 512*sizeof(uint32_t), 0);
                ggml_metal_encoder_dispatch_threadgroups(enc, (ne0 + 63)/64, (ne1 + 63)/64, ne12*ne13, 256, 1, 1);
                return 1;
            }
            // else fall through to the normal paths
        }
    }

    // first try to use small-batch mat-mv kernels
    // these should be efficient for BS [2, ~8]
    // ext kernel derives its row split from the simdgroup width, so it runs on both
    // wave32 and wave64. wave64 rides on wave64_decode (matches the mat-vec gating).
    static const bool ext_disabled = TOSH_ENV("TOSH_EXT_DISABLE") != nullptr;
    if (!ext_disabled &&
        (props_dev->simd_width == 32 || (props_dev->simd_width == 64 && props_dev->wave64_decode)) &&
        !(props_dev->simd_width == 64 && ext_loses_w64(op->src[0]->type, ne11)) &&
        ggml_metal_mv_nc2_rows(op->src[0]->type, props_dev->simd_width, ne11, ne01, ne00) == 0 &&
        op->src[1]->type == GGML_TYPE_F32 && (ne00%128 == 0) &&
        (
         (
          (
           op->src[0]->type == GGML_TYPE_F32  || // TODO: helper function
           op->src[0]->type == GGML_TYPE_F16  ||
           op->src[0]->type == GGML_TYPE_BF16 ||
           op->src[0]->type == GGML_TYPE_Q1_0 ||
           op->src[0]->type == GGML_TYPE_Q2_0 ||
           op->src[0]->type == GGML_TYPE_PQ2_0 ||
           op->src[0]->type == GGML_TYPE_Q4_0 ||
           op->src[0]->type == GGML_TYPE_Q4_1 ||
           op->src[0]->type == GGML_TYPE_Q5_0 ||
           op->src[0]->type == GGML_TYPE_Q5_1 ||
           op->src[0]->type == GGML_TYPE_Q8_0 ||
           op->src[0]->type == GGML_TYPE_MXFP4 ||
           op->src[0]->type == GGML_TYPE_IQ4_NL ||
           false) && (ne11 >= 2 && ne11 <= ext_ceil(props_dev->simd_width, op->src[0]->type))
         ) ||
         (
          (
           op->src[0]->type == GGML_TYPE_Q4_K ||
           op->src[0]->type == GGML_TYPE_Q5_K ||
           op->src[0]->type == GGML_TYPE_Q6_K ||
           op->src[0]->type == GGML_TYPE_Q2_K ||
           op->src[0]->type == GGML_TYPE_Q3_K ||
           false) && (ne11 >= kquant_ext_floor() && ne11 <= ext_ceil(props_dev->simd_width, op->src[0]->type))
         ) ||
         (ggml_metal_op_mul_mat_ext_iq(op->src[0]->type) && ne11 >= ggml_metal_op_mul_mat_ext_iq_min() && ne11 <= 8)
        )
       ) {
        // TODO: determine the optimal parameters based on grid utilization
        //       I still don't know why we should not always use the maximum available threads:
        //
        //       nsg = pipeline.maxTotalThreadsPerThreadgroup / 32
        //
        //       my current hypothesis is that the work grid is not evenly divisible for different nsg
        //       values and there can be some tail effects when nsg is high. need to confirm this
        //
        const int nsg    = 1;                 // num simdgroups per threadgroup

        // num threads along row per simdgroup
        // the thresholds below split a 32-lane simdgroup; TOSH_EXT_NXPSG overrides them so the
        // split can be swept on a 64-lane one, where the row/column balance is not the same
        static const int nxpsg_env = []() {
            const char * s = TOSH_ENV("TOSH_EXT_NXPSG");
            return s ? atoi(s) : 0;
        }();

        int16_t nxpsg = 0;
        if (nxpsg_env) {
            nxpsg = (int16_t) nxpsg_env;
        } else {
            if (ne00 % 256 == 0 && ne11 < 3) {
                nxpsg = 16;
            } else if (ne00 % 128 == 0) {
                nxpsg = 8;
            } else {
                nxpsg = 4;
            }
            // a 64-lane simdgroup wants twice the reduction width: +9.0% with four requests.
            // TOSH_EXT_NXPSG_X2 tries it on 32 lanes, so that width can be A/B'd before the gate goes
            // only where the contracted dimension is wide enough to keep four-wide loads per thread:
            // at ne00 128 the doubled split leaves too few elements and the result goes wrong
            if (ne00 % 256 == 0 &&
                (props_dev->simd_width == 64 || TOSH_ENV("TOSH_EXT_NXPSG_X2") != nullptr) &&
                TOSH_ENV("TOSH_EXT_NXPSG_X2_DISABLE") == nullptr) {
                nxpsg = std::min<int16_t>(16, nxpsg*2);
            }
        }

        const int16_t nypsg  = props_dev->simd_width/nxpsg;   // num threads along col per simdgroup (i.e. a simdgroup processes that many src0 rows at a time)
              int16_t r1ptg  = 4;                 // num src1 rows per threadgroup

        // note: not sure how optimal are those across all different hardware. there might be something cleverer
        switch (ne11) {
            case 2:
                r1ptg = 2; break;
            case 3:
            case 6:
                r1ptg = 3; break;
            case 4:
            case 7:
            case 8:
                r1ptg = 4; break;
            case 5:
                r1ptg = 5; break;
            default:
                r1ptg = 4; break;
        };

        // 2 rows per thread: the activation chunk is loaded once and reused across them
        const bool nr0_2 = op->src[0]->type == GGML_TYPE_Q4_K ||
                           op->src[0]->type == GGML_TYPE_Q5_K ||
                           op->src[0]->type == GGML_TYPE_Q6_K ||
                           op->src[0]->type == GGML_TYPE_Q2_K ||
                           op->src[0]->type == GGML_TYPE_Q3_K;

        static const int nr0_env = []() {
            const char * s = TOSH_ENV("TOSH_EXT_NR0");
            return s ? atoi(s) : 0;
        }();

        // only the K-quants have a two-row variant; forcing the override on the rest asks the
        // library for a kernel that was never instantiated
        // Four rows per thread doubles what a threadgroup covers, so a weight with few
        // rows stops filling the grid. The threshold is per row count, not per quant type.
        static const int64_t nr0_min_rows = []() {
            const char * s = TOSH_ENV("TOSH_EXT_NR0_MINROWS");
            // off by default: the win depends on the weight's shape and one measured model
            // still regresses at four columns, so it stays opt-in until that is understood
            return s ? (int64_t) atoll(s) : (int64_t) 0;
        }();
        const bool nr0_wide = nr0_min_rows > 0 && props_dev->simd_width == 32 &&
                              op->src[0]->type == GGML_TYPE_Q4_K && ne01 >= nr0_min_rows;

        // on 64 lanes a q4_K group of four or more columns runs out of the two-row variant's
        // registers and ends up as slow as one pass per column (measured 4.0x one column at four
        // columns, 2.4x with four rows); two and three columns stay faster with two rows
        const bool nr0_w64 = props_dev->simd_width == 64 && op->src[0]->type == GGML_TYPE_Q4_K && ne11 >= 4 &&
                             TOSH_ENV("TOSH_EXT_NR0_W64_DISABLE") == nullptr;

        const int16_t nr0   = nr0_2 ? (nr0_env ? nr0_env : ((nr0_wide || nr0_w64) ? 4 : 2)) : 1;

        // each group makes a full pass over the weights, so the cost is groups by group width:
        // nine and ten columns fit in two groups of five instead of three of four
        if (ext_wide_group(op->src[0]->type) && (ne11 == 9 || ne11 == 10)) {
            r1ptg = 5;
        }

        // the wave64 q6_K variant keeps four rows per simdgroup, so the row split does not apply
        // keep the gate in sync with the pipeline picker in ggml_metal_library_get_pipeline_mul_mv_ext
        static const bool ext_w64_disable = TOSH_ENV("TOSH_EXT_W64_DISABLE") != nullptr;
        const bool ext_w64_q6k = !ext_w64_disable &&
                                 props_dev->simd_width == 64 && op->src[0]->type == GGML_TYPE_Q6_K;

        // from six src1 rows the eight-column variant reads the weights once instead of twice
        if (ext_w64_q6k && ne11 >= 6) {
            r1ptg = 8;
        }

        const int16_t r0ptg = ext_w64_q6k ? (r1ptg == 8 ? 2 : 4)*nsg : nypsg*nsg*nr0; // num src0 rows per threadgroup

        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_ext(lib, op, nsg, nxpsg, r1ptg, nr0);

        ggml_metal_kargs_mul_mv_ext args = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne10  =*/ ne10,
            /*.ne11  =*/ ne11,
            /*.ne12  =*/ ne12,
            /*.nb10  =*/ nb10,
            /*.nb11  =*/ nb11,
            /*.nb12  =*/ nb12,
            /*.nb13  =*/ nb13,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.r2    =*/ r2,
            /*.r3    =*/ r3,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne01 + r0ptg - 1)/r0ptg), ((ne11 + r1ptg - 1)/r1ptg), ne12*ne13, props_dev->simd_width, nsg, 1);
    // the manual tiled kernels give AMD the same matrix path Apple gets from simdgroup_mm
    } else if (ggml_metal_op_mul_mat_use_mm(op, props_dev->has_simdgroup_mm || props_dev->use_mm_manual)) {
        //GGML_LOG_INFO("matrix: ne00 = %6d, ne01 = %6d, ne02 = %6d, ne11 = %6d, ne12 = %6d\n", ne00, ne01, ne02, ne11, ne12);

        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        auto pipeline = ggml_metal_library_get_pipeline_mul_mm(lib, op);

        ggml_metal_kargs_mul_mm args = {
            /*.ne00 =*/ ne00,
            /*.ne02 =*/ ne02,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.ne12 =*/ ne12,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.r2   =*/ r2,
            /*.r3   =*/ r3,
            /*.band =*/ 0,
        };

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        // bands only pay with a last-level cache that holds them; TOSH_MM_BAND_MB sets another
        // size, 0 the plain order
        if (!props_dev->has_simdgroup_mm && !props_dev->has_unified_memory &&
            (strstr(props_dev->desc, "RX 6") || strstr(props_dev->desc, "W6"))) {
            static const int64_t band_bytes = (int64_t) (getenv("TOSH_MM_BAND_MB") ? atoi(getenv("TOSH_MM_BAND_MB")) : 16) << 20;
            const int64_t tile_bytes = (int64_t) nr1*nb11;
            const int64_t ntx = (ne11 + nr1 - 1)/nr1;
            if (band_bytes > 0 && tile_bytes > 0) {
                const int64_t band = std::max<int64_t>(1, band_bytes/tile_bytes);
                args.band = band < ntx ? (int32_t) band : 0;
            }
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

        const size_t smem = pipeline.smem;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        const int sgw = pipeline.sgw ? pipeline.sgw : 32;

        ggml_metal_encoder_dispatch_threadgroups(enc, ((ne11 + nr1 - 1) / nr1), ((ne01 + nr0 - 1) / nr0), ne12 * ne13, sgw, nsg, 1);
    } else {
        return ggml_metal_op_mul_mat_mv(ctx, idx);
    }

    return 1;
}

// upper bound on the number of entry tiles emitted by kernel_mul_mm_id_map0: the full tiles
// over all entries plus at most one partial tile per expert
static int64_t ggml_metal_op_mul_mat_id_max_tiles(const ggml_tensor * op) {
    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    const int64_t n_rows = ne20*ne21;
    const int64_t tiles  = (n_rows + 31)/32 + (ne02 < n_rows ? ne02 : n_rows) - 1;

    return n_rows < tiles ? n_rows : tiles;
}

size_t ggml_metal_op_mul_mat_id_extra_tpe(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    // row map of kernel_mul_mm_id: total tile count + one
    // (expert, first entry, valid rows) descriptor per tile
    const int64_t max_tiles = ggml_metal_op_mul_mat_id_max_tiles(op);

    return ggml_type_size(GGML_TYPE_I32)*(1 + 3*max_tiles);
}

size_t ggml_metal_op_mul_mat_id_extra_ids(const ggml_tensor * op) {
    assert(op->op == GGML_OP_MUL_MAT_ID);

    const int64_t ne02 = op->src[0]->ne[2]; // n_expert
    const int64_t ne20 = op->src[2]->ne[0]; // n_expert_used
    const int64_t ne21 = op->src[2]->ne[1]; // n_token

    // one entry per (expert used, token) pair of ids plus one padding slot per expert
    // (see kernel_mul_mm_id_map0)
    return ggml_type_size(GGML_TYPE_I32)*(ne20*ne21 + ne02);
}

// gpt-oss decode: the gate and up expert mat-vecs, their per-expert bias adds and the clamped
// swiglu that follow them go as one kernel that writes the swiglu output
static int ggml_metal_op_mul_mat_id_swiglu_oai(ggml_metal_op_t ctx, int idx) {
    static const bool disabled = TOSH_ENV("TOSH_MOE_SWIGLU_OAI_FUSION_DISABLE") != nullptr;
    if (disabled || !ctx->use_fusion() || idx + 4 >= ctx->n_nodes()) {
        return 0;
    }

    const ggml_metal_device_props * props = ggml_metal_device_get_props(ctx->dev);
    if (props->simd_width != 32 || props->has_unified_memory) {
        return 0;
    }

    const ggml_tensor * m0  = ctx->node(idx);
    const ggml_tensor * a0  = ctx->node(idx + 1);
    const ggml_tensor * m1  = ctx->node(idx + 2);
    const ggml_tensor * a1  = ctx->node(idx + 3);
    const ggml_tensor * glu = ctx->node(idx + 4);

    if (m0->op != GGML_OP_MUL_MAT_ID || a0->op != GGML_OP_ADD_ID || m1->op != GGML_OP_MUL_MAT_ID ||
        a1->op != GGML_OP_ADD_ID || glu->op != GGML_OP_GLU || ggml_get_glu_op(glu) != GGML_GLU_OP_SWIGLU_OAI ||
        ggml_get_op_params_i32(glu, 1) != 0 || !glu->src[1] || a0->src[0] != m0 || a1->src[0] != m1) {
        return 0;
    }

    const ggml_tensor * gate_b = glu->src[0];
    const ggml_tensor * up_b   = glu->src[1];
    if (!((gate_b == a0 && up_b == a1) || (gate_b == a1 && up_b == a0))) {
        return 0;
    }

    const ggml_tensor * gate = gate_b->src[0];
    const ggml_tensor * up   = up_b->src[0];
    const ggml_tensor * x    = gate->src[1];
    const ggml_tensor * ids  = gate->src[2];
    const ggml_tensor * bg   = gate_b->src[1];
    const ggml_tensor * bu   = up_b->src[1];

    const ggml_cgraph * gf = ctx->graph();

    const bool ok =
        gate->src[0]->type == GGML_TYPE_MXFP4 && up->src[0]->type == GGML_TYPE_MXFP4 &&
        ggml_are_same_shape(gate->src[0], up->src[0]) && ggml_is_contiguous(gate->src[0]) && ggml_is_contiguous(up->src[0]) &&
        up->src[1] == x && up->src[2] == ids && gate_b->src[2] == ids && up_b->src[2] == ids &&
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        ids->op != GGML_OP_TOSH_DMOE_PUBLISH &&
#endif
        !ggml_metal_op_mul_mat_id_use_mm(gate, props->has_simdgroup_mm || props->use_mm_manual) &&
        x->type == GGML_TYPE_F32 && x->nb[0] == sizeof(float) &&
        bg->type == GGML_TYPE_F32 && bu->type == GGML_TYPE_F32 && bg->nb[0] == sizeof(float) && bu->nb[0] == sizeof(float) &&
        bg->ne[0] == gate->ne[0] && bu->ne[0] == gate->ne[0] &&
        gate->type == GGML_TYPE_F32 && ggml_is_contiguous(gate) && ggml_are_same_shape(gate, up) &&
        glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && ggml_are_same_shape(glu, gate) &&
        gate->src[0]->ne[0] % ggml_blck_size(GGML_TYPE_MXFP4) == 0 &&
        ggml_metal_use_count(gf, m0) == 1 && ggml_metal_use_count(gf, m1) == 1 &&
        ggml_metal_use_count(gf, a0) == 1 && ggml_metal_use_count(gf, a1) == 1 &&
        !(m0->flags & GGML_TENSOR_FLAG_OUTPUT) && !(m1->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        !(a0->flags & GGML_TENSOR_FLAG_OUTPUT) && !(a1->flags & GGML_TENSOR_FLAG_OUTPUT) &&
        !ggml_metal_ranges_overlap(glu, x);
    if (!ok) {
        return 0;
    }

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id_mxfp4_swiglu_oai(ctx->lib);

    const ggml_tensor * w = gate->src[0];

    ggml_metal_kargs_mul_mv_id_swiglu_oai args = {
        /*.mv =*/ {
            /*.nei0 =*/ (int32_t) ids->ne[0],
            /*.nei1 =*/ (int32_t) ids->ne[1],
            /*.nbi1 =*/ ids->nb[1],
            /*.ne00 =*/ (int32_t) w->ne[0],
            /*.ne01 =*/ (int32_t) w->ne[1],
            /*.ne02 =*/ (int32_t) w->ne[2],
            /*.nb00 =*/ w->nb[0],
            /*.nb01 =*/ w->nb[1],
            /*.nb02 =*/ w->nb[2],
            /*.ne10 =*/ (int32_t) x->ne[0],
            /*.ne11 =*/ (int32_t) x->ne[1],
            /*.ne12 =*/ (int32_t) x->ne[2],
            /*.ne13 =*/ (int32_t) x->ne[3],
            /*.nb10 =*/ x->nb[0],
            /*.nb11 =*/ x->nb[1],
            /*.nb12 =*/ x->nb[2],
            /*.ne0  =*/ (int32_t) gate->ne[0],
            /*.ne1  =*/ (int32_t) gate->ne[1],
            /*.nb1  =*/ gate->nb[1],
            /*.nr0  =*/ pipeline.nr0,
        },
        /*.nbg1  =*/ bg->nb[1],
        /*.nbu1  =*/ bu->nb[1],
        /*.alpha =*/ ggml_get_op_params_f32(glu, 2),
        /*.limit =*/ ggml_get_op_params_f32(glu, 3),
    };

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gate->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(up->src[0]),   2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),            3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(glu),          4);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),          5);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(bg),           6);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(bu),           7);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (w->ne[1] + pipeline.nr0 - 1)/pipeline.nr0, 1, ids->ne[0]*ids->ne[1],
                                             pipeline.nr1, pipeline.nsg, 1);

    return 5;
}

int ggml_metal_op_mul_mat_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (const int n_fuse = ggml_metal_op_mul_mat_id_swiglu_oai(ctx, idx)) {
        return n_fuse;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // src2 = ids
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);

    GGML_ASSERT(!ggml_is_transposed(op->src[0]));
    GGML_ASSERT(!ggml_is_transposed(op->src[1]));

    GGML_ASSERT(ne03 == 1);
    GGML_ASSERT(ne13 == 1);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    const uint32_t r2 = 1;
    const uint32_t r3 = 1;
    int32_t ne02_run = ne02;

    // escape hatch: revert wave64 MoE prefill to the mat-vec route
    const bool mm_id_manual_ok = props_dev->simd_width == 32 ||
        TOSH_ENV("TOSH_W64_MMID_PREFILL_DISABLE") == NULL;

    const bool use_mm_id = ggml_metal_op_mul_mat_id_use_mm(op, props_dev->has_simdgroup_mm || (props_dev->use_mm_manual && mm_id_manual_ok));

    // the tile path maps ids without the -1 a mixed expert layer publishes
    // (map0 matches ids as 16-bit expert numbers, so the -1 of a wide publish selects nothing)
    GGML_ASSERT(!(use_mm_id && op->src[2]->op == GGML_OP_TOSH_DMOE_PUBLISH && ggml_get_op_params_i32(op->src[2], 1) == 0));

    if (use_mm_id) {
        // some Metal matrix data types require aligned pointers
        // ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf (Table 2.5)
        //switch (op->src[0]->type) {
        //    case GGML_TYPE_F32:  GGML_ASSERT(nb01 % 16 == 0); break;
        //    case GGML_TYPE_F16:  GGML_ASSERT(nb01 % 8  == 0); break;
        //    case GGML_TYPE_BF16: GGML_ASSERT(nb01 % 8  == 0); break;
        //    default: break;
        //}

        // extra buffers for intermediate id mapping
        ggml_metal_buffer_id bid_tpe = bid_dst;
        bid_tpe.offs += ggml_nbytes(op);

        ggml_metal_buffer_id bid_ids = bid_tpe;
        bid_ids.offs += ggml_metal_op_mul_mat_id_extra_tpe(op);

        {
            ggml_metal_kargs_mul_mm_id_map0 args = {
                ne10,
                ne11, // n_expert_used (bcast)
                nb11,
                nb12,
                ne21, // n_tokens
                ne20, // n_expert_used
                nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_map0(lib, ne02_run, ne20);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(ne02_run <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  2);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  3);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, ne02_run, 1, 1);
        }

        // zero the rows that no expert owns - disjoint from what kernel_mul_mm_id writes
        {
            ggml_metal_kargs_mul_mm_id_zero args = {
                /*.ne0  =*/ ne0,
                /*.nb1  =*/ nb1,
                /*.nb2  =*/ nb2,
                /*.nb21 =*/ nb21,
            };

            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id_zero(lib);

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne20, ne21, 1, 32, 1, 1);
        }

        // this barrier is always needed because the next kernel has to wait for the id maps to be computed
        ggml_metal_op_concurrency_reset(ctx);

        {
            auto pipeline = ggml_metal_library_get_pipeline_mul_mm_id(lib, op);

            ggml_metal_kargs_mul_mm_id args = {
                /*.ne00  =*/ ne00,
                /*.ne02  =*/ ne02_run,
                /*.nb01  =*/ nb01,
                /*.nb02  =*/ nb02,
                /*.nb03  =*/ nb03,
                /*.ne11  =*/ ne11, // n_expert_used (bcast)
                /*.nb10  =*/ nb10,
                /*.nb11  =*/ nb11,
                /*.nb12  =*/ nb12,
                /*.nb13  =*/ nb13,
                /*.ne20  =*/ ne20, // n_expert_used
                /*.ne0   =*/ ne0,
                /*.ne1   =*/ ne1,
                /*.r2    =*/ r2,
                /*.r3    =*/ r3,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline);
            ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
            ggml_metal_encoder_set_buffer  (enc, bid_tpe,  3);
            ggml_metal_encoder_set_buffer  (enc, bid_ids,  4);
            ggml_metal_encoder_set_buffer  (enc, bid_dst,  5);

            const size_t smem = pipeline.smem;

            GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            const int max_tiles = (int) ggml_metal_op_mul_mat_id_max_tiles(op);

            ggml_metal_encoder_dispatch_threadgroups(enc, max_tiles, (ne01 + 63)/64, 1, 128, 1, 1);
        }
    } else {
        auto pipeline = ggml_metal_library_get_pipeline_mul_mv_id(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const size_t smem = pipeline.smem;

        ggml_metal_kargs_mul_mv_id args = {
            /*.nei0 =*/ ne20,
            /*.nei1 =*/ ne21,
            /*.nbi1 =*/ nb21,
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02_run,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.ne10 =*/ ne10,
            /*.ne11 =*/ ne11,
            /*.ne12 =*/ ne12,
            /*.ne13 =*/ ne13,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.ne0  =*/ ne0,
            /*.ne1  =*/ ne1,
            /*.nb1  =*/ nb1,
            /*.nr0  =*/ nr0,
        };

        if (ggml_is_quantized(op->src[0]->type)) {
            GGML_ASSERT(ne00 >= nsg*nr0);
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer(enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer(enc, bid_dst,  3);
        ggml_metal_encoder_set_buffer(enc, bid_src2, 4);

        const int64_t _ne1 = 1;
        const int64_t ne123 = ne20*ne21;

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        // wave64 decode: run the MoE mat-vec on a full 64-wide simdgroup (matches the
        // kernel's FC_mul_mv_nw); 32 for wave32 and the CPU-bound types.
        int id_nth = 32;
        if (props_dev->wave64_decode &&
            (op->src[0]->type == GGML_TYPE_F16 ||
             op->src[0]->type == GGML_TYPE_F32 ||
             op->src[0]->type == GGML_TYPE_BF16 ||
             op->src[0]->type == GGML_TYPE_Q8_0 ||
             op->src[0]->type == GGML_TYPE_Q4_K ||
             op->src[0]->type == GGML_TYPE_Q5_K ||
             op->src[0]->type == GGML_TYPE_Q6_K ||
             op->src[0]->type == GGML_TYPE_Q2_K ||
             op->src[0]->type == GGML_TYPE_Q3_K ||
             op->src[0]->type == GGML_TYPE_MXFP4 ||
             op->src[0]->type == GGML_TYPE_IQ2_XXS ||
             op->src[0]->type == GGML_TYPE_IQ2_XS ||
             op->src[0]->type == GGML_TYPE_IQ3_XXS ||
             op->src[0]->type == GGML_TYPE_IQ3_S ||
             op->src[0]->type == GGML_TYPE_IQ2_S ||
             op->src[0]->type == GGML_TYPE_IQ1_S ||
             op->src[0]->type == GGML_TYPE_IQ1_M ||
             op->src[0]->type == GGML_TYPE_IQ4_NL ||
             op->src[0]->type == GGML_TYPE_IQ4_XS ||
             op->src[0]->type == GGML_TYPE_Q4_0 ||
             op->src[0]->type == GGML_TYPE_Q4_1 ||
             op->src[0]->type == GGML_TYPE_Q5_0 ||
             op->src[0]->type == GGML_TYPE_Q5_1 ||
             op->src[0]->type == GGML_TYPE_Q1_0 ||
             op->src[0]->type == GGML_TYPE_Q2_0 ||
             op->src[0]->type == GGML_TYPE_PQ2_0 ||
             op->src[0]->type == GGML_TYPE_PTQ1_0)) {
            id_nth = props_dev->simd_width;
        }

        if (op->src[0]->type == GGML_TYPE_F32 ||
            op->src[0]->type == GGML_TYPE_F16 ||
            op->src[0]->type == GGML_TYPE_BF16 ||
            op->src[0]->type == GGML_TYPE_Q8_0) {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0 - 1)/(nr0), (_ne1 + nr1 - 1)/nr1, ne123, id_nth, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nr0*nsg - 1)/(nr0*nsg), (_ne1 + nr1 - 1)/nr1, ne123, id_nth, nsg, 1);
        }
    }

    return 1;
}

#ifdef TOSH_ENABLE_DYNAMIC_MOE
// TOSH_DMOE_MASK forces router positions to the host for the correctness matrix:
// all | alt | half | random
static int32_t tosh_dmoe_test_mask() {
    static const int32_t mask = []() {
        const char * s = getenv("TOSH_DMOE_MASK");
        if (s == nullptr) return 0;
        if (strcmp(s, "all")    == 0) return 1;
        if (strcmp(s, "alt")    == 0) return 2;
        if (strcmp(s, "half")   == 0) return 3;
        if (strcmp(s, "random") == 0) return 4;
        return 0;
    }();
    return mask;
}

struct tosh_dmoe_wide_args {
    int32_t  n_embd;
    int32_t  n_used;
    int32_t  n_tokens;
    int32_t  n_expert;
    int32_t  rows;
    int32_t  max_host;
    int32_t  tail;
    int32_t  pad;
    uint32_t off_ids;
    uint32_t off_rmp;
    uint32_t off_map;
    uint32_t off_out;
    uint64_t nb_cur1;
    uint64_t nb_ids1;
};

static tosh_dmoe_wide_args tosh_dmoe_wide_args_of(const ggml_tensor * ids, int n_embd, int n_expert, int rows, uint64_t nb_cur1) {
    const int n_used = ids->ne[0], n_tok = ids->ne[1];
    return {
        n_embd, n_used, n_tok, n_expert, rows, TOSH_DMOE_WIDE_MAX_ROWS, 0, 0,
        (uint32_t) tosh_dmoe_wide_off_ids(n_tok, n_embd, n_used),
        (uint32_t) tosh_dmoe_wide_off_rmp(n_tok, n_embd, n_used),
        (uint32_t) tosh_dmoe_wide_off_map(n_tok, n_embd, n_used),
        (uint32_t) tosh_dmoe_wide_off_out(n_tok, n_embd, n_used),
        nb_cur1, ids->nb[1],
    };
}

static int ggml_metal_op_tosh_dmoe_publish_wide(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    const ggml_tensor * cur = op->src[0];
    const ggml_tensor * ids = op->src[1];
    const ggml_tensor * res = op->src[2];

    const int il       = ggml_get_op_params_i32(op, 0);
    const int n_embd   = cur->ne[0];
    const int n_used   = ids->ne[0];
    const int n_tokens = ids->ne[1];
    const int n_expert = res->ne[0];

    GGML_ASSERT(il >= 0 && il < TOSH_DMOE_MAX_LAYERS);
    GGML_ASSERT(n_tokens <= TOSH_DMOE_WIDE_MAX_TOKENS && n_embd % 4 == 0 && n_expert <= 1024);
    GGML_ASSERT(ggml_is_contiguous(op));

    void * ev = ggml_metal_device_dmoe_event(ctx->dev);
    ggml_metal_buffer_id bid_stage = ggml_metal_device_dmoe_wide_stage(ctx->dev, tosh_dmoe_wide_bytes(n_embd, n_used));

    uint64_t value;
    memcpy(&value, &op->op_params[2], sizeof(value));

    tosh_dmoe_wide_args args = tosh_dmoe_wide_args_of(ids, n_embd, n_expert, ggml_get_op_params_i32(op, 1), cur->nb[1]);
    args.tail = ggml_get_op_params_i32(op, 5);
    if (const int live = tosh_dmoe_rare_rows_live(); live > 0) {
        args.rows = live;
    }

    ggml_metal_buffer_id bid_res = ggml_metal_device_dmoe_tables(ctx->dev);
    if (bid_res.metal != nullptr) {
        bid_res.offs = ((size_t) ggml_get_op_params_i32(op, 4)*TOSH_DMOE_MAX_LAYERS + il)*res->ne[0]*sizeof(int32_t);
    } else {
        bid_res = ggml_metal_get_buffer_id(res);
    }

    ggml_metal_encoder_t enc = ctx->enc;

    auto p_act = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_publish_wide_act", "tosh_dmoe_publish_wide_act", nullptr);
    ggml_metal_encoder_set_pipeline(enc, p_act);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cur), 1);
    ggml_metal_encoder_set_buffer  (enc, bid_stage, 2);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tokens, 1, 1, 256, 1, 1);

    auto p_ids = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_publish_wide_ids", "tosh_dmoe_publish_wide_ids", nullptr);
    ggml_metal_encoder_set_pipeline(enc, p_ids);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids), 1);
    ggml_metal_encoder_set_buffer  (enc, bid_res, 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 3);
    ggml_metal_encoder_set_buffer  (enc, bid_stage, 4);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, (size_t) (n_expert + 1)*sizeof(uint32_t), 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1024, 1, 1);

    ggml_metal_encoder_event(enc, ctx->cmd_buf, ev, value, true);
    ggml_metal_op_concurrency_reset(ctx);

    return 1;
}

static int ggml_metal_op_tosh_dmoe_join_wide(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    const ggml_tensor * gpu = op->src[0];
    const ggml_tensor * ids = op->src[1];

    const int n_embd   = gpu->ne[0];
    const int n_used   = gpu->ne[1];
    const int n_tokens = gpu->ne[2];

    void * ev = ggml_metal_device_dmoe_event(ctx->dev);
    ggml_metal_buffer_id bid_stage = ggml_metal_device_dmoe_wide_stage(ctx->dev, tosh_dmoe_wide_bytes(n_embd, n_used));

    uint64_t value;
    memcpy(&value, &ids->op_params[2], sizeof(value));

    ggml_metal_encoder_event(ctx->enc, ctx->cmd_buf, ggml_metal_device_dmoe_reach_event(ctx->dev), value, true);
    ggml_metal_encoder_event(ctx->enc, ctx->cmd_buf, ev, value + 1, false);
    ggml_metal_op_concurrency_reset(ctx);

    tosh_dmoe_wide_args args = tosh_dmoe_wide_args_of(ids, n_embd, ids->src[2]->ne[0], 0, 0);

    auto pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_join_wide", "tosh_dmoe_join_wide", nullptr);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_stage, 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 2);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_used*n_tokens, 1, 1, 128, 1, 1);
    ggml_metal_op_concurrency_reset(ctx);

    return 1;
}

int ggml_metal_op_tosh_dmoe_publish(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ggml_get_op_params_i32(op, 1) > 0) {
        return ggml_metal_op_tosh_dmoe_publish_wide(ctx, idx);
    }

    const ggml_tensor * cur = op->src[0];
    const ggml_tensor * ids = op->src[1];
    const ggml_tensor * res = op->src[2];

    const int il       = ggml_get_op_params_i32(op, 0);
    const int n_embd   = cur->ne[0];
    const int n_used   = ids->ne[0];
    const int n_tokens = ids->ne[1];

    GGML_ASSERT(il >= 0 && il < TOSH_DMOE_MAX_LAYERS);
    GGML_ASSERT(n_tokens <= TOSH_DMOE_MAX_TOKENS && n_embd % 4 == 0 && n_used <= 256);
    GGML_ASSERT(ggml_is_contiguous(op));

    const size_t stride = tosh_dmoe_layer_stride(n_embd, n_used);

    void * ev = ggml_metal_device_dmoe_event(ctx->dev);
    ggml_metal_buffer_id bid_stage = ggml_metal_device_dmoe_stage(ctx->dev, stride);
    bid_stage.offs = (size_t) il*stride;

    uint64_t value;
    memcpy(&value, &op->op_params[2], sizeof(value));

    struct {
        int32_t  n_embd;
        int32_t  n_used;
        uint32_t off_ids;
        uint32_t off_slots;
        uint64_t nb_cur1;
        uint64_t nb_ids1;
        int32_t  il;
        int32_t  mask;
    } args = {
        n_embd,
        n_used,
        (uint32_t) tosh_dmoe_off_ids(n_embd, n_used),
        (uint32_t) tosh_dmoe_off_slots(n_embd, n_used),
        cur->nb[1],
        ids->nb[1],
        il,
        tosh_dmoe_test_mask(),
    };

    auto pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_publish", "tosh_dmoe_publish", nullptr);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(cur), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids), 2);
    // a dynamic cache reads the table version this graph was submitted with
    ggml_metal_buffer_id bid_res = ggml_metal_device_dmoe_tables(ctx->dev);
    if (bid_res.metal != nullptr) {
        bid_res.offs = ((size_t) ggml_get_op_params_i32(op, 4)*TOSH_DMOE_MAX_LAYERS + il)*res->ne[0]*sizeof(int32_t);
    } else {
        bid_res = ggml_metal_get_buffer_id(res);
    }
    ggml_metal_encoder_set_buffer  (enc, bid_res, 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),  4);
    ggml_metal_encoder_set_buffer  (enc, bid_stage, 5);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_tokens, 1, 1, 256, 1, 1);

    ggml_metal_encoder_event(enc, ctx->cmd_buf, ev, value, true);
    ggml_metal_op_concurrency_reset(ctx);

    return 1;
}

int ggml_metal_op_tosh_dmoe_join(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    if (ggml_get_op_params_i32(op, 1) == 1) {
        return ggml_metal_op_tosh_dmoe_join_wide(ctx, idx);
    }

    const ggml_tensor * gpu   = op->src[0];
    const ggml_tensor * slots = op->src[1];

    const int il       = ggml_get_op_params_i32(op, 0);
    const int n_embd   = gpu->ne[0];
    const int n_used   = gpu->ne[1];
    const int n_tokens = gpu->ne[2];

    const size_t stride = tosh_dmoe_layer_stride(n_embd, n_used);

    void * ev = ggml_metal_device_dmoe_event(ctx->dev);
    ggml_metal_buffer_id bid_stage = ggml_metal_device_dmoe_stage(ctx->dev, stride);
    bid_stage.offs = (size_t) il*stride;

    uint64_t value;
    memcpy(&value, &slots->op_params[2], sizeof(value));

    ggml_metal_encoder_event(ctx->enc, ctx->cmd_buf, ev, value + 1, false);
    ggml_metal_op_concurrency_reset(ctx);

    struct {
        int32_t  n_embd;
        int32_t  n_used;
        uint32_t off_out;
    } args = {
        n_embd,
        n_used,
        (uint32_t) tosh_dmoe_off_out(n_embd, n_used),
    };

    auto pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_join", "tosh_dmoe_join", nullptr);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(gpu),   1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(slots), 2);
    ggml_metal_encoder_set_buffer  (enc, bid_stage, 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),    4);
    ggml_metal_encoder_dispatch_threadgroups(enc, n_used*n_tokens, 1, 1, 128, 1, 1);

    return 1;
}

int ggml_metal_op_tosh_dmoe_observe(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);
    const ggml_tensor * ids = op->src[0];

    ggml_metal_buffer_id bid_ring = ggml_metal_device_dmoe_obs_ring(ctx->dev);
    tosh_dmoe_obs_record * rec = tosh_dmoe_obs_reserve(ggml_get_op_params_i32(op, 0), (int) ids->ne[0], (int) ids->ne[1]);
    if (bid_ring.metal == nullptr || rec == nullptr) {
        return 1;
    }
    bid_ring.offs = rec->offset;

    struct {
        int32_t  n_used;
        int32_t  n_tokens;
        uint64_t nb1;
    } args = { (int32_t) ids->ne[0], (int32_t) ids->ne[1], ids->nb[1] };

    auto pipeline = ggml_metal_library_compile_pipeline(ctx->lib, "tosh_dmoe_observe", "tosh_dmoe_observe", nullptr);

    ggml_metal_encoder_t enc = ctx->enc;
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids), 1);
    ggml_metal_encoder_set_buffer  (enc, bid_ring, 2);
    ggml_metal_encoder_dispatch_threadgroups(enc, args.n_tokens, 1, 1, 32, 1, 1);

    return 1;
}
#endif

int ggml_metal_op_add_id(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[2]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_kargs_add_id args = {
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb11 =*/ nb11,
        /*.nb21 =*/ nb21,
    };

    auto pipeline = ggml_metal_library_get_pipeline_base(lib, GGML_OP_ADD_ID);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne00);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, 1, nth, 1, 1);

    return 1;
}

// ── RX6800 FA experiment (ported idea from Basten7/iRon-Llama-RC2) ──
// Upstream gates FLASH_ATTN_EXT on has_simdgroup_mm (Apple7+), which keeps it
// permanently off on AMD dGPUs. RC2 proved the tiled FA kernels do run on
// RDNA2 (W6800X) with a nsg=8 heuristic. Opt-in experiment:
//   GGML_METAL_FA_ENABLE_AMD=1  -> opens the gate and forces the tiled path
static bool ggml_metal_fa_amd_experiment(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * val = getenv("GGML_METAL_FA_ENABLE_AMD");
        cached = (val && val[0] && val[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

// ── RX6800 FA-RDNA2 自研 kernel（kernels/fa_amd.metal）───────────────────────
// env 总开关 GGML_METAL_FA_AMD（默认开，=0 关闭）。DK256 朴素版已对 CPU 全注意力
// token-identical（40 token 贪心），且量化 KV 下关闭即错（34 splits + PPL 4507），
// 故默认接管；形状/对齐不满足时仍回退上游路径。
// 相位旋钮与 ggml-metal-device.cpp 的 RC2 移植同约定：decode-like = nq==1，
// GGML_METAL_DECODE_FA_* / GGML_METAL_PP_FA_* 优先于通用 GGML_METAL_FA_*。

static bool ggml_metal_fa_amd_enabled(void) {
    static int cached = -1;
    if (cached < 0) {
        const char * val = getenv("GGML_METAL_FA_AMD");
        cached = (!val || !val[0] || val[0] != '0') ? 1 : 0;
    }
    return cached == 1;
}

static int ggml_metal_fa_amd_knob(const ggml_tensor * op, const char * key, int def, int mn, int mx) {
    const bool decode_like = op->src[0]->ne[1] == 1;

    char env[128];
    for (int pass = 0; pass < 3; ++pass) {
        if (pass == 0 && !decode_like) continue;
        if (pass == 1 &&  decode_like) continue;
        if (pass == 0) {
            snprintf(env, sizeof(env), "GGML_METAL_DECODE_FA_%s", key);
        } else if (pass == 1) {
            snprintf(env, sizeof(env), "GGML_METAL_PP_FA_%s", key);
        } else {
            snprintf(env, sizeof(env), "GGML_METAL_FA_%s", key);
        }
        const char * v = getenv(env);
        if (v && v[0]) {
            const int n = atoi(v);
            if (n >= mn && n <= mx) {
                return n;
            }
        }
    }
    return def;
}

bool ggml_metal_op_flash_attn_ext_amd_supported(const ggml_tensor * op) {
    if (!ggml_metal_fa_amd_enabled()) {
        return false;
    }

    if (op->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    // kv-rows gather is not handled by the fa_amd kernels, let upstream decide
    if (op->src[5] != nullptr) {
        return false;
    }

    // 仅 F16 KV（dk==dv∈{64,128,256,512}，256/512 为 P1-A 朴素版，512 tile 用 NBC=32）
    if (op->src[1]->type != GGML_TYPE_F16 || op->src[2]->type != GGML_TYPE_F16) {
        return false;
    }

    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];
    if (dk != dv || (dk != 64 && dk != 128 && dk != 256 && dk != 512)) {
        return false;
    }

    // sinks / ALiBi bias / softcap 不支持（回退）
    if (op->src[4] != nullptr) {
        return false;
    }
    {
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(float));
        memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(float));
        if (max_bias != 0.0f || logit_softcap != 0.0f) {
            return false;
        }
    }

    // 直读 VRAM 的连续性/行距护栏（half4 / float4 读）
    if (op->src[0]->nb[0] != 4 || op->src[0]->nb[1] % 16 != 0) return false;   // Q float4 行读
    if (op->nb[0]        != 4 || op->nb[1]        % 16 != 0) return false;   // dst float4 行写
    if (op->src[1]->nb[0] != 2 || op->src[1]->nb[1] % 8 != 0) return false;    // K half4
    if (op->src[2]->nb[0] != 2 || op->src[2]->nb[1] % 8 != 0) return false;    // V half4/half2
    if (op->src[3] && op->src[3]->nb[0] != 2) return false;                    // mask half 标量

    return true;
}

// 数据指针在 buffer 内的对齐护栏（half4/float4 直读），不满足则回退
static bool ggml_metal_op_flash_attn_ext_amd_align_ok(const ggml_tensor * op) {
    ggml_metal_buffer_id bq = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bk = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bv = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bd = ggml_metal_get_buffer_id(op);

    if ((bq.offs % 16) != 0 || (bd.offs % 16) != 0) return false;
    if ((bk.offs %  8) != 0 || (bv.offs %  8) != 0) return false;

    return true;
}

// Mixed KV: sides in {F16, Q8_0, Q4_0, Q4_1} (not both F16) are handled here;
// quantized sides dequantize to scratch F16 (same design as upstream #27390),
// F16 sides bind directly, then the fa_amd kernels run on the result. Covers
// prefill and decode: keeping FA on Metal for mixed KV avoids spilling the op
// to another backend.
// The F16/F16 path above is untouched.
// NOTE: gate is pure tensor metadata on purpose: it also runs in supports_op
// during graph split, where backend buffers are not assigned yet, so it must
// not query buffer ids (the F16 align check stays encode-side for the same reason).
static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op);
static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op);
static int ggml_metal_op_flash_attn_ext_amd_split_count(const ggml_tensor * op);

// prepass pipeline for one KV side by its own type (K and V dequantize independently)
static ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16_for_type(
        ggml_metal_library_t lib, ggml_type type) {
    char base[256];
    snprintf(base, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(type));
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }
    return res;
}

bool ggml_metal_op_flash_attn_ext_amd_quant_supported(const ggml_tensor * op) {
    if (!ggml_metal_fa_amd_enabled()) {
        return false;
    }
    if (op->op != GGML_OP_FLASH_ATTN_EXT) {
        return false;
    }

    // kv-rows gather is not handled by the fa_amd kernels, let upstream decide
    if (op->src[5] != nullptr) {
        return false;
    }
    const ggml_type ktype = op->src[1]->type;
    const ggml_type vtype = op->src[2]->type;
    const bool k_is_quant = ktype == GGML_TYPE_Q8_0 || ktype == GGML_TYPE_Q4_0 || ktype == GGML_TYPE_Q4_1 || ktype == GGML_TYPE_IQ4_NL || ktype == GGML_TYPE_Q5_0 || ktype == GGML_TYPE_Q5_1;
    const bool v_is_quant = vtype == GGML_TYPE_Q8_0 || vtype == GGML_TYPE_Q4_0 || vtype == GGML_TYPE_Q4_1 || vtype == GGML_TYPE_IQ4_NL || vtype == GGML_TYPE_Q5_0 || vtype == GGML_TYPE_Q5_1;
    if (!k_is_quant && ktype != GGML_TYPE_F16) return false;
    if (!v_is_quant && vtype != GGML_TYPE_F16) return false;
    if (!k_is_quant && !v_is_quant) return false; // F16/F16 stays on the path above
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];
    // P1-B: dk256/512 开给量化 KV（512 tile 用 NBC=32；首验 q8_0，其余组合 P1-C 逐个验）
    if (dk != dv || (dk != 64 && dk != 128 && dk != 256 && dk != 512)) {
        return false;
    }
    if (op->src[4] != nullptr) {
        return false;
    }
    {
        float max_bias      = 0.0f;
        float logit_softcap = 0.0f;
        memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(float));
        memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(float));
        if (max_bias != 0.0f || logit_softcap != 0.0f) {
            return false;
        }
    }
    if (op->src[0]->nb[0] != 4 || op->src[0]->nb[1] % 16 != 0) return false;
    if (op->nb[0]        != 4 || op->nb[1]        % 16 != 0) return false;
    if (op->src[3] && op->src[3]->nb[0] != 2) return false;
    // F16 sides are read directly, so they need the same row guards as the F16 path
    if (!k_is_quant && (op->src[1]->nb[0] != 2 || op->src[1]->nb[1] % 8 != 0)) return false;
    if (!v_is_quant && (op->src[2]->nb[0] != 2 || op->src[2]->nb[1] % 8 != 0)) return false;
    // scratch is always reserved (upstream keeps the extra) and the kernels
    // read it with the same contiguous-F16 layout they already handle
    return ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0;
}

// register-tiled prefill (ToshLLM pf_amd); GGML_METAL_FA_PF_DISABLE keeps the tile kernel
static bool ggml_metal_op_flash_attn_ext_amd_pf(
        ggml_metal_op_t ctx, const ggml_metal_kargs_flash_attn_ext_rdna2 & args, int dk, bool has_mask,
        ggml_metal_buffer_id bid_q, ggml_metal_buffer_id bid_k, ggml_metal_buffer_id bid_v,
        ggml_metal_buffer_id bid_m, ggml_metal_buffer_id bid_dst) {
    static const bool pf_disabled = getenv("GGML_METAL_FA_PF_DISABLE") != nullptr;
    if (pf_disabled || args.ne01 < 32 || (dk != 64 && dk != 128 && dk != 256)) {
        return false;
    }

    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_amd_pf(ctx->lib, dk, has_mask);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, (void *) &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_q,   1);
    ggml_metal_encoder_set_buffer  (enc, bid_k,   2);
    ggml_metal_encoder_set_buffer  (enc, bid_v,   3);
    ggml_metal_encoder_set_buffer  (enc, bid_m,   4);
    ggml_metal_encoder_set_buffer  (enc, bid_dst, 5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, 0, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (args.ne01 + pipeline.nr0 - 1)/pipeline.nr0, args.ne02, args.ne03, 32, 8, 1);

    return true;
}

static int ggml_metal_op_flash_attn_ext_amd_quant(ggml_metal_op_t ctx, int idx, float scale) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS(uint64_t, nbd, op,         nb); // dst [DV,H,N,B]: head stride is nbd1

    const bool has_mask = op->src[3] != nullptr;

    const int32_t  ne31 = has_mask ? (int32_t) op->src[3]->ne[1] : 0;
    const int32_t  ne32 = has_mask ? (int32_t) op->src[3]->ne[2] : 0;
    const int32_t  ne33 = has_mask ? (int32_t) op->src[3]->ne[3] : 0;
    const uint64_t nb31 = has_mask ? op->src[3]->nb[1] : 0;
    const uint64_t nb32 = has_mask ? op->src[3]->nb[2] : 0;
    const uint64_t nb33 = has_mask ? op->src[3]->nb[3] : 0;

    ggml_metal_buffer_id bid_q    = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_m    = has_mask ? ggml_metal_get_buffer_id(op->src[3]) : bid_q;
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    const ggml_type ktype = op->src[1]->type;
    const ggml_type vtype = op->src[2]->type;
    const bool k_is_quant = ktype == GGML_TYPE_Q8_0 || ktype == GGML_TYPE_Q4_0 || ktype == GGML_TYPE_Q4_1 || ktype == GGML_TYPE_IQ4_NL || ktype == GGML_TYPE_Q5_0 || ktype == GGML_TYPE_Q5_1;
    const bool v_is_quant = vtype == GGML_TYPE_Q8_0 || vtype == GGML_TYPE_Q4_0 || vtype == GGML_TYPE_Q4_1 || vtype == GGML_TYPE_IQ4_NL || vtype == GGML_TYPE_Q5_0 || vtype == GGML_TYPE_Q5_1;
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

    const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
    GGML_ASSERT(nblocks1_64 <= INT32_MAX);
    const int32_t nblocks1 = nblocks1_64;

    ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
    bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    if (k_is_quant) {
        auto pipeline_k = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16_for_type(lib, ktype);
        const int nth_k = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_k), 256);

        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_k);
        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,   1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth_k - 1)/nth_k, 1, 1, nth_k, 1, 1);
    }

    if (v_is_quant && !v_is_view_of_k) {
        auto pipeline_v = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16_for_type(lib, vtype);
        const int nth_v = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_v), 256);

        const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
        GGML_ASSERT(nblocks2_64 <= INT32_MAX);
        const int32_t nblocks2 = nblocks2_64;

        ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
            /*.ne0    =*/ ne20,
            /*.ne1    =*/ ne21,
            /*.ne2    =*/ ne22,
            /*.ne3    =*/ ne23,
            /*.nb0    =*/ nb20,
            /*.nb1    =*/ nb21,
            /*.nb2    =*/ nb22,
            /*.nb3    =*/ nb23,
            /*.nblocks =*/ nblocks2,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_v);
        ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src2,  1);
        ggml_metal_encoder_set_buffer  (enc, bid_v_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth_v - 1)/nth_v, 1, 1, nth_v, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    ggml_metal_buffer_id bid_k = k_is_quant ? bid_kv_f16 : bid_src1;
    ggml_metal_buffer_id bid_v = v_is_view_of_k ? bid_k : (v_is_quant ? bid_v_f16 : bid_src2);

    const uint64_t nb10_attn = k_is_quant ? sizeof(ggml_fp16_t) : (uint64_t) nb10;
    const uint64_t nb11_attn = k_is_quant ? nb10_attn*(uint64_t) ne10 : (uint64_t) nb11;
    const uint64_t nb12_attn = k_is_quant ? nb11_attn*(uint64_t) ne11 : (uint64_t) nb12;
    const uint64_t nb13_attn = k_is_quant ? nb12_attn*(uint64_t) ne12 : (uint64_t) nb13;

    uint64_t nb20_attn;
    uint64_t nb21_attn;
    uint64_t nb22_attn;
    uint64_t nb23_attn;

    if (v_is_view_of_k) {
        nb20_attn = nb10_attn;
        nb21_attn = nb11_attn;
        nb22_attn = nb12_attn;
        nb23_attn = nb13_attn;
    } else if (v_is_quant) {
        nb20_attn = sizeof(ggml_fp16_t);
        nb21_attn = nb20_attn*(uint64_t) ne20;
        nb22_attn = nb21_attn*(uint64_t) ne21;
        nb23_attn = nb22_attn*(uint64_t) ne22;
    } else {
        nb20_attn = nb20;
        nb21_attn = nb21;
        nb22_attn = nb22;
        nb23_attn = nb23;
    }

    const int dk = (int) ne10;

    if (ne01 == 1) {
        // decode vec path: split-k partials + reduce merge (same binds as tile)
        // P1-A: dk256/512 只有 nbc64 实例，先钳住（调优见 P4）
        const int nbc   = (dk == 256 || dk == 512) ? 64 : ggml_metal_fa_amd_knob(op, "NBC", 128, 64, 128);
        const int split = ggml_metal_op_flash_attn_ext_amd_split_count(op);
        const int chunk = (int) GGML_PAD(((uint64_t) ne11 + split - 1)/split, (uint64_t) nbc);

        ggml_metal_buffer_id bid_part = bid_dst;
        bid_part.offs += ggml_nbytes(op)
            + ggml_metal_op_flash_attn_ext_extra_pad(op)
            + ggml_metal_op_flash_attn_ext_extra_blk(op)
            + ggml_metal_op_flash_attn_ext_extra_tmp(op)
            + ggml_metal_op_flash_attn_ext_extra_kv_f16(op)
            + ggml_metal_op_flash_attn_ext_extra_idx(op);

        ggml_metal_kargs_flash_attn_ext_rdna2 args = {
            /*.ne01    =*/ ne01,
            /*.ne02    =*/ ne02,
            /*.ne03    =*/ ne03,
            /*.nb01    =*/ nb01,
            /*.nb02    =*/ nb02,
            /*.nb03    =*/ nb03,
            /*.ne11    =*/ ne11,
            /*.ne_12_2 =*/ ne12,
            /*.ne_12_3 =*/ ne13,
            /*.nb11    =*/ nb11_attn,
            /*.nb12    =*/ nb12_attn,
            /*.nb13    =*/ nb13_attn,
            /*.nb21    =*/ nb21_attn,
            /*.nb22    =*/ nb22_attn,
            /*.nb23    =*/ nb23_attn,
            /*.ne31    =*/ ne31,
            /*.ne32    =*/ ne32,
            /*.ne33    =*/ ne33,
            /*.nb31    =*/ nb31,
            /*.nb32    =*/ nb32,
            /*.nb33    =*/ nb33,
            /*.split   =*/ split,
            /*.chunk   =*/ chunk,
            /*.scale   =*/ scale,
            /*.nbq_d   =*/ nbd2,
            /*.nbh_d   =*/ nbd1,
            /*.nbb_d   =*/ nbd3,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_amd_vec(lib, dk, nbc, has_mask);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_q,    1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_m,    4);
        ggml_metal_encoder_set_buffer  (enc, bid_part, 5);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, (nbc + 8)*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, split, ne02, ne03, 32, 4, 1);

        ggml_metal_op_concurrency_reset(ctx);

        auto pipeline_r = ggml_metal_library_get_pipeline_flash_attn_ext_amd_reduce(lib, dk);

        // Hard guard: reduce reuses Q strides for the dst write, which coincides
        // with dst strides only for single-query layout with dk == dv.
        GGML_ASSERT(op->src[0]->ne[1] == 1);
        GGML_ASSERT(op->src[1]->ne[0] == op->src[2]->ne[0]);

        ggml_metal_kargs_flash_attn_ext_amd_reduce args_r = {
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb02  =*/ nbd1,
            /*.nb03  =*/ nbd3,
            /*.split =*/ split,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_r);
        ggml_metal_encoder_set_bytes   (enc, &args_r, sizeof(args_r), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_part, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 0, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne02, ne03, 1, dk/2, 1, 1);

        return 1;
    }

    ggml_metal_kargs_flash_attn_ext_rdna2 args = {
        /*.ne01    =*/ ne01,
        /*.ne02    =*/ ne02,
        /*.ne03    =*/ ne03,
        /*.nb01    =*/ nb01,
        /*.nb02    =*/ nb02,
        /*.nb03    =*/ nb03,
        /*.ne11    =*/ ne11,
        /*.ne_12_2 =*/ ne12,
        /*.ne_12_3 =*/ ne13,
        /*.nb11    =*/ nb11_attn,
        /*.nb12    =*/ nb12_attn,
        /*.nb13    =*/ nb13_attn,
        /*.nb21    =*/ nb21_attn,
        /*.nb22    =*/ nb22_attn,
        /*.nb23    =*/ nb23_attn,
        /*.ne31    =*/ ne31,
        /*.ne32    =*/ ne32,
        /*.ne33    =*/ ne33,
        /*.nb31    =*/ nb31,
        /*.nb32    =*/ nb32,
        /*.nb33    =*/ nb33,
        /*.split   =*/ 1,
        /*.chunk   =*/ 0,
        /*.scale   =*/ scale,
        /*.nbq_d   =*/ nbd2,
        /*.nbh_d   =*/ nbd1,
        /*.nbb_d   =*/ nbd3,
        };

    if (ggml_metal_op_flash_attn_ext_amd_pf(ctx, args, dk, has_mask, bid_q, bid_k, bid_v, bid_m, bid_dst)) {
        return 1;
    }

    // dk512 tile 用 NBC=32（75KB LDS 超 64KB 上限，32 则 40KB），其余 64
    const int nbc_tile = (dk == 512) ? 32 : 64;
    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_amd_tile(lib, dk, nbc_tile, has_mask);

    const size_t smem = (dk/4)*(nbc_tile + 1)*8
        + 32*nbc_tile*sizeof(float)
        + 32*8*sizeof(float)*2
        + 32*sizeof(float)*3;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_q,   1);
    ggml_metal_encoder_set_buffer  (enc, bid_k,   2);
    ggml_metal_encoder_set_buffer  (enc, bid_v,   3);
    ggml_metal_encoder_set_buffer  (enc, bid_m,   4);
    ggml_metal_encoder_set_buffer  (enc, bid_dst, 5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + 31)/32, ne02, ne03, 32, 8, 1);

    return 1;
}

// vec 路径的 kv split 数（0 = 自动）
static int ggml_metal_op_flash_attn_ext_amd_split_count(const ggml_tensor * op) {
    const int knob = ggml_metal_fa_amd_knob(op, "SPLIT", 0, 0, 16);
    if (knob > 0) {
        return knob;
    }

    // 自动：目标 ~96 个 threadgroup，保小模型 decode 占用率
    const int64_t ntg = op->src[0]->ne[2] * op->src[0]->ne[3];
    int64_t split = 1;
    while (split < 16 && ntg*split < 96) {
        split <<= 1;
    }

    const int64_t nblocks = (op->src[1]->ne[1] + 127)/128;   // NBC=128
    return (int) MIN(split, MAX(1, nblocks));
}

size_t ggml_metal_op_flash_attn_ext_extra_amd(const ggml_tensor * op) {
    // vec split partials are needed on both the F16 and the quantized-KV paths;
    // without this the vec kernel writes past the reserved area (GPU faults)
    if (!ggml_metal_op_flash_attn_ext_amd_supported(op) &&
        !ggml_metal_op_flash_attn_ext_amd_quant_supported(op)) {
        return 0;
    }
    if (op->src[0]->ne[1] != 1) {
        return 0;   // 只有 vec 路径用 split 部分结果缓冲
    }

    const int64_t dk    = op->src[1]->ne[0];
    const int     split = ggml_metal_op_flash_attn_ext_amd_split_count(op);

    return op->src[0]->ne[3] * op->src[0]->ne[2] * split * (dk + 2) * sizeof(float);
}

static int ggml_metal_op_flash_attn_ext_amd(ggml_metal_op_t ctx, int idx, float scale) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const bool has_mask = op->src[3] != nullptr;

    const int32_t  ne31 = has_mask ? (int32_t) op->src[3]->ne[1] : 0;
    const int32_t  ne32 = has_mask ? (int32_t) op->src[3]->ne[2] : 0;
    const int32_t  ne33 = has_mask ? (int32_t) op->src[3]->ne[3] : 0;
    const uint64_t nb31 = has_mask ? op->src[3]->nb[1] : 0;
    const uint64_t nb32 = has_mask ? op->src[3]->nb[2] : 0;
    const uint64_t nb33 = has_mask ? op->src[3]->nb[3] : 0;

    ggml_metal_buffer_id bid_q   = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_k   = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_v   = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_m   = has_mask ? ggml_metal_get_buffer_id(op->src[3]) : bid_q;
    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    // split 部分结果缓冲：追加在上游 extra（pad/blk/tmp/kv_f16/idx）之后
    ggml_metal_buffer_id bid_part = bid_dst;
    bid_part.offs += ggml_nbytes(op)
        + ggml_metal_op_flash_attn_ext_extra_pad(op)
        + ggml_metal_op_flash_attn_ext_extra_blk(op)
        + ggml_metal_op_flash_attn_ext_extra_tmp(op)
        + ggml_metal_op_flash_attn_ext_extra_kv_f16(op)
        + ggml_metal_op_flash_attn_ext_extra_idx(op);

    const int dk = (int) ne10;   // == dv（门控保证）

    if (ne01 == 1) {
        // ── decode vec 路径：split-k 部分结果 + reduce 合并 ──
        // P1-A: dk256/512 只有 nbc64 实例，先钳住（调优见 P4）
        const int nbc   = (dk == 256 || dk == 512) ? 64 : ggml_metal_fa_amd_knob(op, "NBC", 128, 64, 128);
        const int split = ggml_metal_op_flash_attn_ext_amd_split_count(op);
        const int chunk = (int) GGML_PAD(((uint64_t) ne11 + split - 1)/split, (uint64_t) nbc);

        ggml_metal_kargs_flash_attn_ext_rdna2 args = {
            /*.ne01    =*/ ne01,
            /*.ne02    =*/ ne02,
            /*.ne03    =*/ ne03,
            /*.nb01    =*/ nb01,
            /*.nb02    =*/ nb02,
            /*.nb03    =*/ nb03,
            /*.ne11    =*/ ne11,
            /*.ne_12_2 =*/ ne12,
            /*.ne_12_3 =*/ ne13,
            /*.nb11    =*/ nb11,
            /*.nb12    =*/ nb12,
            /*.nb13    =*/ nb13,
            /*.nb21    =*/ nb21,
            /*.nb22    =*/ nb22,
            /*.nb23    =*/ nb23,
            /*.ne31    =*/ ne31,
            /*.ne32    =*/ ne32,
            /*.ne33    =*/ ne33,
            /*.nb31    =*/ nb31,
            /*.nb32    =*/ nb32,
            /*.nb33    =*/ nb33,
            /*.split   =*/ split,
            /*.chunk   =*/ chunk,
            /*.scale   =*/ scale,
            /*.nbq_d   =*/ nb2,
            /*.nbh_d   =*/ nb1,
            /*.nbb_d   =*/ nb3,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_amd_vec(lib, dk, nbc, has_mask);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_q,    1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_m,    4);
        ggml_metal_encoder_set_buffer  (enc, bid_part, 5);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, (nbc + 8)*sizeof(float), 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, split, ne02, ne03, 32, 4, 1);

        ggml_metal_op_concurrency_reset(ctx);

        auto pipeline_r = ggml_metal_library_get_pipeline_flash_attn_ext_amd_reduce(lib, dk);

        // Hard guard: reduce reuses Q strides for the dst write, which coincides
        // with dst strides only for single-query layout with dk == dv.
        GGML_ASSERT(op->src[0]->ne[1] == 1);
        GGML_ASSERT(op->src[1]->ne[0] == op->src[2]->ne[0]);

        ggml_metal_kargs_flash_attn_ext_amd_reduce args_r = {
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb02  =*/ nb1,
            /*.nb03  =*/ nb3,
            /*.split =*/ split,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_r);
        ggml_metal_encoder_set_bytes   (enc, &args_r, sizeof(args_r), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_part, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, 0, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne02, ne03, 1, dk/2, 1, 1);

        return 1;
    }

    // ── prefill tile 路径（NQ=32, NBC=64, tg 256）──
    ggml_metal_kargs_flash_attn_ext_rdna2 args = {
        /*.ne01    =*/ ne01,
        /*.ne02    =*/ ne02,
        /*.ne03    =*/ ne03,
        /*.nb01    =*/ nb01,
        /*.nb02    =*/ nb02,
        /*.nb03    =*/ nb03,
        /*.ne11    =*/ ne11,
        /*.ne_12_2 =*/ ne12,
        /*.ne_12_3 =*/ ne13,
        /*.nb11    =*/ nb11,
        /*.nb12    =*/ nb12,
        /*.nb13    =*/ nb13,
        /*.nb21    =*/ nb21,
        /*.nb22    =*/ nb22,
        /*.nb23    =*/ nb23,
        /*.ne31    =*/ ne31,
        /*.ne32    =*/ ne32,
        /*.ne33    =*/ ne33,
        /*.nb31    =*/ nb31,
        /*.nb32    =*/ nb32,
        /*.nb33    =*/ nb33,
        /*.split   =*/ 1,
        /*.chunk   =*/ 0,
        /*.scale   =*/ scale,
        /*.nbq_d   =*/ nb2,
        /*.nbh_d   =*/ nb1,
        /*.nbb_d   =*/ nb3,
    };

    if (ggml_metal_op_flash_attn_ext_amd_pf(ctx, args, dk, has_mask, bid_q, bid_k, bid_v, bid_m, bid_dst)) {
        return 1;
    }

    // dk512 tile 用 NBC=32（75KB LDS 超 64KB 上限，32 则 40KB），其余 64
    const int nbc_tile = (dk == 512) ? 32 : 64;
    auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_amd_tile(lib, dk, nbc_tile, has_mask);

    // 与 fa_amd.metal 的 LDS 布局严格一致：kt + sw + mps + lps + mst + cst + lst
    const size_t smem = (dk/4)*(nbc_tile + 1)*8  // kt half4 × (NBC+1) 行距
        + 32*nbc_tile*sizeof(float)              // sw [NQ][NBC]
        + 32*8*sizeof(float)*2                   // mps + lps
        + 32*sizeof(float)*3;                    // mst + cst + lst

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_q,   1);
    ggml_metal_encoder_set_buffer  (enc, bid_k,   2);
    ggml_metal_encoder_set_buffer  (enc, bid_v,   3);
    ggml_metal_encoder_set_buffer  (enc, bid_m,   4);
    ggml_metal_encoder_set_buffer  (enc, bid_dst, 5);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
    ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + 31)/32, ne02, ne03, 32, 8, 1);

    return 1;
}


// Measured depth where the 32-way split starts to win for a single-row dk128 query on 64 lanes, per
// local query and KV heads; shapes that were not measured keep the full-die depth.
static int64_t ggml_metal_fa_wg_dk128_min_kv_w64(int64_t n_head, int64_t n_head_kv) {
    static const struct { int64_t n_head, n_head_kv, min_kv; } measured[] = {
        {  8, 4, 1536 },
        { 16, 8, 3584 },
        { 16, 4, 3072 },
        { 16, 2, 3584 },
        { 16, 1, 4096 },
        { 20, 4, 4608 },
    };
    for (const auto & m : measured) {
        if (m.n_head == n_head && m.n_head_kv == n_head_kv) {
            return m.min_kv;
        }
    }
    return 16384;
}

// The same for dk64, where the split only pays with few local query heads; 0 keeps it closed.
static int64_t ggml_metal_fa_wg_dk64_min_kv_w64(int64_t n_head, int64_t n_head_kv) {
    static const struct { int64_t n_head, n_head_kv, min_kv; } measured[] = {
        { 16, 4, 6144 },
    };
    for (const auto & m : measured) {
        if (m.n_head == n_head && m.n_head_kv == n_head_kv) {
            return m.min_kv;
        }
    }
    return 0;
}

// A kernel the driver refuses to build (local memory over the device limit) is refused
// every time, and asking again costs a compile per layer and token.
static ggml_metal_pipeline_with_params ggml_metal_fa_amd_pipeline(ggml_metal_library_t lib, const char * name) {
    static std::mutex refused_mtx;
    static std::unordered_set<std::string> refused;

    auto pipeline = ggml_metal_library_get_pipeline(lib, name);
    if (pipeline.pipeline) {
        return pipeline;
    }
    {
        std::lock_guard<std::mutex> lock(refused_mtx);
        if (refused.count(name)) {
            return pipeline;
        }
    }
    pipeline = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    if (!pipeline.pipeline) {
        std::lock_guard<std::mutex> lock(refused_mtx);
        refused.insert(name);
    }
    return pipeline;
}

// Suffix for the AMD vec FA pipeline name, or nullptr for types we don't instantiate.
static const char * ggml_metal_fa_amd_type_suffix(enum ggml_type t) {
    switch (t) {
        case GGML_TYPE_F32:  return "f32";
        case GGML_TYPE_F16:  return "f16";
        case GGML_TYPE_Q8_0: return "q8_0";
        case GGML_TYPE_Q4_0: return "q4_0";
        case GGML_TYPE_TURBO2_0: return "turbo2";
        case GGML_TYPE_TURBO3_0: return "turbo3";
        case GGML_TYPE_TURBO4_0: return "turbo4";
        default:             return nullptr;
    }
}

bool ggml_metal_op_flash_attn_ext_use_vec(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const int64_t ne00 = op->src[0]->ne[0]; // head size
    const int64_t ne01 = op->src[0]->ne[1]; // batch size

    // use vec kernel if the batch size is small and if the head size is supported
    return (ne01 < 20) && (ne00 % 32 == 0);
}

// ref: https://github.com/ggml-org/llama.cpp/pull/27390
// dequantize the quantized KV cache to F16 before running the F16 flash attention kernels
static bool ggml_metal_op_flash_attn_ext_use_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // depending on compute/bandwidth ratio, dequant to f16 kv is not always beneficial
    // ref: https://github.com/ggml-org/llama.cpp/pull/27390#issuecomment-5355152767
    // TODO: tune per device
    // large heads need the upfront dequant to fit the non-vec threadgroup memory
    if (op->src[0]->ne[1] < 32 &&
        (op->src[0]->ne[0] < 512 || ggml_metal_op_flash_attn_ext_use_vec(op))) {
        return false;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            return true;
        default:
            return false;
    }
}

static bool ggml_metal_op_flash_attn_ext_use_tensor(const ggml_tensor * op, bool has_tensor) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    if (!has_tensor || ggml_metal_op_flash_attn_ext_use_vec(op)) {
        return false;
    }

    const int64_t ne01 = op->src[0]->ne[1];
    const int64_t ne02 = op->src[0]->ne[2];
    const int64_t ne03 = op->src[0]->ne[3];

    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 64  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 192 && dv == 128) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return false;
    }

    // large heads use fewer queries per threadgroup, so that the queries fit in threadgroup memory
    const int64_t nqptg = dk >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG;

    // few heads and small batches do not fill the GPU - the half8x8 kernel is faster there
    // TODO: tune per device
    if (((ne01 + nqptg - 1)/nqptg)*ne02*ne03*dk < 8192) {
        return false;
    }

    if (op->src[1]->type != GGML_TYPE_F16 && op->src[1]->type != GGML_TYPE_BF16 && !ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
        return false;
    }

    // with kv_rows, the attention runs over the compacted row lists (padded to 256), not over the cache
    const int64_t ne11_attn = op->src[5] ? op->src[5]->ne[0] : op->src[1]->ne[1];

    return ne11_attn % OP_FLASH_ATTN_EXT_TENSOR_NCPSG == 0;
}

// returns the n_kv_max hint if the sparse path is available for this op, or 0 otherwise
// the mask (src[3]) remains the single source of truth: finite entries are the valid KV positions,
// n_kv_max is only an upper bound on their number per mask row, used to size the index lists
static int ggml_metal_op_flash_attn_ext_n_kv_max_sparse(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // K/V row lists are handled by the dense kernels, not by the sparse path
    if (op->src[5] != nullptr) {
        return 0;
    }

    int32_t n_kv_max = 0;
    memcpy(&n_kv_max, ((const int32_t *) op->op_params) + 4, sizeof(n_kv_max));

    if (n_kv_max <= 0) {
        return 0;
    }

    // the sparse indices are gathered from the mask
    if (!op->src[3]) {
        return 0;
    }

    // bound the size of the index lists
    if (n_kv_max > 4096) {
        return 0;
    }

    // vec kernel instantiations exist for these (type, dk, dv) combinations only
    const int64_t dk = op->src[1]->ne[0];
    const int64_t dv = op->src[2]->ne[0];

    const bool dk_dv_ok = (dk == 32  && dv == 32)  ||
                          (dk == 64  && dv == 64)  ||
                          (dk == 96  && dv == 96)  ||
                          (dk == 96  && dv == 64)  ||
                          (dk == 128 && dv == 128) ||
                          (dk == 128 && dv == 96)  ||
                          (dk == 192 && dv == 128) ||
                          (dk == 192 && dv == 192) ||
                          (dk == 256 && dv == 256) ||
                          (dk == 320 && dv == 256) ||
                          (dk == 512 && dv == 512) ||
                          (dk == 576 && dv == 512);

    if (!dk_dv_ok) {
        return 0;
    }

    switch (op->src[1]->type) {
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
        case GGML_TYPE_F32:
        case GGML_TYPE_Q4_0:
        case GGML_TYPE_Q4_1:
        case GGML_TYPE_Q5_0:
        case GGML_TYPE_Q5_1:
        case GGML_TYPE_Q8_0:
            break;
        default:
            return 0;
    }

    return n_kv_max;
}

// in some models (e.g. MLA-based), V is a view of K (the first ne20 elements of each K row);
// the dequantized V is then a view of the dequantized K and does not need its own dequant or scratch
// - ref: https://github.com/ggml-org/llama.cpp/pull/13435
static bool ggml_metal_op_flash_attn_ext_v_is_view_of_k(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * K = op->src[1];
    const ggml_tensor * V = op->src[2];

    return V->view_src && (V->view_src == K || (V->view_src == K->view_src && V->view_offs == K->view_offs));
}

// size of the F16 dequantized K tensor; the dequantized V tensor follows it in the same scratch buffer
static size_t ggml_metal_op_flash_attn_ext_kv_f16_k_size(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    return GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11*ne12*ne13, 16);
}

size_t ggml_metal_op_flash_attn_ext_extra_pad(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;
    const bool use_kv_f16 = ggml_metal_op_flash_attn_ext_use_kv_f16(op);

    // when the KV is dequantized to F16, the pad kernel copies the tail chunk from the F16 scratch buffer
    // note: when V is a view of K, the dequantized V is read from the dequantized K with K's row stride
    const bool v_is_view_of_k = use_kv_f16 && ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    uint64_t nb11_pad = nb11;
    uint64_t nb21_pad = nb21;

    if (use_kv_f16) {
        nb11_pad = sizeof(ggml_fp16_t)*ne10;
        nb21_pad = sizeof(ggml_fp16_t)*(v_is_view_of_k ? ne10 : ne20);
    }

    // note: the non-vec kernel requires more extra memory, so always reserve for it
    GGML_ASSERT(OP_FLASH_ATTN_EXT_NCPSG >= OP_FLASH_ATTN_EXT_VEC_NCPSG);

    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (false) {
        // note: always reserve the padding space to avoid graph reallocations
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_VEC_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_VEC_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    } else {
        //const bool has_kvpad = ne11 % OP_FLASH_ATTN_EXT_NCPSG != 0;
        const bool has_kvpad = true;

        if (has_kvpad) {
            res += OP_FLASH_ATTN_EXT_NCPSG*(
                nb11_pad*ne12*ne13 +
                nb21_pad*ne22*ne23 +
                (has_mask ? ggml_type_size(GGML_TYPE_F16)*ne31*ne32*ne33 : 0));
        }
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_blk(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    const bool has_mask = op->src[3] != nullptr;

    if (!has_mask) {
        return res;
    }

    const bool is_vec = ggml_metal_op_flash_attn_ext_use_vec(op) && !ggml_metal_fa_amd_experiment();

    // this optimization is not useful for the vector kernels
    // note: always reserve the blk buffer to avoid graph reallocations
    //if (is_vec) {
    //    return res;
    //}

    const int nqptg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NQPSG : OP_FLASH_ATTN_EXT_NQPSG;
    const int ncpsg = is_vec ? OP_FLASH_ATTN_EXT_VEC_NCPSG : OP_FLASH_ATTN_EXT_NCPSG;

    const int64_t ne1 = (ne01 + nqptg - 1)/nqptg;
    const int64_t ne0 = (ne30 + ncpsg - 1)/ncpsg;

    res += GGML_PAD(ggml_type_size(GGML_TYPE_I8)*ne0*ne1*ne32*ne33, 32);

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
  //GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
  //GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);

    size_t res = 0;

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (ggml_metal_op_flash_attn_ext_use_vec(op)) {
    if (true) {
        const int64_t nwg = 32;
        const int64_t ne01_max = std::min(ne01, 32);

        // temp buffer for writing the results from each workgroup
        // - ne20: the size of the Value head
        // -  + 2: the S and M values for each intermediate result
        res += ggml_type_size(GGML_TYPE_F32)*(ne01_max*ne02*ne03*nwg*(ne20 + 2));
    }

    return res;
}

size_t ggml_metal_op_flash_attn_ext_extra_kv_f16(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    // note: always reserve the temp buffer to avoid graph reallocations
    //if (!ggml_metal_op_flash_attn_ext_use_kv_f16(op)) {
    //    return 0;
    //}

    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const size_t k_size = ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

    // when V is a view of K, the dequantized V is a view of the dequantized K
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);
    if (v_is_view_of_k) {
        return k_size;
    }

    const size_t v_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*ne21*ne22*ne23, 16);

    return k_size + v_size;
}

// size of the sparse index lists: one list of KV indices per mask row,
// padded with -1 up to a multiple of OP_FLASH_ATTN_EXT_VEC_NCPSG
size_t ggml_metal_op_flash_attn_ext_extra_idx(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);

    const int n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);

    if (n_kv_max <= 0) {
        return 0;
    }

    const int n_kv_max_padded = GGML_PAD(n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG);

    return GGML_PAD(sizeof(int32_t)*(size_t) n_kv_max_padded*ne31*ne32*ne33, 16);
}

// size of the contiguous F16 buffers holding the K/V rows referenced by the kv_rows lists
// the attention over these buffers is a standard dense op (no per-tile gather in the main kernel)
size_t ggml_metal_op_flash_attn_ext_extra_rows(const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    const ggml_tensor * kv_rows = op->src[5];

    if (!kv_rows) {
        return 0;
    }

    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);

    const int64_t n_rows   = kv_rows->ne[0];
    const int64_t n_slices = kv_rows->ne[1];

    // note: when V is a view of K, the gathered V is read from the gathered K, so only K is gathered
    const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

    const size_t k_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*n_rows*ne12*n_slices, 16);
    const size_t v_size = v_is_view_of_k ? 0 : GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne20*n_rows*ne22*n_slices, 16);

    return k_size + v_size;
}

int ggml_metal_op_flash_attn_ext(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne2, op->src[2], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb2, op->src[2], nb);
    GGML_TENSOR_LOCALS( int32_t, ne3, op->src[3], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb3, op->src[3], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS( int32_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);

    //GGML_ASSERT(ggml_are_same_shape (src1, src2));
    GGML_ASSERT(ne11 == ne21);
    GGML_ASSERT(ne12 == ne22);

    GGML_ASSERT(!op->src[3] || op->src[3]->type == GGML_TYPE_F16);
    GGML_ASSERT(!op->src[3] || op->src[3]->ne[1] >= op->src[0]->ne[1] &&
            "the Flash-Attention Metal kernel requires the mask to be at least n_queries big");

    // K/V row lists (ggml_flash_attn_ext_rows): src[5] holds one list of cache rows per q slice
    // the attention loops then run over the mask columns and read K/V rows through the lists
    const ggml_tensor * kv_rows = op->src[5];
    const bool has_rows = kv_rows != nullptr;

    GGML_ASSERT(!has_rows || (kv_rows->type == GGML_TYPE_I32 && ggml_is_contiguous(kv_rows)));
    GGML_ASSERT(!has_rows || (kv_rows->ne[1] == ne03 && kv_rows->ne[2] == 1 && kv_rows->ne[3] == 1));
    GGML_ASSERT(!has_rows || op->src[3]);

    // the loop bound over the mask columns / K/V row lists
    const int64_t ne11_attn = has_rows ? kv_rows->ne[0] : ne11;

    // the row lists are padded by the KV cache to a multiple of 256, so the pad path never triggers
    GGML_ASSERT(!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_NCPSG == 0);
    GGML_ASSERT(!has_rows || ne11_attn % OP_FLASH_ATTN_EXT_VEC_NCPSG == 0);

    float scale;
    float max_bias;
    float logit_softcap;

    memcpy(&scale,         ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias,      ((const int32_t *) op->op_params) + 1, sizeof(max_bias));
    memcpy(&logit_softcap, ((const int32_t *) op->op_params) + 2, sizeof(logit_softcap));

    if (logit_softcap != 0.0f) {
        scale /= logit_softcap;
    }

    const bool has_mask  = op->src[3] != NULL;
    const bool has_sinks = op->src[4] != NULL;

    GGML_ASSERT(!has_rows || has_mask);
    const bool has_bias  = max_bias != 0.0f;
    const bool has_scap  = logit_softcap != 0.0f;

    const uint32_t n_head      = op->src[0]->ne[2];
    const  int32_t n_head_log2 = 1u << (uint32_t) floorf(log2f((float) n_head));

    const float m0 = powf(2.0f, -(max_bias       ) / n_head_log2);
    const float m1 = powf(2.0f, -(max_bias / 2.0f) / n_head_log2);

    GGML_ASSERT(ne01 < 65536);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_src2 = ggml_metal_get_buffer_id(op->src[2]);
    ggml_metal_buffer_id bid_src3 = has_mask  ? ggml_metal_get_buffer_id(op->src[3]) : bid_src0;
    ggml_metal_buffer_id bid_src4 = has_sinks ? ggml_metal_get_buffer_id(op->src[4]) : bid_src0;

    ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_pad = bid_dst;
    bid_pad.offs += ggml_nbytes(op);

    ggml_metal_buffer_id bid_blk = bid_pad;
    bid_blk.offs += ggml_metal_op_flash_attn_ext_extra_pad(op);

    ggml_metal_buffer_id bid_tmp = bid_blk;
    bid_tmp.offs += ggml_metal_op_flash_attn_ext_extra_blk(op);

    // AMD discrete GPUs miscompile the templated vec kernel, so route attention (decode
    // and prefill) to the dedicated AMD kernel for the standard K/V pairs it covers.
    // Keeps quantized KV on the GPU instead of the CPU fallback. Gated by TOSH_FA_AMD.
    char fa_amd_pipeline[80] = {0};
    // dk 72 is instantiated for f16 K/V only (vision towers); see the .metal side.
    const bool fa_amd_f16_kv =
        op->src[1]->type == GGML_TYPE_F16 && op->src[2]->type == GGML_TYPE_F16;
    const bool fa_amd_dk72 = ne00 == 72 && fa_amd_f16_kv;
    // UNet diffusion heads, f16 only like dk72: SD 1.5 runs 40 at full resolution,
    // where the score matrix costs 2.7 GB at 768 and does not fit at all at 1024
    const bool fa_amd_unet = (ne00 == 40 || ne00 == 80 || ne00 == 160) && fa_amd_f16_kv;
    // dk 384/640 are instantiated for Turbo KV only; see the .metal side
    const auto fa_amd_is_turbo = [](enum ggml_type t) {
        return t == GGML_TYPE_TURBO2_0 || t == GGML_TYPE_TURBO3_0 || t == GGML_TYPE_TURBO4_0;
    };
    const bool fa_amd_turbo_kv = fa_amd_is_turbo(op->src[1]->type) || fa_amd_is_turbo(op->src[2]->type);
    if (ne00 == ne20 &&
        (ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 ||
         ((ne00 == 384 || ne00 == 640) && fa_amd_turbo_kv) ||
         fa_amd_dk72 || fa_amd_unet)) {
        const char * ks = ggml_metal_fa_amd_type_suffix(op->src[1]->type);
        const char * vs = ggml_metal_fa_amd_type_suffix(op->src[2]->type);
        if (ks && vs) {
            snprintf(fa_amd_pipeline, sizeof(fa_amd_pipeline), "kernel_flash_attn_ext_vec_amd_dk%d_k%s_v%s%s", (int) ne00, ks, vs,
                     props_dev->simd_width == 64 ? "_w64" : "");
        }
    } else if (((ne00 == 576 && ne20 == 512) || (ne00 == 320 && ne20 == 256)) &&
               op->src[1]->type == op->src[2]->type) {
        const char * ks = ggml_metal_fa_amd_type_suffix(op->src[1]->type);
        const char * vs = ggml_metal_fa_amd_type_suffix(op->src[2]->type);
        if (ks && vs) {
            snprintf(fa_amd_pipeline, sizeof(fa_amd_pipeline), "kernel_flash_attn_ext_vec_amd_dk%d_dv%d_k%s_v%s%s",
                     (int) ne00, (int) ne20, ks, vs, props_dev->simd_width == 64 ? "_w64" : "");
        }
    }
    // wave32 rides on simdgroup_reduction; wave64 uses the _w64 instantiations and
    // is gated by wave64_decode (simdgroup_reduction is forced off there).
    const bool fa_amd_width_ok = props_dev->simd_width == 32
        ? props_dev->has_simdgroup_reduction
        : props_dev->wave64_decode;
    // sparse attention (QSA / DeepSeek-V4): the mask already carries the selection, the
    // index kernel compacts it and the vec kernel walks the list instead of the whole cache
    const int amd_n_kv_max = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool amd_sparse  = amd_n_kv_max > 0 && ne01 == 1;
    const int amd_n_kv_pad = amd_sparse ? GGML_PAD(amd_n_kv_max, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    if (TOSH_ENV("TOSH_FA_AMD") != NULL && ggml_metal_fa_amd_enabled() && fa_amd_pipeline[0] != '\0' &&
        !props_dev->has_simdgroup_mm && fa_amd_width_ok &&
        !has_bias && !has_rows) {

        ggml_metal_kargs_flash_attn_ext_amd args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ ne11,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ int32_t(nb11/nb10),
            /*.nb11          =*/ nb11,
            /*.nb12          =*/ nb12,
            /*.nb13          =*/ nb13,
            /*.ns20          =*/ int32_t(nb21/nb20),
            /*.nb21          =*/ nb21,
            /*.nb22          =*/ nb22,
            /*.nb23          =*/ nb23,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.has_sinks     =*/ has_sinks ? 1 : 0,
            /*.has_mask      =*/ has_mask ? 1 : 0,
            /*.n_kv_max_padded =*/ amd_n_kv_pad,
        };

        // queries per threadgroup, matching the instantiation for this head size
        static const bool pf512 = TOSH_ENV("TOSH_FA_PF_DK512") != nullptr;
        // MLA keeps K wider than V; the tile kernel it used to fall back to measures ~26% slower
        static const bool pfmla = TOSH_ENV("TOSH_FA_PF_MLA_DISABLE") == nullptr;
        const bool fa_pf_512 = pf512 && ne00 == 512 && ne00 == ne20;
        const bool fa_pf_mla = pfmla && ne00 == 576 && ne20 == 512 &&
                               op->src[2]->type == GGML_TYPE_F16;
        const bool fa_pf_mla320 = ne00 == 320 && ne20 == 256 && op->src[1]->type == op->src[2]->type;
        // BQ must match the instantiation: dispatching in wider blocks than the kernel
        // covers leaves most queries uncomputed
        int fa_pf_bq = (ne00 == 256 || ne00 == 160) ? 32
            : ((fa_pf_512 || fa_pf_mla || fa_pf_mla320) ? 16 : 64);
        const bool fa_pf_head_ok =
            ne00 == 64 || ne00 == 128 || ne00 == 256 || fa_amd_dk72 || fa_amd_unet || fa_pf_512 ||
            fa_pf_mla || fa_pf_mla320;
        // PF always dispatches 256 threads and subdivides them into logical
        // 16-lane query groups, so the same specialization works on wave32
        // (8 physical simdgroups) and wave64 (4 physical simdgroups).
        if (TOSH_ENV("TOSH_FA_PF_DEBUG") != nullptr) {
            fprintf(stderr, "FA_PF gate dk=%d dv=%d ne01=%d head_ok=%d ktype=%s vtype=%s\n",
                    (int) ne00, (int) ne20, (int) ne01, (int) fa_pf_head_ok,
                    ggml_type_name(op->src[1]->type), ggml_type_name(op->src[2]->type));
        }
        if ((props_dev->simd_width == 32 || props_dev->simd_width == 64) &&
            ne01 >= 32 && fa_pf_head_ok && TOSH_ENV("TOSH_FA_PF_DISABLE") == NULL &&
            TOSH_ENV("GGML_METAL_FA_PF_DISABLE") == NULL) {
            char fa_pf_pipeline[80];
            bool fa_pf_asym = false;
            if (fa_pf_mla) {
                snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk576_dv512_k%s_vf16",
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type));
                fa_pf_asym = true;
            } else if (ne20 != ne00) {
                snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk%d_dv%d_k%s_v%s",
                         (int) ne00, (int) ne20,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type));
                fa_pf_asym = true;
            } else {
            snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "kernel_flash_attn_ext_pf_amd_dk%d_k%s_v%s", (int) ne00,
                     ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type));

                // the 64-lane twin lives in fa_w64.metal, with the lane group as a parameter.
                // Only the symmetric heads have one; MLA and the asymmetric pairs stay above.
                // TOSH_FA_PF_LG picks the group so the three can be compared; 16 reproduces
                // the 32-lane kernel exactly and is the control.
                static const int pf_lg = []() {
                    // the cross sweep left the group at sixteen: wider only wins at depth 0,
                    // where this kernel weighs least, and loses 3 to 37% deeper in
                    const char * s = TOSH_ENV("TOSH_FA_PF_LG");
                    return s ? atoi(s) : 16;
                }();
                // TOSH_FA_PF_BQ picks the query-block twin of that group, which is what keeps
                // the rows per thread fixed while the group widens
                static const int pf_bq = []() {
                    const char * s = TOSH_ENV("TOSH_FA_PF_BQ");
                    return s ? atoi(s) : 0;
                }();
                // TOSH_FA_PF_BK picks the key-block twin: the counterpart of BQ on the other
                // side of the K/V reuse, so a bigger block pays the same traffic fewer times
                static const int pf_bk = []() {
                    const char * s = TOSH_ENV("TOSH_FA_PF_BK");
                    return s ? atoi(s) : 0;
                }();
                if (props_dev->simd_width == 64 && (pf_lg == 16 || pf_lg == 32 || pf_lg == 64) &&
                    (ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512) &&
                    TOSH_ENV("TOSH_FA_PF_W64_DISABLE") == nullptr) {
                    char w64_name[96];
                    if (pf_bq > 0) {
                        snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%dq%dk%d", fa_pf_pipeline, pf_lg, pf_bq,
                                 pf_bk > 0 ? pf_bk : 64);
                    } else {
                        snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%d", fa_pf_pipeline, pf_lg);
                    }
                    if (ggml_metal_library_get_pipeline(lib, w64_name).pipeline ||
                        ggml_metal_library_compile_pipeline(lib, w64_name, w64_name, nullptr).pipeline) {
                        snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "%s", w64_name);
                        // the grid steps in query blocks, so it has to follow the twin
                        if (pf_bq > 0) {
                            fa_pf_bq = pf_bq;
                        }
                    }
                }
            }

            // the asymmetric heads compose their name in the branches above, so their twin is
            // picked here instead. Same group of sixteen; the fallback is the 32-lane kernel.
            if (fa_pf_asym && props_dev->simd_width == 64 &&
                TOSH_ENV("TOSH_FA_PF_W64_DISABLE") == nullptr) {
                char w64_name[96];
                // sixteen reproduces the 32-lane kernel exactly, so it is the control; the
                // sweep that closed the symmetric heads there never covered an asymmetric one
                const char * slg = TOSH_ENV("TOSH_FA_PF_LG");
                snprintf(w64_name, sizeof(w64_name), "%s_w64_lg%d", fa_pf_pipeline, slg ? atoi(slg) : 16);
                if (ggml_metal_fa_amd_pipeline(lib, w64_name).pipeline) {
                    snprintf(fa_pf_pipeline, sizeof(fa_pf_pipeline), "%s", w64_name);
                }
            }

            auto pipeline = ggml_metal_fa_amd_pipeline(lib, fa_pf_pipeline);

            // Which twin a head actually lands on cannot be read from the timings: a switch that
            // never reaches the kernel gives the same number on both sides.
            if (TOSH_ENV("TOSH_FA_PF_DEBUG") != nullptr) {
                fprintf(stderr, "FA_PF dk=%d dv=%d bq=%d -> %s%s\n",
                        (int) ne00, (int) ne20, fa_pf_bq, fa_pf_pipeline,
                        pipeline.pipeline ? "" : "  (NO EXISTE, cae al tile)");
            }

            // it fits in registers with nothing to spare; a compiler bump could
            // push it under the 256 threads it dispatches. A driver that refused to build
            // the kernel leaves the pipeline null, which the tile path below covers: older
            // GCN cards reject these over their local storage limit.
            if (pipeline.pipeline &&
                ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) >= 256) {
                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
                ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
                ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
                ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
                ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

                ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + fa_pf_bq - 1)/fa_pf_bq, ne02, ne03,
                                                         props_dev->simd_width, 256/props_dev->simd_width, 1);

                return 1;
            }
        }

        // prefill goes to the tile kernel (K/V shared across 16 queries);
        // decode keeps the per-query vec kernel
        if (ne01 >= 16 && TOSH_ENV("TOSH_FA_TILE_DISABLE") == NULL) {
            char fa_tile_pipeline[80];
            if (ne20 != ne00) {
                // MLA heads carry the value width in the name: 576/512 and 320/256
                snprintf(fa_tile_pipeline, sizeof(fa_tile_pipeline), "kernel_flash_attn_ext_tile_amd_dk%d_dv%d_k%s_v%s%s",
                         (int) ne00, (int) ne20,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type),
                         props_dev->simd_width == 64 ? "_w64" : "");
            } else {
                snprintf(fa_tile_pipeline, sizeof(fa_tile_pipeline), "kernel_flash_attn_ext_tile_amd_dk%d_k%s_v%s%s", (int) ne00,
                         ggml_metal_fa_amd_type_suffix(op->src[1]->type), ggml_metal_fa_amd_type_suffix(op->src[2]->type),
                         props_dev->simd_width == 64 ? "_w64" : "");
            }

            auto pipeline = ggml_metal_fa_amd_pipeline(lib, fa_tile_pipeline);

            // a missed instantiation leaves a null pipeline; fall through to the vec route
            if (pipeline.pipeline) {
                ggml_metal_encoder_set_pipeline(enc, pipeline);
                ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
                ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
                ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
                ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
                ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
                ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

                // queries per threadgroup must match the instantiation: 16 on wave32, 8 on wave64
                const int fa_nq = props_dev->simd_width == 64 ? 8 : 16;
                ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + fa_nq - 1)/fa_nq, ne02, ne03, props_dev->simd_width, fa_nq, 1);

                return 1;
            }
        }

        // must match the .metal instantiations: q4_0 dequant caps the pipeline, so those
        // pairs drop to 24 (dk128 q4_0 keys, dk64 subw=8 with q4_0 in either K or V).
        int fa_amd_nsg = ne00 == 128 ? 16 : ne00 <= 128 ? 32 : 16;
        if (props_dev->simd_width == 32) {
            const bool k_q4 = op->src[1]->type == GGML_TYPE_Q4_0;
            const bool v_q4 = op->src[2]->type == GGML_TYPE_Q4_0;
            // the Turbo macro instantiates every head at 16, and pairing q4_0 with it
            // leaves the pipeline at 512 threads, so 24 would overrun what it admits
            if (!fa_amd_turbo_kv && ((ne00 == 128 && k_q4) || (ne00 == 64 && (k_q4 || v_q4)))) {
                fa_amd_nsg = 24;
            }
        }
        if (props_dev->simd_width == 64) {
            fa_amd_nsg /= 2;
        }
        // the simdgroup count per head was never swept on either width; the pipeline caps it,
        // so an override that overruns is dropped below rather than dispatched
        // one value for every head size overruns the pipelines that cap lower, so the
        // override is per head size: TOSH_FA_AMD_NSG_64, _128, _WIDE
        static const int nsg_env_64 = []() {
            const char * s = TOSH_ENV("TOSH_FA_AMD_NSG_64");
            return s ? atoi(s) : 0;
        }();
        static const int nsg_env_128 = []() {
            const char * s = TOSH_ENV("TOSH_FA_AMD_NSG_128");
            return s ? atoi(s) : 0;
        }();
        static const int nsg_env_wide = []() {
            const char * s = TOSH_ENV("TOSH_FA_AMD_NSG_WIDE");
            return s ? atoi(s) : 0;
        }();
        const int nsg_env = ne00 == 64 ? nsg_env_64 : ne00 == 128 ? nsg_env_128 : nsg_env_wide;
        if (nsg_env > 0) {
            fa_amd_nsg = nsg_env;
        }

        // one threadgroup per query head leaves most of the GPU idle on models with
        // few heads; split the cache across 32 and let upstream's reduce merge them
        // Head 128 needs a deeper cache than the wide heads before the 32-way reduce pays
        // for itself, so it carries its own threshold. The depth it turns at is not the same
        // on both widths, so each keeps its own.
        static const int64_t wg_dk128_env = []() {
            const char * s = TOSH_ENV("TOSH_FA_WG_DK128_MIN_KV");
            return s ? (int64_t) atoll(s) : (int64_t) -1;
        }();
        const bool wg_wide_head =
            ne00 == 256 || ne00 == 512 || (ne00 == 576 && ne20 == 512) ||
            (ne00 == 320 && ne20 == 256);
        const int64_t wg_dk128_min_kv = wg_dk128_env >= 0 ? wg_dk128_env
                                      : (props_dev->simd_width == 64 ? ggml_metal_fa_wg_dk128_min_kv_w64(ne02, ne12) : 24576);
        const bool wg_dk128_width_ok = props_dev->simd_width == 32 || props_dev->simd_width == 64;
        // TOSH_FA_WG_DK64=<kv> sets the head-64 depth for every shape, 0 closes it
        static const int64_t wg_dk64_env = []() {
            const char * s = TOSH_ENV("TOSH_FA_WG_DK64");
            return s ? (int64_t) atoll(s) : (int64_t) -1;
        }();
        const int64_t wg_dk64_min_kv = wg_dk64_env >= 0 ? wg_dk64_env
                                     : (props_dev->simd_width == 64 ? ggml_metal_fa_wg_dk64_min_kv_w64(ne02, ne12) : 0);
        const bool wg_dk64 = ne00 == 64 && ne20 == 64 && wg_dk128_width_ok &&
                             wg_dk64_min_kv > 0 && ne11 >= wg_dk64_min_kv;
        const bool wg_dk128 = (ne00 == 128 && ne20 == 128 && wg_dk128_width_ok &&
                               wg_dk128_min_kv > 0 && ne11 >= wg_dk128_min_kv) || wg_dk64;
        const int fa_amd_nwg =
            (wg_wide_head || wg_dk128) && ne01 == 1 && !has_sinks &&
            (props_dev->simd_width == 32 || props_dev->simd_width == 64) &&
            (wg_dk128 || ne11 >= 1024) &&
            TOSH_ENV("TOSH_FA_WG_DISABLE") == NULL ? 32 : 1;

        char fa_amd_wg_pipeline[96];
        if (fa_amd_nwg > 1) {
            snprintf(fa_amd_wg_pipeline, sizeof(fa_amd_wg_pipeline), "%s_wg32", fa_amd_pipeline);
        }

        // sparse: build the compacted index list first, on the same encoder
        ggml_metal_buffer_id bid_amd_idx = bid_tmp;
        if (amd_sparse) {
            bid_amd_idx.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op)
                              + ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

            ggml_metal_kargs_flash_attn_ext_vec_idx argsi = {
                /*.ne30            =*/ ne30,
                /*.ne31            =*/ ne31,
                /*.ne32            =*/ ne32,
                /*.ne33            =*/ ne33,
                /*.nb31            =*/ nb31,
                /*.nb32            =*/ nb32,
                /*.nb33            =*/ nb33,
                /*.n_kv_max        =*/ amd_n_kv_max,
                /*.n_kv_max_padded =*/ amd_n_kv_pad,
            };

            auto pipei = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipei);
            ggml_metal_encoder_set_bytes   (enc, &argsi, sizeof(argsi), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_amd_idx, 2);

            int nthi = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipei), 256);
            // whole simdgroups only: a partial one loses its slice of the count
            const int nwi = props_dev->simd_width;
            nthi = std::max(nwi, (nthi/nwi)*nwi);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nthi, 1, 1);
            ggml_metal_op_concurrency_reset(ctx);
        }

        auto pipeline = ggml_metal_fa_amd_pipeline(lib, fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline);
        // 32 KB of threadgroup memory (GCN 1.0) does not hold the 64-lane kernel; its twin
        // with half the simdgroups does. TOSH_FA_AMD_LDS32=1 takes the twin on any wave64 card.
        static const bool lds32_force = TOSH_ENV("TOSH_FA_AMD_LDS32") != NULL;
        if ((!pipeline.pipeline || lds32_force) && props_dev->simd_width == 64) {
            char lds_pipeline[112];
            snprintf(lds_pipeline, sizeof(lds_pipeline), "%s_lds32", fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline);
            auto lds = ggml_metal_fa_amd_pipeline(lib, lds_pipeline);
            if (lds.pipeline) {
                pipeline = lds;
                fa_amd_nsg /= 2;
            }
        }

        // dispatching more threads than the pipeline admits hangs the GPU, and how many
        // it admits depends on the K type's dequant; fall through rather than risk it
        if (!pipeline.pipeline) {
            // a missed instantiation leaves a null pipeline; fall through rather than crash
            GGML_LOG_WARN("%s: %s is not instantiated; using the generic kernels\n",
                          __func__, fa_amd_nwg > 1 ? fa_amd_wg_pipeline : fa_amd_pipeline);
        } else if (ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < fa_amd_nsg*props_dev->simd_width) {
            GGML_LOG_WARN("%s: %s wants %d threads, pipeline allows %d; using the generic kernels\n",
                          __func__, fa_amd_pipeline, fa_amd_nsg*props_dev->simd_width,
                          ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        } else {

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_src2, 3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, fa_amd_nwg > 1 ? bid_tmp : bid_dst, 7);
        ggml_metal_encoder_set_buffer  (enc, bid_amd_idx, 8);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03*fa_amd_nwg, props_dev->simd_width, fa_amd_nsg, 1);

        if (fa_amd_nwg > 1) {
            ggml_metal_op_concurrency_reset(ctx);

            ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = { (int32_t) (ne1*ne2*ne3) };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, fa_amd_nwg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne1*ne2*ne3, 1, 1, 32*fa_amd_nwg, 1, 1);
        }

        return 1;
        }
    }

    // FA-RDNA2 (kernels/fa_amd.metal) covers what the ToshLLM AMD kernels above did not take,
    // or all of it when TOSH_FA_AMD is unset; GGML_METAL_FA_AMD=0 turns both off.
    // Only on GPUs without simdgroup matrix (Apple7): that is the same condition supports_op
    // uses to allow FA-RDNA2, and it keeps Apple Silicon on the upstream tensor/vec kernels.
    const bool fa_rdna2_dev_ok = !props_dev->has_simdgroup_mm;
    if (fa_rdna2_dev_ok && ggml_metal_op_flash_attn_ext_amd_supported(op) && ggml_metal_op_flash_attn_ext_amd_align_ok(op)) {
        return ggml_metal_op_flash_attn_ext_amd(ctx, idx, scale);
    }
    // Quantized KV: dequant prepass to scratch F16, then fa_amd vec/tile (any nq)
    if (fa_rdna2_dev_ok && ggml_metal_op_flash_attn_ext_amd_quant_supported(op)) {
        return ggml_metal_op_flash_attn_ext_amd_quant(ctx, idx, scale);
    }

    // The generic Metal kernels below need K and V to share a type.
    GGML_ASSERT(op->src[1]->type == op->src[2]->type);

    ggml_metal_buffer_id bid_kv_f16 = bid_tmp;
    bid_kv_f16.offs += ggml_metal_op_flash_attn_ext_extra_tmp(op);

    // sparse path: gather the finite mask entries into index lists and run the vec kernels over them
    const int n_kv_max_sparse = ggml_metal_op_flash_attn_ext_n_kv_max_sparse(op);
    const bool use_sparse = n_kv_max_sparse > 0 && !ggml_metal_fa_amd_experiment();
    const int n_kv_max_padded = use_sparse ? GGML_PAD(n_kv_max_sparse, OP_FLASH_ATTN_EXT_VEC_NCPSG) : 0;

    const bool use_vec = ggml_metal_op_flash_attn_ext_use_vec(op);

    // with kv_rows, the non-vec path gathers the referenced rows into a contiguous F16 buffer and
    // attends over it with the F16 kernel - quantized KV is first dequantized with the F16 pass
    const bool rows_gather = has_rows && !use_vec;

    // the vec kernels dequantize the KV inline; no need for the F16 dequant pass in the sparse path
    const bool dequant_kv = !use_sparse && (rows_gather ?
        ggml_is_quantized(op->src[1]->type) : ggml_metal_op_flash_attn_ext_use_kv_f16(op));

    const bool use_kv_f16 = !use_sparse && (rows_gather ? true : dequant_kv);

    ggml_metal_buffer_id bid_idx = bid_kv_f16;
    bid_idx.offs += ggml_metal_op_flash_attn_ext_extra_kv_f16(op);

    // contiguous K/V buffers holding the rows referenced by the kv_rows lists
    ggml_metal_buffer_id bid_rows = bid_idx;
    bid_rows.offs += ggml_metal_op_flash_attn_ext_extra_idx(op);

    ggml_metal_buffer_id bid_k = bid_src1;
    ggml_metal_buffer_id bid_v = bid_src2;

    uint64_t nb10_attn = nb10;
    uint64_t nb11_attn = nb11;
    uint64_t nb12_attn = nb12;
    uint64_t nb13_attn = nb13;
    uint64_t nb20_attn = nb20;
    uint64_t nb21_attn = nb21;
    uint64_t nb22_attn = nb22;
    uint64_t nb23_attn = nb23;

    if (dequant_kv) {
        assert(ggml_metal_op_flash_attn_ext_extra_kv_f16(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        const int64_t nblocks1_64 = (ne10/ggml_blck_size(op->src[1]->type))*(int64_t) ne11*ne12*ne13;
        GGML_ASSERT(nblocks1_64 <= INT32_MAX);
        const int32_t nblocks1 = nblocks1_64;

        ggml_metal_buffer_id bid_v_f16 = bid_kv_f16;
        bid_v_f16.offs += ggml_metal_op_flash_attn_ext_kv_f16_k_size(op);

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);

        // K
        ggml_metal_kargs_flash_attn_ext_kv_f16 args_k = {
            /*.ne0    =*/ ne10,
            /*.ne1    =*/ ne11,
            /*.ne2    =*/ ne12,
            /*.ne3    =*/ ne13,
            /*.nb0    =*/ nb10,
            /*.nb1    =*/ nb11,
            /*.nb2    =*/ nb12,
            /*.nb3    =*/ nb13,
            /*.nblocks =*/ nblocks1,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args_k, sizeof(args_k), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src1,        1);
        ggml_metal_encoder_set_buffer  (enc, bid_kv_f16, 2);

        ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks1 + nth - 1)/nth, 1, 1, nth, 1, 1);

        // V (skip when V is a view of K: the dequantized V is a view of the dequantized K)
        if (!v_is_view_of_k) {
            const int64_t nblocks2_64 = (ne20/ggml_blck_size(op->src[2]->type))*(int64_t) ne21*ne22*ne23;
            GGML_ASSERT(nblocks2_64 <= INT32_MAX);
            const int32_t nblocks2 = nblocks2_64;

            ggml_metal_kargs_flash_attn_ext_kv_f16 args_v = {
                /*.ne0    =*/ ne20,
                /*.ne1    =*/ ne21,
                /*.ne2    =*/ ne22,
                /*.ne3    =*/ ne23,
                /*.nb0    =*/ nb20,
                /*.nb1    =*/ nb21,
                /*.nb2    =*/ nb22,
                /*.nb3    =*/ nb23,
                /*.nblocks =*/ nblocks2,
            };

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args_v, sizeof(args_v), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src2,        1);
            ggml_metal_encoder_set_buffer  (enc, bid_v_f16,       2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (nblocks2 + nth - 1)/nth, 1, 1, nth, 1, 1);
        }

        // the pad and attention kernels read the dequantized KV
        ggml_metal_op_concurrency_reset(ctx);

        bid_k = bid_kv_f16;
        bid_v = v_is_view_of_k ? bid_k : bid_v_f16;

        // contiguous F16 layout of the dequantized K
        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the dequantized V is read from the dequantized K with K's strides
        if (v_is_view_of_k) {
            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            // contiguous F16 layout of the dequantized V
            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne21;
            nb23_attn = nb22_attn*ne22;
        }
    }

    const bool fa_amd = ggml_metal_fa_amd_experiment();
    // with kv_rows, gather the cache rows referenced by the lists into contiguous F16 buffers
    // the attention then runs as a standard dense op over the gathered rows (tensor or half8x8 kernel)
    if (rows_gather) {
        assert(ggml_metal_op_flash_attn_ext_extra_rows(op) != 0);

        const bool v_is_view_of_k = ggml_metal_op_flash_attn_ext_v_is_view_of_k(op);

        GGML_ASSERT(ne20 % 4 == 0);

        const size_t k_size = GGML_PAD(sizeof(ggml_fp16_t)*(size_t) ne10*ne11_attn*ne12*ne03, 16);

        ggml_metal_kargs_flash_attn_ext_rows args0 = {
            /*.ne11    =*/ (int32_t) ne11_attn,
            /*.ne_12_2 =*/ ne12,
            /*.ne10    =*/ ne10,
            /*.ne20    =*/ v_is_view_of_k ? 0 : ne20, // the gathered V is a view of the gathered K
            /*.v_off   =*/ k_size,
            /*.nb11    =*/ nb11_attn,
            /*.nb12    =*/ nb12_attn,
            /*.nb21    =*/ nb21_attn,
            /*.nb22    =*/ nb22_attn,
        };

        auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_rows(lib, op);
        const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 64);

        ggml_metal_encoder_set_pipeline(enc, pipeline0);
        ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_k, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_v, 2);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(kv_rows), 3);
        ggml_metal_encoder_set_buffer  (enc, bid_rows, 4);

        ggml_metal_encoder_dispatch_threadgroups(enc, ne11_attn, ne03, 1, nth, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        // contiguous F16 layout of the gathered rows: [DK, n_rows, n_head_kv, n_streams]
        bid_k = bid_rows;

        nb10_attn = sizeof(ggml_fp16_t);
        nb11_attn = nb10_attn*ne10;
        nb12_attn = nb11_attn*ne11_attn;
        nb13_attn = nb12_attn*ne12;

        // if V is a view of K, the gathered V is read from the gathered K with K's strides
        if (v_is_view_of_k) {
            bid_v = bid_k;

            nb20_attn = nb10_attn;
            nb21_attn = nb11_attn;
            nb22_attn = nb12_attn;
            nb23_attn = nb13_attn;
        } else {
            bid_v = bid_rows;
            bid_v.offs += k_size;

            nb20_attn = sizeof(ggml_fp16_t);
            nb21_attn = nb20_attn*ne20;
            nb22_attn = nb21_attn*ne11_attn;
            nb23_attn = nb22_attn*ne22;
        }
    }

    if (!use_sparse && ggml_metal_op_flash_attn_ext_use_tensor(op, props_dev->has_tensor)) {
        // tensor API kernel
        const int nqptg = ne00 >= 512 ? OP_FLASH_ATTN_EXT_TENSOR_NQPSG_LARGE : OP_FLASH_ATTN_EXT_TENSOR_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_TENSOR_NCPSG; // cache values per threadgroup
        const int nsg   = OP_FLASH_ATTN_EXT_TENSOR_NSG;

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            ggml_metal_op_concurrency_reset(ctx);
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ has_rows ? ne03 : ne13, // the gathered K/V holds one copy per stream
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        // shared memory layout: queries (16-bit), scores (float), probabilities (16-bit), row scale (float), rescale flag (int)
        const size_t smem = GGML_PAD(nqptg*ne00*sizeof(ggml_fp16_t) + nqptg*ncpsg*(sizeof(float) + sizeof(ggml_fp16_t)) + nqptg*sizeof(float) + sizeof(int32_t), 16);

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_tensor(lib, op, has_mask, has_sinks, has_bias, has_scap);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  7);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else if (!use_sparse && (!ggml_metal_op_flash_attn_ext_use_vec(op) || fa_amd)) {
        // half8x8 kernel
        const int nqptg = OP_FLASH_ATTN_EXT_NQPSG; // queries per threadgroup
        const int ncpsg = OP_FLASH_ATTN_EXT_NCPSG; // cache values per simdgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg  % 8  == 0);
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = ne11_attn % ncpsg != 0;

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (has_mask) {
            assert(ggml_metal_op_flash_attn_ext_extra_blk(op) != 0);

            ggml_metal_kargs_flash_attn_ext_blk args0 = {
                /*.ne01 =*/ ne01,
                /*.ne30 =*/ ne30,
                /*.ne31 =*/ ne31,
                /*.ne32 =*/ ne32,
                /*.ne33 =*/ ne33,
                /*.nb31 =*/ nb31,
                /*.nb32 =*/ nb32,
                /*.nb33 =*/ nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_blk(lib, op, nqptg, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_blk,  2);

            const int32_t nblk1 = ((ne01 + nqptg - 1)/nqptg);
            const int32_t nblk0 = ((ne30 + ncpsg - 1)/ncpsg);

            ggml_metal_encoder_dispatch_threadgroups(enc, nblk0, nblk1, ne32*ne33, 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        const int is_q = !use_kv_f16 && ggml_is_quantized(op->src[1]->type) ? 1 : 0;

        // shared memory layout (halfs unless noted):
        //   queries/attn/result: Q*(DK + 2*PAD2(DV,64) + 4*C)
        //   quantized KV scratch: 16*32*NSG (only when is_q)
        const int64_t dv_pad = GGML_PAD(ne20, 64);

        auto fa_smem = [&](int32_t nsg) -> size_t {
            const size_t smem_half = nqptg*(ne00 + 2*dv_pad + 4*ncpsg) + is_q*(16*32*nsg);
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t), 16);
        };

        // simdgroups per threadgroup (a.k.a. warps)
        int32_t nsg = ne00 >= 512 ? 8 : 4;
        // RC2 AMD heuristic: more simdgroups help when the KV cache is deep
        if (ggml_metal_fa_amd_experiment() && nsg < 8 && ne11 >= 8*ncpsg &&
            fa_smem(8) <= props_dev->max_theadgroup_memory_size) {
            nsg = 8;
        }

        const size_t smem = fa_smem(nsg);

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ has_rows ? ne03 : ne13, // the gathered K/V holds one copy per stream
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, nsg, use_kv_f16, ns10, ns20);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, bid_pad,  6);
        ggml_metal_encoder_set_buffer  (enc, bid_blk,  7);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  8);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, ne02, ne03, 32, nsg, 1);
    } else {
        // half4x4 kernel
        // sparse: the index lists are per query row, so a threadgroup can share KV with Q == 1 only
        auto cfg = use_sparse
                ? ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20)
                : ggml_metal_tuning::fa_vec_pick(
                          props_dev->gpu_family,
                          (int) op->src[1]->type,
                          (int) ne00, (int) ne20,   // dk, dv (ne00 == dk for FA)
                          ne11, ne01);

        int nqptg = cfg.Q; // queries per threadgroup

        const int ncpsg = OP_FLASH_ATTN_EXT_VEC_NCPSG; // cache values per simdgroup !! sync with kernel template arguments !!
        const int nhptg = 1;                           // heads per threadgroup

        GGML_ASSERT(nqptg <= 32);
        GGML_ASSERT(nqptg == 1 || nqptg == 2 || nqptg == 4);  // only instantiated Q values
        GGML_ASSERT(ncpsg  % 32 == 0);

        bool need_sync = false;

        const bool has_kvpad = !use_sparse && ne11_attn % ncpsg != 0;

        if (use_sparse) {
            assert(ggml_metal_op_flash_attn_ext_extra_idx(op) != 0);

            GGML_ASSERT(ne30 == ne11);

            ggml_metal_kargs_flash_attn_ext_vec_idx args0 = {
                /*.ne30              =*/ ne30,
                /*.ne31              =*/ ne31,
                /*.ne32              =*/ ne32,
                /*.ne33              =*/ ne33,
                /*.nb31              =*/ nb31,
                /*.nb32              =*/ nb32,
                /*.nb33              =*/ nb33,
                /*.n_kv_max          =*/ n_kv_max_sparse,
                /*.n_kv_max_padded   =*/ n_kv_max_padded,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_idx,  2);

            int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline0), 256);
            // whole simdgroups only: a partial one loses its slice of the count
            const int nw0 = props_dev->simd_width;
            nth = std::max(nw0, (nth/nw0)*nw0);

            ggml_metal_encoder_dispatch_threadgroups(enc, ne31, ne32, ne33, nth, 1, 1);

            need_sync = true;
        }

        if (has_kvpad) {
            assert(ggml_metal_op_flash_attn_ext_extra_pad(op) != 0);

            ggml_metal_kargs_flash_attn_ext_pad args0 = {
                /*.ne11    =*/ne11,
                /*.ne_12_2 =*/ne12,
                /*.ne_12_3 =*/ne13,
                /*.nb11    =*/nb11_attn,
                /*.nb12    =*/nb12_attn,
                /*.nb13    =*/nb13_attn,
                /*.nb21    =*/nb21_attn,
                /*.nb22    =*/nb22_attn,
                /*.nb23    =*/nb23_attn,
                /*.ne31    =*/ne31,
                /*.ne32    =*/ne32,
                /*.ne33    =*/ne33,
                /*.nb31    =*/nb31,
                /*.nb32    =*/nb32,
                /*.nb33    =*/nb33,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_pad(lib, op, has_mask, ncpsg);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_k,    1);
            ggml_metal_encoder_set_buffer  (enc, bid_v,    2);
            ggml_metal_encoder_set_buffer  (enc, bid_src3, 3);
            ggml_metal_encoder_set_buffer  (enc, bid_pad,  4);

            assert(ne12 == ne22);
            assert(ne13 == ne23);

            ggml_metal_encoder_dispatch_threadgroups(enc, ncpsg, std::max(ne12, ne32), std::max(ne13, ne33), 32, 1, 1);

            need_sync = true;
        }

        if (need_sync) {
            ggml_metal_op_concurrency_reset(ctx);
        }

        // note: for simplicity assume the K is larger or equal than V
        GGML_ASSERT(ne10 >= ne20);

        // shared memory layout (halfs unless noted):
        //   queries:      Q*NSG*PAD2(ne00, 128)
        //   attn + mask:  NSG*4*Q*C
        //   results:      2*NSG*Q*PAD2(ne20, 128)
        //   sparse idx:   NSG*C ints (when use_sparse, or when gathering K/V rows)
        const int64_t dk_pad = GGML_PAD(ne00, 128);
        const int64_t dv_pad = GGML_PAD(ne20, 128);

        auto fa_vec_smem = [&](int64_t nsg, int32_t nqptg) -> size_t {
            const size_t smem_half = (size_t) (dk_pad + 4*ncpsg + 2*dv_pad)*nqptg*nsg;
            return GGML_PAD(smem_half*sizeof(ggml_fp16_t) + ((use_sparse || has_rows) ? (size_t) nsg*ncpsg*sizeof(int) : 0), 16);
        };

        int64_t nsg = 1;

        // workgroups
        // each workgroup handles nsg*nkpsg cache values
        int32_t nwg = 1;
        if (use_sparse) {
            if (ne01 > 32) {
                // large sparse batch
                nwg = 1;
                nsg = 1;
                if (n_kv_max_padded == 640) {
                    nsg = 4; // 640 % (4*32) == 0
                } else {
                    while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                        nsg *= 2;
                    }
                }
            } else {
                // small sparse batch
                nwg = 32;
                nsg = 1;
                while (2*nwg*nsg*ncpsg < n_kv_max_padded && nsg < 4) {
                    nsg *= 2;
                }
            }
        } else {
            nwg = 32;
            nsg = 1;
            while (2*nwg*nsg*ncpsg < ne11_attn && nsg < 4) {
                nsg *= 2;
            }
        }

        // fall back to baseline (Q=1) if the tuned config exceeds threadgroup memory
        if (fa_vec_smem(nsg, nqptg) > props_dev->max_theadgroup_memory_size) {
            cfg   = ggml_metal_tuning::fa_vec_baseline_cfg((int) ne00, (int) ne20);
            nqptg = cfg.Q;  // = 1
        }

        const int32_t ns10 = nb11_attn/nb10_attn;
        const int32_t ns20 = nb21_attn/nb20_attn;

        ggml_metal_kargs_flash_attn_ext_vec args = {
            /*.ne01          =*/ ne01,
            /*.ne02          =*/ ne02,
            /*.ne03          =*/ ne03,
            /*.nb01          =*/ nb01,
            /*.nb02          =*/ nb02,
            /*.nb03          =*/ nb03,
            /*.ne11          =*/ use_sparse ? n_kv_max_padded : (int32_t) ne11_attn,
            /*.ne_12_2       =*/ ne12,
            /*.ne_12_3       =*/ ne13,
            /*.ns10          =*/ ns10,
            /*.nb11          =*/ nb11_attn,
            /*.nb12          =*/ nb12_attn,
            /*.nb13          =*/ nb13_attn,
            /*.ns20          =*/ ns20,
            /*.nb21          =*/ nb21_attn,
            /*.nb22          =*/ nb22_attn,
            /*.nb23          =*/ nb23_attn,
            /*.ne31          =*/ ne31,
            /*.ne32          =*/ ne32,
            /*.ne33          =*/ ne33,
            /*.nb31          =*/ nb31,
            /*.nb32          =*/ nb32,
            /*.nb33          =*/ nb33,
            /*.ne1           =*/ ne1,
            /*.ne2           =*/ ne2,
            /*.ne3           =*/ ne3,
            /*.scale         =*/ scale,
            /*.max_bias      =*/ max_bias,
            /*.m0            =*/ m0,
            /*.m1            =*/ m1,
            /*.n_head_log2   =*/ n_head_log2,
            /*.logit_softcap =*/ logit_softcap,
            /*.n_kv_max_padded =*/ n_kv_max_padded,
        };

        auto pipeline = ggml_metal_library_get_pipeline_flash_attn_ext_vec(lib, op, has_mask, has_sinks, has_bias, has_scap, has_kvpad, use_sparse, has_rows, nqptg, cfg.NE, nsg, nwg, use_kv_f16, ns10, ns20);

        GGML_ASSERT(nsg*32 <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_k,    2);
        ggml_metal_encoder_set_buffer  (enc, bid_v,    3);
        ggml_metal_encoder_set_buffer  (enc, bid_src3, 4);
        ggml_metal_encoder_set_buffer  (enc, bid_src4, 5);
        ggml_metal_encoder_set_buffer  (enc, use_sparse ? bid_idx : (has_rows ? ggml_metal_get_buffer_id(kv_rows) : bid_src0), 8);

        const size_t smem = fa_vec_smem(nsg, nqptg);

        GGML_ASSERT(smem <= props_dev->max_theadgroup_memory_size);

        if (nwg == 1) {
            // using 1 workgroup -> write the result directly into dst
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_dst, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);
        } else {
            // sanity checks
            assert(ggml_metal_op_flash_attn_ext_extra_tmp(op) != 0);

            GGML_ASSERT(ne01*ne02*ne03 == ne1*ne2*ne3);
            GGML_ASSERT((uint64_t)ne1*ne2*ne3 <= (1u << 31));

            // write the results from each workgroup into a temp buffer
            ggml_metal_encoder_set_buffer(enc, bid_pad, 6);
            ggml_metal_encoder_set_buffer(enc, bid_tmp, 7);

            ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
            ggml_metal_encoder_dispatch_threadgroups(enc, (ne01 + nqptg - 1)/nqptg, (ne02 + nhptg - 1)/nhptg, ne03*nwg, 32, nsg, 1);

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            // reduce the results from the workgroups
            {
                const int32_t nrows = ne1*ne2*ne3;

                ggml_metal_kargs_flash_attn_ext_vec_reduce args0 = {
                    nrows,
                };

                auto pipeline0 = ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(lib, op, ne20, nwg);

                ggml_metal_encoder_set_pipeline(enc, pipeline0);
                ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

                ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, 32*nwg, 1, 1);
            }
        }
    }

    return 1;
}

int ggml_metal_op_bin(ggml_metal_op_t ctx, int idx) {
    {
        ggml_tensor * op = ctx->node(idx);
        ggml_metal_buffer_id bid_peer;
        if (op->op == GGML_OP_ADD && ggml_metal_fb_residual_take(ctx->dev, op->src[0], &bid_peer)) {
            // the norm that reads the sum and its weight can run in the same kernel when nothing
            // else reads the unweighted norm
            static int fuse_rms = -1;
            if (fuse_rms < 0) { const char * e = getenv("TOSH_MGPU_FB_RMS"); fuse_rms = (e && e[0] == '0') ? 0 : 1; }
            ggml_tensor * rms = idx + 2 < ctx->n_nodes() ? ctx->node(idx + 1) : nullptr;
            ggml_tensor * mul = idx + 2 < ctx->n_nodes() ? ctx->node(idx + 2) : nullptr;
            bool fuse = fuse_rms && rms != nullptr && mul != nullptr &&
                        rms->op == GGML_OP_RMS_NORM && rms->src[0] == op && ggml_is_contiguous(rms) &&
                        mul->op == GGML_OP_MUL && mul->src[0] == rms && ggml_is_contiguous(mul) &&
                        mul->type == GGML_TYPE_F32 && mul->src[1]->type == GGML_TYPE_F32 &&
                        ggml_is_contiguous(mul->src[1]) && mul->src[1]->ne[0] == op->ne[0] && ggml_nrows(mul->src[1]) == 1 &&
                        ggml_are_same_shape(mul, op) && op->ne[0] % 4 == 0;
            for (int j = idx + 3; fuse && j < ctx->n_nodes(); j++) {
                const ggml_tensor * nd = ctx->node(j);
                for (int k = 0; k < GGML_MAX_SRC && fuse; k++) {
                    for (const ggml_tensor * sx = nd->src[k]; sx != nullptr; sx = sx->view_src) {
                        if (sx == rms) { fuse = false; break; }
                    }
                }
            }
            auto prms = fuse ? ggml_metal_device_pipeline_add3_rms_mul(ctx->dev) : ggml_metal_pipeline_with_params{};
            if (prms.pipeline != nullptr) {
                const uint32_t n4   = (uint32_t) (op->ne[0]/4);
                const uint32_t ne00 = (uint32_t) op->ne[0];
                float eps;
                memcpy(&eps, rms->op_params, sizeof(float));
                const int nmax = ggml_metal_pipeline_max_theads_per_threadgroup(prms);
                int nth = 32;
                while (nth < (int) n4 && nth < nmax) {
                    nth *= 2;
                }
                nth = std::min(nth, nmax);
                nth = std::min(nth, ((int) n4 + 31)/32*32);
                ggml_metal_encoder_set_pipeline(ctx->enc, prms);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 0);
                ggml_metal_encoder_set_buffer  (ctx->enc, bid_peer, 1);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op->src[1]), 2);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op), 3);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(mul->src[1]), 4);
                ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(mul), 5);
                ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &n4,   sizeof(n4),   6);
                ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &eps,  sizeof(eps),  7);
                ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &ne00, sizeof(ne00), 8);
                ggml_metal_encoder_set_threadgroup_memory_size(ctx->enc, 64*sizeof(float), 0);
                ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (int) ggml_nrows(op), 1, 1, nth, 1, 1);
                return 3;
            }
            const uint32_t n4 = (uint32_t) (ggml_nelements(op)/4);
            ggml_metal_encoder_set_pipeline(ctx->enc, ggml_metal_device_pipeline_add3(ctx->dev));
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op->src[0]), 0);
            ggml_metal_encoder_set_buffer  (ctx->enc, bid_peer, 1);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op->src[1]), 2);
            ggml_metal_encoder_set_buffer  (ctx->enc, ggml_metal_get_buffer_id(op), 3);
            ggml_metal_encoder_set_bytes   (ctx->enc, (void *) &n4, sizeof(n4), 4);
            ggml_metal_encoder_dispatch_threadgroups(ctx->enc, (n4 + 255)/256, 1, 1, 256, 1, 1);
            return 1;
        }
    }

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = nullptr;

    const bool use_fusion = ctx->use_fusion();

    if (use_fusion) {
        int n = 1;
        fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);
        n_fuse = n;

        // snake activation autofuse: mul -> sin -> sqr -> mul -> add
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_SNAKE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_snake_fused(ctx, idx);
        }

        // MoE output reduction: experts * weights -> weighted sum
        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_MOE_REDUCE) {
            ctx->count_fusions(fusion);
            return ggml_metal_op_moe_reduce(ctx, idx);
        }
    }

    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[1]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_src1 = ggml_metal_get_buffer_id(op->src[1]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne10 =*/ ne10,
        /*.ne11 =*/ ne11,
        /*.ne12 =*/ ne12,
        /*.ne13 =*/ ne13,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_src1.offs },
    };

    // c[0] = add(a,    b[0])
    // c[1] = add(c[0], b[1])
    // c[2] = add(c[1], b[2])
    // ...
    if (use_fusion && fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_ADD_CHAIN) {
        // the offsets of the fused addends are relative to the start of the src1 buffer
        for (int i = 1; i < n_fuse; i++) {
            args.o1[i] = ggml_metal_get_buffer_id(ctx->node(idx + i)->src[1]).offs;
        }

        ctx->count_fusions(fusion);

        if (debug_fusion > 1) {
            GGML_LOG_DEBUG("%s: fuse: ADD x %d\n", __func__, n_fuse);
        }
    }

    // the offsets of src1 and all fused buffers are relative to the start of the src1 buffer
    bid_src1.offs = 0;

    struct ggml_metal_pipeline_with_params pipeline;

    pipeline = ggml_metal_library_get_pipeline_bin(lib, op, n_fuse);

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne10 = ne10/4;
        args.ne0  = ne0/4;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_src1, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  3);

    if (pipeline.cnt) {
        const int nth = MIN(256, args.ne0);

        ggml_metal_encoder_dispatch_threadgroups(enc, (args.ne0 + nth - 1)/nth, ggml_nrows(op), 1, nth, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;

        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return n_fuse;
}

int ggml_metal_op_silu_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_silu_back(lib, op);

    const int64_t ne = ggml_nelements(op);

    ggml_metal_kargs_silu_back args = {
        /*.ne =*/ ne,
    };

    int arg_idx{0};

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), arg_idx++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), arg_idx++);

    const int nth = std::min<int64_t>(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne);
    const int64_t n = (ne + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_l2_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_l2_norm args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.eps   =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_l2_norm(lib, op);

    if (pipeline.c4) {
        args.ne00 = ne00/4;
        args.ne0  = ne0/4;
    }

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_group_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t ngrp = ((const int32_t *) op->op_params)[0];

    float eps;
    memcpy(&eps, op->op_params + 1, sizeof(float));

    ggml_metal_kargs_group_norm args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.ngrp =*/ ngrp,
        /*.eps  =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_group_norm(lib, op);

    int nth = 32; // SIMD width
    //while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
    //    nth *= 2;
    //}

    //nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    //nth = std::min(nth, ne00/4);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ngrp, 1, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_norm(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool use_fusion = ctx->use_fusion();

    const int debug_fusion = ggml_metal_fusion_info_debug(ctx->finfo);

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_kargs_norm args = {
        /*.ne00   =*/ ne00,
        /*.ne00_t =*/ ne00 % 4 == 0 ? ne00/4 : ne00,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
        /*.nef1   =*/ { ne01 },
        /*.nef2   =*/ { ne02 },
        /*.nef3   =*/ { ne03 },
        /*.nbf1   =*/ { nb01 },
        /*.nbf2   =*/ { nb02 },
        /*.nbf3   =*/ { nb03 },
        /*.scale =*/ 1.0f,
    };

    int n_fuse = 1;
    bool fused_norm_scale = false;

    ggml_metal_buffer_id bid_fuse[2] = { bid_src0, bid_src0 };

    // d[0] = norm(a)
    // d[1] = mul(d[0], b) or scale(d[0])
    // d[2] = add(d[1], c)
    if (use_fusion) {
        int n = 1;
        const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n);

        if (fusion && (ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL || ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_MUL_ADD)) {
            n_fuse = n;

            ctx->count_fusions(fusion);

            for (int i = 1; i < n_fuse; i++) {
                const ggml_tensor * fn = ctx->node(idx + i);
                const ggml_tensor * fs = fn->src[0] == ctx->node(idx + i - 1) ? fn->src[1] : fn->src[0];

                bid_fuse[i - 1] = ggml_metal_get_buffer_id(fs);

                args.nef1[i] = fs->ne[1];
                args.nef2[i] = fs->ne[2];
                args.nef3[i] = fs->ne[3];

                args.nbf1[i] = fs->nb[1];
                args.nbf2[i] = fs->nb[2];
                args.nbf3[i] = fs->nb[3];
            }

            if (debug_fusion > 1) {
                if (n_fuse == 2) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL\n", __func__, ggml_op_name(op->op));
                }
                if (n_fuse == 3) {
                    GGML_LOG_DEBUG("%s: fuse: %s + MUL + ADD\n", __func__, ggml_op_name(op->op));
                }
            }
        }

        if (fusion && ggml_metal_fusion_get_id(fusion) == GGML_METAL_FUSION_NORM_SCALE) {
            n_fuse = n;
            fused_norm_scale = true;

            ctx->count_fusions(fusion);

            const ggml_tensor * scale_node = ctx->node(idx + 1);
            args.scale = ggml_get_op_params_f32(scale_node, 0);

            if (debug_fusion > 1) {
                GGML_LOG_DEBUG("%s: fuse: %s + SCALE\n", __func__, ggml_op_name(op->op));
            }
        }
    }

    if (n_fuse > 1) {
        bid_dst = ggml_metal_get_buffer_id(ctx->node(idx + n_fuse - 1));

        for (int i = 1; i < n_fuse; ++i) {
            if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
                ggml_metal_op_concurrency_reset(ctx);

                break;
            }
        }
    }

    auto pipeline = fused_norm_scale ?
        ggml_metal_library_get_pipeline_norm_scale(lib, op) :
        ggml_metal_library_get_pipeline_norm(lib, op, n_fuse);

    int nth = 32; // SIMD width

    while (nth < args.ne00_t && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, (args.ne00_t + 31)/32*32);

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0,    1);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[0], 2);
    ggml_metal_encoder_set_buffer  (enc, bid_fuse[1], 3);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return n_fuse;
}

int ggml_metal_op_rope(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // make sure we have one or more position id(ne10) per token(ne02)
    GGML_ASSERT(ne10 % ne02 == 0);
    GGML_ASSERT(ne10 >= ne02);

    const int nth = std::min(1024, ne00);

    const int n_past     = ((const int32_t *) op->op_params)[0];
    const int n_dims     = ((const int32_t *) op->op_params)[1];
  //const int mode       = ((const int32_t *) op->op_params)[2];
    // skip 3, n_ctx, used in GLM RoPE, unimplemented in metal
    const int n_ctx_orig = ((const int32_t *) op->op_params)[4];

    float freq_base;
    float freq_scale;
    float ext_factor;
    float attn_factor;
    float beta_fast;
    float beta_slow;

    memcpy(&freq_base,   (const int32_t *) op->op_params +  5, sizeof(float));
    memcpy(&freq_scale,  (const int32_t *) op->op_params +  6, sizeof(float));
    memcpy(&ext_factor,  (const int32_t *) op->op_params +  7, sizeof(float));
    memcpy(&attn_factor, (const int32_t *) op->op_params +  8, sizeof(float));
    memcpy(&beta_fast,   (const int32_t *) op->op_params +  9, sizeof(float));
    memcpy(&beta_slow,   (const int32_t *) op->op_params + 10, sizeof(float));

    // mrope
    const int sect_0 = ((const int32_t *) op->op_params)[11];
    const int sect_1 = ((const int32_t *) op->op_params)[12];
    const int sect_2 = ((const int32_t *) op->op_params)[13];
    const int sect_3 = ((const int32_t *) op->op_params)[14];

    const int n_offs = ((const int32_t *) op->op_params)[15];

    // when dst aliases src0, the channels outside the rotated window already hold the correct data
    const bool inplace = op->data == op->src[0]->data;

    ggml_metal_kargs_rope args = {
        /*.ne00        =*/ ne00,
        /*.ne01        =*/ ne01,
        /*.ne02        =*/ ne02,
        /*.ne03        =*/ ne03,
        /*.nb00        =*/ nb00,
        /*.nb01        =*/ nb01,
        /*.nb02        =*/ nb02,
        /*.nb03        =*/ nb03,
        /*.ne0         =*/ ne0,
        /*.ne1         =*/ ne1,
        /*.ne2         =*/ ne2,
        /*.ne3         =*/ ne3,
        /*.nb0         =*/ nb0,
        /*.nb1         =*/ nb1,
        /*.nb2         =*/ nb2,
        /*.nb3         =*/ nb3,
        /*.n_past      =*/ n_past,
        /*.n_dims      =*/ n_dims,
        /*.n_offs      =*/ n_offs,
        /*.n_ctx_orig  =*/ n_ctx_orig,
        /*.freq_base   =*/ freq_base,
        /*.freq_scale  =*/ freq_scale,
        /*.ext_factor  =*/ ext_factor,
        /*.attn_factor =*/ attn_factor,
        /*.beta_fast   =*/ beta_fast,
        /*.beta_slow   =*/ beta_slow,
        /* sect_0      =*/ sect_0,
        /* sect_1      =*/ sect_1,
        /* sect_2      =*/ sect_2,
        /* sect_3      =*/ sect_3,
        /* src2        =*/ op->src[2] != nullptr,
        /* inplace     =*/ inplace,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rope(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    if (op->src[2]) {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    } else {
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 3);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_im2col(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];
    const int32_t p1 = ((const int32_t *)(op->op_params))[3];
    const int32_t d0 = ((const int32_t *)(op->op_params))[4];
    const int32_t d1 = ((const int32_t *)(op->op_params))[5];

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;

    const int32_t N  = op->src[1]->ne[is_2D ? 3 : 2];
    const int32_t IC = op->src[1]->ne[is_2D ? 2 : 1];
    const int32_t IH = is_2D ? op->src[1]->ne[1] : 1;
    const int32_t IW =         op->src[1]->ne[0];

    const int32_t KH = is_2D ? op->src[0]->ne[1] : 1;
    const int32_t KW =         op->src[0]->ne[0];

    const int32_t OH = is_2D ? op->ne[2] : 1;
    const int32_t OW =         op->ne[1];

    const int32_t CHW = IC * KH * KW;

    const uint64_t ofs0 = op->src[1]->nb[is_2D ? 3 : 2] / 4;
    const uint64_t ofs1 = op->src[1]->nb[is_2D ? 2 : 1] / 4;

    ggml_metal_kargs_im2col args = {
        /*.ofs0 =*/ ofs0,
        /*.ofs1 =*/ ofs1,
        /*.IW   =*/ IW,
        /*.IH   =*/ IH,
        /*.CHW  =*/ CHW,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.N    =*/ N,
        /*.KH   =*/ KH,
        /*.KW   =*/ KW,
        /*.KHW  =*/ KH * KW,
    };

    auto pipeline = ggml_metal_library_get_pipeline_im2col(lib, op);

    if (KH*KW <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        const uint64_t ntptg0 = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)/(KH*KW), N);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, IC, OH, OW, ntptg0, KH, KW);
    } else {
        const uint64_t n_threads = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), N);
        const int64_t  quotient  = N / n_threads + (N % n_threads > 0 ? 1 : 0);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

        ggml_metal_encoder_dispatch_threadgroups(enc, quotient * CHW, OH, OW, n_threads, 1, 1);
    }

    return 1;
}

// number of reduction splits of the conv_2d matmul kernels
// A small output with a long reduction leaves the GPU idle, so the reduction is split until there are about 1M partial sums.
static int ggml_metal_op_conv_2d_nsplit(const ggml_tensor * op, int32_t * nk) {
    const int64_t K      = op->src[0]->ne[0]*op->src[0]->ne[1]*op->src[0]->ne[2];
    const int64_t nsteps = (K + 31)/32;
    const int64_t ne     = ggml_nelements(op);

    // at least 3 reduction steps of 32 per split, at most 32 splits
    const int64_t nsplit = std::max<int64_t>(1, std::min<int64_t>({ (1024*1024 + ne - 1)/ne, nsteps/3, 32 }));
    const int64_t steps  = (nsteps + nsplit - 1)/nsplit;

    if (nk) {
        *nk = (int32_t) (steps*32);
    }

    return (int) ((nsteps + steps - 1)/steps);
}

size_t ggml_metal_op_conv_2d_extra_tmp(const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    const int nsplit = ggml_metal_op_conv_2d_nsplit(op, nullptr);

    // partial sums of the reduction splits
    return nsplit > 1 ? nsplit*ggml_nelements(op)*sizeof(float) : 0;
}

int ggml_metal_op_conv_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.IC   =*/ ne02,
        /*.OC   =*/ ne03,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
        /*.nk   =*/ 0,
    };

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx->dev);

    // the matmul kernels walk the patches with 32-bit byte offsets inside an image
    // larger images, possibly a strided view, take the generic kernel
    const uint64_t max_off = (ne12 - 1)*nb12 + (ne11 - 1)*nb11 + (ne10 - 1)*nb10 + sizeof(float);
    const bool     fits_32 = max_off <= UINT32_MAX;

    if (props_dev->has_simdgroup_mm && fits_32) {
        auto pipeline = ggml_metal_library_get_pipeline_conv_2d_mm(lib, op);

        const int nr0 = pipeline.nr0;
        const int nr1 = pipeline.nr1;
        const int nsg = pipeline.nsg;

        const int64_t NP = ne3*ne1*ne0;
        const int64_t OC = ne03;

        const int nsplit = ggml_metal_op_conv_2d_nsplit(op, &args.nk);

        ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
        ggml_metal_buffer_id bid_tmp = bid_dst;
        bid_tmp.offs += ggml_nbytes(op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer  (enc, nsplit > 1 ? bid_tmp : bid_dst,       3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

        if (props_dev->has_tensor) {
            // per image: nr0 positions times nr1 channels per threadgroup
            const int64_t OHW = ne1*ne0;
            ggml_metal_encoder_dispatch_threadgroups(enc, (OC + nr1 - 1)/nr1, (OHW + nr0 - 1)/nr0, ne3*nsplit, 32, nsg, 1);
        } else {
            ggml_metal_encoder_dispatch_threadgroups(enc, (NP + nr1 - 1)/nr1, (OC + nr0 - 1)/nr0, nsplit, 32, nsg, 1);
        }

        if (nsplit > 1) {
            GGML_ASSERT(ggml_is_contiguous(op));

            // sync the 2 kernels
            ggml_metal_op_concurrency_reset(ctx);

            ggml_metal_kargs_conv_2d_reduce args0 = {
                /*.ne     =*/ (int32_t) ggml_nelements(op),
                /*.nsplit =*/ nsplit,
            };

            auto pipeline0 = ggml_metal_library_get_pipeline_conv_2d_mm_reduce(lib, op);

            ggml_metal_encoder_set_pipeline(enc, pipeline0);
            ggml_metal_encoder_set_bytes   (enc, &args0, sizeof(args0), 0);
            ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
            ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

            ggml_metal_encoder_dispatch_threadgroups(enc, (args0.ne + 255)/256, 1, 1, 256, 1, 1);
        }

        return 1;
    }

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d(lib, op);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const uint64_t n_out = ggml_nelements(op);

    uint64_t tg = (n_out + nth - 1)/nth;
    tg = std::max<uint64_t>(tg, 1);
    tg = std::min<uint64_t>(tg, (uint64_t) std::numeric_limits<int>::max());

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_2d_dw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);

    const int32_t s0 = ((const int32_t *) op->op_params)[0];
    const int32_t s1 = ((const int32_t *) op->op_params)[1];
    const int32_t p0 = ((const int32_t *) op->op_params)[2];
    const int32_t p1 = ((const int32_t *) op->op_params)[3];
    const int32_t d0 = ((const int32_t *) op->op_params)[4];
    const int32_t d1 = ((const int32_t *) op->op_params)[5];

    ggml_metal_kargs_conv_2d_dw args = {
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb03,
        /*.nb10 =*/ nb10,
        /*.nb11 =*/ nb11,
        /*.nb12 =*/ nb12,
        /*.nb13 =*/ nb13,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.IW   =*/ ne10,
        /*.IH   =*/ ne11,
        /*.KW   =*/ ne00,
        /*.KH   =*/ ne01,
        /*.C    =*/ ne12,
        /*.OW   =*/ ne0,
        /*.OH   =*/ ne1,
        /*.N    =*/ ne13,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.p0   =*/ p0,
        /*.p1   =*/ p1,
        /*.d0   =*/ d0,
        /*.d1   =*/ d1,
    };

    const bool use_tiled = (nb12 < nb10);

    auto pipeline = ggml_metal_library_get_pipeline_conv_2d_dw(lib, op, use_tiled);

    int nth = ggml_metal_pipeline_max_theads_per_threadgroup(pipeline);
    nth = std::min(nth, 256);
    nth = std::max(nth, 1);

    const int32_t OW = ne0;
    const int32_t OH = ne1;
    const int32_t C  = ne12;
    const int32_t N  = ne13;

    const int tg_x = use_tiled ? (C + nth - 1) / nth : (OW + nth - 1) / nth;
    const int tg_y = OH;
    const int tg_z = use_tiled ? OW * N : C * N;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, tg_x, tg_y, tg_z, nth, 1, 1);

    return 1;
}

int ggml_metal_op_conv_3d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    // 1. Extract standard dimensions and byte strides
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    // 2. Extract hyperparams from op_params
    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t s1 = ((const int32_t *)(op->op_params))[1];
    const int32_t s2 = ((const int32_t *)(op->op_params))[2];
    const int32_t p0 = ((const int32_t *)(op->op_params))[3];
    const int32_t p1 = ((const int32_t *)(op->op_params))[4];
    const int32_t p2 = ((const int32_t *)(op->op_params))[5];
    const int32_t d0 = ((const int32_t *)(op->op_params))[6];
    const int32_t d1 = ((const int32_t *)(op->op_params))[7];
    const int32_t d2 = ((const int32_t *)(op->op_params))[8];
    const int32_t IC = ((const int32_t *)(op->op_params))[9];
    const int32_t N  = ((const int32_t *)(op->op_params))[10];
    const int32_t OC = ((const int32_t *)(op->op_params))[11];

    // 3. Build the parameter struct using the macro-generated variables
    ggml_metal_kargs_conv_3d args = {
        /*.IW =*/ (int32_t)op->src[1]->ne[0],
        /*.IH =*/ (int32_t)op->src[1]->ne[1],
        /*.ID =*/ (int32_t)op->src[1]->ne[2],
        /*.OW =*/ (int32_t)op->ne[0],
        /*.OH =*/ (int32_t)op->ne[1],
        /*.OD =*/ (int32_t)op->ne[2],
        /*.KW =*/ (int32_t)op->src[0]->ne[0],
        /*.KH =*/ (int32_t)op->src[0]->ne[1],
        /*.KD =*/ (int32_t)op->src[0]->ne[2],
        s0, s1, s2,
        p0, p1, p2,
        d0, d1, d2,
        IC, N, OC,
        nb00, nb01, nb02, nb03, // Weight strides
        nb10, nb11, nb12, nb13, // Input strides
        nb0,  nb1,  nb2,  nb3   // Output strides
    };

    // 4. Fetch the JIT pipeline
    auto pipeline = ggml_metal_library_get_pipeline_conv_3d(lib, op);

    // 5. Grid mapping
    int nth0 = 32; // Standard SIMD width for Apple Silicon
    int nth1 = 1;
    int nth2 = 1;

    int64_t spatial_volume = args.OW * args.OH * args.OD;

    int ntg0 = (spatial_volume + nth0 - 1) / nth0;
    int ntg1 = args.OC;
    int ntg2 = args.N;

    // 6. Bind and Dispatch via the ggml C wrapper
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg0, ntg1, ntg2, nth0, nth1, nth2);

    return 1;
}

int ggml_metal_op_conv_transpose_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[1];
    const int32_t IL = op->src[1]->ne[0];

    const int32_t K  = op->src[0]->ne[0];

    const int32_t OL = op->ne[0];
    const int32_t OC = op->ne[1];

    ggml_metal_kargs_conv_transpose_1d args = {
        /*.IC  =*/ IC,
        /*.IL  =*/ IL,
        /*.K   =*/ K,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_1d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_dispatch_threadgroups(enc, OL, OC, 1, 1, 1, 1);

    return 1;
}

int ggml_metal_op_col2im_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];
    const int32_t OC = ((const int32_t *)(op->op_params))[1];
    const int32_t p0 = ((const int32_t *)(op->op_params))[2];

    const int32_t K_OC  = (int32_t) op->src[0]->ne[0];
    const int32_t T_in  = (int32_t) op->src[0]->ne[1];
    const int32_t K     = K_OC / OC;
    const int32_t T_out = (int32_t) op->ne[0];

    ggml_metal_kargs_col2im_1d args = {
        /*.T_in  =*/ T_in,
        /*.T_out =*/ T_out,
        /*.OC    =*/ OC,
        /*.K     =*/ K,
        /*.K_OC  =*/ K_OC,
        /*.s0    =*/ s0,
        /*.p0    =*/ p0,
    };

    auto pipeline = ggml_metal_library_get_pipeline_col2im_1d(lib, op);

    const int total = T_out * OC;
    const int nth   = 256;
    const int ntg   = (total + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 1;
}

// Dispatch the fused snake kernel from the matched mul -> sin -> sqr -> mul -> add chain.
// idx points at the leading mul. The caller has validated the chain.
int ggml_metal_op_snake_fused(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * mul0 = ctx->node(idx + 0);
    const ggml_tensor * sqr  = ctx->node(idx + 2);
    const ggml_tensor * mul1 = ctx->node(idx + 3);
    ggml_tensor *       add  = ctx->node(idx + 4);

    const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
    const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];
    const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

    const int T     = (int) x->ne[0];
    const int C     = (int) x->ne[1];
    const int total = T * C;

    // the encode loop pre-checked the leading mul only, check the rest of the chain
    for (int i = 1; i < 5; ++i) {
        if (!ggml_metal_op_concurrency_check(ctx, ctx->node(idx + i))) {
            ggml_metal_op_concurrency_reset(ctx);

            break;
        }
    }

    auto pipeline = ggml_metal_library_get_pipeline_snake(lib, x->type);

    ggml_metal_kargs_snake args = {
        /*.T =*/ T,
        /*.C =*/ C,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(x),     1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(a),     2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(inv_b), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(add),   4);

    const int nth = 256;
    const int ntg = (total + nth - 1) / nth;
    ggml_metal_encoder_dispatch_threadgroups(enc, ntg, 1, 1, nth, 1, 1);

    return 5;
}

int ggml_metal_op_conv_transpose_2d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ((const int32_t *)(op->op_params))[0];

    const int32_t IC = op->src[1]->ne[2];
    const int32_t IH = op->src[1]->ne[1];
    const int32_t IW = op->src[1]->ne[0];

    const int32_t KH = op->src[0]->ne[1];
    const int32_t KW = op->src[0]->ne[0];

    const int32_t OW = op->ne[0];
    const int32_t OH = op->ne[1];
    const int32_t OC = op->ne[2];
    const int32_t N  = op->src[1]->ne[3];

    ggml_metal_kargs_conv_transpose_2d args = {
        /*.IC  =*/ IC,
        /*.IH  =*/ IH,
        /*.IW  =*/ IW,
        /*.KH  =*/ KH,
        /*.KW  =*/ KW,
        /*.OC  =*/ OC,
        /*.s0  =*/ s0,
        /*.nb0 =*/ nb0,
        /*.nb1 =*/ nb1,
        /*.nb2 =*/ nb2,
        /*.nb3 =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_conv_transpose_2d(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    // Metal requires buffer size to be multiple of 16 bytes
    const size_t smem = GGML_PAD(KW * KH * sizeof(float), 16);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, OW, OH, OC * N, KW, KH, 1);

    return 1;
}

int ggml_metal_op_upscale(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float sf0 = (float)ne0/op->src[0]->ne[0];
    float sf1 = (float)ne1/op->src[0]->ne[1];
    float sf2 = (float)ne2/op->src[0]->ne[2];
    float sf3 = (float)ne3/op->src[0]->ne[3];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);

    float poffs = 0.5f;

    if (mode_flags & GGML_SCALE_FLAG_ALIGN_CORNERS) {
        poffs = 0.0f;
        sf0 = ne0 > 1 && ne00 > 1 ? (float)(ne0 - 1) / (ne00 - 1) : sf0;
        sf1 = ne1 > 1 && ne01 > 1 ? (float)(ne1 - 1) / (ne01 - 1) : sf1;
    }

    ggml_metal_kargs_upscale args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
        /*.sf0   =*/ sf0,
        /*.sf1   =*/ sf1,
        /*.sf2   =*/ sf2,
        /*.sf3   =*/ sf3,
        /*.poffs =*/ poffs,
    };

    auto pipeline = ggml_metal_library_get_pipeline_upscale(lib, op);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_roll(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int32_t s0 = ggml_get_op_params_i32(op, 0);
    const int32_t s1 = ggml_get_op_params_i32(op, 1);
    const int32_t s2 = ggml_get_op_params_i32(op, 2);
    const int32_t s3 = ggml_get_op_params_i32(op, 3);

    ggml_metal_kargs_roll args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.s0   =*/ s0,
        /*.s1   =*/ s1,
        /*.s2   =*/ s2,
        /*.s3   =*/ s3
    };

    auto pipeline = ggml_metal_library_get_pipeline_roll(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.lp0  =*/ ggml_get_op_params_i32(op, 0),
        /*.lp1  =*/ ggml_get_op_params_i32(op, 2),
        /*.lp2  =*/ ggml_get_op_params_i32(op, 4),
        /*.lp3  =*/ ggml_get_op_params_i32(op, 6),
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad(lib, op);

    const int nth_max = MIN(64, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int nth = MIN(args.ne0, nth_max);
    const int nk0 = (args.ne0 + 1024 - 1)/1024; // note: 1024 is hardcoded in the kernel!

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, nk0*ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_pad_reflect_1d(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_pad_reflect_1d args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
        /*.p0 =*/ ((const int32_t *)(op->op_params))[0],
        /*.p1 =*/ ((const int32_t *)(op->op_params))[1]
    };

    auto pipeline = ggml_metal_library_get_pipeline_pad_reflect_1d(lib, op);

    const int nth = std::min(1024, ne0);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_arange(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float start;
    float step;

    memcpy(&start, ((const int32_t *) op->op_params) + 0, sizeof(float));
    memcpy(&step,  ((const int32_t *) op->op_params) + 2, sizeof(float));

    ggml_metal_kargs_arange args = {
        /*.ne0   =*/ ne0,
        /*.start =*/ start,
        /*.step  =*/ step
    };

    const int nth = std::min(1024, ne0);

    auto pipeline = ggml_metal_library_get_pipeline_arange(lib, op);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_timestep_embedding(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int dim        = op->op_params[0];
    const int max_period = op->op_params[1];

    ggml_metal_kargs_timestep_embedding args = {
        /*.nb1 =*/ nb1,
        /*.dim =*/ dim,
        /*.max_period =*/ max_period,
    };

    auto pipeline = ggml_metal_library_get_pipeline_timestep_embedding(lib, op);

    const int nth = std::max(1, std::min(1024, dim/2));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne00, 1, 1, nth, 1, 1);

    return 1;
}


// the MoE router emits softmax -> argsort -> get_rows [-> sum_rows -> clamp -> div].
// ggml_can_fuse_ext cannot express it: the argsort output is also read by the later
// mul_mat_id, so it has two uses. Match it by hand like the CUDA backend does.
struct ggml_metal_tosh_topk_moe_plan {
    int n_fuse = 0;
    bool with_norm = false;
    bool delayed = false;
    float clamp = 0.0f;
    ggml_tensor * logits = nullptr;
    ggml_tensor * ids = nullptr;
    ggml_tensor * weights = nullptr;
};

static ggml_tensor * ggml_metal_view_base(ggml_tensor * t) {
    while (t && t->view_src) {
        t = t->view_src;
    }
    return t;
}

static bool ggml_metal_tosh_topk_moe_same(ggml_tensor * a, ggml_tensor * b) {
    return a && b && ggml_metal_view_base(a) == ggml_metal_view_base(b);
}

static bool ggml_metal_tosh_topk_moe_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_tosh_topk_moe_plan & plan) {
    // decode on these cards is bound by the host, and the router chain is five small dispatches
    // per MoE layer; where the experts run on the CPU instead it measures neutral
    static const bool enabled = TOSH_ENV("GGML_METAL_TOPK_MOE_DISABLE") == nullptr;
    if (!enabled || !ctx->use_fusion()) {
        return false;
    }

    if (idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * softmax = ctx->node(idx + 0);
    ggml_tensor * argsort = ctx->node(idx + 1);
    ggml_tensor * getrows = ctx->node(idx + 2);

    if (softmax->op != GGML_OP_SOFT_MAX || argsort->op != GGML_OP_ARGSORT || getrows->op != GGML_OP_GET_ROWS) {
        return false;
    }

    // a mask or ALiBi slope is not part of the router chain
    if (softmax->src[1] || softmax->src[2]) {
        return false;
    }

    if (softmax->type != GGML_TYPE_F32 || softmax->src[0]->type != GGML_TYPE_F32 ||
        argsort->type != GGML_TYPE_I32 || getrows->type != GGML_TYPE_F32) {
        return false;
    }

    float scale = 1.0f;
    float bias  = 0.0f;
    memcpy(&scale, ((const int32_t *) softmax->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) softmax->op_params) + 1, sizeof(float));
    if (scale != 1.0f || bias != 0.0f) {
        return false;
    }

    if (!ggml_metal_tosh_topk_moe_same(argsort->src[0], softmax) ||
        !ggml_metal_tosh_topk_moe_same(getrows->src[0], softmax) ||
        !ggml_metal_tosh_topk_moe_same(getrows->src[1], argsort)) {
        return false;
    }

    if (!ggml_is_contiguous(softmax->src[0]) || !ggml_is_contiguous(softmax) ||
        !ggml_is_contiguous(argsort) || !ggml_is_contiguous(getrows)) {
        return false;
    }

    // the normalising router reads get_rows twice, from sum_rows and from div, either directly or
    // through one reshape; without that chain get_rows already is the weights
    bool norm = false;
    if (idx + 5 < ctx->n_nodes()) {
        ggml_tensor * sumrows = ctx->node(idx + 3);
        ggml_tensor * clamp   = ctx->node(idx + 4);
        ggml_tensor * div     = ctx->node(idx + 5);

        bool uses_ok = ctx->has_n_uses(idx + 2, 2);
        if (!uses_ok && ctx->has_n_uses(idx + 2, 1)) {
            const ggml_cgraph * gf = ctx->graph();
            for (int j = ctx->graph_index(idx + 2) + 1; j < ctx->graph_index(idx + 3); ++j) {
                if (ggml_metal_view_base(gf->nodes[j]) == getrows) {
                    uses_ok = ggml_node_get_use_count(gf, j) == 2 &&
                        !(gf->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT);
                    break;
                }
            }
        }

        float cmin = 0.0f;
        float cmax = 0.0f;
        if (clamp->op == GGML_OP_CLAMP) {
            memcpy(&cmin, ((const int32_t *) clamp->op_params) + 0, sizeof(float));
            memcpy(&cmax, ((const int32_t *) clamp->op_params) + 1, sizeof(float));
        }

        norm = uses_ok &&
            sumrows->op == GGML_OP_SUM_ROWS && clamp->op == GGML_OP_CLAMP && div->op == GGML_OP_DIV &&
            ggml_metal_tosh_topk_moe_same(sumrows->src[0], getrows) &&
            ggml_metal_tosh_topk_moe_same(clamp->src[0], sumrows) &&
            ggml_metal_tosh_topk_moe_same(div->src[0], getrows) &&
            ggml_metal_tosh_topk_moe_same(div->src[1], clamp) &&
            ggml_is_contiguous(div) && div->ne[0] <= 32 &&
            ctx->has_n_uses(idx + 3, 1) &&
            ctx->has_n_uses(idx + 4, 1) &&
            std::isinf(cmax) && cmax > 0.0f;

        if (norm) {
            plan.n_fuse    = 6;
            plan.with_norm = true;
            plan.clamp     = cmin;
            plan.weights   = div;
        }
    }
    if (!norm) {
        if (!ctx->has_n_uses(idx + 2, 1)) {
            return false;
        }
        plan.n_fuse  = 3;
        plan.weights = getrows;
    }

    const int64_t n_expert = softmax->ne[0];
    if (n_expert > 512 || n_expert < 1) {
        return false;
    }

    plan.logits = softmax->src[0];
    plan.ids    = argsort;
    return true;
}

// the router that picks on the raw logits and takes the softmax over the picked ones only
// (gpt-oss): argsort, get_rows, softmax
static bool ggml_metal_tosh_topk_moe_delayed_fusion(ggml_metal_op_t ctx, int idx, ggml_metal_tosh_topk_moe_plan & plan) {
    static const bool enabled = TOSH_ENV("GGML_METAL_TOPK_MOE_DISABLE") == nullptr;
    if (!enabled || !ctx->use_fusion() || idx + 2 >= ctx->n_nodes()) {
        return false;
    }

    ggml_tensor * argsort = ctx->node(idx + 0);
    ggml_tensor * getrows = ctx->node(idx + 1);
    ggml_tensor * softmax = ctx->node(idx + 2);

    if (argsort->op != GGML_OP_ARGSORT || getrows->op != GGML_OP_GET_ROWS || softmax->op != GGML_OP_SOFT_MAX) {
        return false;
    }
    if (ggml_get_op_params_i32(argsort, 0) != GGML_SORT_ORDER_DESC || softmax->src[1] || softmax->src[2]) {
        return false;
    }
    if (argsort->src[0]->type != GGML_TYPE_F32 || argsort->type != GGML_TYPE_I32 ||
        getrows->type != GGML_TYPE_F32 || softmax->type != GGML_TYPE_F32) {
        return false;
    }

    float scale = 1.0f;
    float bias  = 0.0f;
    memcpy(&scale, ((const int32_t *) softmax->op_params) + 0, sizeof(float));
    memcpy(&bias,  ((const int32_t *) softmax->op_params) + 1, sizeof(float));
    if (scale != 1.0f || bias != 0.0f) {
        return false;
    }

    ggml_tensor * logits = argsort->src[0];
    if (!ggml_metal_tosh_topk_moe_same(getrows->src[0], logits) ||
        !ggml_metal_tosh_topk_moe_same(getrows->src[1], argsort) ||
        !ggml_metal_tosh_topk_moe_same(softmax->src[0], getrows)) {
        return false;
    }
    if (!ggml_is_contiguous(logits) || !ggml_is_contiguous(argsort) ||
        !ggml_is_contiguous(getrows) || !ggml_is_contiguous(softmax) ||
        !ctx->has_n_uses(idx + 1, 1)) {
        return false;
    }

    const int64_t n_expert = logits->ne[0];
    if (n_expert > 512 || n_expert < 1 || softmax->ne[0] != getrows->src[1]->ne[0]) {
        return false;
    }

    plan.n_fuse    = 3;
    plan.delayed   = true;
    plan.with_norm = false;
    plan.logits    = logits;
    plan.ids       = argsort;
    plan.weights   = softmax;
    return true;
}

size_t ggml_metal_op_argmax_extra(const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGMAX);

    const int64_t ne00 = op->src[0]->ne[0];
    const int64_t ne01 = op->src[0]->ne[1];

    int npart = 1;
    if (ne00 > N_ARGMAX_CHUNK) {
        npart = (int) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    if (npart == 1) {
        return 0;
    }

    // partial max + partial argmax for every row
    return (sizeof(float) + sizeof(int32_t)) * npart * ne01;
}

int ggml_metal_op_argmax(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    const int64_t nrows = ggml_nrows(op->src[0]);

    const int simd_width = ggml_metal_device_get_props(ctx->dev)->simd_width;

    // the part/final kernels reduce over 32 lanes
    int npart = 1;
    if (ne00 > N_ARGMAX_CHUNK && simd_width == 32) {
        npart = (int) std::min<int64_t>(N_ARGMAX_NPART_MAX, (ne00 + N_ARGMAX_CHUNK - 1)/N_ARGMAX_CHUNK);
    }

    ggml_metal_kargs_argmax args = {
        /*.ne00  = */ ne00,
        /*.nb01  = */ nb01,
        /*.npart = */ npart,
        /*.ne01  = */ ne01,
    };

    const size_t smem = 32*(sizeof(float) + sizeof(int32_t));

    if (npart > 1) {
        auto pipeline_part = ggml_metal_library_get_pipeline_argmax_part(lib);
        auto pipeline_final = ggml_metal_library_get_pipeline_argmax_final(lib);

        ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(op);
        ggml_metal_buffer_id bid_tmp = bid_dst;
        bid_tmp.offs += ggml_nbytes(op);

        ggml_metal_encoder_set_pipeline(enc, pipeline_part);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, npart, nrows, 1, 256, 1, 1);

        ggml_metal_op_concurrency_reset(ctx);

        int nth = 32; // SIMD width
        while (nth < npart && nth < 256) {
            nth *= 2;
        }

        ggml_metal_encoder_set_pipeline(enc, pipeline_final);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);
        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

        return 1;
    }

    auto pipeline = ggml_metal_library_get_pipeline_argmax(lib, op);

    int nth = simd_width;
    while (nth < ne00 && nth*ne01*ne02*ne03 < 256) {
        nth *= 2;
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, pipeline.smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_tosh_topk_moe(ggml_metal_op_t ctx, int idx) {
    ggml_metal_tosh_topk_moe_plan plan;
    const bool matched = ctx->node(idx)->op == GGML_OP_ARGSORT ?
        ggml_metal_tosh_topk_moe_delayed_fusion(ctx, idx, plan) :
        ggml_metal_tosh_topk_moe_fusion(ctx, idx, plan);
    if (!matched) {
        return 0;
    }

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    auto pipeline = ggml_metal_library_get_pipeline_tosh_topk_moe(lib);
    if (!pipeline.pipeline) {
        return 0;
    }

    const int32_t n_expert      = plan.logits->ne[0];
    const int32_t nrows         = ggml_nrows(plan.logits);
    // get_rows leaves [1, n_used, rows], div and the delayed softmax [n_used, rows]
    const int32_t n_expert_used = ggml_nelements(plan.weights)/nrows;

    int nth = 32;
    while (2*nth <= n_expert && 2*nth <= (int) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const size_t smem_f = GGML_PAD((2*n_expert + nth)*sizeof(float), 16);
    const size_t smem_i = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_kargs_tosh_topk_moe args = {
        /*.n_expert        =*/ n_expert,
        /*.n_expert_used   =*/ n_expert_used,
        /*.nrows           =*/ nrows,
        /*.mode            =*/ plan.delayed ? 3 : 0,
        /*.with_norm       =*/ plan.with_norm ? 1 : 0,
        /*.delayed_softmax =*/ plan.delayed ? 1 : 0,
        /*.has_bias        =*/ 0,
        /*.clamp           =*/ plan.clamp,
        /*.scale           =*/ 1.0f,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.logits),  2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.weights), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(plan.ids),     4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_f, 0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_i, 1);

    ggml_metal_encoder_dispatch_threadgroups(enc, nrows, 1, 1, nth, 1, 1);

    // two outputs leave the fused range, and only the first node's dst is tracked
    ggml_metal_op_concurrency_reset(ctx);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fused %d nodes into topk_moe (norm = %d)\n", __func__, plan.n_fuse, (int) plan.with_norm);
    }

    return plan.n_fuse;
}

int ggml_metal_op_argsort(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_argsort(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    const int npr = (ne00 + nth - 1)/nth;

    // Metal kernels require the buffer size to be multiple of 16 bytes
    // https://developer.apple.com/documentation/metal/mtlcomputecommandencoder/1443142-setthreadgroupmemorylength
    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += ggml_nbytes(op);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ nth,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_argsort_merge(lib, op);

    int len = nth;

    while (len < ne00) {
        ggml_metal_op_concurrency_reset(ctx);

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ ne00,
            /*.len   =*/ len,
        };

        // merges per row
        const int nm = (ne00 + 2*len - 1) / (2*len);

        const int nth = std::min(512, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge));

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }

    return 1;
}

// bitonic-sort + merge fallback: efficient when k is small and there are few rows,
// where the single-workgroup-per-row radix-select cannot reach enough parallelism
static void ggml_metal_op_top_k_bitonic(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k(lib, op);

    // bitonic sort requires the number of elements to be power of 2
    int nth = 1;
    while (nth < ne00 && 2*nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    // blocks per row
    const int npr = (ne00 + nth - 1)/nth;

    const size_t smem = GGML_PAD(nth*sizeof(int32_t), 16);

    ggml_metal_buffer_id bid_src0 = ggml_metal_get_buffer_id(op->src[0]);
    ggml_metal_buffer_id bid_dst  = ggml_metal_get_buffer_id(op);

    ggml_metal_buffer_id bid_tmp = bid_dst;
    bid_tmp.offs += sizeof(int32_t)*ggml_nelements(op->src[0]);

    if ((int) ceil(std::log(npr) / std::log(2)) % 2 == 1) {
        std::swap(bid_dst, bid_tmp);
    }

    const int top_k = ne0;

    ggml_metal_kargs_argsort args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.top_k =*/ std::min(nth, top_k), // for each block, keep just the top_k indices
    };

    if (npr > 1) {
        args.ne0 = (npr - 1)*args.top_k + std::min(ne00 - (npr - 1)*nth, args.top_k);
    }

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, npr*ne01, ne02, ne03, nth, 1, 1);

    auto pipeline_merge = ggml_metal_library_get_pipeline_top_k_merge(lib, op);

    int len = args.top_k;

    while (len < args.ne0) {
        ggml_metal_op_concurrency_reset(ctx);

        // merges per row
        const int nm = (args.ne0 + 2*len - 1) / (2*len);

        const int nth = std::min(512, std::min(len, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline_merge)));

        ggml_metal_kargs_argsort_merge args_merge = {
            /*.ne00  =*/ ne00,
            /*.ne01  =*/ ne01,
            /*.ne02  =*/ ne02,
            /*.ne03  =*/ ne03,
            /*.nb00  =*/ nb00,
            /*.nb01  =*/ nb01,
            /*.nb02  =*/ nb02,
            /*.nb03  =*/ nb03,
            /*.ne0   =*/ args.ne0,
            /*.ne1   =*/ ne1,
            /*.ne2   =*/ ne2,
            /*.ne3   =*/ ne3,
            /*.top_k =*/ nm == 1 ? top_k : args.ne0, // the final merge outputs top_k elements
            /*.len   =*/ len,
        };

        ggml_metal_encoder_set_pipeline(enc, pipeline_merge);
        ggml_metal_encoder_set_bytes   (enc, &args_merge, sizeof(args_merge), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_src0, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_dst,  2);
        ggml_metal_encoder_set_buffer  (enc, bid_tmp,  3);

        ggml_metal_encoder_dispatch_threadgroups(enc, nm*ne01, ne02, ne03, nth, 1, 1);

        std::swap(bid_dst, bid_tmp);

        len <<= 1;
    }
}

// radix-select: one workgroup per row. Maps each float to an order-preserving unsigned
// key, finds the k-th largest via 4 radix-8 histogram passes, then compacts the top-k
// indices. Fast for large k and/or many rows.
static void ggml_metal_op_top_k_radix(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);

    auto pipeline = ggml_metal_library_get_pipeline_top_k_radix(lib, op);

    // one workgroup per row; radix-select the k-th largest value
    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_kargs_top_k args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.top_k =*/ (int32_t) op->ne[0],
    };

    // shared memory: 256-entry histogram + bucket/above scalars
    const size_t smem_histo  = GGML_PAD(256*sizeof(uint32_t), 16);
    const size_t smem_bucket = GGML_PAD(    sizeof(uint32_t), 16);
    const size_t smem_above  = GGML_PAD(    sizeof(uint32_t), 16);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_histo,  0);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_bucket, 1);
    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem_above,  2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
}

int ggml_metal_op_topk_moe(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_TOPK_MOE) {
        return 1;
    }

    ggml_tensor * first  = ctx->node(idx);
    ggml_tensor * logits = first->src[0];

    // the unary chains score with sigmoid or sqrt(softplus) and rank biased scores
    const bool with_unary = first->op == GGML_OP_UNARY;
    const int  gating     = !with_unary ? 0 : ggml_get_unary_op(first) == GGML_UNARY_OP_SOFTPLUS ? 1 : 2;

    ggml_tensor * get_rows  = nullptr;
    ggml_tensor * sum_rows  = nullptr;
    ggml_tensor * prob_bias = nullptr;
    int sum_rows_pos = -1;
    for (int j = 0; j < n_fuse; ++j) {
        ggml_tensor * node = ctx->node(idx + j);
        if (node->op == GGML_OP_GET_ROWS) { get_rows = node; }
        if (node->op == GGML_OP_ADD)      { prob_bias = node->src[1]; }
        if (node->op == GGML_OP_SUM_ROWS) { sum_rows = node; sum_rows_pos = j; }
    }

    ggml_tensor * ids     = get_rows->src[1];
    ggml_tensor * weights = ctx->node(idx + n_fuse - 1);

    const bool with_norm  = sum_rows != nullptr;
    const bool with_scale = weights->op == GGML_OP_SCALE;

    const int64_t n_expert      = logits->ne[0];
    const int64_t n_tokens      = logits->ne[1];
    const int64_t n_expert_used = ids->ne[0];

    float clamp = -INFINITY;
    float bias  = 0.0f;
    if (with_norm) {
        ggml_tensor * denom_node = ctx->node(idx + sum_rows_pos + 1);
        if (denom_node->op == GGML_OP_SCALE) {
            bias = ggml_get_op_params_f32(denom_node, 1);
        } else {
            clamp = ggml_get_op_params_f32(denom_node, 0);
        }
    }

    float scale = 1.0f;
    if (with_scale) {
        scale = ggml_get_op_params_f32(weights, 0);
    }

    ggml_metal_kargs_topk_moe args = {
        /*.ne01      =*/ (int32_t) n_tokens,
        /*.nb01      =*/ logits->nb[1],
        /*.nb1_ids   =*/ ids->nb[1],
        /*.clamp =*/ clamp,
        /*.bias  =*/ bias,
        /*.scale =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_topk_moe(lib, (int32_t) n_expert, (int32_t) n_expert_used,
            with_norm, gating, prob_bias != nullptr);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(logits),  1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(ids),     3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(prob_bias ? prob_bias : logits), 4);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) n_tokens, 1, 1, 32, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: %s + ARGSORT + GET_ROWS\n", __func__,
                gating == 1 ? "SOFTPLUS + SQRT" : gating == 2 ? "SIGMOID" : "SOFT_MAX");
    }

    return n_fuse;
}

int ggml_metal_op_moe_reduce(ggml_metal_op_t ctx, int idx) {
    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    int n_fuse = 1;
    const ggml_metal_fusion * fusion = ctx->can_fuse(idx, GGML_METAL_FUSION_FULL, &n_fuse);
    if (!fusion || ggml_metal_fusion_get_id(fusion) != GGML_METAL_FUSION_MOE_REDUCE) {
        return 1;
    }

    ggml_tensor * mul     = ctx->node(idx);
    ggml_tensor * experts = mul->src[0];
    ggml_tensor * weights = mul->src[1];
    ggml_tensor * dst     = ctx->node(idx + n_fuse - 1);

    ggml_metal_kargs_moe_reduce args = {
        /*.ne00 =*/ (int32_t) experts->ne[0],
        /*.ne02 =*/ (int32_t) experts->ne[2],
    };

    auto pipeline = ggml_metal_library_get_pipeline_moe_reduce(lib, (int32_t) experts->ne[1]);

    const int nth = std::min(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    const int n_col_tiles = (args.ne00 + nth - 1) / nth;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(experts), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(weights), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(dst),     3);

    ggml_metal_encoder_dispatch_threadgroups(enc, (uint32_t) args.ne02, (uint32_t) n_col_tiles, 1, nth, 1, 1);

    ctx->count_fusions(fusion);

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 1) {
        GGML_LOG_DEBUG("%s: fuse: MOE_REDUCE\n", __func__);
    }

    return n_fuse;
}

int ggml_metal_op_top_k(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    // radix-select has a fixed single-workgroup-per-row cost (~50-60us) that is only
    // amortized for long rows, many rows, or a large k; otherwise the bitonic path wins
    const int ncols = op->src[0]->ne[0];
    const int k     = op->ne[0];
    const int nrows = ggml_nrows(op->src[0]);

    const bool use_radix =
        ncols > 2048 && (k > 64 || (nrows > 4 && ncols >= 8192));

    if (use_radix) {
        ggml_metal_op_top_k_radix(ctx, idx);
    } else {
        ggml_metal_op_top_k_bitonic(ctx, idx);
    }

    return 1;
}

int ggml_metal_op_tri(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    ggml_metal_kargs_tri args = {
        /*.ne00  =*/ ne00,
        /*.ne01  =*/ ne01,
        /*.ne02  =*/ ne02,
        /*.ne03  =*/ ne03,
        /*.nb00  =*/ nb00,
        /*.nb01  =*/ nb01,
        /*.nb02  =*/ nb02,
        /*.nb03  =*/ nb03,
        /*.ne0   =*/ ne0,
        /*.ne1   =*/ ne1,
        /*.ne2   =*/ ne2,
        /*.ne3   =*/ ne3,
        /*.nb0   =*/ nb0,
        /*.nb1   =*/ nb1,
        /*.nb2   =*/ nb2,
        /*.nb3   =*/ nb3,
    };

    auto pipeline = ggml_metal_library_get_pipeline_tri(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));
    nth = std::min(nth, ne00);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_adamw(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_adamw(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_adamw args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[3]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[4]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_opt_step_sgd(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_opt_step_sgd(lib, op);

    const int64_t np = ggml_nelements(op->src[0]);
    ggml_metal_kargs_opt_step_sgd args = {
        /*.np =*/ np,
    };

    int ida = 0;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), ida++);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), ida++);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);
    const int64_t n = (np + nth - 1) / nth;

    ggml_metal_encoder_dispatch_threadgroups(enc, n, 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS(int32_t,  ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);

    {
        ggml_metal_kargs_memset args = { /*.val =*/ 0 };

        auto pipeline = ggml_metal_library_get_pipeline_memset(lib, op);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 1);

        ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 1, 1, 1);
    }

    ggml_metal_op_concurrency_reset(ctx);

    {
        ggml_metal_kargs_count_equal args = {
            /*.ne00 =*/ ne00,
            /*.ne01 =*/ ne01,
            /*.ne02 =*/ ne02,
            /*.ne03 =*/ ne03,
            /*.nb00 =*/ nb00,
            /*.nb01 =*/ nb01,
            /*.nb02 =*/ nb02,
            /*.nb03 =*/ nb03,
            /*.nb10 =*/ nb10,
            /*.nb11 =*/ nb11,
            /*.nb12 =*/ nb12,
            /*.nb13 =*/ nb13,
        };

        auto pipeline = ggml_metal_library_get_pipeline_count_equal(lib, op);

        const size_t smem = pipeline.smem;

        const int nth = ggml_metal_device_get_props(ctx->dev)->simd_width*pipeline.nsg;

        GGML_ASSERT(nth <= ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes(enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[0]), 1);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[1]), 2);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op), 3);

        ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);
        ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);
    }

    return 1;
}

int ggml_metal_op_soft_max_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    float scale    = 1.0f;
    float max_bias = 0.0f;

    memcpy(&scale,    ((const int32_t *) op->op_params) + 0, sizeof(scale));
    memcpy(&max_bias, ((const int32_t *) op->op_params) + 1, sizeof(max_bias));

    GGML_ASSERT(max_bias == 0.0f);

    const bool use_vec4 = (ne00 % 4) == 0;

    ggml_metal_kargs_soft_max_back args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.nb02   =*/ nb02,
        /*.nb03   =*/ nb03,
        /*.nb11   =*/ nb11,
        /*.nb12   =*/ nb12,
        /*.nb13   =*/ nb13,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.scale  =*/ scale,
    };

    auto pipeline = ggml_metal_library_get_pipeline_soft_max_back(lib, op);

    int nth = 32; // SIMD width

    if (use_vec4) {
        const int ne00_4 = ne00/4;
        while (nth < ne00_4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }
    } else {
        while (nth < ne00 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
            nth *= 2;
        }
    }

    nth = std::max(1, nth);
    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_rms_norm_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb1, op->src[1], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    GGML_ASSERT(ne00 % 4 == 0);

    float eps;
    memcpy(&eps, op->op_params, sizeof(float));

    ggml_metal_kargs_rms_norm_back args = {
        /*.ne00   =*/ ne00,
        /*.ne00_4 =*/ ne00/4,
        /*.nb01   =*/ nb01,
        /*.nb02   =*/ nb02,
        /*.nb03   =*/ nb03,
        /*.nb11   =*/ nb11,
        /*.nb12   =*/ nb12,
        /*.nb13   =*/ nb13,
        /*.nb1    =*/ nb1,
        /*.nb2    =*/ nb2,
        /*.nb3    =*/ nb3,
        /*.eps    =*/ eps,
    };

    auto pipeline = ggml_metal_library_get_pipeline_rms_norm_back(lib, op);

    int nth = 32; // SIMD width

    while (nth < ne00/4 && nth < ggml_metal_pipeline_max_theads_per_threadgroup(pipeline)) {
        nth *= 2;
    }

    nth = std::min(nth, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    const size_t smem = pipeline.smem;

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         3);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, smem, 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne01, ne02, ne03, nth, 1, 1);

    return 1;
}

int ggml_metal_op_repeat_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS(uint64_t, nb0, op->src[0], nb);
    GGML_TENSOR_LOCALS( int32_t, ne,  op,         ne);
    GGML_TENSOR_LOCALS(uint64_t, nb,  op,         nb);

    auto pipeline = ggml_metal_library_get_pipeline_repeat_back(lib);

    ggml_metal_kargs_repeat args = {
        /*.ne00 =*/ ne00,
        /*.ne01 =*/ ne01,
        /*.ne02 =*/ ne02,
        /*.ne03 =*/ ne03,
        /*.nb00 =*/ nb00,
        /*.nb01 =*/ nb01,
        /*.nb02 =*/ nb02,
        /*.nb03 =*/ nb03,
        /*.ne0  =*/ ne0,
        /*.ne1  =*/ ne1,
        /*.ne2  =*/ ne2,
        /*.ne3  =*/ ne3,
        /*.nb0  =*/ nb0,
        /*.nb1  =*/ nb1,
        /*.nb2  =*/ nb2,
        /*.nb3  =*/ nb3,
    };

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         2);

    const int nth = std::min(ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), ne0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ne1, ne2, ne3, nth, 1, 1);

    return 1;
}

int ggml_metal_op_cross_entropy_rows(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const bool back = op->op == GGML_OP_CROSS_ENTROPY_ROWS_BACK;
    const ggml_tensor * logits = back ? op->src[1] : op->src[0];

    ggml_metal_kargs_cross_entropy_rows args = {
        /*.ne00 =*/ (int32_t) logits->ne[0],
        /*.nb01 =*/ logits->nb[1],
        /*.nb1  =*/ op->nb[1],
    };

    auto pipeline = ggml_metal_library_get_pipeline_by_name(lib, back ? "kernel_cross_entropy_rows_back_f32" : "kernel_cross_entropy_rows_f32");

    const int nth = std::min(1024, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    int ib = 1;
    for (int j = 0; j < (back ? 3 : 2); ++j) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(op->src[j]), ib++);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op), ib);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*sizeof(float), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, ggml_nrows(logits), 1, 1, nth, 1, 1);

    return 1;
}

int ggml_metal_op_count_equal_masked(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    ggml_metal_kargs_count_equal_masked args = {
        /*.n =*/ (int32_t) op->src[0]->ne[0],
    };

    auto pipeline = ggml_metal_library_get_pipeline_by_name(lib, "kernel_count_equal_masked");

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[0]), 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[1]), 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op->src[2]), 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),         4);

    ggml_metal_encoder_set_threadgroup_memory_size(enc, 32*sizeof(int32_t), 0);

    ggml_metal_encoder_dispatch_threadgroups(enc, 1, 1, 1, 256, 1, 1);

    return 1;
}

int ggml_metal_op_flash_attn_ext_back(ggml_metal_op_t ctx, int idx) {
    ggml_tensor * op = ctx->node(idx);

    ggml_metal_library_t lib = ctx->lib;
    ggml_metal_encoder_t enc = ctx->enc;

    const ggml_tensor * q      = op->src[0];
    const ggml_tensor * k      = op->src[1];
    const ggml_tensor * v      = op->src[2];
    const ggml_tensor * mask   = op->src[3];
    const ggml_tensor * o      = op->src[4];
    const ggml_tensor * grad_o = op->src[5];
    const ggml_tensor * stats  = op->src[6];

    const int32_t target = ggml_get_op_params_i32(op, 5);

    float scale;
    memcpy(&scale, op->op_params, sizeof(float));

    ggml_metal_kargs_flash_attn_ext_back args = {
        /*.ne01 =*/ (int32_t) q->ne[1],
        /*.ne02 =*/ (int32_t) q->ne[2],
        /*.ne11 =*/ (int32_t) k->ne[1],
        /*.ne12 =*/ (int32_t) k->ne[2],
        /*.ne13 =*/ (int32_t) k->ne[3],
        q->nb[1], q->nb[2], q->nb[3],
        k->nb[1], k->nb[2], k->nb[3],
        v->nb[1], v->nb[2], v->nb[3],
        mask ? mask->nb[1] : 0, mask ? mask->nb[2] : 0, mask ? mask->nb[3] : 0,
        mask ? (int32_t) mask->ne[2] : 1, mask ? (int32_t) mask->ne[3] : 1,
        mask ? 1 : 0,
        o->nb[1], o->nb[2], o->nb[3],
        grad_o->nb[1], grad_o->nb[2], grad_o->nb[3],
        stats ? stats->nb[1] : 0, stats ? stats->nb[2] : 0, stats ? stats->nb[3] : 0,
        op->nb[1], op->nb[2], op->nb[3],
        /*.scale =*/ scale,
    };

    const char * pass = target == GGML_FLASH_ATTN_EXT_BACK_STATS ? "stats" : (target == GGML_FLASH_ATTN_EXT_BACK_DQ ? "dq" : "dkv");
    char name[96];
    snprintf(name, sizeof(name), "kernel_flash_attn_ext_back_%s_d%d_%s", pass, (int) k->ne[0], ggml_type_name(k->type));
    auto pipeline = ggml_metal_library_get_pipeline_by_name(lib, name);

    // an absent mask binds q in its slot; the kernels read it only when has_mask is set
    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(q),                 1);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(k),                 2);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(v),                 3);
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(mask ? mask : q),   4);
    if (target == GGML_FLASH_ATTN_EXT_BACK_STATS) {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(o),      5);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(grad_o), 6);
    } else {
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(grad_o), 5);
        ggml_metal_encoder_set_buffer(enc, ggml_metal_get_buffer_id(stats),  6);
    }
    ggml_metal_encoder_set_buffer  (enc, ggml_metal_get_buffer_id(op),                7);

    const int nsg = 8;
    if (target == GGML_FLASH_ATTN_EXT_BACK_DKV) {
        ggml_metal_encoder_dispatch_threadgroups(enc, (k->ne[1] + nsg - 1)/nsg, k->ne[2], k->ne[3], 32, nsg, 1);
    } else {
        ggml_metal_encoder_dispatch_threadgroups(enc, (q->ne[1] + nsg - 1)/nsg, q->ne[2], q->ne[3], 32, nsg, 1);
    }

    return 1;
}
