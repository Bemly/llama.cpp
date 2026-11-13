#include "ggml-metal.h"

#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-metal-device.h"
#include "ggml-metal-fusion.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-hostcache.h"
#include "tosh-dmoe.h"
#endif
#include "ggml-metal-context.h"
#include "ggml-metal-ops.h"
#include "ggml-metal-tuning.h"

#include <mutex>
#include <string>
#ifdef __APPLE__
#include <dispatch/dispatch.h>
#include <atomic>
#endif

#define GGML_METAL_NAME "MTL"
#define GGML_METAL_MAX_DEVICES 16

// number of Metal devices
// note: can be overridden with GGML_METAL_DEVICES env to simulate virtual devices
static int g_devices = 1;

// forward declaration
static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer);

////////////////////////////////////////////////////////////////////////////////
// backend interface
////////////////////////////////////////////////////////////////////////////////

// shared buffer

static void ggml_backend_metal_buffer_shared_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_shared_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_shared_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_shared_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static void ggml_backend_metal_buffer_shared_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static bool ggml_backend_metal_buffer_shared_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    for (size_t i = 0; i < n_tensors; ++i) {
        if (!ggml_backend_buffer_is_metal(srcs[i]->buffer)) {
            return false;
        }
    }

    return ggml_metal_buffer_cpy_tensor_n(ctx, srcs, dsts, n_tensors);
}

static void ggml_backend_metal_buffer_shared_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_shared_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_shared_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_shared_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_shared_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_shared_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_shared_get_tensor,
    /* .set_tensor_2d = */ NULL,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_shared_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_shared_clear,
    /* .reset         = */ NULL,
    /* .set_tensor_n  = */ ggml_backend_metal_buffer_shared_set_tensor_n,
    /* .get_tensor_n  = */ ggml_backend_metal_buffer_shared_get_tensor_n,
    /* .cpy_tensor_n  = */ ggml_backend_metal_buffer_shared_cpy_tensor_n,
};

// private buffer

static void ggml_backend_metal_buffer_private_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_free(ctx);
}

static void * ggml_backend_metal_buffer_private_get_base(ggml_backend_buffer_t buffer) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    return ggml_metal_buffer_get_base(ctx);
}

static void ggml_backend_metal_buffer_private_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_memset_tensor(ctx, tensor, value, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_buffer_private_set_tensor_n(ggml_backend_buffer_t buffer, struct ggml_tensor * const * tensors, const void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static void ggml_backend_metal_buffer_private_get_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * tensors, void * const * data, const size_t * offsets, const size_t * sizes, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_get_tensor_n(ctx, tensors, data, offsets, sizes, n_tensors);
}

static void ggml_backend_metal_buffer_private_set_tensor_2d(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size,
        size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_set_tensor_2d(ctx, tensor, data, offset, size, n_copies, stride_tensor, stride_data);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    if (!ggml_backend_buffer_is_metal(src->buffer)) {
        return false;
    }

    return ggml_metal_buffer_cpy_tensor(ctx, src, dst);
}

static bool ggml_backend_metal_buffer_private_cpy_tensor_n(ggml_backend_buffer_t buffer, const struct ggml_tensor * const * srcs, struct ggml_tensor * const * dsts, size_t n_tensors) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    for (size_t i = 0; i < n_tensors; ++i) {
        if (!ggml_backend_buffer_is_metal(srcs[i]->buffer)) {
            return false;
        }
    }

    return ggml_metal_buffer_cpy_tensor_n(ctx, srcs, dsts, n_tensors);
}

static void ggml_backend_metal_buffer_private_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_metal_buffer_t ctx = (ggml_metal_buffer_t)buffer->context;

    GGML_ASSERT(!ggml_metal_buffer_is_shared(ctx));

    ggml_metal_buffer_clear(ctx, value);
}

static ggml_backend_buffer_i ggml_backend_metal_buffer_private_i = {
    /* .free_buffer   = */ ggml_backend_metal_buffer_private_free_buffer,
    /* .get_base      = */ ggml_backend_metal_buffer_private_get_base,
    /* .init_tensor   = */ NULL,
    /* .memset_tensor = */ ggml_backend_metal_buffer_private_memset_tensor,
    /* .set_tensor    = */ ggml_backend_metal_buffer_private_set_tensor,
    /* .get_tensor    = */ ggml_backend_metal_buffer_private_get_tensor,
    /* .set_tensor_2d = */ ggml_backend_metal_buffer_private_set_tensor_2d,
    /* .get_tensor_2d = */ NULL,
    /* .cpy_tensor    = */ ggml_backend_metal_buffer_private_cpy_tensor,
    /* .clear         = */ ggml_backend_metal_buffer_private_clear,
    /* .reset         = */ NULL,
    /* .set_tensor_n  = */ ggml_backend_metal_buffer_private_set_tensor_n,
    /* .get_tensor_n  = */ ggml_backend_metal_buffer_private_get_tensor_n,
    /* .cpy_tensor_n  = */ ggml_backend_metal_buffer_private_cpy_tensor_n,
};

static bool ggml_backend_buffer_is_metal(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_metal_buffer_shared_free_buffer ||
           buffer->iface.free_buffer == ggml_backend_metal_buffer_private_free_buffer;
}

//
// buffer types
//

struct ggml_backend_metal_buffer_type {
    int device;
    std::string name;
};

struct ggml_backend_metal_buffer_type_deleter {
    void operator()(ggml_backend_metal_buffer_type * ctx) const {
        delete ctx;
    }
};

typedef std::unique_ptr<ggml_backend_metal_buffer_type, ggml_backend_metal_buffer_type_deleter> ggml_backend_metal_buffer_type_ptr;

// common method for allocating shread or private Metal buffers
static ggml_backend_buffer_t ggml_backend_metal_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size, bool shared) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;
    ggml_metal_buffer_t res = ggml_metal_buffer_init(ctx_dev, size, shared);

    // an allocation that fails returns null here, and reading is_shared off it turned an
    // out-of-memory into a segfault. Diagnosed by Slice on the RX 570:
    // https://www.insanelymac.com/forum/profile/112217-slice/
    if (res == NULL) {
        return NULL;
    }

    if (res == NULL) {
        GGML_LOG_ERROR("%s: failed to allocate Metal buffer of %zu bytes (out of memory)\n", __func__, size);
        return NULL;
    }

    ggml_backend_buffer_i buf_i = ggml_metal_buffer_is_shared(res)
        ? ggml_backend_metal_buffer_shared_i
        : ggml_backend_metal_buffer_private_i;

    return ggml_backend_buffer_init(buft, buf_i, res, size);
}

static size_t ggml_backend_metal_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    size_t res = ggml_nbytes(tensor);

    // some operations require additional memory for fleeting data:
    switch (tensor->op) {
        case GGML_OP_MUL_MAT_ID:
            {
                res += ggml_metal_op_mul_mat_id_extra_tpe(tensor);
                res += ggml_metal_op_mul_mat_id_extra_ids(tensor);
            } break;
        case GGML_OP_FLASH_ATTN_EXT:
            {
                res += ggml_metal_op_flash_attn_ext_extra_pad(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_blk(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_tmp(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_kv_f16(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_idx(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_amd(tensor);
                res += ggml_metal_op_flash_attn_ext_extra_rows(tensor);
            } break;
        case GGML_OP_GATED_DELTA_NET_BACK:
            {
                res += ggml_metal_op_gated_delta_net_back_extra(tensor);
            } break;
        case GGML_OP_OUT_PROD:
            {
                res += ggml_metal_op_out_prod_extra(tensor);
            } break;
        case GGML_OP_CUMSUM:
        case GGML_OP_ARGSORT:
            {
                res *= 2;
            } break;
        case GGML_OP_TOP_K:
            {
                res = 2*sizeof(int32_t)*ggml_nelements(tensor->src[0]);
            } break;
        case GGML_OP_ARGMAX:
            {
                res += ggml_metal_op_argmax_extra(tensor);
            } break;
        default:
            break;
    }

    return res;

    GGML_UNUSED(buft);
}

// default (shared) buffer type

static const char * ggml_backend_metal_buffer_type_shared_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_shared_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_shared_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_shared_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_shared_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_shared_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_shared(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx =
                new ggml_backend_metal_buffer_type {
                    /* .device = */ i,
                    /* .name   = */ GGML_METAL_NAME + std::to_string(i),
                };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_shared_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_shared_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_shared_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_shared_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_shared_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_shared_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// default (private) buffer type

static const char * ggml_backend_metal_buffer_type_private_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_private_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, false);
}

static size_t ggml_backend_metal_buffer_type_private_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_private_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_private_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_private_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_private(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Private"
            };
            ctxs.emplace_back(raw_ctx);

            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_private_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_private_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_private_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_private_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_private_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_private_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// mapped buffer type

static const char * ggml_backend_metal_buffer_type_mapped_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_metal_buffer_type * ctx = (ggml_backend_metal_buffer_type *)buft->context;

    return ctx->name.c_str();
}

static ggml_backend_buffer_t ggml_backend_metal_buffer_type_mapped_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    // for mapped buffers, prefer shared memory
    return ggml_backend_metal_buffer_type_alloc_buffer(buft, size, true);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alignment(ggml_backend_buffer_type_t buft) {
    return 32;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_metal_buffer_type_mapped_get_max_size(ggml_backend_buffer_type_t buft) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)buft->device->context;

    return ggml_metal_device_get_props(ctx_dev)->max_buffer_size;
}

static size_t ggml_backend_metal_buffer_type_mapped_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    return ggml_backend_metal_buffer_type_get_alloc_size(buft, tensor);
}

static bool ggml_backend_metal_buffer_type_mapped_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_type_t ggml_backend_metal_buffer_type_mapped(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    static std::vector<ggml_backend_buffer_type> bufts;
    static std::vector<ggml_backend_metal_buffer_type_ptr> ctxs;

    static bool initialized = false;
    if (!initialized) {
        bufts.reserve(g_devices);
        ctxs.reserve(g_devices);

        for (int i = 0; i < g_devices; ++i) {
            ggml_backend_metal_buffer_type * raw_ctx = new ggml_backend_metal_buffer_type{
                /* .device = */ i,
                /* .name   = */ GGML_METAL_NAME + std::to_string(i) + "_Mapped"
            };
            ctxs.emplace_back(raw_ctx);

            // note: not obvious, but this buffer type still needs to implement .alloc_buffer:
            //       https://github.com/ggml-org/llama.cpp/pull/15832#discussion_r2333177099
            ggml_backend_buffer_type buft = {
                /* .iface = */ {
                    /* .get_name            = */ ggml_backend_metal_buffer_type_mapped_get_name,
                    /* .alloc_buffer        = */ ggml_backend_metal_buffer_type_mapped_alloc_buffer,
                    /* .alloc_buffer_n      = */ NULL,
                    /* .get_alignment       = */ ggml_backend_metal_buffer_type_mapped_get_alignment,
                    /* .get_max_size        = */ ggml_backend_metal_buffer_type_mapped_get_max_size,
                    /* .get_alloc_size      = */ ggml_backend_metal_buffer_type_mapped_get_alloc_size,
                    /* .get_alloc_size_n    = */ NULL,
                    /* .is_host             = */ ggml_backend_metal_buffer_type_mapped_is_host,
                },
                /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_metal_reg(), i),
                /* .context = */ raw_ctx,
            };

            bufts.emplace_back(buft);
        }

        initialized = true;
    }

    return &bufts[device];
}

// backend

static const char * ggml_backend_metal_name(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_get_name(ctx);
}

static void ggml_backend_metal_free(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    // wait for any ongoing async operations to finish
    ggml_metal_synchronize(ctx);

    ggml_metal_free(ctx);

    free(backend);
}

static void ggml_backend_metal_synchronize(ggml_backend_t backend) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_synchronize(ctx);
}

static void ggml_backend_metal_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_tensor_async(ctx, tensor, data, offset, size);
}

static void ggml_backend_metal_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_get_tensor_async(ctx, tensor, data, offset, size);
}

#ifdef GGML_BACKEND_READ_BATCH
static void ggml_backend_metal_read_batch_begin(ggml_backend_t backend) {
    ggml_metal_read_batch_begin((ggml_metal_t) backend->context);
}

static void ggml_backend_metal_read_batch_end(ggml_backend_t backend) {
    ggml_metal_read_batch_end((ggml_metal_t) backend->context);
}
#endif

static bool ggml_backend_metal_exchange_reduce(ggml_backend_t backend_self, ggml_backend_t backend_peer,
                                               ggml_tensor * t_self, ggml_tensor * t_peer,
                                               ggml_tensor * tmp_self, uint64_t seq) {
    ggml_metal_t ctx_self = (ggml_metal_t) backend_self->context;
    ggml_metal_t ctx_peer = (ggml_metal_t) backend_peer->context;

    return ggml_metal_exchange_reduce(ctx_self, ctx_peer, t_self, t_peer, tmp_self, seq);
}

static bool ggml_backend_metal_cpy_tensor_async_ex(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst, bool prefer_events, bool hold_dst) {
    if (!ggml_backend_is_metal(backend_src) || !ggml_backend_is_metal(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_metal(src->buffer) || !ggml_backend_buffer_is_metal(dst->buffer)) {
        return false;
    }

    ggml_metal_t ctx_src = (ggml_metal_t)backend_src->context;
    ggml_metal_t ctx_dst = (ggml_metal_t)backend_dst->context;

    //ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    //ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    //ggml_metal_buffer_t buf_ctx_src = (ggml_metal_buffer_t)buf_src->context;
    //ggml_metal_buffer_t buf_ctx_dst = (ggml_metal_buffer_t)buf_dst->context;

    return ggml_metal_cpy_tensor_async_ex(ctx_src, ctx_dst, src, dst, prefer_events, hold_dst);
}

// Peer costs a third of the decode, so it only pays from a batch up. Both the layer
// hand-off below and the allreduce read the same threshold.
static int64_t ggml_metal_peer_min_batch(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = getenv("TOSH_MGPU_PEER_MIN_BATCH");
        v = s ? atoll(s) : 32;
    }
    return v;
}

static bool ggml_backend_metal_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    // A layer hand-off crosses one token at a time, which is the shape peer is worst at.
    const int64_t batch = src->ne[0] > 0 ? ggml_nelements(src)/src->ne[0] : 1;
    const bool prefer_events = batch < ggml_metal_peer_min_batch();
    return ggml_backend_metal_cpy_tensor_async_ex(backend_src, backend_dst, src, dst, prefer_events, /*hold_dst =*/ false);
}

static enum ggml_status ggml_backend_metal_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_graph_compute(ctx, cgraph);
}

static void ggml_backend_metal_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_record(ctx, ev);
}

static void ggml_backend_metal_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_metal_t ctx = (ggml_metal_t)backend->context;
    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_event_wait(ctx, ev);
}

static void ggml_backend_metal_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    GGML_ASSERT(params && params->add_alloc_dep);

    // keep the MoE weighted-reduction inputs alive until the fused output so the
    // allocator cannot reuse them while the fused kernel is still reading them
    ggml_metal_fusion_add_alloc_deps(params->user_data, params->add_alloc_dep, cgraph);

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_graph_optimize(ctx, cgraph);
}

static void ggml_backend_metal_set_n_cb(ggml_backend_t backend, int n_cb) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_n_cb(ctx, n_cb);
}

static ggml_backend_i ggml_backend_metal_i = {
    /* .get_name                = */ ggml_backend_metal_name,
    /* .free                    = */ ggml_backend_metal_free,
    /* .set_tensor_async        = */ ggml_backend_metal_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_metal_get_tensor_async,
    /* .set_tensor_2d_async     = */ NULL,
    /* .get_tensor_2d_async     = */ NULL,
    /* .cpy_tensor_async        = */ ggml_backend_metal_cpy_tensor_async, // only needed for multi-GPU setups
    /* .synchronize             = */ ggml_backend_metal_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_metal_graph_compute,
    /* .event_record            = */ ggml_backend_metal_event_record,
    /* .event_wait              = */ ggml_backend_metal_event_wait,
    /* .graph_optimize          = */ ggml_backend_metal_graph_optimize,
#ifdef GGML_BACKEND_READ_BATCH
    /* .read_batch_begin        = */ ggml_backend_metal_read_batch_begin,
    /* .read_batch_end          = */ ggml_backend_metal_read_batch_end,
#endif
};

static ggml_guid_t ggml_backend_metal_guid(void) {
    static ggml_guid guid = { 0x81, 0xa1, 0x8b, 0x1e, 0x71, 0xec, 0x79, 0xed, 0x2b, 0x85, 0xdc, 0x8a, 0x61, 0x98, 0x30, 0xe6 };
    return &guid;
}

ggml_backend_t ggml_backend_metal_init(void) {
    ggml_backend_dev_t dev = ggml_backend_reg_dev_get(ggml_backend_metal_reg(), 0);
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev, false);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;
}

bool ggml_backend_is_metal(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_metal_guid());
}

void ggml_backend_metal_set_abort_callback(ggml_backend_t backend, ggml_abort_callback abort_callback, void * user_data) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_set_abort_callback(ctx, abort_callback, user_data);
}

bool ggml_backend_metal_supports_family(ggml_backend_t backend, int family) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    return ggml_metal_supports_family(ctx, family);
}

void ggml_backend_metal_capture_next_compute(ggml_backend_t backend) {
    GGML_ASSERT(ggml_backend_is_metal(backend));

    ggml_metal_t ctx = (ggml_metal_t)backend->context;

    ggml_metal_capture_next_compute(ctx);
}

// backend device

static const char * ggml_backend_metal_device_get_name(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->name;
}

static const char * ggml_backend_metal_device_get_description(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_get_props(ctx_dev)->desc;
}

static void ggml_backend_metal_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_device_get_memory(ctx_dev, free, total);
}

static enum ggml_backend_dev_type ggml_backend_metal_device_get_type(ggml_backend_dev_t dev) {
    return GGML_BACKEND_DEVICE_TYPE_GPU;

    GGML_UNUSED(dev);
}

static void ggml_backend_metal_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    props->name        = ggml_backend_metal_device_get_name(dev);
    props->description = ggml_backend_metal_device_get_description(dev);
    props->type        = ggml_backend_metal_device_get_type(dev);

    ggml_backend_metal_device_get_memory(dev, &props->memory_free, &props->memory_total);

    // zero-copy only loses when the fallback would be private VRAM: on a discrete GPU a mapped weight buffer streams weights over PCIe on every access (~25x slower decode).
    // eGPUs default to shared buffers (#17866), so their fallback is shared memory too and they keep zero-copy.
    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props((ggml_metal_device_t) dev->context);

    props->caps = {
        /* .async                = */ true,
        /* .host_buffer          = */ false,
        /* .buffer_from_host_ptr = */ props_dev->has_unified_memory || props_dev->use_shared_buffers,
        /* .events               = */ true,
        /* .mmap_support         = */ true,
    };
}

static ggml_backend_t ggml_backend_metal_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_t ctx = ggml_metal_init(ctx_dev, params != NULL && strcmp(params, "prefetch") == 0);
    if (ctx == NULL) {
        GGML_LOG_ERROR("%s: error: failed to allocate context\n", __func__);
        return NULL;
    }

    ggml_backend_t backend = (ggml_backend_t) malloc(sizeof(ggml_backend));

    *backend = {
        /* .guid      = */ ggml_backend_metal_guid(),
        /* .interface = */ ggml_backend_metal_i,
        /* .device    = */ dev,
        /* .context   = */ ctx,
    };

    ggml_backend_metal_set_n_cb(backend, 1);

    return backend;

    GGML_UNUSED(params);
}

static ggml_backend_buffer_type_t ggml_backend_metal_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return props_dev->use_shared_buffers ? ggml_backend_metal_buffer_type_shared(props_dev->device) : ggml_backend_metal_buffer_type_private(props_dev->device);
}

static ggml_backend_buffer_t ggml_backend_metal_device_buffer_mapped(ggml_backend_dev_t dev, void * ptr, size_t size, size_t max_tensor_size) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_buffer_t res = ggml_metal_buffer_map(ctx_dev, ptr, size, max_tensor_size);

    const ggml_metal_device_props * props_dev = ggml_metal_device_get_props(ctx_dev);

    return ggml_backend_buffer_init(ggml_backend_metal_buffer_type_mapped(props_dev->device), ggml_backend_metal_buffer_shared_i, res, size);
}

static bool ggml_backend_metal_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_supports_op(ctx_dev, op);
}

static bool ggml_backend_metal_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    return
        buft->device == dev && (
        buft->iface.get_name == ggml_backend_metal_buffer_type_shared_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_private_get_name ||
        buft->iface.get_name == ggml_backend_metal_buffer_type_mapped_get_name);

    GGML_UNUSED(dev);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_metal_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    int min_batch = ggml_metal_device_get_props(ctx_dev)->op_offload_min_batch_size;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    // mixed execution takes every batch above its host path: the CPU backend would compute the
    // resident experts too, and a bounded host cache leaves it no bytes. The device read the
    // threshold before a plan could lower it
    if (tosh_dmoe_enabled() || tosh_hostcache_on()) min_batch = std::min(min_batch, TOSH_DMOE_MAX_TOKENS + 1);
#endif
    return (op->op == GGML_OP_MUL_MAT ||
            op->op == GGML_OP_MUL_MAT_ID) &&
            get_op_batch_size(op) >= min_batch;
}

static ggml_backend_event_t ggml_backend_metal_device_event_new(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t event = ggml_metal_device_event_init(ctx_dev);
    GGML_ASSERT(event);

    ggml_backend_event_t ev = new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };

    return ev;
}

static void ggml_backend_metal_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t ev = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_free(ctx_dev, ev);

    delete event;
}

static void ggml_backend_metal_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    ggml_metal_event_t evt = (ggml_metal_event_t)event->context;

    ggml_metal_device_event_synchronize(ctx_dev, evt);
}

static ggml_backend_device_i ggml_backend_metal_device_i = {
    /* .get_name             = */ ggml_backend_metal_device_get_name,
    /* .get_description      = */ ggml_backend_metal_device_get_description,
    /* .get_memory           = */ ggml_backend_metal_device_get_memory,
    /* .get_type             = */ ggml_backend_metal_device_get_type,
    /* .get_props            = */ ggml_backend_metal_device_get_props,
    /* .init_backend         = */ ggml_backend_metal_device_init_backend,
    /* .get_buffer_type      = */ ggml_backend_metal_device_get_buffer_type,
    /* .get_host_buffer_type = */ NULL,
    /* .buffer_from_host_ptr = */ ggml_backend_metal_device_buffer_mapped,
    /* .supports_op          = */ ggml_backend_metal_device_supports_op,
    /* .supports_buft        = */ ggml_backend_metal_device_supports_buft,
    /* .offload_op           = */ ggml_backend_metal_device_offload_op,
    /* .event_new            = */ ggml_backend_metal_device_event_new,
    /* .event_free           = */ ggml_backend_metal_device_event_free,
    /* .event_synchronize    = */ ggml_backend_metal_device_event_synchronize,
};

// backend registry

struct ggml_backend_metal_reg {
    std::vector<ggml_backend_dev_t> devices;
};

typedef struct ggml_backend_metal_reg * ggml_backend_metal_reg_t;

static ggml_backend_metal_reg_t ggml_backend_metal_reg_init(void) {
    ggml_backend_metal_reg_t ctx = new struct ggml_backend_metal_reg;

    return ctx;
}

static void ggml_backend_metal_reg_free(ggml_backend_metal_reg_t ctx) {
    delete ctx;
}

struct ggml_backend_metal_reg_deleter {
    void operator()(ggml_backend_metal_reg_t ctx) {
        ggml_backend_metal_reg_free(ctx);
    }
};

typedef std::unique_ptr<struct ggml_backend_metal_reg, ggml_backend_metal_reg_deleter> ggml_backend_metal_reg_ptr;

static const char * ggml_backend_metal_reg_get_name(ggml_backend_reg_t reg) {
    return GGML_METAL_NAME;

    GGML_UNUSED(reg);
}

static size_t ggml_backend_metal_reg_device_count(ggml_backend_reg_t reg) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_metal_reg_device_get(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_metal_reg_t ctx = (ggml_backend_metal_reg_t)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature g_ggml_backend_metal_features[] = {
#if defined(GGML_METAL_EMBED_LIBRARY)
    { "EMBED_LIBRARY", "1" },
#endif
    { NULL, NULL },
};

static ggml_backend_feature * ggml_backend_metal_get_features(ggml_backend_reg_t reg) {
    return g_ggml_backend_metal_features;

    GGML_UNUSED(reg);
}

// test/tune-only override for the FA vec (Q, NE) selection, reached via proc_address.
static void ggml_backend_metal_tuning_set_fa_vec_override(int Q, int NE) {
    ggml_metal_tuning::fa_vec_set_override({ (int8_t) Q, (int8_t) NE });
}

static void ggml_backend_metal_tuning_clear_fa_vec_override(void) {
    ggml_metal_tuning::fa_vec_clear_override();
}

static int ggml_backend_metal_tuning_fa_vec_ne11_bucket(int64_t ne11) {
    return ggml_metal_tuning::fa_vec_ne11_bucket(ne11);
}

static int ggml_backend_metal_tuning_fa_vec_ne01_bucket(int64_t ne01) {
    return ggml_metal_tuning::fa_vec_ne01_bucket(ne01);
}

static int ggml_backend_metal_tuning_fa_vec_baseline_ne(int dk, int dv) {
    return ggml_metal_tuning::fa_vec_baseline_ne(dk, dv);
}

static const char * ggml_backend_metal_tuning_device_token(ggml_backend_dev_t dev) {
    ggml_metal_device_t ctx_dev = (ggml_metal_device_t)dev->context;

    return ggml_metal_device_id_token(ggml_metal_device_get_props(ctx_dev)->device_id);
}

// generic fusion debugging API (ad-hoc proc-address mechanism): the test resolves the device
// fusion context once and passes that opaque handle to the rest of the functions
typedef void * ggml_backend_fusion_t;

static ggml_backend_fusion_t ggml_backend_metal_fusion_get(ggml_backend_dev_t dev) {
    return ggml_metal_device_get_fusion_info((ggml_metal_device_t)dev->context);
}

static void ggml_backend_metal_fusion_stats_init(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_init((struct ggml_metal_fusion_info *) finfo);
}

static void ggml_backend_metal_fusion_stats_reset(ggml_backend_fusion_t finfo) {
    ggml_metal_fusion_info_stats_reset((struct ggml_metal_fusion_info *) finfo);
}

static int ggml_backend_metal_fusion_stats_get(ggml_backend_fusion_t finfo, const char ** labels, uint64_t * counts, int n) {
    return ggml_metal_fusion_info_stats_get((struct ggml_metal_fusion_info *) finfo, labels, counts, n);
}

static void ggml_backend_metal_fusion_set_enabled(ggml_backend_fusion_t finfo, bool enabled) {
    ggml_metal_fusion_info_set_enabled((struct ggml_metal_fusion_info *) finfo, enabled);
}

// the shared buffer type is host-resident on a discrete GPU when the expert cache asks for
// it, so offer it by name and let -ot place the expert banks there
// Tensor split reduces after every subgraph. The meta backend's generic path spends a full
// graph submission per device on each of those adds; here the copy and the add are issued
// directly, which is what the CUDA and SYCL backends already do for themselves.
struct ggml_backend_metal_comm_context {
    std::vector<ggml_backend_t>         backends;
    // scratch per device and reduction step: reusing one per device would let a step
    // overwrite what the previous one is still reducing
    std::vector<ggml_backend_buffer_t>  tmp;
    std::vector<size_t>                 tmp_size;
    // f16 transport staging (TOSH_MGPU_STAGE_F16): same slotting as the f32 scratch
    std::vector<ggml_backend_buffer_t>  tmp_f16;
    std::vector<size_t>                 tmp_f16_size;
    size_t                              n_steps = 0;
    bool                                hierarchical = false;
    uint64_t                            seq_round = 0;
    size_t                              leaders[2] = {};
    size_t                              followers[2] = {};
};

// A four-device machine can expose two peer groups of two devices. A flat butterfly starts
// with four transfers across the slow boundary. Reduce inside each group first, exchange only
// the two group leaders, then copy the result back to their followers: two cross-group
// transfers instead of four, and four adds instead of eight.
static bool ggml_backend_metal_comm_hierarchical_init(ggml_backend_metal_comm_context * comm_ctx) {
    if (comm_ctx->backends.size() != 4) {
        return false;
    }

    uint64_t groups[2] = {};
    size_t members[2][2] = {};
    size_t counts[2] = {};
    size_t n_groups = 0;

    for (size_t i = 0; i < comm_ctx->backends.size(); ++i) {
        ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[i]->context;
        const uint64_t group = ggml_metal_peer_group_id(ctx);
        if (group == 0) {
            return false;
        }

        size_t j = 0;
        while (j < n_groups && groups[j] != group) {
            ++j;
        }
        if (j == n_groups) {
            if (n_groups == 2) {
                return false;
            }
            groups[n_groups++] = group;
        }
        if (counts[j] == 2) {
            return false;
        }
        members[j][counts[j]++] = i;
    }

    if (n_groups != 2 || counts[0] != 2 || counts[1] != 2) {
        return false;
    }

    for (size_t j = 0; j < 2; ++j) {
        comm_ctx->leaders[j] = members[j][0];
        comm_ctx->followers[j] = members[j][1];
    }
    return true;
}

// Butterfly reduction over the largest power of two that fits, same schedule as the generic
// path: fold the devices past it, exchange, then copy the result back to them.
static size_t ggml_backend_metal_comm_offset_max(size_t n_backends) {
    size_t offset = n_backends/2;
    while ((offset & (offset - 1)) != 0) {
        offset--;
    }
    return offset;
}

static size_t ggml_backend_metal_comm_n_steps(size_t n_backends) {
    const size_t offset_max = ggml_backend_metal_comm_offset_max(n_backends);

    size_t n_steps = n_backends > 2*offset_max ? 1 : 0;
    for (size_t offset = offset_max; offset >= 1; offset /= 2) {
        n_steps++;
    }

    return n_steps;
}

static void * ggml_backend_metal_comm_init(ggml_backend_t * backends, size_t n_backends) {
    if (n_backends < 2 || getenv("TOSH_MGPU_COMM_DISABLE") != NULL) {
        return nullptr;
    }

    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_metal(backends[i])) {
            return nullptr;
        }
    }

    auto * comm_ctx = new ggml_backend_metal_comm_context;
    comm_ctx->backends.assign(backends, backends + n_backends);
    comm_ctx->n_steps = ggml_backend_metal_comm_n_steps(n_backends);
    comm_ctx->tmp.assign(n_backends*comm_ctx->n_steps, nullptr);
    comm_ctx->tmp_size.assign(n_backends*comm_ctx->n_steps, 0);

    const char * hierarchical = getenv("TOSH_MGPU_HIERARCHICAL");
    const char * peer = getenv("TOSH_MGPU_PEER");
    const bool peer_enabled = peer != nullptr && peer[0] == '1';
    if (peer_enabled && (hierarchical == nullptr || hierarchical[0] != '0')) {
        comm_ctx->hierarchical = ggml_backend_metal_comm_hierarchical_init(comm_ctx);
        if (comm_ctx->hierarchical) {
            GGML_LOG_WARN("%s: topology-aware four-device allreduce enabled (%zu,%zu) <-> (%zu,%zu)\n",
                          __func__, comm_ctx->leaders[0], comm_ctx->followers[0],
                          comm_ctx->leaders[1], comm_ctx->followers[1]);
        } else if (hierarchical != nullptr) {
            GGML_LOG_WARN("%s: topology-aware allreduce requested, but the devices are not two peer groups of two\n",
                          __func__);
        }
    } else if (!peer_enabled && hierarchical != nullptr && hierarchical[0] == '1') {
        GGML_LOG_WARN("%s: topology-aware allreduce requires peer transfers\n", __func__);
    }

    if (n_backends > 2) {
        GGML_LOG_WARN("%s: reducing across %zu devices in %zu steps\n", __func__, n_backends, comm_ctx->n_steps);
    }

    return comm_ctx;
}

static void ggml_backend_metal_comm_free(void * comm_ctx_v) {
    auto * comm_ctx = (ggml_backend_metal_comm_context *) comm_ctx_v;
    for (auto & buf : comm_ctx->tmp) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    for (auto & buf : comm_ctx->tmp_f16) {
        if (buf != nullptr) {
            ggml_backend_buffer_free(buf);
        }
    }
    delete comm_ctx;
}

// TOSH_MGPU_ALLREDUCE_NOOP=1 keeps the split graph and every subgraph submission but performs
// no reduction and no cross-device synchronisation at all. The output is wrong by construction.
// It exists for one measurement: what tensor-split decode would cost if communication were free.
static bool ggml_backend_metal_comm_allreduce_noop(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = getenv("TOSH_MGPU_ALLREDUCE_NOOP");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_ALLREDUCE_NOOP=1, the reduction is "
                            "skipped and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

static bool ggml_backend_metal_comm_allreduce_tensor(void * comm_ctx_v, ggml_tensor ** tensors) {
    if (ggml_backend_metal_comm_allreduce_noop()) {
        return true;
    }

    auto * comm_ctx = (ggml_backend_metal_comm_context *) comm_ctx_v;

    const size_t n_backends = comm_ctx->backends.size();

    static int tosh_trace_al = -1;
    if (tosh_trace_al < 0) { tosh_trace_al = getenv("TOSH_MGPU_TRACE") ? 1 : 0; }

    if (tensors[0] == nullptr || tensors[0]->type != GGML_TYPE_F32) {
        if (tosh_trace_al) fprintf(stderr, "allreduce bail: t0=%p type=%d\n", (void*)tensors[0], tensors[0] ? (int)tensors[0]->type : -1);
        return false;
    }

    const int64_t ne = ggml_nelements(tensors[0]);
    if (ne == 0) {
        return true;
    }

    // a device with an empty slice carries the node but never computes it, so its memory is
    // stale scratch. The generic butterfly zeroes those before the add; the metal path now
    // does the same with a blit fill instead of handing the whole collective to the fallback.
    size_t zero_mask = 0;

    for (size_t i = 0; i < n_backends; i++) {
        ggml_tensor * t = tensors[i];
        if (t == nullptr || t->type != GGML_TYPE_F32 || ggml_nelements(t) != ne) {
            if (tosh_trace_al) fprintf(stderr, "allreduce bail: elem i=%zu ne=%lld\n", i, t ? (long long)ggml_nelements(t) : -1);
            return false;
        }
        if (!ggml_are_same_shape(t, tensors[0]) || !ggml_is_contiguous(t)) {
            if (tosh_trace_al) fprintf(stderr, "allreduce bail: shape/contig i=%zu %s\n", i, t->name);
            return false;
        }
        if (!ggml_backend_buffer_is_metal(t->buffer)) {
            if (tosh_trace_al) fprintf(stderr, "allreduce bail: buft i=%zu %s\n", i, t->name);
            return false;
        }
        if (!(t->flags & GGML_TENSOR_FLAG_COMPUTE)) {
            zero_mask |= (size_t) 1 << i;
        }
    }

    const size_t size = ggml_nbytes(tensors[0]);
    const size_t n_steps = comm_ctx->n_steps;

    for (size_t k = 0; k < n_backends*n_steps; k++) {
        if (comm_ctx->tmp_size[k] < size) {
            ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(comm_ctx->backends[k/n_steps], size);
            if (buf == nullptr) {
                return false;
            }
            if (comm_ctx->tmp[k] != nullptr) {
                ggml_backend_buffer_free(comm_ctx->tmp[k]);
            }
            comm_ctx->tmp[k] = buf;
            comm_ctx->tmp_size[k] = size;
        }
    }

    // stack views onto the scratch buffers, so the existing cross-device copy is reused as is
    std::vector<ggml_tensor> tmp(n_backends);
    auto view_tmp = [&](size_t i, size_t step) {
        tmp[i] = *tensors[i];
        tmp[i].buffer    = comm_ctx->tmp[i*n_steps + step];
        tmp[i].data      = ggml_backend_buffer_get_base(tmp[i].buffer);
        tmp[i].view_src  = nullptr;
        tmp[i].view_offs = 0;
        tmp[i].src[0]    = nullptr;
        tmp[i].src[1]    = nullptr;
    };

    for (size_t i = 0; i < n_backends; i++) {
        view_tmp(i, 0);
        ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[i]->context;
        if (!ggml_metal_add_inplace_supported(ctx, tensors[i], &tmp[i])) {
            if (tosh_trace_al) fprintf(stderr, "allreduce bail: add_inplace i=%zu %s op=%s\n", i, tensors[i]->name, ggml_op_name(tensors[i]->op));
            return false;
        }
    }

    // The peer copy has the higher bandwidth and the event hand-off the lower fixed cost, so
    // the batch decides: peer wins the prefill, events the decode.
    const bool prefer_events = ne / tensors[0]->ne[0] < ggml_metal_peer_min_batch();

    // TOSH_MGPU_STAGE_F16=1: carry the partials across the bridge as f16 and fold them in
    // with an upcast add. At prefill payloads the copy butterfly is fabric-bound, so halving
    // the bytes is the lever; decode is latency-bound and stays f32. All-or-nothing per
    // collective: every device must support both kernels or today's path runs unchanged.
    // bf16 was the first attempt: the scalar bf16e kernels (these cards have no bfloat type)
    // faulted the AMD driver within minutes - see .bench/bf16-stage/fault-hunt.md.
    static int stage = -1;
    if (stage < 0) {
        const char * v = getenv("TOSH_MGPU_STAGE_F16");
        stage = (v && v[0] == '1') ? 1 : 0;
    }

    // v < n_backends: inbox view, v >= n_backends: outbox view (see the registry below)
    std::vector<ggml_tensor> tmp_f16(2*n_backends);
    // the lever only exists at fabric-sized payloads; decode rounds are latency, and the two
    // extra kernel dispatches per push would cost more there than the bytes they save
    // Keep the topology-aware path in f32: its directed schedule already halves the slow-link
    // traffic, and combining two numerical/transport changes needs its own validation.
    bool stage_f16 = stage == 1 && !prefer_events && !comm_ctx->hierarchical;
    if (stage_f16) {
        for (size_t i = 0; i < n_backends; i++) {
            ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[i]->context;
            if (!ggml_metal_f16_stage_supported(ctx, tensors[i])) {
                stage_f16 = false;
                break;
            }
        }
    }
    if (stage_f16) {
        const size_t size_f16 = (size + 1)/2;
        // [0, n_backends*n_steps): inboxes, where a partner's staged partial is pulled to.
        // [n_backends*n_steps, 2*n_backends*n_steps): outboxes, where this card stages its own
        // conversion for the partner to pull. The f32 butterfly only ever stages incoming
        // partials (a sender pushes its tensor straight across), so it can share one slot space;
        // f16 stages both sides, and a shared slot means cvt overwrites the partial the
        // partner's blit just landed there - silently wrong sums and cross-device read races.
        if (comm_ctx->tmp_f16.size() < 2*n_backends*n_steps) {
            comm_ctx->tmp_f16.resize(2*n_backends*n_steps, nullptr);
            comm_ctx->tmp_f16_size.resize(2*n_backends*n_steps, 0);
        }
        for (size_t k = 0; k < 2*n_backends*n_steps; k++) {
            if (comm_ctx->tmp_f16_size[k] < size_f16) {
                ggml_backend_buffer_t buf = ggml_backend_alloc_buffer(comm_ctx->backends[(k/n_steps) % n_backends], size_f16);
                if (buf == nullptr) {
                    stage_f16 = false;
                    break;
                }
                if (comm_ctx->tmp_f16[k] != nullptr) {
                    ggml_backend_buffer_free(comm_ctx->tmp_f16[k]);
                }
                comm_ctx->tmp_f16[k]      = buf;
                comm_ctx->tmp_f16_size[k] = size_f16;
            }
        }
    }

    // v indexes the doubled slot space: inbox v, outbox v = n_backends + i
    auto view_tmp_f16 = [&](size_t v, size_t step) {
        tmp_f16[v] = *tensors[v % n_backends];
        tmp_f16[v].type     = GGML_TYPE_F16;
        tmp_f16[v].buffer   = comm_ctx->tmp_f16[v*n_steps + step];
        tmp_f16[v].data     = ggml_backend_buffer_get_base(tmp_f16[v].buffer);
        tmp_f16[v].view_src = nullptr;
        tmp_f16[v].view_offs = 0;
        tmp_f16[v].nb[0]    = 2;
        tmp_f16[v].nb[1]    = 2*tensors[v % n_backends]->ne[0];
        tmp_f16[v].nb[2]    = tmp_f16[v].nb[1]*tensors[v % n_backends]->ne[1];
        tmp_f16[v].nb[3]    = tmp_f16[v].nb[2]*tensors[v % n_backends]->ne[2];
        tmp_f16[v].src[0]   = nullptr;
        tmp_f16[v].src[1]   = nullptr;
    };

    // the add goes into the command buffer the copy already opened; kill switch because that
    // buffer stays open between the two calls
    static int fuse = -1;
    if (fuse < 0) {
        fuse = getenv("TOSH_MGPU_FUSE_ADD_DISABLE") == NULL ? 1 : 0;
    }
    const bool hold_dst = fuse == 1;

    // Decode batches reduce in one round: every card pushes its partial into every partner's
    // inbox and sums from local VRAM, instead of the butterfly's log2(n) host-staged rounds.
    // The staged round trips are pure latency at 16 KiB, which is why TP4 decode loses to one
    // card today. Encodes nothing and falls through if the hive is not peer-direct.
    if (prefer_events && n_backends >= 2 && n_backends <= 4) {
        ggml_metal_t ctxs[4];
        for (size_t i = 0; i < n_backends; i++) {
            ctxs[i] = (ggml_metal_t) comm_ctx->backends[i]->context;
        }
        if (ggml_metal_allreduce_oneshot(ctxs, tensors, n_backends, comm_ctx->seq_round + 1, (uint32_t) zero_mask)) {
            comm_ctx->seq_round++;
            return true;
        }
    }

    // The one-shot encodes its own zero fills in the merged blit encoder; only paths that
    // continue past it need the standalone ones, and they must land before the first publish.
    for (size_t i = 0; i < n_backends; i++) {
        if (zero_mask & ((size_t) 1 << i)) {
            ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[i]->context;
            if (!ggml_metal_zero_tensor(ctx, tensors[i])) {
                return false;
            }
        }
    }

    // Bailing out is only safe while no device has reduced anything: past that the generic
    // path would add the partial sums a second time.
    bool reduced = false;
    auto push = [&](size_t j_src, size_t j_dst, size_t step, bool hold, bool events) {
        view_tmp(j_dst, step);
        bool ok;
        if (stage_f16) {
            view_tmp_f16(n_backends + j_src, step);   // this card's outbox
            view_tmp_f16(j_dst, step);                // the receiving card's inbox
            ggml_metal_t ctx_src = (ggml_metal_t) comm_ctx->backends[j_src]->context;
            ok = ggml_metal_cvt_f32_f16(ctx_src, tensors[j_src], &tmp_f16[n_backends + j_src]) &&
                 ggml_backend_metal_cpy_tensor_async_ex(comm_ctx->backends[j_src], comm_ctx->backends[j_dst],
                                                        &tmp_f16[n_backends + j_src], &tmp_f16[j_dst], events, hold);
        } else {
            ok = ggml_backend_metal_cpy_tensor_async_ex(comm_ctx->backends[j_src], comm_ctx->backends[j_dst],
                                                        tensors[j_src], &tmp[j_dst], events, hold);
        }
        if (!ok && reduced) {
            // the partial sums are already folded in, so the fallback would add them a
            // second time - a failed butterfly has no safe way out
            GGML_ABORT("%s: allreduce push failed after partial reduction\n", __func__);
        }
        return ok;
    };
    auto reduce = [&](size_t j_dst) {
        ggml_metal_t ctx = (ggml_metal_t) comm_ctx->backends[j_dst]->context;
        const bool ok = stage_f16
            ? ggml_metal_add_inplace_f16_src(ctx, tensors[j_dst], &tmp_f16[j_dst])
            : ggml_metal_add_inplace_async(ctx, tensors[j_dst], &tmp[j_dst]);
        if (!ok) {
            if (reduced) {
                GGML_ABORT("%s: allreduce add failed after partial reduction\n", __func__);
            }
            return false;
        }
        reduced = true;
        return true;
    };

    if (comm_ctx->hierarchical && !prefer_events) {
        // Phase 1: one directed reduction inside each peer group. The leaders retain the
        // complete per-group sums; follower tensors may be overwritten only after phase 2.
        for (size_t j = 0; j < 2; ++j) {
            if (!push(comm_ctx->followers[j], comm_ctx->leaders[j], 0, hold_dst, false)) {
                return false;
            }
        }
        for (size_t j = 0; j < 2; ++j) {
            if (!reduce(comm_ctx->leaders[j])) {
                return false;
            }
        }

        // Phase 2: only the leaders cross the peer-group boundary. Passing no peer tensor
        // forces the destination-side staged path on machines without a direct bridge.
        const uint64_t seq_round = ++comm_ctx->seq_round;
        for (size_t j = 0; j < 2; ++j) {
            view_tmp(comm_ctx->leaders[j], 1);
        }
        for (size_t j = 0; j < 2; ++j) {
            const size_t self = comm_ctx->leaders[j];
            const size_t peer = comm_ctx->leaders[j ^ 1];
            if (!ggml_backend_metal_exchange_reduce(comm_ctx->backends[self], comm_ctx->backends[peer],
                                                    tensors[self], nullptr, &tmp[self], seq_round)) {
                GGML_ABORT("%s: hierarchical leader exchange failed after local reduction\n", __func__);
            }
        }
        reduced = true;

        // Phase 3: destination-side pulls copy the final sum back inside each peer group.
        // The copy primitives signal from the leader queue after phase 2 and make the follower
        // wait, so the next subgraph cannot observe a partial result.
        for (size_t j = 0; j < 2; ++j) {
            const size_t src = comm_ctx->leaders[j];
            const size_t dst = comm_ctx->followers[j];
            if (!ggml_backend_metal_cpy_tensor_async_ex(comm_ctx->backends[src], comm_ctx->backends[dst],
                                                        tensors[src], tensors[dst], /*prefer_events =*/ false,
                                                        /*hold_dst =*/ false)) {
                GGML_ABORT("%s: hierarchical local broadcast failed\n", __func__);
            }
        }
        return true;
    }

    const size_t offset_max = ggml_backend_metal_comm_offset_max(n_backends);

    size_t step = 0;
    if (n_backends > 2*offset_max) {
        for (size_t j_src = 2*offset_max; j_src < n_backends; j_src++) {
            if (!push(j_src, j_src - 2*offset_max, step, hold_dst, prefer_events)) {
                return false;
            }
        }
        for (size_t j_src = 2*offset_max; j_src < n_backends; j_src++) {
            if (!reduce(j_src - 2*offset_max)) {
                return false;
            }
        }
        step++;
    }

    // Every exchange of a step must read the partner's tensor before its own add rewrites it,
    // so the copies all go out before any of the adds.
    static int fused = -1;
    if (fused < 0) {
        fused = getenv("TOSH_MGPU_FUSED_EXCHANGE_DISABLE") == NULL ? 1 : 0;
    }

    for (size_t offset = offset_max; offset >= 1; offset /= 2) {
        if (fused && prefer_events) {
            comm_ctx->seq_round++;
            const uint64_t seq_round = comm_ctx->seq_round;
            bool ok = true;
            for (size_t j = 0; j < 2*offset_max; j++) {
                view_tmp(j, step);
            }
            // Only the tensor-to-tensor variant needs the partner's tensor, and it is the one
            // that would thrash its view cache when a step's partner changes between rounds,
            // so it is handed the partner for a pair and nothing otherwise. The default path
            // publishes into an inbox instead and engages at any device count: on four dies it
            // takes the two intra-card links and leaves the cross-card ones to the staged path,
            // which is what a two-card TensorMesh row already relies on.
            const bool pair = 2*offset_max == 2;
            // links and their staging blocks are created lazily; two threads racing through
            // exchange_reduce on the same pair would both try to create the shared link, so
            // materialize and reserve every directed pair of this round serially first
            for (size_t j = 0; j < 2*offset_max && ok; j++) {
                ggml_metal_t ctx_j    = (ggml_metal_t) comm_ctx->backends[j]->context;
                ggml_metal_t ctx_peer = (ggml_metal_t) comm_ctx->backends[j ^ offset]->context;
                ok = ggml_metal_xdev_prepare(ctx_j, ctx_peer, size);
            }
            if (!ok) {
                return false;
            }
            // TOSH_MGPU_SUBMIT_ORDER, benchmark only: 0 normal, 1 reversed, 2 alternating.
            // With the encodes on GCD threads the order is advisory, and the modes only mean
            // anything for the serial path below (which mirrors pre-parallelism behaviour).
            static int order = -1;
            if (order < 0) {
                const char * v = getenv("TOSH_MGPU_SUBMIT_ORDER");
                order = v ? atoi(v) : 0;
            }
            const bool flip = order == 1 || (order == 2 && (seq_round & 1));

            // TOSH_MGPU_COMM_PARALLEL=1 fans the round's exchanges out on GCD threads.
            // Default off: measured worse at both device counts (TP2 41 vs 57, TP4 36.5 vs
            // 38.0, with ±9 t/s variance) - the per-card encodes are too short to pay for
            // the thread handoff, and concurrent event encoding on peer-group devices shows
            // driver contention. Kept because the failure mode is a clue for later work.
            static int comm_parallel = -1;
            if (comm_parallel < 0) {
                const char * s_cp = getenv("TOSH_MGPU_COMM_PARALLEL");
                comm_parallel = s_cp ? atoi(s_cp) : 0;
            }
#ifdef __APPLE__
            if (comm_parallel) {
                // C++ blocks capture by value, so hand the block only pointers and scalars,
                // same pattern the meta layer's parallel encode uses
                std::atomic<bool> failed{ false };
                std::atomic<bool> * p_failed = &failed;
                ggml_tensor * tmp_data = tmp.data();
                dispatch_apply(2*offset_max, DISPATCH_APPLY_AUTO, ^(size_t jj) {
                    if (p_failed->load(std::memory_order_relaxed)) {
                        return;
                    }
                    const size_t j = flip ? (2*offset_max - 1 - jj) : jj;
                    if (!ggml_backend_metal_exchange_reduce(comm_ctx->backends[j], comm_ctx->backends[j ^ offset],
                                                            tensors[j], pair ? tensors[j ^ offset] : nullptr,
                                                            &tmp_data[j], seq_round)) {
                        p_failed->store(true, std::memory_order_relaxed);
                    }
                });
                ok = !failed.load(std::memory_order_relaxed);
            } else
#endif
            for (size_t jj = 0; jj < 2*offset_max && ok; jj++) {
                const size_t j = flip ? (2*offset_max - 1 - jj) : jj;
                ok = ggml_backend_metal_exchange_reduce(comm_ctx->backends[j], comm_ctx->backends[j ^ offset],
                                                        tensors[j], pair ? tensors[j ^ offset] : nullptr,
                                                        &tmp[j], seq_round);
            }
            if (ok) {
                reduced = true;
                step++;
                continue;
            }
            if (!ok) {
                return false;
            }
        }
        for (size_t j = 0; j < 2*offset_max; j++) {
            if (!push(j, j ^ offset, step, hold_dst, prefer_events)) {
                return false;
            }
        }
        for (size_t j = 0; j < 2*offset_max; j++) {
            if (!reduce(j)) {
                return false;
            }
        }
        step++;
    }

    for (size_t j = 2*offset_max; j < n_backends; j++) {
        const size_t j_src = j - 2*offset_max;
        // past the reduction, so there is no fallback left if this copy fails
        if (!ggml_backend_metal_cpy_tensor_async_ex(comm_ctx->backends[j_src], comm_ctx->backends[j],
                                                    tensors[j_src], tensors[j], prefer_events,
                                                    /*hold_dst =*/ false)) {
            GGML_ABORT("%s: allreduce tail copy failed\n", __func__);
        }
    }

    return true;
}

static void * ggml_backend_metal_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_metal_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_metal_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_metal_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_metal_get_features;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_set_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_set_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_clear_fa_vec_override") == 0) {
        return (void *)ggml_backend_metal_tuning_clear_fa_vec_override;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne11_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne11_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_ne01_bucket") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_ne01_bucket;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_fa_vec_baseline_ne") == 0) {
        return (void *)ggml_backend_metal_tuning_fa_vec_baseline_ne;
    }
    if (strcmp(name, "ggml_backend_metal_tuning_device_token") == 0) {
        return (void *)ggml_backend_metal_tuning_device_token;
    }
    // generic fusion debugging API (ad-hoc proc-address mechanism, not part of the official
    // ggml backend interface yet; a backend that adopts it exports these exact names)
    if (strcmp(name, "ggml_backend_fusion_get") == 0) {
        return (void *)ggml_backend_metal_fusion_get;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_init") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_init;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_reset") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_reset;
    }
    if (strcmp(name, "ggml_backend_fusion_stats_get") == 0) {
        return (void *)ggml_backend_metal_fusion_stats_get;
    }
    if (strcmp(name, "ggml_backend_fusion_set_enabled") == 0) {
        return (void *)ggml_backend_metal_fusion_set_enabled;
    }

    return NULL;

    GGML_UNUSED(reg);
}

static ggml_backend_reg_i ggml_backend_metal_reg_i = {
    /* .get_name         = */ ggml_backend_metal_reg_get_name,
    /* .get_device_count = */ ggml_backend_metal_reg_device_count,
    /* .get_device       = */ ggml_backend_metal_reg_device_get,
    /* .get_proc_address = */ ggml_backend_metal_get_proc_address,
};

static ggml_backend_dev_t ggml_backend_metal_device_init(ggml_backend_reg_t reg, int device) {
    return new ggml_backend_device {
        /* .iface   = */ ggml_backend_metal_device_i,
        /* .reg     = */ reg,
        /* .context = */ ggml_metal_device_get(device, g_devices),
    };
}

static void ggml_backend_metal_device_free(ggml_backend_dev_t dev) {
    delete dev;
}

struct ggml_backend_device_deleter {
    void operator()(ggml_backend_dev_t ctx) {
        ggml_backend_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_backend_device, ggml_backend_device_deleter> ggml_backend_device_ptr;

ggml_backend_reg_t ggml_backend_metal_reg(void) {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);

        const char * env = getenv("GGML_METAL_DEVICES");
        if (env) {
            g_devices = atoi(env);
        }

        // an explicit device list registers one slot per listed physical GPU
        const char * env_list = getenv("GGML_METAL_DEVICE_LIST");
        if (env_list && *env_list) {
            g_devices = 1;
            for (const char * p = env_list; *p; ++p) {
                if (*p == ',') {
                    ++g_devices;
                }
            }
        }

        static std::vector<ggml_backend_device_ptr> devs;

        if (!initialized) {
            // workaround macOS limitation (kIOGPUCommandBufferCallbackErrorImpactingInteractivity) until proper fix becomes possible
            // ref: https://github.com/ggml-org/llama.cpp/issues/20141#issuecomment-4272947703
            setenv("AGX_RELAX_CDM_CTXSTORE_TIMEOUT", "1", true);

            static ggml_backend_metal_reg_ptr reg_ctx(ggml_backend_metal_reg_init());

            for (int i = 0; i < g_devices; ++i) {
                // i is the logical slot (0..g_devices-1) used everywhere as the
                // device index (buffer types, naming). Physical GPU selection lives
                // in ggml_metal_device_init, which maps the slot to a real Metal
                // device via GGML_METAL_DEVICE_INDEX (see there).
                auto * dev = ggml_backend_metal_device_init(&reg, i);
                devs.emplace_back(dev);

                reg_ctx->devices.push_back(dev);
            }

            reg = {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_metal_reg_i,
                /* .context     = */ reg_ctx.get(),
            };
        }

        initialized = true;
    }

    return &reg;
}

GGML_BACKEND_DL_IMPL(ggml_backend_metal_reg)
