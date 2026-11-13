#include "ggml-metal-device.h"
#include "ggml-metal-common.h"

#include "ggml-metal-impl.h"
#include "ggml-metal-tuning.h"

#include "ggml-impl.h"

#include <cassert>
#include <memory>
#include <string>
#include <unordered_map>

struct ggml_metal_device_deleter {
    void operator()(ggml_metal_device_t ctx) {
        ggml_metal_device_free(ctx);
    }
};

typedef std::unique_ptr<ggml_metal_device, ggml_metal_device_deleter> ggml_metal_device_ptr;

ggml_metal_device_t ggml_metal_device_get(int device, int n_devices) {
    static std::vector<ggml_metal_device_ptr> devs;

    devs.emplace_back(ggml_metal_device_init(device, n_devices));

    return devs.back().get();
}

struct ggml_metal_pipelines {
    std::unordered_map<std::string, ggml_metal_pipeline_t> data;
};

ggml_metal_pipelines_t ggml_metal_pipelines_init(void) {
    ggml_metal_pipelines_t res = new ggml_metal_pipelines();

    return res;
}

void ggml_metal_pipelines_free(ggml_metal_pipelines_t ppls) {
    if (!ppls) {
        return;
    }

    for (auto it = ppls->data.begin(); it != ppls->data.end(); ++it) {
        ggml_metal_pipeline_free(it->second);
    }

    delete ppls;
}

void ggml_metal_pipelines_add(ggml_metal_pipelines_t ppls, const char * name, ggml_metal_pipeline_t pipeline) {
    ppls->data[name] = pipeline;
}

ggml_metal_pipeline_t ggml_metal_pipelines_get(ggml_metal_pipelines_t ppls, const char * name) {
    if (ppls->data.find(name) == ppls->data.end()) {
        return nullptr;
    }

    return ppls->data[name];
}

struct ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_base(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op) {
        case GGML_OP_ADD_ID: op_str = "add_id"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s", op_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// bf16 read as raw bits and widened by hand. Cards without the Metal 3 family have no bfloat
// at all, and the ones that do have no arithmetic for it either, so the manual path is also
// the faster one wherever the simdgroup matrix unit is missing.
static bool ggml_metal_bf16_manual(ggml_metal_library_t lib) {
    const struct ggml_metal_device_props * props =
        ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    // escape hatch for the A/B; a card with no bfloat has no other path
    static const bool off = TOSH_ENV("TOSH_BF16_MANUAL_DISABLE") != nullptr;

    return !props->has_bfloat || (!props->has_simdgroup_mm && !off);
}

static const char * ggml_metal_tname(ggml_metal_library_t lib, ggml_type t) {
    if (t == GGML_TYPE_BF16 && ggml_metal_bf16_manual(lib)) {
        return "bf16e";
    }

    return ggml_type_name(t);
}

// Only the pairs the manual kernels are instantiated for: bf16 weights against f32.
static const char * ggml_metal_tname_pair(ggml_metal_library_t lib, ggml_type t, ggml_type other) {
    if (t == GGML_TYPE_BF16 && other == GGML_TYPE_F32) {
        return ggml_metal_tname(lib, t);
    }

    return ggml_type_name(t);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst, bool contig) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cpy_%s%s_%s", contig ? "contig_" : "", ggml_metal_tname_pair(lib, tsrc, tdst), ggml_metal_tname_pair(lib, tdst, tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

// f32 and f16 only
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cpy_flat(ggml_metal_library_t lib, ggml_type tsrc, ggml_type tdst) {
    char name[256];

    snprintf(name, 256, "kernel_cpy_flat_%s_%s", ggml_type_name(tsrc), ggml_type_name(tdst));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_1d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, sizeof(base), "kernel_pool_1d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, sizeof(name), "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pool_2d(ggml_metal_library_t lib, const ggml_tensor * op, ggml_op_pool op_pool) {
    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 && op->src[0]->type == op->type);

    const char * pool_str = "undefined";
    switch (op_pool) {
        case GGML_OP_POOL_AVG: pool_str = "avg"; break;
        case GGML_OP_POOL_MAX: pool_str = "max"; break;
        default: GGML_ASSERT(false && "not implemented");
    };

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pool_2d_%s_%s", pool_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_get_rows(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_get_rows_%s", ggml_metal_tname(lib, tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_set_rows(ggml_metal_library_t lib, const ggml_tensor * op, bool wave64) {
    char base[256];
    char name[256];

    const auto tsrc = op->src[0]->type;
    const auto tidx = op->src[1]->type;
    const auto tdst = op->type;

    // TurboQuant store kernels are named kernel_set_rows_turbo{2,3,4}_{i32,i64}
    // (WHT + PolarQuant quantize on write), distinct from the generic quant path.
    if (tdst == GGML_TYPE_TURBO2_0 || tdst == GGML_TYPE_TURBO3_0 || tdst == GGML_TYPE_TURBO4_0) {
        const char * tn = tdst == GGML_TYPE_TURBO2_0 ? "turbo2" : (tdst == GGML_TYPE_TURBO3_0 ? "turbo3" : "turbo4");
        const char * idx = tidx == GGML_TYPE_I64 ? "i64" : "i32";
        snprintf(base, 256, "kernel_set_rows_%s_%s%s", tn, idx,
                wave64 && tdst != GGML_TYPE_TURBO2_0 ? "_w64" : "");
    } else {
        snprintf(base, 256, "kernel_set_rows_%s_%s_%s", ggml_type_name(tsrc), ggml_type_name(tidx), ggml_type_name(tdst));
    }
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_diag(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int n = op->src[0]->ne[0];

    snprintf(base, 256, "kernel_diag_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_n=%d", base, n);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.nsg  = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_repeat_%s", ggml_type_name(tsrc));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_concat(ggml_metal_library_t lib, ggml_type tsrc) {
    char base[256];
    char name[256];

    const auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    // only the block types that are not a multiple of four bytes have an aligned variant
    const bool align = props->needs_aligned_loads &&
        (tsrc == GGML_TYPE_Q4_0 || tsrc == GGML_TYPE_Q5_0 || tsrc == GGML_TYPE_Q8_0);

    snprintf(base, 256, "kernel_concat_%s%s", ggml_type_name(tsrc), align ? "_al" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_unary(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SCALE:      op_num = OP_UNARY_NUM_SCALE;      break;
        case GGML_OP_FILL:       op_num = OP_UNARY_NUM_FILL;       break;
        case GGML_OP_CLAMP:      op_num = OP_UNARY_NUM_CLAMP;      break;
        case GGML_OP_SQR:        op_num = OP_UNARY_NUM_SQR;        break;
        case GGML_OP_SQRT:       op_num = OP_UNARY_NUM_SQRT;       break;
        case GGML_OP_SIN:        op_num = OP_UNARY_NUM_SIN;        break;
        case GGML_OP_COS:        op_num = OP_UNARY_NUM_COS;        break;
        case GGML_OP_LOG:        op_num = OP_UNARY_NUM_LOG;        break;
        case GGML_OP_LEAKY_RELU: op_num = OP_UNARY_NUM_LEAKY_RELU; break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_TANH:        op_num = OP_UNARY_NUM_TANH;        break;
                case GGML_UNARY_OP_RELU:        op_num = OP_UNARY_NUM_RELU;        break;
                case GGML_UNARY_OP_SIGMOID:     op_num = OP_UNARY_NUM_SIGMOID;     break;
                case GGML_UNARY_OP_GELU:        op_num = OP_UNARY_NUM_GELU;        break;
                case GGML_UNARY_OP_GELU_ERF:    op_num = OP_UNARY_NUM_GELU_ERF;    break;
                case GGML_UNARY_OP_GELU_QUICK:  op_num = OP_UNARY_NUM_GELU_QUICK;  break;
                case GGML_UNARY_OP_SILU:        op_num = OP_UNARY_NUM_SILU;        break;
                case GGML_UNARY_OP_ELU:         op_num = OP_UNARY_NUM_ELU;         break;
                case GGML_UNARY_OP_NEG:         op_num = OP_UNARY_NUM_NEG;         break;
                case GGML_UNARY_OP_ABS:         op_num = OP_UNARY_NUM_ABS;         break;
                case GGML_UNARY_OP_SGN:         op_num = OP_UNARY_NUM_SGN;         break;
                case GGML_UNARY_OP_STEP:        op_num = OP_UNARY_NUM_STEP;        break;
                case GGML_UNARY_OP_HARDSWISH:   op_num = OP_UNARY_NUM_HARDSWISH;   break;
                case GGML_UNARY_OP_HARDSIGMOID: op_num = OP_UNARY_NUM_HARDSIGMOID; break;
                case GGML_UNARY_OP_EXP:         op_num = OP_UNARY_NUM_EXP;         break;
                case GGML_UNARY_OP_SOFTPLUS:    op_num = OP_UNARY_NUM_SOFTPLUS;    break;
                case GGML_UNARY_OP_EXPM1:       op_num = OP_UNARY_NUM_EXPM1;       break;
                case GGML_UNARY_OP_FLOOR:       op_num = OP_UNARY_NUM_FLOOR;       break;
                case GGML_UNARY_OP_CEIL:        op_num = OP_UNARY_NUM_CEIL;        break;
                case GGML_UNARY_OP_ROUND:       op_num = OP_UNARY_NUM_ROUND;       break;
                case GGML_UNARY_OP_TRUNC:       op_num = OP_UNARY_NUM_TRUNC;       break;
                case GGML_UNARY_OP_XIELU:       op_num = OP_UNARY_NUM_XIELU;       break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;
    const bool is_cnt = ggml_is_contiguous(op->src[0]) && ggml_nelements(op) < 32768;

    snprintf(base, 256, "kernel_unary_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_cnt=%d", base, op_num, is_cnt);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_UNARY + 0);
        ggml_metal_cv_set_bool (cv, is_cnt, FC_UNARY + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_cnt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_glu(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));

    char base[256];
    char name[256];

    const char * op_str = "undefined";
    switch (op->op) {
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:        op_str = "reglu";        break;
                case GGML_GLU_OP_GEGLU:        op_str = "geglu";        break;
                case GGML_GLU_OP_SWIGLU:       op_str = "swiglu";       break;
                case GGML_GLU_OP_SWIGLU_OAI:   op_str = "swiglu_oai";   break;
                case GGML_GLU_OP_GEGLU_ERF:    op_str = "geglu_erf";    break;
                case GGML_GLU_OP_GEGLU_QUICK:  op_str = "geglu_quick";  break;
                case GGML_GLU_OP_SWIGLU_CLAMP: op_str = "swiglu_clamp"; break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_%s_%s", op_str, ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_op_sum_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_sum_rows(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_SUM_ROWS: op_num = OP_SUM_ROWS_NUM_SUM_ROWS; break;
        case GGML_OP_MEAN:     op_num = OP_SUM_ROWS_NUM_MEAN;     break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    snprintf(base, 256, "kernel_sum_rows_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d", base, op_num);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_SUM_ROWS + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // one element per lane of the real SIMD width; the kernel indexes shmem[tiisg]
    res.smem = (size_t) ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width*sizeof(float);

    if (is_c4) {
        res.smem *= 4;
    }

    res.c4  = is_c4;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_blk(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_blk_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_cumsum_add(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_CUMSUM);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_cumsum_add_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_TRI);
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    const char * op_str = "tri";
    const int ttype = op->op_params[0];

    snprintf(base, 256, "kernel_%s_%s_%d", op_str, ggml_type_name(op->src[0]->type), ttype);

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(!op->src[1] || op->src[1]->type == GGML_TYPE_F16 || op->src[1]->type == GGML_TYPE_F32);

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[0]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    const ggml_type tsrc1 = op->src[1] ? op->src[1]->type : GGML_TYPE_F32;

    snprintf(base, 256, "kernel_soft_max_%s%s", ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Reduction buffer sized by the real SIMD width so buf[tiisg] fits on wave64.
    auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));
    res.smem = (size_t) props->simd_width*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer(
        ggml_metal_library_t lib,
        const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    char base[256];
    char name[256];

    const int16_t nh = op->src[0]->ne[1];

    snprintf(base, 256, "kernel_lightning_indexer_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", base, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_lightning_indexer_amd(
        ggml_metal_library_t lib,
        const ggml_tensor * op) {
    GGML_ASSERT(op->op == GGML_OP_LIGHTNING_INDEXER);

    char base[256];
    char name[256];

    const int16_t nh = op->src[0]->ne[1];

    snprintf(base, 256, "kernel_lightning_indexer_amd_%s", ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s_nh=%d", base, nh);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nh, FC_LIGHTNING_INDEXER + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_dsv4_hc(ggml_metal_library_t lib, const ggml_tensor * op) {
    char name[256];
    const char * base = nullptr;

    const bool w64 = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width == 64;

    switch (op->op) {
        case GGML_OP_DSV4_HC_COMB:
            base = "kernel_dsv4_hc_comb_f32";
            snprintf(name, 256, "%s", base);
            break;
        case GGML_OP_DSV4_HC_PRE:
            if (w64) {
                base = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32_w64" : "kernel_dsv4_hc_pre_f32_w64";
            } else {
                base = ggml_get_op_params_i32(op, 1) != 0 ? "kernel_dsv4_hc_pre_gated_f32" : "kernel_dsv4_hc_pre_f32";
            }
            snprintf(name, 256, "%s_n_hc=%d", base, (int) op->src[0]->ne[1]);
            break;
        case GGML_OP_DSV4_HC_POST:
            if (w64 && !op->src[3]) {
                base = "kernel_dsv4_hc_post_nocomb_f32_w64";
            } else {
                base = op->src[3] ? "kernel_dsv4_hc_post_f32" : "kernel_dsv4_hc_post_nocomb_f32";
            }
            snprintf(name, 256, "%s", base);
            break;
        default:
            GGML_ABORT("fatal error");
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = nullptr;

        if (op->op == GGML_OP_DSV4_HC_PRE) {
            cv = ggml_metal_cv_init();
            ggml_metal_cv_set_int32(cv, (int32_t) op->src[0]->ne[1], FC_DSV4_HC + 0);
        }

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        if (cv) {
            ggml_metal_cv_free(cv);
        }
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv(
        ggml_metal_library_t lib, const ggml_tensor * op, int32_t nc, bool use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";

    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_nc=%d_silu=%d", base, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, use_silu, FC_SSM_CONV + 1);
        ggml_metal_cv_set_int32(cv, nc,      FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_decode(ggml_metal_library_t lib, const ggml_tensor * op, bool silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_decode%s%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), silu ? "_silu" : "", suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_conv_batched(
        ggml_metal_library_t lib, const ggml_tensor * op, int ssm_conv_bs, int32_t nc, bool use_silu) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->src[1]->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    snprintf(base, 256, "kernel_ssm_conv_%s_%s_batched%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type), suffix);
    snprintf(name, 256, "%s_ssm_conv_bs=%d_nc=%d_silu=%d", base, ssm_conv_bs, nc, use_silu ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ssm_conv_bs, FC_SSM_CONV + 0);
        ggml_metal_cv_set_bool(cv, use_silu,     FC_SSM_CONV + 1);
        ggml_metal_cv_set_int32(cv, nc,          FC_SSM_CONV + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan(ggml_metal_library_t lib, const ggml_tensor * op, bool tail)  {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);

    char base[256];
    char name[256];

    const int nw  = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const int nsg = (ne00 + nw - 1)/nw;

    snprintf(base, 256, "kernel_ssm_scan_%s%s", ggml_type_name(op->src[0]->type), tail ? "_tail" : "");
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // Shared memory layout:
    // - sgptg * NW floats for partial sums (nsg * 32)
    // - sgptg floats for shared_x_dt (nsg)
    // - sgptg floats for shared_dA (nsg)
    // Total: nsg * (NW + 2) floats
    // per simdgroup: NW floats of partial sums plus one x_dt and one dA
    res.smem = GGML_PAD((nw + 2)*sizeof(float)*nsg, 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_ssm_scan_ssd_mma(ggml_metal_library_t lib, const ggml_tensor * op)  {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_ssm_scan_ssd_mma_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    // acs/exp(acs)/state-decay vectors + dtX + SAM rows + two 8x8 tiles per simdgroup
    res.smem = (3*OP_SSM_SCAN_SSD_CS +
                OP_SSM_SCAN_SSD_CS*OP_SSM_SCAN_SSD_HD +
                OP_SSM_SCAN_SSD_NSG*8*OP_SSM_SCAN_SSD_CS +
                OP_SSM_SCAN_SSD_NSG*2*8*8)*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rwkv(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const int64_t C = op->ne[0];
    const int64_t H = op->src[0]->ne[1];

    switch (op->op) {
        case GGML_OP_RWKV_WKV6:
            {
                GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv6_%s", ggml_type_name(op->src[0]->type));
            } break;
        case GGML_OP_RWKV_WKV7:
            {
                GGML_ASSERT(op->src[6]->type == GGML_TYPE_F32);
                GGML_ASSERT(C % H == 0);
                GGML_ASSERT(C / H == 64);

                snprintf(base, 256, "kernel_rwkv_wkv7_%s", ggml_type_name(op->src[0]->type));
            } break;
        default:
            GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net(ggml_metal_library_t lib, const ggml_tensor * op, bool rows, bool fold, bool raw, bool l2q, bool l2k) {
    char base[256];
    char name[256];

    // v is src[2], dimensions: S_v = ne[0], H = ne[1]
    const int ne20 = op->src[2]->ne[0]; // S_v
    const int ne21 = op->src[2]->ne[1]; // H
    const int ne30 = op->src[3]->ne[0]; // G
    // state is src[5], 4D [S_v, S_v, H_v, n_seqs] (s0 only); K is op param 0.
    const int K = ggml_get_op_params_i32(op, 0);

    const int nsg = op->src[2]->ne[0]/32;

    GGML_ASSERT(op->src[5]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->ne[0] == ne20 * ne21);
    GGML_ASSERT(ne20 % 32 == 0);

    // rows of 128 split over 16 lanes instead of 32: the per-token reductions get shorter and each
    // lane reads q and k as whole vectors. TOSH_GDN_LPR picks another split, 0 the 32-lane kernel
    static const int lpr_env = []() { const char * e = TOSH_ENV("TOSH_GDN_LPR"); return e ? atoi(e) : 16; }();
    const int lpr = (ne20 == 128 && ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1])) ? lpr_env : 0;

    if (lpr) {
        snprintf(base, 256, "kernel_gated_delta_net_%s_lpr%d", ggml_type_name(op->src[0]->type), lpr);
    } else {
        snprintf(base, 256, "kernel_gated_delta_net_%s_%d", ggml_type_name(op->src[0]->type), nsg);
    }
    snprintf(name, 256, "%s_ne20=%d_ne30=%d_K=%d_rows=%d_fold=%d_raw=%d_l2=%d%d", base, ne20, ne30, K, rows, fold, raw, l2q, l2k);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, ne20, FC_GATED_DELTA_NET + 0);
        ggml_metal_cv_set_int16(cv, ne30, FC_GATED_DELTA_NET + 1);
        ggml_metal_cv_set_int16(cv, K,    FC_GATED_DELTA_NET + 2);
        ggml_metal_cv_set_bool (cv, rows, FC_GATED_DELTA_NET + 3);
        ggml_metal_cv_set_bool (cv, fold, FC_GATED_DELTA_NET + 4);
        ggml_metal_cv_set_bool (cv, raw,  FC_GATED_DELTA_NET + 5);
        ggml_metal_cv_set_bool (cv, l2q,  FC_GATED_DELTA_NET + 6);
        ggml_metal_cv_set_bool (cv, l2k,  FC_GATED_DELTA_NET + 7);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = lpr ? -lpr : nsg;

    return res;
}

// out_prod on the manual GEMM: src0 read transposed, the 64x64 tile of 8 simdgroups, f32 accumulation
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_ta(ggml_metal_library_t lib, const ggml_tensor * op, bool cs) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const int64_t   K     = op->src[0]->ne[1];

    const bool bc_inp = K % 32 != 0;
    const bool double_buffer = TOSH_ENV("GGML_METAL_MM_DOUBLE_BUFFER_DISABLE") == nullptr;

    const int16_t ne12 = (int16_t) op->ne[2];
    const int16_t ne13 = (int16_t) op->ne[3];

    // with the column scales the gradient fits the half tile of the forward kernel and its packed accumulators
    if (cs) {
        snprintf(base, 256, "kernel_mul_mm_%s_f32", ggml_type_name(tsrc0));
    } else if (tsrc0 == GGML_TYPE_F32) {
        snprintf(base, 256, "kernel_mul_mm_f32_f32_gb");
    } else {
        snprintf(base, 256, "kernel_mul_mm_ta_%s_f32", ggml_type_name(tsrc0));
    }
    snprintf(name, 256, "%s_ta_cs=%d_bci=%d_ne12=%d_ne13=%d_db=%d", base, cs, bc_inp, ne12, ne13, double_buffer);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool (cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool (cv, false,  FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, ne12,   FC_MUL_MM + 2);
        ggml_metal_cv_set_int16(cv, ne13,   FC_MUL_MM + 3);
        ggml_metal_cv_set_int16(cv, 1,      FC_MUL_MM + 4);
        ggml_metal_cv_set_int16(cv, 1,      FC_MUL_MM + 5);
        ggml_metal_cv_set_bool (cv, true,   FC_MUL_MM + 6);
        ggml_metal_cv_set_bool (cv, false,  FC_MUL_MM + 7);
        ggml_metal_cv_set_bool (cv, double_buffer, FC_MUL_MM + 8);
        ggml_metal_cv_set_bool (cv, !cs,    FC_MUL_MM + 9);
        ggml_metal_cv_set_bool (cv, false,  FC_MUL_MM + 10);
        ggml_metal_cv_set_bool (cv, true,   FC_MUL_MM + 12);
        ggml_metal_cv_set_bool (cv, cs,     FC_MUL_MM + 13);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = 64;
    res.nr1  = 64;
    res.nsg  = 8;
    res.smem = double_buffer ? 16384 : 8192;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_gated_delta_net_back(ggml_metal_library_t lib, const char * stage) {
    char name[256];
    snprintf(name, 256, "kernel_gated_delta_net_back_%s", stage);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_turbo_wht(ggml_metal_library_t lib) {
    const char * name = "kernel_turbo_wht";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    res.nsg = 1;
    res.smem = 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_solve_tri(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    // 256 threads per threadgroup regardless of the simd width (wave64: 4 groups of 64)
    const int nw  = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const int nsg = 256/nw;
    const int n   = op->src[1]->ne[1];
    const int k   = op->src[1]->ne[0];

    snprintf(base, 256, "kernel_solve_tri_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d_n=%d_k=%d_nw=%d", base, nsg, n, k, nw);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_SOLVE_TRI + 0);
        ggml_metal_cv_set_int16(cv, n,   FC_SOLVE_TRI + 1);
        ggml_metal_cv_set_int16(cv, k,   FC_SOLVE_TRI + 2);
        ggml_metal_cv_set_int16(cv, nw,  FC_SOLVE_TRI + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = GGML_PAD(GGML_PAD(n, nw)*nsg*sizeof(float), 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_ext(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nxpsg, int r1ptg, int nr0) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;
    const int       ne12  = op->src[1]->ne[2];
    const int       r2    = ne12 / op->src[0]->ne[2];
    const int       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // q4_K keeps its quants raw and applies the scale once per chunk; the other K-quants
    // read the block header once per block via the _hdr kernels
    static const bool no_raw = TOSH_ENV("TOSH_EXT_NO_RAW") != nullptr;

    char tname[64];
    // q6_K measured slower with the header cached (its dequant is register-heavy), so it stays generic
    const bool is_khdr = tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q2_K || tsrc0 == GGML_TYPE_Q3_K;
    // staging the activation chunk in shared memory pays from three columns up: with two, half the
    // simdgroup idles in the staging loop. Measured on 64 lanes only, so it stays gated there;
    // TOSH_EXT_LDS asks for it on 32 lanes to A/B it
    const int simd_width_lds = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const bool use_lds =
        (simd_width_lds == 64 || TOSH_ENV("TOSH_EXT_LDS") != nullptr) &&
        TOSH_ENV("TOSH_EXT_LDS_DISABLE") == nullptr &&
        nr0 > 1 && r1ptg >= 3;

    if (!no_raw && tsrc0 == GGML_TYPE_Q4_K) {
        snprintf(tname, sizeof(tname), use_lds ? "%s_raw_lds" : "%s_raw", ggml_type_name(tsrc0));
    } else if (!no_raw && is_khdr) {
        snprintf(tname, sizeof(tname), "%s_hdr", ggml_type_name(tsrc0));
    } else {
        snprintf(tname, sizeof(tname), "%s", ggml_metal_tname(lib, tsrc0));
    }

    // q6_K on 64 lanes uses the wave64-shaped variant; TOSH_EXT_W64_DISABLE falls back to the generic kernel
    static const bool w64_disable = TOSH_ENV("TOSH_EXT_W64_DISABLE") != nullptr;
    const bool is_w64_q6k = !w64_disable && tsrc0 == GGML_TYPE_Q6_K && simd_width_lds == 64;
    // q8_0 reads its quads as halves on 64 lanes only: the same read is slower on 32
    const bool is_w64_q8  = !w64_disable && tsrc0 == GGML_TYPE_Q8_0 && simd_width_lds == 64 && nr0 == 1;

    if (is_w64_q6k || is_w64_q8) {
        snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d_w64", ggml_type_name(tsrc0), ggml_type_name(tsrc1), r1ptg);
    } else if (nr0 > 1) {
        snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d_nr0_%d", tname, ggml_type_name(tsrc1), r1ptg, nr0);
    } else {
        snprintf(base, 256, "kernel_mul_mv_ext_%s_%s_r1_%d", tname, ggml_type_name(tsrc1), r1ptg);
    }
    snprintf(name, 256, "%s_nsg=%d_nxpsg=%d_ne12=%d_r2=%d_r3=%d", base, nsg, nxpsg, ne12, r2, r3);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, nxpsg,          FC_MUL_MV + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV + 4);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

// threadgroups needed to fill the GPU
// TODO: dedup
static constexpr int64_t GGML_METAL_MIN_THREADGROUPS = 128;

// few-row MMA mat-mul: 8x8 simdgroup matrices for the src1 rows of ggml_metal_mul_mat_use_mma
static constexpr int     GGML_METAL_MMA_TILE          = 8;
static constexpr int     GGML_METAL_MMA_NT_MAX        = 4;
static constexpr int     GGML_METAL_MMA_SMEM_MAX      = 16384;

// simdgroups per threadgroup for at most FEW_ROWS, at most MANY_ROWS, and more src0 rows
static constexpr int64_t GGML_METAL_MMA_FEW_ROWS      = 64;
static constexpr int64_t GGML_METAL_MMA_MANY_ROWS     = 6144;
static constexpr int     GGML_METAL_MMA_NSG_FEW_ROWS  = 32;
static constexpr int     GGML_METAL_MMA_NSG_MID_ROWS  = 16;
static constexpr int     GGML_METAL_MMA_NSG_MANY_ROWS = 8;

struct ggml_metal_mma_tiling {
    int nsg; // simdgroups per threadgroup, each over a slice of K
    int nt;  // 8-row src0 tiles per threadgroup
    int rt;  // 8-row src1 tiles per threadgroup
};

// halves n while it is above limit
static int ggml_metal_halve_to_limit(int n, int64_t limit) {
    while (n > 1 && n > limit) {
        n /= 2;
    }
    return n;
}

static size_t ggml_metal_mul_mv_mma_smem(int nsg, int nt, int rt) {
    // one 8x8 float simdgroup matrix per output tile
    constexpr size_t tile_bytes = 8*8*sizeof(float);
    return (size_t) nsg*nt*rt*tile_bytes;
}

// fewer src0 rows need more simdgroups per threadgroup (a finer K split) to fill the GPU.
// the tiles become narrower when the K-slice reduction buffer is too large or when too few threadgroups fill the GPU.
static ggml_metal_mma_tiling ggml_metal_op_mul_mat_mma_tiling(const ggml_tensor * op) {
    const ggml_type type = op->src[0]->type;
    const int64_t   ne00 = op->src[0]->ne[0];
    const int64_t   ne01 = op->src[0]->ne[1];

    ggml_metal_mma_tiling res;
    res.rt = ggml_metal_mul_mv_mma_rt(op);

    const int64_t n_steps = ne00/ggml_metal_mul_mv_mma_k_step(type, res.rt);

    int nsg = GGML_METAL_MMA_NSG_MANY_ROWS;
    if (ne01 <= GGML_METAL_MMA_FEW_ROWS) {
        nsg = GGML_METAL_MMA_NSG_FEW_ROWS;
    } else if (ne01 <= GGML_METAL_MMA_MANY_ROWS) {
        nsg = GGML_METAL_MMA_NSG_MID_ROWS;
    }
    res.nsg = ggml_metal_halve_to_limit(nsg, n_steps);

    const int64_t nt_smem   = GGML_METAL_MMA_SMEM_MAX/(int64_t) ggml_metal_mul_mv_mma_smem(res.nsg, 1, res.rt);
    const int64_t nt_rows   = ne01/(GGML_METAL_MIN_THREADGROUPS*GGML_METAL_MMA_TILE);
    res.nt = ggml_metal_halve_to_limit(GGML_METAL_MMA_NT_MAX, std::min(nt_smem, nt_rows));

    return res;
}

// the pipeline for the tiling, with fewer simdgroups when the device cannot run that many threads
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma_auto(ggml_metal_library_t lib, const ggml_tensor * op, bool add) {
    ggml_metal_mma_tiling tiling = ggml_metal_op_mul_mat_mma_tiling(op);

    auto pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, add);
    while (tiling.nsg > 1 && ggml_metal_pipeline_max_theads_per_threadgroup(pipeline) < tiling.nsg*32) {
        tiling.nsg /= 2;
        pipeline = ggml_metal_library_get_pipeline_mul_mv_mma(lib, op, tiling.nsg, tiling.nt, tiling.rt, add);
    }

    return pipeline;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_mma(ggml_metal_library_t lib, const ggml_tensor * op, int nsg, int nt, int rt, bool add) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;
    const int       ne12  = op->src[1]->ne[2];
    const int       r2    = ne12 / op->src[0]->ne[2];
    const int       r3    = op->src[1]->ne[3] / op->src[0]->ne[3];

    GGML_ASSERT(ne12 <= INT16_MAX && r2 <= INT16_MAX && r3 <= INT16_MAX);

    // the specialized kernels unroll over a compile-time row length
    const int ne00 = ggml_metal_mul_mv_mma_kind(tsrc0, rt) != GGML_METAL_MMA_KIND_GEN ? op->src[0]->ne[0] : 0;

    snprintf(base, 256, "kernel_mul_mv_mma_%s_%s_nt%d_rt%d", ggml_type_name(tsrc0), ggml_type_name(tsrc1), nt, rt);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_ne00=%d_add=%d", base, nsg, ne12, r2, r3, ne00, add);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, ne00,           FC_MUL_MV_MMA + 4);
        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV_MMA + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV_MMA + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) r2,   FC_MUL_MV_MMA + 2);
        ggml_metal_cv_set_int16(cv, (int16_t) r3,   FC_MUL_MV_MMA + 3);
        ggml_metal_cv_set_bool (cv, add,            FC_MUL_MV_MMA + 5);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.smem = ggml_metal_mul_mv_mma_smem(nsg, nt, rt);

    res.nr0 = GGML_METAL_MMA_TILE*nt;
    res.nr1 = GGML_METAL_MMA_TILE*rt;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    constexpr int NRA = SZ_SIMDGROUP * N_MM_BLOCK_Y * N_MM_SIMD_GROUP_Y;
    constexpr int NRB = SZ_SIMDGROUP * N_MM_BLOCK_X * N_MM_SIMD_GROUP_X;

    const bool has_tensor    = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_tensor;
    const bool use_mm_manual = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->use_mm_manual;
    const bool use_mm_double_buffer = use_mm_manual && TOSH_ENV("GGML_METAL_MM_DOUBLE_BUFFER_DISABLE") == nullptr;

    const bool bc_out = has_tensor
        ? (op->ne[0] % NRA != 0 || op->ne[1] % NRB != 0)
        : (op->ne[0] % 64  != 0 || op->ne[1] % 32  != 0);

    GGML_ASSERT(op->src[1]->ne[2] <= INT16_MAX && op->src[1]->ne[3] <= INT16_MAX);
    const int16_t ne12 = (int16_t) op->src[1]->ne[2];
    const int16_t ne13 = (int16_t) op->src[1]->ne[3];
    const int16_t r2   = (int16_t) (ne12 / op->src[0]->ne[2]);
    const int16_t r3   = (int16_t) (ne13 / op->src[0]->ne[3]);

    // The 8x4 tile halves the simdgroups per threadgroup, so it needs enough tokens to keep
    // the GPU in threadgroups; below ~192 it loses (measured 8.7% at 64 tokens on q8_0).
    static const int wide_floor_env = []() {
        const char * s = TOSH_ENV("TOSH_MM_WIDE_FLOOR");
        return s ? atoi(s) : 0;
    }();

    // the 64-token tile rounds the batch up, so a tail of 32 or fewer pays for a half-empty
    // group that this one covers exactly. Past 128 the padding is a smaller share and the
    // grid-lookup quants lose more re-dequantizing than they save.
    // the tail keeps its own bound: it beats the 8x4 tile on a 32-token remainder at any size,
    // so it must not follow the wide floor down
    // The 192 bound was measured on 32 lanes. On 64 the tail path stops paying earlier: 128 gains
    // 1.8 to 3.6% between 130 and 160 tokens and is neutral elsewhere, while raising it costs 8%.
    // TOSH_MM_TAIL_MAX overrides it on either width.
    static const int tail_max_env = []() {
        const char * s = TOSH_ENV("TOSH_MM_TAIL_MAX");
        return s ? atoi(s) : 0;
    }();

    const int simd_width_tail = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const int tail_max = tail_max_env ? tail_max_env : (simd_width_tail == 64 ? 128 : 192);

    bool tail_32 = op->ne[1] < tail_max && (op->ne[1] % 64) >= 1 && (op->ne[1] % 64) <= 32 &&
                   TOSH_ENV("TOSH_MM_NARROW_TAIL_DISABLE") == nullptr;
    if (tail_32 && op->ne[1] > 128) {
        switch (tsrc0) {
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_S:
                tail_32 = false; break;
            default: break;
        }
    }
    const bool narrow = use_mm_manual && !has_tensor && tsrc1 == GGML_TYPE_F32 &&
                        (op->ne[1] <= 32 || tail_32);

    // wave64 wants a far higher floor: between 288 and 384 the tile loses on two models,
    // and from 448 up it wins 4 to 6%
    const int simd_width = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const bool wide_width_ok = simd_width != 64 || TOSH_ENV("TOSH_MM_WIDE_W64_DISABLE") == nullptr;
    // Both floors were derived against the 32-lane tile. The 64-lane rewrite moved the
    // crossover, and it moved by a different amount per type, so each carries its own:
    // a single number would have to pick whom to hurt (q8_0 gains 3.7% where q4_K loses 3.1%).
    // Types with no measurement of their own keep the width default.
    int wide_floor = wide_floor_env ? wide_floor_env : (simd_width == 64 ? 448 : 112);
    if (!wide_floor_env && simd_width == 64) {
        switch (tsrc0) {
            // measured one type at a time over 128/192/256/384 tokens: 192 is where the tile
            // starts paying for these, and below it every type loses (128 costs 4 to 19%)
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_IQ3_S: wide_floor = 192; break;
            // q4_0 turns one step later: at 192 it loses 10% on a 192-token prompt
            case GGML_TYPE_Q4_0: wide_floor = 256; break;
            // left at the width default because the shorter floor measured worse: f16 (-4.8%),
            // q5_0 (-1.3%), q5_1 (-1.4%), and q4_K, which gains on a 4B and loses on an 8B
            default: break;
        }
    }

    // the tile also needs enough output rows, not just enough tokens. Halving the simdgroups per
    // threadgroup costs more where the rows run out: on 64 lanes 2560 rows lose 7 to 10% at 512
    // tokens while 4096 and up keep winning
    static const int ne0_floor_env = []() {
        const char * s = TOSH_ENV("TOSH_MM_NE0_FLOOR");
        return s ? atoi(s) : 0;
    }();

    int ne0_floor = ne0_floor_env ? ne0_floor_env : (simd_width == 64 ? 4096 : 2048);
    if (!ne0_floor_env && simd_width == 64) {
        switch (tsrc0) {
            // these two win with the shorter weights the 64-lane tile now handles; q4_K and
            // q6_K measured worse there and keep the higher floor
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q4_0: ne0_floor = 2048; break;
            default: break;
        }
    }

    static const int wide_ext_floor_env = []() {
        const char * s = TOSH_ENV("TOSH_MM_WIDE_EXT_FLOOR");
        return s ? atoi(s) : 0;
    }();
    const int wide_ext_floor = wide_ext_floor_env ? wide_ext_floor_env : 2048;

    bool wide = use_mm_manual && !has_tensor && !narrow && tsrc1 == GGML_TYPE_F32 && wide_width_ok &&
                op->ne[1] >= wide_floor && (op->ne[0] >= ne0_floor || TOSH_ENV("TOSH_MM_NE0_FLOOR_DISABLE") != nullptr) &&
                TOSH_ENV("TOSH_MM_WIDE_DISABLE") == nullptr;
    if (wide) {
        // it needs 64 accumulators, so only where the dequant leaves the registers
        switch (tsrc0) {
            case GGML_TYPE_F16:
            case GGML_TYPE_BF16:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_PQ2_0:
            case GGML_TYPE_PTQ1_0: break;
            // these leave the registers the 64 accumulators need; iq3_xxs, iq2_xs,
            // iq2_xxs and iq1_m do not, so they keep the narrow tile

            case GGML_TYPE_Q6_K:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS:
                // third floor, for the types whose dequant leaves less room: it was picked
                // against the 32-lane tile too, so TOSH_MM_WIDE_EXT_FLOOR opens it for sweeping
                wide = op->ne[0] >= wide_ext_floor && TOSH_ENV("TOSH_MM_WIDE_EXT_DISABLE") == nullptr; break;
            // f32 has never been through this gate; the override opens it so the exclusion can
            // be measured instead of inherited
            default: wide = TOSH_ENV("TOSH_MM_WIDE_ALL") != nullptr && op->ne[0] >= wide_ext_floor; break;
        }
    }

    // attention matmuls read a permuted KV cache view and can exceed half range in the packed
    // accumulator. bf16 weights already reach the tile as half, so they sum like f16.
    const bool acc_f32 = use_mm_manual &&
                         (!ggml_is_contiguous(op->src[0]) ||
                          ggml_get_op_params_i32(op, 1) == GGML_HINT_GRADIENT ||
                          (tsrc0 == GGML_TYPE_BF16 && TOSH_ENV("TOSH_MM_BF16_ACC_F32") != nullptr)) &&
                         TOSH_ENV("TOSH_MM_ACC_F32_DISABLE") == nullptr;

    // the 64-lane GEMM lives in mul_mm_w64.metal: one wave is one tile there, so it needs the
    // dispatch to hand it whole waves instead of the 32-thread groups the shared kernel expects
    // iq4_nl is the one type this kernel does not suit at every batch: it costs 3.9% at 256
    // tokens and gains 1.0% at 512, so it takes a floor rather than lose the gain.
    static const int w64_floor_env = []() {
        const char * s = TOSH_ENV("TOSH_MM_W64_FLOOR");
        return s ? atoi(s) : 0;
    }();
    const int w64_floor = w64_floor_env ? w64_floor_env :
                          (tsrc0 == GGML_TYPE_IQ4_NL ? 384 : 0);

    const bool w64 = simd_width == 64 && use_mm_manual && !has_tensor &&
                     op->ne[1] >= w64_floor &&
                     TOSH_ENV("TOSH_MM_W64_DISABLE") == nullptr;

    // a gradient can pass the half range: its f32 operand goes through the bfloat tile instead
    const bool grad_bf16 = acc_f32 && !w64 && ggml_get_op_params_i32(op, 1) == GGML_HINT_GRADIENT &&
                           ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_bfloat &&
                           tsrc1 == GGML_TYPE_F32 && (tsrc0 == GGML_TYPE_F32 || tsrc0 == GGML_TYPE_F16) &&
                           !(wide && tsrc0 == GGML_TYPE_F32) &&
                           TOSH_ENV("TOSH_MM_GRAD_BF16_DISABLE") == nullptr;

    snprintf(base, 256, "kernel_mul_mm_%s_%s%s%s%s", ggml_metal_tname_pair(lib, tsrc0, tsrc1), ggml_type_name(tsrc1),
             grad_bf16 ? "_gb" : "", wide ? "_w" : "", w64 ? "_w64" : "");
    // measured on 32 lanes only; the 64-lane tile lives in its own kernel
    const bool uniform_scale = (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K) && wide && !w64 &&
                               use_mm_manual && !has_tensor && simd_width == 32 &&
                               TOSH_ENV("TOSH_MM_UNIFORM_SCALE_DISABLE") == nullptr;

    snprintf(name, 256, "%s_bci=%d_bco=%d_ne12=%d_ne13=%d_r2=%d_r3=%d_mm=%d_db=%d_af=%d_us=%d",
             base, bc_inp, bc_out, ne12, ne13, r2, r3, use_mm_manual + 2*narrow, use_mm_double_buffer, acc_f32, uniform_scale);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, narrow, FC_MUL_MM + 7);
        ggml_metal_cv_set_bool(cv, bc_inp, FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, bc_out, FC_MUL_MM + 1);
        ggml_metal_cv_set_int16(cv, ne12,  FC_MUL_MM + 2);
        ggml_metal_cv_set_int16(cv, ne13,  FC_MUL_MM + 3);
        ggml_metal_cv_set_int16(cv, r2,    FC_MUL_MM + 4);
        ggml_metal_cv_set_int16(cv, r3,    FC_MUL_MM + 5);
        ggml_metal_cv_set_bool(cv, use_mm_manual, FC_MUL_MM + 6);
        ggml_metal_cv_set_bool(cv, use_mm_double_buffer, FC_MUL_MM + 8);
        ggml_metal_cv_set_bool(cv, acc_f32, FC_MUL_MM + 9);
        ggml_metal_cv_set_bool(cv, uniform_scale, FC_MUL_MM + 10);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    if (has_tensor) {
        res.nr0 = NRA;
        res.nr1 = NRB;

        const size_t smem_a = NRA * N_MM_NK_TOTAL * sizeof(ggml_fp16_t);
        res.smem = smem_a;
    } else {
        res.nr0 = 64;
        res.nr1 = use_mm_manual ? (narrow ? 32 : 64) : 32;

        res.smem = use_mm_double_buffer ? 16384 : ((bc_out || use_mm_manual) ? 8192 : (4096 + 2048));
    }

    res.nsg = (!has_tensor && use_mm_manual) ? ((wide || narrow) ? 4 : 8) : N_MM_SIMD_GROUP_X * N_MM_SIMD_GROUP_Y;

    if (w64) {
        // same thread count as the 32-lane path, grouped into whole waves
        res.sgw = 64;
        res.nsg = (wide || narrow) ? 2 : 4;
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_pq2_swiglu(ggml_metal_library_t lib) {
    const auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    const bool    h     = props->simd_width == 64;
    const int16_t nsg   = N_SG_PQ2_0;
    const int16_t nw    = props->wave64_decode ? (int16_t) props->simd_width : 32;
    const bool    align = props->needs_aligned_loads;

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_pq2_0_swiglu_f32%s", h ? "_h" : "");
    snprintf(name, 256, "%s_nsg=%d_nw=%d_al=%d", base, nsg, nw, align);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,         FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,           FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,           FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,           FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, nw,          FC_MUL_MV + 5);
        ggml_metal_cv_set_bool (cv, false,       FC_MUL_MV + 6);
        ggml_metal_cv_set_bool (cv, false,       FC_MUL_MV + 7);
        ggml_metal_cv_set_bool (cv, align,       FC_MUL_MV + 8);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg = nsg;
    res.nr0 = h ? N_R0_PQ2_0_H : N_R0_PQ2_0;
    res.nr1 = nw; // threads per simdgroup for the dispatch

    return res;
}

// gate, up and their swiglu in one 32-lane dispatch, with the geometry of the standalone mat-vec
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_dense_swiglu(ggml_metal_library_t lib, const ggml_tensor * w) {
    const auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));

    const ggml_type type = w->type;

    int16_t nsg = 0;
    int     nr0 = 0;
    switch (type) {
        case GGML_TYPE_Q4_0: nsg = N_SG_Q4_0; nr0 = N_R0_Q4_0; break;
        case GGML_TYPE_Q4_K: nsg = N_SG_Q4_K; nr0 = N_R0_Q4_K; break;
        case GGML_TYPE_Q5_K: nsg = N_SG_Q5_K; nr0 = N_R0_Q5_K; break;
        case GGML_TYPE_Q8_0: nsg = N_SG_Q8_0; nr0 = N_R0_Q8_0; break;
        case GGML_TYPE_MXFP4: nsg = N_SG_MXFP4; nr0 = N_R0_MXFP4; break;
        default: GGML_ABORT("unsupported dense swiglu type");
    }

    // the word-wide reads the standalone q8_0 mat-vec takes for this stride
    const bool row8  = type == GGML_TYPE_Q8_0 && w->nb[1] % 4 == 0 &&
                       TOSH_ENV("TOSH_MV_ROW4_DISABLE") == nullptr && TOSH_ENV("TOSH_MV_ROW8_DISABLE") == nullptr;
    const bool align = props->needs_aligned_loads;

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mv_%s_swiglu_f32", ggml_type_name(type));
    snprintf(name, 256, "%s_nsg=%d_r8=%d_al=%d", base, nsg, row8, align);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,   FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,     FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,     FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,     FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, 32,    FC_MUL_MV + 5);
        ggml_metal_cv_set_bool (cv, row8,  FC_MUL_MV + 6);
        ggml_metal_cv_set_bool (cv, row8,  FC_MUL_MV + 7);
        ggml_metal_cv_set_bool (cv, align, FC_MUL_MV + 8);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = nsg;
    res.nr0  = nr0/2; // rows of each matrix, so a group reads what it did alone
    res.nr1  = 32;    // threads per simdgroup for the dispatch
    // q8_0 splits each row across its simdgroups and reduces gate and up rows in threadgroup memory;
    // mxfp4 keeps its value table there
    res.smem = type == GGML_TYPE_Q8_0 ? 32*sizeof(float)*nr0 : type == GGML_TYPE_MXFP4 ? 32*sizeof(float) : 0;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id_mxfp4_swiglu_oai(ggml_metal_library_t lib) {
    const char * name = "kernel_mul_mv_id_mxfp4_swiglu_oai_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, 32, FC_MUL_MV + 5);

        res = ggml_metal_library_compile_pipeline(lib, name, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nsg  = 1;
    res.nr0  = N_R0_MXFP4/2; // rows of each matrix, so a group reads what it did alone
    res.nr1  = 32;
    res.smem = 32*sizeof(float);

    return res;
}

// a pair of columns shares one read of each block, where one pass per column would read and
// unpack the weights twice. Four columns go as two pairs where that measured faster.
// TOSH_MV_NC2_PROBE=R opens the untuned widths with R rows, for measuring
static int nc2_probe() {
    static const int r = TOSH_ENV("TOSH_MV_NC2_PROBE") ? atoi(TOSH_ENV("TOSH_MV_NC2_PROBE")) : 0;
    return r;
}

int ggml_metal_mv_nc2_rows(ggml_type type, int simd_width, int ne11, int ne01, int ne00) {
    static const bool disabled = TOSH_ENV("TOSH_MV_NC2_DISABLE") != nullptr;
    if (disabled || ne11 < 2 || ne11 > 4) {
        return 0;
    }
    const bool w64 = simd_width == 64;
    // three columns in one pass, where that beat the batched kernel
    if (ne11 == 3) {
        static const bool nc3_off = TOSH_ENV("TOSH_MV_NC3_DISABLE") != nullptr;
        if (nc3_off) {
            return 0;
        }
        if (w64) {
            // q8_0 on 64 lanes: few rows want more groups, a short row stays on the batched kernel
            if (type == GGML_TYPE_Q8_0) {
                return ne01 < 2048 ? 2 : (ne00 >= 2048 || ne01 >= 8192 ? 8 : 0);
            }
            return 0;
        }
        switch (type) {
            case GGML_TYPE_Q5_K:    return 4;
            default: return 0;
        }
    }
    switch (type) {
        case GGML_TYPE_Q4_K: return w64 ? (ne11 == 2 ? 8 : 0) : 4;
        // on 32 lanes four columns of q4_0/q4_1 stay on the batched kernel, which reads them
        // faster; on 64 lanes eight rows per simdgroup beat it at two and four columns. q4_0 only
        // keeps eight for a pair where the grid comes out in whole waves or is large enough
        case GGML_TYPE_Q4_0: return w64 ? (ne11 != 2 || (ne01 % 2048 == 0 && ne01 >= 4096) || ne01 >= 16384 ? 8 : 4)
                                        : (ne11 == 2 ? 4 : 0);
        case GGML_TYPE_Q4_1: return w64 ? 8 : (ne11 == 2 ? 4 : 0);
        case GGML_TYPE_Q5_0: return w64 ? 8 : 4;
        case GGML_TYPE_Q5_1: return 4;
        // four q8_0 columns in one pass on 64 lanes; two passes of two cost twice the reads
        case GGML_TYPE_Q8_0: return w64 ? (ne11 == 4 ? (ne01 >= 2048 ? 4 : 2) : 8) : 0;
        case GGML_TYPE_IQ3_S: return w64 ? nc2_probe() : 8;
        case GGML_TYPE_IQ3_XXS: return w64 ? nc2_probe() : 8;
        // 64 lanes only gain on matrices of few rows, with two rows per simdgroup
        case GGML_TYPE_Q5_K: return w64 ? (nc2_probe() ? nc2_probe() : (ne11 == 2 && ne01 <= 5120 ? 2 : 0)) : 4;
        case GGML_TYPE_Q6_K: return w64 ? nc2_probe() : (ne11 == 2 ? 2 : 0);
        default: return 0;
    }
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const int simd_width = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;

    const char * suffix = "";
    bool w64 = false;
    char sweep_sfx[16] = {0};


    // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                if (ne00 < 32) {
                    nsg = 1;
                    nr0 = 32;
                    nr1 = 1;
                    suffix = "_short";
                } else {
                    nsg = std::min(4, (ne00 + 127) / 128);
                    nr0 = 2;
                    nr1 = 1;
                    smem = 32*sizeof(float)*nr0;
                    suffix = ne00 % 4 == 0 ? "_4" : "";
                }
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } break;
        case GGML_TYPE_PQ2_0:
            {
                nsg = N_SG_PQ2_0;
                nr0 = N_R0_PQ2_0;
                // 64 lanes are bound by instructions here: products in half with float accumulation
                // take the byte conversions off the loop
                if (simd_width == 64) {
                    suffix = "_h";
                    nr0 = N_R0_PQ2_0_H;
                }
                // a few columns share one pass over the weights; five to eight take two
                if (ne11 >= 2) {
                    nr0 = N_R0_PQ2_0_NC;
                    nr1 = ne11 <= 4 ? ne11 : (ne11 + 1)/2;
                    snprintf(sweep_sfx, sizeof(sweep_sfx), "_c%d", nr1);
                    suffix = sweep_sfx;
                }
            } break;
        case GGML_TYPE_PTQ1_0:
            {
                nsg = N_SG_PTQ1_0;
                nr0 = N_R0_PTQ1_0;
                smem = 256*2*sizeof(uint32_t); // the trit table
                // 64 lanes want more rows per lane to amortize the staged activations
                if (simd_width == 64 && ne11 == 1) {
                    nsg    = N_SG_PTQ1_0_W64;
                    nr0    = N_R0_PTQ1_0_W64;
                    suffix = "_r8";
                }
                if (ne11 >= 2) {
                    nr0 = N_R0_PTQ1_0_NC;
                    nr1 = ne11 <= 4 ? ne11 : (ne11 + 1)/2;
                    snprintf(sweep_sfx, sizeof(sweep_sfx), "_c%d", nr1);
                    suffix = sweep_sfx;
                }
            } break;
        case GGML_TYPE_Q4_0:
            {
                // eight lanes per block with four rows each: +5.8% of generation
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_Q4_0_W64_DISABLE") == nullptr) {
                    nsg    = 1;
                    nr0    = 4;
                    suffix = "_L8r4";
                    w64    = true;
                } else {
                    nsg = N_SG_Q4_0;
                    nr0 = N_R0_Q4_0;
                }
            } break;
        case GGML_TYPE_Q4_1:
            {
                // four lanes per block with four rows each: +6.2% of generation
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_Q4_1_W64_DISABLE") == nullptr) {
                    nsg    = 1;
                    nr0    = 4;
                    suffix = "_L4r4";
                    w64    = true;
                } else {
                    nsg = N_SG_Q4_1;
                    nr0 = N_R0_Q4_1;
                }
            } break;
        case GGML_TYPE_Q5_0:
            {
                // eight rows per thread measured worse on 64 lanes for this type, so the
                // inherited pair stays; TOSH_MV_Q5_0_R8 opens it again for measuring
                if (TOSH_ENV("TOSH_MV_Q5_0_R8") != nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_Q5_0;
                    nr0 = N_R0_Q5_0;
                }
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_Q5_1:
            {
                // eight rows per thread measured worse on 64 lanes for this type, so the
                // inherited pair stays; TOSH_MV_Q5_1_R8 opens it again for measuring
                if (TOSH_ENV("TOSH_MV_Q5_1_R8") != nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else if (simd_width == 32 && !ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_unified_memory &&
                           TOSH_ENV("TOSH_MV_Q5_1_L2_DISABLE") == nullptr) {
                    // two lanes per block with two rows each on discrete 32-lane cards: +10% of generation
                    nsg    = N_SG_Q5_1;
                    nr0    = 2;
                    suffix = "_L2r2";
                } else {
                    nsg = N_SG_Q5_1;
                    nr0 = N_R0_Q5_1;
                }
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_Q8_0:
            {
                // TOSH_MV_Q8_0_R8 asks for eight rows instead of the four this width takes,
                // so the three values can be compared
                if (TOSH_ENV("TOSH_MV_Q8_0_R8") != nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 4;
                    suffix = "_r4";
                } else {
                    nsg = N_SG_Q8_0;
                    nr0 = N_R0_Q8_0;
                    // a row shorter than the simdgroups' blocks leaves some with nothing to sum but zeros
                    if (simd_width == 32 && !ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_unified_memory &&
                        TOSH_ENV("TOSH_MV_Q8_0_NSG_DISABLE") == nullptr) {
                        nsg = std::max(1, std::min(nsg, (ne00/32 + 7)/8));
                    }
                }
                // the reduction buffer scales with the rows, so it follows the effective nr0
                smem = 32*sizeof(float)*nr0;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_MXFP4:
            {
                // Fewer than four rows per simdgroup falls off a cliff here: two rows measured
                // 3904 us against 122 for four on the same matrix, so both widths take four.
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_Q2_K:
            {
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_Q2_K;
                    nr0 = N_R0_Q2_K;
                }
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_Q3_K:
            {
                // A 64-lane variant with thirty-two lanes per block and the mask tables derived
                // instead of held: measured 127.2 against 126.2, so it stays behind the switch.
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_Q3K_W64") != nullptr) {
                    nsg = N_SG_Q3_K;
                    nr0 = N_R0_Q3_K;
                    suffix = "_w64";
                    break;
                }
                // 64 lanes want eight rows in one group (+7.2% tg); two groups there costs 30%
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_Q3_K;
                    nr0 = N_R0_Q3_K;
                }
            } break;
        case GGML_TYPE_Q4_K:
            {
                // small matrices: one simdgroup with 4 rows amortizes the
                // per-threadgroup setup better than two with 2. Unlike q5_K and q6_K,
                // 64 lanes do NOT want the four-row path at any size here, so the choice
                // stays by shape; TOSH_MV_Q4K_R4 opens it again for measuring
                const bool q4k_r4_wide = TOSH_ENV("TOSH_MV_Q4K_R4") != nullptr &&
                                         TOSH_ENV("TOSH_MV_Q4K_R4_DISABLE") == nullptr;
                const bool q4k_small = ne00 <= 2048 || ne01 <= 2048 || q4k_r4_wide;

                // 64 lanes have their own decomposition: sixteen cover a block instead of
                // eight, which brings the kernel from 246 registers down to 184. It does not
                // cover the small-matrix case, which still wants one group with four rows.
                // Sixteen lanes with four rows each: +5.8% over eight lanes. Sixteen lanes
                // with the inherited rows loses, so the win lives in the crossing of the two.
                if (simd_width == 64 && !q4k_small && TOSH_ENV("TOSH_MV_Q4K_W64_DISABLE") == nullptr) {
                    nsg    = 1;
                    nr0    = 4;
                    suffix = "_L16r4";
                    w64    = true;
                    break;
                }
                if (q4k_small) {
                    nsg = 1;
                    nr0 = 4;
                    suffix = "_r4";
                } else {
                    nsg = N_SG_Q4_K;
                    nr0 = N_R0_Q4_K;
                }
            } break;
        case GGML_TYPE_Q5_K:
            {
                // the shipped 2 rows per thread is a 32-lane number; 64 lanes want 8 (+15.9% tg).
                // TOSH_MV_R8 forces it on 32 too, so that width can be A/B'd without a rebuild
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_Q5_K;
                    nr0 = N_R0_Q5_K;
                }

                // Sixteen lanes per block with two rows each: measured +10.9% of generation
                // over the inherited pair, five alternating pairs. Four lanes with eight rows
                // loses, which is why the two fronts have to be walked as a grid and not one
                // at a time.
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_Q5K_W64_DISABLE") == nullptr) {
                    nsg    = 1;
                    nr0    = 2;
                    suffix = "_L16r2";
                    w64    = true;
                }
            } break;
        case GGML_TYPE_Q6_K:
            {
                // 64 lanes want four rows in one group (+16.3% tg); 32 keep the shipped pair
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 4;
                    suffix = "_r4";
                } else {
                    nsg = N_SG_Q6_K;
                    nr0 = N_R0_Q6_K;
                }
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_IQ3_XXS;
                    nr0 = N_R0_IQ3_XXS;
                }
                smem = 256*4+128;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
                smem = 1024*8;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
                smem = 2048*4;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
                smem = 2048*4;
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                // eight rows per thread measured worse on 64 lanes for this type, so the
                // inherited pair stays; TOSH_MV_IQ4_NL_R8 opens it again for measuring
                if (TOSH_ENV("TOSH_MV_IQ4_NL_R8") != nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_IQ4_NL;
                    nr0 = N_R0_IQ4_NL;
                }
                smem = 32*sizeof(float);
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                // eight rows per thread measured worse on 64 lanes for this type, so the
                // inherited pair stays; TOSH_MV_IQ4_XS_R8 opens it again for measuring
                if (TOSH_ENV("TOSH_MV_IQ4_XS_R8") != nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_IQ4_XS;
                    nr0 = N_R0_IQ4_XS;
                }
                smem = 32*sizeof(float);
                // the 64-lane twin of this kernel lives in mul_mv_w64.metal
                if (simd_width == 64 && TOSH_ENV("TOSH_MV_W64_DISABLE") == nullptr) {
                    w64 = true;
                }
                // up to three columns share one pass over the weights; four columns spill
                // the registers, so wider batches split into even groups of two or three
                if (ne11 >= 2 && suffix[0] == '\0' && TOSH_ENV("TOSH_MV_IQ4_XS_NC_DISABLE") == nullptr) {
                    const int passes = (ne11 + 2)/3;
                    nr1 = (ne11 + passes - 1)/passes;
                    if (w64) {
                        nr0 = N_R0_IQ4_XS_NC_W64;
                    }
                    snprintf(sweep_sfx, sizeof(sweep_sfx), "_c%d", nr1);
                    suffix = sweep_sfx;
                }
            } break;
        case GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int) tsrc0);
                GGML_ABORT("not implemented");
            }
    };

    GGML_ASSERT(ne12 <= INT16_MAX && ne13 <= INT16_MAX);
    const int16_t r2 = (int16_t) (ne12 / ne02);
    const int16_t r3 = (int16_t) (ne13 / ne03);

    // wave64 decode: run the mat-vec on a 64-wide simdgroup. The kernels read FC_mul_mv_nw.
    // f16/f32/q8_0 reduce through helper_mv_reduce_and_write (needs NW*nr0 floats of shmem);
    // the K-quant kernels reduce with simd_sum directly, so they only need the width.
    auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));
    int16_t nw = 32;
    if (props->wave64_decode) {
        const bool helper_reduce = (tsrc0 == GGML_TYPE_F16 || tsrc0 == GGML_TYPE_F32 ||
                                    tsrc0 == GGML_TYPE_BF16 || tsrc0 == GGML_TYPE_Q8_0);
        const bool simd_reduce   = (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K ||
                                    tsrc0 == GGML_TYPE_Q2_K || tsrc0 == GGML_TYPE_Q3_K ||
                                    tsrc0 == GGML_TYPE_Q4_0 || tsrc0 == GGML_TYPE_Q4_1 ||
                                    tsrc0 == GGML_TYPE_Q5_0 || tsrc0 == GGML_TYPE_Q5_1 ||
                                    tsrc0 == GGML_TYPE_Q1_0 || tsrc0 == GGML_TYPE_Q2_0 ||
                                    tsrc0 == GGML_TYPE_PQ2_0 || tsrc0 == GGML_TYPE_PTQ1_0 ||
                                    tsrc0 == GGML_TYPE_IQ2_XXS || tsrc0 == GGML_TYPE_IQ2_XS ||
                                    tsrc0 == GGML_TYPE_IQ3_XXS || tsrc0 == GGML_TYPE_IQ3_S ||
                                    tsrc0 == GGML_TYPE_IQ2_S || tsrc0 == GGML_TYPE_IQ1_S ||
                                    tsrc0 == GGML_TYPE_IQ1_M);
        // mxfp4 keeps its kvalues table in shmem, one float per lane
        const bool lut = (tsrc0 == GGML_TYPE_MXFP4 || tsrc0 == GGML_TYPE_IQ4_NL || tsrc0 == GGML_TYPE_IQ4_XS);
        if (helper_reduce || simd_reduce || lut) {
            nw = (int16_t) props->simd_width;
        }
        if (helper_reduce) {
            smem = (size_t) nw*sizeof(float)*nr0;
        }
        if (lut) {
            smem = (size_t) nw*sizeof(float);
        }
    }

    // q6_K's 210-byte block leaves the compiler assuming byte alignment, so it reads its quant
    // arrays a byte at a time; an even stride lets the kernel read them two bytes at a time
    const bool row4 = (op->src[0]->nb[1] % 2 == 0) && (op->src[0]->nb[2] % 2 == 0) &&
                      TOSH_ENV("TOSH_MV_ROW4_DISABLE") == nullptr;

    // A stride that is a multiple of four lets the quant arrays be read a word at a time.
    // The two widths are decided apart: 64 lanes have no last-level cache to hide the extra
    // loads and every type gains, while 32 lanes only measured a gain on q8_0.
    const bool row8_aligned = row4 && (op->src[0]->nb[1] % 4 == 0) && (op->src[0]->nb[2] % 4 == 0);
    const bool row8_width   = nw == 64 || (nw == 32 && tsrc0 == GGML_TYPE_Q8_0);
    const bool row8 = row8_aligned && row8_width && TOSH_ENV("TOSH_MV_ROW8_DISABLE") == nullptr;

    // Cards that require natural alignment read the wrong bytes from the odd blocks of the
    // 22-, 34-, 98-, 110- and 210-byte types, so the wide read splits in two there.
    const bool align = props->needs_aligned_loads;

    // the low-bit rewrites keep the standalone geometry, so every 32-lane device takes them
    {
        static const bool v2 = TOSH_ENV("TOSH_MV_V2_DISABLE") == nullptr;
        if (v2 && suffix[0] == '\0' && nw == 32 &&
            (tsrc0 == GGML_TYPE_IQ3_XXS || tsrc0 == GGML_TYPE_IQ3_S || tsrc0 == GGML_TYPE_IQ2_S ||
             tsrc0 == GGML_TYPE_IQ2_XXS || tsrc0 == GGML_TYPE_IQ2_XS || tsrc0 == GGML_TYPE_IQ1_S ||
             tsrc0 == GGML_TYPE_IQ1_M   || tsrc0 == GGML_TYPE_Q3_K  || tsrc0 == GGML_TYPE_Q2_K  ||
             tsrc0 == GGML_TYPE_Q5_K)) {
            suffix = "_v2";
        }
    }

    // Sweep hook: TOSH_SWEEP_TYPE names one ggml type and TOSH_SWEEP_NR0 / TOSH_SWEEP_NSG
    // force its rows per lane and simdgroups, so the register front can be measured without
    // rebuilding. The pipeline name carries both, so a run can be checked against what it asked.
    {
        const char * st = TOSH_ENV("TOSH_SWEEP_TYPE");

        if (st != nullptr && strcmp(st, ggml_type_name(tsrc0)) == 0) {
            const char * snr = TOSH_ENV("TOSH_SWEEP_NR0");
            const char * sns = TOSH_ENV("TOSH_SWEEP_NSG");

            const char * sln = TOSH_ENV("TOSH_SWEEP_LANES");

            if (snr != nullptr && strcmp(suffix, "_v2") == 0) {
                nr0 = atoi(snr);
                snprintf(sweep_sfx, sizeof(sweep_sfx), "_v2_r%d", nr0);
                suffix = sweep_sfx;
            } else if (snr != nullptr && sln != nullptr) {
                // cross sweep: the 32-block family carries one entry point per (lanes, rows)
                nr0 = atoi(snr);
                snprintf(sweep_sfx, sizeof(sweep_sfx), "_L%dr%d", atoi(sln), nr0);
                suffix = sweep_sfx;
                w64 = simd_width == 64;
            } else if (snr != nullptr) {
                nr0 = atoi(snr);
                snprintf(sweep_sfx, sizeof(sweep_sfx), "_r%d", nr0);
                suffix = sweep_sfx;
                w64 = simd_width == 64;
            }

            if (sns != nullptr) {
                nsg = atoi(sns);
            }
        }
    }

    if (const int rows = ggml_metal_mv_nc2_rows(tsrc0, simd_width, ne11, ne01, ne00); rows > 0) {
        nr0 = rows;
        nr1 = ne11 == 3 ? 3 : (ne11 == 4 && tsrc0 == GGML_TYPE_Q8_0 && simd_width == 64 ? 4 : 2);
        nsg = simd_width == 64 ? 1 : 2;
        static thread_local char nc_sfx[16];
        snprintf(nc_sfx, sizeof(nc_sfx), "_c%d_r%d", nr1, rows);
        suffix = nc_sfx;
        // q4_K and q5_K have a 64-lane twin; the others run their one kernel at either width
        w64 = simd_width == 64 && (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K);
        if (const char * sg = TOSH_ENV("TOSH_MV_NC2_SG")) {
            nsg = atoi(sg);
        }
    }

    char sfx[64];
    if (w64) {
        snprintf(sfx, sizeof(sfx), "%s_w64", suffix);
        suffix = sfx;
    }

    snprintf(base, 256, "kernel_mul_mv_%s_%s%s", ggml_metal_tname_pair(lib, tsrc0, tsrc1), ggml_type_name(tsrc1), suffix);
    snprintf(name, 256, "%s_nsg=%d_ne12=%d_r2=%d_r3=%d_nw=%d_a4=%d_a8=%d_al=%d", base, nsg, ne12, r2, r3, nw, row4, row8, align);

    if (TOSH_ENV("TOSH_SWEEP_DEBUG") != nullptr) {
        fprintf(stderr, "SWEEP pipeline = %s (nsg=%d nr0=%d)\n", name, nsg, nr0);
    }

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg,            FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) ne12, FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, r2,             FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, r3,             FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, nw,             FC_MUL_MV + 5);
        ggml_metal_cv_set_bool (cv, row4,           FC_MUL_MV + 6);
        ggml_metal_cv_set_bool (cv, row8,           FC_MUL_MV + 7);
        ggml_metal_cv_set_bool (cv, align,          FC_MUL_MV + 8);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    // a grid kernel stages its table once per threadgroup, which measured worth amortizing from
    // 2 KB up except on iq2_xxs; 640 threadgroups still fill the largest cards it runs on.
    // TOSH_MV_NTG overrides, 0 is no cap
    if (strncmp(suffix, "_v2", 3) == 0 && smem > 0) {
        static const int ntg_env = TOSH_ENV("TOSH_MV_NTG") ? atoi(TOSH_ENV("TOSH_MV_NTG")) : -1;
        res.ntg = ntg_env >= 0 ? ntg_env : (smem >= 2048 && tsrc0 != GGML_TYPE_IQ2_XXS ? 640 : 0);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax_part(ggml_metal_library_t lib) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_amax_part_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_amax(ggml_metal_library_t lib) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_amax_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id_map0(ggml_metal_library_t lib, int ne02, int ne20) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_mul_mm_id_map0_ne20_%d", ne20);
    snprintf(name, 256, "%s_ne02=%d", base, ne02);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ne02*ne20*sizeof(uint16_t);
    res.smem = GGML_PAD(res.smem, 16);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mm_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    char base[256];
    char name[256];

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const bool bc_inp = op->src[0]->ne[0] % 32 != 0;

    const bool use_mm_manual = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->use_mm_manual;

    // src1 prec [TAG_GGML_PREC]
    const bool amax = ggml_get_op_params_i32(op, 3) == GGML_PREC_F32;

    // a rescaled src1 still overflows the half partial sums, so those ops accumulate in float
    const bool acc_f32 = use_mm_manual && (!ggml_is_contiguous(op->src[0]) || tsrc0 == GGML_TYPE_BF16 || amax) &&
                         TOSH_ENV("TOSH_MM_ACC_F32_DISABLE") == nullptr;

    // one grid per expert already keeps the GPU in threadgroups, so the 8x4 tile does not need
    // the dense token floor. What it needs is an average expert that fills the narrow tile:
    // below that its 64 columns are mostly padding and it loses (measured -5% at 14 tokens
    // per expert, +2.3% at 45)
    static const int wide_floor = []() {
        const char * s = TOSH_ENV("TOSH_MMID_WIDE_FLOOR");
        return s ? atoi(s) : 32;
    }();

    // wave64 is off by the dense tile's result, not by one measured here: the expert path splits
    // its work per expert, so the opt-in exists for a GCN card to settle it
    const int simd_width = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const bool wide_width_ok = simd_width != 64 || TOSH_ENV("TOSH_MMID_WIDE_W64_ENABLE") != nullptr;

    const int64_t tpe = op->src[0]->ne[2] > 0 ? op->src[2]->ne[0]*op->src[2]->ne[1] / op->src[0]->ne[2] : 0;

    bool wide = use_mm_manual && tsrc1 == GGML_TYPE_F32 && wide_width_ok &&
                tpe >= wide_floor && TOSH_ENV("TOSH_MMID_WIDE_DISABLE") == nullptr;
    if (wide) {
        // it needs 64 accumulators, so only where the dequant leaves the registers
        switch (tsrc0) {
            case GGML_TYPE_F16:
            case GGML_TYPE_Q4_0:
            case GGML_TYPE_Q4_1:
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q8_0:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_Q2_K:
            case GGML_TYPE_Q3_K:
            case GGML_TYPE_Q4_K:
            case GGML_TYPE_Q5_K:
            case GGML_TYPE_Q6_K:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ3_XXS: break;
            default: wide = false; break;
        }
    }

    // the 64-lane twin lives in mul_mm_w64.metal, beside the dense one; it has no src1 rescale
    const int simd_width_id = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;
    const bool w64 = simd_width_id == 64 && use_mm_manual && !amax &&
                     TOSH_ENV("TOSH_MMID_W64_DISABLE") == nullptr;

    // the narrow mxfp4 kernel is bfloat when the device has the type, and the manual path keeps
    // its packed sums in that type; the wide and 64-lane variants are already half, and so is the
    // narrow one without bfloat
    const bool half_id = use_mm_manual && !wide && !w64 && tsrc0 == GGML_TYPE_MXFP4 && tsrc1 == GGML_TYPE_F32 &&
                         ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_bfloat &&
                         TOSH_ENV("TOSH_MMID_HALF_DISABLE") == nullptr;

    snprintf(base, 256, "kernel_mul_mm_id_%s_%s%s%s%s", ggml_metal_tname_pair(lib, tsrc0, tsrc1), ggml_type_name(tsrc1), wide ? "_w" : "", w64 ? "_w64" : "", half_id ? "_h" : "");
    snprintf(name, 256, "%s_bci=%d_mm=%d_af=%d_amax=%d", base, bc_inp, use_mm_manual, acc_f32, amax);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, bc_inp,        FC_MUL_MM + 0);
        ggml_metal_cv_set_bool(cv, use_mm_manual, FC_MUL_MM + 6);
        ggml_metal_cv_set_bool(cv, acc_f32,       FC_MUL_MM + 9);
        ggml_metal_cv_set_bool(cv, amax,          FC_MUL_MM + 11);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 8192;
    res.nr0  = 64;
    res.nr1  = wide ? 64 : 32;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_mul_mv_id(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_TENSOR_LOCALS( int32_t, ne0, op->src[0], ne);
    GGML_TENSOR_LOCALS( int32_t, ne1, op->src[1], ne);

    char base[256];
    char name[256];

    int nsg = 0; // number of simdgroups
    int nr0 = 0; // number of src0 rows per simdgroup
    int nr1 = 1; // number of src1 rows per threadgroup

    size_t smem = 0; // shared memory

    const ggml_type tsrc0 = op->src[0]->type;
    const ggml_type tsrc1 = op->src[1]->type;

    const int simd_width = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;

    const char * suffix = "";


        // use custom matrix x vector kernel
    switch (tsrc0) {
        case GGML_TYPE_F32:
        case GGML_TYPE_F16:
        case GGML_TYPE_BF16:
            {
                nsg = std::min(4, (ne00 + 127) / 128);
                nr0 = 2;
                nr1 = 1;
                smem = 32*sizeof(float)*nr0;
                suffix = ne00 % 4 == 0 ? "_4" : "";
            } break;
        case GGML_TYPE_Q1_0:
            {
                nsg = N_SG_Q1_0;
                nr0 = N_R0_Q1_0;
            } break;
        case GGML_TYPE_Q2_0:
            {
                nsg = N_SG_Q2_0;
                nr0 = N_R0_Q2_0;
            } break;
        case GGML_TYPE_PQ2_0:
            {
                nsg = N_SG_PQ2_0;
                nr0 = N_R0_PQ2_0;
            } break;
        case GGML_TYPE_PTQ1_0:
            {
                nsg = N_SG_PTQ1_0;
                nr0 = N_R0_PTQ1_0;
                smem = 256*2*sizeof(uint32_t); // the trit table
            } break;
        case GGML_TYPE_Q4_0:
            {
                nsg = N_SG_Q4_0;
                nr0 = N_R0_Q4_0;
            } break;
        case GGML_TYPE_Q4_1:
            {
                nsg = N_SG_Q4_1;
                nr0 = N_R0_Q4_1;
            } break;
        case GGML_TYPE_Q5_0:
            {
                nsg = N_SG_Q5_0;
                nr0 = N_R0_Q5_0;
            } break;
        case GGML_TYPE_Q5_1:
            {
                nsg = N_SG_Q5_1;
                nr0 = N_R0_Q5_1;
            } break;
        case GGML_TYPE_Q8_0:
            {
                nsg = N_SG_Q8_0;
                nr0 = N_R0_Q8_0;
                smem = 32*sizeof(float)*N_R0_Q8_0;
                // as in the dense path, short rows leave simdgroups with only zeros to sum
                if (simd_width == 32 && !ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->has_unified_memory &&
                    TOSH_ENV("TOSH_MV_Q8_0_NSG_DISABLE") == nullptr) {
                    nsg = std::max(1, std::min(nsg, (ne00/32 + 7)/8));
                }
            } break;
        case GGML_TYPE_MXFP4:
            {
                nsg = N_SG_MXFP4;
                nr0 = N_R0_MXFP4;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_Q2_K:
            {
                nsg = N_SG_Q2_K;
                nr0 = N_R0_Q2_K;
            } break;
        case GGML_TYPE_Q3_K:
            {
                nsg = N_SG_Q3_K;
                nr0 = N_R0_Q3_K;
            } break;
        case GGML_TYPE_Q4_K:
            {
                nsg = N_SG_Q4_K;
                nr0 = N_R0_Q4_K;
            } break;
        case GGML_TYPE_Q5_K:
            {
                // the expert mat-vec was short too: 8 rows measure +12.1% of tg on a 64-wide card
                if ((simd_width == 64 || TOSH_ENV("TOSH_MV_R8") != nullptr) && TOSH_ENV("TOSH_MV_R8_DISABLE") == nullptr) {
                    nsg = 1;
                    nr0 = 8;
                    suffix = "_r8";
                } else {
                    nsg = N_SG_Q5_K;
                    nr0 = N_R0_Q5_K;
                }
            } break;
        case GGML_TYPE_Q6_K:
            {
                nsg = N_SG_Q6_K;
                nr0 = N_R0_Q6_K;
            } break;
        case GGML_TYPE_IQ2_XXS:
            {
                nsg = N_SG_IQ2_XXS;
                nr0 = N_R0_IQ2_XXS;
                smem = 256*8+128;
            } break;
        case GGML_TYPE_IQ2_XS:
            {
                nsg = N_SG_IQ2_XS;
                nr0 = N_R0_IQ2_XS;
                smem = 512*8+128;
            } break;
        case GGML_TYPE_IQ3_XXS:
            {
                nsg = N_SG_IQ3_XXS;
                nr0 = N_R0_IQ3_XXS;
                smem = 256*4+128;
            } break;
        case GGML_TYPE_IQ3_S:
            {
                nsg = N_SG_IQ3_S;
                nr0 = N_R0_IQ3_S;
                smem = 512*4;
            } break;
        case GGML_TYPE_IQ2_S:
            {
                nsg = N_SG_IQ2_S;
                nr0 = N_R0_IQ2_S;
                smem = 1024*8;
            } break;
        case GGML_TYPE_IQ1_S:
            {
                nsg = N_SG_IQ1_S;
                nr0 = N_R0_IQ1_S;
                smem = 2048*4;
            } break;
        case GGML_TYPE_IQ1_M:
            {
                nsg = N_SG_IQ1_M;
                nr0 = N_R0_IQ1_M;
                smem = 2048*4;
            } break;
        case GGML_TYPE_IQ4_NL:
            {
                nsg = N_SG_IQ4_NL;
                nr0 = N_R0_IQ4_NL;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_IQ4_XS:
            {
                nsg = N_SG_IQ4_XS;
                nr0 = N_R0_IQ4_XS;
                smem = 32*sizeof(float);
            } break;
        case GGML_TYPE_TQ2_0:
            {
                nsg = N_SG_TQ2_0;
                nr0 = N_R0_TQ2_0;
            } break;
        default:
            {
                GGML_LOG_ERROR("Asserting on type %d\n", (int)op->src[2]->type);
                GGML_ABORT("not implemented");
            }
    };

    // the 64-lane expert matvec lives in mul_mv_w64.metal and carries the same lane
    // decomposition as the standalone one, so the expert path stops falling back to 32.
    // Types whose shared memory holds a lookup grid keep the rows and the buffer the
    // switch above computed: only the reduction-buffer ones move.
    bool id_w64 = simd_width == 64 && TOSH_ENV("TOSH_MVID_W64_DISABLE") == nullptr;
    if (id_w64) {
        switch (tsrc0) {
            case GGML_TYPE_Q4_0: nr0 = 4; break;
            case GGML_TYPE_Q4_1: nr0 = 4; break;
            case GGML_TYPE_Q8_0: nr0 = 4; break;
            case GGML_TYPE_Q2_K: nr0 = 8; break;
            // q3_K keeps the 32-lane eight-row kernel: its 64-lane twin is opt-in in the
            // dense path too, and forcing it here measured 4.3% below on a 60-expert model
            case GGML_TYPE_Q4_K: nr0 = 4; break;
            case GGML_TYPE_Q5_K: nr0 = 2; break;
            case GGML_TYPE_Q6_K: nr0 = 4; break;
            case GGML_TYPE_Q5_0:
            case GGML_TYPE_Q5_1:
            case GGML_TYPE_Q1_0:
            case GGML_TYPE_Q2_0:
            case GGML_TYPE_MXFP4:
            case GGML_TYPE_IQ1_S:
            case GGML_TYPE_IQ1_M:
            case GGML_TYPE_IQ2_S:
            case GGML_TYPE_IQ2_XS:
            case GGML_TYPE_IQ2_XXS:
            case GGML_TYPE_IQ3_S:
            case GGML_TYPE_IQ3_XXS:
            case GGML_TYPE_IQ4_NL:
            case GGML_TYPE_IQ4_XS: break;
            // the rest have no 64-lane twin yet
            default: id_w64 = false; break;
        }
    }
    if (id_w64) {
        // one wave per threadgroup, and the twins carry no suffix of their own. The rows go
        // in before the block below, so the shared buffer is sized for the rows actually used.
        nsg = 1;
        suffix = "";
    }

    // wave64 decode: MoE experts reuse the same width-parameterized mat-vec impls,
    // so run them on a 64-wide simdgroup (nw) like the dense mul_mv path.
    auto * props_id = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));
    int16_t nw = 32;
    if (props_id->wave64_decode) {
        const bool helper = (tsrc0 == GGML_TYPE_F16 || tsrc0 == GGML_TYPE_F32 ||
                             tsrc0 == GGML_TYPE_BF16 || tsrc0 == GGML_TYPE_Q8_0);
        const bool sr     = (tsrc0 == GGML_TYPE_Q4_K || tsrc0 == GGML_TYPE_Q5_K || tsrc0 == GGML_TYPE_Q6_K ||
                             tsrc0 == GGML_TYPE_Q2_K || tsrc0 == GGML_TYPE_Q3_K ||
                             tsrc0 == GGML_TYPE_Q4_0 || tsrc0 == GGML_TYPE_Q4_1 ||
                             tsrc0 == GGML_TYPE_Q5_0 || tsrc0 == GGML_TYPE_Q5_1 ||
                             tsrc0 == GGML_TYPE_Q1_0 || tsrc0 == GGML_TYPE_Q2_0 ||
                             tsrc0 == GGML_TYPE_PQ2_0 || tsrc0 == GGML_TYPE_PTQ1_0 ||
                             tsrc0 == GGML_TYPE_IQ2_XXS || tsrc0 == GGML_TYPE_IQ2_XS ||
                             tsrc0 == GGML_TYPE_IQ3_XXS || tsrc0 == GGML_TYPE_IQ3_S ||
                             tsrc0 == GGML_TYPE_IQ2_S || tsrc0 == GGML_TYPE_IQ1_S ||
                             tsrc0 == GGML_TYPE_IQ1_M);
        const bool lut = (tsrc0 == GGML_TYPE_MXFP4 || tsrc0 == GGML_TYPE_IQ4_NL || tsrc0 == GGML_TYPE_IQ4_XS);
        if (helper) { nw = (int16_t) props_id->simd_width; smem = (size_t) nw*sizeof(float)*nr0; }
        else if (sr) { nw = (int16_t) props_id->simd_width; }
        else if (lut) { nw = (int16_t) props_id->simd_width; smem = (size_t) nw*sizeof(float); }
    }

    // the 32-lane expert mat-vec takes the same kernels as the dense one
    if (!id_w64 && nw == 32 && suffix[0] == '\0') {
        static const bool v2 = TOSH_ENV("TOSH_MV_V2_DISABLE") == nullptr;
        if (v2 && (tsrc0 == GGML_TYPE_IQ3_XXS || tsrc0 == GGML_TYPE_IQ3_S  || tsrc0 == GGML_TYPE_IQ2_S  ||
                   tsrc0 == GGML_TYPE_IQ2_XXS || tsrc0 == GGML_TYPE_IQ2_XS || tsrc0 == GGML_TYPE_IQ1_S  ||
                   tsrc0 == GGML_TYPE_IQ1_M   || tsrc0 == GGML_TYPE_Q3_K   || tsrc0 == GGML_TYPE_Q2_K   ||
                   tsrc0 == GGML_TYPE_Q5_K)) {
            suffix = "_v2";
        }
        if (tsrc0 == GGML_TYPE_Q5_1 && !props_id->has_unified_memory && TOSH_ENV("TOSH_MV_Q5_1_L2_DISABLE") == nullptr) {
            nr0    = 2;
            suffix = "_L2r2";
        }
        // rows of two super-blocks leave half the lane groups idle, the _s2 twin gives them a second set of rows
        if (tsrc0 == GGML_TYPE_Q4_K && ne00 == 2*256 && ne01 % (2*N_R0_Q4_K*N_SG_Q4_K) == 0 && !props_id->has_unified_memory &&
            TOSH_ENV("TOSH_MVID_SPLIT_DISABLE") == nullptr) {
            nr0    = 2*N_R0_Q4_K;
            suffix = "_s2";
        }
    }

    const bool row4 = (op->src[0]->nb[1] % 2 == 0) && (op->src[0]->nb[2] % 2 == 0) &&
                      TOSH_ENV("TOSH_MV_ROW4_DISABLE") == nullptr;

    // The expert kernels share the implementations of the dense path, so they can read the
    // quant arrays a word at a time under the same conditions.
    const bool row8 = row4 && (op->src[0]->nb[1] % 4 == 0) && (op->src[0]->nb[2] % 4 == 0) &&
                      (nw == 64 || (nw == 32 && tsrc0 == GGML_TYPE_Q8_0)) &&
                      TOSH_ENV("TOSH_MV_ROW8_DISABLE") == nullptr;

    const bool align = props_id->needs_aligned_loads;

    // only the ids a mixed expert layer publishes can be -1
    const bool id_skip = op->src[2] != nullptr && op->src[2]->op == GGML_OP_TOSH_DMOE_PUBLISH;

    snprintf(base, 256, "kernel_mul_mv_id_%s_%s%s%s", ggml_metal_tname_pair(lib, tsrc0, tsrc1), ggml_type_name(tsrc1), suffix, id_w64 ? "_w64" : "");
    snprintf(name, 256, "%s_nsg=%d_nw=%d_a4=%d_a8=%d_al=%d_sk=%d", base, nsg, nw, row4, row8, align, id_skip);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_MUL_MV + 0);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 2);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 3);
        ggml_metal_cv_set_int16(cv, 1,   FC_MUL_MV + 4);
        ggml_metal_cv_set_int16(cv, nw,  FC_MUL_MV + 5);
        ggml_metal_cv_set_bool (cv, row4, FC_MUL_MV + 6);
        ggml_metal_cv_set_bool (cv, row8, FC_MUL_MV + 7);
        ggml_metal_cv_set_bool (cv, align, FC_MUL_MV + 8);
        ggml_metal_cv_set_bool (cv, id_skip, FC_MUL_MV + 9);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.nr0  = nr0;
    res.nr1  = nr1;
    res.nsg  = nsg;
    res.smem = smem;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argmax(ggml_metal_library_t lib, const ggml_tensor * op) {
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous_1(op->src[0]));
    GGML_ASSERT(op->src[0]->nb[0] == ggml_type_size(op->src[0]->type));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_argmax_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width*(sizeof(float) + sizeof(int32_t));

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_tosh_topk_moe(ggml_metal_library_t lib) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_tosh_topk_moe_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_argsort_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARGSORT);

    char base[256];
    char name[256];

    ggml_sort_order order = (ggml_sort_order) op->op_params[0];

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht(ggml_metal_library_t lib, int n, ggml_type tsrc, bool w64) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_fwht_%s_%d%s", ggml_type_name(tsrc), n, w64 ? "_w64" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_fwht_fused(ggml_metal_library_t lib, int n, int hd, int rep, int nk, bool w64) {
    char base[256];
    char name[256];

    const bool perm = hd > 0;

    snprintf(base, 256, "kernel_fwht_fused_f32_%d%s%s", n, perm ? "_p" : "", w64 ? "_w64" : "");
    snprintf(name, 256, "%s_hd=%d_rep=%d_nk=%d", base, hd, rep, nk);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, (int16_t) hd,  FC_FWHT + 0);
        ggml_metal_cv_set_int16(cv, (int16_t) rep, FC_FWHT + 1);
        ggml_metal_cv_set_int16(cv, (int16_t) nk,  FC_FWHT + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

// note: reuse the argsort kernel for the bitonic top_k fallback
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    // note: the top_k kernel is always descending order
    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_radix(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_top_k_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_top_k_merge(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TOP_K);

    char base[256];
    char name[256];

    ggml_sort_order order = GGML_SORT_ORDER_DESC;

    const char * order_str = "undefined";
    switch (order) {
        case GGML_SORT_ORDER_ASC:  order_str = "asc";  break;
        case GGML_SORT_ORDER_DESC: order_str = "desc"; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_argsort_merge_%s_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->type), order_str);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_topk_moe(
        ggml_metal_library_t lib, int32_t n_expert, int32_t top_k, bool with_norm) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_topk_moe_f32");
    snprintf(name, 256, "%s_n_expert=%d_top_k=%d_with_norm=%d", base, n_expert, top_k, with_norm ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool (cv, with_norm, FC_TOPK_MOE + 0);
        ggml_metal_cv_set_int32(cv, n_expert,  FC_TOPK_MOE + 1);
        ggml_metal_cv_set_int32(cv, top_k,     FC_TOPK_MOE + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_moe_reduce(ggml_metal_library_t lib, int32_t n_expert_used) {
    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_moe_reduce_f32");
    snprintf(name, 256, "%s_n_expert_used=%d", base, n_expert_used);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int32(cv, n_expert_used, FC_MOE_REDUCE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_pad(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        bool    has_mask,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_pad");

    snprintf(name, 256, "%s_mask=%d_ncpsg=%d",
            base,
            has_mask,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_PAD + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_PAD + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_PAD + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_PAD + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_PAD + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_PAD + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_PAD + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_PAD + 23);
        //ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_PAD + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_PAD + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_kv_f16(
        ggml_metal_library_t lib,
        const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];

    snprintf(base, 256, "kernel_flash_attn_ext_kv_%s_f16", ggml_type_name(op->src[1]->type));

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, base);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, base, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_blk(
        ggml_metal_library_t lib,
        const struct ggml_tensor * op,
        int32_t nqptg,
        int32_t ncpsg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_%s",
            "flash_attn_ext_blk");

    snprintf(name, 256, "%s_nqptg=%d_ncpsg=%d",
            base,
            nqptg,
            ncpsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        //ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_BLK + 0);
        //ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_BLK + 1);
        //ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_BLK + 2);
        //ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_BLK + 3);

        //ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_BLK + 20);
        //ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_BLK + 21);
        //ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_BLK + 22);
        //ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_BLK + 23);
        ggml_metal_cv_set_int32(cv, nqptg, FC_FLASH_ATTN_EXT_BLK + 24);
        ggml_metal_cv_set_int32(cv, ncpsg, FC_FLASH_ATTN_EXT_BLK + 25);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_tensor(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool has_mask,
        bool has_sinks,
        bool has_bias,
        bool has_scap) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    snprintf(base, 256, "kernel_flash_attn_ext_tensor_f16_dk%d_dv%d", dk, dv);
    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d", base, has_mask, has_sinks, has_bias, has_scap);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_TENSOR + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_TENSOR + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_TENSOR + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_TENSOR + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        int32_t nsg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const char * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    // do bounds checks for the mask?
    const bool bc_mask = op->src[3] && (op->src[3]->ne[1] % 8 != 0);

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d",
            "flash_attn_ext",
            type,
            dk,
            dv);

    snprintf(name, 256, "%s_mask=%d_sinks=%d_bias=%d_scap=%d_kvpad=%d_bcm=%d_ns10=%d_ns20=%d_nsg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            bc_mask,
            ns10,
            ns20,
            nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad, FC_FLASH_ATTN_EXT + 4);

        ggml_metal_cv_set_bool(cv, bc_mask, FC_FLASH_ATTN_EXT + 10);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT + 22);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_idx(
        ggml_metal_library_t lib,
        const ggml_tensor * op) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);
    assert(op->src[3]);

    char name[256];

    const int nw = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;

    snprintf(name, 256, "kernel_flash_attn_ext_vec_idx%s", nw == 64 ? "_w64" : "");

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    GGML_UNUSED(op);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        bool    has_mask,
        bool    has_sinks,
        bool    has_bias,
        bool    has_scap,
        bool    has_kvpad,
        bool    has_sparse,
        int32_t nqpsg,
        int32_t ne,
        int32_t nsg,
        int32_t nwg,
        bool    use_kv_f16,
        int32_t ns10,
        int32_t ns20) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    const int32_t dk = (int32_t) op->src[1]->ne[0];
    const int32_t dv = (int32_t) op->src[2]->ne[0];

    const char * type = use_kv_f16 ? "f16" : ggml_type_name(op->src[1]->type);

    char qne_suffix[16] = {0};
    if (!(nqpsg == 1 && ne == ggml_metal_tuning::fa_vec_baseline_ne(dk, dv))) {
        snprintf(qne_suffix, sizeof(qne_suffix), "_q%d_ne%d", nqpsg, ne);
    }

    snprintf(base, 256, "kernel_%s_%s_dk%d_dv%d%s",
            "flash_attn_ext_vec",
            type,
            dk,
            dv,
            qne_suffix);

    snprintf(name, 256, "%s_mask=%d_sink=%d_bias=%d_scap=%d_kvpad=%d_sparse=%d_ns10=%d_ns20=%d_nsg=%d_nwg=%d",
            base,
            has_mask,
            has_sinks,
            has_bias,
            has_scap,
            has_kvpad,
            has_sparse,
            ns10,
            ns20,
            nsg, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, has_mask,  FC_FLASH_ATTN_EXT_VEC + 0);
        ggml_metal_cv_set_bool(cv, has_sinks, FC_FLASH_ATTN_EXT_VEC + 1);
        ggml_metal_cv_set_bool(cv, has_bias,  FC_FLASH_ATTN_EXT_VEC + 2);
        ggml_metal_cv_set_bool(cv, has_scap,  FC_FLASH_ATTN_EXT_VEC + 3);
        ggml_metal_cv_set_bool(cv, has_kvpad,  FC_FLASH_ATTN_EXT_VEC + 4);
        ggml_metal_cv_set_bool(cv, has_sparse, FC_FLASH_ATTN_EXT_VEC + 5);

        ggml_metal_cv_set_int32(cv, ns10, FC_FLASH_ATTN_EXT_VEC + 20);
        ggml_metal_cv_set_int32(cv, ns20, FC_FLASH_ATTN_EXT_VEC + 21);
        ggml_metal_cv_set_int32(cv, nsg,  FC_FLASH_ATTN_EXT_VEC + 22);
        ggml_metal_cv_set_int32(cv, nwg,  FC_FLASH_ATTN_EXT_VEC + 23);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_flash_attn_ext_vec_reduce(
        ggml_metal_library_t lib,
        const ggml_tensor * op,
        int32_t dv,
        int32_t nwg) {
    assert(op->op == GGML_OP_FLASH_ATTN_EXT);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_flash_attn_ext_vec_reduce");
    snprintf(name, 256, "%s_dv=%d_nwg=%d", base, dv, nwg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int32(cv, dv,  FC_FLASH_ATTN_EXT_VEC_REDUCE + 0);
        ggml_metal_cv_set_int32(cv, nwg, FC_FLASH_ATTN_EXT_VEC_REDUCE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;

    GGML_UNUSED(op);
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin(ggml_metal_library_t lib, const ggml_tensor * op, int32_t n_fuse) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op->op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t1_str = ggml_type_name(op->src[1]->type);
    const char * t_str  = ggml_type_name(op->type);

    const bool is_c4 = (op->src[0]->ne[0] % 4 == 0) && (op->src[1]->ne[0] % 4 == 0);

    const bool is_cb = op->src[0]->ne[0] != op->src[1]->ne[0];
    const bool is_rb = ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]) && (ggml_nrows(op->src[1]) == 1) && ggml_nelements(op) < 65536;

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s%s", t0_str, t1_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s_op=%d_nf=%d_rb=%d_cb=%d", base, op_num, n_fuse, is_rb, is_cb);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, n_fuse, FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, is_rb,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, is_cb,  FC_BIN + 3);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.c4  = is_c4;
    res.cnt = is_rb;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one(ggml_metal_library_t lib, ggml_op op) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", "f32", "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

// The f16 allreduce transport reduces f32 += f16 with an upcast. Same shape as the f32-only
// one-shot above, but src1 goes through the same name-pair helper the cpy pipelines use, so
// the instantiation is resolved by type the same way everywhere else.
ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_bin_one_src1(ggml_metal_library_t lib, ggml_op op, ggml_type tsrc1) {
    char base[256];
    char name[256];

    int op_num = -1;

    switch (op) {
        case GGML_OP_ADD: op_num = 0; break;
        case GGML_OP_SUB: op_num = 1; break;
        case GGML_OP_MUL: op_num = 2; break;
        case GGML_OP_DIV: op_num = 3; break;
        default: GGML_ABORT("fatal error");
    };

    const char * t1_str = ggml_metal_tname_pair(lib, tsrc1, GGML_TYPE_F32);

    snprintf(base, 256, "kernel_bin_fuse_%s_%s_%s", "f32", t1_str, "f32");
    snprintf(name, 256, "%s_op=%d_nf=%d", base, op_num, 1);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, op_num, FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,      FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, false,  FC_BIN + 2);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_l2_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_L2_NORM);

    char base[256];
    char name[256];

    const bool is_c4 = op->src[0]->ne[0] % 4 == 0;

    const char * t0_str = ggml_type_name(op->src[0]->type);
    const char * t_str  = ggml_type_name(op->type);

    snprintf(base, 256, "kernel_l2_norm_%s_%s%s", t0_str, t_str, is_c4 ? "_4" : "");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.c4   = is_c4;
    // wave64 needs one reduction slot per lane of the real simd width, not 32.
    res.smem = (size_t) ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_group_norm(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_GROUP_NORM);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_group_norm_f32");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = (size_t) ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm(ggml_metal_library_t lib, const ggml_tensor * op, int n_fuse) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        case GGML_OP_RMS_NORM:
            switch (n_fuse) {
                case 1: snprintf(base, 256, "kernel_rms_norm_f32%s", suffix);         break;
                case 2: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix);     break;
                case 3: snprintf(base, 256, "kernel_rms_norm_mul_add_f32%s", suffix); break;
                default: GGML_ABORT("fatal error");
            } break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, false, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_norm_scale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_NORM || op->op == GGML_OP_RMS_NORM);

    GGML_ASSERT(ggml_is_contiguous_rows(op->src[0]));

    char base[256];
    char name[256];

    const char * suffix = "";
    if (op->ne[0] % 4 == 0) {
        suffix = "_4";
    }

    switch (op->op) {
        case GGML_OP_NORM:     snprintf(base, 256, "kernel_norm_mul_f32%s", suffix);     break;
        case GGML_OP_RMS_NORM: snprintf(base, 256, "kernel_rms_norm_mul_f32%s", suffix); break;
        default: GGML_ABORT("fatal error");
    }

    snprintf(name, 256, "%s_use_scale", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_bool(cv, true, FC_NORM + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    // Reduction buffer sized by the real SIMD width so shmem[tiisg] fits on wave64.
    auto * props = ggml_metal_device_get_props(ggml_metal_library_get_device(lib));
    res.smem = (size_t) props->simd_width*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rope(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROPE || op->op == GGML_OP_ROPE_BACK);

    const bool is_back = op->op == GGML_OP_ROPE_BACK;

    char base[256];
    char name[256];

    const int mode = ((const int32_t *) op->op_params)[2];

    const bool is_neox   = mode & GGML_ROPE_TYPE_NEOX;
    const bool is_mrope  = mode & GGML_ROPE_TYPE_MROPE;
    const bool is_imrope = mode == GGML_ROPE_TYPE_IMROPE;
    const bool is_vision = mode == GGML_ROPE_TYPE_VISION;

    if (is_neox) {
        snprintf(base, 256, "kernel_rope_neox_%s", ggml_type_name(op->src[0]->type));
    } else if ((is_mrope || is_imrope) && !is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_multi_%s", ggml_type_name(op->src[0]->type));
    } else if (is_vision) {
        GGML_ASSERT(op->src[1]->ne[0]*4 >= op->src[0]->ne[2]); // need at least 4 pos per token
        snprintf(base, 256, "kernel_rope_vision_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_rope_norm_%s", ggml_type_name(op->src[0]->type));
    }

    snprintf(name, 256, "%s_imrope=%d_is_back=%d", base, is_imrope ? 1 : 0, is_back ? 1 : 0);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, is_imrope, FC_ROPE + 0);
        ggml_metal_cv_set_bool(cv, is_back,   FC_ROPE + 1);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_im2col(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_IM2COL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F16 || op->type == GGML_TYPE_F32);

    const bool is_2D = ((const int32_t *)(op->op_params))[6] == 1;
    const int64_t KH = is_2D ? ne01 : 1;
    const int64_t KW = ne00;

    char base[256];
    char name[256];

    if (KH*KW <= 1024) {
        snprintf(base, 256, "kernel_im2col_%s", ggml_type_name(op->type));
    } else {
        snprintf(base, 256, "kernel_im2col_ext_%s", ggml_type_name(op->type));
    }
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_1d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_col2im_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_COL2IM_1D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_BF16);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_col2im_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_snake(ggml_metal_library_t lib, enum ggml_type type) {
    GGML_ASSERT(type == GGML_TYPE_F32 || type == GGML_TYPE_F16 || type == GGML_TYPE_BF16);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_snake_%s", ggml_type_name(type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_transpose_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_TRANSPOSE_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(ggml_is_contiguous(op->src[1]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_transpose_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_2D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_2d_dw(ggml_metal_library_t lib, const ggml_tensor * op, bool tiled) {
    assert(op->op == GGML_OP_CONV_2D_DW);

    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_2d_dw%s_%s_%s",
             tiled ? "_tiled" : "",
             ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_conv_3d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_CONV_3D);

    GGML_ASSERT(ggml_is_contiguous(op->src[0]));
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_F16 || op->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->src[1]->type == GGML_TYPE_F32);
    GGML_ASSERT(op->type         == GGML_TYPE_F32);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_conv_3d_%s_%s", ggml_type_name(op->src[0]->type), ggml_type_name(op->src[1]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_upscale(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_UPSCALE);

    char base[256];
    char name[256];

    const int32_t mode_flags = ggml_get_op_params_i32(op, 0);
    const ggml_scale_mode mode = (ggml_scale_mode) (mode_flags & 0xFF);

    const bool antialias = (mode_flags & GGML_SCALE_FLAG_ANTIALIAS);

    if (mode == GGML_SCALE_MODE_BILINEAR) {
        snprintf(base, 256, "kernel_upscale_bilinear_%s", ggml_type_name(op->src[0]->type));
    } else if (mode == GGML_SCALE_MODE_BICUBIC) {
        snprintf(base, 256, "kernel_upscale_bicubic_%s", ggml_type_name(op->src[0]->type));
    } else {
        snprintf(base, 256, "kernel_upscale_nearest_%s", ggml_type_name(op->src[0]->type));
    }
    snprintf(name, 256, "%s_aa=%d", base, antialias);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, antialias, FC_UPSCALE + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_roll(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ROLL);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_roll_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD);

    char base[256];
    char name[256];

    const bool circular = ggml_get_op_params_i32(op, 8) != 0;

    snprintf(base, 256, "kernel_pad_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_circular=%d", base, circular);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_bool(cv, circular, FC_PAD + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_pad_reflect_1d(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_PAD_REFLECT_1D);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_pad_reflect_1d_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_arange(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_ARANGE);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_arange_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_timestep_embedding(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_TIMESTEP_EMBEDDING);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_timestep_embedding_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_adamw(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_ADAMW);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_adamw_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_opt_step_sgd(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_OPT_STEP_SGD);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_opt_step_sgd_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_silu_back(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SILU_BACK);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_silu_back_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_memset(ggml_metal_library_t lib, const ggml_tensor *  op) {
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_memset_%s", ggml_type_name(op->type));
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_count_equal(ggml_metal_library_t lib, const ggml_tensor *  op) {
    assert(op->op == GGML_OP_COUNT_EQUAL);

    GGML_TENSOR_LOCALS(int64_t, ne0, op->src[0], ne);

    GGML_ASSERT(op->src[0]->type == op->src[1]->type);
    GGML_ASSERT(op->src[0]->type == GGML_TYPE_I32);
    GGML_ASSERT(op->type == GGML_TYPE_I64);

    // note: the kernel only supports i32 output due to metal atomic add only supporting atomic_int
    GGML_ASSERT(ggml_nelements(op->src[0]) < (1LL << 31));

    char base[256];
    char name[256];

    const int nw = ggml_metal_device_get_props(ggml_metal_library_get_device(lib))->simd_width;

    // the final reduction is a single simdgroup reading one entry per subgroup
    const int nsg_max = std::min(nw, 1024/nw);

    int nsg = 1;
    while (nw*nsg < ne00 && nsg < nsg_max) {
        nsg *= 2;
    }

    snprintf(base, 256, "kernel_count_equal_%s", ggml_type_name(op->src[0]->type));
    snprintf(name, 256, "%s_nsg=%d", base, nsg);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();

        ggml_metal_cv_set_int16(cv, nsg, FC_COUNT_EQUAL + 0);

        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);

        ggml_metal_cv_free(cv);
    }

    res.smem = (size_t) nsg * sizeof(int32_t);
    res.nsg  = nsg;

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_soft_max_back(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_SOFT_MAX_BACK);

    char base[256];
    char name[256];

    const char * suffix = (op->src[0]->ne[0] % 4 == 0) ? "_4" : "";

    snprintf(base, 256, "kernel_soft_max_back%s", suffix);
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_rms_norm_back(ggml_metal_library_t lib, const ggml_tensor * op) {
    assert(op->op == GGML_OP_RMS_NORM_BACK);
    GGML_UNUSED(op);

    char base[256];
    char name[256];

    snprintf(base, 256, "kernel_rms_norm_back");
    snprintf(name, 256, "%s", base);

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, base, name, nullptr);
    }

    res.smem = 2*32*sizeof(float);

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_repeat_back(ggml_metal_library_t lib) {
    const char * name = "kernel_repeat_back_f32";

    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}

ggml_metal_pipeline_with_params ggml_metal_library_get_pipeline_by_name(ggml_metal_library_t lib, const char * name) {
    ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        res = ggml_metal_library_compile_pipeline(lib, name, name, nullptr);
    }

    return res;
}
