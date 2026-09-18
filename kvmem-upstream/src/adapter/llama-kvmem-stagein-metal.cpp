// KVMem GPU stage-in for Metal (LLAMA_KVMEM_METAL).
//
// Same API as llama-kvmem-stagein.cu, without CUDA. Discrete Metal has no
// CPU-visible device memory, so every entry point that takes a raw device
// pointer degrades to its documented fallback:
//
// - enqueue_v / h2d_bytes with null dst: return false, the adapter falls
//   back to ggml_backend_tensor_set (synchronous, correct).
// - enqueue_k stages packed K on host; flush runs the K chain on CPU
//   (dequant -> RoPE(orig pos) -> FWHT -> quant) and copies out when dst
//   is host-accessible, else returns false and the caller falls back.
// - The granular dequant/rope/fwht/quantize entry points have no adapter
//   callers (the K chain lives in flush); they return false.
// - stageout / d2d_batched / GPU mean-K: not supported on Metal; return
//   false and the adapter uses its tensor_get / host-accumulate fallbacks.
// - d2h harvest pipe: never armed (d2h_init fails); harvest runs through
//   the synchronous harvest_capture fallback.

#include "llama-kvmem-stagein.h"

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-backend-impl.h"
#include "ggml-common.h"
#include "ggml-quants.h"
#include "ggml-metal/ggml-metal-device.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <vector>

namespace {

struct StageItem {
    bool is_k = false;
    ggml_type ty = GGML_TYPE_Q8_0;
    int64_t nt = 0;
    int64_t n_embd = 0;
    // K chain params
    int nrot = 0;
    int n_head = 0;
    int n_embd_head = 0;
    int n_rot_rope = 0;
    int32_t pos0 = 0;
    std::vector<float> theta;
    // staged bytes
    size_t off = 0;
    size_t nbytes = 0;
    uint8_t * dst = nullptr;
};

struct StageState {
    std::vector<uint8_t> slab;   // host staging, H2D source
    size_t used = 0;
    std::vector<StageItem> items;
    // transform scratch (F32 rows)
    std::vector<float> f32;
};

StageState & stage() {
    static StageState st;
    return st;
}

bool is_pow2(int n) {
    return n >= 64 && (n & (n - 1)) == 0;
}

// In-place normalized Walsh-Hadamard transform over nrot floats.
void fwht_rows(float * x, int64_t n_rows, int64_t stride, int nrot) {
    const float scale = 1.0f / sqrtf((float) nrot);
    std::vector<float> row((size_t) nrot);
    for (int64_t r = 0; r < n_rows; ++r) {
        float * xr = x + r * stride;
        for (int i = 0; i < nrot; ++i) {
            row[i] = xr[i] * scale;
        }
        for (int len = 1; len < nrot; len *= 2) {
            for (int i = 0; i < nrot; i += 2 * len) {
                for (int j = 0; j < len; ++j) {
                    const float u = row[i + j];
                    const float v = row[i + j + len];
                    row[i + j] = u + v;
                    row[i + j + len] = u - v;
                }
            }
        }
        for (int i = 0; i < nrot; ++i) {
            xr[i] = row[i];
        }
    }
}

// GPT-NeoX RoPE over the first n_rot dims of each head.
void rope_neox_rows(float * x, int64_t n_tokens, int n_head, int n_embd_head,
                    int n_rot, int32_t pos0, const float * theta) {
    for (int64_t t = 0; t < n_tokens; ++t) {
        const float pos = (float) (pos0 + t);
        for (int h = 0; h < n_head; ++h) {
            float * head = x + (t * n_head + h) * n_embd_head;
            for (int i = 0; i < n_rot / 2; ++i) {
                const float th = pos * theta[i];
                const float c = cosf(th);
                const float s = sinf(th);
                const float x0 = head[i];
                const float x1 = head[i + n_rot / 2];
                head[i] = x0 * c - x1 * s;
                head[i + n_rot / 2] = x0 * s + x1 * c;
            }
        }
    }
}

bool dequant_block(ggml_type ty, const uint8_t * src, float * dst, int64_t n_rows, int64_t n_embd) {
    const size_t row_size = ggml_row_size(ty, n_embd);
    for (int64_t r = 0; r < n_rows; ++r) {
        float * y = dst + r * n_embd;
        const void * x = src + r * row_size;
        if (ty == GGML_TYPE_Q8_0) {
            dequantize_row_q8_0((const block_q8_0 *) x, y, n_embd);
        } else if (ty == GGML_TYPE_Q4_0) {
            dequantize_row_q4_0((const block_q4_0 *) x, y, n_embd);
        } else if (ty == GGML_TYPE_F32) {
            std::memcpy(y, x, (size_t) n_embd * sizeof(float));
        } else {
            return false;
        }
    }
    return true;
}

bool quant_block(ggml_type ty, const float * src, uint8_t * dst, int64_t n_rows, int64_t n_embd) {
    const size_t row_size = ggml_row_size(ty, n_embd);
    for (int64_t r = 0; r < n_rows; ++r) {
        const float * x = src + r * n_embd;
        void * y = dst + r * row_size;
        if (ty == GGML_TYPE_Q8_0) {
            quantize_row_q8_0_ref(x, (block_q8_0 *) y, n_embd);
        } else if (ty == GGML_TYPE_Q4_0) {
            quantize_row_q4_0_ref(x, (block_q4_0 *) y, n_embd);
        } else if (ty == GGML_TYPE_F32) {
            std::memcpy(y, x, (size_t) n_embd * sizeof(float));
        } else {
            return false;
        }
    }
    return true;
}

bool slab_reserve(size_t nbytes) {
    StageState & st = stage();
    if (st.used + nbytes > st.slab.size()) {
        return false;
    }
    return true;
}

} // namespace

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed) {
    StageState & st = stage();
    const size_t need = n_f32 * sizeof(float) + n_packed;
    st.slab.assign(need > 0 ? need : 1, 0);
    st.used = 0;
    st.items.clear();
    st.f32.clear();
    return true;
}

void kvmem_stagein_gpu_free() {
    StageState & st = stage();
    st.slab.clear();
    st.slab.shrink_to_fit();
    st.used = 0;
    st.items.clear();
    st.f32.clear();
}

bool kvmem_stagein_fwht_ok(int nrot) {
    return nrot > 0 && nrot <= 8192 && is_pow2(nrot);
}

bool kvmem_stagein_quant_ok(ggml_type ty) {
    return ty == GGML_TYPE_Q8_0 || ty == GGML_TYPE_Q4_0 || ty == GGML_TYPE_F32;
}

bool kvmem_stagein_h2d_f32(const float * host, int64_t n) {
    if (!host || n <= 0) {
        return false;
    }
    const size_t nbytes = (size_t) n * sizeof(float);
    if (!slab_reserve(nbytes)) {
        return false;
    }
    StageState & st = stage();
    std::memcpy(st.slab.data() + st.used, host, nbytes);
    st.used += nbytes;
    return true;
}

bool kvmem_stagein_h2d_packed(const void * host, size_t n) {
    if (!host || n == 0) {
        return false;
    }
    if (!slab_reserve(n)) {
        return false;
    }
    StageState & st = stage();
    std::memcpy(st.slab.data() + st.used, host, n);
    st.used += n;
    return true;
}

bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd) {
    // Operates on the tail of the slab in place is not possible across
    // widths; callers stage packed bytes then read back F32 via flush.
    // Kept for API completeness; the K chain below is the real user.
    (void) ty;
    (void) n_rows;
    (void) n_embd;
    return false;
}

bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta) {
    (void) n_tokens;
    (void) n_head;
    (void) n_embd_head;
    (void) n_rot;
    (void) pos0;
    (void) theta;
    (void) n_theta;
    return false;
}

bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot) {
    (void) n_rows;
    (void) n_embd;
    (void) nrot;
    return false;
}

bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd) {
    (void) ty;
    (void) gpu_dst;
    (void) n_rows;
    (void) n_embd;
    return false;
}

bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n) {
    if (!gpu_dst || !host || n == 0) {
        return false;
    }
    std::memcpy(gpu_dst, host, n);
    return true;
}

void kvmem_stagein_sync() {
    // All Metal stage-in work is synchronous.
}

bool kvmem_stagein_enqueue_k(
        ggml_type ty, const void * packed, size_t nbytes, uint8_t * dst,
        int64_t nt, int64_t n_embd, int nrot,
        int n_head, int n_embd_head, int n_rot_rope, int32_t pos0,
        const float * theta, int n_theta,
        int64_t * copy_us, int64_t * rope_us, int64_t * hadamard_us, int64_t * set_us) {
    (void) copy_us;
    (void) rope_us;
    (void) hadamard_us;
    (void) set_us;
    if (!packed || nbytes == 0 || nt <= 0 || n_embd <= 0) {
        return false;
    }
    if (!slab_reserve(nbytes)) {
        return false;
    }
    StageState & st = stage();
    StageItem it;
    it.is_k = true;
    it.ty = ty;
    it.nt = nt;
    it.n_embd = n_embd;
    it.nrot = nrot;
    it.n_head = n_head;
    it.n_embd_head = n_embd_head;
    it.n_rot_rope = n_rot_rope;
    it.pos0 = pos0;
    if (theta && n_theta > 0) {
        it.theta.assign(theta, theta + n_theta);
    }
    it.off = st.used;
    it.nbytes = nbytes;
    it.dst = dst;
    std::memcpy(st.slab.data() + st.used, packed, nbytes);
    st.used += nbytes;
    st.items.push_back(std::move(it));
    return true;
}

bool kvmem_stagein_enqueue_v(const void * packed, size_t nbytes, uint8_t * dst,
                             int64_t * set_us) {
    (void) set_us;
    if (!packed || !dst || nbytes == 0) {
        return false;
    }
    if (!slab_reserve(nbytes)) {
        return false;
    }
    StageState & st = stage();
    StageItem it;
    it.is_k = false;
    it.off = st.used;
    it.nbytes = nbytes;
    it.dst = dst;
    std::memcpy(st.slab.data() + st.used, packed, nbytes);
    st.used += nbytes;
    st.items.push_back(std::move(it));
    return true;
}

bool kvmem_stagein_flush(int64_t * copy_us, int64_t * rope_us,
                         int64_t * hadamard_us, int64_t * set_us) {
    (void) copy_us;
    (void) rope_us;
    (void) hadamard_us;
    (void) set_us;
    StageState & st = stage();
    if (st.items.empty()) {
        return true;
    }
    bool ok = true;
    for (const StageItem & it : st.items) {
        if (!it.dst || it.nbytes == 0) {
            ok = false;
            break;
        }
        if (!it.is_k) {
            std::memcpy(it.dst, st.slab.data() + it.off, it.nbytes);
            continue;
        }
        // K chain on CPU: dequant -> RoPE(orig pos) -> FWHT -> quant.
        const uint8_t * src = st.slab.data() + it.off;
        st.f32.assign((size_t) it.nt * (size_t) it.n_embd, 0.0f);
        if (!dequant_block(it.ty, src, st.f32.data(), it.nt, it.n_embd)) {
            ok = false;
            break;
        }
        if (!it.theta.empty() && it.n_rot_rope > 0) {
            rope_neox_rows(st.f32.data(), it.nt, it.n_head, it.n_embd_head,
                           it.n_rot_rope, it.pos0, it.theta.data());
        }
        if (it.nrot > 0) {
            if (!kvmem_stagein_fwht_ok(it.nrot) || (it.n_embd % it.nrot) != 0) {
                ok = false;
                break;
            }
            fwht_rows(st.f32.data(), it.nt, it.n_embd, it.nrot);
        }
        std::vector<uint8_t> out((size_t) it.nt * ggml_row_size(it.ty, it.n_embd));
        if (!quant_block(it.ty, st.f32.data(), out.data(), it.nt, it.n_embd)) {
            ok = false;
            break;
        }
        std::memcpy(it.dst, out.data(), out.size());
    }
    st.used = 0;
    st.items.clear();
    return ok;
}

// Reverse slab: Metal has no device-slab gather; harvest falls back to
// ggml_backend_tensor_get, so these stay inert.
bool kvmem_stageout_enqueue(const void * gpu_src, size_t nbytes) {
    (void) gpu_src;
    (void) nbytes;
    return false;
}

size_t kvmem_stageout_used() {
    return 0;
}

int kvmem_stageout_submit(int64_t * copy_us) {
    (void) copy_us;
    return -1;
}

bool kvmem_stageout_wait(int slot, int64_t * copy_us) {
    (void) slot;
    (void) copy_us;
    return false;
}

const uint8_t * kvmem_stageout_slot_base(int slot) {
    (void) slot;
    return nullptr;
}

void kvmem_stageout_clear() {
}

// Layout gather/scatter runs on the host fallback path instead.
bool kvmem_d2d_batched(const void * const * src, void * const * dst,
                       const size_t * nbytes, int n) {
    (void) src;
    (void) dst;
    (void) nbytes;
    (void) n;
    return false;
}

// Decode mean-K runs on the host accumulator instead.
bool kvmem_meank_ready(uint32_t n_layer, uint32_t n_embd) {
    (void) n_layer;
    (void) n_embd;
    return false;
}

void kvmem_meank_free() {
}

void kvmem_meank_zero(uint32_t il) {
    (void) il;
}

bool kvmem_meank_add(uint32_t il, ggml_type ty, const void * gpu_k,
                     uint32_t tok0, uint32_t n_keep, uint32_t n_embd,
                     int64_t ne0, size_t nb0, size_t nb1, size_t nb2) {
    (void) il;
    (void) ty;
    (void) gpu_k;
    (void) tok0;
    (void) n_keep;
    (void) n_embd;
    (void) ne0;
    (void) nb0;
    (void) nb1;
    (void) nb2;
    return false;
}

bool kvmem_meank_d2h(uint32_t il, float * host, uint32_t n_embd) {
    (void) il;
    (void) host;
    (void) n_embd;
    return false;
}

// Metal layout fast path: tensor-handle batched blit (see header).
// Resolves each tensor to (MTLBuffer, offset) and issues one blit command
// buffer per call. Synchronous (commit + wait); the caller (reselect layout)
// runs between graphs on the same serial MTLCommandQueue, so commit order
// preserves the read-after-write hazard with prior compute.
namespace {

void * metal_buf_ctx(const ggml_tensor * t) {
    if (!t) {
        return nullptr;
    }
    ggml_backend_buffer_t buf = t->view_src ? t->view_src->buffer : t->buffer;
    if (!buf || ggml_backend_buffer_is_host(buf)) {
        return nullptr;
    }
    return buf->context;
}

} // namespace

bool kvmem_metal_blit_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char * e = getenv("KVMEM_METAL_BLIT");
        v = (!e || e[0] == '\0' || e[0] != '0') ? 1 : 0;
    }
    return v == 1;
}

void * kvmem_metal_scratch_alloc_for(const ggml_tensor * ref, size_t nbytes) {
    void * ctx = metal_buf_ctx(ref);
    if (!ctx || nbytes == 0) {
        return nullptr;
    }
    void * queue = ggml_metal_queue_for_buf(ctx);
    if (!queue) {
        return nullptr;
    }
    return ggml_metal_scratch_alloc(queue, nbytes);
}

void kvmem_metal_scratch_free_buf(void * scratch) {
    ggml_metal_scratch_free(scratch);
}

void * kvmem_metal_staging_alloc_for(const ggml_tensor * ref, size_t nbytes) {
    void * ctx = metal_buf_ctx(ref);
    if (!ctx || nbytes == 0) {
        return nullptr;
    }
    void * queue = ggml_metal_queue_for_buf(ctx);
    if (!queue) {
        return nullptr;
    }
    return ggml_metal_staging_alloc(queue, nbytes);
}

const void * kvmem_metal_staging_bytes(const void * staging) {
    if (!staging) {
        return nullptr;
    }
    return ggml_metal_staging_bytes(staging);
}

void kvmem_metal_staging_free_buf(void * staging) {
    ggml_metal_scratch_free(staging);
}

bool kvmem_metal_blit_moves(const struct kvmem_metal_move * moves, int n, void * scratch) {
    if (!moves || n <= 0 || !scratch) {
        return false;
    }
    void * queue = nullptr;
    for (int i = 0; i < n; ++i) {
        if (!moves[i].t || moves[i].nbytes == 0) {
            return false;
        }
        void * ctx = metal_buf_ctx(moves[i].t);
        if (!ctx) {
            return false;
        }
        if (!queue) {
            queue = ggml_metal_queue_for_buf(ctx);
            if (!queue) {
                return false;
            }
        }
    }
    std::vector<const void *> src_buf((size_t) n);
    std::vector<size_t> src_off((size_t) n);
    std::vector<void *> dst_buf((size_t) n);
    std::vector<size_t> dst_off((size_t) n);
    std::vector<size_t> nbytes((size_t) n);
    for (int i = 0; i < n; ++i) {
        void * ctx = metal_buf_ctx(moves[i].t);
        struct ggml_metal_buffer_id bid =
            ggml_metal_buffer_get_id((ggml_metal_buffer_t) ctx, moves[i].t);
        if (!bid.metal) {
            return false;
        }
        nbytes[(size_t) i] = moves[i].nbytes;
        if (moves[i].to_scratch) {
            src_buf[(size_t) i] = bid.metal;
            src_off[(size_t) i] = bid.offs + moves[i].t_off;
            dst_buf[(size_t) i] = scratch;
            dst_off[(size_t) i] = moves[i].scratch_off;
        } else {
            src_buf[(size_t) i] = scratch;
            src_off[(size_t) i] = moves[i].scratch_off;
            dst_buf[(size_t) i] = bid.metal;
            dst_off[(size_t) i] = bid.offs + moves[i].t_off;
        }
    }
    return ggml_metal_blit_batched(queue, src_buf.data(), src_off.data(),
                                   dst_buf.data(), dst_off.data(),
                                   nbytes.data(), n);
}
