#pragma once

// GPU stage-in for quantized raw-K:
//   H2D packed q8/q4 → dequant F32 → NeoX RoPE → Walsh-Hadamard → quant into cache.
// F32-H2D path remains for fallback / unrotated V. Host Hadamard/RoPE is fallback only.

#include "ggml.h"

#include <cstddef>
#include <cstdint>

bool kvmem_stagein_gpu_ready(size_t n_f32, size_t n_packed = 0);
void kvmem_stagein_gpu_free();

bool kvmem_stagein_fwht_ok(int nrot);
bool kvmem_stagein_quant_ok(ggml_type ty);

// Host buffer is free to reuse after return (H2D is fenced). Later kernels stay async.
bool kvmem_stagein_h2d_f32(const float * host, int64_t n);
bool kvmem_stagein_h2d_packed(const void * host, size_t n);
bool kvmem_stagein_dequant(ggml_type ty, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_rope_neox(int64_t n_tokens, int n_head, int n_embd_head, int n_rot,
                             int32_t pos0, const float * theta, int n_theta);
bool kvmem_stagein_fwht(int64_t n_rows, int64_t n_embd, int nrot);
bool kvmem_stagein_quantize(ggml_type ty, void * gpu_dst, int64_t n_rows, int64_t n_embd);
bool kvmem_stagein_h2d_bytes(void * gpu_dst, const void * host, size_t n);
void kvmem_stagein_sync();

// 32 MiB packed host+GPU slab. Enqueue (block,layer) rows; flush does one H2D
// then dequant/RoPE/Hadamard/quant (K) or D2D (packed V). Extra VRAM = 32 MiB.
bool kvmem_stagein_enqueue_k(
        ggml_type ty, const void * packed, size_t nbytes, uint8_t * dst,
        int64_t nt, int64_t n_embd, int nrot,
        int n_head, int n_embd_head, int n_rot_rope, int32_t pos0,
        const float * theta, int n_theta,
        int64_t * copy_us, int64_t * rope_us, int64_t * hadamard_us, int64_t * set_us);
bool kvmem_stagein_enqueue_v(const void * packed, size_t nbytes, uint8_t * dst,
                             int64_t * set_us);
bool kvmem_stagein_flush(int64_t * copy_us, int64_t * rope_us,
                         int64_t * hadamard_us, int64_t * set_us);

// Reverse slab: packed GPU V → pin. Gather kernel packs scattered rows into
// the 32 MiB GPU slab, then one D2H. Two host pins; no extra VRAM slab.
bool kvmem_stageout_enqueue(const void * gpu_src, size_t nbytes);
size_t kvmem_stageout_used();
int kvmem_stageout_submit(int64_t * copy_us);
bool kvmem_stageout_wait(int slot, int64_t * copy_us);
const uint8_t * kvmem_stageout_slot_base(int slot);
void kvmem_stageout_clear();

// Batched device copies (layout gather/scatter). One kernel; false → caller D2D.
bool kvmem_d2d_batched(const void * const * src, void * const * dst,
                       const size_t * nbytes, int n);

#if defined(LLAMA_KVMEM_METAL)
// Metal layout fast path (tensor handles, blit D2D, synchronous).
// Discrete GPUs have no CPU-visible device memory, so the raw-pointer API
// above can never resolve there; these carry ggml tensors instead and
// resolve (MTLBuffer, offset) via ggml_metal_buffer_get_id internally.
// Default on; KVMEM_METAL_BLIT=0 forces the host round-trip fallback.
struct kvmem_metal_move {
    const ggml_tensor * t; // device K/V cache tensor (same MTLDevice for all)
    size_t t_off;          // byte offset of the row range within the tensor
    size_t scratch_off;    // byte offset within scratch
    size_t nbytes;
    int to_scratch;        // 1: tensor -> scratch (gather), 0: scratch -> tensor (scatter)
};
bool kvmem_metal_blit_enabled(void);
// scratch is a private MTLBuffer of at least nbytes; ref supplies the device.
void * kvmem_metal_scratch_alloc_for(const ggml_tensor * ref, size_t nbytes);
void kvmem_metal_scratch_free_buf(void * scratch);
bool kvmem_metal_blit_moves(const struct kvmem_metal_move * moves, int n, void * scratch);
// Harvest D2H: same move list (to_scratch=1) into a SHARED staging buffer,
// then read back on the host with zero extra copies. One blit + one wait
// replaces N per-row tensor_get calls (each = temp buffer + blit + wait).
void * kvmem_metal_staging_alloc_for(const ggml_tensor * ref, size_t nbytes);
const void * kvmem_metal_staging_bytes(const void * staging);
void kvmem_metal_staging_free_buf(void * staging);
#endif

// Decode mean-K running sum on GPU. Extra VRAM = n_layer * n_embd * 4.
bool kvmem_meank_ready(uint32_t n_layer, uint32_t n_embd);
void kvmem_meank_free();
void kvmem_meank_zero(uint32_t il);
bool kvmem_meank_add(uint32_t il, ggml_type ty, const void * gpu_k,
                     uint32_t tok0, uint32_t n_keep, uint32_t n_embd,
                     int64_t ne0, size_t nb0, size_t nb1, size_t nb2);
bool kvmem_meank_d2h(uint32_t il, float * host, uint32_t n_embd);
