#pragma once

#include "ggml-metal-device.h"

#ifdef __cplusplus
extern "C" {
#endif

//
// backend context
//

typedef struct ggml_metal * ggml_metal_t;

// prefetch contexts get their own command queue and the no-copy upload path
// (see ggml_metal_set_tensor_async); everything else shares the device queue
ggml_metal_t ggml_metal_init(ggml_metal_device_t dev, bool prefetch);
void ggml_metal_free(ggml_metal_t ctx);

const char * ggml_metal_get_name(ggml_metal_t ctx);
uint64_t     ggml_metal_peer_group_id(ggml_metal_t ctx);

void ggml_metal_synchronize(ggml_metal_t ctx);

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size);
void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size);

// batch small synchronous reads into one device round-trip
void ggml_metal_read_batch_begin(ggml_metal_t ctx);
void ggml_metal_read_batch_end  (ggml_metal_t ctx);
// prefer_events picks the event hand-off over the peer copy when both are enabled; the caller
// decides, since peer wins the prefill and events the decode.
// hold_dst leaves the destination command buffer open for the add that follows the copy, which
// is only correct when the caller does issue that add right away. The source side is never held:
// it has to start running while the rest is still being encoded.
bool ggml_metal_cpy_tensor_async_ex(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst, bool prefer_events, bool hold_dst);
bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst);

// acc += add, both f32 and contiguous. Encodes the kernel directly so a tensor-split
// allreduce does not pay a full graph submission per step.
bool ggml_metal_add_inplace_supported(ggml_metal_t ctx, const struct ggml_tensor * acc, const struct ggml_tensor * add);
bool ggml_metal_add_inplace_async    (ggml_metal_t ctx, struct ggml_tensor * acc, const struct ggml_tensor * add);

// f16 transport for the prefill allreduce (TOSH_MGPU_STAGE_F16): halve the bytes crossing
// the peer bridge by carrying partials as f16 and folding them into the f32 accumulator.
bool ggml_metal_f16_stage_supported(ggml_metal_t ctx, const struct ggml_tensor * t);
bool ggml_metal_cvt_f32_f16        (ggml_metal_t ctx, const struct ggml_tensor * src, struct ggml_tensor * dst);
bool ggml_metal_add_inplace_f16_src(ggml_metal_t ctx, struct ggml_tensor * acc, const struct ggml_tensor * add);

// One command buffer per card per butterfly step: both copies and the add together.
// t_peer lets a peer-connected pair read the partner's partial straight out of its VRAM;
// pass NULL and the exchange always stages through the shared host block.
bool ggml_metal_exchange_reduce(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                struct ggml_tensor * t_self, struct ggml_tensor * t_peer,
                                struct ggml_tensor * tmp_self, uint64_t seq);

// Create and reserve both directed links of a pair, so exchange_reduce calls running on
// different host threads cannot race to create the same lazy link or resize its block while
// a partner is encoding against it. Must run serially before a parallel encode round.
bool ggml_metal_xdev_prepare(ggml_metal_t ctx_self, ggml_metal_t ctx_peer, size_t size);

// One-shot all-reduce for decode batches: every card pushes its partial into every partner's
// inbox in a single round and reduces from local VRAM, replacing the staged butterfly. The
// seq must come from the same counter the exchanges use: links may skip values but never go
// backwards. Returns false, having encoded nothing, when the group cannot run it.
bool ggml_metal_allreduce_oneshot(ggml_metal_t * ctxs, struct ggml_tensor ** tensors,
                                  size_t n, uint64_t seq, uint32_t zero_mask);

// Zero a tensor with a blit fill on its own device. The allreduce uses it for the slices no
// device computed, so the metal path can contribute zeros instead of falling back to the
// generic butterfly. Queued ahead of whatever the collective encodes next on this device.
bool ggml_metal_zero_tensor(ggml_metal_t ctx, struct ggml_tensor * t);

enum ggml_status ggml_metal_graph_compute (ggml_metal_t ctx, struct ggml_cgraph * gf);
void             ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf);

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev);
void ggml_metal_event_wait  (ggml_metal_t ctx, ggml_metal_event_t ev);

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx);

void ggml_metal_set_n_cb            (ggml_metal_t ctx, int n_cb);
void ggml_metal_set_abort_callback  (ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data);

bool ggml_metal_supports_family     (ggml_metal_t ctx, int family);
void ggml_metal_capture_next_compute(ggml_metal_t ctx);

#ifdef __cplusplus
}
#endif
