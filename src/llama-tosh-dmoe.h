#pragma once

#include "ggml.h"

#include <vector>

struct llama_layer;
struct llama_hparams;

// One MoE layer under mixed execution: the canonical bank stays in host memory and a VRAM arena
// holds only the resident experts, in slot order.
struct llama_tosh_dmoe_layer {
    int il = -1;

    ggml_tensor * arena_gate = nullptr;
    ggml_tensor * arena_up   = nullptr;
    ggml_tensor * arena_down = nullptr;
    ggml_tensor * residency  = nullptr; // [n_expert] i32, slot or -1

    int n_slots = 0;
};

// Enabled by TOSH_DMOE_RESIDENCY=<file of "layer expert" lines>. No-op otherwise.
void llama_tosh_dmoe_init(std::vector<llama_layer> & layers, const llama_hparams & hparams);

// Whether the host executor can run this layer's experts: act 0 is silu, 1 swiglu_oai(alpha, limit), 2 gelu,
// and the gate/up biases the graph adds must be the ones copied at load.
bool llama_tosh_dmoe_accept(const llama_tosh_dmoe_layer * dl, int act, float alpha, float limit, bool gate_b, bool up_b);

// Whether wide batches report their routing to the residency policy.
bool llama_tosh_dmoe_observe_prefill();

// Wide batches: experts routed to at most this many rows of the chunk and not resident run on
// the host (TOSH_DMOE_RARE_ROWS); 0 when off. Only batches the GPU takes, from the offload size.
int llama_tosh_dmoe_rare_rows();
// the last tokens of a chunk keep their experts on the GPU, where prefill can make them resident
int llama_tosh_dmoe_rare_tail();
int llama_tosh_dmoe_wide_min_tokens();

// TOSH_DMOE_CACHE_MIB=auto defers the arena until a context knows its own VRAM: the arena is
// what remains of the card after it, the prefill slots and a reserve (TOSH_DMOE_RESERVE_MIB).
bool llama_tosh_dmoe_budget_pending();
void llama_tosh_dmoe_budget(size_t gpu_bytes);
// A context created while no other is alive may resize an automatic arena; contexts are counted.
bool llama_tosh_dmoe_rebudget_possible();
void llama_tosh_dmoe_context_opened();
void llama_tosh_dmoe_context_closed();

// Stops the host executor and the residency policy before the model's buffers go away.
void llama_tosh_dmoe_shutdown();

// The mixed-execution state for the layer whose gate bank is `gate_exps`, or null.
const llama_tosh_dmoe_layer * llama_tosh_dmoe_layer_of(const ggml_tensor * gate_exps);
