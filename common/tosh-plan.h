#pragma once

#include "ggml.h"

#include <string>
#include <vector>

struct common_params;

// Context-first memory plan for MoE models: the requested context and a VRAM reserve come first,
// expert residency takes what is left. Enabled by TOSH_AUTO (auto, or full / dmoe / legacy to
// force an execution mode); TOSH_AUTO_KV picks the cache type (auto, f16, q8_0, turbo4).

enum tosh_plan_state {
    TOSH_PLAN_NOT_PLANNED,          // dense model, or planning off
    TOSH_PLAN_FULL_GPU_OPTIMAL,     // weights and the requested workload fit with the reserve
    TOSH_PLAN_DMOE_CONTEXT_OPTIMAL, // the weights fit, the requested context does not
    TOSH_PLAN_DMOE_CAPACITY_REQUIRED,
    TOSH_PLAN_LEGACY_OFFLOAD_BETTER,
    TOSH_PLAN_UNSUPPORTED,
    TOSH_PLAN_DMOE_BOUNDED_HOST,    // experimental: part of the expert bank in RAM, the rest read from the file
};

// execution mode of a plan, for callers; tosh_plan_mode_name gives its stable code
enum tosh_plan_mode {
    TOSH_PLAN_MODE_NONE,            // not planned
    TOSH_PLAN_MODE_FULL_GPU,
    TOSH_PLAN_MODE_FULL_HOST_DMOE,
    TOSH_PLAN_MODE_BOUNDED_DMOE,
    TOSH_PLAN_MODE_CLASSIC_NCMOE,
    TOSH_PLAN_MODE_UNSUPPORTED,
};

// version of the "product" object in the plan JSON
#define TOSH_PLAN_SCHEMA_VERSION 1

struct tosh_plan_candidate {
    std::string name;
    std::string mode;               // full_gpu, dmoe, legacy_offload
    ggml_type   kv      = GGML_TYPE_F16;
    int         ubatch  = 512;
    int         ncmoe   = 0;
    bool        valid   = false;
    std::string reason;
    double      private_mib = 0;    // projected VRAM this process holds
    double      free_mib    = 0;    // projected VRAM left on the card
    double      arena_mib   = 0;
    double      kv_mib      = 0;
    double      compute_mib = 0;
    double      need_vram_mib = 0;  // VRAM with the reserve at the mode's smallest useful size
    double      need_host_mib = 0;
};

struct tosh_plan {
    tosh_plan_state state = TOSH_PLAN_NOT_PLANNED;
    tosh_plan_mode  plan_mode = TOSH_PLAN_MODE_NONE;
    std::string primary_reason;     // why this mode and not the preferred one, e.g. CURRENT_RAM_LIMIT
    std::vector<std::string> warnings;  // e.g. COVERAGE_CONSTRAINED, RESERVE_LOWERED, REPLANNED
    std::string limiting_resource;  // VRAM, HOST_RAM, WIRE, METADATA or NONE
    std::string coverage_state;     // GOOD, CONSTRAINED or REJECT for DMoE plans
    double      kv_mib = 0;
    // what the smallest mode would need against what is there, set when nothing fits
    double      required_host_mib = 0, available_host_mib = 0, required_vram_mib = 0, available_vram_mib = 0;
    double      recheck_ms = 0;     // from the memory snapshot to the check before committing
    double      prompt_cache_mib = 0;  // the server's RAM prompt cache the plan leaves room for
    int         n_ctx_checkpoints = 0; // context checkpoints per sequence the plan leaves room for
    double      checkpoint_mib = 0;    // one checkpoint: the sliding-window cache of a sequence
    std::string mode;               // selected execution mode
    // PLAN_FULL_GPU, PLAN_FULL_HOST_DMOE, PLAN_BOUNDED_DMOE, PLAN_CLASSIC_NCMOE or PLAN_UNSUPPORTED
    std::string code;
    std::vector<std::string> limits;  // what ruled out the preferred modes, e.g. VRAM_LIMIT, COVERAGE_TOO_LOW
    std::string reason;
    std::string fallback;           // why a preferred mode was not used
    ggml_type   kv     = GGML_TYPE_F16;
    int         ubatch = 512;
    int         n_ctx  = 0;
    int         ncmoe  = 0;
    double      reserve_mib        = 0;
    double      arena_mib          = 0;
    double      min_arena_mib      = 0;
    double      private_mib        = 0;
    double      free_mib           = 0;
    double      vram_total_mib     = 0;
    double      vram_free_mib      = 0;
    double      host_required_mib  = 0;
    double      host_ram_mib       = 0;
    double      host_available_mib = 0;
    double      host_reserve_mib   = 0;
    double      bank_mib           = 0;
    double      model_mib          = 0;  // every tensor, across all shards
    double      host_weights_mib   = 0;  // weights the backends keep in host memory besides the bank
    double      host_lazy_mib      = 0;  // weights read from the file on demand, outside host_weights_mib
    double      warm_floor_mib     = 0;  // smallest RAM cache that holds one layer per expert size
    int         n_split            = 1;
    std::string mlock;              // required, not_required, not_lockable
    std::string dispersion;         // normal, high
    // TOSH_AUTO_HOST_POLICY=runtime (experimental): host memory budget from the machine's state now
    std::string host_policy;        // static, runtime
    std::string host_pressure;      // NORMAL, ELEVATED, CRITICAL
    std::string full_host_code;     // FULL_HOST_SAFE or why the whole bank is not taken
    std::string bounded_code;       // BOUNDED_HOST_SELECTED or why not
    bool        host_simulated    = false;
    bool        mtp_head_dropped  = false;   // the MTP head left no room and the plan goes on without it
    bool        mtp_head_arena    = false;   // dropped because it took too much of the expert arena
    bool        swap_growing      = false;
    double      reclaimable_mib   = 0;
    double      wired_mib         = 0;
    double      compressed_mib    = 0;
    double      swap_used_mib     = 0;
    double      self_rss_mib      = 0;
    double      static_reserve_mib = 0;
    double      headroom_mib      = 0;
    double      static_cap_mib    = 0;  // every budget below is bytes this load may still add
    double      dynamic_cap_mib   = 0;
    double      lockable_mib      = 0;
    double      effective_mib     = 0;
    double      warm_mib          = 0;
    double      coverage          = 0;
    double      projected_rss_mib = 0;
    double      projected_left_mib = 0; // reclaimable memory left to the system after the load
    std::vector<tosh_plan_candidate> candidates;
};

const char * tosh_plan_state_name(tosh_plan_state s);
const char * tosh_plan_mode_name(tosh_plan_mode m);
std::string  tosh_plan_json(const tosh_plan & plan);

// Plans and applies the result to params. Returns false when planning is off or not needed, and
// when no plan exists: then out->state is TOSH_PLAN_UNSUPPORTED and the model must not be loaded.
bool tosh_plan_apply(common_params & params, tosh_plan * out = nullptr);

// For a model the plan did not take: checkpoints of a recurrent state stay exact while enough of them fit in
// the free memory, and drop to 8 bits otherwise; their number is cut to what fits either way.
void tosh_plan_checkpoints(common_params & params);

// Outcome of the load that followed the last plan: READY, PLAN_REFUSED, MODEL_LOAD_FAILED,
// ALLOCATION_FAILED or CONTEXT_FAILED, with the measured RSS and VRAM next to the projection.
// Written to <TOSH_AUTO_PLAN_FILE>.actual; no-op when nothing was planned.
void tosh_plan_report_load(const char * result, const char * detail = nullptr);
