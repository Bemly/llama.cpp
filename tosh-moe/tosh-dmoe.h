// Rendezvous between the GPU side of mixed expert execution and the host executor.
// The GPU publishes a layer's routing into a shared stage and signals an odd value; the host
// computes the non-resident experts, writes them into the same stage and signals value + 1,
// which the GPU waits on.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

#ifdef __cplusplus
extern "C" {
#endif

#define TOSH_DMOE_MAX_LAYERS 128
#define TOSH_DMOE_MAX_TOKENS 8
#define TOSH_DMOE_RING       1024
#define TOSH_DMOE_VERSIONS   4

struct tosh_dmoe_job {
    uint64_t value;     // the publish value this job answers
    int32_t  il;
    int32_t  n_tokens;
    int32_t  n_used;
    int32_t  n_embd;
    int32_t  version;   // residency table the GPU read, dynamic residency only
    uint32_t tick;      // the graph it belongs to
};

// Stage layout of one layer, in bytes from the layer base.
static inline size_t tosh_dmoe_off_act  (int n_embd, int n_used) { (void) n_embd; (void) n_used; return 0; }
static inline size_t tosh_dmoe_off_ids  (int n_embd, int n_used) { (void) n_used; return (size_t) n_embd*TOSH_DMOE_MAX_TOKENS*4; }
static inline size_t tosh_dmoe_off_slots(int n_embd, int n_used) { return tosh_dmoe_off_ids(n_embd, n_used) + (size_t) n_used*TOSH_DMOE_MAX_TOKENS*4; }
static inline size_t tosh_dmoe_off_out  (int n_embd, int n_used) {
    const size_t o = tosh_dmoe_off_slots(n_embd, n_used) + (size_t) n_used*TOSH_DMOE_MAX_TOKENS*4;
    return (o + 255) & ~(size_t) 255;
}
static inline size_t tosh_dmoe_layer_stride(int n_embd, int n_used) {
    const size_t o = tosh_dmoe_off_out(n_embd, n_used) + (size_t) n_embd*n_used*TOSH_DMOE_MAX_TOKENS*4;
    return (o + 4095) & ~(size_t) 4095;
}

// Wide batches (prefill): experts routed to few rows of the chunk run on the host while the rest
// is uploaded. One stage serves every layer, since a layer's join precedes the next publish.
#define TOSH_DMOE_WIDE_MAX_TOKENS 2048
#define TOSH_DMOE_WIDE_MAX_ROWS   1024
static inline size_t tosh_dmoe_wide_off_ids(int n_tok, int n_embd, int n_used) { (void) n_used; return (size_t) n_tok*n_embd*4; }
static inline size_t tosh_dmoe_wide_off_rmp(int n_tok, int n_embd, int n_used) { return tosh_dmoe_wide_off_ids(n_tok, n_embd, n_used) + (size_t) n_tok*n_used*4; }
static inline size_t tosh_dmoe_wide_off_map(int n_tok, int n_embd, int n_used) { return tosh_dmoe_wide_off_rmp(n_tok, n_embd, n_used) + (size_t) n_tok*n_used*4; }
static inline size_t tosh_dmoe_wide_off_out(int n_tok, int n_embd, int n_used) {
    return (tosh_dmoe_wide_off_map(n_tok, n_embd, n_used) + (size_t) n_tok*n_used*4 + 255) & ~(size_t) 255;
}
static inline size_t tosh_dmoe_wide_bytes(int n_embd, int n_used) {
    return tosh_dmoe_wide_off_out(TOSH_DMOE_WIDE_MAX_TOKENS, n_embd, n_used) + (size_t) TOSH_DMOE_WIDE_MAX_ROWS*n_embd*4;
}
void   tosh_dmoe_set_wide_stage(void * host, size_t bytes);
void * tosh_dmoe_wide_stage(size_t * bytes);

// Rare-row threshold the next wide publish uses. With adapt, it moves one row at a time to keep
// the share of layers whose join is already waiting when the host finishes near a target
// (TOSH_DMOE_RARE_TARGET): a short wait means the host took rows the uploads no longer carry.
void tosh_dmoe_rare_rows_init(int rows, int max_rows, bool adapt);
int  tosh_dmoe_rare_rows_live(void);
void tosh_dmoe_rare_host_done(int il, double t_start, double t_end, bool gpu_waiting);
// the GPU signals this event with a wide layer's publish value when it reaches that layer's join
void     tosh_dmoe_set_reach_event(void * ev, uint64_t (*get)(void *));
uint64_t tosh_dmoe_reach_get(void);
void tosh_dmoe_rare_dma_done(int il, double t_end);
void tosh_dmoe_rare_chunk(double t_start, int n_tokens);
// a decode step ran: the next chunk's period would include it, so it is not a measurement
void tosh_dmoe_rare_interrupt(void);

// Prefill timeline for overlap studies (TOSH_DMOE_TIMELINE=<file>): kind, layer, host seconds.
void tosh_dmoe_timeline(int kind, int il, double t0, double t1);
double tosh_dmoe_host_seconds(void);

// Free VRAM before this process allocated anything, and the card's total, as the driver sees them.
void tosh_dmoe_set_vram(size_t free_at_start, size_t total);
bool tosh_dmoe_vram(size_t * free_at_start, size_t * total);
// The driver's live counters (free, and in use by every process), set by the GPU backend.
void tosh_dmoe_set_vram_query(bool (*fn)(int64_t * vfree, int64_t * vused));
// Free VRAM as the card's size less what every process holds, median of three samples. The
// driver's free counter lags behind released memory; a disagreement is logged, not used.
bool tosh_vram_read(size_t * vfree, size_t * total);
// Mixed execution chosen at run time, after the environment was first read.
void tosh_dmoe_enable(void);
// VRAM this process holds in the GPU backend's own buffers, 0 when the backend does not say
void   tosh_dmoe_set_private_query(size_t (*fn)(void));
size_t tosh_dmoe_private_bytes(void);

// Whether mixed expert execution is configured for this process; nothing runs otherwise.
bool tosh_dmoe_enabled(void);

// Set by the GPU backend.
void     tosh_dmoe_set_stage(void * host, size_t layer_stride);
void *   tosh_dmoe_stage_layer(int il);
void     tosh_dmoe_set_event(void * ev, uint64_t (*get)(void *), void (*set)(void *, uint64_t));
uint64_t tosh_dmoe_event_get(void);
void     tosh_dmoe_event_set(uint64_t value);

// Publish values are odd and grow in GPU order; the answer is value + 1. They are handed out
// when a graph is submitted, so a publish and its join agree however the graph is split.
uint64_t tosh_dmoe_next_value(void);

struct tosh_dmoe_job * tosh_dmoe_job_of(uint64_t value);

// The host executor only spins while a graph with mixed layers is in flight.
void tosh_dmoe_set_active(bool active);
bool tosh_dmoe_wait_active(int timeout_us);
bool tosh_dmoe_is_active(void);

// Dynamic residency. The GPU reads residency from host memory laid out as
// [TOSH_DMOE_VERSIONS][TOSH_DMOE_MAX_LAYERS][n_expert] int32; each graph takes the published version when it is
// submitted, so a table never changes under a running token.
void      tosh_dmoe_set_tables(int32_t * host, size_t bytes, int n_layers, int n_expert);
int32_t * tosh_dmoe_tables(size_t * bytes, int * n_layers, int * n_expert);
void      tosh_dmoe_publish_version(int version);
int       tosh_dmoe_published_version(void);

// Graphs are counted as they are submitted. The completed count grows at every synchronize,
// which waits for every graph submitted before it; submissions and synchronizes come from one
// thread, so whatever a table published before sync k named is no longer read after sync k + 2.
uint64_t tosh_dmoe_tick_next(void);
// the last submitted graph was a wide batch (prefill) rather than mixed decode
void     tosh_dmoe_note_wide(bool wide);
bool     tosh_dmoe_wide(void);
uint64_t tosh_dmoe_tick(void);
void     tosh_dmoe_mark_completed(void);
uint64_t tosh_dmoe_completed_value(void);
uint64_t tosh_dmoe_last_value(void);
bool     tosh_dmoe_wait_completed(uint64_t seen, int timeout_us);

// Uploads host bytes into GPU tensors on a queue of their own, through a bounded stage.
struct tosh_dmoe_copy {
    const void * dst;       // ggml_tensor
    size_t       offset;
    const void * src;
    size_t       size;
};
struct tosh_dmoe_uploader {
    void *   ctx;
    uint64_t (*upload)(void * ctx, const struct tosh_dmoe_copy * copies, int n);   // 0 on failure
    int      (*status)(void * ctx, uint64_t ticket);    // 0 pending, 1 done, -1 failed
    double   (*gpu_us)(void * ctx, uint64_t ticket);    // device time of a finished upload
    // the other way: src is the ggml_tensor and dst host memory; returns once the bytes are there
    bool     (*download)(void * ctx, const struct tosh_dmoe_copy * copies, int n);
};
void                              tosh_dmoe_set_uploader(const struct tosh_dmoe_uploader * u);
const struct tosh_dmoe_uploader * tosh_dmoe_get_uploader(void);
// the GPU backend makes the uploader the first time it is asked for
void tosh_dmoe_set_uploader_factory(void (*make)(void));

// Prefill. A bank upload asks for the layer's residency so resident experts come from the
// VRAM arena and only the rest from host memory. Registered by whoever owns the banks.
struct tosh_dmoe_bank_plan {
    const void *    arena;          // ggml_tensor holding the resident copies, slot-major
    const int32_t * slot_of;        // n_expert entries, slot or -1
    int             n_expert;
    size_t          expert_bytes;
    // experts to copy from the freshly filled upload into free arena slots, for decode
    const int32_t * promo_expert;
    const int32_t * promo_slot;
    int             n_promo;
    // rows per expert of the chunk being uploaded, or null to carry the whole bank
    const int32_t * rows;
    int             il;
    int             which;          // 0 gate, 1 up, 2 down
};
typedef bool (*tosh_dmoe_bank_plan_fn)(const void * host, size_t size, struct tosh_dmoe_bank_plan * plan);
void tosh_dmoe_set_bank_planner(tosh_dmoe_bank_plan_fn fn);
bool tosh_dmoe_bank_planner(const void * host, size_t size, struct tosh_dmoe_bank_plan * plan);

// Prefill routing: the GPU copies a layer's ids into a host-visible ring and the policy reads
// them once the graph has completed. Records are overwritten when the ring wraps.
#define TOSH_DMOE_OBS_RING_BYTES (32u << 20)
#define TOSH_DMOE_OBS_RECORDS    4096
struct tosh_dmoe_obs_record {
    uint64_t seq;           // 0 while unused
    uint64_t sync;          // completed count when it was encoded
    uint64_t pos;           // bytes reserved before it, never wraps
    uint32_t offset;        // bytes into the ring
    int32_t  il;
    int32_t  n_used;
    int32_t  n_tokens;
    uint32_t tick;
};
void                          tosh_dmoe_set_obs_ring(void * host);
void *                        tosh_dmoe_obs_ring(void);
struct tosh_dmoe_obs_record * tosh_dmoe_obs_reserve(int il, int n_used, int n_tokens);   // null if too big
struct tosh_dmoe_obs_record * tosh_dmoe_obs_record_at(uint64_t seq);
uint64_t                      tosh_dmoe_obs_head(void);
bool                          tosh_dmoe_obs_intact(const struct tosh_dmoe_obs_record * r);

// A failure latches until cleared: every later publish is answered at once so the GPU never
// waits forever, and the caller aborts the batch.
void         tosh_dmoe_fail(const char * why);
const char * tosh_dmoe_failed(void);
void         tosh_dmoe_clear_failure(void);

// Compact prefill: a wide batch uploads only the experts its chunk routes to. The scheduler reads
// the chunk's ids and sets the rows per expert right before the bank upload that consumes them.
bool            tosh_dmoe_compact(void);
void            tosh_dmoe_set_chunk_rows(const int32_t * rows, const float * tail, int n_expert);
const int32_t * tosh_dmoe_chunk_rows(int * n_expert);
// per expert, its routing weighted by closeness to the chunk's end (TOSH_DMOE_PREFILL_TOKEN_DECAY)
const float *   tosh_dmoe_chunk_tail(void);
float           tosh_dmoe_token_decay(void);
// prefill time accounting: 0 waiting for the chunk's ids, 1 issuing bank uploads, 2 upload GPU time,
// 3 counting rows per expert
void            tosh_dmoe_note_time(int what, int64_t us);

#ifdef __cplusplus
}
#endif
