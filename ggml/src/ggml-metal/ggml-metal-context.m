#include <os/lock.h>
#import "ggml-metal-context.h"

#import "ggml-impl.h"
#import "ggml-backend-impl.h"

#import "ggml-metal-impl.h"
#import "ggml-metal-common.h"
#import "ggml-metal-ops.h"
#import "ggml-metal-fusion.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-dmoe.h"
#include "tosh-hostcache.h"
#endif

#import <Foundation/Foundation.h>

#import <Metal/Metal.h>

#undef MIN
#undef MAX
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#define MAX(a, b) ((a) > (b) ? (a) : (b))

// max number of extra MTLCommandBuffers used to submit a graph for processing
// (the main thread owns one more on top of n_cb)
#define GGML_METAL_MAX_COMMAND_BUFFERS 16
#define GGML_METAL_MAX_PENDING_WAITS   8
#define GGML_METAL_MAX_PREFETCH_STAGES 8

#include <stdatomic.h>
#include <time.h>

// TOSH_CB_PROFILE=1: per-command-buffer GPU accounting, bucketed by graph size.
// Off by default, and it attaches no completion handler when off.
#define TOSH_CBP_MAX_BUCKETS 24

struct tosh_cbp_bucket {
    _Atomic int      n_nodes;    // 0 = free slot
    _Atomic uint64_t n_graphs;
    _Atomic uint64_t n_cbs;
    _Atomic uint64_t gpu_ns;     // sum of GPUEndTime - GPUStartTime
    _Atomic uint64_t ker_ns;     // sum of kernelEndTime - kernelStartTime
    _Atomic uint64_t enc_ns;     // CPU time inside ggml_metal_graph_compute
    _Atomic uint64_t span_ns;    // per graph: max(GPUEndTime) - min(GPUStartTime)
    _Atomic uint64_t gpu_min_ns; // busiest/idlest single graph, to expose the warmup
    _Atomic uint64_t gpu_max_ns;
};

// CPU side of synchronize, kept apart from the encode so a blocked wait is not read
// as scheduler waste
static _Atomic uint64_t g_cbp_wait_ns;   // blocked in waitUntilCompleted
static _Atomic uint64_t g_cbp_drain_ns;  // blocked draining staged uploads
static _Atomic uint64_t g_cbp_sync_n;
static _Atomic uint64_t g_cbp_ops_ns;    // encoding the ops of a command buffer
static _Atomic uint64_t g_cbp_commit_ns; // the commit call itself
static _Atomic uint64_t g_cbp_create_ns; // creating and enqueuing the buffers of a graph

static struct tosh_cbp_bucket g_cbp[TOSH_CBP_MAX_BUCKETS];
static _Atomic int g_cbp_state = 0; // 0 unknown, 1 off, 2 on

static bool tosh_cbp_on(void) {
    int st = atomic_load_explicit(&g_cbp_state, memory_order_relaxed);
    if (st == 0) {
        const char * e = TOSH_ENV("TOSH_CB_PROFILE");
        st = (e != NULL && strcmp(e, "0") != 0) ? 2 : 1;
        atomic_store_explicit(&g_cbp_state, st, memory_order_relaxed);
    }
    return st == 2;
}

static struct tosh_cbp_bucket * tosh_cbp_bucket_for(int n_nodes) {
    for (int i = 0; i < TOSH_CBP_MAX_BUCKETS; ++i) {
        int cur = atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed);
        if (cur == n_nodes) {
            return &g_cbp[i];
        }
        if (cur == 0) {
            int expected = 0;
            if (atomic_compare_exchange_strong(&g_cbp[i].n_nodes, &expected, n_nodes)) {
                return &g_cbp[i];
            }
            if (atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed) == n_nodes) {
                return &g_cbp[i];
            }
        }
    }
    return NULL;
}

// per-graph GPU timeline, filled by the completion handlers of that graph
struct tosh_cbp_graph {
    struct tosh_cbp_bucket * bucket;
    _Atomic uint64_t t_first_ns;  // min GPUStartTime, ns
    _Atomic uint64_t t_last_ns;   // max GPUEndTime, ns
    _Atomic uint64_t gpu_ns;      // this graph's own GPU busy time
    _Atomic int      n_left;      // command buffers still running
};

static void tosh_cbp_attach(id<MTLCommandBuffer> cb, struct tosh_cbp_graph * g) {
    if (g == NULL) {
        return;
    }
    atomic_fetch_add_explicit(&g->n_left, 1, memory_order_relaxed);
    [cb addCompletedHandler:^(id<MTLCommandBuffer> b) {
        const double t0 = [b GPUStartTime];
        const double t1 = [b GPUEndTime];
        const double k0 = [b kernelStartTime];
        const double k1 = [b kernelEndTime];

        atomic_fetch_add_explicit(&g->bucket->n_cbs, 1, memory_order_relaxed);
        if (t1 > t0) {
            const uint64_t d = (uint64_t)((t1 - t0)*1e9);
            atomic_fetch_add_explicit(&g->bucket->gpu_ns, d, memory_order_relaxed);
            atomic_fetch_add_explicit(&g->gpu_ns,         d, memory_order_relaxed);
        }
        if (k1 > k0) {
            atomic_fetch_add_explicit(&g->bucket->ker_ns, (uint64_t)((k1 - k0)*1e9), memory_order_relaxed);
        }

        const uint64_t s_ns = (uint64_t)(t0*1e9);
        const uint64_t e_ns = (uint64_t)(t1*1e9);
        uint64_t cur = atomic_load_explicit(&g->t_first_ns, memory_order_relaxed);
        while (s_ns < cur || cur == 0) {
            if (atomic_compare_exchange_weak(&g->t_first_ns, &cur, s_ns)) break;
        }
        cur = atomic_load_explicit(&g->t_last_ns, memory_order_relaxed);
        while (e_ns > cur) {
            if (atomic_compare_exchange_weak(&g->t_last_ns, &cur, e_ns)) break;
        }

        if (atomic_fetch_sub_explicit(&g->n_left, 1, memory_order_acq_rel) == 1) {
            const uint64_t a = atomic_load_explicit(&g->t_first_ns, memory_order_relaxed);
            const uint64_t z = atomic_load_explicit(&g->t_last_ns,  memory_order_relaxed);
            if (z > a) {
                atomic_fetch_add_explicit(&g->bucket->span_ns, z - a, memory_order_relaxed);
            }
            const uint64_t own = atomic_load_explicit(&g->gpu_ns, memory_order_relaxed);
            uint64_t m = atomic_load_explicit(&g->bucket->gpu_min_ns, memory_order_relaxed);
            while (m == 0 || own < m) {
                if (atomic_compare_exchange_weak(&g->bucket->gpu_min_ns, &m, own)) break;
            }
            m = atomic_load_explicit(&g->bucket->gpu_max_ns, memory_order_relaxed);
            while (own > m) {
                if (atomic_compare_exchange_weak(&g->bucket->gpu_max_ns, &m, own)) break;
            }
            free(g);
        }
    }];
}

// definidos en ggml-metal-device.m
extern _Atomic uint64_t g_tosh_d2h_ns, g_tosh_d2h_bytes, g_tosh_d2h_n;
extern _Atomic uint64_t g_tosh_h2d_ns, g_tosh_h2d_bytes, g_tosh_h2d_n;

static void tosh_cbp_dump_dev(ggml_metal_device_t dev) {
    if (!tosh_cbp_on()) {
        return;
    }
    {
        const uint64_t dn = atomic_load_explicit(&g_tosh_d2h_n, memory_order_relaxed);
        const uint64_t hn = atomic_load_explicit(&g_tosh_h2d_n, memory_order_relaxed);
        fprintf(stderr, "TOSH_CBP  readback: %llu lecturas, %.2f MiB, %.1f ms | subidas: %llu, %.2f MiB, %.1f ms\n",
                (unsigned long long) dn,
                atomic_load_explicit(&g_tosh_d2h_bytes, memory_order_relaxed)/1048576.0,
                atomic_load_explicit(&g_tosh_d2h_ns,    memory_order_relaxed)/1e6,
                (unsigned long long) hn,
                atomic_load_explicit(&g_tosh_h2d_bytes, memory_order_relaxed)/1048576.0,
                atomic_load_explicit(&g_tosh_h2d_ns,    memory_order_relaxed)/1e6);
        atomic_store_explicit(&g_tosh_d2h_n, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_d2h_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_d2h_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_n, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_bytes, 0, memory_order_relaxed);
        atomic_store_explicit(&g_tosh_h2d_ns, 0, memory_order_relaxed);
    }
    {
        const uint64_t n = atomic_load_explicit(&g_cbp_sync_n, memory_order_relaxed);
        fprintf(stderr, "TOSH_CBP  pid=%d  syncs=%llu  cpu_wait=%.1f  upload_drain=%.1f  "
                "encode_ops=%.1f  commit=%.1f  create=%.1f  (ms totales)\n",
                getpid(), (unsigned long long) n,
                atomic_load_explicit(&g_cbp_wait_ns,   memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_drain_ns,  memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_ops_ns,    memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_commit_ns, memory_order_relaxed)/1e6,
                atomic_load_explicit(&g_cbp_create_ns, memory_order_relaxed)/1e6);
        atomic_store_explicit(&g_cbp_wait_ns,   0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_drain_ns,  0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_sync_n,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_ops_ns,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_commit_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp_create_ns, 0, memory_order_relaxed);
    }
    fprintf(stderr, "TOSH_CBP  %8s %8s %8s %12s %12s %12s %12s %12s %12s\n",
            "n_nodes", "graphs", "cbs", "gpu_us/g", "gpu_us_min", "gpu_us_max", "kern_us/g", "span_us/g", "encode_us/g");
    for (int i = 0; i < TOSH_CBP_MAX_BUCKETS; ++i) {
        const int nn = atomic_load_explicit(&g_cbp[i].n_nodes, memory_order_relaxed);
        if (nn == 0) {
            continue;
        }
        const uint64_t ng = atomic_load_explicit(&g_cbp[i].n_graphs, memory_order_relaxed);
        if (ng == 0) {
            continue;
        }
        fprintf(stderr, "TOSH_CBP  %8d %8llu %8llu %12.1f %12.1f %12.1f %12.1f %12.1f %12.1f\n", nn,
                (unsigned long long) ng,
                (unsigned long long) atomic_load_explicit(&g_cbp[i].n_cbs, memory_order_relaxed),
                atomic_load_explicit(&g_cbp[i].gpu_ns,     memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].gpu_min_ns, memory_order_relaxed)/1000.0,
                atomic_load_explicit(&g_cbp[i].gpu_max_ns, memory_order_relaxed)/1000.0,
                atomic_load_explicit(&g_cbp[i].ker_ns,     memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].span_ns,    memory_order_relaxed)/1000.0/ng,
                atomic_load_explicit(&g_cbp[i].enc_ns,     memory_order_relaxed)/1000.0/ng);

        // reset so each dump covers only the graphs since the previous one
        atomic_store_explicit(&g_cbp[i].n_graphs,   0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].n_cbs,      0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].ker_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].span_ns,    0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].enc_ns,     0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_min_ns, 0, memory_order_relaxed);
        atomic_store_explicit(&g_cbp[i].gpu_max_ns, 0, memory_order_relaxed);
    }
    fflush(stderr);
}

struct ggml_metal_command_buffer {
    id<MTLCommandBuffer> obj;
};

struct ggml_metal {
    char name[128];

    ggml_metal_device_t  dev;
    ggml_metal_library_t lib;

    // shared device queue, or an owned one for secondary (prefetch) contexts
    // so their uploads overlap the primary context's compute
    id<MTLCommandQueue> queue;
    bool owns_queue;
    bool is_prefetch;

    // Bounded host-visible source ring for expert uploads. Keeping one no-copy
    // MTLBuffer per CPU tensor can pin many GiB of host pages on discrete GPUs.
    id<MTLBuffer>        prefetch_stages[GGML_METAL_MAX_PREFETCH_STAGES];
    id<MTLCommandBuffer> prefetch_stage_cmds[GGML_METAL_MAX_PREFETCH_STAGES];
    int prefetch_stage_count;
    int prefetch_stage_cur;

#ifdef TOSH_ENABLE_DYNAMIC_MOE
    // compact prefill uploads: bounded host ring, one command buffer per chunk
    id<MTLBuffer>        dmoe_ring;
    id<MTLBuffer>        dmoe_mark;
    id<MTLCommandBuffer> dmoe_ring_cmds[8];
    int                  dmoe_ring_cur;
#endif

    ggml_metal_event_t ev_cpy; // for async copies
    ggml_metal_event_t ev_sync; // destination completion signal

    id<MTLCounterSampleBuffer> ct_buf;   // trace only, TOSH_MGPU_CTIME
    id<MTLSharedEvent>         ev_wait_dummy;   // benchmark only, TOSH_MGPU_WAIT_SATISFIED
    _Atomic uint64_t           ct_idx;

    // Synchronization and resources shared by both cross-device copy paths.
    struct ggml_metal_xdev_link {
        ggml_metal_device_t  src_dev;
        void *               host;
        size_t               cap;
        id<MTLBuffer>        wrap_src, wrap_dst;
        // Retaining the source keeps its cached remote view valid.
        id<MTLBuffer>        peer_src, peer_view;
        // The collective caches its own view: it orders on seq_x, and an invalidation that
        // waited on the copy path's seq would let go of a view a queued exchange still reads.
        id<MTLBuffer>        peer_x_src, peer_x_view;
        // The partial this card publishes for its partner to read. Keeping it apart from the
        // tensor is what lets the add overwrite the tensor without waiting for the peer.
        // Two of them, alternating by round: with one, publishing round n has to wait for the
        // peer to finish reading round n-1, and that wait serialises the two cards every round.
        id<MTLBuffer>        shadow[2];
        id<MTLBuffer>        peer_x_view2[2];
        size_t               shadow_cap;
        int                  peer_x_ok; // -1 unknown, 0 ineligible, 1 eligible
        id<MTLSharedEvent>   ev_ready, ev_done;
        id<MTLSharedEvent>   ev_x_ready, ev_x_done;
        uint64_t             seq;
        uint64_t             seq_x;
        // per-slot back-pressure for the one-shot: the last seq whose publish landed in the
        // wrap block at that slot offset. Two slots let a card publish while a lagging
        // partner is still reading the other slot, the same one-round slack the butterfly
        // gets by touching each link only every other round.
        uint64_t             x_slot[2];
        bool                 peer_logged;
        bool                 peer_x_logged;
    } xlinks[4];
    int n_xlinks;

    dispatch_queue_t d_queue;

    // additional, inference-time compiled pipelines
    ggml_metal_pipelines_t pipelines_ext;

    bool use_concurrency;
    bool use_graph_optimize;

    int debug_graph;

    struct ggml_metal_fusion_info * finfo;

    // capture state
    int capture_compute;
    bool capture_started;

    id<MTLCaptureScope> capture_scope;

    // command buffer state
    int n_cb;           // number of extra threads used to submit the command buffers

    // first node of each extra command buffer, then the end of the graph
    int cb_start[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // the split of the current graph: GGML_METAL_DECODE_SCHED can fold it into the main
    // command buffer, so its extra buffer count can be below n_cb
    int n_cb_graph;
    int n_nodes_0;      // nodes encoded by the main thread, cb_start[0]
    int n_nodes_1;      // nodes encoded by the n_cb_graph extra command buffers

    // main thread's share sized from its measured encode and GPU time per node, per graph size
    bool     adapt_n_main;
    int      am_nodes[4];
    int      am_main[4];
    float    am_enc[4];   // seconds per node, moving averages
    float    am_gpu[4];
    int      am_prev;     // node count of the graph whose buffers are still held, 0 if unusable
    uint64_t am_enc_ns;

    struct ggml_cgraph * gf;

    // the callback given to the thread pool
    void (^encode_async)(size_t ith);

    // n_cb command buffers + 1 used by the main thread
    struct ggml_metal_command_buffer cmd_bufs[GGML_METAL_MAX_COMMAND_BUFFERS + 1];

    // extra command buffers for things like getting, setting and copying tensors
    NSMutableArray * cmd_bufs_ext;

    // buffers to release after async Metal operations complete
    // if Metal released them, it would do so on a Metal-internal thread without an autorelease pool, which could cause leaks
    NSMutableArray * buf_refs;

    // the last command buffer queued into the Metal queue with operations relevant to the current Metal backend
    id<MTLCommandBuffer> cmd_buf_last;

    // abort ggml_metal_graph_compute if callback returns true
    ggml_abort_callback abort_callback;
    void *              abort_callback_data;

    // error state - set when a command buffer fails during synchronize
    // once set, graph_compute will return GGML_STATUS_FAILED until the backend is recreated
    bool has_error;

    ggml_metal_event_t ev_wait_pending[GGML_METAL_MAX_PENDING_WAITS];
    int                n_ev_wait_pending;

    // a peer copy reads this context's memory from the other GPU's queue, so nothing
    // it submits next may overwrite the source until that read reports done
    struct ggml_metal_peer_wait {
        id<MTLSharedEvent> ev;
        uint64_t           val;
    } peer_waits[GGML_METAL_MAX_PENDING_WAITS];
    int n_peer_waits;

    // destination command buffer a cross-device copy left open, so the reduction that follows
    // it is encoded into the same one instead of opening another
    id<MTLCommandBuffer> cmd_buf_hold;
};

// TOSH_MGPU_DEFER_WAITS folds a cross-device wait into the next command buffer this context
// submits instead of spending a whole one on it. Opt-in: on a single card both contexts share
// the device queue, so it measures slightly negative there and only 2 real GPUs can settle it.
static bool ggml_metal_defer_waits(void) {
    static int val = -1;
    if (val < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_DEFER_WAITS");
        val = (v && v[0] == '1') ? 1 : 0;
    }
    return val;
}

static _Atomic uint64_t g_tr_cmd_all     = 0; // command buffers created through the tracked paths
static _Atomic uint64_t g_tr_graphs      = 0; // graph submissions, one per subgraph per card
static _Atomic uint64_t g_tr_merge_reuse = 0; // subgraphs that continued the collective's buffer
static _Atomic uint64_t g_tr_merge_waits = 0; // waits that had queued in between, and would have
                                              // been dropped before this was fixed
static _Atomic uint64_t g_tr_signals     = 0; // cross-device event signals encoded
static _Atomic uint64_t g_tr_waits       = 0; // cross-device event waits encoded


// TOSH_MGPU_TRACE: where the collective's time and bytes go. The counters are only touched
// when it is on, so a release run pays one cached branch per crossing.
static _Atomic uint64_t g_tr_reduce      = 0; // exchange_reduce calls
static _Atomic uint64_t g_tr_oneshot     = 0; // one-shot all-reduce collectives
static _Atomic uint64_t g_tr_copy        = 0; // plain cross-device copies
static _Atomic uint64_t g_tr_bytes_peer  = 0; // crossed through a remote buffer view
static _Atomic uint64_t g_tr_bytes_host  = 0; // crossed through the shared host block
static _Atomic uint64_t g_tr_cmd_bufs    = 0; // command buffers the collective created
static _Atomic uint64_t g_tr_rv_new      = 0; // remote views built
static _Atomic uint64_t g_tr_rv_hit      = 0; // remote views reused
static _Atomic uint64_t g_tr_encode_ns   = 0; // CPU time inside the collective
static _Atomic uint64_t g_tr_gpu_ns      = 0; // GPU execution of the collective's buffers
static _Atomic uint64_t g_tr_sched_ns    = 0; // committed to running: queueing plus event waits

static bool tosh_mgpu_trace(void) {
    static int on = -1;
    if (on < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_TRACE");
        on = (v && v[0] == '1') ? 1 : 0;
    }
    return on == 1;
}

static uint64_t tosh_now_ns(void) {
    return clock_gettime_nsec_np(CLOCK_MONOTONIC);
}

// Separates three things the old instrumentation conflated: when the host committed each
// side, when each GPU started the collective, and the signed order of both. The collective
// loop is serial on the CPU, so a GPU-start difference is not evidence of compute imbalance
// until the commit difference is subtracted from it.
#define TOSH_SKEW_SLOTS 8192
static struct {
    _Atomic uint64_t seq[2];
    _Atomic uint64_t commit_ns[2];
    _Atomic uint64_t gpu_ns[2];
} g_skew[TOSH_SKEW_SLOTS];

static _Atomic uint64_t g_sk_n;
static _Atomic uint64_t g_sk_gpu_hist[8], g_sk_com_hist[8];
static _Atomic uint64_t g_sk_gpu_sum, g_sk_com_sum, g_sk_gpu_max;
static _Atomic uint64_t g_sk_gpu_first0, g_sk_com_first0;   // how often peerIndex 0 leads

static int tosh_skew_bucket(double us) {
    return us < 1 ? 0 : us < 2 ? 1 : us < 5 ? 2 : us < 10 ? 3 :
           us < 20 ? 4 : us < 50 ? 5 : us < 200 ? 6 : 7;
}

static void tosh_skew_commit(uint64_t seq, int side, uint64_t t_ns) {
    const int slot = (int) (seq % TOSH_SKEW_SLOTS);
    atomic_store(&g_skew[slot].gpu_ns[side], 0);
    atomic_store(&g_skew[slot].commit_ns[side], t_ns);
    atomic_store(&g_skew[slot].seq[side], seq);
}

static void tosh_skew_started(uint64_t seq, int side, double gpu_start) {
    const int slot = (int) (seq % TOSH_SKEW_SLOTS);
    atomic_store(&g_skew[slot].gpu_ns[side], (uint64_t) (gpu_start*1e9));

    const int other = side ^ 1;
    if (atomic_load(&g_skew[slot].seq[other]) != seq) {
        return;
    }
    const uint64_t g0 = atomic_load(&g_skew[slot].gpu_ns[0]);
    const uint64_t g1 = atomic_load(&g_skew[slot].gpu_ns[1]);
    const uint64_t c0 = atomic_load(&g_skew[slot].commit_ns[0]);
    const uint64_t c1 = atomic_load(&g_skew[slot].commit_ns[1]);
    if (g0 == 0 || g1 == 0 || c0 == 0 || c1 == 0) {
        return;                                  // the other side has not started yet
    }
    // only the side that completes second gets here with both present
    if (side != 1 && atomic_load(&g_skew[slot].gpu_ns[other]) == 0) {
        return;
    }
    static _Atomic uint64_t claimed[TOSH_SKEW_SLOTS];
    uint64_t was = atomic_exchange(&claimed[slot], seq);
    if (was == seq) {
        return;                                  // already accounted
    }

    const uint64_t dg = g0 > g1 ? g0 - g1 : g1 - g0;
    const uint64_t dc = c0 > c1 ? c0 - c1 : c1 - c0;
    atomic_fetch_add(&g_sk_n, 1);
    atomic_fetch_add(&g_sk_gpu_sum, dg);
    atomic_fetch_add(&g_sk_com_sum, dc);
    if (g0 <= g1) { atomic_fetch_add(&g_sk_gpu_first0, 1); }
    if (c0 <= c1) { atomic_fetch_add(&g_sk_com_first0, 1); }
    uint64_t m = atomic_load(&g_sk_gpu_max);
    while (dg > m && !atomic_compare_exchange_weak(&g_sk_gpu_max, &m, dg)) { }
    atomic_fetch_add(&g_sk_gpu_hist[tosh_skew_bucket(dg/1000.0)], 1);
    atomic_fetch_add(&g_sk_com_hist[tosh_skew_bucket(dc/1000.0)], 1);
}

static void tosh_mgpu_trace_report(void);
static bool ggml_metal_collective_nowait(void);

// The subgraph command buffer is timed start and end separately, because a difference in
// completion can come from starting late or from running longer and those need different
// answers. Note what this buffer contains: ggml_metal_encode_pending_waits runs at its top,
// so its duration is graph compute plus any event wait encoded there, not pure compute.
static _Atomic uint64_t g_gr_ord[2];
static struct {
    _Atomic uint64_t ord[2];
    _Atomic uint64_t start_ns[2];
    _Atomic uint64_t end_ns[2];
} g_gr[TOSH_SKEW_SLOTS];

static _Atomic uint64_t g_gr_n;
static _Atomic uint64_t g_gr_hs[8], g_gr_hd[8], g_gr_he[8];      // |start|, |duration|, |end| deltas
static _Atomic uint64_t g_gr_ss, g_gr_sd, g_gr_se;               // their sums, ns
static _Atomic uint64_t g_gr_sign_start, g_gr_sign_dur;          // sign agreement with end delta
static _Atomic uint64_t g_gr_start0_first, g_gr_dur0_less;       // is either side systematically ahead
static _Atomic uint64_t g_gr_resid;                              // |end - (start + dur)| residual
static _Atomic uint64_t g_dur_n[2], g_dur_sum[2], g_dur_h[2][8]; // per-side duration

static int tosh_dur_bucket(double us) {
    return us <  40 ? 0 : us <  80 ? 1 : us < 120 ? 2 : us < 160 ? 3 :
           us < 200 ? 4 : us < 300 ? 5 : us < 500 ? 6 : 7;
}

static void tosh_trace_graph_end(id<MTLCommandBuffer> cmd_buf, int side) {
    if (!tosh_mgpu_trace()) {
        return;
    }
    static _Atomic int reg = 0;                              // a single GPU runs no collective
    int exp = 0;
    if (atomic_compare_exchange_strong(&reg, &exp, 1)) { atexit(tosh_mgpu_trace_report); }
    const uint64_t ord = atomic_fetch_add(&g_gr_ord[side], 1);
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.GPUStartTime <= 0 || cb.GPUEndTime <= 0) {
            return;
        }
        const int slot = (int) (ord % TOSH_SKEW_SLOTS);
        const uint64_t st = (uint64_t) (cb.GPUStartTime*1e9);
        const uint64_t en = (uint64_t) (cb.GPUEndTime*1e9);

        atomic_fetch_add(&g_dur_n[side], 1);
        atomic_fetch_add(&g_dur_sum[side], en - st);
        atomic_fetch_add(&g_dur_h[side][tosh_dur_bucket((en - st)/1000.0)], 1);

        atomic_store(&g_gr[slot].start_ns[side], st);
        atomic_store(&g_gr[slot].end_ns[side], en);
        atomic_store(&g_gr[slot].ord[side], ord + 1);
        if (atomic_load(&g_gr[slot].ord[side ^ 1]) != ord + 1) {
            return;
        }
        const int64_t s0 = (int64_t) atomic_load(&g_gr[slot].start_ns[0]);
        const int64_t s1 = (int64_t) atomic_load(&g_gr[slot].start_ns[1]);
        const int64_t e0 = (int64_t) atomic_load(&g_gr[slot].end_ns[0]);
        const int64_t e1 = (int64_t) atomic_load(&g_gr[slot].end_ns[1]);
        if (s0 == 0 || s1 == 0 || e0 == 0 || e1 == 0) {
            return;
        }
        static _Atomic uint64_t claimed[TOSH_SKEW_SLOTS];
        uint64_t was = atomic_exchange(&claimed[slot], ord + 1);
        if (was == ord + 1) {
            return;
        }

        const int64_t ds = s0 - s1;
        const int64_t dd = (e0 - s0) - (e1 - s1);
        const int64_t de = e0 - e1;
        const uint64_t as = ds < 0 ? -ds : ds, ad = dd < 0 ? -dd : dd, ae = de < 0 ? -de : de;
        const int64_t r = de - (ds + dd);

        atomic_fetch_add(&g_gr_n, 1);
        atomic_fetch_add(&g_gr_ss, as);
        atomic_fetch_add(&g_gr_sd, ad);
        atomic_fetch_add(&g_gr_se, ae);
        atomic_fetch_add(&g_gr_resid, (uint64_t) (r < 0 ? -r : r));
        if (ds <= 0) { atomic_fetch_add(&g_gr_start0_first, 1); }
        if (dd <= 0) { atomic_fetch_add(&g_gr_dur0_less, 1); }
        if ((ds >= 0) == (de >= 0)) { atomic_fetch_add(&g_gr_sign_start, 1); }
        if ((dd >= 0) == (de >= 0)) { atomic_fetch_add(&g_gr_sign_dur, 1); }
        atomic_fetch_add(&g_gr_hs[tosh_skew_bucket(as/1000.0)], 1);
        atomic_fetch_add(&g_gr_hd[tosh_skew_bucket(ad/1000.0)], 1);
        atomic_fetch_add(&g_gr_he[tosh_skew_bucket(ae/1000.0)], 1);
    }];
}

static void tosh_trace_exchange(id<MTLCommandBuffer> cmd_buf, uint64_t seq, int side) {
    if (!tosh_mgpu_trace()) {
        return;
    }
    tosh_skew_commit(seq, side, clock_gettime_nsec_np(CLOCK_MONOTONIC));
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        if (cb.GPUStartTime > 0) {
            tosh_skew_started(seq, side, cb.GPUStartTime);
        }
    }];
}

// Steady-state decode only. Every counter is snapshotted the first time a collective runs on a
// single-row batch, which is the first generated token, so nothing from load, warm-up or the
// prompt is amortised into the per-token figures.
extern _Atomic uint64_t * ggml_backend_meta_pass_counter(void);

static _Atomic int      g_snap_taken = 0;
static uint64_t g_snap_reduce, g_snap_cmd, g_snap_graphs, g_snap_sig, g_snap_wait,
                g_snap_peer_b, g_snap_host_b, g_snap_passes;

static void tosh_decode_snapshot(int64_t batch) {
    if (batch != 1 || atomic_load(&g_snap_taken)) {
        return;
    }
    int expected = 0;
    if (!atomic_compare_exchange_strong(&g_snap_taken, &expected, 1)) {
        return;
    }
    g_snap_reduce = atomic_load(&g_tr_reduce);
    g_snap_cmd    = atomic_load(&g_tr_cmd_all);
    g_snap_graphs = atomic_load(&g_tr_graphs);
    g_snap_sig    = atomic_load(&g_tr_signals);
    g_snap_wait   = atomic_load(&g_tr_waits);
    g_snap_peer_b = atomic_load(&g_tr_bytes_peer);
    g_snap_host_b = atomic_load(&g_tr_bytes_host);
    g_snap_passes = atomic_load(ggml_backend_meta_pass_counter());
}

// TOSH_MGPU_CTIME=1, trace only. GPU timestamps inside the staged exchange, the path batch-1
// decode actually takes. Four samples, all from one device, so no cross-GPU clock correlation
// is involved: S0/S1 bracket the publish blit, S2/S3 the staging read. S1 to S2 is a
// WAIT-WINDOW, not a pure event wait: it spans the signal, the peer wait and the encoder switch.
#define TOSH_CT_SLOTS 4096
#define TOSH_CT_PER   4

static bool ggml_metal_ctime(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_CTIME"); v = (s && s[0] == '1') ? 1 : 0; }
    return v == 1;
}

static bool ggml_metal_ctime_barrier(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_CTIME_BARRIER"); v = (s && s[0] == '1') ? 1 : 0; }
    return v == 1;
}

static _Atomic uint64_t g_tr_fused_bnd;
static _Atomic uint64_t g_tr_res_fold;
static _Atomic uint64_t g_ct_n, g_ct_pub, g_ct_wait, g_ct_read, g_ct_fail;
static _Atomic uint64_t g_ct_hw[8], g_ct_hp[8], g_ct_hr[8];
static double g_ct_ns_per_tick = 0.0;

static int tosh_ct_bucket(double us) {
    return us < 1 ? 0 : us < 5 ? 1 : us < 10 ? 2 : us < 20 ? 3 :
           us < 50 ? 4 : us < 100 ? 5 : us < 200 ? 6 : 7;
}

static id<MTLCounterSampleBuffer> tosh_ct_buffer(ggml_metal_t ctx) {
    if (!ggml_metal_ctime()) { return nil; }
    if (ctx->ct_buf != nil)  { return ctx->ct_buf; }
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> dev = ggml_metal_device_get_obj(ctx->dev);
        id<MTLCounterSet> ts = nil;
        for (id<MTLCounterSet> cs in dev.counterSets) {
            if ([cs.name isEqualToString:MTLCommonCounterSetTimestamp]) { ts = cs; break; }
        }
        if (ts == nil) { return nil; }
        MTLCounterSampleBufferDescriptor * d = [MTLCounterSampleBufferDescriptor new];
        d.counterSet = ts; d.sampleCount = TOSH_CT_SLOTS*TOSH_CT_PER; d.storageMode = MTLStorageModeShared;
        NSError * err = nil;
        ctx->ct_buf = [dev newCounterSampleBufferWithDescriptor:d error:&err];
        if (ctx->ct_buf == nil) {
            GGML_LOG_WARN("%s: counter sample buffer unavailable\n", __func__);
            return nil;
        }
        if (g_ct_ns_per_tick == 0.0) {
            g_ct_ns_per_tick = 1.0;
            MTLTimestamp c0 = 0, g0 = 0, c1 = 0, g1 = 0;
            [dev sampleTimestamps:&c0 gpuTimestamp:&g0];
            usleep(20000);
            [dev sampleTimestamps:&c1 gpuTimestamp:&g1];
            if (g1 > g0) { g_ct_ns_per_tick = (double) (c1 - c0)/(double) (g1 - g0); }
            GGML_LOG_INFO("%s: GPU timestamp sampling on, %.4f ns per tick\n", __func__, g_ct_ns_per_tick);
        }
    }
    return ctx->ct_buf;
}

static void tosh_ct_record(id<MTLCounterSampleBuffer> cs, uint64_t base) {
    if (@available(macOS 10.15, *)) {
        NSData * d = [cs resolveCounterRange:NSMakeRange(base, TOSH_CT_PER)];
        if (d == nil || d.length < TOSH_CT_PER*sizeof(MTLCounterResultTimestamp)) {
            atomic_fetch_add(&g_ct_fail, 1); return;
        }
        const MTLCounterResultTimestamp * t = (const MTLCounterResultTimestamp *) d.bytes;
        for (int i = 0; i < TOSH_CT_PER; i++) {
            if (t[i].timestamp == MTLCounterErrorValue) { atomic_fetch_add(&g_ct_fail, 1); return; }
        }
        const double k = g_ct_ns_per_tick;
        const double pub  = (double) (t[1].timestamp - t[0].timestamp)*k;
        const double wait = (double) (t[2].timestamp - t[1].timestamp)*k;
        const double read = (double) (t[3].timestamp - t[2].timestamp)*k;
        if (pub < 0 || wait < 0 || read < 0) { atomic_fetch_add(&g_ct_fail, 1); return; }
        atomic_fetch_add(&g_ct_n, 1);
        atomic_fetch_add(&g_ct_pub,  (uint64_t) pub);
        atomic_fetch_add(&g_ct_wait, (uint64_t) wait);
        atomic_fetch_add(&g_ct_read, (uint64_t) read);
        atomic_fetch_add(&g_ct_hp[tosh_ct_bucket(pub /1000.0)], 1);
        atomic_fetch_add(&g_ct_hw[tosh_ct_bucket(wait/1000.0)], 1);
        atomic_fetch_add(&g_ct_hr[tosh_ct_bucket(read/1000.0)], 1);
    }
}

// TOSH_MGPU_WAIT_SATISFIED=1, benchmark only. The staged collective keeps every encoder, every
// signal, the same command buffer and the same sampling points; the only change is that the
// peer-ready wait names a private event that is already signalled, so it can never stall. The
// read then observes whatever the partner left behind and the output is wrong by construction.
// Subtracting this run's central window from the normal one isolates the real peer stall
// without adding an encoder to the interval being measured.
// The same trick for the other wait in the path, the one guarding reuse of the shared host
// block. TOSH_MGPU_BP_SATISFIED=1, benchmark only, same warning applies.
static bool ggml_metal_bp_satisfied(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_BP_SATISFIED");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_BP_SATISFIED=1, the collective does not "
                            "wait before reusing the staging block. Benchmark only.\n");
        }
    }
    return v == 1;
}

static int64_t ggml_metal_peer_min_batch(void);

static bool ggml_metal_tp2_fused_boundary(int64_t batch) {
    static int v = -2;
    if (v == -2) {
        const char * s = TOSH_ENV("TOSH_MGPU_TP2_FUSED_BOUNDARY");
        v = s ? (s[0] == '1' ? 1 : 0) : -1;
    }
    if (v >= 0) {
        return v == 1;
    }
    return batch < ggml_metal_peer_min_batch();
}

static bool ggml_metal_wait_satisfied(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_WAIT_SATISFIED");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_WAIT_SATISFIED=1, the collective never "
                            "waits for the peer and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

static id<MTLSharedEvent> tosh_wait_dummy(ggml_metal_t ctx) {
    if (ctx->ev_wait_dummy == nil) {
        ctx->ev_wait_dummy = [ggml_metal_device_get_obj(ctx->dev) newSharedEvent];
        ctx->ev_wait_dummy.signaledValue = 1;   // satisfied before anything is encoded
    }
    return ctx->ev_wait_dummy;
}

static void tosh_mgpu_trace_report(void) {
    const uint64_t reduce = atomic_load(&g_tr_reduce);
    const uint64_t peer   = atomic_load(&g_tr_bytes_peer);
    const uint64_t host   = atomic_load(&g_tr_bytes_host);

    fprintf(stderr, "TOSH_MGPU_TRACE\n");
    fprintf(stderr, "  allreduce exchanges   %llu\n", (unsigned long long) reduce);
    fprintf(stderr, "  plain xdev copies     %llu\n", (unsigned long long) atomic_load(&g_tr_copy));
    fprintf(stderr, "  peer-direct bytes     %llu\n", (unsigned long long) peer);
    fprintf(stderr, "  host-staged bytes     %llu\n", (unsigned long long) host);
    fprintf(stderr, "  peer share            %.1f%%\n",
            peer + host ? 100.0*(double) peer/(double)(peer + host) : 0.0);
    fprintf(stderr, "  collective cmd bufs   %llu\n", (unsigned long long) atomic_load(&g_tr_cmd_bufs));
    fprintf(stderr, "  tracked cmd bufs      %llu\n", (unsigned long long) atomic_load(&g_tr_cmd_all));
    fprintf(stderr, "  merge reuse / waits   %llu/%llu\n",
            (unsigned long long) atomic_load(&g_tr_merge_reuse),
            (unsigned long long) atomic_load(&g_tr_merge_waits));
    fprintf(stderr, "  graph submissions     %llu\n", (unsigned long long) atomic_load(&g_tr_graphs));
    {
        const uint64_t n = atomic_load(&g_sk_n);
        if (n > 0) {
            static const char * lbl[8] = {"<1","<2","<5","<10","<20","<50","<200",">=200"};
            const double nn = (double) n;
            fprintf(stderr, "  skew n=%llu   commit mean %.1f us   GPU-start mean %.1f us   GPU max %.1f us\n",
                    (unsigned long long) n, atomic_load(&g_sk_com_sum)/1000.0/nn,
                    atomic_load(&g_sk_gpu_sum)/1000.0/nn, atomic_load(&g_sk_gpu_max)/1000.0);
            fprintf(stderr, "    side0 leads: commit %.1f%%   GPU %.1f%%\n",
                    100.0*atomic_load(&g_sk_com_first0)/nn, 100.0*atomic_load(&g_sk_gpu_first0)/nn);
            fprintf(stderr, "    commit us   ");
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&g_sk_com_hist[i])/nn); }
            fprintf(stderr, "\n    GPUstart us ");
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&g_sk_gpu_hist[i])/nn); }
            fprintf(stderr, "\n");
        }
    }
    {
        const uint64_t n = atomic_load(&g_gr_n);
        if (n > 0) {
            static const char * lbl[8] = {"<1","<2","<5","<10","<20","<50","<200",">=200"};
            const double nn = (double) n;
            fprintf(stderr, "  subgraph deltas n=%llu  |start| %.1f us  |duration| %.1f us  |end| %.1f us  residual %.3f us\n",
                    (unsigned long long) n, atomic_load(&g_gr_ss)/1000.0/nn,
                    atomic_load(&g_gr_sd)/1000.0/nn, atomic_load(&g_gr_se)/1000.0/nn,
                    atomic_load(&g_gr_resid)/1000.0/nn);
            fprintf(stderr, "    sign of end delta agrees with start %.1f%%, with duration %.1f%%\n",
                    100.0*atomic_load(&g_gr_sign_start)/nn, 100.0*atomic_load(&g_gr_sign_dur)/nn);
            fprintf(stderr, "    side0 starts first %.1f%%   side0 shorter %.1f%%\n",
                    100.0*atomic_load(&g_gr_start0_first)/nn, 100.0*atomic_load(&g_gr_dur0_less)/nn);
            const char * nm[3] = {"|start|  ", "|duration|", "|end|    "};
            _Atomic uint64_t * hh[3] = {g_gr_hs, g_gr_hd, g_gr_he};
            for (int k = 0; k < 3; k++) {
                fprintf(stderr, "    %s us ", nm[k]);
                for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&hh[k][i])/nn); }
                fprintf(stderr, "\n");
            }
        }
        static const char * dlb2[8] = {"<40","<80","<120","<160","<200","<300","<500",">=500"};
        for (int sd = 0; sd < 2; sd++) {
            const uint64_t dn = atomic_load(&g_dur_n[sd]);
            if (!dn) continue;
            fprintf(stderr, "  side%d subgraph duration mean %.1f us  n=%llu  ", sd,
                    atomic_load(&g_dur_sum[sd])/1000.0/(double) dn, (unsigned long long) dn);
            for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", dlb2[i], 100.0*atomic_load(&g_dur_h[sd][i])/(double) dn); }
            fprintf(stderr, "\n");
        }
    }
    {
        const uint64_t n = atomic_load(&g_ct_n);
        if (n > 0 || atomic_load(&g_ct_fail) > 0) {
            static const char * lbl[8] = {"<1","<5","<10","<20","<50","<100","<200",">=200"};
            const double nn = (double) (n ? n : 1);
            fprintf(stderr, "  staged timeline n=%llu (failed %llu)  publish %.1f us  WAIT-WINDOW %.1f us  read %.1f us\n",
                    (unsigned long long) n, (unsigned long long) atomic_load(&g_ct_fail),
                    atomic_load(&g_ct_pub)/1000.0/nn, atomic_load(&g_ct_wait)/1000.0/nn,
                    atomic_load(&g_ct_read)/1000.0/nn);
            const char * rn[3] = {"publish    ", "wait-window", "read       "};
            _Atomic uint64_t * hh[3] = {g_ct_hp, g_ct_hw, g_ct_hr};
            for (int k = 0; k < 3; k++) {
                fprintf(stderr, "    %s us ", rn[k]);
                for (int i = 0; i < 8; i++) { fprintf(stderr, "%s %.1f%%  ", lbl[i], 100.0*atomic_load(&hh[k][i])/nn); }
                fprintf(stderr, "\n");
            }
        }
    }
    fprintf(stderr, "  fused boundaries      %llu\n", (unsigned long long) atomic_load(&g_tr_fused_bnd));
    fprintf(stderr, "  residual folds        %llu\n", (unsigned long long) atomic_load(&g_tr_res_fold));
    fprintf(stderr, "  oneshot collectives   %llu\n", (unsigned long long) atomic_load(&g_tr_oneshot));
    fprintf(stderr, "  remote views new/hit  %llu/%llu\n",
            (unsigned long long) atomic_load(&g_tr_rv_new),
            (unsigned long long) atomic_load(&g_tr_rv_hit));
    fprintf(stderr, "  encode CPU            %.1f ms\n", atomic_load(&g_tr_encode_ns)/1e6);
    fprintf(stderr, "  collective GPU        %.1f ms\n", atomic_load(&g_tr_gpu_ns)/1e6);
    fprintf(stderr, "  commit to GPU start   %.1f ms\n", atomic_load(&g_tr_sched_ns)/1e6);
    // decode-only, per card per token
    if (atomic_load(&g_snap_taken)) {
        const double t = (double) (atomic_load(ggml_backend_meta_pass_counter()) - g_snap_passes);
        if (t > 0) {
            fprintf(stderr, "  --- steady-state decode, per card per token (%.0f passes) ---\n", t);
            fprintf(stderr, "    collectives         %.2f\n", (atomic_load(&g_tr_reduce) - g_snap_reduce)/t);
            fprintf(stderr, "    tracked cmd bufs    %.2f\n", (atomic_load(&g_tr_cmd_all) - g_snap_cmd)/t);
            fprintf(stderr, "    graph submissions   %.2f\n", (atomic_load(&g_tr_graphs) - g_snap_graphs)/t);
            fprintf(stderr, "    event signals       %.2f\n", (atomic_load(&g_tr_signals) - g_snap_sig)/t);
            fprintf(stderr, "    event waits         %.2f\n", (atomic_load(&g_tr_waits) - g_snap_wait)/t);
            fprintf(stderr, "    peer bytes          %.0f\n", (atomic_load(&g_tr_bytes_peer) - g_snap_peer_b)/t);
            fprintf(stderr, "    host-staged bytes   %.0f\n", (atomic_load(&g_tr_bytes_host) - g_snap_host_b)/t);
        }
    }

    if (reduce) {
        fprintf(stderr, "  per exchange          %.1f us GPU, %.1f us to start, %llu B\n",
                atomic_load(&g_tr_gpu_ns)/1e3/(double) reduce,
                atomic_load(&g_tr_sched_ns)/1e3/(double) reduce,
                (unsigned long long) ((peer + host)/reduce));
    }
}

// Metal reports both when the buffer was handed over and when the GPU actually ran it; the
// gap is the queueing and the event waits, which is the number the collective is judged on.
static void ggml_metal_encode_pending_waits(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    for (int i = 0; i < ctx->n_ev_wait_pending; ++i) {
        ggml_metal_event_encode_wait(ctx->ev_wait_pending[i], (ggml_metal_cmd_buf_t) cmd_buf);
    }
    ctx->n_ev_wait_pending = 0;

    for (int i = 0; i < ctx->n_peer_waits; ++i) {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:ctx->peer_waits[i].ev value:ctx->peer_waits[i].val];
    }
    ctx->n_peer_waits = 0;
}

static void ggml_metal_cmd_buf_submit(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    [cmd_buf commit];
    [ctx->cmd_bufs_ext addObject:cmd_buf];
    ctx->cmd_buf_last = cmd_buf;
    [cmd_buf retain];
}


static void ggml_metal_cmd_buf_hold_flush(ggml_metal_t ctx) {
    if (ctx->cmd_buf_hold == nil) {
        return;
    }

    id<MTLCommandBuffer> cmd_buf = ctx->cmd_buf_hold;
    ctx->cmd_buf_hold = nil;
    ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    [cmd_buf release];
}

static id<MTLCommandBuffer> ggml_metal_cmd_buf_new(ggml_metal_t ctx) {
    if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
    id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
    ggml_metal_encode_pending_waits(ctx, cmd_buf);

    return cmd_buf;
}

// Continues the buffer a copy left open, if there is one. Only the reduction that follows the
// copy may do this: a buffer that stays open is a buffer that has not started running.
static id<MTLCommandBuffer> ggml_metal_cmd_buf_continue(ggml_metal_t ctx) {
    if (ctx->cmd_buf_hold != nil) {
        id<MTLCommandBuffer> cmd_buf = ctx->cmd_buf_hold;
        ctx->cmd_buf_hold = nil;
        return [cmd_buf autorelease];
    }

    return ggml_metal_cmd_buf_new(ctx);
}

static void ggml_metal_cmd_buf_end(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf, bool hold) {
    if (hold) {
        ctx->cmd_buf_hold = [cmd_buf retain];
    } else {
        ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    }
}

// folded into the next command buffer this context submits, so the back-edge costs
// no command buffer of its own
static void ggml_metal_peer_wait_add(ggml_metal_t ctx, id<MTLSharedEvent> ev, uint64_t val) {
    for (int i = 0; i < ctx->n_peer_waits; ++i) {
        if (ctx->peer_waits[i].ev == ev) {
            ctx->peer_waits[i].val = MAX(ctx->peer_waits[i].val, val);
            return;
        }
    }

    if (ctx->n_peer_waits < GGML_METAL_MAX_PENDING_WAITS) {
        ctx->peer_waits[ctx->n_peer_waits].ev  = ev;
        ctx->peer_waits[ctx->n_peer_waits].val = val;
        ctx->n_peer_waits++;
        return;
    }

    @autoreleasepool {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:ev value:val];
        ggml_metal_cmd_buf_submit(ctx, cmd_buf);
    }
}

ggml_metal_t ggml_metal_init(ggml_metal_device_t dev, bool prefetch) {
    GGML_LOG_INFO("%s: allocating\n", __func__);

    @autoreleasepool {
#if TARGET_OS_OSX && !GGML_METAL_NDEBUG
        // Show all the Metal device instances in the system
        NSArray * devices = MTLCopyAllDevices();
        for (id<MTLDevice> device in devices) {
            GGML_LOG_INFO("%s: found device: %s\n", __func__, [[device name] UTF8String]);
        }
        [devices release]; // since it was created by a *Copy* C method
#endif

        // init context
        ggml_metal_t res = calloc(1, sizeof(struct ggml_metal));

        id<MTLDevice> device = ggml_metal_device_get_obj(dev);

        GGML_LOG_INFO("%s: picking default device: %s\n", __func__, [[device name] UTF8String]);

        // TODO: would it be better to have one queue for the backend and one queue for the device?
        //       the graph encoders and async ops would use the backend queue while the sync ops would use the device queue?
        //res->queue = [device newCommandQueue]; [TAG_QUEUE_PER_BACKEND]
        res->queue = prefetch ? ggml_metal_device_acquire_queue(dev, &res->owns_queue)
                              : ggml_metal_device_get_queue(dev);
        if (res->queue == nil) {
            GGML_LOG_ERROR("%s: error: failed to create command queue\n", __func__);
            free(res);
            return NULL;
        }
        res->is_prefetch = prefetch;
        if (prefetch) {
            const char * val = TOSH_ENV("GGML_METAL_PREFETCH_STAGE_COUNT");
            const int requested = val ? atoi(val) : 1;
            res->prefetch_stage_count = MAX(1, MIN(requested, GGML_METAL_MAX_PREFETCH_STAGES));
        }

        res->dev = dev;
        res->lib = ggml_metal_device_get_library(dev);
        if (res->lib == NULL) {
            GGML_LOG_WARN("%s: the device does not have a precompiled Metal library - this is unexpected\n", __func__);
            GGML_LOG_WARN("%s: will try to compile it on the fly\n", __func__);

            res->lib = ggml_metal_library_init(dev);
            if (res->lib == NULL) {
                GGML_LOG_ERROR("%s: error: failed to initialize the Metal library\n", __func__);

                free(res);

                return NULL;
            }
        }

        res->ev_cpy  = ggml_metal_device_event_init(dev);
        res->ev_sync = ggml_metal_device_event_init(dev);

        const struct ggml_metal_device_props * props_dev = ggml_metal_device_get_props(dev);

        snprintf(res->name, sizeof(res->name), "%s", props_dev->name);

        res->d_queue = dispatch_queue_create("ggml-metal", DISPATCH_QUEUE_CONCURRENT);

        // discrete GPUs (AMD) corrupt their output with a concurrent encoder: the driver does
        // not honour the memory barrier. GGML_METAL_CONCURRENCY_ENABLE (or GGML_METAL_CONCURRENCY_FORCE)
        // forces it back on to test.
        res->use_concurrency = props_dev->has_unified_memory
            ? TOSH_ENV("GGML_METAL_CONCURRENCY_DISABLE") == nil
            : (TOSH_ENV("GGML_METAL_CONCURRENCY_ENABLE") != nil || TOSH_ENV("GGML_METAL_CONCURRENCY_FORCE") != nil);

        res->adapt_n_main = !props_dev->has_unified_memory && TOSH_ENV("TOSH_N_MAIN_FIXED") == nil;

        {
            const char * val = TOSH_ENV("GGML_METAL_GRAPH_DEBUG");
            res->debug_graph = val ? atoi(val) : 0;
        }

        res->use_graph_optimize = true;

        if (TOSH_ENV("GGML_METAL_GRAPH_OPTIMIZE_DISABLE") != NULL) {
            res->use_graph_optimize = false;
        }

        res->finfo = ggml_metal_device_get_fusion_info(dev);
        if (ggml_metal_fusion_info_stats(res->finfo)) {
            ggml_metal_fusion_info_labels_init(res->finfo);
            res->n_cb = 0;
        }

        GGML_LOG_INFO("%s: use fusion         = %s\n", __func__, ggml_metal_fusion_info_enabled(res->finfo) ? "true" : "false");
        GGML_LOG_INFO("%s: use concurrency    = %s\n", __func__, res->use_concurrency    ? "true" : "false");
        GGML_LOG_INFO("%s: use graph optimize = %s\n", __func__, res->use_graph_optimize ? "true" : "false");

        res->capture_compute = 0;
        res->capture_started = false;
        res->capture_scope = nil;

        {
            const char * val = TOSH_ENV("GGML_METAL_CAPTURE_COMPUTE");
            if (val) {
                res->capture_compute = atoi(val);
            }
        }

        res->has_error = false;

        res->gf = nil;
        res->encode_async = nil;
        for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
            res->cmd_bufs[i].obj = nil;
        }

        res->cmd_bufs_ext = [[NSMutableArray alloc] init];
        res->buf_refs     = [[NSMutableArray alloc] init];

        res->cmd_buf_last = nil;

        res->pipelines_ext = ggml_metal_pipelines_init();

        return res;
    }
}

void ggml_metal_free(ggml_metal_t ctx) {
    tosh_cbp_dump_dev(ctx->dev);

    GGML_LOG_INFO("%s: deallocating\n", __func__);

    ggml_metal_cmd_buf_hold_flush(ctx);

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] waitUntilCompleted];
            [ctx->prefetch_stage_cmds[i] release];
        }
        [ctx->prefetch_stages[i] release];
    }

#ifdef TOSH_ENABLE_DYNAMIC_MOE
    for (int i = 0; i < 8; ++i) {
        if (ctx->dmoe_ring_cmds[i]) {
            [ctx->dmoe_ring_cmds[i] waitUntilCompleted];
            [ctx->dmoe_ring_cmds[i] release];
        }
    }
    [ctx->dmoe_ring release];
    [ctx->dmoe_mark release];
#endif

    for (int i = 0; i < ctx->n_xlinks; ++i) {
        if (ctx->xlinks[i].seq > 0) {
            [ctx->xlinks[i].ev_done waitUntilSignaledValue:ctx->xlinks[i].seq timeoutMS:10000];
        }
        if (ctx->xlinks[i].seq_x > 0) {
            [ctx->xlinks[i].ev_x_done waitUntilSignaledValue:ctx->xlinks[i].seq_x timeoutMS:10000];
        }
        [ctx->xlinks[i].wrap_src release];
        [ctx->xlinks[i].wrap_dst release];
        [ctx->xlinks[i].peer_view release];
        [ctx->xlinks[i].peer_src release];
        [ctx->xlinks[i].peer_x_view release];
        [ctx->xlinks[i].peer_x_src release];
        for (int k = 0; k < 2; k++) {
            [ctx->xlinks[i].shadow[k] release];
            [ctx->xlinks[i].peer_x_view2[k] release];
        }
        [ctx->xlinks[i].ev_ready release];
        [ctx->xlinks[i].ev_done release];
        [ctx->xlinks[i].ev_x_ready release];
        [ctx->xlinks[i].ev_x_done release];
        free(ctx->xlinks[i].host);
    }

    for (int i = 0; i < GGML_METAL_MAX_COMMAND_BUFFERS; ++i) {
        if (ctx->cmd_bufs[i].obj) {
            [ctx->cmd_bufs[i].obj release];
        }
    }

    for (int i = 0; i < (int) ctx->cmd_bufs_ext.count; ++i) {
        if (ctx->cmd_bufs_ext[i]) {
            [ctx->cmd_bufs_ext[i] release];
        }
    }

    [ctx->cmd_bufs_ext removeAllObjects];
    [ctx->cmd_bufs_ext release];

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
        [ctx->buf_refs release];
    }

    if (ctx->pipelines_ext) {
        ggml_metal_pipelines_free(ctx->pipelines_ext);
        ctx->pipelines_ext = nil;
    }

    if (ggml_metal_fusion_info_debug(ctx->finfo) > 0) {
        GGML_LOG_DEBUG("%s: fusion stats:\n", __func__);

        const int n_fusions = ggml_metal_fusion_info_n_fusions(ctx->finfo);
        for (int i = 0; i < n_fusions; i++) {
            const uint64_t count = ggml_metal_fusion_info_count(ctx->finfo, i);
            if (count == 0) {
                continue;
            }

            // note: cannot use ggml_log here
            GGML_LOG_DEBUG("%s: - %s: %" PRIu64 "\n", __func__, ggml_metal_fusion_info_label(ctx->finfo, i), count);
        }
    }

    Block_release(ctx->encode_async);

    ggml_metal_device_release_queue(ctx->dev, ctx->queue, ctx->owns_queue);

    dispatch_release(ctx->d_queue);

    ggml_metal_device_event_free(ctx->dev, ctx->ev_cpy);
    ggml_metal_device_event_free(ctx->dev, ctx->ev_sync);

    free(ctx);
}

const char * ggml_metal_get_name(ggml_metal_t ctx) {
    return ctx->name;
}

uint64_t ggml_metal_peer_group_id(ggml_metal_t ctx) {
#if TARGET_OS_OSX
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
        return device.peerGroupID;
    }
#else
    GGML_UNUSED(ctx);
#endif
    return 0;
}

void ggml_metal_synchronize(ggml_metal_t ctx) {
    // nothing completes while a command buffer is still held open
    ggml_metal_cmd_buf_hold_flush(ctx);

    // pending waits guard memory a peer blit can still be reading, so they must run and complete before synchronize returns
    if (ctx->n_ev_wait_pending > 0 || ctx->n_peer_waits > 0) {
        @autoreleasepool {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            ggml_metal_cmd_buf_submit(ctx, cmd_buf);
        }
    }

    // wait for any backend operations to finish
    const bool cbp_on = tosh_cbp_on();
    uint64_t cbp_t = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
    if (ctx->cmd_buf_last) {
        [ctx->cmd_buf_last waitUntilCompleted];
        ctx->cmd_buf_last = nil;
    }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    if (tosh_dmoe_enabled()) {
        // a wide batch synchronizes mid-graph to read its routing; its join still needs the host
        if (tosh_dmoe_event_get() >= tosh_dmoe_last_value() + 1) {
            tosh_dmoe_set_active(false);
        }
        tosh_dmoe_mark_completed();
    }
    if (tosh_dmoe_failed() != NULL && !ctx->has_error) {
        GGML_LOG_ERROR("%s: mixed expert execution failed: %s\n", __func__, tosh_dmoe_failed());
        ctx->has_error = true;
    }
#endif
    if (cbp_on) {
        const uint64_t t1 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
        atomic_fetch_add_explicit(&g_cbp_wait_ns, t1 - cbp_t, memory_order_relaxed);
        atomic_fetch_add_explicit(&g_cbp_sync_n, 1, memory_order_relaxed);
        cbp_t = t1;
    }

    // in-flight staged uploads count as pending backend operations
    ggml_metal_device_upload_drain(ctx->dev);
    if (cbp_on) {
        atomic_fetch_add_explicit(&g_cbp_drain_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_t, memory_order_relaxed);
    }

    // check status of all command buffers
    {
        const int n_cb = ctx->n_cb;

        for (int cb_idx = 0; cb_idx <= n_cb; ++cb_idx) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;
            if (!cmd_buf) {
                continue;
            }

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, cb_idx, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    NSError * cb_err = [cmd_buf error];
                    GGML_LOG_ERROR("%s: Metal CB error: status=%d domain=%s code=%ld desc=%s\n", __func__, (int) status,
                        [[cb_err domain] UTF8String], (long) [cb_err code], [[cb_err localizedDescription] UTF8String]);
                }
                ctx->has_error = true;
                return;
            }
        }
    }

    // release any completed extra command buffers
    if (ctx->cmd_bufs_ext.count > 0) {
        for (size_t i = 0; i < ctx->cmd_bufs_ext.count; ++i) {
            id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs_ext[i];

            MTLCommandBufferStatus status = [cmd_buf status];
            if (status != MTLCommandBufferStatusCompleted) {
                GGML_LOG_ERROR("%s: error: command buffer %d failed with status %d\n", __func__, (int) i, (int) status);
                if (status == MTLCommandBufferStatusError) {
                    NSError * cb_err = [cmd_buf error];
                    GGML_LOG_ERROR("%s: Metal CB error: status=%d domain=%s code=%ld desc=%s\n", __func__, (int) status,
                        [[cb_err domain] UTF8String], (long) [cb_err code], [[cb_err localizedDescription] UTF8String]);
                }

                // release this and all remaining command buffers before returning
                for (size_t j = i; j < ctx->cmd_bufs_ext.count; ++j) {
                    [ctx->cmd_bufs_ext[j] release];
                }
                [ctx->cmd_bufs_ext removeAllObjects];

                ctx->has_error = true;
                return;
            }

            [cmd_buf release];
        }

        [ctx->cmd_bufs_ext removeAllObjects];
    }

    @autoreleasepool {
        [ctx->buf_refs removeAllObjects];
    }

    for (int i = 0; i < ctx->prefetch_stage_count; ++i) {
        if (ctx->prefetch_stage_cmds[i]) {
            [ctx->prefetch_stage_cmds[i] release];
            ctx->prefetch_stage_cmds[i] = nil;
        }
    }
}

static struct ggml_metal_buffer_id ggml_metal_get_buffer_id(const struct ggml_tensor * t) {
    if (!t) {
        return (struct ggml_metal_buffer_id) { nil, 0 };
    }

    ggml_backend_buffer_t buffer = t->view_src ? t->view_src->buffer : t->buffer;

    return ggml_metal_buffer_get_id(buffer->context, t);
}

void ggml_metal_read_batch_begin(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_begin(ctx->dev);
}

void ggml_metal_read_batch_end(ggml_metal_t ctx) {
    ggml_metal_device_read_batch_end(ctx->dev);
}

#ifdef TOSH_ENABLE_DYNAMIC_MOE
static _Atomic uint64_t g_dmoe_pf_host_bytes;
static _Atomic uint64_t g_dmoe_pf_vram_bytes;
static _Atomic uint64_t g_dmoe_pf_skip_bytes;
static _Atomic uint64_t g_dmoe_pf_rare_bytes[3];
static _Atomic uint64_t g_dmoe_pf_uploads;
static _Atomic uint64_t g_dmoe_pf_used;
static _Atomic uint64_t g_dmoe_pf_used_res;

static void ggml_metal_dmoe_pf_report(void) {
    const uint64_t h = atomic_load(&g_dmoe_pf_host_bytes), v = atomic_load(&g_dmoe_pf_vram_bytes);
    if (h + v > 0 && getenv("TOSH_DMOE_STATS")) {
        const double G = 1073741824.0;
        fprintf(stderr, "tosh_dmoe: prefill bank uploads %.2f GiB from host, %.2f GiB from the VRAM arena, %.2f GiB not routed and skipped\n",
                h/G, v/G, atomic_load(&g_dmoe_pf_skip_bytes)/G);
        const uint64_t n = atomic_load(&g_dmoe_pf_uploads);
        if (n > 0) {
            fprintf(stderr, "tosh_dmoe: compact uploads %llu, routed experts per upload %.1f (%.1f resident) | host bytes of experts with 1 / <=2 / <=4 rows %.2f / %.2f / %.2f GiB\n",
                    (unsigned long long) n, (double) atomic_load(&g_dmoe_pf_used)/n, (double) atomic_load(&g_dmoe_pf_used_res)/n,
                    atomic_load(&g_dmoe_pf_rare_bytes[0])/G, atomic_load(&g_dmoe_pf_rare_bytes[1])/G, atomic_load(&g_dmoe_pf_rare_bytes[2])/G);
        }
    }
}

#define DMOE_RING_CHUNKS 8
#define DMOE_RING_CHUNK  ((size_t) 16 << 20)

struct dmoe_piece { size_t src, dst, len; const char * host; };   // host: a cached expert, else data + src

static void ggml_metal_dmoe_submit(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf, int il) {
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        tosh_dmoe_note_time(2, (int64_t) ((cb.GPUEndTime - cb.GPUStartTime)*1e6));
        tosh_dmoe_timeline(2, il, cb.GPUStartTime, cb.GPUEndTime);
    }];
    [cmd_buf commit];
    [ctx->cmd_bufs_ext addObject:cmd_buf];
    ctx->cmd_buf_last = cmd_buf;
    [cmd_buf retain];
}

// Copies the pieces into their ring chunk on several threads and blits them to the slot.
static void ggml_metal_dmoe_ring_flush(ggml_metal_t ctx, const void * data, id<MTLBuffer> dst, size_t dst_offs,
        const struct dmoe_piece * p, int n, bool * first, int il) {
    if (n == 0) {
        return;
    }
    const int c = ctx->dmoe_ring_cur;
    ctx->dmoe_ring_cur = (c + 1) % DMOE_RING_CHUNKS;
    const double tl_w = tosh_dmoe_host_seconds();
    if (ctx->dmoe_ring_cmds[c]) {
        [ctx->dmoe_ring_cmds[c] waitUntilCompleted];
        [ctx->dmoe_ring_cmds[c] release];
        ctx->dmoe_ring_cmds[c] = nil;
    }
    const double tl_c = tosh_dmoe_host_seconds();
    tosh_dmoe_timeline(7, il, tl_w, tl_c);
    char * ring = (char *) ctx->dmoe_ring.contents + (size_t) c*DMOE_RING_CHUNK;
    const size_t sub = (size_t) 1 << 20;
    size_t n_sub = 0;
    for (int i = 0; i < n; ++i) n_sub += (p[i].len + sub - 1)/sub;
    dispatch_apply(n_sub, DISPATCH_APPLY_AUTO, ^(size_t k) {
        size_t kk = k;
        for (int i = 0; i < n; ++i) {
            const size_t m = (p[i].len + sub - 1)/sub;
            if (kk < m) {
                const size_t o = kk*sub, l = MIN(sub, p[i].len - o);
                memcpy(ring + p[i].dst + o, (p[i].host ? p[i].host : (const char *) data + p[i].src) + o, l);
                return;
            }
            kk -= m;
        }
    });
    tosh_dmoe_timeline(5, il, tl_c, tosh_dmoe_host_seconds());
    id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
    if (*first) {
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        *first = false;
    }
    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
    for (int i = 0; i < n; ++i) {
        [encoder copyFromBuffer:ctx->dmoe_ring sourceOffset:(size_t) c*DMOE_RING_CHUNK + p[i].dst
                       toBuffer:dst destinationOffset:dst_offs + p[i].src size:p[i].len];
    }
    [encoder endEncoding];
    ggml_metal_dmoe_submit(ctx, cmd_buf, il);
    ctx->dmoe_ring_cmds[c] = [cmd_buf retain];
}

// Only the experts the chunk routes to: residents device to device, the rest from host memory
// through the ring. Promotions copy from the filled slot once its blits are done.
static bool ggml_metal_dmoe_compact_upload(ggml_metal_t ctx, const struct tosh_dmoe_bank_plan * plan,
        struct ggml_metal_buffer_id bid_dst, struct ggml_metal_buffer_id bid_arena, const void * data) {
    if (ctx->dmoe_ring == nil) {
        ctx->dmoe_ring = [ggml_metal_device_get_obj(ctx->dev) newBufferWithLength:DMOE_RING_CHUNKS*DMOE_RING_CHUNK
                                                                         options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
        if (ctx->dmoe_ring == nil) {
            return false;
        }
    }
    const size_t eb = plan->expert_bytes;
    const int32_t * rows = plan->rows;
    bool first = true;
    uint64_t host_bytes = 0, vram_bytes = 0, skip_bytes = 0, rare[3] = { 0, 0, 0 }, used = 0, used_res = 0;

    // residents first: they need no host work and can run while the ring fills
    {
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        first = false;
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        for (int e = 0; e < plan->n_expert; ++e) {
            if (rows[e] == 0) {
                skip_bytes += eb;
                continue;
            }
            used++;
            if (plan->slot_of[e] >= 0) {
                [encoder copyFromBuffer:bid_arena.metal sourceOffset:bid_arena.offs + (size_t) plan->slot_of[e]*eb
                               toBuffer:bid_dst.metal destinationOffset:bid_dst.offs + (size_t) e*eb size:eb];
                vram_bytes += eb;
                used_res++;
            } else {
                if (rows[e] <= 1) rare[0] += eb;
                if (rows[e] <= 2) rare[1] += eb;
                if (rows[e] <= 4) rare[2] += eb;
            }
        }
        [encoder endEncoding];
        ggml_metal_dmoe_submit(ctx, cmd_buf, plan->il);
    }

    struct dmoe_piece p[64];
    int n = 0;
    size_t fill = 0;
    if (tosh_hostcache_on()) {
        // experimental bounded host cache: pin the chunk's missing experts (their reads overlap),
        // copy each from its cached part, then let them go
        int32_t need[512];
        int n_need = 0;
        for (int e = 0; e < plan->n_expert && n_need < 512; ++e) {
            if (rows[e] > 0 && plan->slot_of[e] < 0) need[n_need++] = e;
        }
        if (!tosh_hostcache_acquire(plan->il, need, n_need, TOSH_HC_PREFILL)) {
            tosh_dmoe_fail(tosh_hostcache_error());
            return false;
        }
        for (int i = 0; i < n_need; ++i) {
            if (fill + eb > DMOE_RING_CHUNK || n == 64) {
                ggml_metal_dmoe_ring_flush(ctx, data, bid_dst.metal, bid_dst.offs, p, n, &first, plan->il);
                n = 0;
                fill = 0;
            }
            p[n++] = (struct dmoe_piece) { (size_t) need[i]*eb, fill, eb, tosh_hostcache_part(plan->il, need[i], plan->which) };
            fill += eb;
            host_bytes += eb;
        }
        ggml_metal_dmoe_ring_flush(ctx, data, bid_dst.metal, bid_dst.offs, p, n, &first, plan->il);
        tosh_hostcache_release(plan->il, need, n_need);
        n = 0;
    }
    for (int e = 0; e < plan->n_expert && !tosh_hostcache_on();) {
        if (rows[e] == 0 || plan->slot_of[e] >= 0) {
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < plan->n_expert && rows[e1] > 0 && plan->slot_of[e1] < 0) ++e1;
        size_t off = (size_t) e*eb, len = (size_t) (e1 - e)*eb;
        host_bytes += len;
        while (len > 0) {
            if (fill == DMOE_RING_CHUNK || n == 64) {
                ggml_metal_dmoe_ring_flush(ctx, data, bid_dst.metal, bid_dst.offs, p, n, &first, plan->il);
                n = 0;
                fill = 0;
            }
            const size_t take = MIN(len, DMOE_RING_CHUNK - fill);
            p[n++] = (struct dmoe_piece) { off, fill, take, NULL };
            fill += take;
            off  += take;
            len  -= take;
        }
        e = e1;
    }
    if (!tosh_hostcache_on()) {
        ggml_metal_dmoe_ring_flush(ctx, data, bid_dst.metal, bid_dst.offs, p, n, &first, plan->il);
    }

    if (plan->n_promo > 0) {
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        for (int i = 0; i < plan->n_promo; ++i) {
            const int e = plan->promo_expert[i], sl = plan->promo_slot[i];
            if (e < 0 || e >= plan->n_expert || sl < 0 || rows[e] == 0) continue;
            [encoder copyFromBuffer:bid_dst.metal sourceOffset:bid_dst.offs + (size_t) e*eb
                           toBuffer:bid_arena.metal destinationOffset:bid_arena.offs + (size_t) sl*eb size:eb];
        }
        [encoder endEncoding];
        ggml_metal_dmoe_submit(ctx, cmd_buf, plan->il);
    }

    // when the host runs rare experts, report when this layer's down bank is on the GPU: a read of
    // the slot waits for every blit that wrote it
    if (plan->which == 2 && tosh_dmoe_rare_rows_live() > 0) {
        if (ctx->dmoe_mark == nil) {
            ctx->dmoe_mark = [ggml_metal_device_get_obj(ctx->dev) newBufferWithLength:256 options:MTLResourceStorageModeShared];
        }
        id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
        [encoder copyFromBuffer:bid_dst.metal sourceOffset:bid_dst.offs toBuffer:ctx->dmoe_mark destinationOffset:0 size:16];
        [encoder endEncoding];
        const int il = plan->il;
        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            tosh_dmoe_rare_dma_done(il, cb.GPUEndTime);
        }];
        [cmd_buf commit];
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;
        [cmd_buf retain];
    }

    atomic_fetch_add(&g_dmoe_pf_host_bytes, host_bytes);
    atomic_fetch_add(&g_dmoe_pf_vram_bytes, vram_bytes);
    atomic_fetch_add(&g_dmoe_pf_skip_bytes, skip_bytes);
    for (int i = 0; i < 3; ++i) atomic_fetch_add(&g_dmoe_pf_rare_bytes[i], rare[i]);
    atomic_fetch_add(&g_dmoe_pf_uploads, 1);
    atomic_fetch_add(&g_dmoe_pf_used, used);
    atomic_fetch_add(&g_dmoe_pf_used_res, used_res);
    return true;
}

// A bank upload for prefill: experts already in the VRAM arena are copied device to device,
// the rest come from host memory. Same bytes land in the slot as a full upload.
static bool ggml_metal_dmoe_bank_upload(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t size) {
    struct tosh_dmoe_bank_plan plan;
    if (!tosh_dmoe_enabled() || !tosh_dmoe_bank_planner(data, size, &plan)) {
        return false;
    }
    if (plan.expert_bytes*plan.n_expert != size) {
        return false;
    }
    struct ggml_metal_buffer_id bid_dst   = ggml_metal_get_buffer_id(tensor);
    struct ggml_metal_buffer_id bid_arena = ggml_metal_get_buffer_id((const struct ggml_tensor *) plan.arena);
    if (bid_dst.metal == nil || bid_arena.metal == nil) {
        return false;
    }

    static bool registered = false;
    if (!registered) {
        registered = true;
        atexit(ggml_metal_dmoe_pf_report);
    }
    if (plan.rows != NULL && ggml_metal_dmoe_compact_upload(ctx, &plan, bid_dst, bid_arena, data)) {
        return true;
    }
    if (tosh_hostcache_on()) {
        // the bank holds no bytes: without the chunk's routing every expert is taken from the cache
        static int32_t all_rows[1024];
        static uint64_t n_all = 0;
        if (plan.n_expert > 1024) {
            GGML_ABORT("bounded host cache: a bank upload without routing and more than 1024 experts");
        }
        for (int e = 0; e < plan.n_expert; ++e) all_rows[e] = 1;
        plan.rows = all_rows;
        if (n_all++ == 0) GGML_LOG_WARN("%s: bounded host cache: a bank upload without routing takes every expert\n", __func__);
        if (ggml_metal_dmoe_compact_upload(ctx, &plan, bid_dst, bid_arena, data)) {
            return true;
        }
        GGML_ABORT("bounded host cache: a bank upload found no cached path: %s", tosh_hostcache_error() ? tosh_hostcache_error() : "?");
    }

    int stage = -1;
    size_t offs_src = 0;
    id<MTLBuffer> buf_src = ggml_metal_device_wrap_host(ctx->dev, data, size, &offs_src);
    if (buf_src == nil) {
        stage = ctx->prefetch_stage_cur;
        ctx->prefetch_stage_cur = (ctx->prefetch_stage_cur + 1) % ctx->prefetch_stage_count;
        if (ctx->prefetch_stage_cmds[stage]) {
            [ctx->prefetch_stage_cmds[stage] waitUntilCompleted];
            [ctx->prefetch_stage_cmds[stage] release];
            ctx->prefetch_stage_cmds[stage] = nil;
        }
        buf_src = ctx->prefetch_stages[stage];
        if (buf_src == nil || buf_src.length < size) {
            [buf_src release];
            buf_src = [ggml_metal_device_get_obj(ctx->dev) newBufferWithLength:size
                                                                        options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
            ctx->prefetch_stages[stage] = buf_src;
        }
        if (buf_src == nil) {
            return false;
        }
        offs_src = 0;
    }

    const size_t eb = plan.expert_bytes;
    // with the chunk's routing only the routed experts are carried; the rest of the slot is never read
    const int32_t * rows = plan.rows;
    id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
    ggml_metal_encode_pending_waits(ctx, cmd_buf);
    id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];
    uint64_t host_bytes = 0, vram_bytes = 0, skip_bytes = 0, rare[3] = { 0, 0, 0 }, used = 0, used_res = 0;
    for (int e = 0; e < plan.n_expert;) {
        if (rows && rows[e] == 0) {
            skip_bytes += eb;
            ++e;
            continue;
        }
        used++;
        if (plan.slot_of[e] >= 0) {
            [encoder copyFromBuffer:bid_arena.metal sourceOffset:bid_arena.offs + (size_t) plan.slot_of[e]*eb
                           toBuffer:bid_dst.metal destinationOffset:bid_dst.offs + (size_t) e*eb size:eb];
            vram_bytes += eb;
            used_res++;
            ++e;
            continue;
        }
        int e1 = e + 1;
        while (e1 < plan.n_expert && plan.slot_of[e1] < 0 && (!rows || rows[e1] > 0)) ++e1;
        if (rows) {
            for (int i = e; i < e1; ++i) {
                if (rows[i] <= 1) rare[0] += eb;
                if (rows[i] <= 2) rare[1] += eb;
                if (rows[i] <= 4) rare[2] += eb;
            }
            used += e1 - e - 1;
        }
        const size_t off = (size_t) e*eb, len = (size_t) (e1 - e)*eb;
        if (stage >= 0) {
            memcpy((char *) buf_src.contents + off, (const char *) data + off, len);
        }
        [encoder copyFromBuffer:buf_src sourceOffset:offs_src + off
                       toBuffer:bid_dst.metal destinationOffset:bid_dst.offs + off size:len];
        host_bytes += len;
        e = e1;
    }
    // promotions ride on the upload: the expert is already in VRAM, so it costs a device copy
    for (int i = 0; i < plan.n_promo; ++i) {
        const int e = plan.promo_expert[i], s = plan.promo_slot[i];
        if (e < 0 || e >= plan.n_expert || s < 0) continue;
        [encoder copyFromBuffer:bid_dst.metal sourceOffset:bid_dst.offs + (size_t) e*eb
                       toBuffer:bid_arena.metal destinationOffset:bid_arena.offs + (size_t) s*eb size:eb];
    }
    [encoder endEncoding];
    if (rows) {
        [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
            tosh_dmoe_note_time(2, (int64_t) ((cb.GPUEndTime - cb.GPUStartTime)*1e6));
        }];
    }
    [cmd_buf commit];

    [ctx->cmd_bufs_ext addObject:cmd_buf];
    ctx->cmd_buf_last = cmd_buf;
    if (stage >= 0) {
        ctx->prefetch_stage_cmds[stage] = [cmd_buf retain];
    }
    [cmd_buf retain];

    atomic_fetch_add(&g_dmoe_pf_host_bytes, host_bytes);
    atomic_fetch_add(&g_dmoe_pf_vram_bytes, vram_bytes);
    if (rows) {
        atomic_fetch_add(&g_dmoe_pf_skip_bytes, skip_bytes);
        for (int i = 0; i < 3; ++i) atomic_fetch_add(&g_dmoe_pf_rare_bytes[i], rare[i]);
        atomic_fetch_add(&g_dmoe_pf_uploads, 1);
        atomic_fetch_add(&g_dmoe_pf_used, used);
        atomic_fetch_add(&g_dmoe_pf_used_res, used_res);
    }
    return true;
}
#endif

void ggml_metal_set_tensor_async(ggml_metal_t ctx, struct ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, false);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(tensor);
        if (bid_dst.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_dst.offs += offset;

        // Expert prefetch keeps the zero-copy source path while it fits the
        // device budget, then falls back to a bounded shared staging ring.
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        if (ctx->is_prefetch && offset == 0 && ggml_metal_dmoe_bank_upload(ctx, tensor, data, size)) {
            return;
        }
#endif
        if (ctx->is_prefetch) {
            int stage = -1;
            size_t offs_src = 0;
            id<MTLBuffer> buf_src = ggml_metal_device_wrap_host(ctx->dev, data, size, &offs_src);

            if (buf_src == nil) {
                stage = ctx->prefetch_stage_cur;
                ctx->prefetch_stage_cur = (ctx->prefetch_stage_cur + 1) % ctx->prefetch_stage_count;

                if (ctx->prefetch_stage_cmds[stage]) {
                    [ctx->prefetch_stage_cmds[stage] waitUntilCompleted];
                    [ctx->prefetch_stage_cmds[stage] release];
                    ctx->prefetch_stage_cmds[stage] = nil;
                }

                buf_src = ctx->prefetch_stages[stage];
                if (buf_src == nil || buf_src.length < size) {
                    [buf_src release];
                    buf_src = [device newBufferWithLength:size
                                                  options:MTLResourceStorageModeShared | MTLResourceCPUCacheModeWriteCombined];
                    ctx->prefetch_stages[stage] = buf_src;
                }

                if (buf_src != nil) {
                    memcpy(buf_src.contents, data, size);
                    offs_src = 0;
                }
            }

            if (buf_src != nil) {
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
                id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBuffer];
                ggml_metal_encode_pending_waits(ctx, cmd_buf);
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:buf_src
                           sourceOffset:offs_src
                               toBuffer:bid_dst.metal
                      destinationOffset:bid_dst.offs
                                   size:size];

                [encoder endEncoding];
                [cmd_buf commit];

                [ctx->cmd_bufs_ext addObject:cmd_buf];
                ctx->cmd_buf_last = cmd_buf;
                if (stage >= 0) {
                    ctx->prefetch_stage_cmds[stage] = [cmd_buf retain];
                }

                [cmd_buf retain];

                return;
            }
        }

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ctx->owns_queue && !ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_set(ctx->dev, bid_dst, data, size)) {
            return;
        }

        // wrap the source data into a Metal buffer
        id<MTLBuffer> buf_src = [device newBufferWithBytes:data
                                                    length:size
                                                   options:MTLResourceStorageModeShared];

        GGML_ASSERT(buf_src);

        // queue the copy operation into the queue of the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:buf_src
                   sourceOffset:0
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:size];

        [encoder endEncoding];
        [cmd_buf commit];

        [ctx->buf_refs addObject:buf_src];
        [buf_src release];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_get_tensor_async(ggml_metal_t ctx, const struct ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    if (size == 0) {
        return;
    }

    @autoreleasepool {
        ggml_metal_prof_note(tensor->name, size, true);
        id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(tensor);
        if (bid_src.metal == nil) {
            GGML_ABORT("%s: failed to find buffer for tensor '%s'\n", __func__, tensor->name);
        }

        bid_src.offs += offset;

        // private buffers route small transfers through the device staging
        // buffer (synchronous) instead of allocating a Metal buffer per call
        ggml_backend_buffer_t buffer = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;
        if (!ggml_metal_buffer_is_shared(buffer->context)
            && ggml_metal_device_stage_get(ctx->dev, bid_src, data, size)) {
            return;
        }

        id<MTLCommandQueue> queue = ctx->queue;

        id<MTLBuffer> buf_dst = [device newBufferWithBytesNoCopy:data
                                                          length:size
                                                         options:MTLResourceStorageModeShared
                                                     deallocator:nil];

        if (buf_dst != nil) {
            // queue the copy operation into the queue of the Metal context
            // this will be queued at the end, after any currently ongoing GPU operations
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

            [encoder copyFromBuffer:bid_src.metal
                       sourceOffset:bid_src.offs
                           toBuffer:buf_dst
                  destinationOffset:0
                               size:size];

            [encoder endEncoding];
            [cmd_buf commit];

            [ctx->buf_refs addObject:buf_dst];
            [buf_dst release];

            // do not wait here for completion
            //[cmd_buf waitUntilCompleted];

            // instead, remember a reference to the command buffer and wait for it later if needed
            [ctx->cmd_bufs_ext addObject:cmd_buf];
            ctx->cmd_buf_last = cmd_buf;

            [cmd_buf retain];

            return;
        }

        // newBufferWithBytesNoCopy requires page-aligned data, and some drivers cap the size of
        // host-visible allocations; copy through a small staging buffer in chunks instead. The
        // staging path is synchronous (commit + waitUntilCompleted) so callers that would later
        // wait on cmd_buf_last still observe a completed transfer.
        size_t stage_size = MIN(size, (size_t) 8*1024*1024);
        id<MTLBuffer> buf_stage = nil;
        while (stage_size > 0) {
            buf_stage = [device newBufferWithLength:stage_size options:MTLResourceStorageModeShared];
            if (buf_stage != nil) {
                break;
            }
            stage_size /= 2;
        }
        if (buf_stage == nil) {
            GGML_LOG_ERROR("%s: failed to allocate staging buffer, size = %zu\n", __func__, size);
        }
        GGML_ASSERT(buf_stage);

        for (size_t done = 0; done < size;) {
            const size_t n = MIN(stage_size, size - done);

            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);

            {
                id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

                [encoder copyFromBuffer:bid_src.metal
                           sourceOffset:bid_src.offs + done
                               toBuffer:buf_stage
                      destinationOffset:0
                                   size:n];

                [encoder endEncoding];
            }

            [cmd_buf commit];
            [cmd_buf waitUntilCompleted];

            memcpy((char *) data + done, buf_stage.contents, n);

            done += n;
        }
    }
}

static struct ggml_metal_xdev_link * ggml_metal_xdev_link_get(
        ggml_metal_t ctx_src, ggml_metal_t ctx_dst) {
    struct ggml_metal_xdev_link * link = NULL;
    for (int i = 0; i < ctx_dst->n_xlinks; i++) {
        if (ctx_dst->xlinks[i].src_dev == ctx_src->dev) {
            link = &ctx_dst->xlinks[i];
            break;
        }
    }
    if (link == NULL) {
        if (ctx_dst->n_xlinks == 4) {
            return NULL;
        }
        link = &ctx_dst->xlinks[ctx_dst->n_xlinks++];
        link->src_dev  = ctx_src->dev;
        link->peer_x_ok = -1;
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->ev_ready = [dev_src newSharedEvent];
        link->ev_done  = [dev_dst newSharedEvent];
        link->ev_x_ready = [dev_src newSharedEvent];
        link->ev_x_done  = [dev_dst newSharedEvent];
        if (link->ev_ready == nil || link->ev_done == nil ||
            link->ev_x_ready == nil || link->ev_x_done == nil) {
            [link->ev_ready release];
            [link->ev_done release];
            [link->ev_x_ready release];
            [link->ev_x_done release];
            memset(link, 0, sizeof(*link));
            ctx_dst->n_xlinks--;
            return NULL;
        }
    }

    return link;
}

static bool ggml_metal_xdev_link_reserve(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                        struct ggml_metal_xdev_link * link, size_t size) {
    if (link->cap == 0) {
        GGML_LOG_INFO("%s: cross-device hand-off via shared events and host staging (TOSH_MGPU_EVENTS)\n", __func__);
    }

    if (link->cap < size) {
        // resizing frees pages a queued blit may still read; both paths use these buffers, so
        // wait each one out on the events it signals
        if (link->seq > 0 && ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
            return false;
        }
        if (link->seq_x > 0 && ![link->ev_x_done waitUntilSignaledValue:link->seq_x timeoutMS:10000]) {
            return false;
        }
        [link->wrap_src release];
        [link->wrap_dst release];
        free(link->host);
        link->cap = MAX(size, (size_t) 1*1024*1024);
        if (posix_memalign(&link->host, (size_t) sysconf(_SC_PAGESIZE), link->cap) != 0) {
            link->host = NULL;
            link->cap  = 0;
            return false;
        }
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        link->wrap_src = [dev_src newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        link->wrap_dst = [dev_dst newBufferWithBytesNoCopy:link->host
                              length:link->cap options:MTLResourceStorageModeShared deallocator:nil];
        if (link->wrap_src == nil || link->wrap_dst == nil) {
            // leaving cap set would make the next call skip this block and hand out the nil wraps
            [link->wrap_src release];
            [link->wrap_dst release];
            link->wrap_src = nil;
            link->wrap_dst = nil;
            free(link->host);
            link->host = NULL;
            link->cap  = 0;
            return false;
        }
    }
    return true;
}


// Must be called before the buffer is committed: Metal refuses a handler added after.
static void tosh_trace_cmd_buf(id<MTLCommandBuffer> cmd_buf) {
    if (!tosh_mgpu_trace()) {
        return;
    }
    atomic_fetch_add(&g_tr_cmd_bufs, 1);
    [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) {
        const double gpu   = cb.GPUEndTime   - cb.GPUStartTime;
        const double sched = cb.GPUStartTime - cb.kernelEndTime;
        if (gpu > 0) {
            atomic_fetch_add(&g_tr_gpu_ns, (uint64_t) (gpu*1e9));
        }
        if (sched > 0) {
            atomic_fetch_add(&g_tr_sched_ns, (uint64_t) (sched*1e9));
        }
    }];
}

// TOSH_MGPU_NOWAIT=1, benchmark only: the collective still publishes, still signals and still
// reduces, but never waits for the partner. The output is wrong by construction. Subtracting
// this from the normal run measures the stall at the wait inside the real collective, instead
// of inferring it from whole-command-buffer completion timestamps.
static bool ggml_metal_collective_nowait(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_NOWAIT");
        v = (s && s[0] == '1') ? 1 : 0;
        if (v) {
            fprintf(stderr, "ggml_metal: WARNING TOSH_MGPU_NOWAIT=1, the collective does not wait "
                            "for the peer and the output is meaningless. Benchmark only.\n");
        }
    }
    return v == 1;
}

// TOSH_MGPU_PEER_SELF runs the peer hand-off between two logical devices backed by the
// same GPU (GGML_METAL_DEVICE_LIST=0,0), which needs no bridge: the queues are still
// separate, so the cross-device ordering is testable on a single card.
static bool ggml_metal_peer_self(void) {
    static int val = -1;
    if (val < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_PEER_SELF");
        val = (v && v[0] == '1') ? 1 : 0;
    }
    return val;
}

// Direct copy between GPUs in the same Metal peer group.
static bool ggml_metal_cpy_xdev_peer(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                     const struct ggml_tensor * src, struct ggml_tensor * dst) {
#if TARGET_OS_OSX
    if (@available(macOS 10.15, *)) {
        id<MTLDevice> dev_src = ggml_metal_device_get_obj(ctx_src->dev);
        id<MTLDevice> dev_dst = ggml_metal_device_get_obj(ctx_dst->dev);
        const bool same_dev = dev_src == dev_dst && ggml_metal_peer_self();
        const uint64_t peer_group = dev_src.peerGroupID;
        if (!same_dev && (peer_group == 0 || peer_group != dev_dst.peerGroupID)) {
            return false;
        }

        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }
        id<MTLBuffer> metal_src = (id<MTLBuffer>) bid_src.metal;
        id<MTLBuffer> metal_dst = (id<MTLBuffer>) bid_dst.metal;

        const size_t size = ggml_nbytes(src);
        if (bid_src.offs > metal_src.length || size > metal_src.length - bid_src.offs ||
            bid_dst.offs > metal_dst.length || size > metal_dst.length - bid_dst.offs) {
            return false;
        }

        struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
        if (link == NULL) {
            return false;
        }

        if (link->peer_src == metal_src) {
            if (tosh_mgpu_trace()) {
                atomic_fetch_add(&g_tr_rv_hit, 1);
            }
        }

        if (link->peer_src != metal_src) {
            // Wait before replacing a remote view still used by the last blit.
            if (link->peer_view != nil && link->seq > 0 &&
                ![link->ev_done waitUntilSignaledValue:link->seq timeoutMS:10000]) {
                return false;
            }

            [link->peer_view release];
            [link->peer_src release];
            link->peer_view = same_dev ? [metal_src retain]
                                       : [metal_src newRemoteBufferViewForDevice:dev_dst];
            link->peer_src  = link->peer_view != nil ? [metal_src retain] : nil;
            if (link->peer_view == nil) {
                return false;
            }
            if (tosh_mgpu_trace()) {
                atomic_fetch_add(&g_tr_rv_new, 1);
            }
        }

        if (tosh_mgpu_trace()) {
            atomic_fetch_add(&g_tr_copy, 1);
            atomic_fetch_add(&g_tr_bytes_peer, size);
        }

        const uint64_t seq = ++link->seq;
        @autoreleasepool {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_src = [ctx_src->queue commandBuffer];
            tosh_trace_cmd_buf(cmd_src);
            ggml_metal_encode_pending_waits(ctx_src, cmd_src);
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_src encodeSignalEvent:link->ev_ready value:seq];
            [cmd_src commit];
            [ctx_src->cmd_bufs_ext addObject:cmd_src];
            ctx_src->cmd_buf_last = cmd_src;
            [cmd_src retain];

            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_dst = [ctx_dst->queue commandBuffer];
            tosh_trace_cmd_buf(cmd_dst);
            ggml_metal_encode_pending_waits(ctx_dst, cmd_dst);
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
            id<MTLBlitCommandEncoder> encoder = [cmd_dst blitCommandEncoder];
            [encoder copyFromBuffer:link->peer_view sourceOffset:bid_src.offs
                           toBuffer:metal_dst destinationOffset:bid_dst.offs size:size];
            [encoder endEncoding];
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_dst encodeSignalEvent:link->ev_done value:seq];
            [cmd_dst commit];
            [ctx_dst->cmd_bufs_ext addObject:cmd_dst];
            ctx_dst->cmd_buf_last = cmd_dst;
            [cmd_dst retain];
        }

        ggml_metal_peer_wait_add(ctx_src, link->ev_done, seq);

        if (!link->peer_logged) {
            // fprintf, not GGML_LOG_INFO, so the confirmation shows at the default
            // log level like the SIMD-width line. Deduped by peer group across contexts:
            // a tool that reloads the model per run rebuilds the link every time.
            static uint64_t logged[8];
            static int      n_logged = 0;

            bool seen = false;
            for (int i = 0; i < n_logged; ++i) {
                seen = seen || logged[i] == peer_group;
            }

            if (!seen) {
                fprintf(stderr, same_dev
                            ? "ggml_metal: peer transfer self-test: %s -> %s (no bridge, peer group %llu)\n"
                            : "ggml_metal: Infinity Fabric peer transfer enabled: %s -> %s (peer group %llu)\n",
                        dev_src.name.UTF8String, dev_dst.name.UTF8String,
                        (unsigned long long) peer_group);
                if (n_logged < (int) (sizeof(logged)/sizeof(logged[0]))) {
                    logged[n_logged++] = peer_group;
                }
            }

            link->peer_logged = true;
        }
        return true;
    }
#else
    GGML_UNUSED(ctx_src);
    GGML_UNUSED(ctx_dst);
    GGML_UNUSED(src);
    GGML_UNUSED(dst);
#endif
    return false;
}

// event-chained hand-off: both blits share one host block and order through
// MTLSharedEvents, so neither GPU drains and the CPU never blocks
static _Atomic uint64_t g_xdev_ev = 0, g_xdev_peer = 0, g_xdev_gen = 0;

static void tosh_xdev_report(void) {
    fprintf(stderr, "TOSH_XDEV  events=%llu  peer=%llu  generic=%llu\n",
            (unsigned long long) atomic_load(&g_xdev_ev),
            (unsigned long long) atomic_load(&g_xdev_peer),
            (unsigned long long) atomic_load(&g_xdev_gen));
}

static bool ggml_metal_cpy_xdev_events(ggml_metal_t ctx_src, ggml_metal_t ctx_dst,
                                       const struct ggml_tensor * src, struct ggml_tensor * dst,
                                       bool hold_dst) {
    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
    if (bid_src.metal == nil || bid_dst.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(src);

    struct ggml_metal_xdev_link * link = ggml_metal_xdev_link_get(ctx_src, ctx_dst);
    if (link == NULL) {
        return false;
    }
    if (!ggml_metal_xdev_link_reserve(ctx_src, ctx_dst, link, size)) {
        return false;
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_copy, 1);
        // twice: the block is written on one card and read back on the other
        atomic_fetch_add(&g_tr_bytes_host, 2*size);
    }

    const uint64_t seq = ++link->seq;

    @autoreleasepool {
        // reuse the host block only after the destination consumed the last hand-off
        id<MTLCommandBuffer> cmd_src = ggml_metal_cmd_buf_continue(ctx_src);
        if (seq > 1) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_src encodeWaitForEvent:link->ev_done value:seq - 1];
        }
        id<MTLBlitCommandEncoder> enc_src = [cmd_src blitCommandEncoder];
        [enc_src copyFromBuffer:bid_src.metal sourceOffset:bid_src.offs
                       toBuffer:link->wrap_src destinationOffset:0 size:size];
        [enc_src endEncoding];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_src encodeSignalEvent:link->ev_ready value:seq];
        ggml_metal_cmd_buf_end(ctx_src, cmd_src, false);

        id<MTLCommandBuffer> cmd_dst = ggml_metal_cmd_buf_new(ctx_dst);
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_dst encodeWaitForEvent:link->ev_ready value:seq];
        id<MTLBlitCommandEncoder> enc_dst = [cmd_dst blitCommandEncoder];
        [enc_dst copyFromBuffer:link->wrap_dst sourceOffset:0
                       toBuffer:bid_dst.metal destinationOffset:bid_dst.offs size:size];
        [enc_dst endEncoding];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_dst encodeSignalEvent:link->ev_done value:seq];
        // the signal lands when the blit does, so holding the buffer open does not keep the
        // source waiting on whatever gets encoded into it afterwards
        ggml_metal_cmd_buf_end(ctx_dst, cmd_dst, hold_dst);
    }

    return true;
}

bool ggml_metal_cpy_tensor_async(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    return ggml_metal_cpy_tensor_async_ex(ctx_src, ctx_dst, src, dst, /*prefer_events =*/ false, /*hold_dst =*/ false);
}

bool ggml_metal_cpy_tensor_async_ex(ggml_metal_t ctx_src, ggml_metal_t ctx_dst, const struct ggml_tensor * src, struct ggml_tensor * dst, bool prefer_events, bool hold_dst) {
    // A Metal blit can't reach a buffer owned by another device, so a multi-GPU layer
    // hand-off goes through host memory (GGML_METAL_CROSS_STAGING_DISABLE uses the generic copy).
    if (ctx_src->dev != ctx_dst->dev) {
        if (TOSH_ENV("GGML_METAL_CROSS_STAGING_DISABLE") != NULL) {
            return false;
        }

        // both buffers must be private for the cross-device blits
        static int use_events = -1;
        if (use_events < 0) {
            const char * v = TOSH_ENV("TOSH_MGPU_EVENTS");
            use_events = (v && v[0] == '1') ? 1 : 0;
        }
        // Infinity Fabric peer copy is opt-in until validated on bridged hardware:
        // a success that silently returns wrong data has no fallback.
        static int use_peer = -1;
        if (use_peer < 0) {
            const char * v = TOSH_ENV("TOSH_MGPU_PEER");
            use_peer = (v && v[0] == '1') ? 1 : 0;
        }
        ggml_backend_buffer_t ebsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t ebdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        // which of the three hand-offs actually runs is not visible otherwise, and the flag
        // alone does not tell: TOSH_MGPU_XDEV_COUNT reports the tally at exit
        static int count = -1;
        if (count < 0) {
            count = TOSH_ENV("TOSH_MGPU_XDEV_COUNT") != NULL;
            if (count) {
                atexit(tosh_xdev_report);
            }
        }

        if (!ggml_metal_buffer_is_shared(ebsrc->context) &&
            !ggml_metal_buffer_is_shared(ebdst->context)) {
            if (prefer_events && use_events && ggml_metal_cpy_xdev_events(ctx_src, ctx_dst, src, dst, hold_dst)) {
                atomic_fetch_add(&g_xdev_ev, 1);
                return true;
            }
            if (use_peer && ggml_metal_cpy_xdev_peer(ctx_src, ctx_dst, src, dst)) {
                atomic_fetch_add(&g_xdev_peer, 1);
                return true;
            }
            if (use_events && ggml_metal_cpy_xdev_events(ctx_src, ctx_dst, src, dst, hold_dst)) {
                atomic_fetch_add(&g_xdev_ev, 1);
                return true;
            }
        }
        atomic_fetch_add(&g_xdev_gen, 1);
        // drain both GPUs: the source before reading it, the destination before
        // overwriting an input its in-flight graph may still be consuming
        ggml_metal_synchronize(ctx_src);
        ggml_metal_synchronize(ctx_dst);

        ggml_backend_buffer_t bsrc = src->view_src ? src->view_src->buffer : src->buffer;
        ggml_backend_buffer_t bdst = dst->view_src ? dst->view_src->buffer : dst->buffer;
        ggml_metal_buffer_t buf_src = (ggml_metal_buffer_t) bsrc->context;
        ggml_metal_buffer_t buf_dst = (ggml_metal_buffer_t) bdst->context;

        const size_t size = ggml_nbytes(src);
        void * stage = malloc(size);
        if (!stage) {
            return false;
        }
        ggml_metal_buffer_get_tensor(buf_src, src, stage, 0, size);
        ggml_metal_buffer_set_tensor(buf_dst, dst, stage, 0, size);
        free(stage);
        return true;
    }

    @autoreleasepool {
        struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
        struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);

        if (bid_src.metal == nil || bid_dst.metal == nil) {
            return false;
        }

        id<MTLCommandQueue> dst_queue = ctx_dst->queue;
        id<MTLCommandBuffer> sync_cmd_buf = [dst_queue commandBuffer];

        ggml_metal_event_encode_signal(ctx_dst->ev_sync, sync_cmd_buf);

        [sync_cmd_buf commit];

        [ctx_dst->cmd_bufs_ext addObject:sync_cmd_buf];
        ctx_dst->cmd_buf_last = sync_cmd_buf;

        [sync_cmd_buf retain];

        // queue the copy operation into the Metal context
        // this will be queued at the end, after any currently ongoing GPU operations
        id<MTLCommandQueue> queue = ctx_src->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx_src, cmd_buf);
        ggml_metal_event_encode_wait(ctx_dst->ev_sync, cmd_buf);
        id<MTLBlitCommandEncoder> encoder = [cmd_buf blitCommandEncoder];

        [encoder copyFromBuffer:bid_src.metal
                   sourceOffset:bid_src.offs
                       toBuffer:bid_dst.metal
              destinationOffset:bid_dst.offs
                           size:ggml_nbytes(src)];

        [encoder endEncoding];

        ggml_metal_event_t ev_cpy = ggml_metal_get_ev_cpy(ctx_src);
        ggml_metal_event_encode_signal(ev_cpy, cmd_buf);

        [cmd_buf commit];

        // do not wait here for completion
        //[cmd_buf waitUntilCompleted];

        // instead, remember a reference to the command buffer and wait for it later if needed
        [ctx_src->cmd_bufs_ext addObject:cmd_buf];
        ctx_src->cmd_buf_last = cmd_buf;

        [cmd_buf retain];

        ggml_metal_event_wait(ctx_dst, ev_cpy);

        return true;
    }
}

bool ggml_metal_add_inplace_supported(ggml_metal_t ctx, const struct ggml_tensor * acc, const struct ggml_tensor * add) {
    if (ctx->has_error) {
        return false;
    }

    if (acc->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32) {
        return false;
    }

    if (!ggml_are_same_shape(acc, add) || !ggml_is_contiguous(acc) || !ggml_is_contiguous(add)) {
        return false;
    }

    if (ggml_metal_get_buffer_id(acc).metal == nil || ggml_metal_get_buffer_id(add).metal == nil) {
        return false;
    }

    return ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD).pipeline != NULL;
}

static bool ggml_metal_add_inplace_into(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf_in, struct ggml_tensor * acc, const struct ggml_tensor * add) {
    if (!ggml_metal_add_inplace_supported(ctx, acc, add)) {
        // a copy may be holding its buffer open for this add, and nothing else would submit it
        ggml_metal_cmd_buf_hold_flush(ctx);
        return false;
    }

    struct ggml_metal_buffer_id bid_acc = ggml_metal_get_buffer_id(acc);
    struct ggml_metal_buffer_id bid_add = ggml_metal_get_buffer_id(add);

    struct ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD);

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ (int32_t) acc->ne[0],
        /*.ne01 =*/ (int32_t) acc->ne[1],
        /*.ne02 =*/ (int32_t) acc->ne[2],
        /*.ne03 =*/ (int32_t) acc->ne[3],
        /*.nb00 =*/ acc->nb[0],
        /*.nb01 =*/ acc->nb[1],
        /*.nb02 =*/ acc->nb[2],
        /*.nb03 =*/ acc->nb[3],
        /*.ne10 =*/ (int32_t) add->ne[0],
        /*.ne11 =*/ (int32_t) add->ne[1],
        /*.ne12 =*/ (int32_t) add->ne[2],
        /*.ne13 =*/ (int32_t) add->ne[3],
        /*.nb10 =*/ add->nb[0],
        /*.nb11 =*/ add->nb[1],
        /*.nb12 =*/ add->nb[2],
        /*.nb13 =*/ add->nb[3],
        /*.ne0  =*/ (int32_t) acc->ne[0],
        /*.ne1  =*/ (int32_t) acc->ne[1],
        /*.ne2  =*/ (int32_t) acc->ne[2],
        /*.ne3  =*/ (int32_t) acc->ne[3],
        /*.nb0  =*/ acc->nb[0],
        /*.nb1  =*/ acc->nb[1],
        /*.nb2  =*/ acc->nb[2],
        /*.nb3  =*/ acc->nb[3],
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_add.offs },
    };

    // the kernel adds the fused offsets to the src1 base itself
    bid_add.offs = 0;

    if (pipeline.c4) {
        args.ne00 /= 4;
        args.ne10 /= 4;
        args.ne0  /= 4;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = cmd_buf_in != nil ? cmd_buf_in : ggml_metal_cmd_buf_continue(ctx);

        ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_add, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 3);

        if (pipeline.cnt) {
            ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, (int) ggml_nrows(acc), 1, 1, 1, 1);
        } else {
            const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

            int nth = 1;
            while (2*nth < args.ne0 && nth < nth_max) {
                nth *= 2;
            }

            ggml_metal_encoder_dispatch_threadgroups(enc, args.ne01, args.ne02, args.ne03, nth, 1, 1);
        }

        ggml_metal_encoder_end_encoding(enc);
        ggml_metal_encoder_free(enc);

        if (cmd_buf_in == nil) {
            ggml_metal_cmd_buf_end(ctx, cmd_buf, false);
        }
    }

    return true;
}

// acc += a plain buffer laid out exactly like acc. The collective's partner publishes its
// partial that way, so the add can read it straight across the fabric with no tensor to wrap.
// TOSH_MGPU_FB_ADD_RB=0 keeps the generic add below. A single row goes through the row-broadcast
// float4 variant the graph already uses for the residual, since bin_one strides a few threads
// across the whole row and is several times slower on it.
static bool ggml_metal_fb_add_rb(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_FB_ADD_RB"); v = (s && s[0] == '0') ? 0 : 1; }
    return v == 1;
}

static struct ggml_metal_pipeline_with_params ggml_metal_pipeline_add_rb4(ggml_metal_library_t lib) {
    const char * base = "kernel_bin_fuse_f32_f32_f32_4";
    const char * name = "kernel_bin_fuse_f32_f32_f32_4_op=0_nf=1_rb=1_cb=0";
    struct ggml_metal_pipeline_with_params res = ggml_metal_library_get_pipeline(lib, name);
    if (!res.pipeline) {
        ggml_metal_cv_t cv = ggml_metal_cv_init();
        ggml_metal_cv_set_int16(cv, 0,     FC_BIN + 0);
        ggml_metal_cv_set_int16(cv, 1,     FC_BIN + 1);
        ggml_metal_cv_set_bool (cv, true,  FC_BIN + 2);
        ggml_metal_cv_set_bool (cv, false, FC_BIN + 3);
        res = ggml_metal_library_compile_pipeline(lib, base, name, cv);
        ggml_metal_cv_free(cv);
    }
    res.c4  = true;
    res.cnt = true;
    return res;
}

static bool ggml_metal_add_inplace_buffer(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf,
                                          struct ggml_tensor * acc,
                                          id<MTLBuffer> add, size_t add_offs) {
    const bool rb4 = ggml_metal_fb_add_rb() && ggml_nrows(acc) == 1 && acc->ne[0] % 4 == 0 &&
                     ggml_is_contiguous(acc) && ggml_nelements(acc) < 65536;
    struct ggml_metal_pipeline_with_params pipeline = rb4 ? ggml_metal_pipeline_add_rb4(ctx->lib)
                                                          : ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD);
    // Runtime compilation of the specialized row-broadcast kernel is optional. Keep the
    // generic ADD as a correctness fallback if this device cannot create that pipeline.
    if (pipeline.pipeline == NULL && rb4) {
        pipeline = ggml_metal_library_get_pipeline_bin_one(ctx->lib, GGML_OP_ADD);
    }
    if (pipeline.pipeline == NULL) {
        return false;
    }

    struct ggml_metal_buffer_id bid_acc = ggml_metal_get_buffer_id(acc);
    struct ggml_metal_buffer_id bid_add = { .metal = add, .offs = add_offs };

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ (int32_t) acc->ne[0],
        /*.ne01 =*/ (int32_t) acc->ne[1],
        /*.ne02 =*/ (int32_t) acc->ne[2],
        /*.ne03 =*/ (int32_t) acc->ne[3],
        /*.nb00 =*/ acc->nb[0],
        /*.nb01 =*/ acc->nb[1],
        /*.nb02 =*/ acc->nb[2],
        /*.nb03 =*/ acc->nb[3],
        /*.ne10 =*/ (int32_t) acc->ne[0],
        /*.ne11 =*/ (int32_t) acc->ne[1],
        /*.ne12 =*/ (int32_t) acc->ne[2],
        /*.ne13 =*/ (int32_t) acc->ne[3],
        /*.nb10 =*/ acc->nb[0],
        /*.nb11 =*/ acc->nb[1],
        /*.nb12 =*/ acc->nb[2],
        /*.nb13 =*/ acc->nb[3],
        /*.ne0  =*/ (int32_t) acc->ne[0],
        /*.ne1  =*/ (int32_t) acc->ne[1],
        /*.ne2  =*/ (int32_t) acc->ne[2],
        /*.ne3  =*/ (int32_t) acc->ne[3],
        /*.nb0  =*/ acc->nb[0],
        /*.nb1  =*/ acc->nb[1],
        /*.nb2  =*/ acc->nb[2],
        /*.nb3  =*/ acc->nb[3],
        /*.offs =*/ 0,
        /*.o1   =*/ { add_offs },
    };
    bid_add.offs = 0;

    if (pipeline.c4) {
        args.ne00 /= 4;
        args.ne10 /= 4;
        args.ne0  /= 4;
    }

    ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_acc, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_add, 2);
    ggml_metal_encoder_set_buffer  (enc, bid_acc, 3);

    if (pipeline.cnt) {
        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne0, (int) ggml_nrows(acc), 1, 1, 1, 1);
    } else {
        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;
        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne01, args.ne02, args.ne03, nth, 1, 1);
    }

    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);

    return true;
}

// A TP2 boundary of one or two rows leaves the partner's partial in its slot of the host block and the
// residual add that consumes the reduced tensor folds it in as a third addend, which saves the
// boundary's own add and encoder. The slot is released at the end of that subgraph, so the
// staged exchange alternates two slots and back-pressure waits on the round that used the slot.
// TOSH_MGPU_FB_RESIDUAL=0 keeps the add inside the boundary.
static bool ggml_metal_fb_residual(void) {
    static int v = -1;
    if (v < 0) { const char * s = TOSH_ENV("TOSH_MGPU_FB_RESIDUAL"); v = (s && s[0] == '0') ? 0 : 1; }
    return v == 1;
}

struct ggml_metal_fb_residual_state {
    ggml_metal_device_t        dev;
    const struct ggml_tensor * t;       // reduced tensor still missing the partner's partial
    const struct ggml_tensor * alias;   // a same-storage reshape of t that the residual add reads
    id<MTLBuffer>              buf;
    size_t                     off;
    id<MTLSharedEvent>         ev_done;
    uint64_t                   seq;
    int                        state;   // 0 idle, 1 pending, 2 armed for the residual add, 3 folded
};

static struct ggml_metal_fb_residual_state g_fb_res[8];
static os_unfair_lock g_fb_res_lock = OS_UNFAIR_LOCK_INIT;

// one entry per device, and only that device's encoding thread touches it after lookup
static struct ggml_metal_fb_residual_state * ggml_metal_fb_residual_get(ggml_metal_device_t dev) {
    struct ggml_metal_fb_residual_state * res = NULL;
    os_unfair_lock_lock(&g_fb_res_lock);
    for (int i = 0; i < 8; i++) {
        if (g_fb_res[i].dev == dev || g_fb_res[i].dev == NULL) {
            g_fb_res[i].dev = dev;
            res = &g_fb_res[i];
            break;
        }
    }
    os_unfair_lock_unlock(&g_fb_res_lock);
    return res;
}

bool ggml_metal_fb_residual_armed(ggml_metal_device_t dev, const struct ggml_tensor * t) {
    if (!ggml_metal_fb_residual()) {
        return false;
    }
    struct ggml_metal_fb_residual_state * st = ggml_metal_fb_residual_get(dev);
    return st != NULL && st->state == 2 && t != NULL && (st->t == t || st->alias == t);
}

bool ggml_metal_fb_residual_take(ggml_metal_device_t dev, const struct ggml_tensor * t, struct ggml_metal_buffer_id * peer) {
    if (!ggml_metal_fb_residual()) {
        return false;
    }
    struct ggml_metal_fb_residual_state * st = ggml_metal_fb_residual_get(dev);
    if (st == NULL || st->state != 2 || (st->t != t && st->alias != t)) {
        return false;
    }
    peer->metal = st->buf;
    peer->offs  = st->off;
    st->state = 3;
    if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_res_fold, 1);
    return true;
}

// before a subgraph: arm the pending partial when its first user here is a residual add
// with the reduced tensor as src0 and encoded by this command buffer, otherwise fold it the
// usual way and release the slot
static void ggml_metal_fb_residual_before(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf,
                                          struct ggml_cgraph * gf, int n_main) {
    struct ggml_metal_fb_residual_state * st = ggml_metal_fb_residual() ? ggml_metal_fb_residual_get(ctx->dev) : NULL;
    if (st == NULL || st->state != 1) {
        return;
    }
    const struct ggml_tensor * t = st->t;
    const struct ggml_tensor * alias = NULL;
    bool arm = false;
    for (int i = 0; i < gf->n_nodes; i++) {
        const struct ggml_tensor * node = gf->nodes[i];
        bool uses = node->view_src == t;
        for (int k = 0; k < GGML_MAX_SRC && !uses; k++) {
            for (const struct ggml_tensor * src = node->src[k]; src != NULL; src = src->view_src) {
                if (src == t) { uses = true; break; }
            }
        }
        if (!uses) {
            continue;
        }
        // a reshape or view of the whole tensor at offset 0 moves no data: follow it once
        if (alias == NULL && (node->op == GGML_OP_RESHAPE || node->op == GGML_OP_VIEW) &&
            node->view_src == t && node->view_offs == 0 && ggml_is_contiguous(node) &&
            ggml_nelements(node) == ggml_nelements(t)) {
            alias = node;
            continue;
        }
        const struct ggml_tensor * src0 = node->src[0];
        arm = i < n_main && node->op == GGML_OP_ADD && (src0 == t || (alias != NULL && src0 == alias)) &&
              node->src[1] != NULL &&
              node->type == GGML_TYPE_F32 && node->src[1]->type == GGML_TYPE_F32 &&
              ggml_is_contiguous(node) && ggml_is_contiguous(node->src[1]) &&
              ggml_are_same_shape(node, src0) && ggml_are_same_shape(node->src[1], src0) &&
              ggml_metal_device_pipeline_add3(ctx->dev).pipeline != NULL;
        break;
    }
    st->alias = arm ? alias : NULL;
    if (arm) {
        st->state = 2;
        return;
    }
    const bool added = ggml_metal_add_inplace_buffer(ctx, cmd_buf, (struct ggml_tensor *) t, st->buf, st->off);
    GGML_ASSERT(added);
    if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
    [cmd_buf encodeSignalEvent:st->ev_done value:st->seq];
    st->state = 0;
}

// after the subgraph's encoder closed: the folded slot has been read
static void ggml_metal_fb_residual_after(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf) {
    struct ggml_metal_fb_residual_state * st = ggml_metal_fb_residual() ? ggml_metal_fb_residual_get(ctx->dev) : NULL;
    if (st == NULL) {
        return;
    }
    GGML_ASSERT(st->state != 2 && "an armed TP2 partial was not folded");
    if (st->state == 3) {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:st->ev_done value:st->seq];
        st->state = 0;
    }
}

bool ggml_metal_add_inplace_async(ggml_metal_t ctx, struct ggml_tensor * acc, const struct ggml_tensor * add) {
    return ggml_metal_add_inplace_into(ctx, nil, acc, add);
}

// --- f16 transport for the prefill allreduce -------------------------------------------------
// The copy butterfly moves f32 partials across the peer bridge and at prefill payloads the
// fabric, not the wait, is the limit. Convert locally to f16 (half the bytes in flight), move
// those across, and fold them into the f32 accumulator with an upcast add.

bool ggml_metal_f16_stage_supported(ggml_metal_t ctx, const struct ggml_tensor * t) {
    if (ctx->has_error) {
        return false;
    }
    if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t) || ggml_nelements(t) == 0) {
        return false;
    }
    if (ggml_metal_get_buffer_id(t).metal == nil) {
        return false;
    }
    if (ggml_metal_library_get_pipeline_cpy(ctx->lib, GGML_TYPE_F32, GGML_TYPE_F16, true).pipeline == NULL) {
        return false;
    }
    if (ggml_metal_library_get_pipeline_bin_one_src1(ctx->lib, GGML_OP_ADD, GGML_TYPE_F16).pipeline == NULL) {
        return false;
    }
    return true;
}

// local dispatch, its own command buffer: the peer copy that follows is ordered after it by
// the same queue/event rules that already order a computed source tensor before its push
bool ggml_metal_cvt_f32_f16(ggml_metal_t ctx, const struct ggml_tensor * src, struct ggml_tensor * dst) {
    if (ctx->has_error || src->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F16) {
        return false;
    }

    struct ggml_metal_pipeline_with_params pipe_cpy = ggml_metal_library_get_pipeline_cpy(ctx->lib, GGML_TYPE_F32, GGML_TYPE_F16, true);
    if (pipe_cpy.pipeline == NULL) {
        return false;
    }

    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = ggml_metal_get_buffer_id(dst);
    if (bid_src.metal == nil || bid_dst.metal == nil) {
        return false;
    }

    const int64_t n = ggml_nelements(src);

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ n,
        /*.ne00 =*/ src->ne[0], /*.ne01 =*/ src->ne[1], /*.ne02 =*/ src->ne[2], /*.ne03 =*/ src->ne[3],
        /*.nb00 =*/ src->nb[0], /*.nb01 =*/ src->nb[1], /*.nb02 =*/ src->nb[2], /*.nb03 =*/ src->nb[3],
        /*.ne0  =*/ dst->ne[0], /*.ne1  =*/ dst->ne[1], /*.ne2  =*/ dst->ne[2], /*.ne3  =*/ dst->ne[3],
        /*.nb0  =*/ dst->nb[0], /*.nb1  =*/ dst->nb[1], /*.nb2  =*/ dst->nb[2], /*.nb3  =*/ dst->nb[3],
    };

    const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipe_cpy));

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx);

        ggml_metal_encoder_t ec = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);
        ggml_metal_encoder_set_pipeline(ec, pipe_cpy);
        ggml_metal_encoder_set_bytes   (ec, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (ec, bid_src, 1);
        ggml_metal_encoder_set_buffer  (ec, bid_dst, 2);
        ggml_metal_encoder_dispatch_threadgroups(ec, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
        ggml_metal_encoder_end_encoding(ec);
        ggml_metal_encoder_free(ec);

        ggml_metal_cmd_buf_end(ctx, cmd_buf, false);
    }

    return true;
}

// same encode as ggml_metal_add_inplace_into, but src1 is f16 and upcast inside the kernel
bool ggml_metal_add_inplace_f16_src(ggml_metal_t ctx, struct ggml_tensor * acc, const struct ggml_tensor * add) {
    if (ctx->has_error) {
        return false;
    }
    if (acc->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F16) {
        return false;
    }
    if (!ggml_are_same_shape(acc, add) || !ggml_is_contiguous(acc) || !ggml_is_contiguous(add)) {
        return false;
    }

    struct ggml_metal_buffer_id bid_acc = ggml_metal_get_buffer_id(acc);
    struct ggml_metal_buffer_id bid_add = ggml_metal_get_buffer_id(add);
    if (bid_acc.metal == nil || bid_add.metal == nil) {
        return false;
    }

    struct ggml_metal_pipeline_with_params pipeline = ggml_metal_library_get_pipeline_bin_one_src1(ctx->lib, GGML_OP_ADD, GGML_TYPE_F16);
    if (pipeline.pipeline == NULL) {
        return false;
    }

    ggml_metal_kargs_bin args = {
        /*.ne00 =*/ (int32_t) acc->ne[0],
        /*.ne01 =*/ (int32_t) acc->ne[1],
        /*.ne02 =*/ (int32_t) acc->ne[2],
        /*.ne03 =*/ (int32_t) acc->ne[3],
        /*.nb00 =*/ acc->nb[0],
        /*.nb01 =*/ acc->nb[1],
        /*.nb02 =*/ acc->nb[2],
        /*.nb03 =*/ acc->nb[3],
        /*.ne10 =*/ (int32_t) add->ne[0],
        /*.ne11 =*/ (int32_t) add->ne[1],
        /*.ne12 =*/ (int32_t) add->ne[2],
        /*.ne13 =*/ (int32_t) add->ne[3],
        /*.nb10 =*/ add->nb[0],
        /*.nb11 =*/ add->nb[1],
        /*.nb12 =*/ add->nb[2],
        /*.nb13 =*/ add->nb[3],
        /*.ne0  =*/ (int32_t) acc->ne[0],
        /*.ne1  =*/ (int32_t) acc->ne[1],
        /*.ne2  =*/ (int32_t) acc->ne[2],
        /*.ne3  =*/ (int32_t) acc->ne[3],
        /*.nb0  =*/ acc->nb[0],
        /*.nb1  =*/ acc->nb[1],
        /*.nb2  =*/ acc->nb[2],
        /*.nb3  =*/ acc->nb[3],
        /*.offs =*/ 0,
        /*.o1   =*/ { bid_add.offs },
    };

    // the kernel adds the fused offsets to the src1 base itself
    bid_add.offs = 0;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx);

        ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

        ggml_metal_encoder_set_pipeline(enc, pipeline);
        ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 1);
        ggml_metal_encoder_set_buffer  (enc, bid_add, 2);
        ggml_metal_encoder_set_buffer  (enc, bid_acc, 3);

        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipeline));

        int nth = 1;
        while (2*nth < args.ne0 && nth < nth_max) {
            nth *= 2;
        }

        ggml_metal_encoder_dispatch_threadgroups(enc, args.ne01, args.ne02, args.ne03, nth, 1, 1);

        ggml_metal_encoder_end_encoding(enc);
        ggml_metal_encoder_free(enc);

        ggml_metal_cmd_buf_end(ctx, cmd_buf, false);
    }

    return true;
}

// TOSH_MGPU_PEER_MIN_BATCH, the same knob the copy path reads: below it a crossing is all
// latency and the shared host block is faster.
// rows in the exchanged tensor: one for a generated token, the whole prompt for a prefill
static int64_t tosh_batch_of(const struct ggml_tensor * t) {
    if (t == NULL || t->ne[0] <= 0) { return 1; }
    return ggml_nelements(t)/t->ne[0];
}

static int64_t ggml_metal_peer_min_batch(void) {
    static int64_t v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_PEER_MIN_BATCH");
        v = s ? atoll(s) : 32;
    }
    return v;
}

// Leaves the exchange's command buffer open so whatever the graph encodes next lands in it,
// one submission fewer. On its own this is neutral, which is what the first measurement said;
// it only pays together with the merge that makes the next subgraph continue the held buffer,
// and then only when the batch is small. A token of one row crosses 72 times and each crossing
// carries its own pair of command buffers, so halving them is worth ~3% of decode; a prompt
// amortises the submissions over the whole batch and loses ~2% instead, hence the batch gate.
// TOSH_MGPU_HOLD=1 forces it on, =0 off, unset follows the batch.
static bool ggml_metal_collective_hold_batch(int64_t batch) {
    static int v = -2;
    if (v == -2) {
        const char * s = TOSH_ENV("TOSH_MGPU_HOLD");
        v = s ? (s[0] == '1' ? 1 : 0) : -1;
    }
    if (v >= 0) {
        return v == 1;
    }
    return batch < ggml_metal_peer_min_batch();
}

// Two cards of one Metal peer group can read each other's VRAM, so the collective has no
// reason to bounce its partials through a host block. Decided once per link.
static bool ggml_metal_tp2_peer_eligible(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                         struct ggml_metal_xdev_link * link) {
#if TARGET_OS_OSX
    if (link->peer_x_ok >= 0) {
        return link->peer_x_ok == 1;
    }

    static int use_peer = -1;
    if (use_peer < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_PEER");
        use_peer = (v && v[0] == '1') ? 1 : 0;
    }

    link->peer_x_ok = 0;

    if (use_peer && @available(macOS 10.15, *)) {
        id<MTLDevice> dev_self = ggml_metal_device_get_obj(ctx_self->dev);
        id<MTLDevice> dev_peer = ggml_metal_device_get_obj(ctx_peer->dev);
        const bool same_dev = dev_self == dev_peer && ggml_metal_peer_self();
        const uint64_t group = dev_peer.peerGroupID;

        if (same_dev || (group != 0 && group == dev_self.peerGroupID)) {
            link->peer_x_ok = 1;
        }

        if (!link->peer_x_logged) {
            link->peer_x_logged = true;
            fprintf(stderr, link->peer_x_ok
                        ? "ggml_metal: TP2 peer-direct allreduce enabled for %s <-> %s (peer group %llu)\n"
                        : "ggml_metal: TP2 peer-direct unavailable, using staged allreduce (%s <-> %s, peer group %llu)\n",
                    dev_self.name.UTF8String, dev_peer.name.UTF8String, (unsigned long long) group);
        }
    }

    return link->peer_x_ok == 1;
#else
    GGML_UNUSED(ctx_self);
    GGML_UNUSED(ctx_peer);
    link->peer_x_ok = 0;
    return false;
#endif
}

// A view of the peer's buffer, valid on this card. Rebuilt only when the peer hands over a
// different MTLBuffer, and never while an exchange that reads the old one is still queued.
static id<MTLBuffer> ggml_metal_tp2_peer_view(struct ggml_metal_xdev_link * link,
                                              id<MTLBuffer> metal_peer, id<MTLDevice> dev_self,
                                              bool same_dev) {
#if TARGET_OS_OSX
    if (link->peer_x_src == metal_peer && link->peer_x_view != nil) {
        if (tosh_mgpu_trace()) {
            atomic_fetch_add(&g_tr_rv_hit, 1);
        }
        return link->peer_x_view;
    }

    if (link->peer_x_view != nil && link->seq_x > 0 &&
        ![link->ev_x_done waitUntilSignaledValue:link->seq_x timeoutMS:10000]) {
        return nil;
    }

    [link->peer_x_view release];
    [link->peer_x_src  release];
    link->peer_x_view = same_dev ? [metal_peer retain]
                                 : [metal_peer newRemoteBufferViewForDevice:dev_self];
    link->peer_x_src  = link->peer_x_view != nil ? [metal_peer retain] : nil;

    if (link->peer_x_view != nil && tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_rv_new, 1);
    }

    return link->peer_x_view;
#else
    GGML_UNUSED(link); GGML_UNUSED(metal_peer); GGML_UNUSED(dev_self); GGML_UNUSED(same_dev);
    return nil;
#endif
}

// Each card keeps one buffer holding the partial it publishes. It is stable across rounds, so
// the remote view of it is built once, and the partner reading it never touches the tensor the
// add is about to rewrite: the exchange needs no wait for the peer to finish reading.
static bool ggml_metal_xdev_shadow_reserve(ggml_metal_t ctx_owner, struct ggml_metal_xdev_link * link,
                                           size_t size) {
    if (link->shadow_cap >= size && link->shadow[0] != nil && link->shadow[1] != nil) {
        return true;
    }

    // resizing frees pages a queued exchange may still be reading through a remote view
    if (link->seq_x > 0 && ![link->ev_x_done waitUntilSignaledValue:link->seq_x timeoutMS:10000]) {
        return false;
    }

    id<MTLDevice> dev = ggml_metal_device_get_obj(ctx_owner->dev);
    link->shadow_cap = MAX(size, (size_t) 1*1024*1024);

    for (int k = 0; k < 2; k++) {
        [link->shadow[k] release];
        [link->peer_x_view2[k] release];
        link->peer_x_view2[k] = nil;
        link->shadow[k] = [dev newBufferWithLength:link->shadow_cap options:MTLResourceStorageModePrivate];
        if (link->shadow[k] == nil) {
            link->shadow_cap = 0;
            return false;
        }
    }

    return true;
}

// One view per inbox, built once: the inboxes are stable for the life of the link.
static id<MTLBuffer> ggml_metal_tp2_inbox_view(struct ggml_metal_xdev_link * link, int slot,
                                               id<MTLDevice> dev_self, bool same_dev) {
#if TARGET_OS_OSX
    if (link->peer_x_view2[slot] != nil) {
        if (tosh_mgpu_trace()) {
            atomic_fetch_add(&g_tr_rv_hit, 1);
        }
        return link->peer_x_view2[slot];
    }

    link->peer_x_view2[slot] = same_dev ? [link->shadow[slot] retain]
                                        : [link->shadow[slot] newRemoteBufferViewForDevice:dev_self];
    if (link->peer_x_view2[slot] != nil && tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_rv_new, 1);
    }

    return link->peer_x_view2[slot];
#else
    GGML_UNUSED(link); GGML_UNUSED(slot); GGML_UNUSED(dev_self); GGML_UNUSED(same_dev);
    return nil;
#endif
}

// The fused peer path. One local copy to publish, then an add that reads the partner's copy
// straight out of its VRAM:
//
//   wait   done(out)@prev     the peer has read what I published last round
//   blit   t_self -> shadow   local VRAM, so the tensor is free from here on
//   signal ready(out)         my partial is published
//   wait   ready(in)          the peer's is
//   add    t_self += remote(peer shadow)
//   signal done(in)           I have read it
//
// No host block, no temporary, and no round trip waiting on the peer's reader.
static bool ggml_metal_exchange_reduce_peer_fused(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                                  struct ggml_metal_xdev_link * link_out,
                                                  struct ggml_metal_xdev_link * link_in,
                                                  struct ggml_tensor * t_self,
                                                  struct ggml_tensor * tmp_self,
                                                  bool via_blit, uint64_t seq) {
#if TARGET_OS_OSX
    if (!ggml_metal_tp2_peer_eligible(ctx_self, ctx_peer, link_out)) {
        return false;
    }

    struct ggml_metal_buffer_id bid_self = ggml_metal_get_buffer_id(t_self);
    if (bid_self.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(t_self);
    id<MTLBuffer> metal_self = (id<MTLBuffer>) bid_self.metal;
    if (bid_self.offs > metal_self.length || size > metal_self.length - bid_self.offs) {
        return false;
    }

    // both shadows, because the card encoded first would otherwise find the partner's missing
    if (!ggml_metal_xdev_shadow_reserve(ctx_self, link_out, size) ||
        !ggml_metal_xdev_shadow_reserve(ctx_peer, link_in,  size)) {
        return false;
    }

    id<MTLDevice> dev_self = ggml_metal_device_get_obj(ctx_self->dev);
    id<MTLDevice> dev_peer = ggml_metal_device_get_obj(ctx_peer->dev);
    const bool same_dev = dev_self == dev_peer && ggml_metal_peer_self();

    const int slot = (int) (seq & 1);

    id<MTLBuffer> view = ggml_metal_tp2_inbox_view(link_in, slot, dev_self, same_dev);
    if (view == nil) {
        return false;
    }

    link_out->seq_x = seq;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx_self);

        // the peer read this slot two rounds ago, so by now it is long done and the wait does
        // not stall; with a single inbox the same guard names seq-1 and serialises the pair
        if (seq > 2 && !ggml_metal_collective_nowait()) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
            [cmd_buf encodeWaitForEvent:link_out->ev_x_done value:seq - 2];
        }

        id<MTLBlitCommandEncoder> enc = [cmd_buf blitCommandEncoder];
        [enc copyFromBuffer:metal_self sourceOffset:bid_self.offs
                   toBuffer:link_out->shadow[slot] destinationOffset:0 size:size];
        [enc endEncoding];

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_out->ev_x_ready value:seq];
        if (!ggml_metal_collective_nowait()) {
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:link_in->ev_x_ready value:seq];
        }

        bool ok;
        if (via_blit) {
            // the same transport read by the blit engine instead of the add's threads, which
            // is the one thing that differs between this and the fused variant
            struct ggml_metal_buffer_id bid_tmp = ggml_metal_get_buffer_id(tmp_self);
            id<MTLBlitCommandEncoder> enc_in = [cmd_buf blitCommandEncoder];
            [enc_in copyFromBuffer:view sourceOffset:0
                          toBuffer:bid_tmp.metal destinationOffset:bid_tmp.offs size:size];
            [enc_in endEncoding];
            ok = ggml_metal_add_inplace_into(ctx_self, cmd_buf, t_self, tmp_self);
        } else {
            ok = ggml_metal_add_inplace_buffer(ctx_self, cmd_buf, t_self, view, 0);
        }

        if (!ok) {
            // the buffer already carries the publish and the signal, so it has to go out
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq];
            tosh_trace_cmd_buf(cmd_buf);
            ggml_metal_cmd_buf_end(ctx_self, cmd_buf, false);
            return false;
        }

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq];

        tosh_trace_cmd_buf(cmd_buf);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_reduce, 1);
        atomic_fetch_add(&g_tr_bytes_peer, size);
    }

    return true;
#else
    GGML_UNUSED(ctx_self); GGML_UNUSED(ctx_peer); GGML_UNUSED(link_out);
    GGML_UNUSED(link_in); GGML_UNUSED(t_self); GGML_UNUSED(seq);
    return false;
#endif
}

// Copy a tensor into a plain buffer laid out the same way. Used to publish this card's
// partial into the partner's inbox, which is a remote view: the blit engine refuses one as a
// destination, a compute kernel does not.
static bool ggml_metal_cpy_to_buffer(ggml_metal_t ctx, id<MTLCommandBuffer> cmd_buf,
                                     const struct ggml_tensor * src, id<MTLBuffer> dst, size_t dst_offs) {
    struct ggml_metal_pipeline_with_params pipeline =
        ggml_metal_library_get_pipeline_cpy(ctx->lib, src->type, src->type, false);
    if (pipeline.pipeline == NULL) {
        return false;
    }

    struct ggml_metal_buffer_id bid_src = ggml_metal_get_buffer_id(src);
    struct ggml_metal_buffer_id bid_dst = { .metal = dst, .offs = dst_offs };

    ggml_metal_kargs_cpy args = {
        /*.nk0  =*/ src->ne[0],
        /*.ne00 =*/ src->ne[0],
        /*.ne01 =*/ src->ne[1],
        /*.ne02 =*/ src->ne[2],
        /*.ne03 =*/ src->ne[3],
        /*.nb00 =*/ src->nb[0],
        /*.nb01 =*/ src->nb[1],
        /*.nb02 =*/ src->nb[2],
        /*.nb03 =*/ src->nb[3],
        /*.ne0  =*/ src->ne[0],
        /*.ne1  =*/ src->ne[1],
        /*.ne2  =*/ src->ne[2],
        /*.ne3  =*/ src->ne[3],
        /*.nb0  =*/ src->nb[0],
        /*.nb1  =*/ src->nb[1],
        /*.nb2  =*/ src->nb[2],
        /*.nb3  =*/ src->nb[3],
    };

    ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

    ggml_metal_encoder_set_pipeline(enc, pipeline);
    ggml_metal_encoder_set_bytes   (enc, &args, sizeof(args), 0);
    ggml_metal_encoder_set_buffer  (enc, bid_src, 1);
    ggml_metal_encoder_set_buffer  (enc, bid_dst, 2);

    const int nth = MIN((int64_t) ggml_metal_pipeline_max_theads_per_threadgroup(pipeline), src->ne[0]);

    ggml_metal_encoder_dispatch_threadgroups(enc, args.ne01, args.ne02, args.ne03, nth, 1, 1);

    ggml_metal_encoder_end_encoding(enc);
    ggml_metal_encoder_free(enc);

    return true;
}

// The push. A fabric read stalls on latency; a fabric write is posted and the card carries on.
// The blit engine will not write through a remote view ("RemoteView supports read-only
// operation and cannot be used as Destination"), but a compute kernel will, so the publish is
// a cpy kernel and every read the add makes is local.
//
//   wait   done(out)@prev     the peer has consumed what I pushed last round
//   cpy    t_self -> remote(peer's inbox)
//   signal ready(out)         it has landed
//   wait   ready(in)          the peer's push has landed in mine
//   add    t_self += my inbox (local)
//   signal done(in)           consumed
static bool ggml_metal_exchange_reduce_peer_push(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                                 struct ggml_metal_xdev_link * link_out,
                                                 struct ggml_metal_xdev_link * link_in,
                                                 struct ggml_tensor * t_self,
                                                 uint64_t seq) {
#if TARGET_OS_OSX
    if (!ggml_metal_tp2_peer_eligible(ctx_self, ctx_peer, link_out)) {
        return false;
    }

    struct ggml_metal_buffer_id bid_self = ggml_metal_get_buffer_id(t_self);
    if (bid_self.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(t_self);

    // both kernels resolved up front: a half-encoded buffer would have to be submitted, and the
    // caller would then reduce the same tensor a second time through the staged path
    if (ggml_metal_library_get_pipeline_cpy(ctx_self->lib, t_self->type, t_self->type, false).pipeline == NULL ||
        ggml_metal_library_get_pipeline_bin_one(ctx_self->lib, GGML_OP_ADD).pipeline == NULL) {
        return false;
    }

    // an inbox on each card, owned by the card that reads it
    if (!ggml_metal_xdev_shadow_reserve(ctx_peer, link_out, size) ||
        !ggml_metal_xdev_shadow_reserve(ctx_self, link_in,  size)) {
        return false;
    }

    id<MTLDevice> dev_self = ggml_metal_device_get_obj(ctx_self->dev);
    id<MTLDevice> dev_peer = ggml_metal_device_get_obj(ctx_peer->dev);
    const bool same_dev = dev_self == dev_peer && ggml_metal_peer_self();

    const int slot = (int) (seq & 1);

    id<MTLBuffer> view = ggml_metal_tp2_inbox_view(link_out, slot, dev_self, same_dev);
    if (view == nil) {
        return false;
    }

    link_out->seq_x = seq;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx_self);

        if (seq > 2 && !ggml_metal_collective_nowait()) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
            [cmd_buf encodeWaitForEvent:link_out->ev_x_done value:seq - 2];
        }

        ggml_metal_cpy_to_buffer(ctx_self, cmd_buf, t_self, view, 0);

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_out->ev_x_ready value:seq];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:link_in->ev_x_ready value:seq];

        ggml_metal_add_inplace_buffer(ctx_self, cmd_buf, t_self, link_in->shadow[slot], 0);

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq];
        tosh_trace_cmd_buf(cmd_buf);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_reduce, 1);
        atomic_fetch_add(&g_tr_bytes_peer, size);
    }

    return true;
#else
    GGML_UNUSED(ctx_self); GGML_UNUSED(ctx_peer); GGML_UNUSED(link_out);
    GGML_UNUSED(link_in); GGML_UNUSED(t_self); GGML_UNUSED(seq);
    return false;
#endif
}

// The peer-direct half of the exchange. Nothing is encoded until every check has passed, so a
// refusal here costs the caller only the staged path it would have taken anyway.
//
//   signal ready(out)          my partial is final, the peer may read it
//   wait   ready(in)           the peer's partial is final
//   blit   remote(t_peer) -> tmp        one crossing, GPU to GPU
//   signal done(in)            I have read the peer
//   wait   done(out)           the peer has read me, so the add may overwrite my partial
//   add    t_self += tmp
//
// That last wait is the whole write-after-read hazard: without it the add rewrites the very
// bytes the other card is still pulling across the fabric.
static bool ggml_metal_exchange_reduce_peer(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                            struct ggml_metal_xdev_link * link_out,
                                            struct ggml_metal_xdev_link * link_in,
                                            struct ggml_tensor * t_self,
                                            struct ggml_tensor * t_peer,
                                            struct ggml_tensor * tmp_self,
                                            uint64_t seq) {
#if TARGET_OS_OSX
    if (t_peer == NULL) {
        return false;
    }
    if (!ggml_metal_tp2_peer_eligible(ctx_self, ctx_peer, link_out)) {
        return false;
    }

    struct ggml_metal_buffer_id bid_self = ggml_metal_get_buffer_id(t_self);
    struct ggml_metal_buffer_id bid_peer = ggml_metal_get_buffer_id(t_peer);
    struct ggml_metal_buffer_id bid_tmp  = ggml_metal_get_buffer_id(tmp_self);
    if (bid_self.metal == nil || bid_peer.metal == nil || bid_tmp.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(t_self);
    if (ggml_nbytes(t_peer) != size) {
        return false;
    }

    id<MTLBuffer> metal_peer = (id<MTLBuffer>) bid_peer.metal;
    id<MTLBuffer> metal_tmp  = (id<MTLBuffer>) bid_tmp.metal;
    if (bid_peer.offs > metal_peer.length || size > metal_peer.length - bid_peer.offs ||
        bid_tmp.offs  > metal_tmp.length  || size > metal_tmp.length  - bid_tmp.offs) {
        return false;
    }

    if (!ggml_metal_add_inplace_supported(ctx_self, t_self, tmp_self)) {
        return false;
    }

    id<MTLDevice> dev_self = ggml_metal_device_get_obj(ctx_self->dev);
    id<MTLDevice> dev_peer = ggml_metal_device_get_obj(ctx_peer->dev);
    const bool same_dev = dev_self == dev_peer && ggml_metal_peer_self();

    // the view lives on the link the peer's data travels in on, which is the one the wait
    // below names, so the two cannot get out of step
    id<MTLBuffer> view = ggml_metal_tp2_peer_view(link_in, metal_peer, dev_self, same_dev);
    if (view == nil) {
        return false;
    }

    link_out->seq_x = seq;

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx_self);

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_out->ev_x_ready value:seq];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:link_in->ev_x_ready value:seq];

        id<MTLBlitCommandEncoder> enc = [cmd_buf blitCommandEncoder];
        [enc copyFromBuffer:view sourceOffset:bid_peer.offs
                   toBuffer:metal_tmp destinationOffset:bid_tmp.offs size:size];
        [enc endEncoding];

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq];
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        [cmd_buf encodeWaitForEvent:link_out->ev_x_done value:seq];

        ggml_metal_add_inplace_into(ctx_self, cmd_buf, t_self, tmp_self);

        tosh_trace_cmd_buf(cmd_buf);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_reduce, 1);
        atomic_fetch_add(&g_tr_bytes_peer, size);
    }

    return true;
#else
    GGML_UNUSED(ctx_self); GGML_UNUSED(ctx_peer); GGML_UNUSED(link_out); GGML_UNUSED(link_in);
    GGML_UNUSED(t_self); GGML_UNUSED(t_peer); GGML_UNUSED(tmp_self); GGML_UNUSED(seq);
    return false;
#endif
}

// TOSH_MGPU_ONESHOT=1 engages the one-shot round for decode batches. Default off: at n=4 the
// measurements say the staged butterfly wins, 38.0 vs 35.8 t/s. The one-shot's publish is one
// round instead of two, but every card then waits on every partner in the same held command
// buffer, so each queue stalls on the slowest of the four every collective: NOWAIT decode is
// 45.7 against 35.8 with the waits in. The butterfly's pairwise rounds stall far less.
static bool ggml_metal_oneshot_enabled(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_ONESHOT");
        v = (s && s[0] == '1') ? 1 : 0;
    }
    return v == 1;
}

// The one-shot all-reduce: a single round for the whole group instead of the butterfly.
// Every card publishes its partial once into the host block it shares with each partner,
// then pulls every partner's partial once and sums locally:
//
//   wait   done(out,j)@prev    partner j finished reading this block last round
//   blit   t_self -> wrap_src(j)        local PCIe write, one per partner
//   signal ready(out,j)        the partial is published
//   wait   ready(in,j)         partner j's partial is published
//   add    t_self += wrap_dst(j)         host read in the add
//   signal done(in,j)          it has been read
//
// Why the host block when the cards can read each other's VRAM: compute-initiated accesses
// through remote buffer views measure ~600 us for 16 KiB on this stack, one-way either
// direction, so a peer-direct collective is all latency with none of the bandwidth. The host
// block is the medium the staged path already proves fast at decode sizes, and the fused
// boundary from 0065 reads it inside the add. The win over the butterfly is one round instead
// of log2(n), each card touching the block once out and once in for the whole reduction.
// Nothing is encoded until every link is reserved and every pipeline resolved, so a refusal
// leaves the caller's butterfly exactly as it was. seq comes from the counter the staged
// exchanges use: links may skip values but never go backwards.
// A tensor no device computed (an empty split slice carries the node but no COMPUTE flag, so
// its memory is stale from the scratch pool). The generic butterfly zeroes those before the
// add; the metal collectives need the same and get it with a blit fill on the owning device.
// Queued on the device's own buffer chain, so it lands before anything the collective
// publishes from this device.
bool ggml_metal_zero_tensor(ggml_metal_t ctx, struct ggml_tensor * t) {
    struct ggml_metal_buffer_id bid = ggml_metal_get_buffer_id(t);
    if (bid.metal == nil) {
        return false;
    }
    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx);
        id<MTLBlitCommandEncoder> enc = [cmd_buf blitCommandEncoder];
        [enc fillBuffer:bid.metal range:NSMakeRange(bid.offs, ggml_nbytes(t)) value:0];
        [enc endEncoding];
        ggml_metal_cmd_buf_end(ctx, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t)));
    }
    return true;
}

bool ggml_metal_allreduce_oneshot(ggml_metal_t * ctxs, struct ggml_tensor ** tensors,
                                  size_t n, uint64_t seq, uint32_t zero_mask) {
#if TARGET_OS_OSX
    // xlinks[4] leaves room for three partners per card; beyond that the butterfly's
    // folding takes over and this schedule does not fit the link tables. At n=2 it is a
    // strict loss: the butterfly is already one round and every extra event op costs
    // (49 t/s one-shot vs 61.6 staged), so the two-card case stays on the staged path.
    if (n < 3 || n > 4 || !ggml_metal_oneshot_enabled()) {
        if (tosh_mgpu_trace()) fprintf(stderr, "oneshot: bail n=%zu enabled=%d\n", n, ggml_metal_oneshot_enabled());
        return false;
    }

    const size_t size = ggml_nbytes(tensors[0]);
    const size_t slot = (seq & 1) * size;   // two slots side by side, alternating by round
    const int    sp   = (int) (seq & 1);

    struct ggml_metal_xdev_link * out_l[4][4]; // out_l[i][j]: card i's publish channel to card j
    struct ggml_metal_xdev_link * in_l[4][4];  // in_l[i][j]:  card j's publish channel to card i
    uint64_t prev[4][4];                       // last seq that wrote this slot on that link, read before any card re-stamps it

    // resolved in the pre-pass so the encode phase cannot fail halfway through the group
    struct ggml_metal_pipeline_with_params pipelines[4];
    ggml_metal_kargs_bin args[4];

    for (size_t i = 0; i < n; i++) {
        ggml_metal_t ctx_i = ctxs[i];

        if (ggml_metal_get_buffer_id(tensors[i]).metal == nil ||
            ggml_nbytes(tensors[i]) != size ||
            ggml_metal_library_get_pipeline_bin_one(ctx_i->lib, GGML_OP_ADD).pipeline == NULL) {
            if (tosh_mgpu_trace()) fprintf(stderr, "oneshot: bail basic i=%zu n=%zu size=%zu\n", i, n, size);
            return false;
        }

        // the same pipeline resolution add_inplace_buffer does, hoisted so the args can be
        // built once: every partner's publish sits at the same slot, so only the buffer bound
        // at index 2 differs between the n-1 dispatches of one card
        const bool rb4 = ggml_metal_fb_add_rb() && ggml_nrows(tensors[i]) == 1 && tensors[i]->ne[0] % 4 == 0 &&
                         ggml_is_contiguous(tensors[i]) && ggml_nelements(tensors[i]) < 65536;
        struct ggml_metal_pipeline_with_params pl = rb4 ? ggml_metal_pipeline_add_rb4(ctx_i->lib)
                                                        : ggml_metal_library_get_pipeline_bin_one(ctx_i->lib, GGML_OP_ADD);
        if (pl.pipeline == NULL) {
            pl = ggml_metal_library_get_pipeline_bin_one(ctx_i->lib, GGML_OP_ADD);
            if (pl.pipeline == NULL) {
                return false;
            }
        }
        pipelines[i] = pl;

        const struct ggml_tensor * acc = tensors[i];
        args[i] = (ggml_metal_kargs_bin) {
            /*.ne00 =*/ (int32_t) acc->ne[0],
            /*.ne01 =*/ (int32_t) acc->ne[1],
            /*.ne02 =*/ (int32_t) acc->ne[2],
            /*.ne03 =*/ (int32_t) acc->ne[3],
            /*.nb00 =*/ acc->nb[0],
            /*.nb01 =*/ acc->nb[1],
            /*.nb02 =*/ acc->nb[2],
            /*.nb03 =*/ acc->nb[3],
            /*.ne10 =*/ (int32_t) acc->ne[0],
            /*.ne11 =*/ (int32_t) acc->ne[1],
            /*.ne12 =*/ (int32_t) acc->ne[2],
            /*.ne13 =*/ (int32_t) acc->ne[3],
            /*.nb10 =*/ acc->nb[0],
            /*.nb11 =*/ acc->nb[1],
            /*.nb12 =*/ acc->nb[2],
            /*.nb13 =*/ acc->nb[3],
            /*.ne0  =*/ (int32_t) acc->ne[0],
            /*.ne1  =*/ (int32_t) acc->ne[1],
            /*.ne2  =*/ (int32_t) acc->ne[2],
            /*.ne3  =*/ (int32_t) acc->ne[3],
            /*.nb0  =*/ acc->nb[0],
            /*.nb1  =*/ acc->nb[1],
            /*.nb2  =*/ acc->nb[2],
            /*.nb3  =*/ acc->nb[3],
            /*.offs =*/ 0,
            /*.o1   =*/ { slot },
        };
        if (pl.c4) {
            args[i].ne00 /= 4;
            args[i].ne10 /= 4;
            args[i].ne0  /= 4;
        }

        for (size_t j = 0; j < n; j++) {
            if (j == i) {
                continue;
            }
            ggml_metal_t ctx_j = ctxs[j];

            out_l[i][j] = ggml_metal_xdev_link_get(ctx_i, ctx_j);
            in_l[i][j]  = ggml_metal_xdev_link_get(ctx_j, ctx_i);
            if (out_l[i][j] == NULL || in_l[i][j] == NULL) {
                if (tosh_mgpu_trace()) fprintf(stderr, "oneshot: bail link i=%zu j=%zu\n", i, j);
                return false;
            }
            prev[i][j] = out_l[i][j]->x_slot[sp];

            // one host block per link, both devices' views of it, reserved up front so no
            // card can fail on allocation after a partner has already committed.
            // the in-direction is a different directed link and would otherwise be checked
            // (here) before it is reserved (at iteration j,i), so reserve it now too
            if (!ggml_metal_xdev_link_reserve(ctx_i, ctx_j, out_l[i][j], 2 * size) ||
                !ggml_metal_xdev_link_reserve(ctx_j, ctx_i, in_l[i][j], 2 * size)) {
                if (tosh_mgpu_trace()) fprintf(stderr, "oneshot: bail reserve i=%zu j=%zu\n", i, j);
                return false;
            }
            if (out_l[i][j]->cap < 2 * size || in_l[i][j]->cap < 2 * size) {
                if (tosh_mgpu_trace()) fprintf(stderr, "oneshot: bail cap i=%zu j=%zu cap=%zu/%zu need=%zu\n",
                                               i, j, out_l[i][j]->cap, in_l[i][j]->cap, 2*size);
                return false;
            }
        }
    }

    static bool logged = false;
    if (!logged) {
        logged = true;
        fprintf(stderr, "ggml_metal: one-shot all-reduce enabled for %zu-card groups\n", n);
    }

    for (size_t i = 0; i < n; i++) {
        ggml_metal_t ctx_i = ctxs[i];
        struct ggml_tensor * t_self = tensors[i];
        const uint64_t t_enter = tosh_mgpu_trace() ? tosh_now_ns() : 0;

        @autoreleasepool {
            id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx_i);

            // back-pressure names the last round this very link was used at, the same
            // protocol the staged exchange uses: seq comes from a group-wide counter and a
            // link only sees a fraction of the values, so a fixed seq-2 could name a round
            // that link never signalled. prev[] was captured before any card re-stamped it.
            if (!ggml_metal_collective_nowait()) {
                for (size_t j = 0; j < n; j++) {
                    if (j == i) {
                        continue;
                    }
                    if (prev[i][j] == 0) {
                        continue;
                    }
                    if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
                    if (ggml_metal_bp_satisfied()) {
                        [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_i) value:1];
                    } else {
                        [cmd_buf encodeWaitForEvent:out_l[i][j]->ev_x_done value:prev[i][j]];
                    }
                }
            }

            // publish before waiting: every card signals every partner first, so no subset
            // of the group can sit on an event the rest are waiting behind.
            // one blit encoder for all partners, one compute encoder for all adds; encoder
            // switches are a known per-dispatch cost on this stack and the staged round
            // spends only ~2 dispatches, so paying them n-1 times was the regression
            struct ggml_metal_buffer_id bid_self = ggml_metal_get_buffer_id(t_self);

            {
                id<MTLBlitCommandEncoder> enc_blit = [cmd_buf blitCommandEncoder];
                // an empty-slice partial joins the sum as zeros, filled in this same encoder
                if (zero_mask & (1u << i)) {
                    [enc_blit fillBuffer:bid_self.metal range:NSMakeRange(bid_self.offs, size) value:0];
                }
                for (size_t j = 0; j < n; j++) {
                    if (j == i) {
                        continue;
                    }
                    [enc_blit copyFromBuffer:bid_self.metal sourceOffset:bid_self.offs
                                    toBuffer:out_l[i][j]->wrap_src destinationOffset:slot size:size];
                }
                [enc_blit endEncoding];
            }
            for (size_t j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
                [cmd_buf encodeSignalEvent:out_l[i][j]->ev_x_ready value:seq];
            }
            for (size_t j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
                if (ggml_metal_collective_nowait() || ggml_metal_wait_satisfied()) {
                    [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_i) value:1];
                } else {
                    [cmd_buf encodeWaitForEvent:in_l[i][j]->ev_x_ready value:seq];
                }
            }

            {
                ggml_metal_encoder_t enc = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);

                ggml_metal_encoder_set_pipeline(enc, pipelines[i]);
                ggml_metal_encoder_set_bytes   (enc, &args[i], sizeof(args[i]), 0);
                ggml_metal_encoder_set_buffer  (enc, bid_self, 1);
                ggml_metal_encoder_set_buffer  (enc, bid_self, 3);

                for (size_t j = 0; j < n; j++) {
                    if (j == i) {
                        continue;
                    }
                    struct ggml_metal_buffer_id bid_add = { .metal = in_l[i][j]->wrap_dst, .offs = 0 };
                    ggml_metal_encoder_set_buffer(enc, bid_add, 2);

                    if (pipelines[i].cnt) {
                        ggml_metal_encoder_dispatch_threadgroups(enc, args[i].ne0, (int) ggml_nrows(t_self), 1, 1, 1, 1);
                    } else {
                        const int nth_max = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipelines[i]));

                        int nth = 1;
                        while (2*nth < args[i].ne0 && nth < nth_max) {
                            nth *= 2;
                        }

                        ggml_metal_encoder_dispatch_threadgroups(enc, args[i].ne01, args[i].ne02, args[i].ne03, nth, 1, 1);
                    }
                }

                ggml_metal_encoder_end_encoding(enc);
                ggml_metal_encoder_free(enc);
            }
            for (size_t j = 0; j < n; j++) {
                if (j == i) {
                    continue;
                }
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
                [cmd_buf encodeSignalEvent:in_l[i][j]->ev_x_done value:seq];
            }

    // stamp every link only after every card has encoded: all four contexts live in one
    // address space, so an early stamp would make a later card wait on a value its partner
    // is about to signal behind that very wait
    for (size_t i = 0; i < n; i++) {
        for (size_t j = 0; j < n; j++) {
            if (j != i) {
                out_l[i][j]->seq_x   = seq;
                out_l[i][j]->x_slot[sp] = seq;
            }
        }
    }

            tosh_trace_cmd_buf(cmd_buf);
            tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_i->dev) peerIndex]) & 1));
            ggml_metal_cmd_buf_end(ctx_i, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
        }

        if (tosh_mgpu_trace()) {
            atomic_fetch_add(&g_tr_reduce, 1);
            atomic_fetch_add(&g_tr_bytes_host, 2 * size * (n - 1));
            atomic_fetch_add(&g_tr_encode_ns, tosh_now_ns() - t_enter);
        }
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_oneshot, 1);
    }

    return true;
#else
    GGML_UNUSED(ctxs); GGML_UNUSED(tensors); GGML_UNUSED(n); GGML_UNUSED(seq); GGML_UNUSED(zero_mask);
    return false;
#endif
}

// A buffer per copy is most of what a token creates, and creating one contends on Metal's
// shared pools. Every card signals before it waits, so no pair can lock up.
bool ggml_metal_xdev_prepare(ggml_metal_t ctx_self, ggml_metal_t ctx_peer, size_t size) {
    struct ggml_metal_xdev_link * link_out = ggml_metal_xdev_link_get(ctx_self, ctx_peer);
    struct ggml_metal_xdev_link * link_in  = ggml_metal_xdev_link_get(ctx_peer, ctx_self);
    if (link_out == NULL || link_in == NULL) {
        return false;
    }
    return ggml_metal_xdev_link_reserve(ctx_self, ctx_peer, link_out, size) &&
           ggml_metal_xdev_link_reserve(ctx_peer, ctx_self, link_in,  size);
}

bool ggml_metal_exchange_reduce(ggml_metal_t ctx_self, ggml_metal_t ctx_peer,
                                struct ggml_tensor * t_self, struct ggml_tensor * t_peer,
                                struct ggml_tensor * tmp_self, uint64_t seq) {
    struct ggml_metal_xdev_link * link_out = ggml_metal_xdev_link_get(ctx_self, ctx_peer);
    struct ggml_metal_xdev_link * link_in  = ggml_metal_xdev_link_get(ctx_peer, ctx_self);
    if (link_out == NULL || link_in == NULL) {
        return false;
    }

    if (tosh_mgpu_trace()) {
        tosh_decode_snapshot(t_self->ne[0] > 0 ? ggml_nelements(t_self)/t_self->ne[0] : 1);
    }

    // 0 the tensor-to-tensor peer path, 1 the add pulls the partner across the fabric,
    // 2 a blit pulls it into a local temporary, 3 push into the partner and add locally.
    // 1 is the default: on the Vega II Duo 3 ties it on prefill and is a shade behind on
    // decode, and it costs a second kernel.
    static int fused = -1;
    if (fused < 0) {
        const char * v = TOSH_ENV("TOSH_MGPU_PEER_FUSED");
        fused = v == NULL ? 1 : atoi(v);
    }

    // One row per crossing is the shape the fabric is worst at: the crossing is latency, not
    // bytes, and the shared host block wins. Same threshold the copy path uses. Below it the
    // peer attempt is skipped, not failed: failing here would drop the whole allreduce to the
    // meta layer's generic butterfly, which is far slower than the staged path below.
    const int64_t batch = t_self->ne[0] > 0 ? ggml_nelements(t_self)/t_self->ne[0] : 1;
    const bool try_peer = batch >= ggml_metal_peer_min_batch();

    if (try_peer) {
        if (fused == 3 && ggml_metal_exchange_reduce_peer_push(ctx_self, ctx_peer, link_out, link_in,
                                                               t_self, seq)) {
            return true;
        }

        if (fused > 0 && fused < 3 &&
            ggml_metal_exchange_reduce_peer_fused(ctx_self, ctx_peer, link_out, link_in,
                                                  t_self, tmp_self, fused == 2, seq)) {
            return true;
        }

        if (fused == 0 && ggml_metal_exchange_reduce_peer(ctx_self, ctx_peer, link_out, link_in,
                                                          t_self, t_peer, tmp_self, seq)) {
            return true;
        }
    }

    struct ggml_metal_buffer_id bid_self = ggml_metal_get_buffer_id(t_self);
    struct ggml_metal_buffer_id bid_tmp  = ggml_metal_get_buffer_id(tmp_self);
    if (bid_self.metal == nil || bid_tmp.metal == nil) {
        return false;
    }

    const size_t size = ggml_nbytes(t_self);
    if (!ggml_metal_xdev_link_reserve(ctx_self, ctx_peer, link_out, size) ||
        !ggml_metal_xdev_link_reserve(ctx_peer, ctx_self, link_in,  size)) {
        return false;
    }

    if (!ggml_metal_add_inplace_supported(ctx_self, tmp_self, t_self)) {
        return false;
    }

    const uint64_t t_enter = tosh_mgpu_trace() ? tosh_now_ns() : 0;

    // The add reads the shared block in place of a temporary: same element type, same shape,
    // contiguous, and the block is a page-aligned host allocation at least this long.
    const bool fused_bnd = ggml_metal_tp2_fused_boundary(batch) && !ggml_metal_ctime() &&
                           link_in->wrap_dst != nil &&
                           [link_in->wrap_dst length] >= size &&
                           t_self->type == GGML_TYPE_F32 &&
                           ggml_is_contiguous(t_self) &&
                           ggml_metal_library_get_pipeline_bin_one(ctx_self->lib, GGML_OP_ADD).pipeline != NULL;

    // Both sides are handed the same number: reading it off the link races with whichever
    // card the loop reached first. Links skip values, so the back-pressure wait names its own.
    // its own events and counter: the copies number per link and this path numbers by round,
    // and one event cannot carry both without a value landing below the last one signalled
    const uint64_t seq_prev = link_out->seq_x;
    const uint64_t seq_out  = seq;
    const uint64_t seq_in   = seq;
    link_out->seq_x = seq;
    // the staged exchange writes offset 0, which is the one-shot's even slot, and a prefill
    // payload can span the odd one too. both slot trackers have to know a publish landed here
    const bool fb_res = fused_bnd && ggml_metal_fb_residual() && ggml_nrows(t_self) <= 2 &&
                        2*size <= [link_out->wrap_src length] && 2*size <= [link_in->wrap_dst length];
    const int    res_slot = (int) (seq & 1);
    const size_t res_off  = fb_res && res_slot ? [link_out->wrap_src length]/2 : 0;
    const uint64_t bp_seq = fb_res ? link_out->x_slot[res_slot] : seq_prev;
    if (fb_res) {
        link_out->x_slot[res_slot] = seq;
    } else {
        link_out->x_slot[0] = seq;
        link_out->x_slot[1] = seq;
    }

    @autoreleasepool {
        id<MTLCommandBuffer> cmd_buf = ggml_metal_cmd_buf_continue(ctx_self);

        if (bp_seq > 0) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
            if (ggml_metal_bp_satisfied()) {
                [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_self) value:1];
            } else {
                [cmd_buf encodeWaitForEvent:link_out->ev_x_done value:bp_seq];
            }
        }

        id<MTLCounterSampleBuffer> cts = tosh_ct_buffer(ctx_self);
        uint64_t cbase = 0;
        const bool cbar = ggml_metal_ctime_barrier();
        if (cts != nil) {
            cbase = (atomic_fetch_add(&ctx_self->ct_idx, 1) % TOSH_CT_SLOTS)*TOSH_CT_PER;
        }

        struct ggml_metal_pipeline_with_params pipe_cpy = { 0 };
        if (fused_bnd) {
            pipe_cpy = ggml_metal_library_get_pipeline_cpy(ctx_self->lib, GGML_TYPE_F32, GGML_TYPE_F32, true);
        }
        if (pipe_cpy.pipeline != NULL) {
            const int64_t n = ggml_nelements(t_self);
            ggml_metal_kargs_cpy args = {
                /*.nk0  =*/ n,
                /*.ne00 =*/ t_self->ne[0], /*.ne01 =*/ t_self->ne[1], /*.ne02 =*/ t_self->ne[2], /*.ne03 =*/ t_self->ne[3],
                /*.nb00 =*/ t_self->nb[0], /*.nb01 =*/ t_self->nb[1], /*.nb02 =*/ t_self->nb[2], /*.nb03 =*/ t_self->nb[3],
                /*.ne0  =*/ t_self->ne[0], /*.ne1  =*/ t_self->ne[1], /*.ne2  =*/ t_self->ne[2], /*.ne3  =*/ t_self->ne[3],
                /*.nb0  =*/ t_self->nb[0], /*.nb1  =*/ t_self->nb[1], /*.nb2  =*/ t_self->nb[2], /*.nb3  =*/ t_self->nb[3],
            };
            const int nth = MIN(256, ggml_metal_pipeline_max_theads_per_threadgroup(pipe_cpy));
            struct ggml_metal_buffer_id bid_blk = { .metal = link_out->wrap_src, .offs = res_off };
            ggml_metal_encoder_t ec = ggml_metal_encoder_init((ggml_metal_cmd_buf_t) cmd_buf, false);
            ggml_metal_encoder_set_pipeline(ec, pipe_cpy);
            ggml_metal_encoder_set_bytes   (ec, &args, sizeof(args), 0);
            ggml_metal_encoder_set_buffer  (ec, bid_self, 1);
            ggml_metal_encoder_set_buffer  (ec, bid_blk,  2);
            ggml_metal_encoder_dispatch_threadgroups(ec, (n + nth - 1)/nth, 1, 1, nth, 1, 1);
            ggml_metal_encoder_end_encoding(ec);
            ggml_metal_encoder_free(ec);
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_fused_bnd, 1);
        } else {
            id<MTLBlitCommandEncoder> enc_out = [cmd_buf blitCommandEncoder];
            if (cts != nil) { [enc_out sampleCountersInBuffer:cts atSampleIndex:cbase + 0 withBarrier:cbar]; }
            [enc_out copyFromBuffer:bid_self.metal sourceOffset:bid_self.offs
                           toBuffer:link_out->wrap_src destinationOffset:0 size:size];
            if (cts != nil) { [enc_out sampleCountersInBuffer:cts atSampleIndex:cbase + 1 withBarrier:cbar]; }
            [enc_out endEncoding];
        }
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
        [cmd_buf encodeSignalEvent:link_out->ev_x_ready value:seq_out];

        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_waits, 1);
        if (ggml_metal_wait_satisfied()) {
            [cmd_buf encodeWaitForEvent:tosh_wait_dummy(ctx_self) value:1];
        } else {
            [cmd_buf encodeWaitForEvent:link_in->ev_x_ready value:seq_in];
        }
        if (fused_bnd) {
            // the partner may reuse the block the moment done is signalled, so it goes after
            // the add that reads the block
            if (fb_res) {
                struct ggml_metal_fb_residual_state * st = ggml_metal_fb_residual_get(ctx_self->dev);
                GGML_ASSERT(st != NULL && st->state == 0);
                st->t       = t_self;
                st->buf     = link_in->wrap_dst;
                st->off     = res_off;
                st->ev_done = link_in->ev_x_done;
                st->seq     = seq_in;
                st->state   = 1;
            } else {
                const bool added = ggml_metal_add_inplace_buffer(ctx_self, cmd_buf, t_self, link_in->wrap_dst, 0);
                GGML_ASSERT(added);
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
                [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq_in];
            }
        } else {
            id<MTLBlitCommandEncoder> enc_in = [cmd_buf blitCommandEncoder];
            if (cts != nil) { [enc_in sampleCountersInBuffer:cts atSampleIndex:cbase + 2 withBarrier:cbar]; }
            [enc_in copyFromBuffer:link_in->wrap_dst sourceOffset:0
                          toBuffer:bid_tmp.metal destinationOffset:bid_tmp.offs size:size];
            if (cts != nil) { [enc_in sampleCountersInBuffer:cts atSampleIndex:cbase + 3 withBarrier:cbar]; }
            [enc_in endEncoding];
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_signals, 1);
            [cmd_buf encodeSignalEvent:link_in->ev_x_done value:seq_in];

            ggml_metal_add_inplace_into(ctx_self, cmd_buf, t_self, tmp_self);
        }

        if (cts != nil) {
            id<MTLCounterSampleBuffer> csr = cts;
            const uint64_t cb2 = cbase;
            [cmd_buf addCompletedHandler:^(id<MTLCommandBuffer> cb) { (void) cb; tosh_ct_record(csr, cb2); }];
        }

        tosh_trace_cmd_buf(cmd_buf);
        tosh_trace_exchange(cmd_buf, seq, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx_self->dev) peerIndex]) & 1));
        ggml_metal_cmd_buf_end(ctx_self, cmd_buf, ggml_metal_collective_hold_batch(tosh_batch_of(t_self)));
    }

    if (tosh_mgpu_trace()) {
        atomic_fetch_add(&g_tr_reduce, 1);
        // out through the shared block and back in: this card touches it twice
        atomic_fetch_add(&g_tr_bytes_host, 2*size);
        atomic_fetch_add(&g_tr_encode_ns, tosh_now_ns() - t_enter);
    }

    return true;
}



// TOSH_MGPU_MERGE=1: a tensor split cuts the graph at every partial node, so a token costs one
// command buffer for each collective plus one for each subgraph that follows it. With this on,
// the collective leaves its buffer open and the next subgraph encodes into it instead of
// submitting its own, which halves the command buffers a split token creates.
// The other thread must finish encoding before the GPU drains the main thread's share, so
// enc*(n - m) <= gpu*m with both per node, measured on the share itself; never below the default.
static int ggml_metal_adapt_n_main(ggml_metal_t ctx, int n_nodes, int n_main) {
    const int prev = ctx->am_prev;
    ctx->am_prev = 0;

    if (prev > 0 && ctx->n_nodes_0 > 0 && ctx->n_nodes_1 > 0 && ctx->am_enc_ns > 0) {
        id<MTLCommandBuffer> cb0 = ctx->cmd_bufs[ctx->n_cb].obj;
        const float enc = 1e-9f*(float) ctx->am_enc_ns/ctx->n_nodes_0;
        // a graph that compiled pipelines while encoding says nothing about the next one
        if (cb0 != nil && [cb0 status] == MTLCommandBufferStatusCompleted && enc < 20e-6f) {
            const float g0  = ([cb0 GPUEndTime] - [cb0 GPUStartTime])/ctx->n_nodes_0;

            int slot = -1;
            for (int i = 0; i < 4 && slot < 0; ++i) {
                if (ctx->am_nodes[i] == prev) slot = i;
            }
            if (slot < 0) {
                // a new size takes the first entry; the oldest one falls off the end
                int n_keep = 3;
                for (int i = 0; i < 4; ++i) {
                    if (ctx->am_nodes[i] == 0) { n_keep = i; break; }
                }
                memmove(&ctx->am_nodes[1], &ctx->am_nodes[0], (size_t) n_keep*sizeof(int));
                memmove(&ctx->am_main[1],  &ctx->am_main[0],  (size_t) n_keep*sizeof(int));
                memmove(&ctx->am_enc[1],   &ctx->am_enc[0],   (size_t) n_keep*sizeof(float));
                memmove(&ctx->am_gpu[1],   &ctx->am_gpu[0],   (size_t) n_keep*sizeof(float));
                slot = 0;
                ctx->am_nodes[0] = prev;
                ctx->am_enc[0]   = enc;
                ctx->am_gpu[0]   = g0;
            }
            ctx->am_enc[slot] += 0.125f*(enc - ctx->am_enc[slot]);
            ctx->am_gpu[slot] += 0.125f*(g0  - ctx->am_gpu[slot]);

            const float e = ctx->am_enc[slot];
            const float g = ctx->am_gpu[slot];
            const int   m = e > 0.0f && g > 0.0f ? (int) ceilf(prev*e/(e + g)) : 0;
            // one node stays on the other thread, or the next graph would have nothing to measure
            ctx->am_main[slot] = MIN(m, prev - 1);
        }
    }

    for (int i = 0; i < 4; ++i) {
        if (ctx->am_nodes[i] == n_nodes && ctx->am_main[i] > n_main) {
            return ctx->am_main[i];
        }
    }
    return n_main;
}

static bool ggml_metal_merge_graph(void) {
    static int v = -1;
    if (v < 0) {
        const char * s = TOSH_ENV("TOSH_MGPU_MERGE");
        // on by default: it is inert unless a collective left a buffer held, which only the
        // batch gate above does, so a prompt never sees it
        v = s ? (s[0] == '1' ? 1 : 0) : 1;
    }
    return v == 1;
}

enum ggml_status ggml_metal_graph_compute(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    if (tosh_mgpu_trace()) { atomic_fetch_add(&g_tr_graphs, 1); }
    // command buffers run in commit order, so a held one has to go in before this graph
    if (!ggml_metal_merge_graph()) {
        ggml_metal_cmd_buf_hold_flush(ctx);
    }

    if (ctx->has_error) {
        GGML_LOG_ERROR("%s: backend is in error state from a previous command buffer failure - recreate the backend to recover\n", __func__);
        return GGML_STATUS_FAILED;
    }

    if (gf->n_nodes == 0) {
        return GGML_STATUS_SUCCESS;
    }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    // mixed layers get their event values in submission order, before any parallel encoding;
    // a dynamic cache also pins the residency table this graph reads
    const bool dmoe_on = tosh_dmoe_enabled();
    const int dmoe_version = dmoe_on ? tosh_dmoe_published_version() : 0;
    bool dmoe_ticked = false;
    for (int i = 0; dmoe_on && i < gf->n_nodes; ++i) {
        struct ggml_tensor * node = gf->nodes[i];
        if (node->op == GGML_OP_TOSH_DMOE_OBSERVE) {
            tosh_dmoe_note_wide(true);
            continue;
        }
        if (node->op != GGML_OP_TOSH_DMOE_PUBLISH) {
            continue;
        }
        tosh_dmoe_note_wide(ggml_get_op_params_i32(node, 1) > 0);
        const uint64_t value = tosh_dmoe_next_value();
        memcpy(&node->op_params[2], &value, sizeof(value));

        ggml_set_op_params_i32(node, 4, dmoe_version);
        if (!dmoe_ticked) {
            tosh_dmoe_tick_next();
            dmoe_ticked = true;
        }

        struct tosh_dmoe_job * job = tosh_dmoe_job_of(value);
        job->version  = dmoe_version;
        job->tick     = (uint32_t) tosh_dmoe_tick();
        job->il       = ggml_get_op_params_i32(node, 0);
        job->n_tokens = (int32_t) node->src[1]->ne[1];
        job->n_used   = (int32_t) node->src[1]->ne[0];
        job->n_embd   = (int32_t) node->src[0]->ne[0];
        __atomic_store_n(&job->value, value, __ATOMIC_RELEASE);
        tosh_dmoe_set_active(true);
    }
#endif

    // TOSH_CB_MAX_NODES=N: command buffers of at most N nodes, so a long graph does not block the display
    static int cb_max_nodes = -1;
    if (cb_max_nodes < 0) {
        const char * v = TOSH_ENV("TOSH_CB_MAX_NODES");
        cb_max_nodes = v ? MAX(0, atoi(v)) : 0;
    }
    if (cb_max_nodes > 0 && gf->n_nodes > cb_max_nodes && ctx->capture_compute <= 0 && !ggml_metal_merge_graph()) {
        @autoreleasepool {
            ctx->gf = gf;
            for (int i0 = 0; i0 < gf->n_nodes; i0 += cb_max_nodes) {
                const int i1 = MIN(i0 + cb_max_nodes, gf->n_nodes);

                id<MTLCommandBuffer> cmd_buf = [ctx->queue commandBufferWithUnretainedReferences];
                ggml_metal_encode_pending_waits(ctx, cmd_buf);

                ggml_metal_op_t ctx_op = ggml_metal_op_init(ctx->dev, cmd_buf, gf, ctx->finfo, i0, i1,
                        ctx->use_concurrency, false, ctx->debug_graph);
                for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
                    const int res = ggml_metal_op_encode(ctx_op, idx);
                    if (res == 0) {
                        break;
                    }
                    idx += res - 1;
                }
                ggml_metal_op_free(ctx_op);

                ggml_metal_cmd_buf_submit(ctx, cmd_buf);
            }
        }
        return GGML_STATUS_SUCCESS;
    }

    // number of nodes encoded by the main thread (empirically determined)
    // the split was tuned on unified memory; TOSH_N_MAIN overrides it to A/B the share here
    static int n_main_env = -2;
    if (n_main_env == -2) {
        const char * v = TOSH_ENV("TOSH_N_MAIN");
        n_main_env = v ? atoi(v) : -1;
    }
    int n_main = n_main_env >= 0 ? n_main_env : MAX(64, 0.1*gf->n_nodes);

    // number of threads in addition to the main thread
    int n_cb = ctx->n_cb;

    // ── RX6800 decode scheduling (ported from Basten7/iRon-Llama-RC2) ──
    // Fold decode-like graphs into a single command buffer: per token the
    // 2-CB scheme costs an extra commit + a dispatch_apply worker handoff,
    // which dominates short graphs. A graph is decode-like when its first
    // MUL_MAT sees a single activation row (ne11 == 1); prefill batches have
    // ne11 > 1 and keep the parallel-encode path. Opt-in only:
    //   GGML_METAL_DECODE_SCHED=1
    //   GGML_METAL_DECODE_STATS=1  (periodic fold counter)
    {
        static bool  sched_init = false;
        static bool  sched_on   = false;
        static bool  stats_on   = false;
        static uint64_t calls = 0, folded = 0;
        if (!sched_init) {
            sched_init = true;
            sched_on   = getenv("GGML_METAL_DECODE_SCHED") != NULL;
            stats_on   = getenv("GGML_METAL_DECODE_STATS")  != NULL;
        }
        if (sched_on) {
            calls++;
            bool decode_like = false;
            for (int i = 0; i < gf->n_nodes; i++) {
                struct ggml_tensor * t = gf->nodes[i];
                if (t->op == GGML_OP_MUL_MAT && t->src[1]) {
                    // first batched matmul reveals the token count of this pass
                    decode_like = (t->src[1]->ne[1] == 1);
                    break;
                }
            }
            if (decode_like) {
                folded++;
                n_cb = 0;   // single command buffer: existing n_cb==0 path below
            }
            if (stats_on && (calls % 512) == 0) {
                GGML_LOG_INFO("%s: decode_sched graphs=%llu folded=%llu\n", __func__,
                              (unsigned long long) calls, (unsigned long long) folded);
            }
        }
    }

    if (n_main_env < 0 && ctx->adapt_n_main) {
        n_main = ggml_metal_adapt_n_main(ctx, gf->n_nodes, n_main);
    }

    // keep the memory wired
    ggml_metal_device_rsets_keep_alive(ctx->dev);

    // submit the ggml compute graph to the GPU by creating command buffers and encoding the ops in them
    // the first n_main nodes are encoded and submitted for processing directly by the calling thread
    // while these nodes are processing, we start n_cb threads to enqueue the rest of the nodes
    // each thread creates it's own command buffer and enqueues the ops in parallel
    //
    // tests on M1 Pro and M2 Ultra using LLaMA models, show that optimal values for n_cb are 1 or 2

    struct tosh_cbp_graph * cbp = NULL;
    uint64_t cbp_t0 = 0;
    if (tosh_cbp_on()) {
        struct tosh_cbp_bucket * b = tosh_cbp_bucket_for(gf->n_nodes);
        if (b != NULL) {
            cbp = (struct tosh_cbp_graph *) calloc(1, sizeof(struct tosh_cbp_graph));
            if (cbp != NULL) {
                cbp->bucket = b;
                atomic_store_explicit(&cbp->n_left, 1, memory_order_relaxed); // released below
                atomic_fetch_add_explicit(&b->n_graphs, 1, memory_order_relaxed);
                cbp_t0 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            }
        }
    }

    @autoreleasepool {
        ctx->gf = gf;

        // the main thread encodes [0, cb_start[0]) and the extra command buffer i encodes [cb_start[i], cb_start[i + 1])
        // a fusion cannot span two command buffers, so each boundary moves to the end of the fusable group it falls
        // into and the encoders fuse the same nodes as with a single command buffer
        {
            const int n_rest   = gf->n_nodes - MIN(n_main, gf->n_nodes);
            const int n_per_cb = n_cb > 0 ? (n_rest + n_cb - 1) / n_cb : 0;

            int i = 0;
            for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
                const int target = MIN(n_main + cb_idx*n_per_cb, gf->n_nodes);
                while (i < target) {
                    i += ggml_metal_fusion_max(gf, i);
                }
                ctx->cb_start[cb_idx] = MIN(i, gf->n_nodes);
            }
            ctx->cb_start[n_cb] = gf->n_nodes;

            ctx->n_cb_graph = n_cb;
            ctx->n_nodes_0  = ctx->cb_start[0];
            ctx->n_nodes_1  = gf->n_nodes - ctx->cb_start[0];
        }

        if (ctx->capture_compute >= 0) {
            ctx->capture_compute--;
        }

        const bool use_capture = ctx->capture_compute == 0;
        if (use_capture) {
            // make sure all previous computations have finished before starting the capture
            if (ctx->cmd_buf_last) {
                [ctx->cmd_buf_last waitUntilCompleted];
                ctx->cmd_buf_last = nil;
            }

            if (!ctx->capture_started) {
                NSString * path = [NSString stringWithFormat:@"/tmp/perf-metal-%d.gputrace", getpid()];

                GGML_LOG_WARN("%s: capturing graph in %s\n", __func__, [path UTF8String]);

                // create capture scope
                id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);
                ctx->capture_scope = [[MTLCaptureManager sharedCaptureManager] newCaptureScopeWithDevice:device];

                MTLCaptureDescriptor * descriptor = [MTLCaptureDescriptor new];
                descriptor.captureObject = ctx->capture_scope;
                descriptor.destination = MTLCaptureDestinationGPUTraceDocument;
                descriptor.outputURL = [NSURL fileURLWithPath:path];

                NSError * error = nil;
                const bool ok = [[MTLCaptureManager sharedCaptureManager] startCaptureWithDescriptor:descriptor error:&error];

                [descriptor release];

                if (!ok) {
                    GGML_LOG_ERROR("%s: error: unable to start capture '%s' (did you set METAL_CAPTURE_ENABLED=1 ?)\n", __func__, [[error localizedDescription] UTF8String]);
                    [ctx->capture_scope release];
                    ctx->capture_scope = nil;
                } else {
                    [ctx->capture_scope beginScope];
                    ctx->capture_started = true;
                }
            }
        }

        // short-hand
        id<MTLCommandQueue> queue = ctx->queue;

        const uint64_t cbp_tc = cbp != NULL ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        // the main thread commits the first few commands immediately
        // cmd_buf[n_cb]
        bool am_measure = ctx->adapt_n_main;
        {
            // continue the collective's buffer when it left one open, otherwise start a new one
            id<MTLCommandBuffer> cmd_buf = nil;
            if (ggml_metal_merge_graph() && ctx->cmd_buf_hold != nil) {
                cmd_buf = ctx->cmd_buf_hold;
                ctx->cmd_buf_hold = nil;
                // this buffer also carries the collective, so its time says nothing about the graph
                am_measure = false;
                // the collective is already encoded in here; anything that queued a wait since
                // then has to land between it and this subgraph, not be dropped
                if (tosh_mgpu_trace()) {
                    if (ctx->n_ev_wait_pending > 0 || ctx->n_peer_waits > 0) {
                        atomic_fetch_add(&g_tr_merge_waits, ctx->n_ev_wait_pending + ctx->n_peer_waits);
                    }
                    atomic_fetch_add(&g_tr_merge_reuse, 1);
                }
            } else {
                if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
                cmd_buf = [queue commandBufferWithUnretainedReferences];
                [cmd_buf retain];
            }
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            tosh_cbp_attach(cmd_buf, cbp);

            if (ctx->cmd_bufs[n_cb].obj) {
                [ctx->cmd_bufs[n_cb].obj release];
            }
            ctx->cmd_bufs[n_cb].obj = cmd_buf;

            tosh_trace_graph_end(cmd_buf, (int) (([(id<MTLDevice>) ggml_metal_device_get_obj(ctx->dev) peerIndex]) & 1));

            [cmd_buf enqueue];

            ggml_metal_fb_residual_before(ctx, cmd_buf, ctx->gf, ctx->n_nodes_0);
            const uint64_t t_enc = am_measure ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
            ctx->encode_async(n_cb);
            if (am_measure) {
                ctx->am_enc_ns = clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - t_enc;
                ctx->am_prev   = ctx->gf->n_nodes;
            }
        }

        // remember the command buffer for the next iteration
        ctx->cmd_buf_last = ctx->cmd_bufs[n_cb].obj;

        // the main thread already encoded every node: the remaining command buffers would be
        // created, dispatched and committed empty. Tensor split hits this on every subgraph.
        const bool encoded_by_main = ctx->n_nodes_1 == 0 && !use_capture;

        if (!encoded_by_main) {
        // prepare the rest of the command buffers asynchronously (optional)
        // cmd_buf[0.. n_cb)
        for (int cb_idx = 0; cb_idx < n_cb; ++cb_idx) {
            if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
            id<MTLCommandBuffer> cmd_buf = [queue commandBufferWithUnretainedReferences];
            ggml_metal_encode_pending_waits(ctx, cmd_buf);
            tosh_cbp_attach(cmd_buf, cbp);
            [cmd_buf retain];

            if (ctx->cmd_bufs[cb_idx].obj) {
                [ctx->cmd_bufs[cb_idx].obj release];
            }
            ctx->cmd_bufs[cb_idx].obj = cmd_buf;

            // always enqueue the first two command buffers
            // enqueue all of the command buffers if we don't need to abort
            if (cb_idx < 2 || ctx->abort_callback == NULL) {
                [cmd_buf enqueue];

                // update the pointer to the last queued command buffer
                // this is needed to implement synchronize()
                ctx->cmd_buf_last = cmd_buf;
            }
        }

        if (cbp != NULL) {
            atomic_fetch_add_explicit(&g_cbp_create_ns,
                    clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_tc, memory_order_relaxed);
        }

        dispatch_apply(n_cb, ctx->d_queue, ctx->encode_async);
        }

        // for debugging: block until graph is computed
        //[ctx->cmd_buf_last waitUntilCompleted];

        // enter here only when capturing in order to wait for all computation to finish
        // otherwise, we leave the graph to compute asynchronously
        if (use_capture && ctx->capture_started) {
            enum ggml_status res = GGML_STATUS_SUCCESS;

            // wait for completion and check status of each command buffer
            // needed to detect if the device ran out-of-memory for example (#1881)
            {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[n_cb].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, n_cb, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }

                    res = GGML_STATUS_FAILED;
                }
            }

            for (int i = 0; i < n_cb && res == GGML_STATUS_SUCCESS; ++i) {
                id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[i].obj;
                [cmd_buf waitUntilCompleted];

                MTLCommandBufferStatus status = [cmd_buf status];
                if (status != MTLCommandBufferStatusCompleted) {
                    GGML_LOG_INFO("%s: command buffer %d failed with status %lu\n", __func__, i, status);
                    if (status == MTLCommandBufferStatusError) {
                        GGML_LOG_INFO("error: %s\n", [[cmd_buf error].localizedDescription UTF8String]);
                    }
                    res = GGML_STATUS_FAILED;
                    break;
                }

                id<MTLCommandBuffer> next_buffer = (i + 1 < n_cb ? ctx->cmd_bufs[i + 1].obj : nil);
                if (!next_buffer) {
                    continue;
                }

                const bool next_queued = ([next_buffer status] != MTLCommandBufferStatusNotEnqueued);
                if (next_queued) {
                    continue;
                }

                if (ctx->abort_callback && ctx->abort_callback(ctx->abort_callback_data)) {
                    GGML_LOG_INFO("%s: command buffer %d aborted", __func__, i);
                    res = GGML_STATUS_ABORTED;
                    break;
                }

                [next_buffer commit];
            }

            [ctx->capture_scope endScope];
            [[MTLCaptureManager sharedCaptureManager] stopCapture];

            [ctx->capture_scope release];
            ctx->capture_scope = nil;

            ctx->capture_started = false;

            if (res != GGML_STATUS_SUCCESS) {
                return res;
            }
        }
    }

    if (cbp != NULL) {
        atomic_fetch_add_explicit(&cbp->bucket->enc_ns,
                clock_gettime_nsec_np(CLOCK_UPTIME_RAW) - cbp_t0, memory_order_relaxed);
        // drop the reference held while encoding; the last completion handler frees it
        if (atomic_fetch_sub_explicit(&cbp->n_left, 1, memory_order_acq_rel) == 1) {
            free(cbp);
        }
    }

    return GGML_STATUS_SUCCESS;
}

// the reorder gains on dense decode but disturbs the expert chain on MoE, so this stays
// narrow on purpose
static bool ggml_metal_graph_is_moe_decode(const struct ggml_cgraph * gf, int max_tokens) {
    bool has_experts = false;

    for (int i = 0; i < gf->n_nodes; i++) {
        const struct ggml_tensor * node = gf->nodes[i];

        // mul_mat_id carries the experts in ne[1] and the tokens in ne[2]
        const int dim = node->op == GGML_OP_MUL_MAT ? 1 : (node->op == GGML_OP_MUL_MAT_ID ? 2 : -1);
        if (dim < 0) {
            continue;
        }

        if (node->src[1] && node->src[1]->ne[dim] > max_tokens) {
            return false;
        }

        has_experts |= node->op == GGML_OP_MUL_MAT_ID;
    }

    return has_experts;
}

void ggml_metal_graph_optimize(ggml_metal_t ctx, struct ggml_cgraph * gf) {
    //const int64_t t_start = ggml_time_us();

    if (ctx->use_graph_optimize) {
        static int max_tokens = -1;
        if (max_tokens < 0) {
            const char * s = TOSH_ENV("GGML_METAL_GRAPH_OPTIMIZE_MIN_BATCH");
            max_tokens = s ? atoi(s) - 1 : 7;
        }

        // a wide batch whose rare experts run on the host is not reordered: moving nodes around its
        // publish produced non-finite activations on small batches (32-63 tokens)
        bool host_rows = false;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        for (int i = 0; i < gf->n_nodes && !host_rows; ++i) {
            host_rows = gf->nodes[i]->op == GGML_OP_TOSH_DMOE_PUBLISH && ggml_get_op_params_i32(gf->nodes[i], 1) > 0;
        }
#endif
        if (!host_rows && (max_tokens < 0 || !ggml_metal_graph_is_moe_decode(gf, max_tokens))) {
            ggml_graph_optimize(gf);
        }
    }

    //printf("%s: graph optimize took %.3f ms\n", __func__, (ggml_time_us() - t_start) / 1000.0);
}

void ggml_metal_event_record(ggml_metal_t ctx, ggml_metal_event_t ev) {
    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);

        ggml_metal_event_encode_signal(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

void ggml_metal_event_wait(ggml_metal_t ctx, ggml_metal_event_t ev) {
    if (ggml_metal_defer_waits() && ctx->n_ev_wait_pending < GGML_METAL_MAX_PENDING_WAITS) {
        ctx->ev_wait_pending[ctx->n_ev_wait_pending++] = ev;
        return;
    }

    @autoreleasepool {
        id<MTLCommandQueue> queue = ctx->queue;
        if (tosh_mgpu_trace()) atomic_fetch_add(&g_tr_cmd_all, 1);
        id<MTLCommandBuffer> cmd_buf = [queue commandBuffer];
        ggml_metal_encode_pending_waits(ctx, cmd_buf);

        ggml_metal_event_encode_wait(ev, cmd_buf);

        [cmd_buf commit];

        [ctx->cmd_bufs_ext addObject:cmd_buf];
        ctx->cmd_buf_last = cmd_buf;

        [cmd_buf retain];
    }
}

ggml_metal_event_t ggml_metal_get_ev_cpy(ggml_metal_t ctx) {
    return ctx->ev_cpy;
}

void ggml_metal_set_n_cb(ggml_metal_t ctx, int n_cb) {
    // opt-in split override.
    // GGML_METAL_N_CB counts ADDITIONAL async command buffers; the main
    // thread owns one more on top, so the graph runs on n_cb + 1 buffers.
    // Large graphs on discrete GPUs can exceed the GPU watchdog when too
    // much work lands in one buffer (kIOGPUCommandBufferCallbackErrorTimeout);
    // raising the split count keeps each buffer under the limit.
    {
        const char * env = getenv("GGML_METAL_N_CB");
        if (env && env[0]) {
            int req = atoi(env);
            if (req < 1) {
                req = 1;
            }
            if (req > GGML_METAL_MAX_COMMAND_BUFFERS) {
                req = GGML_METAL_MAX_COMMAND_BUFFERS;
            }
            n_cb = req;
        }
    }

    // when fusion stats are collected the graph must be encoded by a single thread so the
    // counters are race-free; override whatever the caller requested (env included)
    if (ggml_metal_fusion_info_stats(ctx->finfo)) {
        n_cb = 0;
    }

    if (ctx->n_cb != n_cb) {
        ctx->n_cb = MIN(n_cb, GGML_METAL_MAX_COMMAND_BUFFERS);

        GGML_LOG_INFO("%s: n_cb = %d (plus one main-thread command buffer)\n", __func__, ctx->n_cb);

        if (ctx->n_cb > 2) {
            // the >2 caution comes from Apple-Silicon tuning; on discrete GPUs
            // more splits are sometimes required to stay under the GPU watchdog
            GGML_LOG_INFO("%s: n_cb > 2 adds submit overhead, only use as much as needed\n", __func__);
        }
    }

    if (ctx->encode_async) {
        Block_release(ctx->encode_async);
    }

    ctx->encode_async = Block_copy(^(size_t iter) {
        const int cb_idx = iter;
        const int n_cb_l = ctx->n_cb_graph;

        int idx_start = 0;
        int idx_end   = ctx->cb_start[0];

        if (cb_idx < n_cb_l) {
            idx_start = ctx->cb_start[cb_idx];
            idx_end   = ctx->cb_start[cb_idx + 1];
        }

        id<MTLCommandBuffer> cmd_buf = ctx->cmd_bufs[cb_idx].obj;

        ggml_metal_op_t ctx_op = ggml_metal_op_init(
            ctx->dev,
            cmd_buf,
            ctx->gf,
            ctx->finfo,
            idx_start,
            idx_end,
            ctx->use_concurrency,
            ctx->capture_compute == 0,
            ctx->debug_graph);

        const bool cbp_on = tosh_cbp_on();
        const uint64_t cbp_t0 = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;

        for (int idx = 0; idx < ggml_metal_op_n_nodes(ctx_op); ++idx) {
            const int res = ggml_metal_op_encode(ctx_op, idx);
            if (res == 0) {
                break;
            }

            idx += res - 1;
        }

        ggml_metal_op_free(ctx_op);
        if (cb_idx == n_cb_l) {
            ggml_metal_fb_residual_after(ctx, cmd_buf);
        }

        const uint64_t cbp_t1 = cbp_on ? clock_gettime_nsec_np(CLOCK_UPTIME_RAW) : 0;
        if (cb_idx < 2 || ctx->abort_callback == NULL) {
            [cmd_buf commit];
        }
        if (cbp_on) {
            const uint64_t cbp_t2 = clock_gettime_nsec_np(CLOCK_UPTIME_RAW);
            atomic_fetch_add_explicit(&g_cbp_ops_ns,    cbp_t1 - cbp_t0, memory_order_relaxed);
            atomic_fetch_add_explicit(&g_cbp_commit_ns, cbp_t2 - cbp_t1, memory_order_relaxed);
        }
    });
}

void ggml_metal_set_abort_callback(ggml_metal_t ctx, ggml_abort_callback abort_callback, void * user_data) {
    ctx->abort_callback = abort_callback;
    ctx->abort_callback_data = user_data;
}

bool ggml_metal_supports_family(ggml_metal_t ctx, int family) {
    GGML_ASSERT(ctx->dev != nil);

    id<MTLDevice> device = ggml_metal_device_get_obj(ctx->dev);

    return [device supportsFamily:(MTLGPUFamilyApple1 + family - 1)];
}

void ggml_metal_capture_next_compute(ggml_metal_t ctx) {
    ctx->capture_compute = 1;
}
