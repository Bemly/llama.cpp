#include "tosh-dmoe.h"

#include <atomic>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <chrono>
#include <condition_variable>
#include <algorithm>
#include <mutex>
#include <thread>

namespace {

std::atomic<char *>   g_stage{nullptr};
std::atomic<size_t>   g_stride{0};

void *                g_ev = nullptr;
uint64_t            (*g_ev_get)(void *) = nullptr;
void                (*g_ev_set)(void *, uint64_t) = nullptr;

std::atomic<uint64_t> g_value{1};

tosh_dmoe_job         g_jobs[TOSH_DMOE_RING];

std::mutex              g_active_mutex;
std::condition_variable g_active_cv;
std::atomic<bool>       g_active{false};

std::atomic<const char *> g_failed{nullptr};

int32_t *            g_tables = nullptr;
size_t               g_tables_bytes = 0;
int                  g_tables_layers = 0;
int                  g_tables_experts = 0;
std::atomic<int>     g_version{0};

std::atomic<uint64_t> g_tick{0};
std::atomic<uint64_t> g_completed{0};
std::mutex              g_done_mutex;
std::condition_variable g_done_cv;

tosh_dmoe_uploader    g_uploader = {};
std::atomic<bool>     g_has_uploader{false};

std::atomic<tosh_dmoe_bank_plan_fn> g_planner{nullptr};

void *                g_obs_ring = nullptr;
tosh_dmoe_obs_record  g_obs[TOSH_DMOE_OBS_RECORDS];
std::atomic<uint64_t> g_obs_seq{0};
uint32_t              g_obs_off = 0;
std::atomic<uint64_t> g_obs_pos{0};
std::mutex            g_obs_mutex;

}

static std::atomic<bool> g_forced{false};
void tosh_dmoe_enable(void) { g_forced.store(true); }

bool tosh_dmoe_enabled(void) {
    static const bool on = getenv("TOSH_DMOE_RESIDENCY") != nullptr || getenv("TOSH_DMOE_CACHE_MIB") != nullptr;
    return on || g_forced.load(std::memory_order_relaxed);
}

void tosh_dmoe_set_stage(void * host, size_t layer_stride) {
    g_stride.store(layer_stride);
    g_stage.store((char *) host);
}

void * tosh_dmoe_stage_layer(int il) {
    char * s = g_stage.load();
    return s ? s + (size_t) il*g_stride.load() : nullptr;
}

void tosh_dmoe_set_event(void * ev, uint64_t (*get)(void *), void (*set)(void *, uint64_t)) {
    g_ev_get = get;
    g_ev_set = set;
    g_ev     = ev;
}

uint64_t tosh_dmoe_event_get(void) {
    return g_ev ? g_ev_get(g_ev) : 0;
}

void tosh_dmoe_event_set(uint64_t value) {
    if (g_ev) {
        g_ev_set(g_ev, value);
    }
}

uint64_t tosh_dmoe_next_value(void) { return g_value.fetch_add(2); }
uint64_t tosh_dmoe_last_value(void) { return g_value.load() - 2; }

void tosh_dmoe_set_tables(int32_t * host, size_t bytes, int n_layers, int n_expert) {
    g_tables_bytes   = bytes;
    g_tables_layers  = n_layers;
    g_tables_experts = n_expert;
    g_tables         = host;
}

int32_t * tosh_dmoe_tables(size_t * bytes, int * n_layers, int * n_expert) {
    if (bytes)    *bytes    = g_tables_bytes;
    if (n_layers) *n_layers = g_tables_layers;
    if (n_expert) *n_expert = g_tables_experts;
    return g_tables;
}

void tosh_dmoe_publish_version(int version) { g_version.store(version, std::memory_order_release); }
int  tosh_dmoe_published_version(void)      { return g_version.load(std::memory_order_acquire); }

uint64_t tosh_dmoe_tick_next(void) { return g_tick.fetch_add(1) + 1; }
static std::atomic<bool> g_wide{false};
void tosh_dmoe_note_wide(bool wide) { g_wide.store(wide); }
bool tosh_dmoe_wide(void)           { return g_wide.load(); }
uint64_t tosh_dmoe_tick(void)      { return g_tick.load(); }

static void * g_wide_stage = nullptr;
static size_t g_wide_bytes = 0;
void tosh_dmoe_set_wide_stage(void * host, size_t bytes) { g_wide_stage = host; g_wide_bytes = bytes; }
void * tosh_dmoe_wide_stage(size_t * bytes) {
    if (bytes) *bytes = g_wide_bytes;
    return g_wide_stage;
}

namespace {
struct tl_event { int kind, il; double t0, t1; };
std::mutex g_tl_mutex;
std::vector<tl_event> g_tl;
void tl_dump() {
    const char * p = getenv("TOSH_DMOE_TIMELINE");
    if (!p) return;
    std::lock_guard<std::mutex> lock(g_tl_mutex);
    if (FILE * f = fopen(p, "w")) {
        for (auto & e : g_tl) fprintf(f, "%d %d %.6f %.6f\n", e.kind, e.il, e.t0, e.t1);
        fclose(f);
    }
}
}
// steady_clock runs on the same uptime clock Metal stamps command buffers with
double tosh_dmoe_host_seconds(void) {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
void tosh_dmoe_timeline(int kind, int il, double t0, double t1) {
    static const bool on = getenv("TOSH_DMOE_TIMELINE") != nullptr;
    if (!on) return;
    std::lock_guard<std::mutex> lock(g_tl_mutex);
    if (g_tl.empty()) { g_tl.reserve(1 << 20); atexit(tl_dump); }
    if (g_tl.size() < (1 << 22)) g_tl.push_back({ kind, il, t0, t1 });
}

static size_t g_vram_free = 0, g_vram_total = 0;
void tosh_dmoe_set_vram(size_t free_at_start, size_t total) {
    if (g_vram_total == 0) { g_vram_free = free_at_start; g_vram_total = total; }
}
static size_t (*g_private_query)(void) = nullptr;
void   tosh_dmoe_set_private_query(size_t (*fn)(void)) { g_private_query = fn; }
size_t tosh_dmoe_private_bytes(void) { return g_private_query ? g_private_query() : 0; }

bool tosh_dmoe_vram(size_t * free_at_start, size_t * total) {
    *free_at_start = g_vram_free; *total = g_vram_total;
    return g_vram_total > 0;
}

static bool (*g_vram_query)(int64_t *, int64_t *) = nullptr;
void tosh_dmoe_set_vram_query(bool (*fn)(int64_t *, int64_t *)) { g_vram_query = fn; }

bool tosh_vram_read(size_t * vfree, size_t * total) {
    if (g_vram_query == nullptr) {
        return false;
    }
    constexpr int N = 3;
    int64_t f[N], u[N];
    int n = 0;
    for (int i = 0; i < N; ++i) {
        if (i > 0) std::this_thread::sleep_for(std::chrono::milliseconds(2));
        if (g_vram_query(&f[n], &u[n]) && f[n] >= 0 && u[n] >= 0) n++;
    }
    if (n == 0) {
        return false;
    }
    std::sort(f, f + n);
    std::sort(u, u + n);
    const int64_t mf = f[n/2], mu = u[n/2];
    // the card's size is taken once; after that free is what nobody holds, since the driver's own
    // free counter lags behind memory that was released and can still be claimed
    static int64_t size = 0;
    if (size == 0) size = mf + mu;
    const int64_t free_now = std::max<int64_t>(0, size - mu);
    if (llabs(mf - free_now) > (256ll << 20)) {
        fprintf(stderr, "tosh_vram: driver reports %lld MiB free, %lld MiB are not in use; using the latter\n",
                (long long) (mf >> 20), (long long) (free_now >> 20));
    }
    *vfree = (size_t) free_now;
    *total = (size_t) size;
    return true;
}

namespace {
struct rare_ctl {
    std::mutex m;
    std::atomic<int> rows{0};
    int  max_rows = 8;
    bool adapt = false;
    struct layer { uint64_t host_seq = 0, dma_seq = 0; double dma_end[8] = {}; };
    std::vector<layer> layers = std::vector<layer>(TOSH_DMOE_MAX_LAYERS);
    int n = 0, n_late = 0;
    double late_s = 0;
    double prev_start = -1;
    int    prev_tokens = 0;
    double host_s = 0;              // host time of the chunk in progress
    double share = -1;              // smoothed host share of a chunk
    double target = 0.50;
    double waited = -1;                 // smoothed share of layers whose join waited on the host
    int    n_decode = 0;                // decode jobs since the last chunk
    int    n_wait = 0;                  // layers whose join was already waiting when the host finished
};
rare_ctl g_rare;
}

void tosh_dmoe_rare_rows_init(int rows, int max_rows, bool adapt) {
    std::lock_guard<std::mutex> lock(g_rare.m);
    g_rare.rows = rows; g_rare.max_rows = max_rows; g_rare.adapt = adapt;
    if (const char * t = getenv("TOSH_DMOE_RARE_TARGET")) g_rare.target = atof(t);
}
int tosh_dmoe_rare_rows_live(void) { return g_rare.rows.load(std::memory_order_relaxed); }

void tosh_dmoe_rare_dma_done(int il, double t_end) {
    if (il < 0 || il >= TOSH_DMOE_MAX_LAYERS) return;
    std::lock_guard<std::mutex> lock(g_rare.m);
    auto & L = g_rare.layers[il];
    L.dma_end[L.dma_seq % 8] = t_end;
    L.dma_seq++;
}

static void * g_reach_ev = nullptr;
static uint64_t (*g_reach_get)(void *) = nullptr;
void tosh_dmoe_set_reach_event(void * ev, uint64_t (*get)(void *)) { g_reach_ev = ev; g_reach_get = get; }
uint64_t tosh_dmoe_reach_get(void) { return g_reach_get ? g_reach_get(g_reach_ev) : 0; }

void tosh_dmoe_rare_host_done(int il, double t_start, double t_end, bool gpu_waiting) {
    if (il < 0 || il >= TOSH_DMOE_MAX_LAYERS) return;
    std::lock_guard<std::mutex> lock(g_rare.m);
    g_rare.host_s += t_end - t_start;
    g_rare.n_wait += gpu_waiting;
    auto & L = g_rare.layers[il];
    const uint64_t h = L.host_seq++;
    // the chunk's down bank already landed: the join waits on the host from here
    if (L.dma_seq > h && L.dma_seq - h <= 8) {
        g_rare.n_late++;
        g_rare.late_s += t_end - L.dma_end[h % 8];
    }
    g_rare.n++;
}

void tosh_dmoe_rare_interrupt(void) {
    std::lock_guard<std::mutex> lock(g_rare.m);
    g_rare.n_decode++;
}

void tosh_dmoe_rare_chunk(double t_start, int n_tokens) {
    std::lock_guard<std::mutex> lock(g_rare.m);
    // a model that narrows its last layer to the output rows runs one decode job per chunk; more
    // than a few means decode ran in between and the period is not a measurement
    const double prev = g_rare.n_decode > 8 ? -1.0 : g_rare.prev_start;
    const int    prev_tokens = g_rare.prev_tokens;
    g_rare.prev_start  = t_start;
    g_rare.prev_tokens = n_tokens;
    g_rare.n_decode    = 0;
    if (prev < 0) {
        g_rare.host_s = 0;
        g_rare.n_wait = 0;
        g_rare.n = 0;
    }
    // a gap longer than a chunk is decode or idle in between, not a measurement
    if (prev < 0 || prev_tokens <= 0 || t_start - prev > 5.0) {
        return;
    }
    const double share = g_rare.host_s/(t_start - prev);
    g_rare.host_s = 0;
    g_rare.share = g_rare.share < 0 ? share : 0.5*g_rare.share + 0.5*share;
    if (g_rare.n > 0) {
        const double w = (double) g_rare.n_wait/g_rare.n;
        g_rare.waited = g_rare.waited < 0 ? w : 0.5*g_rare.waited + 0.5*w;
    }
    int r = g_rare.rows.load();
    const int was = r;
    if (g_rare.adapt && g_rare.waited >= 0) {
        // two rows at a time while far from the target, so a short prompt still gets there
        const int step = g_rare.waited < 0.10 || g_rare.waited > 0.90 ? 2 : 1;
        if (g_rare.waited < g_rare.target - 0.15) r = std::min(r + step, g_rare.max_rows);
        else if (g_rare.waited > g_rare.target + 0.15) r = std::max(r - step, 1);
    }
    if (getenv("TOSH_DMOE_STATS")) {
        fprintf(stderr, "tosh_dmoe: rare-row threshold %d -> %d (host share %.2f, %.3f ms per token; join waited in %d of %d layers, %d finished after their down bank)\n",
                was, r, g_rare.share, (t_start - prev)/prev_tokens*1e3, g_rare.n_wait, g_rare.n, g_rare.n_late);
    }
    g_rare.n_wait = 0;
    g_rare.rows = r;
    g_rare.n = 0; g_rare.n_late = 0; g_rare.late_s = 0;
}

bool tosh_dmoe_compact(void) {
    static const bool on = getenv("TOSH_DMOE_COMPACT") == nullptr || atoi(getenv("TOSH_DMOE_COMPACT")) != 0;
    return on && tosh_dmoe_enabled();
}
static const int32_t * g_chunk_rows = nullptr;
static const float *   g_chunk_tail = nullptr;
static int g_chunk_experts = 0;
void tosh_dmoe_set_chunk_rows(const int32_t * rows, const float * tail, int n_expert) {
    g_chunk_rows = rows; g_chunk_tail = tail; g_chunk_experts = n_expert;
}
const float * tosh_dmoe_chunk_tail(void) { return g_chunk_tail; }
float tosh_dmoe_token_decay(void) {
    static const float d = getenv("TOSH_DMOE_PREFILL_TOKEN_DECAY") ? (float) atof(getenv("TOSH_DMOE_PREFILL_TOKEN_DECAY")) : 0.95f;
    return d;
}
static std::atomic<int64_t> g_pf_us[4];
static std::atomic<int64_t> g_pf_n[4];
static void pf_time_report() {
    if (getenv("TOSH_DMOE_STATS") && g_pf_n[1].load() > 0) {
        fprintf(stderr, "tosh_dmoe: prefill time waiting for ids %.1f ms (%lld), counting rows %.1f ms, issuing uploads %.1f ms (%lld), upload gpu %.1f ms (%lld)\n",
                g_pf_us[0].load()/1e3, (long long) g_pf_n[0].load(), g_pf_us[3].load()/1e3, g_pf_us[1].load()/1e3, (long long) g_pf_n[1].load(),
                g_pf_us[2].load()/1e3, (long long) g_pf_n[2].load());
    }
}
void tosh_dmoe_note_time(int what, int64_t us) {
    static std::once_flag once;
    std::call_once(once, [] { atexit(pf_time_report); });
    g_pf_us[what] += us;
    g_pf_n[what]++;
}
const int32_t * tosh_dmoe_chunk_rows(int * n_expert) {
    if (n_expert) *n_expert = g_chunk_experts;
    return g_chunk_rows;
}

void tosh_dmoe_mark_completed(void) {
    {
        std::lock_guard<std::mutex> lock(g_done_mutex);
        g_completed.fetch_add(1);
    }
    g_done_cv.notify_all();
}

uint64_t tosh_dmoe_completed_value(void) { return g_completed.load(); }

bool tosh_dmoe_wait_completed(uint64_t seen, int timeout_us) {
    std::unique_lock<std::mutex> lock(g_done_mutex);
    return g_done_cv.wait_for(lock, std::chrono::microseconds(timeout_us), [seen] { return g_completed.load() != seen; });
}

void tosh_dmoe_set_bank_planner(tosh_dmoe_bank_plan_fn fn) { g_planner.store(fn); }

bool tosh_dmoe_bank_planner(const void * host, size_t size, tosh_dmoe_bank_plan * plan) {
    const tosh_dmoe_bank_plan_fn fn = g_planner.load();
    return fn != nullptr && fn(host, size, plan);
}

void   tosh_dmoe_set_obs_ring(void * host) { g_obs_ring = host; }
void * tosh_dmoe_obs_ring(void)            { return g_obs_ring; }
uint64_t tosh_dmoe_obs_head(void)          { return g_obs_seq.load(); }

tosh_dmoe_obs_record * tosh_dmoe_obs_reserve(int il, int n_used, int n_tokens) {
    const uint32_t bytes = (uint32_t) n_used*n_tokens*sizeof(int32_t);
    if (bytes == 0 || bytes > TOSH_DMOE_OBS_RING_BYTES/4) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(g_obs_mutex);
    if (g_obs_off + bytes > TOSH_DMOE_OBS_RING_BYTES) {
        g_obs_pos.fetch_add(TOSH_DMOE_OBS_RING_BYTES - g_obs_off);
        g_obs_off = 0;
    }
    const uint64_t seq = g_obs_seq.load() + 1;
    tosh_dmoe_obs_record * r = &g_obs[seq % TOSH_DMOE_OBS_RECORDS];
    r->sync     = tosh_dmoe_completed_value();
    r->offset   = g_obs_off;
    r->pos      = g_obs_pos.load();
    r->il       = il;
    r->n_used   = n_used;
    r->n_tokens = n_tokens;
    r->tick     = (uint32_t) tosh_dmoe_tick();
    __atomic_store_n(&r->seq, seq, __ATOMIC_RELEASE);
    g_obs_seq.store(seq);
    g_obs_off += (bytes + 255) & ~255u;
    g_obs_pos.fetch_add((bytes + 255) & ~255u);
    return r;
}

bool tosh_dmoe_obs_intact(const tosh_dmoe_obs_record * r) {
    const uint64_t bytes = (uint64_t) r->n_used*r->n_tokens*sizeof(int32_t);
    return g_obs_pos.load() - r->pos + bytes <= TOSH_DMOE_OBS_RING_BYTES;
}

tosh_dmoe_obs_record * tosh_dmoe_obs_record_at(uint64_t seq) {
    tosh_dmoe_obs_record * r = &g_obs[seq % TOSH_DMOE_OBS_RECORDS];
    return __atomic_load_n(&r->seq, __ATOMIC_ACQUIRE) == seq ? r : nullptr;
}

void tosh_dmoe_set_uploader(const tosh_dmoe_uploader * u) {
    g_uploader = *u;
    g_has_uploader.store(true);
}

static void (*g_uploader_factory)(void) = nullptr;
void tosh_dmoe_set_uploader_factory(void (*make)(void)) { g_uploader_factory = make; }

const tosh_dmoe_uploader * tosh_dmoe_get_uploader(void) {
    static std::mutex m;
    if (!g_has_uploader.load() && g_uploader_factory != nullptr) {
        std::lock_guard<std::mutex> lock(m);
        if (!g_has_uploader.load()) g_uploader_factory();
    }
    return g_has_uploader.load() ? &g_uploader : nullptr;
}

tosh_dmoe_job * tosh_dmoe_job_of(uint64_t value) {
    return &g_jobs[(value >> 1) % TOSH_DMOE_RING];
}

void tosh_dmoe_set_active(bool active) {
    if (g_active.exchange(active) == active) {
        return;
    }
    if (active) {
        std::lock_guard<std::mutex> lock(g_active_mutex);
        g_active_cv.notify_all();
    }
}

bool tosh_dmoe_is_active(void) { return g_active.load(std::memory_order_acquire); }

bool tosh_dmoe_wait_active(int timeout_us) {
    if (g_active.load()) {
        return true;
    }
    std::unique_lock<std::mutex> lock(g_active_mutex);
    return g_active_cv.wait_for(lock, std::chrono::microseconds(timeout_us), [] { return g_active.load(); });
}

void tosh_dmoe_fail(const char * why) {
    const char * none = nullptr;
    g_failed.compare_exchange_strong(none, why);
}

const char * tosh_dmoe_failed(void)  { return g_failed.load(); }
void tosh_dmoe_clear_failure(void)   { g_failed.store(nullptr); }
