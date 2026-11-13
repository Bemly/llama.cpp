#include "llama-tosh-dmoe.h"

#include "llama-impl.h"
#include "llama-model.h"
#include "llama-hparams.h"

#include "ggml-backend.h"
#include "ggml-cpu.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-dmoe.h"
#include "tosh-hostcache.h"
#endif

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <map>
#include <memory>
#include <sstream>
#include <thread>
#include <unordered_map>

#include <sys/mman.h>
#include <pthread.h>
#include <unistd.h>
#include <cerrno>

#if defined(__x86_64__)
#include <immintrin.h>
#define DMOE_PAUSE() _mm_pause()
#else
#define DMOE_PAUSE() do {} while (0)
#endif

static std::unordered_map<const ggml_tensor *, llama_tosh_dmoe_layer> g_layers;

const llama_tosh_dmoe_layer * llama_tosh_dmoe_layer_of(const ggml_tensor * gate_exps) {
    if (g_layers.empty() || gate_exps == nullptr) {
        return nullptr;
    }
    const auto it = g_layers.find(gate_exps);
    return it == g_layers.end() ? nullptr : &it->second;
}

#ifndef TOSH_ENABLE_DYNAMIC_MOE

void llama_tosh_dmoe_init(std::vector<llama_layer> &, const llama_hparams &) {}
bool llama_tosh_dmoe_budget_pending() { return false; }
void llama_tosh_dmoe_budget(size_t) {}
bool llama_tosh_dmoe_rebudget_possible() { return false; }
void llama_tosh_dmoe_context_opened() {}
void llama_tosh_dmoe_context_closed() {}
void llama_tosh_dmoe_shutdown() {}
bool llama_tosh_dmoe_observe_prefill() { return false; }
bool llama_tosh_dmoe_accept(const llama_tosh_dmoe_layer *, int, float, float, bool, bool) { return false; }
int  llama_tosh_dmoe_rare_rows() { return 0; }
int  llama_tosh_dmoe_rare_tail() { return 0; }
int  llama_tosh_dmoe_wide_min_tokens() { return 1 << 30; }

#else

namespace {

using clk = std::chrono::steady_clock;

static double us_since(clk::time_point t0) {
    return std::chrono::duration<double, std::micro>(clk::now() - t0).count();
}

// The canonical bank of one layer, read in place by the host executor.
struct host_bank {
    const ggml_tensor * gate = nullptr;
    const ggml_tensor * up   = nullptr;
    const ggml_tensor * down = nullptr;
    bool fused = false;             // gate holds gate and up, one after the other, and up is null
    // per-expert gate/up biases, copied at load; the down bias is added on the GPU after the join
    std::vector<float> gate_b, up_b;
    // activation the graph registered for this layer: -1 none yet, 0 silu, 1 swiglu_oai, 2 gelu
    std::atomic<int> act{-1};
    float alpha = 0.0f, limit = 0.0f;
    // the router as floats, kept only when a bounded cache reads experts ahead of the layer
    std::vector<float> router, router_b;
    host_bank() = default;
    host_bank(const host_bank & o) : gate(o.gate), up(o.up), down(o.down), fused(o.fused), gate_b(o.gate_b), up_b(o.up_b),
        act(o.act.load()), alpha(o.alpha), limit(o.limit), router(o.router), router_b(o.router_b) {}
};

// A layer's gate bank holds gate and up together when the model fuses them.
static ggml_tensor * dmoe_gate(const llama_layer & L) { return L.ffn_gate_exps ? L.ffn_gate_exps : L.ffn_gate_up_exps; }
static bool dmoe_has_bank(const llama_layer & L) {
    return dmoe_gate(L) && L.ffn_down_exps && (L.ffn_gate_exps ? L.ffn_up_exps != nullptr : L.ffn_up_exps == nullptr);
}

struct miss {
    int t;      // token
    int j;      // router position
    int e;      // expert id
    // bounded host cache: the expert's cached gate, up and down, pinned for this job
    const char * g = nullptr, * u = nullptr, * d = nullptr;
};

struct executor {
    int n_threads = 1;
    int n_embd    = 0;
    int n_ff      = 0;

    std::vector<host_bank> banks;   // by layer

    // per-job scratch, sized once for the widest job
    std::vector<uint8_t> act_q;     // n_tokens rows of the activation, quantized for gate/up
    std::vector<float>   h;         // one n_ff row per miss
    std::vector<uint8_t> h_q;       // the same, quantized for down
    size_t act_q_row = 0;
    size_t h_q_row   = 0;

    // current job
    std::vector<miss> misses;
    std::vector<int>  grp;   // start of each run of misses on one expert, then the end
    const host_bank * bank = nullptr;
    const float * act = nullptr;
    float * out = nullptr;
    int n_used = 0;
    bool wide = false;                 // out holds one row per miss instead of one per router position

    // one cache line each: every chunk claim would otherwise invalidate the line the barrier spins on
    alignas(64) std::atomic<int>      next1{0};
    alignas(64) std::atomic<int>      next2{0};
    alignas(64) std::atomic<uint64_t> gen{0};
    alignas(64) std::atomic<int>      arrived{0};
    alignas(64) std::atomic<uint64_t> barrier_gen{0};
    alignas(64) std::atomic<bool>     stop{false};

    // helpers spin while jobs come every few hundred microseconds (decode) and sleep otherwise
    std::mutex              work_mutex;
    std::condition_variable work_cv;
    std::atomic<int>        sleepers{0};
    int                     spin_us = 2000;

    void kick() {
        gen.fetch_add(1);
        if (sleepers.load() > 0) {
            std::lock_guard<std::mutex> lock(work_mutex);
            work_cv.notify_all();
        }
    }

    std::vector<std::thread> helpers;
    std::thread leader;
    std::thread watchdog;
    // the driver cancels a command buffer kept waiting 6 s (4 s measured safe on RDNA2)
    double      watchdog_us = 3e6;

    // dynamic residency: what the last job routed, handed to the policy once the GPU is released
    void (*observe)(void * ctx, int il, const int32_t * ids, const int32_t * slots, int n, bool multi, uint32_t tick) = nullptr;
    void * observe_ctx = nullptr;
    int32_t obs_ids[TOSH_DMOE_MAX_TOKENS*256];
    int32_t obs_slots[TOSH_DMOE_MAX_TOKENS*256];
    int     obs_n = 0;
    std::vector<std::atomic<uint32_t>> hits_by_tick, routed_by_tick;

    // stats
    alignas(64) std::atomic<uint64_t> n_jobs{0};
    std::atomic<uint64_t> n_routed{0}, n_miss{0};
    std::atomic<uint64_t> n_wide{0}, n_wide_miss{0}, n_wide_experts{0}, us_wide{0};
    std::atomic<uint64_t> us_job{0}, us_quant{0}, us_p1{0}, us_p2{0};
    std::vector<std::atomic<uint64_t>> miss_by_layer;
    std::vector<std::atomic<uint64_t>> routed_by_layer;

    static constexpr int rows1 = 16;
    static constexpr int rows2 = 64;

    // misses sorted by expert; a run is capped so a wide batch still spreads over the threads
    void build_groups() {
        static constexpr int max_run = 8;
        grp.clear();
        for (int i = 0; i < (int) misses.size(); ++i) {
            if (i == 0 || misses[i].e != misses[i - 1].e || i - grp.back() >= max_run) {
                grp.push_back(i);
            }
        }
        grp.push_back((int) misses.size());
    }

    void barrier() {
        const uint64_t g = barrier_gen.load(std::memory_order_acquire);
        if (arrived.fetch_add(1, std::memory_order_acq_rel) + 1 == n_threads) {
            arrived.store(0, std::memory_order_relaxed);
            barrier_gen.store(g + 1, std::memory_order_release);
        } else {
            while (barrier_gen.load(std::memory_order_acquire) == g) {
                DMOE_PAUSE();
            }
        }
    }

    // gate/up rows of every miss, then down rows; each thread takes chunks from shared counters
    void run(int tid) {
        const auto * tt_gate = ggml_get_type_traits_cpu(bank->gate->type);
        const auto * tt_up   = ggml_get_type_traits_cpu((bank->fused ? bank->gate : bank->up)->type);
        const ggml_tensor * up_t = bank->fused ? bank->gate : bank->up;
        const size_t up_off = bank->fused ? (size_t) n_ff*bank->gate->nb[1] : 0;
        const auto * tt_down = ggml_get_type_traits_cpu(bank->down->type);

        const int n_miss = (int) misses.size();
        const int n_grp  = (int) grp.size() - 1;
        const int c1 = n_ff/rows1;
        const clk::time_point t1 = clk::now();
        // each weight row is read once per run of misses on its expert and applied to all of them, and
        // unpacked once for up to four of them where the type has a multi-row dot
        const auto mg = ggml_cpu_get_vec_dot_multi(bank->gate->type);
        const auto mu = ggml_cpu_get_vec_dot_multi(up_t->type);
        const auto md = ggml_cpu_get_vec_dot_multi(bank->down->type);
        for (int c; (c = next1.fetch_add(1, std::memory_order_relaxed)) < n_grp*c1; ) {
            const int i0 = grp[c/c1], i1 = grp[c/c1 + 1];
            const miss & m = misses[i0];
            const int r0 = (c % c1)*rows1;
            const char * g = m.g ? m.g : (const char *) bank->gate->data + (size_t) m.e*bank->gate->nb[2];
            const char * u = m.u ? m.u : (const char *) up_t->data + (size_t) m.e*up_t->nb[2] + up_off;
            const float * gb = bank->gate_b.empty() ? nullptr : bank->gate_b.data() + (size_t) m.e*n_ff;
            const float * ub = bank->up_b.empty()   ? nullptr : bank->up_b.data()   + (size_t) m.e*n_ff;
            const int act = bank->act.load(std::memory_order_relaxed);
            auto finish = [&](int i, int r, float sg, float su) {
                float * hr = h.data() + (size_t) i*n_ff;
                if (gb) sg += gb[r];
                if (ub) su += ub[r];
                if (act == 1) {
                    // same form as ggml_swiglu_oai
                    const float x = std::min(sg, bank->limit);
                    const float y = std::clamp(su, -bank->limit, bank->limit);
                    hr[r] = x/(1.0f + expf(-bank->alpha*x))*(y + 1.0f);
                } else if (act == 2) {
                    // the tanh form ggml_geglu uses
                    hr[r] = 0.5f*sg*(1.0f + tanhf(0.7978845608f*sg*(1.0f + 0.044715f*sg*sg)))*su;
                } else {
                    hr[r] = sg/(1.0f + expf(-sg))*su;
                }
            };
            for (int r = r0; r < r0 + rows1; ++r) {
                const char * gr = g + (size_t) r*bank->gate->nb[1];
                const char * ur = u + (size_t) r*up_t->nb[1];
                int i = i0;
                if (mg && mu) {
                    while (i1 - i >= 2) {
                        const int k = std::min(4, i1 - i);
                        const void * av[4];
                        for (int t = 0; t < k; ++t) av[t] = act_q.data() + (size_t) misses[i + t].t*act_q_row;
                        float sg[4], su[4];
                        mg(n_embd, k, sg, gr, av);
                        mu(n_embd, k, su, ur, av);
                        for (int t = 0; t < k; ++t) finish(i + t, r, sg[t], su[t]);
                        i += k;
                    }
                }
                for (; i < i1; ++i) {
                    const void * a = act_q.data() + (size_t) misses[i].t*act_q_row;
                    float sg, su;
                    tt_gate->vec_dot(n_embd, &sg, 0, gr, 0, a, 0, 1);
                    tt_up  ->vec_dot(n_embd, &su, 0, ur, 0, a, 0, 1);
                    finish(i, r, sg, su);
                }
            }
        }
        barrier();
        if (tid == 0) {
            us_p1 += (uint64_t) us_since(t1);
            const ggml_type vt = tt_down->vec_dot_type;
            const auto from_float = ggml_get_type_traits_cpu(vt)->from_float;
            for (int i = 0; i < n_miss; ++i) {
                from_float(h.data() + (size_t) i*n_ff, h_q.data() + (size_t) i*h_q_row, n_ff);
            }
        }
        barrier();
        const clk::time_point t2 = clk::now();
        const int c2 = n_embd/rows2;
        for (int c; (c = next2.fetch_add(1, std::memory_order_relaxed)) < n_grp*c2; ) {
            const int i0 = grp[c/c2], i1 = grp[c/c2 + 1];
            const int r0 = (c % c2)*rows2;
            const miss & m0 = misses[i0];
            const char * d = m0.d ? m0.d : (const char *) bank->down->data + (size_t) m0.e*bank->down->nb[2];
            auto out_row = [&](int i) {
                const miss & m = misses[i];
                return out + (wide ? (size_t) i : (size_t) m.t*n_used + m.j)*n_embd;
            };
            for (int r = r0; r < r0 + rows2; ++r) {
                const char * dr = d + (size_t) r*bank->down->nb[1];
                int i = i0;
                if (md) {
                    while (i1 - i >= 2) {
                        const int k = std::min(4, i1 - i);
                        const void * hv[4];
                        for (int t = 0; t < k; ++t) hv[t] = h_q.data() + (size_t) (i + t)*h_q_row;
                        float o4[4];
                        md(n_ff, k, o4, dr, hv);
                        for (int t = 0; t < k; ++t) out_row(i + t)[r] = o4[t];
                        i += k;
                    }
                }
                for (; i < i1; ++i) {
                    tt_down->vec_dot(n_ff, out_row(i) + r, 0, dr, 0, h_q.data() + (size_t) i*h_q_row, 0, 1);
                }
            }
        }
        barrier();
        if (tid == 0) {
            us_p2 += (uint64_t) us_since(t2);
        }
    }

    void helper_main(int tid) {
        raise_priority();
        uint64_t seen = 0;
        while (!stop.load()) {
            if (!tosh_dmoe_wait_active(1000000)) {
                continue;
            }
            uint64_t g;
            const clk::time_point t0 = clk::now();
            for (int n = 1; (g = gen.load()) == seen && tosh_dmoe_is_active() && !stop.load(); ++n) {
                if ((n & 255) == 0 && us_since(t0) > spin_us) {
                    std::unique_lock<std::mutex> lock(work_mutex);
                    sleepers++;
                    work_cv.wait_for(lock, std::chrono::milliseconds(50),
                        [&] { return gen.load() != seen || !tosh_dmoe_is_active() || stop.load(); });
                    sleepers--;
                    continue;
                }
                DMOE_PAUSE();
            }
            if (g == seen) {
                continue;
            }
            seen = g;
            run(tid);
        }
    }

    std::vector<int32_t> pinned;
    int pinned_il = -1;

    bool pin_misses(int il, int kind) {
        if (!tosh_hostcache_on() || misses.empty()) {
            return true;
        }
        pinned.clear();
        for (const miss & m : misses) pinned.push_back(m.e);
        pinned_il = il;
        if (!tosh_hostcache_acquire(il, pinned.data(), (int) pinned.size(), kind)) {
            tosh_dmoe_fail(tosh_hostcache_error());
            return false;
        }
        const size_t up_off = bank->fused ? (size_t) n_ff*bank->gate->nb[1] : 0;
        for (miss & m : misses) {
            m.g = tosh_hostcache_part(il, m.e, 0);
            m.u = bank->fused ? m.g + up_off : tosh_hostcache_part(il, m.e, 1);
            m.d = tosh_hostcache_part(il, m.e, 2);
        }
        return true;
    }

    void unpin_misses() {
        if (!pinned.empty()) {
            tosh_hostcache_release(pinned_il, pinned.data(), (int) pinned.size());
            pinned.clear();
        }
    }

    // a wide batch: the positions the GPU skipped, grouped by expert so its weights stay in cache
    bool job_wide(int il, const tosh_dmoe_job & jb) {
        const double t0 = tosh_dmoe_host_seconds();
        if (il <= wide_first_il) {
            wide_first_il = il;
            tosh_dmoe_rare_chunk(t0, jb.n_tokens);
        }
        size_t bytes = 0;
        char * stage = (char *) tosh_dmoe_wide_stage(&bytes);
        const int T = jb.n_tokens, nu = jb.n_used;
        if (stage == nullptr || il >= (int) banks.size() || banks[il].gate == nullptr ||
            jb.n_embd != n_embd || T > TOSH_DMOE_WIDE_MAX_TOKENS || nu <= 0 ||
            tosh_dmoe_wide_off_out(T, n_embd, nu) + (size_t) TOSH_DMOE_WIDE_MAX_ROWS*n_embd*4 > bytes) {
            tosh_dmoe_fail("a wide layer has no host bank or stage, or an unexpected shape");
            return false;
        }
        bank   = &banks[il];
        n_used = nu;
        act    = (const float *) stage;
        out    = (float *) (stage + tosh_dmoe_wide_off_out(T, n_embd, nu));
        const int32_t * ids = (const int32_t *) (stage + tosh_dmoe_wide_off_ids(T, n_embd, nu));
        const int32_t * rmp = (const int32_t *) (stage + tosh_dmoe_wide_off_rmp(T, n_embd, nu));
        int32_t       * map = (int32_t *)       (stage + tosh_dmoe_wide_off_map(T, n_embd, nu));

        misses.clear();
        for (int k = 0; k < T*nu; ++k) {
            map[k] = -1;
            if (rmp[k] < 0) misses.push_back({ k/nu, k % nu, ids[k] });
        }
        if ((int) misses.size() > TOSH_DMOE_WIDE_MAX_ROWS) {
            tosh_dmoe_fail("a wide layer sent more host rows than the stage holds");
            return false;
        }
        std::stable_sort(misses.begin(), misses.end(), [](const miss & a, const miss & b) { return a.e < b.e; });
        build_groups();
        int n_exp = 0;
        for (size_t i = 0; i < misses.size(); ++i) {
            map[misses[i].t*nu + misses[i].j] = (int32_t) i;
            n_exp += i == 0 || misses[i].e != misses[i - 1].e;
        }

        if (!misses.empty()) {
            if (act_q.size() < (size_t) T*act_q_row) act_q.resize((size_t) T*act_q_row);
            if (h.size() < misses.size()*n_ff) {
                h.resize(misses.size()*n_ff);
                h_q.resize(misses.size()*h_q_row);
            }
            const auto from_float = ggml_get_type_traits_cpu(ggml_get_type_traits_cpu(bank->gate->type)->vec_dot_type)->from_float;
            wide_seen.assign(T, 0);
            for (const miss & m : misses) {
                if (!wide_seen[m.t]) {
                    from_float(act + (size_t) m.t*n_embd, act_q.data() + (size_t) m.t*act_q_row, n_embd);
                    wide_seen[m.t] = 1;
                }
            }
            if (!pin_misses(il, TOSH_HC_WIDE)) {
                return false;
            }
            wide = true;
            next1.store(0, std::memory_order_relaxed);
            next2.store(0, std::memory_order_relaxed);
            kick();
            run(0);
            wide = false;
            unpin_misses();
        }
        n_wide++;
        n_wide_miss    += misses.size();
        n_wide_experts += n_exp;
        const double t1 = tosh_dmoe_host_seconds();
        us_wide += (uint64_t) ((t1 - t0)*1e6);
        tosh_dmoe_timeline(1, il, t0, t1);
        tosh_dmoe_rare_host_done(il, t0, t1, tosh_dmoe_reach_get() >= jb.value);
        return true;
    }
    std::vector<uint8_t> wide_seen;
    int wide_first_il = 1 << 30;

    // the next layer's input differs from this one's by this layer's update alone, so its router
    // applied here picks most of the experts it will ask for: read the ones neither VRAM nor RAM holds
    std::vector<float> ahead_score;
    std::vector<int>   ahead_order;
    uint64_t n_ahead = 0;
    void read_ahead(int nl) {
        const host_bank & nb = banks[nl];
        size_t tb_bytes = 0;
        int tb_layers = 0, tb_expert = 0;
        const int32_t * tb = tosh_dmoe_tables(&tb_bytes, &tb_layers, &tb_expert);
        const int ne = nb.router.empty() ? 0 : (int) (nb.router.size()/n_embd);
        if (tb == nullptr || ne == 0 || ne != tb_expert) return;
        const int32_t * vram = tb + ((size_t) tosh_dmoe_published_version()*TOSH_DMOE_MAX_LAYERS + nl)*ne;
        const int k = std::min(ne, n_used + n_used/2);
        ahead_score.resize(ne);
        ahead_order.resize(ne);
        for (int e = 0; e < ne; ++e) {
            const float * w = nb.router.data() + (size_t) e*n_embd;
            float acc[8] = {};
            int i = 0;
            for (; i + 8 <= n_embd; i += 8) for (int l = 0; l < 8; ++l) acc[l] += w[i + l]*act[i + l];
            float sum = 0.0f;
            for (; i < n_embd; ++i) sum += w[i]*act[i];
            for (int l = 0; l < 8; ++l) sum += acc[l];
            ahead_score[e] = nb.router_b.empty() ? sum : 1.0f/(1.0f + expf(-sum)) + nb.router_b[e];
            ahead_order[e] = e;
        }
        std::partial_sort(ahead_order.begin(), ahead_order.begin() + k, ahead_order.end(),
                          [this](int a, int b) { return ahead_score[a] > ahead_score[b]; });
        for (int j = 0; j < k; ++j) {
            if (vram[ahead_order[j]] < 0) {
                tosh_hostcache_prefetch(nl, ahead_order[j]);
                n_ahead++;
            }
        }
    }

    // one layer: find the non-resident experts, compute them, answer the GPU
    bool job(int il, const tosh_dmoe_job & jb) {
        const clk::time_point t0 = clk::now();

        char * stage = (char *) tosh_dmoe_stage_layer(il);
        if (stage == nullptr || il >= (int) banks.size() || banks[il].gate == nullptr) {
            tosh_dmoe_fail("a mixed layer has no host bank or stage");
            return false;
        }
        if (jb.n_embd != n_embd || jb.n_tokens > TOSH_DMOE_MAX_TOKENS || jb.n_used <= 0) {
            tosh_dmoe_fail("a mixed layer published an unexpected shape");
            return false;
        }

        // test only: TOSH_DMOE_TEST_FAIL=<job> fails that job, TOSH_DMOE_TEST_STALL=<job> hangs it
        static const long fail_at  = getenv("TOSH_DMOE_TEST_FAIL")  ? atol(getenv("TOSH_DMOE_TEST_FAIL"))  : -1;
        static const long stall_at = getenv("TOSH_DMOE_TEST_STALL") ? atol(getenv("TOSH_DMOE_TEST_STALL")) : -1;
        if ((long) n_jobs.load() == fail_at) {
            tosh_dmoe_fail("injected failure");
            return false;
        }
        if ((long) n_jobs.load() == stall_at) {
            static const long stall_ms = getenv("TOSH_DMOE_TEST_STALL_MS") ? atol(getenv("TOSH_DMOE_TEST_STALL_MS")) : 3000;
            std::this_thread::sleep_for(std::chrono::milliseconds(stall_ms));
        }

        bank   = &banks[il];
        n_used = jb.n_used;
        act    = (const float *) (stage + tosh_dmoe_off_act(n_embd, n_used));
        out    = (float *) (stage + tosh_dmoe_off_out(n_embd, n_used));

        const int32_t * ids   = (const int32_t *) (stage + tosh_dmoe_off_ids(n_embd, n_used));
        const int32_t * slots = (const int32_t *) (stage + tosh_dmoe_off_slots(n_embd, n_used));

        misses.clear();
        bool tok_used[TOSH_DMOE_MAX_TOKENS] = {};
        for (int t = 0; t < jb.n_tokens; ++t) {
            for (int j = 0; j < n_used; ++j) {
                const int k = t*n_used + j;
                if (slots[k] < 0) {
                    misses.push_back({ t, j, ids[k] });
                    tok_used[t] = true;
                }
            }
        }
        // a verify window often routes two tokens to one expert: read its rows once for both
        if (jb.n_tokens > 1) {
            std::stable_sort(misses.begin(), misses.end(), [](const miss & a, const miss & b) { return a.e < b.e; });
        }
        build_groups();

        const uint64_t routed = (uint64_t) jb.n_tokens*n_used;
        n_routed += routed;
        n_miss   += misses.size();
        routed_by_layer[il] += routed;
        miss_by_layer[il]   += misses.size();
        if (jb.tick < hits_by_tick.size()) {
            routed_by_tick[jb.tick] += routed;
            hits_by_tick[jb.tick]   += routed - misses.size();
        }
        if (observe != nullptr && routed <= sizeof(obs_ids)/sizeof(obs_ids[0])) {
            memcpy(obs_ids,   ids,   routed*sizeof(int32_t));
            memcpy(obs_slots, slots, routed*sizeof(int32_t));
            obs_n = (int) routed;
        }

        if (!misses.empty()) {
            const clk::time_point tq = clk::now();
            const auto from_float = ggml_get_type_traits_cpu(ggml_get_type_traits_cpu(bank->gate->type)->vec_dot_type)->from_float;
            for (int t = 0; t < jb.n_tokens; ++t) {
                if (tok_used[t]) {
                    from_float(act + (size_t) t*n_embd, act_q.data() + (size_t) t*act_q_row, n_embd);
                }
            }
            us_quant += (uint64_t) us_since(tq);

            if (!pin_misses(il, TOSH_HC_DECODE)) {
                return false;
            }
            next1.store(0, std::memory_order_relaxed);
            next2.store(0, std::memory_order_relaxed);
            kick();
            // the workers start on this layer's experts while the leader looks one layer ahead
            if (il + 1 < (int) banks.size()) read_ahead(il + 1);
            run(0);
            unpin_misses();
        } else if (il + 1 < (int) banks.size()) {
            read_ahead(il + 1);
        }

        n_jobs++;
        us_job += (uint64_t) us_since(t0);
        return true;
    }

    void leader_main() {
        raise_priority();
        uint64_t answered = 0;
        clk::time_point last = clk::now();
        bool was_active = false;
        while (!stop.load()) {
            if (!tosh_dmoe_wait_active(1000000)) {
                was_active = false;
                continue;
            }
            if (!was_active) {
                was_active = true;
                last = clk::now();
            }
            const uint64_t v = tosh_dmoe_event_get();
            if (v == answered || v % 2 == 0) {
                // a wide layer takes tens of milliseconds: past the decode cadence, poll gently
                if (us_since(last) > spin_us) {
                    std::this_thread::sleep_for(std::chrono::microseconds(100));
                } else {
                    DMOE_PAUSE();
                }
                continue;
            }
            last = clk::now();
            const tosh_dmoe_job * jb = tosh_dmoe_job_of(v);
            obs_n = 0;
            if (tosh_dmoe_failed() == nullptr) {
                if (__atomic_load_n(&jb->value, __ATOMIC_ACQUIRE) != v) {
                    tosh_dmoe_fail("the published value does not match its job");
                } else {
                    try {
                        if (jb->n_tokens > TOSH_DMOE_MAX_TOKENS) {
                            job_wide(jb->il, *jb);
                        } else {
                            tosh_dmoe_rare_interrupt();
                            job(jb->il, *jb);
                        }
                    } catch (...) {
                        tosh_dmoe_fail("the host executor threw");
                    }
                }
            }
            answered = v;
            tosh_dmoe_event_set(v + 1);
            last = clk::now();
            if (obs_n > 0 && jb->n_tokens <= TOSH_DMOE_MAX_TOKENS) {
                observe(observe_ctx, jb->il, obs_ids, obs_slots, obs_n, jb->n_tokens > 1, jb->tick);
            }
        }
    }

    // answers a publish nobody answered, so the GPU cannot wait forever
    void watchdog_main() {
        uint64_t last = 0;
        clk::time_point since = clk::now();
        while (!stop.load()) {
            std::this_thread::sleep_for(std::chrono::milliseconds(200));
            const uint64_t v = tosh_dmoe_event_get();
            const bool pending = tosh_dmoe_is_active() && (v % 2 == 1);
            if (!pending || v != last) {
                last  = v;
                since = clk::now();
                continue;
            }
            // a normal job takes under a millisecond; a starved one must still beat the driver
            if (us_since(since) > watchdog_us) {
                // the failure keeps the pointer, so the message must outlive this scope
                static const std::string why = format("the host executor did not answer within %.1f s", watchdog_us/1e6);
                tosh_dmoe_fail(why.c_str());
                tosh_dmoe_event_set(v + 1);
                since = clk::now();
            }
        }
    }

    // the GPU waits on these threads mid-graph: under a busy desktop they must not queue behind it
    static void raise_priority() {
#ifdef __APPLE__
        pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    }

    void start() {
        miss_by_layer   = std::vector<std::atomic<uint64_t>>(banks.size());
        routed_by_layer = std::vector<std::atomic<uint64_t>>(banks.size());
        hits_by_tick    = std::vector<std::atomic<uint32_t>>(1 << 16);
        routed_by_tick  = std::vector<std::atomic<uint32_t>>(1 << 16);
        for (int i = 1; i < n_threads; ++i) {
            helpers.emplace_back(&executor::helper_main, this, i);
        }
        leader   = std::thread(&executor::leader_main, this);
        watchdog = std::thread(&executor::watchdog_main, this);
    }

    void report() {
        if (n_wide.load() > 0) {
            fprintf(stderr, "tosh_dmoe: wide layers %llu, host rows %llu (%.1f per layer) over %.1f experts per layer, host time %.1f ms (%.1f us per layer)\n",
                    (unsigned long long) n_wide.load(), (unsigned long long) n_wide_miss.load(),
                    (double) n_wide_miss.load()/n_wide.load(), (double) n_wide_experts.load()/n_wide.load(),
                    us_wide.load()/1e3, (double) us_wide.load()/n_wide.load());
        }
        const uint64_t routed = n_routed.load();
        if (routed == 0) {
            return;
        }
        const uint64_t m = n_miss.load();
        fprintf(stderr, "tosh_dmoe: jobs %llu, routed %llu, host %llu, hit rate %.2f%%, read ahead %llu\n",
                (unsigned long long) n_jobs.load(), (unsigned long long) routed, (unsigned long long) m,
                100.0*(double) (routed - m)/(double) routed, (unsigned long long) n_ahead);
        fprintf(stderr, "tosh_dmoe: host time per job %.1f us (quantize %.1f, gate/up %.1f, down %.1f)\n",
                (double) us_job.load()/n_jobs.load(), (double) us_quant.load()/n_jobs.load(),
                (double) us_p1.load()/n_jobs.load(), (double) us_p2.load()/n_jobs.load());
        fprintf(stderr, "tosh_dmoe: host expert time %.3f ms total over %llu host experts (%.1f us each)\n",
                us_job.load()/1e3, (unsigned long long) m, m ? (double) us_job.load()/m : 0.0);
        static const int edges[] = { 1, 9, 17, 33, 65, 129, 257, 513, 1 << 16 };
        std::string line = "tosh_dmoe: hit rate by graph";
        for (int b = 0; b + 1 < (int) (sizeof(edges)/sizeof(edges[0])); ++b) {
            uint64_t h = 0, r = 0;
            for (int t = edges[b]; t < edges[b + 1] && t < (int) hits_by_tick.size(); ++t) {
                h += hits_by_tick[t]; r += routed_by_tick[t];
            }
            if (r) line += format(" | %d-%d %.1f%%", edges[b], edges[b + 1] - 1, 100.0*h/r);
        }
        fprintf(stderr, "%s\n", line.c_str());
        if (const char * p = getenv("TOSH_DMOE_TRACE_HITS")) {
            if (FILE * f = fopen(p, "w")) {
                for (size_t t = 1; t < hits_by_tick.size(); ++t) {
                    if (routed_by_tick[t]) fprintf(f, "%zu %u %u\n", t, hits_by_tick[t].load(), routed_by_tick[t].load());
                }
                fclose(f);
            }
        }
        if (getenv("TOSH_DMOE_STATS_LAYERS")) {
            for (size_t il = 0; il < banks.size(); ++il) {
                const uint64_t r = routed_by_layer[il].load();
                if (r) {
                    fprintf(stderr, "tosh_dmoe: layer %2zu hit %.1f%%\n", il, 100.0*(double) (r - miss_by_layer[il].load())/r);
                }
            }
        }
    }
};


// Online residency: LFU with a lazy per-graph decay, promotion when the decayed count clears a
// cost threshold, the lowest-scoring resident of the layer as victim. Slots move
// FREE -> PENDING (upload in flight, not in any table) -> RESIDENT (published) -> RETIRING
// (dropped from the next table) -> FREE once no graph can still read the old table.
struct policy {
    enum : uint8_t { FREE, PENDING, RESIDENT, RETIRING, PREFILL };

    struct slot {
        uint8_t  state  = FREE;
        int32_t  e      = -1;
        uint32_t gen    = 0;
        uint64_t ticket = 0;
        uint64_t retire_sync = UINT64_MAX;
        uint32_t since  = 0;
        uint32_t uses   = 0;
        clk::time_point issued;
        uint8_t  parts  = 0;              // PREFILL: which of gate/up/down have been copied
        uint64_t reserved_sync = 0;
        uint64_t encoded_sync  = 0;
        bool     claimed = false;         // PREFILL: taken by a bank upload rather than by admission
        bool     drop    = false;         // PREFILL: the expert was already resident or on its way
    };

    // promotions carried by prefill uploads, per layer, consumed by the three bank uploads
    std::mutex vp_mutex;
    std::vector<std::vector<std::pair<int32_t, int32_t>>> vplan;
    std::vector<int> claims;              // slots claimed by bank uploads, not yet seen by the policy
    uint64_t lent = 0, lent_dropped = 0;
    std::vector<uint8_t> vmask;
    std::vector<uint64_t> vmask_sync;    // completed graphs when a layer's mask last changed
    std::vector<uint8_t> full;           // parts a slot needs: 7, or 5 when gate and up share a bank
    std::vector<int> prefill_slots;
    uint64_t vpromotions = 0, vexpired = 0;

    struct obs {
        int16_t  e;
        int16_t  slot;
        uint8_t  multi;
        uint32_t tick;
    };

    static constexpr uint32_t ring_size = 1 << 14;

    int n_layer = 0, n_expert = 0, Q = 0;
    float  decay  = 0.98f;
    float  margin = 0.0f;
    double admit_fixed = 0.0;   // 0: estimate from the running costs
    int    budget = 8;
    long   stop_after = -1;
    bool   warm_prefill = true;
    float  tok_decay = 0.0f;              // per-token decay of prefill routing, 0 = one weight per chunk
    int    reserve   = 0;                 // free slots per layer kept for bank uploads during prefill
    int    reserve_tail = 0;              // how many of them go to the chunk's last tokens
    std::vector<int> retiring_n;
    std::vector<float> chunk_freq;        // routing of recent wide batches, halved per batch
    uint64_t reserve_evictions = 0;
    std::vector<float> tok_w;
    long   verify_every = 0;
    bool   defer_wide = true;
    bool   vram_promote = true;

    std::vector<const llama_tosh_dmoe_layer *> layers;
    const std::vector<host_bank> * banks = nullptr;
    const tosh_dmoe_uploader * up = nullptr;

    std::vector<float>    score;
    std::vector<uint32_t> last_tick;
    std::vector<int32_t>  resident_slot;
    std::vector<uint8_t>  pending;
    std::vector<uint32_t> evicted_tick;
    std::vector<slot>     slots;
    std::vector<std::vector<int>> free_list;
    std::vector<int>      pending_slots;   // l*Q + s
    std::vector<int>      retiring_slots;
    std::vector<float>    pow_decay;

    std::vector<std::vector<obs>> ring;
    std::unique_ptr<std::atomic<uint32_t>[]> head, tail;

    uint32_t budget_tick = 0;
    int      issued_tick = 0;   // promotions issued for the current graph

    uint64_t obs_next = 1;
    uint64_t obs_lost = 0, obs_prefill = 0;

    int32_t * tables = nullptr;
    int cur_version = 0;
    uint64_t version_retired[TOSH_DMOE_VERSIONS] = {};
    bool dirty = false;

    std::thread th;
    std::atomic<bool> stop{false};

    // stats
    std::atomic<uint64_t> dropped{0};
    uint64_t n_cycles = 0, promotions = 0, evictions = 0, repromotions = 0, failed = 0, publishes = 0;
    uint32_t last_stats = 0;
    uint64_t bytes_promoted = 0, upload_done = 0, verified = 0, verify_bad = 0, stale_hits = 0;
    double   us_cycle = 0, us_drain = 0, us_decide = 0, us_publish = 0, us_upload_call = 0;
    double   us_upload_wall = 0, us_upload_gpu = 0;
    uint64_t lifetime_sum = 0, lifetime_n = 0, uses_sum = 0;
    std::vector<uint32_t> lifetimes;
    std::vector<uint16_t> promo_by_tick = std::vector<uint16_t>(1 << 16);
    double   t_cpu_expert_us = 0;
    double   admit_used = 0;
    const executor * ex = nullptr;

    float decayed(size_t key, uint32_t now) const {
        const uint32_t dt = now - last_tick[key];
        return score[key]*(dt < pow_decay.size() ? pow_decay[dt] : 0.0f);
    }

    static void observe_cb(void * ctx, int il, const int32_t * ids, const int32_t * sl, int n, bool multi, uint32_t tick) {
        auto * p = (policy *) ctx;
        const uint32_t h = p->head[il].load(std::memory_order_relaxed);
        const uint32_t t = p->tail[il].load(std::memory_order_acquire);
        if (h - t + n > ring_size) {
            p->dropped += n;
            return;
        }
        auto & r = p->ring[il];
        for (int i = 0; i < n; ++i) {
            r[(h + i) & (ring_size - 1)] = { (int16_t) ids[i], (int16_t) sl[i], (uint8_t) multi, tick };
        }
        p->head[il].store(h + n, std::memory_order_release);
    }

    double admit_threshold() {
        if (ex != nullptr && ex->n_miss.load() > 0) {
            t_cpu_expert_us = (double) ex->us_job.load()/ex->n_miss.load();
        }
        if (admit_fixed > 0) return admit_fixed;
        // expected reuse must pay the upload: T_upload / (T_cpu - T_gpu). This charges the stage
        // copy as if it were serial, so it errs strict; the default is a fixed threshold.
        if (upload_done < 16 || promotions == 0 || t_cpu_expert_us <= 10) return 1.8;
        const double t_up = us_upload_gpu/upload_done + us_upload_call/promotions;
        return std::max(1.0, t_up/(t_cpu_expert_us - 7.0));
    }

    struct exchange { int il, s, e; };
    std::vector<exchange> exchanges;      // evicted this cycle: their bytes are still in the slot
    uint64_t exchanged = 0;
    double   us_exchange = 0;

    // bounded host cache: an expert leaving VRAM goes back to RAM, which holds what VRAM does not
    void demote(int il, int s) {
        const int e = slots[(size_t) il*Q + s].e;
        if (!tosh_hostcache_on() || e < 0) {
            return;
        }
        static const bool copy_back = getenv("TOSH_DMOE_HOST_EXCHANGE") == nullptr || atoi(getenv("TOSH_DMOE_HOST_EXCHANGE")) != 0;
        if (copy_back && up->download != nullptr) {
            if (tosh_hostcache_store(il, e)) exchanges.push_back({ il, s, e });
        } else {
            tosh_hostcache_prefetch(il, e);
        }
    }

    // the slots are not reused before a later cycle, so one copy per cycle takes them all
    void flush_exchanges() {
        if (exchanges.empty()) {
            return;
        }
        const clk::time_point t0 = clk::now();
        std::vector<tosh_dmoe_copy> copies;
        for (const exchange & x : exchanges) {
            const llama_tosh_dmoe_layer * dl = layers[x.il];
            const host_bank & b = (*banks)[x.il];
            int part = -1;
            for (auto [bank, arena] : { std::pair{ b.gate, dl->arena_gate }, { b.up, dl->arena_up }, { b.down, dl->arena_down } }) {
                part++;
                if (bank == nullptr) continue;
                copies.push_back({ tosh_hostcache_part(x.il, x.e, part), (size_t) x.s*arena->nb[2], arena, arena->nb[2] });
            }
        }
        const bool ok = up->download(up->ctx, copies.data(), (int) copies.size());
        for (const exchange & x : exchanges) tosh_hostcache_store_done(x.il, x.e, ok);
        if (ok) exchanged += exchanges.size();
        us_exchange += std::chrono::duration<double, std::micro>(clk::now() - t0).count();
        exchanges.clear();
    }

    bool verify_slot(int il, int s) {
        const llama_tosh_dmoe_layer * dl = layers[il];
        const host_bank & b = (*banks)[il];
        const int e = slots[(size_t) il*Q + s].e;
        bool ok = true;
        const int32_t eid = e;
        const bool hc = tosh_hostcache_on();
        if (hc && !tosh_hostcache_acquire(il, &eid, 1, TOSH_HC_VERIFY)) {
            return false;
        }
        int part = -1;
        for (auto [bank, arena] : { std::pair{ b.gate, dl->arena_gate }, { b.up, dl->arena_up }, { b.down, dl->arena_down } }) {
            part++;
            if (bank == nullptr) continue;
            std::vector<uint8_t> got(arena->nb[2]);
            ggml_backend_tensor_get(arena, got.data(), (size_t) s*arena->nb[2], got.size());
            const char * src = hc ? tosh_hostcache_part(il, e, part) : (const char *) bank->data + (size_t) e*bank->nb[2];
            ok = ok && memcmp(got.data(), src, got.size()) == 0;
        }
        if (hc) tosh_hostcache_release(il, &eid, 1);
        return ok;
    }

    void cycle() {
        const clk::time_point t0 = clk::now();
        const uint64_t sync = tosh_dmoe_completed_value();
        const uint32_t now  = (uint32_t) tosh_dmoe_tick();
        n_cycles++;

        // TOSH_DMOE_STATS_EVERY=<graphs>: a running line for long sessions
        static const uint32_t every = getenv("TOSH_DMOE_STATS_EVERY") ? (uint32_t) atoi(getenv("TOSH_DMOE_STATS_EVERY")) : 0;
        if (every > 0 && now >= last_stats + every) {
            last_stats = now;
            int res = 0, pend = 0, ret = 0, pre = 0;
            for (const slot & sl : slots) {
                res  += sl.state == RESIDENT;
                pend += sl.state == PENDING;
                ret  += sl.state == RETIRING;
                pre  += sl.state == PREFILL;
            }
            fprintf(stderr, "tosh_dmoe: graph %u | slots resident %d pending %d retiring %d prefill %d | promotions %llu evictions %llu re-promotions %llu | changed-slot hits %llu, failed %llu\n",
                    now, res, pend, ret, pre, (unsigned long long) promotions, (unsigned long long) evictions,
                    (unsigned long long) repromotions, (unsigned long long) stale_hits, (unsigned long long) failed);
        }

        // 1. observations since the last cycle
        std::vector<std::pair<float, size_t>> cands;
        for (int il = 0; il < n_layer; ++il) {
            if (layers[il] == nullptr) continue;
            const uint32_t h = head[il].load(std::memory_order_acquire);
            uint32_t t = tail[il].load(std::memory_order_relaxed);
            for (; t != h; ++t) {
                const obs & o = ring[il][t & (ring_size - 1)];
                const size_t key = (size_t) il*n_expert + o.e;
                if (warm_prefill || !o.multi) {
                    score[key] = decayed(key, o.tick) + 1.0f;
                    last_tick[key] = o.tick;
                }
                if (o.slot >= 0) {
                    // a hit must name the slot's occupant: slots are refilled only after every
                    // graph that could read the old table has finished, long after its observation
                    slot & sl = slots[(size_t) il*Q + o.slot];
                    if (sl.e == o.e) sl.uses++; else stale_hits++;
                } else if (resident_slot[key] < 0 && !pending[key]) {
                    cands.push_back({ 0.0f, key });
                }
            }
            tail[il].store(t, std::memory_order_release);
        }

        // routing of wide batches, once the graph that wrote it has completed
        const int32_t * obs_ring = (const int32_t *) tosh_dmoe_obs_ring();
        const uint64_t obs_head = tosh_dmoe_obs_head();
        while (obs_ring != nullptr && obs_next <= obs_head) {
            const tosh_dmoe_obs_record * rec = tosh_dmoe_obs_record_at(obs_next);
            if (rec == nullptr || !tosh_dmoe_obs_intact(rec)) { obs_lost++; obs_next++; continue; }
            if (tosh_dmoe_completed_value() <= rec->sync) break;
            if (rec->il < 0 || rec->il >= n_layer || layers[rec->il] == nullptr) { obs_next++; continue; }
            // one graph of T tokens stands in for T decode steps: weight each use by the mean decay
            const int T = rec->n_tokens;
            const float w = (1.0f - powf(decay, (float) T))/((float) T*(1.0f - decay));
            const int32_t * ids = obs_ring + rec->offset/sizeof(int32_t);
            if ((int) chunk_freq.size() != n_layer*n_expert) chunk_freq.assign((size_t) n_layer*n_expert, 0.0f);
            for (int e = 0; e < n_expert; ++e) chunk_freq[(size_t) rec->il*n_expert + e] *= 0.5f;
            for (int i = 0; i < rec->n_used*T; ++i) {
                if (ids[i] >= 0 && ids[i] < n_expert) chunk_freq[(size_t) rec->il*n_expert + ids[i]] += 1.0f;
            }
            if (warm_prefill && tok_decay > 0.0f) {
                // decode follows the end of the prompt: age the layer by T tokens, then weight each
                // token by its distance to the chunk's end
                const float age = powf(tok_decay, (float) T);
                for (int e = 0; e < n_expert; ++e) {
                    const size_t key = (size_t) rec->il*n_expert + e;
                    score[key] = decayed(key, rec->tick)*age;
                    last_tick[key] = rec->tick;
                }
                tok_w.resize(T);
                for (int t = 0; t < T; ++t) tok_w[t] = powf(tok_decay, (float) (T - 1 - t));
            }
            for (int i = 0; i < rec->n_used*T; ++i) {
                const int e = ids[i];
                if (e < 0 || e >= n_expert) continue;
                const size_t key = (size_t) rec->il*n_expert + e;
                if (warm_prefill && tok_decay > 0.0f) {
                    score[key] += tok_w[i/rec->n_used];
                } else if (warm_prefill) {
                    score[key] = decayed(key, rec->tick) + w;
                    last_tick[key] = rec->tick;
                }
                if (resident_slot[key] < 0 && !pending[key]) cands.push_back({ 0.0f, key });
            }
            obs_prefill += (uint64_t) rec->n_used*T;
            obs_next++;
        }
        const clk::time_point t1 = clk::now();

        // 2. uploads that landed become resident; failed ones free their slot
        for (size_t i = 0; i < pending_slots.size();) {
            const int gs = pending_slots[i];
            slot & sl = slots[gs];
            const int st = up->status(up->ctx, sl.ticket);
            if (st == 0) { ++i; continue; }
            const int il = gs / Q, s = gs % Q;
            const size_t key = (size_t) il*n_expert + sl.e;
            pending[key] = 0;
            if (st == 1) {
                sl.state = RESIDENT; sl.since = now; sl.uses = 0;
                resident_slot[key] = s;
                if (tosh_hostcache_on()) tosh_hostcache_drop(il, sl.e);
                upload_done++;
                us_upload_wall += std::chrono::duration<double, std::micro>(clk::now() - sl.issued).count();
                us_upload_gpu  += up->gpu_us(up->ctx, sl.ticket);
                if (verify_every > 0 && upload_done % verify_every == 0) {
                    verified++;
                    if (!verify_slot(il, s)) verify_bad++;
                }
                dirty = true;
            } else {
                failed++;
                sl.state = FREE; sl.e = -1;
                std::lock_guard<std::mutex> lock(vp_mutex);
                free_list[il].push_back(s);
            }
            pending_slots[i] = pending_slots.back();
            pending_slots.pop_back();
        }

        // 2b. promotions carried by prefill uploads: resident once the graph that copied them is done
        {
            std::lock_guard<std::mutex> lock(vp_mutex);
            for (const int gs : claims) {
                slot & sl = slots[gs];
                const size_t key = (size_t) (gs / Q)*n_expert + sl.e;
                if (resident_slot[key] >= 0 || pending[key]) {
                    sl.drop = true;
                    lent_dropped++;
                } else {
                    pending[key] = 1;
                }
            }
            claims.clear();
            // a batch can end with a layer's plan half carried, and decode brings no upload to finish
            // it: once nothing can still be copying, the mark goes and its slots expire below
            if (!tosh_dmoe_wide()) {
                for (int il = 0; il < n_layer; ++il) {
                    if (vmask[il] != 0 && sync > vmask_sync[il] + 16) vmask[il] = 0;
                }
            }
            for (size_t i = 0; i < prefill_slots.size();) {
                const int gs = prefill_slots[i];
                slot & sl = slots[gs];
                const int il = gs / Q, s = gs % Q;
                const size_t key = (size_t) il*n_expert + sl.e;
                if (sl.drop && ((sl.parts == full[il] && sync > sl.encoded_sync) || sync > sl.reserved_sync + 16) && vmask[il] == 0) {
                    // a duplicate: free the slot once nothing can still be copying into it
                    auto & lst = vplan[il];
                    lst.erase(std::remove_if(lst.begin(), lst.end(), [s](auto & q) { return q.second == s; }), lst.end());
                    sl.state = FREE; sl.e = -1; sl.parts = 0; sl.drop = false; sl.claimed = false;
                    free_list[il].push_back(s);
                } else if (sl.drop) {
                    ++i;
                    continue;
                } else if (sl.parts == full[il] && sync > sl.encoded_sync) {
                    sl.state = RESIDENT; sl.since = now; sl.uses = 0; sl.claimed = false;
                    resident_slot[key] = s;
                    if (tosh_hostcache_on()) tosh_hostcache_drop(il, sl.e);
                    pending[key] = 0;
                    dirty = true;
                    if (verify_every > 0 && vpromotions % verify_every == 0) {
                        verified++;
                        if (!verify_slot(il, s)) verify_bad++;
                    }
                } else if (sl.parts != full[il] && sync > sl.reserved_sync + 16 && vmask[il] == 0) {
                    // never fully carried: drop it, its layer's plan with it
                    auto & lst = vplan[il];
                    lst.erase(std::remove_if(lst.begin(), lst.end(), [s](auto & p) { return p.second == s; }), lst.end());
                    if (lst.empty()) vmask[il] = 0;
                    pending[key] = 0;
                    sl.state = FREE; sl.e = -1; sl.parts = 0; sl.claimed = false;
                    free_list[il].push_back(s);
                    vexpired++;
                } else {
                    ++i;
                    continue;
                }
                prefill_slots[i] = prefill_slots.back();
                prefill_slots.pop_back();
            }
        }

        // 3. retired slots nobody can still read
        for (size_t i = 0; i < retiring_slots.size();) {
            slot & sl = slots[retiring_slots[i]];
            if (sl.retire_sync != UINT64_MAX && sync >= sl.retire_sync + 2) {
                sl.state = FREE; sl.e = -1;
                std::lock_guard<std::mutex> lock(vp_mutex);
                free_list[retiring_slots[i] / Q].push_back(retiring_slots[i] % Q);
                retiring_slots[i] = retiring_slots.back();
                retiring_slots.pop_back();
            } else {
                ++i;
            }
        }

        // 4. admission and eviction
        admit_used = admit_threshold();
        // promotions wait for decode: during a wide batch they would compete with its uploads
        const bool wide   = tosh_dmoe_wide();
        const bool frozen = (stop_after >= 0 && (long) now > stop_after) || (defer_wide && wide);
        if (!frozen && !cands.empty()) {
            std::sort(cands.begin(), cands.end(), [](auto & a, auto & b) { return a.second < b.second; });
            cands.erase(std::unique(cands.begin(), cands.end(), [](auto & a, auto & b) { return a.second == b.second; }), cands.end());
            for (auto & c : cands) c.first = decayed(c.second, now);
            std::sort(cands.begin(), cands.end(), [](auto & a, auto & b) { return a.first > b.first; });
            if (now != budget_tick) {
                budget_tick = now;
                issued_tick = 0;
            }
            int retired = 0;
            for (auto & [sc, key] : cands) {
                if (sc < admit_used || issued_tick >= budget) break;
                const int il = (int) (key / n_expert), e = (int) (key % n_expert);
                if (resident_slot[key] >= 0 || pending[key]) continue;
                bool no_free;
                {
                    std::lock_guard<std::mutex> lock(vp_mutex);
                    no_free = free_list[il].empty();
                }
                if (no_free) {
                    if (retired >= budget) continue;
                    int victim = -1; float vs = 0;
                    for (int s = 0; s < Q; ++s) {
                        const slot & sl = slots[(size_t) il*Q + s];
                        if (sl.state != RESIDENT) continue;
                        const float v = decayed((size_t) il*n_expert + sl.e, now);
                        if (victim < 0 || v < vs) { victim = s; vs = v; }
                    }
                    if (victim < 0 || vs + margin >= sc) continue;
                    demote(il, victim);
                    slot & sl = slots[(size_t) il*Q + victim];
                    const size_t vkey = (size_t) il*n_expert + sl.e;
                    resident_slot[vkey] = -1;
                    evicted_tick[vkey] = now;
                    lifetime_sum += now - sl.since; lifetime_n++; uses_sum += sl.uses;
                    if (lifetimes.size() < (1 << 20)) lifetimes.push_back(now - sl.since);
                    sl.state = RETIRING; sl.retire_sync = UINT64_MAX;
                    retiring_slots.push_back(il*Q + victim);
                    evictions++; retired++;
                    dirty = true;
                    continue;   // the candidate takes a slot once one frees
                }
                // during a wide batch the upload of the layer's bank carries the expert over
                if (wide && vram_promote) {
                    std::lock_guard<std::mutex> lock(vp_mutex);
                    if (vmask[il] != 0 || free_list[il].empty()) continue;   // half consumed, or lent meanwhile
                    const int s = free_list[il].back();
                    free_list[il].pop_back();
                    slot & sl = slots[(size_t) il*Q + s];
                    sl.state = PREFILL; sl.e = e; sl.gen++; sl.parts = 0; sl.reserved_sync = sync;
                    vplan[il].push_back({ e, s });
                    prefill_slots.push_back(il*Q + s);
                    pending[key] = 1;
                    vpromotions++; issued_tick++;
                    continue;
                }
                int s;
                {
                    std::lock_guard<std::mutex> lock(vp_mutex);
                    if (free_list[il].empty()) continue;
                    s = free_list[il].back();
                    free_list[il].pop_back();
                }
                slot & sl = slots[(size_t) il*Q + s];
                const llama_tosh_dmoe_layer * dl = layers[il];
                const host_bank & b = (*banks)[il];
                tosh_dmoe_copy copies[3];
                int n_copies = 0;
                const int32_t eid = e;
                const bool hc = tosh_hostcache_on();
                if (hc && !tosh_hostcache_acquire(il, &eid, 1, TOSH_HC_PROMOTE)) {
                    tosh_dmoe_fail(tosh_hostcache_error());
                }
                int part = -1;
                for (auto [bank, arena] : { std::pair{ b.gate, dl->arena_gate }, { b.up, dl->arena_up }, { b.down, dl->arena_down } }) {
                    part++;
                    if (bank == nullptr) continue;
                    const char * src = hc ? tosh_hostcache_part(il, e, part) : (const char *) bank->data + (size_t) e*bank->nb[2];
                    copies[n_copies++] = { arena, (size_t) s*arena->nb[2], src, arena->nb[2] };
                }
                sl.e = e; sl.gen++; sl.issued = clk::now();
                const clk::time_point tu = clk::now();
                sl.ticket = up->upload(up->ctx, copies, n_copies);
                if (hc) tosh_hostcache_release(il, &eid, 1);
                us_upload_call += std::chrono::duration<double, std::micro>(clk::now() - tu).count();
                if (sl.ticket == 0) {
                    failed++;
                    sl.state = FREE; sl.e = -1;
                    std::lock_guard<std::mutex> lock(vp_mutex);
                    free_list[il].push_back(s);
                    continue;
                }
                sl.state = PENDING;
                pending[key] = 1;
                pending_slots.push_back(il*Q + s);
                promotions++; issued_tick++;
                if (now < promo_by_tick.size()) promo_by_tick[now]++;
                for (int c = 0; c < n_copies; ++c) bytes_promoted += copies[c].size;
                if (evicted_tick[key] != 0 && now - evicted_tick[key] < 64) repromotions++;
            }
        }
        // 4b. during prefill keep a few free slots per layer for the bank uploads to lend
        if (reserve > 0 && wide) {
            retiring_n.assign(n_layer, 0);
            for (const int gs : retiring_slots) retiring_n[gs / Q]++;
            for (int il = 0; il < n_layer; ++il) {
                if (layers[il] == nullptr) continue;
                int n_free;
                {
                    std::lock_guard<std::mutex> lock(vp_mutex);
                    n_free = (int) free_list[il].size();
                }
                for (int need = reserve - n_free - retiring_n[il]; need > 0; --need) {
                    // the least routed resident of recent batches: later chunks will not miss it
                    int victim = -1; float vs = 0;
                    for (int s = 0; s < Q; ++s) {
                        const slot & sl = slots[(size_t) il*Q + s];
                        if (sl.state != RESIDENT) continue;
                        const size_t k = (size_t) il*n_expert + sl.e;
                        const float v = (chunk_freq.empty() ? 0.0f : chunk_freq[k]) + 1e-3f*decayed(k, now);
                        if (victim < 0 || v < vs) { victim = s; vs = v; }
                    }
                    if (victim < 0) break;
                    demote(il, victim);
                    slot & sl = slots[(size_t) il*Q + victim];
                    const size_t vkey = (size_t) il*n_expert + sl.e;
                    resident_slot[vkey] = -1;
                    evicted_tick[vkey] = now;
                    lifetime_sum += now - sl.since; lifetime_n++; uses_sum += sl.uses;
                    sl.state = RETIRING; sl.retire_sync = UINT64_MAX;
                    retiring_slots.push_back(il*Q + victim);
                    evictions++; reserve_evictions++;
                    dirty = true;
                }
            }
        }
        flush_exchanges();
        const clk::time_point t2 = clk::now();

        // 5. publish a table with what is resident now, into a version no graph can be reading
        if (dirty) {
            int next = -1;
            for (int k = 1; k < TOSH_DMOE_VERSIONS; ++k) {
                const int v = (cur_version + k) % TOSH_DMOE_VERSIONS;
                if (sync >= version_retired[v] + 2) { next = v; break; }
            }
            if (next >= 0) {
                for (int il = 0; il < n_layer; ++il) {
                    if (layers[il] == nullptr) continue;
                    int32_t * tb = tables + ((size_t) next*TOSH_DMOE_MAX_LAYERS + il)*n_expert;
                    for (int e = 0; e < n_expert; ++e) tb[e] = -1;
                    for (int s = 0; s < Q; ++s) {
                        const slot & sl = slots[(size_t) il*Q + s];
                        if (sl.state == RESIDENT) tb[sl.e] = s;
                    }
                }
                std::atomic_thread_fence(std::memory_order_release);
                tosh_dmoe_publish_version(next);
                // counted after the switch: a graph submitted before it is done two syncs later,
                // however many syncs passed while this cycle ran
                const uint64_t sync_pub = tosh_dmoe_completed_value();
                version_retired[cur_version] = sync_pub;
                cur_version = next;
                for (int gs : retiring_slots) {
                    if (slots[gs].retire_sync == UINT64_MAX) slots[gs].retire_sync = sync_pub;
                }
                publishes++;
                dirty = false;
            }
        }
        const clk::time_point t3 = clk::now();
        us_drain   += std::chrono::duration<double, std::micro>(t1 - t0).count();
        us_decide  += std::chrono::duration<double, std::micro>(t2 - t1).count();
        us_publish += std::chrono::duration<double, std::micro>(t3 - t2).count();
        us_cycle   += std::chrono::duration<double, std::micro>(t3 - t0).count();
    }

    void main() {
        uint64_t seen = tosh_dmoe_completed_value();
        while (!stop.load()) {
            // uploads in flight are polled; with nothing moving only a finished graph brings work
            const bool moving = !pending_slots.empty() || !prefill_slots.empty() || !retiring_slots.empty();
            tosh_dmoe_wait_completed(seen, moving ? 2000 : 500000);
            seen = tosh_dmoe_completed_value();
            try {
                cycle();
            } catch (...) {
                tosh_dmoe_fail("the residency policy threw");
            }
        }
    }

    void report(uint64_t graphs) {
        const double g = std::max<uint64_t>(1, graphs);
        int used = 0;
        for (const slot & s : slots) used += s.state == RESIDENT;
        std::sort(lifetimes.begin(), lifetimes.end());
        fprintf(stderr, "tosh_dmoe: policy decay %.3f, admit %.2f (measured upload %.1f us wall, %.1f us gpu; cpu expert %.1f us), budget %d/graph, margin %.2f\n",
                decay, admit_used, upload_done ? us_upload_wall/upload_done : 0.0, upload_done ? us_upload_gpu/upload_done : 0.0,
                t_cpu_expert_us, budget, margin);
        fprintf(stderr, "tosh_dmoe: per graph: promotions %.2f, evictions %.2f, MiB promoted %.2f | re-promotions %.1f%% | failed %llu, dropped obs %llu, hits on a changed slot %llu\n",
                promotions/g, evictions/g, bytes_promoted/1048576.0/g, promotions ? 100.0*repromotions/promotions : 0.0,
                (unsigned long long) failed, (unsigned long long) dropped.load(), (unsigned long long) stale_hits);
        fprintf(stderr, "tosh_dmoe: slots %d of %d resident | evicted lifetime mean %.1f median %u graphs, uses per evicted residency %.1f\n",
                used, (int) slots.size(), lifetime_n ? (double) lifetime_sum/lifetime_n : 0.0,
                lifetimes.empty() ? 0u : lifetimes[lifetimes.size()/2], lifetime_n ? (double) uses_sum/lifetime_n : 0.0);
        uint64_t live_uses = 0;
        for (const slot & s : slots) if (s.state == RESIDENT) live_uses += s.uses;
        fprintf(stderr, "tosh_dmoe: uses per promotion (evicted + live) %.1f\n",
                promotions ? (double) (uses_sum + live_uses)/promotions : 0.0);
        fprintf(stderr, "tosh_dmoe: policy thread per graph %.1f us (drain %.1f, decide %.1f incl. upload calls %.1f, publish %.1f), %llu publishes, %llu cycles\n",
                us_cycle/g, us_drain/g, us_decide/g, us_upload_call/g, us_publish/g, (unsigned long long) publishes, (unsigned long long) n_cycles);
        if (const char * p = getenv("TOSH_DMOE_TRACE_PROMO")) {
            if (FILE * f = fopen(p, "w")) {
                for (size_t t = 1; t < promo_by_tick.size() && t <= graphs; ++t) fprintf(f, "%zu %u\n", t, promo_by_tick[t]);
                fclose(f);
            }
        }
        if (exchanged > 0) {
            fprintf(stderr, "tosh_dmoe: experts copied from VRAM back to RAM %llu, %.1f us each\n",
                    (unsigned long long) exchanged, us_exchange/exchanged);
        }
        if (vpromotions + lent > 0) {
            fprintf(stderr, "tosh_dmoe: promotions carried by prefill uploads %llu, expired %llu | free slots lent to prefill uploads %llu, duplicates dropped %llu\n",
                    (unsigned long long) vpromotions, (unsigned long long) vexpired, (unsigned long long) lent, (unsigned long long) lent_dropped);
            fprintf(stderr, "tosh_dmoe: evictions to keep %d free slots per layer during prefill %llu\n", reserve, (unsigned long long) reserve_evictions);
        }
        if (obs_prefill + obs_lost > 0) {
            fprintf(stderr, "tosh_dmoe: prefill routing observed %llu selections, %llu records lost\n",
                    (unsigned long long) obs_prefill, (unsigned long long) obs_lost);
        }
        if (verify_every > 0) {
            fprintf(stderr, "tosh_dmoe: verified %llu promoted slots against the host bank, %llu mismatched\n",
                    (unsigned long long) verified, (unsigned long long) verify_bad);
        }
    }
};

static policy * g_policy = nullptr;

static executor * g_exec = nullptr;

static ggml_backend_buffer_t g_arena_buf = nullptr;

// bank data pointer -> (layer, tensor), for prefill uploads that read residents from the arena
struct bank_ref { int il; int which; const ggml_tensor * bank; };
static std::unordered_map<const void *, bank_ref> g_bank_refs;
static std::vector<const llama_tosh_dmoe_layer *> g_layer_by_il;
static std::vector<std::vector<int32_t>> g_fixed_slots;
static int g_n_expert = 0;

static bool dmoe_bank_plan(const void * host, size_t size, tosh_dmoe_bank_plan * plan) {
    const auto it = g_bank_refs.find(host);
    if (it == g_bank_refs.end() || ggml_nbytes(it->second.bank) != size) {
        return false;
    }
    const bank_ref & r = it->second;
    const llama_tosh_dmoe_layer * dl = r.il < (int) g_layer_by_il.size() ? g_layer_by_il[r.il] : nullptr;
    if (dl == nullptr) {
        return false;
    }
    plan->arena        = r.which == 0 ? dl->arena_gate : r.which == 1 ? dl->arena_up : dl->arena_down;
    plan->n_expert     = g_n_expert;
    plan->expert_bytes = r.bank->nb[2];
    plan->il           = r.il;
    plan->which        = r.which;
    plan->promo_expert = nullptr;
    plan->promo_slot   = nullptr;
    plan->n_promo      = 0;
    int n_rows = 0;
    plan->rows = tosh_dmoe_chunk_rows(&n_rows);
    if (n_rows != g_n_expert) {
        plan->rows = nullptr;
    }
    const int32_t * rows = plan->rows;
    if (g_policy != nullptr) {
        plan->slot_of = g_policy->tables + ((size_t) tosh_dmoe_published_version()*TOSH_DMOE_MAX_LAYERS + r.il)*g_n_expert;
        static std::vector<int32_t> pe, ps;
        static std::vector<std::pair<int32_t, int32_t>> want;
        static const bool lend = getenv("TOSH_DMOE_LEND") == nullptr || atoi(getenv("TOSH_DMOE_LEND")) != 0;
        policy & p = *g_policy;
        std::lock_guard<std::mutex> lock(p.vp_mutex);
        auto & lst = p.vplan[r.il];
        auto & fl  = p.free_list[r.il];
        if (lend && rows && p.vmask[r.il] == 0 && !fl.empty()) {
            // free slots take the chunk's busiest host experts: the upload brings them into VRAM anyway
            // the chunk's last tokens first, since decode continues from them; then its busiest experts
            const float * tail = tosh_dmoe_chunk_tail();
            want.clear();
            for (int e = 0; e < g_n_expert; ++e) {
                if (rows[e] > 0 && plan->slot_of[e] < 0) want.push_back({ rows[e], e });
            }
            for (auto [e, s] : lst) {
                for (auto & w : want) if (w.second == e) w.first = 0;
            }
            std::sort(want.begin(), want.end(), [](auto & a, auto & b) { return a.first > b.first; });
            if (tail != nullptr && p.reserve_tail > 0) {
                std::stable_sort(want.begin(), want.end(), [tail](auto & a, auto & b) {
                    return (a.first > 0 ? tail[a.second] : -1.0f) > (b.first > 0 ? tail[b.second] : -1.0f);
                });
                const size_t k = std::min<size_t>(p.reserve_tail, want.size());
                std::sort(want.begin() + k, want.end(), [](auto & a, auto & b) { return a.first > b.first; });
            }
            const uint64_t sync = tosh_dmoe_completed_value();
            for (auto [n, e] : want) {
                if (fl.empty()) break;
                if (n == 0) continue;
                const int s = fl.back();
                fl.pop_back();
                const int gs = r.il*p.Q + s;
                auto & sl = p.slots[gs];
                sl.state = policy::PREFILL; sl.e = e; sl.gen++; sl.parts = 0; sl.reserved_sync = sync;
                sl.claimed = true; sl.drop = false;
                lst.push_back({ e, s });
                p.prefill_slots.push_back(gs);
                p.claims.push_back(gs);
                p.lent++;
            }
        }
        if (!lst.empty() && !(p.vmask[r.il] & (1u << r.which))) {
            pe.clear(); ps.clear();
            for (auto [e, s] : lst) {
                // only a routed expert lands in the upload; the others wait for a chunk that routes them
                if (rows && rows[e] == 0) continue;
                pe.push_back(e); ps.push_back(s);
                auto & sl = p.slots[(size_t) r.il*p.Q + s];
                sl.parts |= (uint8_t) (1u << r.which);
                if (sl.parts == p.full[r.il]) sl.encoded_sync = tosh_dmoe_completed_value();
            }
            plan->promo_expert = pe.data();
            plan->promo_slot   = ps.data();
            plan->n_promo      = (int) pe.size();
            p.vmask[r.il] |= (uint8_t) (1u << r.which);
            p.vmask_sync[r.il] = tosh_dmoe_completed_value();
            if (p.vmask[r.il] == p.full[r.il]) {
                if (rows) {
                    lst.erase(std::remove_if(lst.begin(), lst.end(), [rows](auto & q) { return rows[q.first] > 0; }), lst.end());
                } else {
                    lst.clear();
                }
                p.vmask[r.il] = 0;
            }
        }
    } else {
        plan->slot_of = g_fixed_slots[r.il].data();
    }
    return true;
}

static void dmoe_report_at_exit() {
    if (g_exec && getenv("TOSH_DMOE_STATS")) {
        g_exec->report();
        if (g_policy) {
            g_policy->report(tosh_dmoe_tick());
        }
    }
}

}

bool llama_tosh_dmoe_accept(const llama_tosh_dmoe_layer * dl, int act, float alpha, float limit, bool gate_b, bool up_b) {
    if (g_exec == nullptr || dl == nullptr || dl->il < 0 || dl->il >= (int) g_exec->banks.size()) {
        return false;
    }
    host_bank & b = g_exec->banks[dl->il];
    if (b.gate == nullptr || gate_b != !b.gate_b.empty() || up_b != !b.up_b.empty()) {
        return false;
    }
    if (act < 0 || act > 2 || (act == 1 && !(alpha > 0.0f && limit > 0.0f))) {
        return false;
    }
    // graphs are built on one thread; the first one fixes the layer's activation
    if (b.act.load() < 0) {
        b.alpha = alpha;
        b.limit = limit;
        b.act.store(act);
    }
    return b.act.load() == act && (act == 0 || (b.alpha == alpha && b.limit == limit));
}

int llama_tosh_dmoe_rare_rows() {
    // a number fixes the threshold; auto starts low and lets the controller move it
    static const int rows = [] {
        const char * v = getenv("TOSH_DMOE_RARE_ROWS");
        if (v == nullptr) return 0;
        const bool adapt = strcmp(v, "auto") == 0;
        const int r = adapt ? 4 : atoi(v);
        tosh_dmoe_rare_rows_init(r, 24, adapt);
        return r;
    }();
    return g_exec != nullptr && tosh_dmoe_compact() ? rows : 0;
}

int llama_tosh_dmoe_rare_tail() {
    static const int n = getenv("TOSH_DMOE_RARE_TAIL") ? atoi(getenv("TOSH_DMOE_RARE_TAIL")) : 32;
    return std::max(n, 0);
}

int llama_tosh_dmoe_wide_min_tokens() {
    // every batch above the host path goes to the GPU, which offloads from the same size
    return TOSH_DMOE_MAX_TOKENS + 1;
}

bool llama_tosh_dmoe_observe_prefill() {
    static const bool on = getenv("TOSH_DMOE_OBSERVE_PREFILL") == nullptr || atoi(getenv("TOSH_DMOE_OBSERVE_PREFILL")) != 0;
    return on && g_policy != nullptr;
}

void llama_tosh_dmoe_shutdown() {
    tosh_dmoe_set_bank_planner(nullptr);
    g_bank_refs.clear();
    g_layer_by_il.clear();
    if (g_policy != nullptr && g_policy->th.joinable()) {
        g_policy->stop = true;
        g_policy->th.join();
    }
    if (g_exec != nullptr) {
        g_exec->stop = true;
        // idle workers sleep until the executor turns active: wake them to see the stop
        tosh_dmoe_set_active(true);
        {
            std::lock_guard<std::mutex> lock(g_exec->work_mutex);
            g_exec->work_cv.notify_all();
        }
        for (auto & t : g_exec->helpers) if (t.joinable()) t.join();
        if (g_exec->leader.joinable())   g_exec->leader.join();
        if (g_exec->watchdog.joinable()) g_exec->watchdog.join();
        tosh_dmoe_set_active(false);
    }
    // the stats stay for the exit report; the arena and the layer map go with the model
    g_layers.clear();
    if (g_arena_buf != nullptr) {
        ggml_backend_buffer_free(g_arena_buf);
        g_arena_buf = nullptr;
    }
}

static void dmoe_init(std::vector<llama_layer> & layers, const llama_hparams & hparams, const char * path, double cache_mib);

// A cache size is a request, not a requirement: a model or budget it cannot serve keeps running
// without mixed execution. A residency file is a test setup and still fails the load.
// TOSH_DMOE_STATUS_FILE: what mixed execution ended up doing, for whoever planned it
static void dmoe_status(bool on, const char * code, const std::string & detail, double arena_mib) {
    const char * path = getenv("TOSH_DMOE_STATUS_FILE");
    if (path == nullptr) {
        return;
    }
    std::string d;
    for (char c : detail) d += (c == '"' || c == '\\') ? ' ' : c;
    if (FILE * f = fopen(path, "w")) {
        fprintf(f, "{\"dmoe\": \"%s\", \"code\": \"%s\", \"detail\": \"%s\", \"arena_mib\": %.0f, \"bank_locked\": %s}\n",
                on ? "on" : "off", code, d.c_str(), arena_mib, on ? "true" : "false");
        fclose(f);
    }
}

// The host executor reads the bank on every miss; a page the kernel took back stalls the GPU
// past its watchdog, so the bank is locked or mixed execution does not start.
static void dmoe_lock_bank(const std::vector<llama_layer> & layers) {
    static bool locked = false;   // an arena rebuild keeps the same bank
    if (locked) {
        return;
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    const bool test_fail = getenv("TOSH_DMOE_TEST_MLOCK_FAIL") != nullptr;
    size_t bytes = 0;
    for (const llama_layer & L : layers) {
        for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
            if (t == nullptr || t->data == nullptr) continue;
            const uintptr_t a = (uintptr_t) t->data & ~(uintptr_t) (page - 1);
            const uintptr_t b = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(uintptr_t) (page - 1);
            char err[256];
            if (test_fail || !tosh_mlock_paced((void *) a, b - a, err, sizeof(err))) {
                throw std::runtime_error(test_fail ? format("DMOE_HOST_BANK_NOT_LOCKABLE: mlock of %.0f MiB of the expert bank failed after %.0f MiB (%s)",
                                                           (b - a)/1048576.0, bytes/1048576.0, strerror(EAGAIN))
                                                   : format("%s, expert bank at %.0f MiB", err, bytes/1048576.0));
            }
            bytes += b - a;
        }
    }
    locked = true;
    LLAMA_LOG_INFO("%s: expert bank locked in host memory, %.0f MiB\n", __func__, bytes/1048576.0);
}

// EXPERIMENTAL: the bank tensors were left unread; the cache reads experts from the file. The
// untouched bank pages are made inaccessible so a reader that bypasses the cache fails loudly.
static void dmoe_start_hostcache(const std::vector<llama_layer> & layers, const llama_hparams & hparams) {
    static bool started = false;
    if (started) {
        return;
    }
    const char * ob = getenv("GGML_OP_OFFLOAD_MIN_BATCH");
    if (ob == nullptr || atoi(ob) > TOSH_DMOE_MAX_TOKENS + 1) {
        throw std::runtime_error(format("the bounded host cache needs GGML_OP_OFFLOAD_MIN_BATCH=%d: smaller batches would compute experts from the bank on the CPU backend", TOSH_DMOE_MAX_TOKENS + 1));
    }
    int n_moe = 0;
    for (int il = 0; il < (int) layers.size(); ++il) {
        const llama_layer & L = layers[il];
        if (!dmoe_gate(L)) continue;
        // parts keep fixed places: 0 gate (or fused gate/up), 1 up, 2 down; biases and scales stay loaded
        int part = 0;
        for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
            if (t != nullptr && !tosh_hostcache_bind(il, part, t->data, t->nb[2], (int) hparams.n_expert)) {
                throw std::runtime_error(tosh_hostcache_error());
            }
            part++;
        }
        n_moe++;
    }
    if (n_moe == 0 || !tosh_hostcache_start((int) layers.size(), (int) hparams.n_expert)) {
        throw std::runtime_error(tosh_hostcache_error() ? tosh_hostcache_error() : "no expert bank");
    }
    const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    for (const llama_layer & L : layers) {
        for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
            if (t == nullptr) continue;
            const uintptr_t a = ((uintptr_t) t->data + page - 1) & ~(uintptr_t) (page - 1);
            const uintptr_t b = ((uintptr_t) t->data + ggml_nbytes(t)) & ~(uintptr_t) (page - 1);
            if (b > a) mprotect((void *) a, b - a, PROT_NONE);
        }
    }
    atexit(tosh_hostcache_report);
    started = true;
}

static void dmoe_init_or_off(std::vector<llama_layer> & layers, const llama_hparams & hparams, const char * path, double cache_mib) {
    if (path != nullptr) {
        dmoe_init(layers, hparams, path, cache_mib);
        return;
    }
    try {
        if (tosh_hostcache_on()) {
            dmoe_start_hostcache(layers, hparams);
        } else {
            dmoe_lock_bank(layers);
            if (getenv("TOSH_DMOE_HOST_BYTECHECK")) {
                // the cache reads the loader's own offsets: check them against the bank it loaded
                int n = 0, bad = 0;
                for (const llama_layer & L : layers) {
                    for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
                        if (t == nullptr) continue;
                        for (int e : { 0, 1, 37, 128, 200, (int) t->ne[2] - 1 }) {
                            if (e >= t->ne[2]) continue;
                            n++;
                            bad += !tosh_hostcache_bytecheck(t->data, t->nb[2], e);
                        }
                    }
                }
                LLAMA_LOG_INFO("%s: expert bytes read from the file against the loaded bank: %d checked in %d file(s), %d differ\n",
                               __func__, n, tosh_hostcache_n_files(), bad);
            }
        }
        dmoe_init(layers, hparams, path, cache_mib);
        dmoe_status(true, "DMOE_ON", "", g_arena_buf ? ggml_backend_buffer_get_size(g_arena_buf)/1048576.0 : 0.0);
    } catch (const std::exception & err) {
        if (tosh_hostcache_on()) {
            throw;   // the bank holds no data: there is no path without the cache
        }
        LLAMA_LOG_WARN("%s: mixed expert execution stays off: %s\n", __func__, err.what());
        llama_tosh_dmoe_shutdown();
        delete g_policy;
        g_policy = nullptr;
        const std::string what = err.what();
        const bool lock = what.rfind("DMOE_HOST_BANK_NOT_LOCKABLE", 0) == 0;
        dmoe_status(false, lock ? "DMOE_HOST_BANK_NOT_LOCKABLE" : "DMOE_UNSUPPORTED_MODEL", what, 0.0);
    }
}

// TOSH_DMOE_CACHE_MIB=auto: the arena waits for the context, so it is what the card has left
static std::vector<llama_layer> * g_defer_layers = nullptr;
static const llama_hparams *      g_defer_hparams = nullptr;

void llama_tosh_dmoe_init(std::vector<llama_layer> & layers, const llama_hparams & hparams) {
    const char * path      = getenv("TOSH_DMOE_RESIDENCY");
    const char * cache_mib = getenv("TOSH_DMOE_CACHE_MIB");
    if (tosh_hostcache_on() && (cache_mib == nullptr || path != nullptr)) {
        throw std::runtime_error("the bounded host cache needs TOSH_DMOE_CACHE_MIB (a VRAM arena) and no residency file");
    }
    if ((path == nullptr && cache_mib == nullptr) || g_exec != nullptr) {
        return;
    }
    if (path == nullptr && strcmp(cache_mib, "auto") == 0) {
        g_defer_layers  = &layers;
        g_defer_hparams = &hparams;
        return;
    }
    dmoe_init_or_off(layers, hparams, path, cache_mib ? atof(cache_mib) : 0.0);
}

bool llama_tosh_dmoe_budget_pending() {
    return g_defer_layers != nullptr && g_exec == nullptr;
}

// an automatic arena is sized again when a context with another footprint is created alone
static std::vector<llama_layer> * g_auto_layers  = nullptr;
static const llama_hparams *      g_auto_hparams = nullptr;
static int                        g_contexts     = 0;

void llama_tosh_dmoe_context_opened() { g_contexts++; }
void llama_tosh_dmoe_context_closed() { g_contexts--; }

bool llama_tosh_dmoe_rebudget_possible() {
    return g_auto_layers != nullptr && g_contexts == 0;
}

// Drops the running policy, executor and arena once nothing can use them: no context is alive,
// and uploads still in flight into the arena are waited for.
static void dmoe_teardown() {
    if (g_policy != nullptr) {
        g_policy->stop = true;
        if (g_policy->th.joinable()) g_policy->th.join();
        for (const int gs : g_policy->pending_slots) {
            const auto & sl = g_policy->slots[gs];
            for (int i = 0; i < 1000 && g_policy->up->status(g_policy->up->ctx, sl.ticket) == 0; ++i) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
            }
        }
    }
    llama_tosh_dmoe_shutdown();
    delete g_policy;
    g_policy = nullptr;
    delete g_exec;
    g_exec = nullptr;
}

void llama_tosh_dmoe_budget(size_t gpu_bytes) {
    const bool first = g_defer_layers != nullptr;
    std::vector<llama_layer> & layers = first ? *g_defer_layers : *g_auto_layers;
    const llama_hparams & hparams = first ? *g_defer_hparams : *g_auto_hparams;
    g_defer_layers = nullptr;
    g_auto_layers  = &layers;
    g_auto_hparams = &hparams;
    // the running arena is part of what the backend holds, not of what this context needs
    const size_t arena_now = g_arena_buf ? ggml_backend_buffer_get_size(g_arena_buf) : 0;
    gpu_bytes -= std::min(gpu_bytes, arena_now);

    auto envmib = [](const char * k, double d) { const char * v = getenv(k); return (v ? atof(v) : d)*1048576.0; };
    size_t free0 = 0, total = 0;
    tosh_dmoe_vram(&free0, &total);
    // the bank uploads of prefill take three slots the size of the largest host bank
    size_t bank_max = 0;
    for (const llama_layer & L : layers) {
        for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
            if (t) bank_max = std::max(bank_max, ggml_nbytes(t));
        }
    }
    const double reserve  = envmib("TOSH_DMOE_RESERVE_MIB", 2560);
    const double overhead = envmib("TOSH_DMOE_OVERHEAD_MIB", 64);
    const double prefetch = 3.0*(double) bank_max;
    const double arena    = (double) free0 - reserve - overhead - prefetch - (double) gpu_bytes;
    const double M = 1048576.0;
    LLAMA_LOG_INFO("%s: VRAM free at start %.0f MiB of %.0f; model, context and compute %.0f, prefill slots %.0f, "
                   "reserve %.0f, overhead %.0f -> expert arena %.0f MiB\n", __func__, free0/M, total/M, gpu_bytes/M,
                   prefetch/M, reserve/M, overhead/M, std::max(0.0, arena)/M);
    if (!first) {
        const double delta = arena - (double) arena_now;
        if (fabs(delta) < 256.0*M) {
            return;
        }
        const auto t0 = clk::now();
        dmoe_teardown();
        if (arena >= 256.0*M) {
            dmoe_init_or_off(layers, hparams, nullptr, arena/M);
        }
        LLAMA_LOG_INFO("%s: expert arena rebuilt for this context, %.0f -> %.0f MiB in %.1f ms\n", __func__,
                       arena_now/M, std::max(0.0, arena)/M, std::chrono::duration<double, std::milli>(clk::now() - t0).count());
        return;
    }
    // a planner passes the smallest arena it judged useful for this model
    const double min_arena = std::max(256.0*M, envmib("TOSH_DMOE_MIN_ARENA_MIB", 0));
    if (arena < min_arena && tosh_hostcache_on()) {
        throw std::runtime_error("the bounded host cache needs mixed expert execution, and the VRAM arena is under its minimum");
    }
    if (arena < min_arena) {
        LLAMA_LOG_WARN("%s: under %.0f MiB left for experts, mixed execution stays off\n", __func__, min_arena/M);
        dmoe_status(false, "DMOE_ARENA_BELOW_MINIMUM", format("%.0f MiB left, %.0f needed", std::max(0.0, arena)/M, min_arena/M), 0.0);
        return;
    }
    dmoe_init_or_off(layers, hparams, nullptr, arena/M);
}

static void dmoe_init(std::vector<llama_layer> & layers, const llama_hparams & hparams, const char * path, double cache_mib) {
    // a residency file pins the set; a cache size lets the policy fill it online
    const bool dynamic = path == nullptr;

    const int n_layer  = (int) layers.size();
    const int n_expert = (int) hparams.n_expert;
    std::vector<std::vector<int>> resident(n_layer);
    std::vector<std::vector<int>> slot_of(n_layer, std::vector<int>(n_expert, -1));

    int quota = 0;
    if (dynamic) {
        size_t per_expert_all_layers = 0;
        for (const llama_layer & L : layers) {
            if (dmoe_has_bank(L)) {
                per_expert_all_layers += dmoe_gate(L)->nb[2] + (L.ffn_up_exps ? L.ffn_up_exps->nb[2] : 0) + L.ffn_down_exps->nb[2];
            }
        }
        if (per_expert_all_layers == 0) {
            throw std::runtime_error("TOSH_DMOE_CACHE_MIB: the model has no layer with separate gate, up and down expert banks");
        }
        quota = (int) std::min<size_t>((size_t) n_expert, (size_t) (cache_mib*1048576.0)/per_expert_all_layers);
        if (quota < 1) {
            throw std::runtime_error("TOSH_DMOE_CACHE_MIB: the cache is smaller than one expert per layer");
        }
    }

    std::ifstream f(dynamic ? "" : path);
    if (!dynamic && !f) {
        throw std::runtime_error(format("TOSH_DMOE_RESIDENCY: cannot open %s", path));
    }
    for (std::string line; !dynamic && std::getline(f, line);) {
        std::istringstream ls(line);
        int il, e;
        if (!(ls >> il >> e)) {
            continue;
        }
        if (il < 0 || il >= n_layer || e < 0 || e >= n_expert) {
            throw std::runtime_error(format("TOSH_DMOE_RESIDENCY: (%d, %d) is out of range", il, e));
        }
        if (slot_of[il][e] < 0) {
            slot_of[il][e] = (int) resident[il].size();
            resident[il].push_back(e);
        }
    }

    ggml_backend_dev_t gpu = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
    if (gpu == nullptr) {
        throw std::runtime_error("TOSH_DMOE_RESIDENCY: no GPU device");
    }
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(gpu);

    std::unique_ptr<executor> ex_owner(new executor());
    auto * ex = ex_owner.get();
    if (const char * v = getenv("TOSH_DMOE_SPIN_US")) {
        ex->spin_us = atoi(v);
    }
    if (const char * v = getenv("TOSH_DMOE_WATCHDOG_MS")) {
        ex->watchdog_us = atof(v)*1e3;
    }
    ex->banks.resize(n_layer);

    ggml_init_params ip = { ggml_tensor_overhead()*(size_t) n_layer*4 + 1024, nullptr, true };
    ggml_context * ctx = ggml_init(ip);

    std::vector<std::pair<int, llama_tosh_dmoe_layer>> made;
    for (int il = 0; il < n_layer; ++il) {
        llama_layer & L = layers[il];
        if (!dmoe_has_bank(L)) {
            continue;
        }
        for (const ggml_tensor * t : { (const ggml_tensor *) dmoe_gate(L), (const ggml_tensor *) L.ffn_up_exps, (const ggml_tensor *) L.ffn_down_exps }) {
            if (t == nullptr) continue;
            if (t->buffer == nullptr || !ggml_backend_buffer_is_host(t->buffer)) {
                throw std::runtime_error("TOSH_DMOE_RESIDENCY needs every expert bank in host memory (-ncmoe covering all layers)");
            }
        }
        host_bank & hb = ex->banks[il];
        hb.gate  = dmoe_gate(L);
        hb.up    = L.ffn_up_exps;
        hb.fused = L.ffn_gate_exps == nullptr;
        hb.down = L.ffn_down_exps;
        ex->n_embd = (int) hb.gate->ne[0];
        ex->n_ff   = (int) (hb.fused ? hb.gate->ne[1]/2 : hb.gate->ne[1]);
        if (hb.fused && L.ffn_gate_up_exps_b != nullptr) {
            throw std::runtime_error("TOSH_DMOE_CACHE_MIB: biases on a fused gate/up bank are not supported");
        }
        // with experts read from disk (TOSH_AUTO_DISK_EXPERTS) decode reads the next layer ahead; TOSH_HC_DECODE_AHEAD=0 turns it off
        static const bool decode_ahead = getenv("TOSH_AUTO_DISK_EXPERTS") != nullptr && atoi(getenv("TOSH_AUTO_DISK_EXPERTS")) != 0 &&
                                         (getenv("TOSH_HC_DECODE_AHEAD") == nullptr || atoi(getenv("TOSH_HC_DECODE_AHEAD")) != 0);
        const ggml_tensor * rt = L.ffn_gate_inp;
        if (decode_ahead && tosh_hostcache_on() && rt != nullptr && rt->ne[1] == n_expert && ggml_get_type_traits(rt->type)->to_float) {
            std::vector<uint8_t> raw(ggml_nbytes(rt));
            ggml_backend_tensor_get(rt, raw.data(), 0, raw.size());
            hb.router.resize((size_t) rt->ne[0]*rt->ne[1]);
            for (int64_t e = 0; e < rt->ne[1]; ++e) {
                ggml_get_type_traits(rt->type)->to_float(raw.data() + e*rt->nb[1], hb.router.data() + e*rt->ne[0], rt->ne[0]);
            }
            if (L.ffn_exp_probs_b != nullptr && L.ffn_exp_probs_b->type == GGML_TYPE_F32) {
                hb.router_b.resize(n_expert);
                ggml_backend_tensor_get(L.ffn_exp_probs_b, hb.router_b.data(), 0, sizeof(float)*n_expert);
            }
        }
        for (auto [b, dst] : { std::pair{ L.ffn_gate_exps_b, &hb.gate_b }, { L.ffn_up_exps_b, &hb.up_b } }) {
            if (b == nullptr) {
                continue;
            }
            if (b->type != GGML_TYPE_F32 || b->ne[0] != ex->n_ff || b->ne[1] != n_expert) {
                throw std::runtime_error("TOSH_DMOE_CACHE_MIB: expert biases of an unexpected type or shape");
            }
            dst->resize(ggml_nelements(b));
            ggml_backend_tensor_get(b, dst->data(), 0, ggml_nbytes(b));
        }

        llama_tosh_dmoe_layer dl;
        dl.il      = il;
        dl.n_slots = dynamic ? quota : (int) resident[il].size();
        const int64_t ns = std::max(1, dl.n_slots);
        dl.arena_gate = ggml_new_tensor_3d(ctx, hb.gate->type, hb.gate->ne[0], hb.gate->ne[1], ns);
        dl.arena_up   = hb.up ? ggml_new_tensor_3d(ctx, hb.up->type, hb.up->ne[0], hb.up->ne[1], ns) : nullptr;
        dl.arena_down = ggml_new_tensor_3d(ctx, L.ffn_down_exps->type, L.ffn_down_exps->ne[0], L.ffn_down_exps->ne[1], ns);
        dl.residency  = ggml_new_tensor_1d(ctx, GGML_TYPE_I32, n_expert);
        ggml_format_name(dl.arena_gate, "dmoe_arena_gate-%d", il);
        if (dl.arena_up) ggml_format_name(dl.arena_up, "dmoe_arena_up-%d", il);
        ggml_format_name(dl.arena_down, "dmoe_arena_down-%d", il);
        ggml_format_name(dl.residency,  "dmoe_residency-%d",  il);
        made.push_back({ il, dl });
    }

    if (made.empty()) {
        throw std::runtime_error("TOSH_DMOE_RESIDENCY: the model has no layer with separate gate, up and down expert banks");
    }

    ggml_backend_buffer_t buf = ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft);
    if (buf == nullptr) {
        throw std::runtime_error("TOSH_DMOE_RESIDENCY: the VRAM arena does not fit");
    }
    ggml_backend_buffer_set_usage(buf, GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    g_arena_buf = buf;

    size_t n_resident = 0;
    for (auto & [il, dl] : made) {
        const llama_layer & L = layers[il];
        for (int s = 0; s < (int) resident[il].size(); ++s) {
            const int e = resident[il][s];
            for (auto [bank, arena] : { std::pair{ dmoe_gate(L), dl.arena_gate }, { L.ffn_up_exps, dl.arena_up }, { L.ffn_down_exps, dl.arena_down } }) {
                if (bank == nullptr) continue;
                ggml_backend_tensor_set(arena, (const char *) bank->data + (size_t) e*bank->nb[2], (size_t) s*arena->nb[2], arena->nb[2]);
            }
        }
        ggml_backend_tensor_set(dl.residency, slot_of[il].data(), 0, sizeof(int32_t)*n_expert);
        n_resident += resident[il].size();
        g_layers[dmoe_gate(L)] = dl;
    }

    const char * nt = getenv("TOSH_DMOE_THREADS");
    ex->n_threads = std::max(1, nt ? atoi(nt) : (int) std::thread::hardware_concurrency()/2);

    const ggml_type at = ggml_get_type_traits_cpu(ex->banks[made.front().first].gate->type)->vec_dot_type;
    ex->act_q_row = ggml_row_size(at, ex->n_embd);
    ex->h_q_row   = ggml_row_size(GGML_TYPE_F32, ex->n_ff);   // wide enough for any down vec_dot_type
    const size_t max_miss = (size_t) TOSH_DMOE_MAX_TOKENS*hparams.n_expert;
    ex->act_q.resize(ex->act_q_row*TOSH_DMOE_MAX_TOKENS);
    ex->h.resize(max_miss*ex->n_ff);
    ex->h_q.resize(max_miss*ex->h_q_row);
    ex->misses.reserve(max_miss);

    if (ex->n_embd % executor::rows2 != 0 || ex->n_ff % executor::rows1 != 0) {
        throw std::runtime_error("TOSH_DMOE_RESIDENCY: expert shape not divisible by the executor's row chunks");
    }

    if (dynamic) {
        const size_t page = 16384;
        const size_t bytes = ((size_t) TOSH_DMOE_VERSIONS*TOSH_DMOE_MAX_LAYERS*n_expert*sizeof(int32_t) + page - 1)/page*page;
        // kept across arena rebuilds: the GPU backend wraps these pages once
        static void * tb_kept = nullptr;
        static size_t tb_bytes = 0;
        void * tb = bytes <= tb_bytes ? tb_kept : nullptr;
        if (tb == nullptr && posix_memalign(&tb, page, bytes) != 0) {
            throw std::runtime_error("TOSH_DMOE_CACHE_MIB: cannot allocate the residency tables");
        }
        if (tb != tb_kept) {
            tb_kept = tb;
            tb_bytes = bytes;
        }
        memset(tb, 0xff, bytes);
        tosh_dmoe_set_tables((int32_t *) tb, bytes, TOSH_DMOE_MAX_LAYERS, n_expert);
        tosh_dmoe_publish_version(0);

        auto * pol = new policy();
        pol->n_layer  = n_layer;
        pol->n_expert = n_expert;
        pol->Q        = quota;
        pol->tables   = (int32_t *) tb;
        pol->banks    = &ex->banks;
        pol->ex       = ex;
        pol->up       = tosh_dmoe_get_uploader();
        if (pol->up == nullptr) {
            throw std::runtime_error("TOSH_DMOE_CACHE_MIB: the GPU backend offers no uploader");
        }
        auto envf = [](const char * k, double d) { const char * v = getenv(k); return v ? atof(v) : d; };
        pol->decay        = (float) envf("TOSH_DMOE_DECAY", 0.98);
        // a promotion costs about one reuse of token time, so 1.8 keeps a margin against churn
        pol->admit_fixed  = envf("TOSH_DMOE_ADMIT", 1.8);
        pol->margin       = (float) envf("TOSH_DMOE_MARGIN", 0.0);
        // 0: no per-graph cap; the stage ring still bounds the uploads in flight
        pol->budget       = (int) envf("TOSH_DMOE_BUDGET", 0);
        if (pol->budget <= 0) pol->budget = n_layer*n_expert;
        pol->stop_after   = (long) envf("TOSH_DMOE_PROMO_STOP", -1);
        pol->warm_prefill = envf("TOSH_DMOE_WARM_PREFILL", 1) != 0;
        pol->tok_decay    = tosh_dmoe_token_decay();
        // a layer with few slots cannot empty eight of them every chunk and still keep a working set
        pol->reserve      = (int) envf("TOSH_DMOE_LEND_RESERVE", std::min(8, quota/8));
        pol->reserve_tail = (int) envf("TOSH_DMOE_LEND_TAIL", 8);
        pol->verify_every = (long) envf("TOSH_DMOE_VERIFY", 0);
        pol->defer_wide   = envf("TOSH_DMOE_DEFER_WIDE", 0) != 0;
        pol->vram_promote = envf("TOSH_DMOE_PREFILL_PROMOTE", 1) != 0;
        pol->vplan.resize(n_layer);
        pol->vmask.assign(n_layer, 0);
        pol->vmask_sync.assign(n_layer, 0);
        pol->full.assign(n_layer, 7);
        for (int il = 0; il < n_layer; ++il) {
            if (layers[il].ffn_gate_exps == nullptr) pol->full[il] = 5;
        }
        pol->layers.assign(n_layer, nullptr);
        pol->free_list.resize(n_layer);
        for (auto & [il, dl] : made) {
            pol->layers[il] = &g_layers[dmoe_gate(layers[il])];
            for (int s = quota - 1; s >= 0; --s) pol->free_list[il].push_back(s);
        }
        const size_t nk = (size_t) n_layer*n_expert;
        pol->score.assign(nk, 0.0f);
        pol->last_tick.assign(nk, 0);
        pol->resident_slot.assign(nk, -1);
        pol->pending.assign(nk, 0);
        pol->evicted_tick.assign(nk, 0);
        pol->slots.resize((size_t) n_layer*quota);
        pol->pow_decay.resize(1 << 16);
        for (size_t i = 0; i < pol->pow_decay.size(); ++i) pol->pow_decay[i] = powf(pol->decay, (float) i);
        pol->ring.assign(n_layer, std::vector<policy::obs>(policy::ring_size));
        pol->head.reset(new std::atomic<uint32_t>[n_layer]());
        pol->tail.reset(new std::atomic<uint32_t>[n_layer]());
        ex->observe     = policy::observe_cb;
        ex->observe_ctx = pol;
        g_policy = pol;
        pol->th = std::thread(&policy::main, pol);
        LLAMA_LOG_INFO("%s: dynamic residency, %d slots per layer, decay %.3f, budget %d promotions per graph\n",
                __func__, quota, pol->decay, pol->budget);
    }

    g_n_expert = n_expert;
    g_layer_by_il.assign(n_layer, nullptr);
    g_fixed_slots = slot_of;
    for (auto & [il, dl] : made) {
        const llama_layer & L = layers[il];
        g_layer_by_il[il] = &g_layers[dmoe_gate(L)];
        g_bank_refs[dmoe_gate(L)->data] = { il, 0, dmoe_gate(L) };
        if (L.ffn_up_exps) g_bank_refs[L.ffn_up_exps->data] = { il, 1, L.ffn_up_exps };
        g_bank_refs[L.ffn_down_exps->data] = { il, 2, L.ffn_down_exps };
    }
    if (getenv("TOSH_DMOE_PREFILL_ARENA") == nullptr || atoi(getenv("TOSH_DMOE_PREFILL_ARENA")) != 0) {
        tosh_dmoe_set_bank_planner(dmoe_bank_plan);
    }

    g_exec = ex_owner.release();
    ex->start();
    static bool reported = false;
    if (!reported) {
        reported = true;
        atexit(dmoe_report_at_exit);
    }

    LLAMA_LOG_INFO("%s: mixed expert execution on %zu layers, %zu resident experts, VRAM arena %.2f MiB, %d host threads\n",
            __func__, made.size(), n_resident, ggml_backend_buffer_get_size(buf)/1048576.0, ex->n_threads);
}

#endif
