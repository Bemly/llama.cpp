#include "tosh-hostcache.h"
#include "tosh-dmoe.h"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <mutex>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#include <fcntl.h>
#include <pthread.h>
#include <sys/mman.h>
#include <sys/uio.h>
#include <sys/param.h>
#ifdef __APPLE__
#include <sys/sysctl.h>
#endif
#include <unistd.h>

namespace {

enum slot_state : uint8_t { EMPTY, LOADING, RESIDENT };

// TOSH_HC_LOOKAHEAD=1 / TOSH_HC_PROMPT_LAST=1: prompt lookahead and prompt-first eviction, off by default
static bool env_on(const char * name) {
    const char * v = getenv(name);
    return v != nullptr && strcmp(v, "0") != 0;
}
static constexpr int LOOKAHEAD_LAYERS = 2;

struct file_tensor {
    size_t   nbytes;
    uint64_t file_off;
    int      file;
    uint64_t file_off_up = UINT64_MAX;   // up half of a gate/up pair the loader fused
};

struct part_ref {
    uint64_t file_off  = 0;     // of expert 0
    int      file      = 0;     // shard of a split model
    size_t   bytes     = 0;     // per expert
    size_t   slot_off  = 0;     // inside a cache slot
};

struct cache {
    std::mutex              m;
    std::condition_variable loaded;
    std::condition_variable work;
    std::deque<int>         queue;      // demand reads in front, read-backs of evicted experts behind
    bool                    stop = false;

    std::unordered_map<const void *, file_tensor> noted;
    std::unordered_map<const void *, std::pair<int, int>> bank_of;   // tensor data -> (il, part)
    std::vector<std::string> paths;     // model files; a split model has one per shard
    std::vector<int> fds;

    int n_layer = 0, n_expert = 0;
    std::vector<part_ref> parts;        // [il*3 + part]
    // one pool per expert size (mixed quantizations), each with its share of the capacity
    struct pool { size_t slot_bytes = 0; int first = 0, n = 0; char * base = nullptr; size_t bytes = 0;
                  std::vector<int> free_slots; int lru_head = -1, lru_tail = -1; };
    std::vector<pool> pools;
    std::vector<int> pool_of_layer, pool_of_slot;
    char * base = nullptr;              // first pool, non-null once started
    size_t cap_bytes = 0, locked_bytes = 0, resident_bytes = 0, peak_bytes = 0;
    int n_slots = 0;

    std::vector<int32_t>    slot_of;    // key -> slot
    std::vector<uint8_t>    ever;       // key was read before
    std::vector<double>     enq_t;      // when a slot's read was queued
    std::vector<uint8_t>    bg;         // queued by a read-back, not by a demand
    std::vector<uint8_t>    bg_unused;  // read back and not used since
    std::vector<uint8_t>    origin;     // 0 read after the server was ready, 1 during warm-up, 2 warm-up fill
    std::vector<uint8_t>    reused;     // a warm-up or fill entry hit after ready
    std::atomic<bool>       ready{false};
    std::thread             locker;
    double   t_start = 0, t_ready = 0, fill_stop_t = 0;
    uint64_t fill_n = 0, fill_bytes = 0, fill_dup = 0, fill_skipped = 0;
    uint64_t reuse_n[3] = {}, reuse_bytes[3] = {}, loaded_n[3] = {}, loaded_bytes[3] = {};
    std::vector<int32_t>    key_of, ref, prev, next;
    std::vector<uint8_t>    state;
    // a slot only a prompt has used goes first on eviction, ahead of the experts decode keeps using
    std::vector<uint8_t>    prompt_only;
    // prompt lookahead: the experts each layer pinned in its latest chunk, read ahead for the next one
    std::vector<std::vector<int32_t>> pf_set;
    std::vector<uint32_t>   pf_chunk_of;
    uint32_t pf_chunk = 0;
    int      pf_il = -1;
    uint64_t ahead_queued = 0;

    std::string error;

    // stats
    uint64_t req[TOSH_HC_KINDS] = {}, hit[TOSH_HC_KINDS] = {}, cold[TOSH_HC_KINDS] = {}, coalesced = 0;
    uint64_t evictions = 0, reloads = 0, reads = 0, read_bytes = 0, prefetches = 0, stored = 0;
    double wait_us[TOSH_HC_KINDS] = {};
    // queue and read time per class (0 demand, 1 read-back), where acquire waits, read-back reuse
    std::vector<float> qwait_us[2], rd_us[2];
    uint64_t rd_n[2] = {}, rd_bytes[2] = {};
    double wait_own_us[TOSH_HC_KINDS] = {}, wait_co_dem_us[TOSH_HC_KINDS] = {}, wait_co_bg_us[TOSH_HC_KINDS] = {};
    uint64_t bg_used = 0, bg_used_loading = 0, bg_wasted = 0, bg_dropped_unused = 0;
    size_t prewarm_bytes = 0; double prewarm_us = 0;
    uint64_t boosted = 0, kept_on_promotion = 0, refreshed = 0;

    // a free slot of layer il's pool, evicting the least recent unpinned one; -1 if all are pinned
    int take_slot(int il) {
        pool & pl = pools[pool_of_layer[il]];
        int s;
        if (!pl.free_slots.empty()) {
            s = pl.free_slots.back();
            pl.free_slots.pop_back();
        } else {
            s = pl.lru_tail;
            if (s < 0) return -1;
            lru_unlink(s);
            slot_of[key_of[s]] = -1;
            evictions++;
            if (bg_unused[s]) { bg_wasted++; bg_unused[s] = 0; }
            resident_bytes -= pl.slot_bytes;
        }
        resident_bytes += pl.slot_bytes;
        peak_bytes = std::max(peak_bytes, resident_bytes);
        return s;
    }
    std::vector<float> read_us;
    std::atomic<uint64_t> cold_total{0};

    char * slot_ptr(int s) { pool & p = pools[pool_of_slot[s]]; return p.base + (size_t) (s - p.first)*p.slot_bytes; }
    void lru_unlink(int s) {
        pool & p = pools[pool_of_slot[s]];
        if (prev[s] >= 0) next[prev[s]] = next[s]; else if (p.lru_head == s) p.lru_head = next[s];
        if (next[s] >= 0) prev[next[s]] = prev[s]; else if (p.lru_tail == s) p.lru_tail = prev[s];
        prev[s] = next[s] = -1;
    }
    void lru_push_tail(int s) {
        pool & p = pools[pool_of_slot[s]];
        next[s] = -1;
        prev[s] = p.lru_tail;
        if (p.lru_tail >= 0) next[p.lru_tail] = s;
        p.lru_tail = s;
        if (p.lru_head < 0) p.lru_head = s;
    }
    void lru_push(int s) {
        pool & p = pools[pool_of_slot[s]];
        prev[s] = -1;
        next[s] = p.lru_head;
        if (p.lru_head >= 0) prev[p.lru_head] = s;
        p.lru_head = s;
        if (p.lru_tail < 0) p.lru_tail = s;
    }
    bool in_lru(int s) const { return prev[s] >= 0 || next[s] >= 0 || pools[pool_of_slot[s]].lru_head == s; }
};

cache g;

double now_us() {
    return std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now().time_since_epoch()).count();
}

// Wired pages cannot be compressed or paged out: locking faster than the system frees memory
// evicts the code pages of every other process, and a desktop stalls until the watchdog reboots it.
bool mlock_paced(void * addr, size_t bytes, char * err, size_t err_len) {
    static const size_t step    = (size_t) 256 << 20;
    static const double wait_us = 30e6;
    char * base = (char *) addr;
    for (size_t o = 0; o < bytes; o += step) {
#ifdef __APPLE__
        const double t0 = now_us();
        for (;;) {
            int lvl = 1; size_t len = sizeof(lvl);
            if (sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &len, nullptr, 0) != 0 || lvl <= 1) {
                break;
            }
            if (lvl >= 4 || now_us() - t0 > wait_us) {
                snprintf(err, err_len, "DMOE_HOST_MEMORY_PRESSURE: the system reported %s memory pressure after %.0f of %.0f MiB were locked",
                         lvl >= 4 ? "critical" : "sustained", o/1048576.0, bytes/1048576.0);
                munlock(base, o);
                return false;
            }
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }
#endif
        if (getenv("TOSH_DMOE_TEST_MLOCK_FAIL") != nullptr || mlock(base + o, std::min(step, bytes - o)) != 0) {
            const int e = getenv("TOSH_DMOE_TEST_MLOCK_FAIL") != nullptr ? EAGAIN : errno;
            snprintf(err, err_len, "DMOE_HOST_BANK_NOT_LOCKABLE: mlock failed after %.0f of %.0f MiB (%s)",
                     o/1048576.0, bytes/1048576.0, strerror(e));
            munlock(base, o);
            return false;
        }
    }
    return true;
}

void reader_main() {
#ifdef __APPLE__
    // the executor's threads run at this class and would otherwise starve the reads they wait on
    pthread_set_qos_class_self_np(QOS_CLASS_USER_INTERACTIVE, 0);
#endif
    for (;;) {
        int s;
        {
            std::unique_lock<std::mutex> lock(g.m);
            g.work.wait(lock, [] { return g.stop || !g.queue.empty(); });
            if (g.stop) return;
            s = g.queue.front();
            g.queue.pop_front();
        }
        const int key = g.key_of[s];
        const int il = key / g.n_expert, e = key % g.n_expert;
        char * dst = g.slot_ptr(s);
        const double t0 = now_us();
        const int cls = g.bg[s];
        const double tq = g.enq_t[s];
        std::string err;
        size_t total = 0;
        for (int p = 0; p < 3 && err.empty(); ++p) {
            const part_ref & pr = g.parts[il*3 + p];
            size_t done = 0;
            while (done < pr.bytes) {
                const ssize_t r = pread(g.fds[pr.file], dst + pr.slot_off + done, pr.bytes - done, (off_t) (pr.file_off + (uint64_t) e*pr.bytes + done));
                if (r <= 0) {
                    err = r == 0 ? "short read of an expert from the model file" : std::string("pread: ") + strerror(errno);
                    break;
                }
                done += (size_t) r;
            }
            total += done;
        }
        static const long fail_at = getenv("TOSH_DMOE_TEST_READ_FAIL") ? atol(getenv("TOSH_DMOE_TEST_READ_FAIL")) : -1;
        if (fail_at >= 0 && (long) g.reads + 1 == fail_at && err.empty()) err = "injected read failure";
        const double us = now_us() - t0;
        {
            std::lock_guard<std::mutex> lock(g.m);
            if (!err.empty() && g.error.empty()) g.error = err;
            if (g.qwait_us[cls].size() < (1u << 22)) { g.qwait_us[cls].push_back((float) (t0 - tq)); g.rd_us[cls].push_back((float) us); }
            g.rd_n[cls]++; g.rd_bytes[cls] += total;
            tosh_dmoe_timeline(8 + 2*cls, il, tq*1e-6, t0*1e-6);
            tosh_dmoe_timeline(9 + 2*cls, il, t0*1e-6, (t0 + us)*1e-6);
            g.state[s] = RESIDENT;
            if (g.ref[s] == 0 && !g.in_lru(s)) g.lru_push(s);
            g.reads++;
            g.read_bytes += total;
            if (g.read_us.size() < (1u << 22)) g.read_us.push_back((float) us);
        }
        g.loaded.notify_all();
    }
}

}  // namespace

extern "C" {

bool tosh_hostcache_on(void) {
    // only a yes is kept: the Auto plan sets the variable after its probes have asked
    static std::atomic<bool> on{false};
    if (on.load(std::memory_order_relaxed)) return true;
    if (getenv("TOSH_DMOE_HOST_CACHE_MIB") == nullptr) return false;
    on.store(true);
    return true;
}

bool tosh_hostcache_is_bank_name(const char * name) {
    const char * p = strstr(name, ".ffn_");
    if (p == nullptr) return false;
    return strcmp(p, ".ffn_gate_exps.weight") == 0 || strcmp(p, ".ffn_up_exps.weight") == 0 || strcmp(p, ".ffn_down_exps.weight") == 0 ||
           strcmp(p, ".ffn_gate_up_exps.weight") == 0;
}

static int file_index(int fd) {
    char buf[MAXPATHLEN];
    if (fcntl(fd, F_GETPATH, buf) != 0) return -1;
    int file = 0;
    while (file < (int) g.paths.size() && g.paths[file] != buf) file++;
    if (file == (int) g.paths.size()) g.paths.push_back(buf);
    return file;
}

void tosh_hostcache_note_file_tensor(const void * data, size_t nbytes, int fd, uint64_t file_off) {
    std::lock_guard<std::mutex> lock(g.m);
    const int file = file_index(fd);
    if (file >= 0) g.noted[data] = { nbytes, file_off, file };
}

void tosh_hostcache_note_fused_tensor(const void * data, size_t nbytes, int fd, uint64_t off_gate, uint64_t off_up) {
    std::lock_guard<std::mutex> lock(g.m);
    const int file = file_index(fd);
    if (file >= 0) g.noted[data] = { nbytes, off_gate, file, off_up };
}

bool tosh_hostcache_bind(int il, int part, const void * data, size_t part_bytes, int n_expert) {
    std::lock_guard<std::mutex> lock(g.m);
    if (g.base != nullptr) return true;   // bound and started by the loader
    auto it = g.noted.find(data);
    if (it == g.noted.end() || it->second.nbytes != part_bytes*(size_t) n_expert || part < 0 || part > 2) {
        g.error = "an expert bank was loaded, or has an unexpected size";
        return false;
    }
    if ((int) g.parts.size() < (il + 1)*3) g.parts.resize((il + 1)*3);
    const file_tensor & ft = it->second;
    if (ft.file_off_up != UINT64_MAX) {
        // fused gate/up: its gate and up halves are the slot's parts 0 and 1, adjacent in the slot
        if (part != 0 || part_bytes % 2 != 0) { g.error = "a fused gate/up bank in an unexpected place"; return false; }
        g.parts[il*3 + 0] = { ft.file_off,    ft.file, part_bytes/2, 0 };
        g.parts[il*3 + 1] = { ft.file_off_up, ft.file, part_bytes/2, 0 };
    } else {
        g.parts[il*3 + part].file_off = ft.file_off;
        g.parts[il*3 + part].file     = ft.file;
        g.parts[il*3 + part].bytes    = part_bytes;
    }
    g.bank_of[data] = { il, part };
    return true;
}

static void prewarm(int n_io);
static void start_warmup_fill();

bool tosh_hostcache_start(int n_layer, int n_expert) {
    std::unique_lock<std::mutex> lock(g.m);
    if (!g.error.empty()) return false;
    if (g.base != nullptr) return true;   // started early by the loader
    g.n_layer = n_layer;
    g.n_expert = n_expert;
    g.parts.resize(n_layer*3);
    // a slot holds gate, up and down of one expert; layers with another expert size get their own pool
    std::vector<size_t> layer_bytes(n_layer, 0);
    for (int il = 0; il < n_layer; ++il) {
        size_t off = 0;
        for (int p = 0; p < 3; ++p) {
            g.parts[il*3 + p].slot_off = off;
            off += g.parts[il*3 + p].bytes;
        }
        layer_bytes[il] = off;
    }
    if (g.paths.empty()) { g.error = "no expert bank to cache"; return false; }
    for (const auto & path : g.paths) {
        const int fd = open(path.c_str(), O_RDONLY);
        if (fd < 0) { g.error = "cannot open the model file for expert reads: " + path; return false; }
        if (getenv("TOSH_DMOE_HOST_NOCACHE") != nullptr) fcntl(fd, F_NOCACHE, 1);
        g.fds.push_back(fd);
    }

    g.cap_bytes = (size_t) (atof(getenv("TOSH_DMOE_HOST_CACHE_MIB"))*1048576.0);
    size_t bank_bytes = 0;
    g.pool_of_layer.assign(n_layer, -1);
    for (int il = 0; il < n_layer; ++il) {
        if (layer_bytes[il] == 0) continue;
        bank_bytes += layer_bytes[il]*(size_t) n_expert;
        int pi = -1;
        for (int k = 0; k < (int) g.pools.size(); ++k) if (g.pools[k].slot_bytes == layer_bytes[il]) pi = k;
        if (pi < 0) { pi = (int) g.pools.size(); g.pools.emplace_back(); g.pools.back().slot_bytes = layer_bytes[il]; }
        g.pool_of_layer[il] = pi;
    }
    if (g.pools.empty()) { g.error = "no expert bank to cache"; return false; }
    // every pool holds at least one layer's full expert set, since a prefill chunk can pin them all;
    // the rest of the budget is shared in proportion to each size class
    size_t floor_bytes = 0;
    for (auto & pl : g.pools) floor_bytes += pl.slot_bytes*(size_t) n_expert;
    if (floor_bytes > g.cap_bytes) { g.error = "the expert cache is smaller than one layer's experts per size class"; return false; }
    for (auto & pl : g.pools) {
        size_t class_bytes = 0;
        for (int il = 0; il < n_layer; ++il) if (g.pool_of_layer[il] >= 0 && g.pools[g.pool_of_layer[il]].slot_bytes == pl.slot_bytes) class_bytes += pl.slot_bytes*(size_t) n_expert;
        const size_t share = (size_t) ((double) (g.cap_bytes - floor_bytes)*class_bytes/bank_bytes) + pl.slot_bytes*(size_t) n_expert;
        pl.n = (int) std::min(share/pl.slot_bytes, class_bytes/pl.slot_bytes);
        pl.first = g.n_slots;
        g.n_slots += pl.n;
        pl.bytes = (size_t) pl.n*pl.slot_bytes;
        void * p = mmap(nullptr, std::max<size_t>(pl.bytes, 1), PROT_READ | PROT_WRITE, MAP_PRIVATE | MAP_ANON, -1, 0);
        if (p == MAP_FAILED) { g.error = "cannot allocate the expert cache"; return false; }
        pl.base = (char *) p;
        g.locked_bytes += pl.bytes;
    }
    g.base = g.pools[0].base;
    const int n_keys = n_layer*n_expert;
    g.slot_of.assign(n_keys, -1);
    g.ever.assign(n_keys, 0);
    g.key_of.assign(g.n_slots, -1);
    g.ref.assign(g.n_slots, 0);
    g.prev.assign(g.n_slots, -1);
    g.next.assign(g.n_slots, -1);
    g.state.assign(g.n_slots, EMPTY);
    g.enq_t.assign(g.n_slots, 0.0);
    g.bg.assign(g.n_slots, 0);
    g.bg_unused.assign(g.n_slots, 0);
    g.origin.assign(g.n_slots, 0);
    g.reused.assign(g.n_slots, 0);
    g.prompt_only.assign(g.n_slots, 0);
    g.pf_set.assign(n_layer, {});
    g.pf_chunk_of.assign(n_layer, UINT32_MAX);
    g.pool_of_slot.assign(g.n_slots, 0);
    for (int k = 0; k < (int) g.pools.size(); ++k) {
        auto & pl = g.pools[k];
        for (int s = pl.first + pl.n - 1; s >= pl.first; --s) { pl.free_slots.push_back(s); g.pool_of_slot[s] = k; }
    }

    const int n_io = std::max(1, getenv("TOSH_DMOE_HOST_IO") ? atoi(getenv("TOSH_DMOE_HOST_IO")) : 4);
    for (int i = 0; i < n_io; ++i) std::thread(reader_main).detach();
    for (auto & pl : g.pools) {
        fprintf(stderr, "tosh_hostcache: pool of %d slots of %.3f MiB (%.0f MiB)\n", pl.n, pl.slot_bytes/1048576.0, pl.bytes/1048576.0);
    }
    fprintf(stderr, "tosh_hostcache: %.0f MiB locked for experts, %d readers, %zu file%s from %s\n", g.locked_bytes/1048576.0, n_io,
            g.paths.size(), g.paths.size() > 1 ? "s" : "", g.paths[0].c_str());
    g.t_start = now_us();
    // locking zero-fills every page; with the warm-up fill on, the pages are filled with experts from
    // the file while the lock proceeds, and the server waits for the lock before it is ready
    const bool overlap = getenv("TOSH_DMOE_HOST_WARMUP_FILL_MIB") != nullptr;
    auto lock_pools = [] {
        for (auto & pl : g.pools) {
            char err[256];
            if (!mlock_paced(pl.base, pl.bytes, err, sizeof(err))) {
                std::lock_guard<std::mutex> lk(g.m);
                if (g.error.empty()) g.error = err;
                return;
            }
        }
    };
    lock.unlock();
    if (overlap) {
        g.locker = std::thread(lock_pools);
        // a load that fails before ready exits with the lock thread still attached
        static const bool joined_at_exit = atexit([] { if (g.locker.joinable()) g.locker.join(); }) == 0;
        (void) joined_at_exit;
        start_warmup_fill();
    } else {
        lock_pools();
    }
    prewarm(n_io);
    return g.error.empty();
}

// EXPERIMENT: TOSH_DMOE_HOST_WARMUP_FILL_MIB (a number, or max) reads experts in file order into free
// slots from the moment the model finishes loading, past ready, until the budget or the slots run
// out. Demand never waits on it: an entry is published only once read, and last in line for eviction.
static void start_warmup_fill() {
    const char * v = getenv("TOSH_DMOE_HOST_WARMUP_FILL_MIB");
    if (v == nullptr) return;
    const size_t budget = strcmp(v, "max") == 0 ? g.locked_bytes : (size_t) (atof(v)*1048576.0);
    if (budget == 0) return;
    const int n_th = std::max(1, getenv("TOSH_DMOE_HOST_FILL_IO") ? atoi(getenv("TOSH_DMOE_HOST_FILL_IO")) : 16);
    auto next = std::make_shared<std::atomic<int>>(0);
    auto left = std::make_shared<std::atomic<int>>(n_th);
    for (int t = 0; t < n_th; ++t) {
        std::thread([budget, next, left] {
            pthread_set_qos_class_self_np(QOS_CLASS_UTILITY, 0);
            const int n_keys = g.n_layer*g.n_expert;
            for (int key; (key = (*next)++) < n_keys;) {
                const int il = key / g.n_expert, e = key % g.n_expert;
                int s;
                {
                    std::lock_guard<std::mutex> lock(g.m);
                    if (g.pool_of_layer[il] < 0 || g.slot_of[key] >= 0) continue;
                    if (g.fill_bytes >= budget) break;
                    auto & pl = g.pools[g.pool_of_layer[il]];
                    if (pl.free_slots.empty()) { g.fill_skipped++; continue; }   // never evicts
                    s = pl.free_slots.back();
                    pl.free_slots.pop_back();
                    g.resident_bytes += pl.slot_bytes;
                    g.peak_bytes = std::max(g.peak_bytes, g.resident_bytes);
                    g.key_of[s] = -1; g.state[s] = LOADING; g.ref[s] = 1;
                    g.fill_bytes += pl.slot_bytes;
                }
                bool ok = true;
                for (int p = 0; p < 3 && ok; ++p) {
                    const part_ref & pr = g.parts[il*3 + p];
                    size_t done = 0;
                    while (ok && done < pr.bytes) {
                        const ssize_t r = pread(g.fds[pr.file], g.slot_ptr(s) + pr.slot_off + done, pr.bytes - done,
                                                (off_t) (pr.file_off + (uint64_t) e*pr.bytes + done));
                        if (r <= 0) ok = false; else done += (size_t) r;
                    }
                }
                std::lock_guard<std::mutex> lock(g.m);
                auto & pl = g.pools[g.pool_of_slot[s]];
                g.ref[s] = 0;
                if (!ok || g.slot_of[key] >= 0) {
                    // a demand read it meanwhile, or the read failed: the slot goes back
                    g.state[s] = EMPTY;
                    pl.free_slots.push_back(s);
                    g.resident_bytes -= pl.slot_bytes;
                    g.fill_bytes -= pl.slot_bytes;
                    g.fill_dup++;
                    continue;
                }
                g.key_of[s] = key; g.slot_of[key] = s; g.state[s] = RESIDENT;
                g.ever[key] = 1; g.origin[s] = 2; g.reused[s] = 0;
                g.lru_push_tail(s);
                g.fill_n++;
                g.loaded_n[2]++; g.loaded_bytes[2] += pl.slot_bytes;
            }
            if (--*left == 0) {
                std::lock_guard<std::mutex> lock(g.m);
                g.fill_stop_t = now_us();
            }
        }).detach();
    }
}

// EXPERIMENT: fill part of the cache before the first request. TOSH_DMOE_HOST_PREWARM_MIB bytes, taken
// in layer order (TOSH_DMOE_HOST_PREWARM=seq) or from a list of "layer expert" lines; read one
// expert at a time in list order, or (TOSH_DMOE_HOST_PREWARM_ORDER=offset) in file order with
// adjacent experts of a tensor merged into one read
static void prewarm(int n_io) {
    const char * mib = getenv("TOSH_DMOE_HOST_PREWARM_MIB");
    if (mib == nullptr || atof(mib) <= 0) return;
    const size_t budget = (size_t) (atof(mib)*1048576.0);
    const char * src = getenv("TOSH_DMOE_HOST_PREWARM") ? getenv("TOSH_DMOE_HOST_PREWARM") : "seq";
    const bool by_offset = getenv("TOSH_DMOE_HOST_PREWARM_ORDER") && strcmp(getenv("TOSH_DMOE_HOST_PREWARM_ORDER"), "offset") == 0;
    const double t0 = now_us();
    std::vector<int> keys;
    if (strcmp(src, "seq") == 0) {
        for (int il = 0; il < g.n_layer; ++il) {
            if (g.pool_of_layer[il] < 0) continue;
            for (int e = 0; e < g.n_expert; ++e) keys.push_back(il*g.n_expert + e);
        }
    } else if (FILE * f = fopen(src, "r")) {
        int il, e;
        while (fscanf(f, "%d %d", &il, &e) == 2) {
            if (il >= 0 && il < g.n_layer && e >= 0 && e < g.n_expert && g.pool_of_layer[il] >= 0) keys.push_back(il*g.n_expert + e);
        }
        fclose(f);
    }
    std::vector<int> slots;
    size_t bytes = 0;
    {
        std::lock_guard<std::mutex> lock(g.m);
        for (int key : keys) {
            const int il = key / g.n_expert;
            auto & pl = g.pools[g.pool_of_layer[il]];
            if (bytes + pl.slot_bytes > budget) break;
            if (g.slot_of[key] >= 0 || pl.free_slots.empty()) continue;
            const int s = g.take_slot(il);
            g.key_of[s] = key; g.slot_of[key] = s; g.state[s] = LOADING; g.ref[s] = 0; g.ever[key] = 1;
            slots.push_back(s);
            bytes += pl.slot_bytes;
        }
    }
    // one read job: a range of one part tensor landing in the parts of one or more slots
    struct job { int il, part, e0; std::vector<int> s; };
    std::vector<job> jobs;
    if (by_offset) {
        std::vector<int> order = slots;
        std::sort(order.begin(), order.end(), [](int a, int b) { return g.key_of[a] < g.key_of[b]; });
        for (int part = 0; part < 3; ++part) {
            for (size_t i = 0; i < order.size();) {
                const int k0 = g.key_of[order[i]];
                if (g.parts[k0 / g.n_expert*3 + part].bytes == 0) { i++; continue; }   // no such part (fused gate/up)
                job j{ k0 / g.n_expert, part, k0 % g.n_expert, { order[i] } };
                size_t k = i + 1;
                // a run stays inside one layer's tensor
                while (k < order.size() && g.key_of[order[k]] == k0 + (int) (k - i) && g.key_of[order[k]] / g.n_expert == j.il &&
                       j.s.size() < 64) j.s.push_back(order[k++]);
                jobs.push_back(std::move(j));
                i = k;
            }
        }
        std::sort(jobs.begin(), jobs.end(), [](const job & a, const job & b) {
            const uint64_t oa = g.parts[a.il*3 + a.part].file_off + (uint64_t) a.e0*g.parts[a.il*3 + a.part].bytes;
            const uint64_t ob = g.parts[b.il*3 + b.part].file_off + (uint64_t) b.e0*g.parts[b.il*3 + b.part].bytes;
            return oa < ob;
        });
    } else {
        for (int s : slots) {
            const int k = g.key_of[s];
            for (int part = 0; part < 3; ++part) {
                if (g.parts[k / g.n_expert*3 + part].bytes > 0) jobs.push_back({ k / g.n_expert, part, k % g.n_expert, { s } });
            }
        }
    }
    std::atomic<size_t> next{0};
    std::atomic<uint64_t> n_reads{0};
    std::vector<std::thread> th;
    for (int t = 0; t < n_io; ++t) {
        th.emplace_back([&] {
            for (size_t i; (i = next++) < jobs.size();) {
                const job & j = jobs[i];
                const part_ref & pr = g.parts[j.il*3 + j.part];
                std::vector<iovec> iov(j.s.size());
                for (size_t k = 0; k < j.s.size(); ++k) iov[k] = { g.slot_ptr(j.s[k]) + pr.slot_off, pr.bytes };
                const size_t want = pr.bytes*j.s.size();
                size_t got = 0;
                const off_t off = (off_t) (pr.file_off + (uint64_t) j.e0*pr.bytes);
                while (got < want) {
                    // resume a short read inside the vector
                    size_t skip = got, k = 0;
                    while (skip >= iov[k].iov_len) { skip -= iov[k].iov_len; k++; }
                    iovec first = { (char *) iov[k].iov_base + skip, iov[k].iov_len - skip };
                    std::vector<iovec> rest(iov.begin() + k, iov.end());
                    rest[0] = first;
                    const ssize_t r = preadv(g.fds[pr.file], rest.data(), (int) rest.size(), off + (off_t) got);
                    if (r <= 0) { std::lock_guard<std::mutex> lock(g.m); if (g.error.empty()) g.error = "prewarm read failed"; return; }
                    got += (size_t) r;
                }
                n_reads++;
            }
        });
    }
    for (auto & t : th) t.join();
    {
        std::lock_guard<std::mutex> lock(g.m);
        for (int s : slots) { g.state[s] = RESIDENT; g.lru_push(s); }
        g.prewarm_bytes = bytes;
        g.prewarm_us = now_us() - t0;
    }
    fprintf(stderr, "tosh_hostcache: prewarm %zu experts, %.0f MiB from %s in %s order, %llu reads, %.0f ms (%.2f GB/s)\n",
            slots.size(), bytes/1048576.0, strcmp(src, "seq") == 0 ? "layer order" : "a list", by_offset ? "file" : "list",
            (unsigned long long) n_reads.load(), g.prewarm_us/1000, bytes/(g.prewarm_us*1e3));
}

bool tosh_mlock_paced(void * addr, size_t bytes, char * err, size_t err_len) {
    return mlock_paced(addr, bytes, err, err_len);
}

static int readback_slot(int il, int e);

bool tosh_hostcache_acquire(int il, const int32_t * ids, int n, int kind) {
    const double t0 = now_us();
    std::unique_lock<std::mutex> lock(g.m);
    std::vector<int> wait;
    std::vector<uint8_t> wait_cls;     // 0 own read, 1 another demand, 2 a read-back
    bool queued = false;
    const bool prompt = kind == TOSH_HC_PREFILL || kind == TOSH_HC_WIDE;
    if (prompt && g.base != nullptr) {
        static const bool lookahead = env_on("TOSH_HC_LOOKAHEAD");
        // a chunk walks the layers in order; host rows may trail the uploads by a layer
        if (il + 2 < g.pf_il) { g.pf_chunk++; g.pf_il = il; } else g.pf_il = std::max(g.pf_il, il);
        if (g.pf_chunk_of[il] != g.pf_chunk) {
            g.pf_chunk_of[il] = g.pf_chunk;
            g.pf_set[il].clear();
            // the next layers still hold the last chunk's experts: their reads go behind the demands
            for (int d = 1; d <= LOOKAHEAD_LAYERS && lookahead && il + d < g.n_layer; ++d) {
                auto & set = g.pf_set[il + d];
                std::sort(set.begin(), set.end());
                set.erase(std::unique(set.begin(), set.end()), set.end());
                for (int32_t e : set) {
                    const int sl = readback_slot(il + d, e);
                    if (sl < 0) continue;
                    g.prompt_only[sl] = 1;
                    g.queue.push_back(sl);
                    g.ahead_queued++;
                    queued = true;
                }
            }
        }
        g.pf_set[il].insert(g.pf_set[il].end(), ids, ids + n);
    }
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= g.n_expert) continue;
        const int key = il*g.n_expert + e;
        g.req[kind]++;
        int s = g.slot_of[key];
        if (s >= 0) {
            if (!prompt) g.prompt_only[s] = 0;
            if (g.ref[s]++ == 0 && g.in_lru(s)) g.lru_unlink(s);
            if (g.bg_unused[s]) {
                g.bg_unused[s] = 0;
                g.bg_used++;
                if (g.state[s] != RESIDENT) g.bg_used_loading++;
            }
            if (g.ready && g.origin[s] != 0 && !g.reused[s]) {
                g.reused[s] = 1;
                g.reuse_n[g.origin[s]]++;
                g.reuse_bytes[g.origin[s]] += g.pools[g.pool_of_slot[s]].slot_bytes;
            }
            if (g.state[s] == RESIDENT) {
                g.hit[kind]++;
            } else {
                g.coalesced++;
                wait.push_back(s);
                wait_cls.push_back(g.bg[s] ? 2 : 1);
                if (g.bg[s]) {
                    // a demand waits on a queued read-back: it goes ahead of the backlog
                    auto it = std::find(g.queue.begin(), g.queue.end(), s);
                    if (it != g.queue.end()) {
                        g.queue.erase(it);
                        g.queue.push_front(s);
                        g.boosted++;
                        queued = true;
                    }
                }
            }
            continue;
        }
        s = g.take_slot(il);
        if (s < 0) {
            if (g.error.empty()) g.error = "the expert cache is smaller than the experts pinned at once";
            return false;
        }
        g.key_of[s] = key;
        g.slot_of[key] = s;
        g.state[s] = LOADING;
        g.ref[s] = 1;
        g.prompt_only[s] = prompt;
        g.cold[kind]++;
        g.cold_total++;
        g.reloads += g.ever[key];
        g.ever[key] = 1;
        g.enq_t[s] = now_us();
        g.bg[s] = 0;
        g.bg_unused[s] = 0;
        g.origin[s] = g.ready ? 0 : 1;
        g.reused[s] = 0;
        g.loaded_n[g.origin[s]]++;
        g.loaded_bytes[g.origin[s]] += g.pools[g.pool_of_slot[s]].slot_bytes;
        g.queue.push_front(s);    // ahead of the read-backs of evicted experts
        queued = true;
        wait.push_back(s);
        wait_cls.push_back(0);
    }
    if (queued) g.work.notify_all();
    for (size_t i = 0; i < wait.size(); ++i) {
        const int s = wait[i];
        const double tw = now_us();
        g.loaded.wait(lock, [s] { return g.state[s] == RESIDENT; });
        const double dt = now_us() - tw;
        if (wait_cls[i] == 0) g.wait_own_us[kind] += dt; else if (wait_cls[i] == 1) g.wait_co_dem_us[kind] += dt; else g.wait_co_bg_us[kind] += dt;
    }
    const double t1 = now_us();
    g.wait_us[kind] += t1 - t0;
    if (!wait.empty()) tosh_dmoe_timeline(30 + kind, il, t0*1e-6, t1*1e-6);
    return g.error.empty();
}

const char * tosh_hostcache_part(int il, int e, int part) {
    const int s = g.slot_of[il*g.n_expert + e];
    return s < 0 ? nullptr : g.slot_ptr(s) + g.parts[il*3 + part].slot_off;
}

void tosh_hostcache_release(int il, const int32_t * ids, int n) {
    std::lock_guard<std::mutex> lock(g.m);
    for (int i = 0; i < n; ++i) {
        const int e = ids[i];
        if (e < 0 || e >= g.n_expert) continue;
        const int s = g.slot_of[il*g.n_expert + e];
        if (s >= 0 && --g.ref[s] == 0) {
            static const bool prompt_last = env_on("TOSH_HC_PROMPT_LAST");
            if (prompt_last && g.prompt_only[s]) g.lru_push_tail(s); else g.lru_push(s);
        }
    }
}

void tosh_hostcache_drop(int il, int e) {
    std::lock_guard<std::mutex> lock(g.m);
    if (g.base == nullptr) return;
    const int key = il*g.n_expert + e;
    const int s = g.slot_of[key];
    if (s < 0 || g.ref[s] != 0 || g.state[s] != RESIDENT) return;
    if (g.bg_unused[s]) { g.bg_unused[s] = 0; g.bg_dropped_unused++; }
    // the copy stays, first in line for eviction: if the expert leaves VRAM before it is evicted
    // here, nothing has to be read back
    g.lru_unlink(s);
    g.lru_push_tail(s);
    g.kept_on_promotion++;
}

// a slot for an expert coming back from VRAM, or -1 when RAM still holds it or has no room; g.m held
static int readback_slot(int il, int e) {
    const int key = il*g.n_expert + e;
    if (g.base == nullptr) return -1;
    if (g.slot_of[key] >= 0) {
        // a copy kept on promotion: it is needed again
        const int s = g.slot_of[key];
        if (g.ref[s] == 0 && g.state[s] == RESIDENT && g.in_lru(s)) { g.lru_unlink(s); g.lru_push(s); g.refreshed++; }
        return -1;
    }
    const int s = g.take_slot(il);
    if (s < 0) return -1;
    g.key_of[s] = key;
    g.slot_of[key] = s;
    g.state[s] = LOADING;
    g.ref[s] = 0;
    g.ever[key] = 1;
    g.enq_t[s] = now_us();
    g.bg[s] = 1;
    g.bg_unused[s] = 1;
    g.origin[s] = g.ready ? 0 : 1;
    g.reused[s] = 0;
    g.prompt_only[s] = 0;
    return s;
}

void tosh_hostcache_prefetch(int il, int e) {
    {
        std::lock_guard<std::mutex> lock(g.m);
        const int s = readback_slot(il, e);
        if (s < 0) return;
        g.prefetches++;
        g.queue.push_back(s);
    }
    g.work.notify_one();
}

bool tosh_hostcache_store(int il, int e) {
    std::lock_guard<std::mutex> lock(g.m);
    return readback_slot(il, e) >= 0;
}

void tosh_hostcache_store_done(int il, int e, bool ok) {
    {
        std::lock_guard<std::mutex> lock(g.m);
        const int s = g.base == nullptr ? -1 : g.slot_of[il*g.n_expert + e];
        if (s < 0 || g.state[s] != LOADING) return;
        if (ok) {
            g.state[s] = RESIDENT;
            if (g.ref[s] == 0 && !g.in_lru(s)) g.lru_push(s);
            g.stored++;
        } else {
            g.prefetches++;
            g.queue.push_back(s);
        }
    }
    if (ok) g.loaded.notify_all(); else g.work.notify_one();
}

bool tosh_hostcache_lookup(const void * data, int * il, int * part) {
    std::lock_guard<std::mutex> lock(g.m);
    auto it = g.bank_of.find(data);
    if (it == g.bank_of.end()) return false;
    *il = it->second.first;
    *part = it->second.second;
    return true;
}

bool tosh_hostcache_bytecheck(const void * data, size_t part_bytes, int e) {
    file_tensor ft;
    std::string path;
    {
        std::lock_guard<std::mutex> lock(g.m);
        auto it = g.noted.find(data);
        if (it == g.noted.end()) return false;
        ft = it->second;
        path = g.paths[ft.file];
    }
    const int fd = open(path.c_str(), O_RDONLY);
    if (fd < 0) return false;
    fcntl(fd, F_NOCACHE, 1);
    const bool fused = ft.file_off_up != UINT64_MAX;
    const size_t n = fused ? part_bytes/2 : part_bytes;
    std::vector<char> buf(part_bytes);
    bool ok = pread(fd, buf.data(), n, (off_t) (ft.file_off + (uint64_t) e*n)) == (ssize_t) n;
    if (ok && fused) ok = pread(fd, buf.data() + n, n, (off_t) (ft.file_off_up + (uint64_t) e*n)) == (ssize_t) n;
    ok = ok && memcmp(buf.data(), (const char *) data + (size_t) e*part_bytes, part_bytes) == 0;
    close(fd);
    return ok;
}

int tosh_hostcache_n_files(void) {
    std::lock_guard<std::mutex> lock(g.m);
    return (int) g.paths.size();
}

const char * tosh_hostcache_error(void) {
    return g.error.empty() ? nullptr : g.error.c_str();
}

uint64_t tosh_hostcache_cold_reads(void) {
    return g.cold_total.load();
}

bool tosh_hostcache_warmup_done(void) {
    if (g.base == nullptr || g.ready) return g.error.empty();
    // never ready with an unlocked cache
    if (g.locker.joinable()) g.locker.join();
    std::lock_guard<std::mutex> lock(g.m);
    g.ready = true;
    g.t_ready = now_us();
    uint64_t hits = 0;
    for (int k = 0; k < TOSH_HC_KINDS; ++k) hits += g.hit[k];
    if (!g.error.empty()) fprintf(stderr, "tosh_hostcache: %s\n", g.error.c_str());
    fprintf(stderr, "tosh_hostcache: ready %.0f ms after the cache started | resident %.0f MiB (%.1f%% of %.0f locked) | warm-up read %llu experts (%.0f MiB), hits %llu | fill %llu experts (%.0f MiB) so far\n",
            (g.t_ready - g.t_start)/1000, g.resident_bytes/1048576.0, 100.0*g.resident_bytes/std::max<size_t>(1, g.locked_bytes), g.locked_bytes/1048576.0,
            (unsigned long long) g.loaded_n[1], g.loaded_bytes[1]/1048576.0, (unsigned long long) hits,
            (unsigned long long) g.fill_n, g.fill_bytes/1048576.0);
    return g.error.empty();
}

void tosh_hostcache_report(void) {
    std::lock_guard<std::mutex> lock(g.m);
    if (g.base == nullptr) return;
    static const char * names[TOSH_HC_KINDS] = { "decode", "wide", "promote", "prefill", "verify" };
    for (int k = 0; k < TOSH_HC_KINDS; ++k) {
        if (g.req[k] == 0) continue;
        fprintf(stderr, "tosh_hostcache: %-7s requests %llu, warm hits %llu (%.2f%%), cold %llu, waited %.1f ms\n", names[k],
                (unsigned long long) g.req[k], (unsigned long long) g.hit[k], 100.0*g.hit[k]/g.req[k], (unsigned long long) g.cold[k], g.wait_us[k]/1000);
    }
    auto pq = [](std::vector<float> v, double f) { if (v.empty()) return 0.0; std::sort(v.begin(), v.end()); return (double) v[std::min(v.size() - 1, (size_t) (f*v.size()))]; };
    for (int c = 0; c < 2; ++c) {
        fprintf(stderr, "tosh_hostcache: %s reads %llu (%.1f MiB) | queue wait p50 %.0f p95 %.0f p99 %.0f us | read p50 %.0f p95 %.0f p99 %.0f us\n",
                c ? "read-back" : "demand   ", (unsigned long long) g.rd_n[c], g.rd_bytes[c]/1048576.0,
                pq(g.qwait_us[c], 0.5), pq(g.qwait_us[c], 0.95), pq(g.qwait_us[c], 0.99), pq(g.rd_us[c], 0.5), pq(g.rd_us[c], 0.95), pq(g.rd_us[c], 0.99));
    }
    for (int k = 0; k < TOSH_HC_KINDS; ++k) {
        if (g.req[k] == 0) continue;
        fprintf(stderr, "tosh_hostcache: %-7s waits: own reads %.1f ms, another demand %.1f ms, a read-back %.1f ms\n", names[k],
                g.wait_own_us[k]/1000, g.wait_co_dem_us[k]/1000, g.wait_co_bg_us[k]/1000);
    }
    const uint64_t bg_left = [] { uint64_t n = 0; for (auto u : g.bg_unused) n += u; return n; }();
    fprintf(stderr, "tosh_hostcache: read-backs used %llu (%llu while loading), evicted unused %llu, dropped unused %llu, still unused %llu\n",
            (unsigned long long) g.bg_used, (unsigned long long) g.bg_used_loading, (unsigned long long) g.bg_wasted,
            (unsigned long long) g.bg_dropped_unused, (unsigned long long) bg_left);
    fprintf(stderr, "tosh_hostcache: prompt lookahead queued %llu reads\n", (unsigned long long) g.ahead_queued);
    fprintf(stderr, "tosh_hostcache: file reads: first touch %llu, capacity miss %llu, read-back %llu\n",
            (unsigned long long) (g.rd_n[0] > g.reloads ? g.rd_n[0] - g.reloads : 0), (unsigned long long) g.reloads, (unsigned long long) g.rd_n[1]);
    fprintf(stderr, "tosh_hostcache: read-backs moved ahead for a waiting demand %llu | promoted copies kept %llu, refreshed on eviction %llu\n",
            (unsigned long long) g.boosted, (unsigned long long) g.kept_on_promotion, (unsigned long long) g.refreshed);
    std::vector<float> v = g.read_us;
    std::sort(v.begin(), v.end());
    auto q = [&](double f) { return v.empty() ? 0.0 : (double) v[std::min(v.size() - 1, (size_t) (f*v.size()))]; };
    fprintf(stderr, "tosh_hostcache: read back on eviction %llu, copied back from VRAM %llu\n",
            (unsigned long long) g.prefetches, (unsigned long long) g.stored);
    if (g.t_ready > 0) {
        fprintf(stderr, "tosh_hostcache: loaded before ready: warm-up %llu (%.0f MiB), fill %llu (%.0f MiB, %llu superseded, stopped %.0f ms after ready) | used after ready: warm-up %llu (%.0f MiB), fill %llu (%.0f MiB)\n",
                (unsigned long long) g.loaded_n[1], g.loaded_bytes[1]/1048576.0, (unsigned long long) g.loaded_n[2], g.loaded_bytes[2]/1048576.0,
                (unsigned long long) g.fill_dup, g.fill_stop_t > 0 ? (g.fill_stop_t - g.t_ready)/1000 : -1.0,
                (unsigned long long) g.reuse_n[1], g.reuse_bytes[1]/1048576.0, (unsigned long long) g.reuse_n[2], g.reuse_bytes[2]/1048576.0);
    }
    fprintf(stderr, "tosh_hostcache: reads %llu (%.1f MiB), coalesced %llu, evictions %llu, reloads %llu | expert read p50 %.0f p95 %.0f p99 %.0f us | resident %.0f MiB, peak %.0f MiB of %.0f locked\n",
            (unsigned long long) g.reads, g.read_bytes/1048576.0, (unsigned long long) g.coalesced, (unsigned long long) g.evictions,
            (unsigned long long) g.reloads, q(0.5), q(0.95), q(0.99), g.resident_bytes/1048576.0,
            g.peak_bytes/1048576.0, g.locked_bytes/1048576.0);
}

}  // extern "C"
