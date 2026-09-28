#include "kvmem/kvmem_runtime.hpp"
#include "kvmem/kvmem_page_table.hpp"
#include "llama-kvmem-quant.h"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <unistd.h>
#include <vector>

using namespace kvmem;

static int g_fail = 0;
#define CHECK(cond)                                                            \
    do {                                                                       \
        if (!(cond)) {                                                         \
            std::printf("FAIL %s:%d  %s\n", __FILE__, __LINE__, #cond);        \
            ++g_fail;                                                          \
        }                                                                      \
    } while (0)

struct RecordingBackend : KvMemBackend {
    int32_t next = 0;
    std::vector<int32_t> allocs;
    std::vector<int32_t> frees;
    std::vector<std::string> ops;

    int32_t alloc_gpu_slot() override {
        const int32_t s = next++;
        allocs.push_back(s);
        ops.push_back("alloc");
        return s;
    }
    void free_gpu_slot(int32_t slot) override {
        frees.push_back(slot);
        ops.push_back("free");
    }
};

static KvMemRuntimeConfig make_cfg() {
    KvMemRuntimeConfig cfg;
    cfg.store.block_tokens = 32;
    cfg.store.select_budget = 32 * 4;
    cfg.store.sink_blocks = 1;
    cfg.store.recent_blocks = 1;
    cfg.store.estimated_block_bytes = 1024;
    cfg.cpu_bytes = 1024 * 16;
    return cfg;
}

static void test_stage_out_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    auto plan = rt.prepare_reselect();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    // All GPU frees (evictions) must precede any new alloc.
    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());
}

static void test_high_overlap_skips_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    rt.reselect();
    const auto first = rt.last_plan();
    CHECK(first.remaps.size() == 4);

    auto second = rt.prepare_reselect();
    CHECK(second.stage_in.empty());
    CHECK(second.gpu_reused_blocks == 4);
    for (const auto &rm : second.remaps) {
        CHECK(rm.skip || rm.working_k_resident);
    }
    rt.finish_reselect();
}

static void test_pressure_keeps_sink_and_tail() {
    KvMemRuntime rt(make_cfg());
    rt.register_append(32 * 10);
    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    CHECK(plan.remaps.front().block_id == 0);
    CHECK(plan.remaps.back().block_id == 9);
    rt.finish_reselect();
}

static void test_maybe_offload_evicts_before_stage_in() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    be.allocs.clear();
    be.frees.clear();

    const uint32_t pool = 32 * 4;
    CHECK(rt.maybe_offload_during_prefill(/*incoming=*/32, /*resident=*/32 * 10, pool));
    CHECK(rt.last_plan().remaps.size() == 4);
    rt.finish_reselect();

    bool saw_alloc = false;
    bool order_ok = true;
    for (const auto &op : be.ops) {
        if (op == "alloc") saw_alloc = true;
        if (op == "free" && saw_alloc) order_ok = false;
    }
    CHECK(order_ok);
    CHECK(!be.frees.empty());

    KvMemRuntime idle(make_cfg());
    idle.register_append(32 * 2);
    CHECK(!idle.maybe_offload_during_prefill(32, 32, 32 * 8));
}

static void test_cpu_full_spills_to_nvme_and_roundtrips() {
    struct MemoryBackend : KvMemBackend {
        uint64_t slot_bytes = 64;
        int32_t next = 0;
        std::map<int32_t, std::vector<uint8_t>> gpu;
        int32_t alloc_gpu_slot() override {
            const int32_t s = next++;
            gpu[s].assign(slot_bytes, static_cast<uint8_t>(s + 1));
            return s;
        }
        void free_gpu_slot(int32_t slot) override { gpu.erase(slot); }
        void copy_block_to_host(uint32_t, int32_t gpu_slot, void *host,
                                uint64_t bytes) override {
            auto it = gpu.find(gpu_slot);
            if (it == gpu.end() || !host) return;
            std::memcpy(host, it->second.data(),
                        static_cast<size_t>(std::min(bytes, slot_bytes)));
        }
        void copy_block_from_host(uint32_t, int32_t gpu_slot, const void *host,
                                  uint64_t bytes) override {
            if (!host) return;
            gpu[gpu_slot].assign(static_cast<const uint8_t *>(host),
                                 static_cast<const uint8_t *>(host) +
                                     static_cast<size_t>(bytes));
        }
    };

    const char *base = std::getenv("TMPDIR");
    if (!base) base = "/tmp";
    const std::string dir = std::string(base) + "/kvmem_p32_nvme";

    MemoryBackend be;
    KvMemRuntimeConfig cfg = make_cfg();
    cfg.store.estimated_block_bytes = 64;
    cfg.cpu_bytes = 64 * 2;          // two CPU slots
    cfg.nvme_bytes = 64 * 8;
    cfg.nvme_dir = dir;
    KvMemRuntime rt(cfg, &be);
    CHECK(rt.cpu_tier() && rt.cpu_tier()->enabled());
    CHECK(rt.nvme_tier() && rt.nvme_tier()->enabled());

    rt.register_append(32 * 10);
    for (uint32_t id = 0; id < 10; ++id) {
        rt.store().set_block_tier(id, KvTier::GPU);
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }

    auto plan = rt.prepare_prefill_pressure();
    CHECK(plan.remaps.size() == 4);
    rt.finish_reselect();

    uint32_t on_nvme = 0;
    for (const auto &b : rt.store().blocks()) {
        if (b.nvme_slot >= 0 || b.tier == KvTier::SSD) {
            ++on_nvme;
        }
    }
    CHECK(on_nvme > 0);

    // Bring a spilled middle block back via retrieval scores.
    std::vector<double> scores(10, 0.0);
    scores[4] = 100.0;
    rt.store().set_retrieval_scores(scores);
    auto back = rt.prepare_reselect();
    bool staged = false;
    for (uint32_t id : back.stage_in) {
        if (id == 4) staged = true;
    }
    CHECK(staged);
    rt.finish_reselect();
    CHECK(rt.store().blocks()[4].gpu_slot >= 0);
    const auto &payload = be.gpu[rt.store().blocks()[4].gpu_slot];
    CHECK(!payload.empty());
    // Original GPU slot for block 4 was 4, filled with byte 5.
    CHECK(payload[0] == 5);
}

static void test_selection_preview_and_resident_commit() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    rt.register_append(32 * 3 + 7);
    for (uint32_t id = 0; id < rt.store().block_count(); ++id) {
        rt.store().set_block_gpu_slot(id, be.alloc_gpu_slot());
    }
    be.ops.clear();
    const auto before = rt.store().blocks();
    const auto selected = rt.preview_reselect();
    CHECK(selected.size() == 4);
    CHECK(be.ops.empty());
    for (uint32_t id = 0; id < before.size(); ++id) {
        const auto & after = rt.store().blocks()[id];
        CHECK(after.gpu_slot == before[id].gpu_slot);
        CHECK(after.in_working_set == before[id].in_working_set);
        CHECK(after.baked_pos == before[id].baked_pos);
    }
    CHECK(!rt.commit_resident_selection({0, 1, 2}));
    CHECK(!rt.store().blocks()[0].in_working_set);
    CHECK(rt.commit_resident_selection(selected));
    CHECK(be.ops.empty());
    CHECK(rt.last_plan().total_window_tokens == 103);
    CHECK(rt.store().blocks().back().n_tokens == 7);
    auto pending = rt.prepare_selection(selected);
    CHECK(!rt.commit_resident_selection(selected));
    rt.finish_reselect();
    CHECK(rt.commit_resident_selection(selected));
}

// P1: host arena extent geometry. make_cfg gives 16 slots of 1024B.
namespace kvmem {
struct kvmem_runtime_test_access {
    static const uint8_t *cpu_ptr(const KvMemRuntime & rt, int32_t slot) {
        return rt.cpu_ptr(slot);
    }
};
} // namespace kvmem
static void test_cpu_ptr_extent_checks() {
    RecordingBackend be;
    KvMemRuntime rt(make_cfg(), &be);
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, -1) == nullptr);
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, 0) != nullptr);
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, 15) != nullptr);
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, 16) == nullptr); // upper bound (was OOB before P1)
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, 1000000) == nullptr);
    CHECK(kvmem_runtime_test_access::cpu_ptr(rt, 2147483647) == nullptr); // multiply-overflow shape
}

// P1: page table generations, ownership, session release.
static void test_page_table_sessions() {
    KvmemPageTable t;
    t.reset(4);
    CHECK(t.valid());
    CHECK(t.n_free() == 4);
    const KvmemPageAlloc a0 = t.alloc(0, 10);
    const KvmemPageAlloc a1 = t.alloc(1, 20);
    CHECK(a0.slot >= 0 && a1.slot >= 0 && a0.slot != a1.slot);
    CHECK(a0.gen >= 1 && a1.gen >= 1);
    CHECK(t.valid());
    CHECK(!t.free(a0.slot, 1, a0.gen)); // wrong owner rejected
    CHECK(!t.free(a0.slot, 0, a0.gen + 1)); // stale generation rejected
    CHECK(t.valid());
    CHECK(t.free(a0.slot, 0, a0.gen));
    CHECK(t.double_free_drops() == 0);
    CHECK(!t.free(a0.slot, -2, 0)); // double free rejected, not re-pushed
    CHECK(t.double_free_drops() == 1);
    CHECK(t.n_free() == 3);
    const KvmemPageAlloc a2 = t.alloc(1, 30);
    CHECK(a2.slot == a0.slot); // LIFO reuse
    CHECK(a2.gen != a0.gen); // generation bumped: old handles are stale
    CHECK(t.note_resident(a1.slot, 21, 1)); // live page: logical refresh ok
    CHECK(t.release_session(0) == 0); // session 0 holds nothing now
    CHECK(t.release_session(1) == 2); // both live pages belong to session 1
    CHECK(t.valid());
    CHECK(t.n_free() == 4);
    const KvmemPageAlloc b0 = t.alloc(-1, 5);
    CHECK(t.release_session(-1) == 1); // negative releases the whole pool
    CHECK(t.valid());
    CHECK(b0.slot >= 0);
}

// P2: F16 <-> Q8 host-row round trip stays within quant noise; identity and
// bad-dims behave.
static void test_p2_host_row_roundtrip() {
    const uint32_t nrows = 5, dim = 1024;
    std::vector<float> orig(nrows * dim);
    for (uint32_t i = 0; i < nrows * dim; ++i) {
        orig[i] = 0.02f * std::sin(i * 0.11f) * std::cos(i * 0.031f);
    }
    std::vector<uint8_t> f16(nrows * dim * 2), q8(nrows * 1088), back(nrows * dim * 2);
    ggml_fp32_to_fp16_row(orig.data(), reinterpret_cast<ggml_fp16_t *>(f16.data()),
                          (int64_t) nrows * dim);
    CHECK(kvmem_rows_f16_to_host(GGML_TYPE_Q8_0, f16.data(), q8.data(), nrows, dim));
    CHECK(kvmem_rows_host_to_f16(GGML_TYPE_Q8_0, q8.data(), back.data(), nrows, dim));
    std::vector<float> rt(nrows * dim);
    ggml_fp16_to_fp32_row(reinterpret_cast<const ggml_fp16_t *>(back.data()), rt.data(),
                          (int64_t) nrows * dim);
    double se = 0.0;
    for (uint32_t i = 0; i < nrows * dim; ++i) {
        const double d = (double) rt[i] - orig[i];
        se += d * d;
    }
    CHECK(std::sqrt(se / (nrows * dim)) < 0.05);
    CHECK(kvmem_rows_f16_to_host(GGML_TYPE_F16, f16.data(), back.data(), nrows, dim));
    CHECK(std::memcmp(f16.data(), back.data(), f16.size()) == 0);
    CHECK(!kvmem_rows_f16_to_host(GGML_TYPE_Q8_0, f16.data(), q8.data(), nrows, 1000));
    CHECK(!kvmem_rows_host_to_f16(GGML_TYPE_Q8_0, q8.data(), back.data(), 0, dim));
}

int main() {
    test_selection_preview_and_resident_commit();
    test_stage_out_before_stage_in();
    test_high_overlap_skips_stage_in();
    test_pressure_keeps_sink_and_tail();
    test_maybe_offload_evicts_before_stage_in();
    test_cpu_full_spills_to_nvme_and_roundtrips();
    test_cpu_ptr_extent_checks();
    test_page_table_sessions();
    test_p2_host_row_roundtrip();
    if (g_fail != 0) {
        std::printf("FAILED: %d check(s)\n", g_fail);
        return 1;
    }
    std::printf("OK\n");
    return 0;
}
