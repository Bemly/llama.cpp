#include "tosh-plan.h"

#include "common.h"
#include "fit.h"
#include "log.h"

#include "gguf.h"
#include "llama.h"

#ifdef TOSH_ENABLE_DYNAMIC_MOE
#include "tosh-dmoe.h"
#endif

#include <algorithm>
#include <climits>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <list>
#include <regex>

#ifdef __APPLE__
#include <chrono>
#include <thread>
#include <mach/mach.h>
#include <sys/sysctl.h>
#endif

static constexpr double MiB = 1048576.0;

// safety reserve: a share of the card, bounded on small and large cards
static constexpr double RESERVE_FRACTION = 0.08;
static constexpr double RESERVE_MIN_MIB  = 768.0;
static constexpr double RESERVE_MAX_MIB  = 1536.0;
// RAM left to the rest of the system once a plan's host memory is taken
static constexpr double HOST_MARGIN_MIB  = 2048.0;
// experimental bounded host cache: (VRAM arena + RAM cache) / bank below this streams every token
static constexpr double BOUNDED_MIN_COVERAGE  = 1.0;
static constexpr double BOUNDED_GOOD_COVERAGE = 1.25;
// TOSH_AUTO_DISK_EXPERTS=1: below full coverage the experts no cache holds are read from the model file
static constexpr double BOUNDED_DISK_MIN_COVERAGE = 0.5;
// TOSH_AUTO_HOST_BANK=lean: RAM holds the experts the VRAM arena does not, plus this margin
static constexpr double LEAN_COVERAGE         = 1.08;
// reclaimable RAM lost between the snapshot and the commit that makes the plan stale
static constexpr double STALE_DROP_MIB = 1024.0;

namespace {

struct moe_info {
    bool   moe       = false;
    int    n_layer   = 0;
    int    n_moe     = 0;       // layers with an expert bank
    int    n_expert  = 0;
    int    n_used    = 0;
    int    n_ctx_train = 0;
    double bank_mib  = 0;       // every expert weight
    double max_tensor_mib = 0;  // one layer's largest expert tensor: a prefill slot
    double embd_mib  = 0;       // token embeddings, kept in host memory
    double lazy_mib  = 0;       // read from the file on demand: page cache, never resident or locked in full
    double total_mib = 0;       // every tensor, across all shards
    // the RAM cache keeps one pool per expert size and each must hold a whole layer
    double warm_floor_mib = 0;
    // sliding-window cache of one sequence per cache element size: what a context checkpoint holds
    double swa_elems = 0;
    int    n_swa     = 0;
    // recurrent state of one sequence, every recurrent layer: what a hybrid model's checkpoint holds
    double recr_elems = 0;
    int    n_split   = 1;
    std::string error;          // the model's metadata could not be read completely
};

struct host_mem {
    double ram_mib        = 0;
    double wired_mib      = 0;
    double wire_limit_mib = 0;
    double available_mib  = 0;   // what the system can hand out now without compressing or swapping
    // runtime policy: free, speculative and purgeable pages plus inactive file cache; inactive
    // anonymous pages only come back through the compressor or swap, so they do not count
    double reclaimable_mib = 0;
    double compressed_mib  = 0;
    double swap_used_mib   = 0;
    double self_rss_mib    = 0;
    int    pressure_level  = 1;  // kern.memorystatus_vm_pressure_level: 1 normal, 2 warn, 4 critical
    bool   swap_growing    = false;
    bool   simulated       = false;
};

struct probe {
    bool   ok = false;
    double model = 0, context = 0, compute = 0;   // GPU, MiB
    double host_model = 0, host_other = 0;
};

double env_mib(const char * k, double d) {
    const char * v = getenv(k);
    return v ? atof(v) : d;
}

// metadata comes from the first shard, tensors from every shard of a split model
moe_info read_moe_info(const char * path, llama_lazy_mode lazy) {
    moe_info mi;
    gguf_init_params gp = { /*.no_alloc =*/ true, /*.ctx =*/ nullptr };
    gguf_context * g = gguf_init_from_file(path, gp);
    if (g == nullptr) {
        mi.error = "cannot read the model file";
        return mi;
    }
    auto get_u32 = [&](const std::string & k) -> int {
        const int64_t id = gguf_find_key(g, k.c_str());
        return id < 0 ? 0 : (int) gguf_get_val_u32(g, id);
    };
    const int64_t aid = gguf_find_key(g, "general.architecture");
    const std::string arch = aid < 0 ? "" : gguf_get_val_str(g, aid);
    mi.n_layer     = get_u32(arch + ".block_count");
    mi.n_expert    = get_u32(arch + ".expert_count");
    mi.n_used      = get_u32(arch + ".expert_used_count");
    mi.n_ctx_train = get_u32(arch + ".context_length");
    // sliding-window layers: heads x (key + value) per token, over the window
    auto u32_or = [&](const std::string & k, int d) { const int64_t id = gguf_find_key(g, k.c_str()); return id < 0 ? d : (int) gguf_get_val_u32(g, id); };
    mi.n_swa = u32_or(arch + ".attention.sliding_window", 0);
    if (mi.n_swa > 0 && mi.n_layer > 0) {
        const int64_t hid = gguf_find_key(g, (arch + ".attention.head_count_kv").c_str());
        const int64_t pid = gguf_find_key(g, (arch + ".attention.sliding_window_pattern").c_str());
        const int kl = u32_or(arch + ".attention.key_length_swa", u32_or(arch + ".attention.key_length", 128));
        const int vl = u32_or(arch + ".attention.value_length_swa", u32_or(arch + ".attention.value_length", 128));
        for (int il = 0; il < mi.n_layer; ++il) {
            // without a pattern every layer is counted, which only overstates
            const bool swa = pid < 0 || gguf_get_arr_type(g, pid) != GGUF_TYPE_BOOL || il >= (int) gguf_get_arr_n(g, pid) ||
                             ((const bool *) gguf_get_arr_data(g, pid))[il];
            if (!swa) continue;
            int heads = 0;
            if (hid >= 0 && gguf_get_kv_type(g, hid) == GGUF_TYPE_ARRAY && il < (int) gguf_get_arr_n(g, hid)) {
                heads = ((const int32_t *) gguf_get_arr_data(g, hid))[il];
            } else if (hid >= 0 && gguf_get_kv_type(g, hid) == GGUF_TYPE_UINT32) {
                heads = (int) gguf_get_val_u32(g, hid);
            }
            mi.swa_elems += (double) heads*(kl + vl);
        }
    }
    const int d_state = u32_or(arch + ".ssm.state_size", 0);
    const int d_inner = u32_or(arch + ".ssm.inner_size", 0);
    if (d_state > 0 && d_inner > 0 && mi.n_layer > 0) {
        const int d_conv   = u32_or(arch + ".ssm.conv_kernel", 0);
        const int n_group  = u32_or(arch + ".ssm.group_count", 0);
        const int interval = u32_or(arch + ".full_attention_interval", 0);
        const int n_main   = mi.n_layer - u32_or(arch + ".nextn_predict_layers", 0);
        int n_recr = 0;
        for (int il = 0; il < n_main; ++il) n_recr += interval <= 0 || (il + 1) % interval != 0;
        mi.recr_elems = (double) n_recr*((double) std::max(0, d_conv - 1)*(d_inner + 2*n_group*d_state) + (double) d_state*d_inner);
    }
    const int64_t sid = gguf_find_key(g, "split.count");
    mi.n_split = sid < 0 ? 1 : std::max(1, (int) gguf_get_val_u16(g, sid));

    std::vector<std::string> paths = { path };
    if (mi.n_split > 1) {
        char prefix[PATH_MAX] = {}, shard[PATH_MAX] = {};
        if (llama_split_prefix(prefix, sizeof(prefix), path, 0, mi.n_split) == 0) {
            mi.error = string_format("SPLIT_GGUF_METADATA_ERROR: %s is not the first of %d shards", path, mi.n_split);
            gguf_free(g);
            return mi;
        }
        for (int i = 1; i < mi.n_split; ++i) {
            llama_split_path(shard, sizeof(shard), prefix, i, mi.n_split);
            paths.push_back(shard);
        }
    }

    const std::regex bank("^blk\\.([0-9]+)\\.ffn_(up|down|gate|gate_up)_exps\\.weight$");
    std::vector<double> layer_mib(mi.n_layer + 1, 0.0);
    for (size_t f = 0; f < paths.size(); ++f) {
        gguf_context * gs = f == 0 ? g : gguf_init_from_file(paths[f].c_str(), gp);
        if (gs == nullptr) {
            mi.error = string_format("SPLIT_GGUF_METADATA_ERROR: cannot read shard %zu of %d: %s", f + 1, mi.n_split, paths[f].c_str());
            break;
        }
        for (int64_t i = 0; i < gguf_get_n_tensors(gs); ++i) {
            const std::string name = gguf_get_tensor_name(gs, i);
            const double mib = gguf_get_tensor_size(gs, i)/MiB;
            mi.total_mib += mib;
            if (name == "token_embd.weight") {
                mi.embd_mib = mib;
            }
            // the loader's rule for the one tensor the archs mark lazy (gemma4, qwen4exp)
            if (name == "per_layer_token_embd.weight" && lazy != LLAMA_LAZY_MODE_OFF &&
                (lazy == LLAMA_LAZY_MODE_ON || mib > 4096.0)) {
                mi.lazy_mib += mib;
            }
            std::smatch m;
            if (!std::regex_match(name, m, bank)) {
                continue;
            }
            mi.bank_mib += mib;
            mi.max_tensor_mib = std::max(mi.max_tensor_mib, mib);
            const int il = std::stoi(m[1]);
            if (il >= 0 && il < (int) layer_mib.size()) layer_mib[il] += mib;
        }
        if (gs != g) gguf_free(gs);
    }
    gguf_free(g);
    std::vector<double> sizes;
    for (double l : layer_mib) {
        if (l <= 0) continue;
        mi.n_moe++;
        if (std::none_of(sizes.begin(), sizes.end(), [&](double s) { return std::fabs(s - l) < 1e-6; })) sizes.push_back(l);
    }
    for (double s : sizes) mi.warm_floor_mib += s;
    mi.moe = mi.error.empty() && mi.n_expert > 0 && mi.n_used > 0 && mi.n_moe > 0;
    return mi;
}

// the product switch plans host memory from the machine's state unless told otherwise
static bool g_runtime_default = false;

static bool disk_experts() {
    const char * v = getenv("TOSH_AUTO_DISK_EXPERTS");
    return v != nullptr && strcmp(v, "0") != 0;
}

bool host_policy_runtime() {
    const char * v = getenv("TOSH_AUTO_HOST_POLICY");
    return v == nullptr ? g_runtime_default : strcmp(v, "runtime") == 0;
}

static bool host_bank_lean() {
    const char * v = getenv("TOSH_AUTO_HOST_BANK");
    return v != nullptr && strcmp(v, "lean") == 0;
}

host_mem read_host_info() {
    host_mem h;
#ifdef __APPLE__
    uint64_t mem = 0; size_t len = sizeof(mem);
    if (sysctlbyname("hw.memsize", &mem, &len, nullptr, 0) == 0) h.ram_mib = mem/MiB;
    uint64_t lim = 0; len = sizeof(lim);
    if (sysctlbyname("vm.user_wire_limit", &lim, &len, nullptr, 0) == 0) h.wire_limit_mib = lim/MiB;
    vm_statistics64_data_t vs;
    mach_msg_type_number_t cnt = HOST_VM_INFO64_COUNT;
    if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vs, &cnt) == KERN_SUCCESS) {
        h.wired_mib = (double) vs.wire_count*vm_page_size/MiB;
        h.available_mib = (double) (vs.free_count + vs.inactive_count + vs.speculative_count + vs.purgeable_count)*vm_page_size/MiB;
    }
#endif
#ifdef __APPLE__
    if (host_policy_runtime()) {
        auto sample = [](vm_statistics64_data_t & v) {
            mach_msg_type_number_t n = HOST_VM_INFO64_COUNT;
            return host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &v, &n) == KERN_SUCCESS;
        };
        vm_statistics64_data_t a, b;
        if (sample(a)) {
            const double pg = (double) vm_page_size/MiB;
            h.reclaimable_mib = (a.free_count + a.speculative_count + a.purgeable_count +
                                 std::min<uint64_t>(a.inactive_count, a.external_page_count))*pg;
            h.compressed_mib = a.compressor_page_count*pg;
            // swap written now, not swap left over from earlier pressure
            std::this_thread::sleep_for(std::chrono::milliseconds(400));
            if (sample(b)) h.swap_growing = b.swapouts > a.swapouts;
        }
        xsw_usage sw = {}; size_t sl = sizeof(sw);
        if (sysctlbyname("vm.swapusage", &sw, &sl, nullptr, 0) == 0) h.swap_used_mib = sw.xsu_used/MiB;
        int lvl = 1; size_t ll = sizeof(lvl);
        if (sysctlbyname("kern.memorystatus_vm_pressure_level", &lvl, &ll, nullptr, 0) == 0) h.pressure_level = lvl;
        mach_task_basic_info_data_t ti; mach_msg_type_number_t tn = MACH_TASK_BASIC_INFO_COUNT;
        if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &ti, &tn) == KERN_SUCCESS) h.self_rss_mib = ti.resident_size/MiB;
    }
#endif
    // planner tests: a pressure level or swap activity this machine is not in
    if (const char * v = getenv("TOSH_AUTO_SIM_PRESSURE_LEVEL")) h.pressure_level = atoi(v);
    if (getenv("TOSH_AUTO_SIM_SWAP_GROWING")) h.swap_growing = true;
    const double sim = env_mib("TOSH_AUTO_SIM_RAM_MIB", 0);
    if (sim > 0) {
        // the runtime policy scales this machine's wired share instead of keeping its free RAM
        h.wired_mib = host_policy_runtime() ? h.wired_mib*sim/h.ram_mib : std::max(0.0, h.wired_mib - h.ram_mib + sim);
        h.wire_limit_mib = h.wire_limit_mib > 0 ? h.wire_limit_mib*sim/h.ram_mib : 0;
        h.available_mib = std::min(h.available_mib, sim);
        h.ram_mib = sim;
        // a simulated machine does not inherit this one's state: idle unless told otherwise
        h.simulated = true;
        h.reclaimable_mib = env_mib("TOSH_AUTO_SIM_RECLAIMABLE_MIB", 0.85*sim);
        h.compressed_mib = 0; h.swap_used_mib = 0; h.self_rss_mib = 0;
        if (!getenv("TOSH_AUTO_SIM_PRESSURE_LEVEL")) h.pressure_level = 1;
        if (!getenv("TOSH_AUTO_SIM_SWAP_GROWING")) h.swap_growing = false;
    }
    return h;
}

// keeps override patterns alive for the lifetime of the params that point at them
std::list<std::string> & pattern_store() {
    static std::list<std::string> s;
    return s;
}

const char * experts_pattern() {
    // weights only: expert biases are small and stay next to the ops that add them
    return "\\.ffn_(up|down|gate|gate_up)_exps\\.weight$";
}

probe run_probe(const common_params & params, int n_ctx, ggml_type kv, int ub, bool experts_on_host, int ncmoe) {
    probe p;
    llama_model_params   mp = common_model_params_to_llama(const_cast<common_params &>(params));
    llama_context_params cp = common_context_params_to_llama(params);
    mp.n_gpu_layers = 999;
    // repacked weights are not host memory the executor or the scheduler's uploads can read
    mp.use_extra_bufts = !(experts_on_host || ncmoe > 0) && mp.use_extra_bufts;
    std::vector<llama_model_tensor_buft_override> ov;
    if (experts_on_host) {
        ov.push_back({ experts_pattern(), ggml_backend_cpu_buffer_type() });
    }
    for (int i = 0; i < ncmoe; ++i) {
        pattern_store().push_back(llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX));
        ov.push_back({ pattern_store().back().c_str(), ggml_backend_cpu_buffer_type() });
    }
    ov.push_back({ nullptr, nullptr });
    mp.tensor_buft_overrides = ov.size() > 1 ? ov.data() : nullptr;
    cp.n_ctx    = n_ctx;
    cp.n_ubatch = ub;
    cp.n_batch  = std::max<uint32_t>(cp.n_batch, ub);
    cp.type_k   = kv;
    cp.type_v   = kv;
    try {
        std::vector<ggml_backend_dev_t> devs;
        uint32_t ngl = 0, ctx_train = 0, n_expert = 0;
        const auto mem = common_get_device_memory_data(params.model.path.c_str(), &mp, &cp, devs, ngl, ctx_train, n_expert,
                                                       GGML_LOG_LEVEL_ERROR);
        if (mem.size() < 2) {
            return p;
        }
        p.model   = mem[0].model/MiB;
        p.context = mem[0].context/MiB;
        p.compute = mem[0].compute/MiB;
        p.host_model = mem.back().model/MiB;
        p.host_other = (mem.back().context + mem.back().compute)/MiB;
        p.ok = true;
    } catch (const std::exception & e) {
        LOG_WRN("%s: memory probe failed: %s\n", __func__, e.what());
    }
    return p;
}

bool read_vram(double & free_mib, double & total_mib) {
    free_mib = total_mib = 0;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    size_t f = 0, t = 0;
    // free before this process allocated anything; the probes allocate nothing
    if (tosh_dmoe_vram(&f, &t) || tosh_vram_read(&f, &t)) {
        free_mib = f/MiB; total_mib = t/MiB;
    }
#endif
    if (total_mib <= 0) {
        ggml_backend_dev_t dev = ggml_backend_dev_by_type(GGML_BACKEND_DEVICE_TYPE_GPU);
        if (dev == nullptr) return false;
        size_t f = 0, t = 0;
        ggml_backend_dev_memory(dev, &f, &t);
        free_mib = f/MiB; total_mib = t/MiB;
    }
    const double sim = env_mib("TOSH_AUTO_SIM_VRAM_MIB", 0);
    if (sim > 0) {
        // what other processes hold on the real card is kept as the simulated card's baseline
        free_mib = std::max(0.0, sim - (total_mib - free_mib));
        total_mib = sim;
    }
    return total_mib > 0;
}

const char * kv_name(ggml_type t) { return ggml_type_name(t); }

} // namespace

const char * tosh_plan_state_name(tosh_plan_state s) {
    switch (s) {
        case TOSH_PLAN_NOT_PLANNED:            return "NOT_PLANNED";
        case TOSH_PLAN_FULL_GPU_OPTIMAL:       return "FULL_GPU_OPTIMAL";
        case TOSH_PLAN_DMOE_CONTEXT_OPTIMAL:   return "DMOE_CONTEXT_OPTIMAL";
        case TOSH_PLAN_DMOE_CAPACITY_REQUIRED: return "DMOE_CAPACITY_REQUIRED";
        case TOSH_PLAN_LEGACY_OFFLOAD_BETTER:  return "LEGACY_OFFLOAD_BETTER";
        case TOSH_PLAN_UNSUPPORTED:            return "UNSUPPORTED";
        case TOSH_PLAN_DMOE_BOUNDED_HOST:      return "DMOE_BOUNDED_HOST";
    }
    return "?";
}

const char * tosh_plan_mode_name(tosh_plan_mode m) {
    switch (m) {
        case TOSH_PLAN_MODE_NONE:           return "PLAN_NONE";
        case TOSH_PLAN_MODE_FULL_GPU:       return "PLAN_FULL_GPU";
        case TOSH_PLAN_MODE_FULL_HOST_DMOE: return "PLAN_FULL_HOST_DMOE";
        case TOSH_PLAN_MODE_BOUNDED_DMOE:   return "PLAN_BOUNDED_DMOE";
        case TOSH_PLAN_MODE_CLASSIC_NCMOE:  return "PLAN_CLASSIC_NCMOE";
        case TOSH_PLAN_MODE_UNSUPPORTED:    return "PLAN_UNSUPPORTED";
    }
    return "?";
}

static std::string jstr(const std::string & s) {
    std::string o = "\"";
    for (char c : s) o += (c == '"' || c == '\\') ? '\'' : c;
    return o + "\"";
}

static std::string jarr(const std::vector<std::string> & v) {
    std::string o = "[";
    for (size_t i = 0; i < v.size(); ++i) o += (i ? ", " : "") + jstr(v[i]);
    return o + "]";
}

static long long bytes(double mib) { return (long long) std::llround(mib*MiB); }

// the stable part for the app: codes and bytes, no sentences
static std::string plan_product_json(const tosh_plan & p) {
    std::string label = tosh_plan_mode_name(p.plan_mode);
    label = "plan.mode." + label.substr(5);
    for (auto & c : label) c = (char) tolower(c);
    const bool fallback = p.plan_mode == TOSH_PLAN_MODE_BOUNDED_DMOE || p.plan_mode == TOSH_PLAN_MODE_CLASSIC_NCMOE;
    std::string o = "{\"mode\": " + jstr(tosh_plan_mode_name(p.plan_mode)) + ", \"mode_label_key\": " + jstr(label);
    o += ", \"reason\": " + jstr(p.primary_reason) + ", \"warnings\": " + jarr(p.warnings) + ", \"limits\": " + jarr(p.limits);
    o += ", \"limiting_resource\": " + jstr(p.limiting_resource);
    o += ", \"fallback\": {\"selected\": " + std::string(fallback ? "true" : "false") + ", \"type\": " +
         jstr(fallback ? tosh_plan_mode_name(p.plan_mode) : "NONE") + "}";
    o += ", \"memory\": {\"host_policy\": " + jstr(p.host_policy.empty() ? "static" : p.host_policy) + ", \"pressure\": " + jstr(p.host_pressure);
    o += string_format(", \"physical_bytes\": %lld, \"reclaimable_bytes\": %lld, \"runtime_headroom_bytes\": %lld, \"static_reserve_bytes\": %lld",
                       bytes(p.host_ram_mib), bytes(p.reclaimable_mib), bytes(p.headroom_mib), bytes(p.static_reserve_mib));
    o += string_format(", \"effective_host_budget_bytes\": %lld, \"lockable_remaining_bytes\": %lld, \"swap_used_bytes\": %lld, \"compressed_bytes\": %lld",
                       bytes(p.effective_mib), bytes(p.lockable_mib), bytes(p.swap_used_mib), bytes(p.compressed_mib));
    o += string_format(", \"projected_rss_bytes\": %lld, \"projected_vram_bytes\": %lld, \"vram_total_bytes\": %lld, \"vram_free_bytes\": %lld, \"prompt_cache_bytes\": %lld, \"context_checkpoints\": %d, \"checkpoint_bytes\": %lld}",
                       bytes(p.projected_rss_mib), bytes(p.private_mib), bytes(p.vram_total_mib), bytes(p.vram_free_mib), bytes(p.prompt_cache_mib),
                       p.n_ctx_checkpoints, bytes(p.checkpoint_mib));
    o += string_format(", \"dmoe\": {\"expert_bank_bytes\": %lld, \"hot_bytes\": %lld, \"warm_bytes\": %lld, \"warm_floor_bytes\": %lld, \"coverage\": %.3f, \"coverage_state\": %s}",
                       bytes(p.bank_mib), bytes(p.arena_mib), bytes(p.warm_mib), bytes(p.warm_floor_mib), p.coverage, jstr(p.coverage_state).c_str());
    o += string_format(", \"runtime\": {\"context\": %d, \"kv_type\": %s, \"kv_bytes\": %lld, \"ubatch\": %d, \"ncmoe_layers\": %d}",
                       p.n_ctx, jstr(ggml_type_name(p.kv)).c_str(), bytes(p.kv_mib), p.ubatch, p.ncmoe);
    o += string_format(", \"model\": {\"model_bytes\": %lld, \"split_count\": %d, \"host_weights_bytes\": %lld, \"host_lazy_bytes\": %lld}",
                       bytes(p.model_mib), p.n_split, bytes(p.host_weights_mib), bytes(p.host_lazy_mib));
    if (p.plan_mode == TOSH_PLAN_MODE_UNSUPPORTED) {
        o += string_format(", \"unsupported\": {\"required_host_bytes\": %lld, \"available_host_bytes\": %lld, \"required_vram_bytes\": %lld, \"available_vram_bytes\": %lld}",
                           bytes(p.required_host_mib), bytes(p.available_host_mib), bytes(p.required_vram_mib), bytes(p.available_vram_mib));
    }
    o += string_format(", \"recheck_ms\": %.0f}", p.recheck_ms);
    return o;
}

std::string tosh_plan_json(const tosh_plan & p) {
    std::string o = "{";
    o += string_format("\"plan_schema_version\": %d, \"product\": ", TOSH_PLAN_SCHEMA_VERSION) + plan_product_json(p) + ", ";
    o += "\"state\": " + jstr(tosh_plan_state_name(p.state));
    o += ", \"mode\": " + jstr(p.mode);
    o += ", \"code\": " + jstr(p.code) + ", \"limits\": [";
    for (size_t i = 0; i < p.limits.size(); ++i) o += (i ? ", " : "") + jstr(p.limits[i]);
    o += "]";
    o += ", \"reason\": " + jstr(p.reason);
    o += ", \"fallback\": " + jstr(p.fallback);
    o += ", \"kv\": " + jstr(kv_name(p.kv));
    o += string_format(", \"n_ctx\": %d, \"ubatch\": %d, \"ncmoe\": %d", p.n_ctx, p.ubatch, p.ncmoe);
    o += string_format(", \"reserve_mib\": %.0f, \"arena_mib\": %.0f, \"min_arena_mib\": %.0f", p.reserve_mib, p.arena_mib, p.min_arena_mib);
    o += string_format(", \"projected_private_mib\": %.0f, \"projected_free_mib\": %.0f", p.private_mib, p.free_mib);
    o += string_format(", \"vram_total_mib\": %.0f, \"vram_free_mib\": %.0f", p.vram_total_mib, p.vram_free_mib);
    o += string_format(", \"host_required_mib\": %.0f, \"host_ram_mib\": %.0f, \"host_available_mib\": %.0f, \"host_reserve_mib\": %.0f, \"bank_mib\": %.0f",
                       p.host_required_mib, p.host_ram_mib, p.host_available_mib, p.host_reserve_mib, p.bank_mib);
    o += string_format(", \"model_mib\": %.0f, \"host_weights_mib\": %.0f, \"warm_floor_mib\": %.0f, \"n_split\": %d",
                       p.model_mib, p.host_weights_mib, p.warm_floor_mib, p.n_split);
    o += ", \"mlock\": " + jstr(p.mlock) + ", \"dispersion\": " + jstr(p.dispersion);
    if (p.host_policy == "runtime") {
        o += ", \"host\": {\"policy\": " + jstr(p.host_policy) + ", \"simulated\": " + (p.host_simulated ? "true" : "false") +
             ", \"pressure\": " + jstr(p.host_pressure) + ", \"swap_growing\": " + (p.swap_growing ? "true" : "false") +
             ", \"full_host\": " + jstr(p.full_host_code) + ", \"bounded\": " + jstr(p.bounded_code);
        o += string_format(", \"reclaimable_mib\": %.0f, \"wired_mib\": %.0f, \"compressed_mib\": %.0f, \"swap_used_mib\": %.0f, \"self_rss_mib\": %.0f",
                           p.reclaimable_mib, p.wired_mib, p.compressed_mib, p.swap_used_mib, p.self_rss_mib);
        o += string_format(", \"static_reserve_mib\": %.0f, \"headroom_mib\": %.0f, \"static_cap_mib\": %.0f, \"dynamic_cap_mib\": %.0f, \"lockable_mib\": %.0f, \"effective_mib\": %.0f",
                           p.static_reserve_mib, p.headroom_mib, p.static_cap_mib, p.dynamic_cap_mib, p.lockable_mib, p.effective_mib);
        o += string_format(", \"warm_mib\": %.0f, \"coverage\": %.3f, \"projected_rss_mib\": %.0f, \"projected_left_mib\": %.0f}",
                           p.warm_mib, p.coverage, p.projected_rss_mib, p.projected_left_mib);
    }
    o += ", \"candidates\": [";
    for (size_t i = 0; i < p.candidates.size(); ++i) {
        const auto & c = p.candidates[i];
        o += i ? ", {" : "{";
        o += "\"name\": " + jstr(c.name) + ", \"mode\": " + jstr(c.mode) + ", \"kv\": " + jstr(kv_name(c.kv)) + ", \"valid\": " + (c.valid ? "true" : "false") + ", \"reason\": " + jstr(c.reason);
        o += string_format(", \"ubatch\": %d, \"ncmoe\": %d, \"private_mib\": %.0f, \"free_mib\": %.0f, \"arena_mib\": %.0f, \"kv_mib\": %.0f, \"compute_mib\": %.0f}",
                           c.ubatch, c.ncmoe, c.private_mib, c.free_mib, c.arena_mib, c.kv_mib, c.compute_mib);
    }
    return o + "]}";
}

static void plan_log(const tosh_plan & p) {
    LOG_DBG("Tosh auto memory plan\n");
    LOG_DBG("  requested context %d, VRAM %.1f GiB (%.1f free), reserve %.1f GiB\n",
            p.n_ctx, p.vram_total_mib/1024, p.vram_free_mib/1024, p.reserve_mib/1024);
    for (const auto & c : p.candidates) {
        LOG_DBG("  %-22s ub %4d  private %6.0f MiB  free %6.0f MiB  arena %6.0f MiB  %s: %s\n", c.name.c_str(), c.ubatch,
                c.private_mib, c.free_mib, c.arena_mib, c.valid ? "valid" : "rejected", c.reason.c_str());
    }
    if (p.host_policy == "runtime") {
        LOG_DBG("  host (%s): RAM %.1f GiB, reclaimable now %.1f, wired %.1f, compressed %.1f, swap %.1f%s, pressure %s\n",
                p.host_simulated ? "simulated machine" : "this machine now", p.host_ram_mib/1024, p.reclaimable_mib/1024, p.wired_mib/1024,
                p.compressed_mib/1024, p.swap_used_mib/1024, p.swap_growing ? " and growing" : "", p.host_pressure.c_str());
        LOG_DBG("  host budget to add: static %.1f GiB (reserve %.1f), current %.1f GiB (headroom %.1f), lockable %.1f GiB -> %.1f GiB\n",
                p.static_cap_mib/1024, p.static_reserve_mib/1024, p.dynamic_cap_mib/1024, p.headroom_mib/1024, p.lockable_mib/1024, p.effective_mib/1024);
        LOG_DBG("  full host bank %.1f GiB: %s | RAM cache %.1f GiB, coverage %.2f: %s | projected RSS %.1f GiB, reclaimable left %.1f GiB\n",
                p.bank_mib/1024, p.full_host_code.c_str(), p.warm_mib/1024, p.coverage, p.bounded_code.c_str(),
                p.projected_rss_mib/1024, p.projected_left_mib/1024);
    }
    std::string limits;
    for (const auto & l : p.limits) limits += (limits.empty() ? "" : ", ") + l;
    LOG_DBG("  model %.1f GiB in %d file%s, expert bank %.1f GiB, other weights in host memory %.1f GiB, read on demand %.1f GiB\n",
            p.model_mib/1024, p.n_split, p.n_split > 1 ? "s" : "", p.bank_mib/1024, p.host_weights_mib/1024, p.host_lazy_mib/1024);
    LOG_DBG("  selected %s (%s, %s): %s, KV %s, ubatch %d, arena %.0f MiB, projected free %.0f MiB, host %.1f GiB%s%s%s%s\n",
            p.mode.c_str(), p.code.c_str(), tosh_plan_state_name(p.state), p.reason.c_str(), kv_name(p.kv), p.ubatch, p.arena_mib,
            p.free_mib, p.host_required_mib/1024, p.fallback.empty() ? "" : "; ", p.fallback.c_str(),
            limits.empty() ? "" : "; limits: ", limits.c_str());
    std::string warn;
    for (const auto & w : p.warnings) warn += (warn.empty() ? "" : ",") + w;
    if (p.plan_mode == TOSH_PLAN_MODE_UNSUPPORTED) {
        LOG_INF("Tosh Dynamic MoE: mode=%s reason=%s limits=%s required_host=%.1fGiB available_host=%.1fGiB required_vram=%.1fGiB available_vram=%.1fGiB\n",
                p.code.c_str(), p.primary_reason.c_str(), limits.c_str(), p.required_host_mib/1024, p.available_host_mib/1024,
                p.required_vram_mib/1024, p.available_vram_mib/1024);
    } else {
        LOG_INF("Tosh Dynamic MoE: mode=%s reason=%s%s%s rss=%.1fGiB vram=%.1fGiB warm=%.1fGiB hot=%.1fGiB coverage=%.2f %s ctx=%d kv=%s ub=%d%s\n",
                p.code.c_str(), p.primary_reason.c_str(), warn.empty() ? "" : " warnings=", warn.c_str(), p.projected_rss_mib/1024,
                p.private_mib/1024, p.warm_mib/1024, p.arena_mib/1024, p.coverage, p.coverage_state.c_str(), p.n_ctx, kv_name(p.kv), p.ubatch,
                p.ncmoe > 0 ? string_format(" ncmoe=%d", p.ncmoe).c_str() : "");
    }
}

static void add_override(common_params & params, const char * pattern) {
    auto & ov = params.tensor_buft_overrides;
    for (auto & o : ov) {
        if (o.pattern == nullptr) {
            o = { pattern, ggml_backend_cpu_buffer_type() };
            break;
        }
    }
    if (ov.empty() || ov.back().pattern != nullptr) {
        ov.push_back({ nullptr, nullptr });
    }
}

// log, plan file and dry run; returns only when the caller should go on
static tosh_plan g_last_plan;
static bool      g_planned = false;

static void plan_emit(const tosh_plan & p) {
    g_last_plan = p;
    g_planned = true;
    plan_log(p);
    if (const char * f = getenv("TOSH_AUTO_PLAN_FILE")) {
        if (FILE * fp = fopen(f, "w")) {
            fprintf(fp, "%s\n", tosh_plan_json(p).c_str());
            fclose(fp);
        }
        if (getenv("TOSH_DMOE_STATUS_FILE") == nullptr) {
            setenv("TOSH_DMOE_STATUS_FILE", (std::string(f) + ".runtime").c_str(), 1);
        }
    }
    const char * dry = getenv("TOSH_AUTO_DRY_RUN");
    if (dry != nullptr && strcmp(dry, "exit") == 0) {
        // a caller previews the plan of these exact arguments without loading anything
        printf("%s\n", tosh_plan_json(p).c_str());
        fflush(stdout);
        exit(p.state == TOSH_PLAN_UNSUPPORTED ? 2 : 0);
    }
}

// a recurrent state is checkpointed exact when at least this many checkpoints fit, in 8 bits below that
static constexpr int RS_EXACT_MIN_CHECKPOINTS = 8;
// what a plan keeps even when the model takes its whole budget: this turn's checkpoints and the last one's
static constexpr int RS_MIN_CHECKPOINTS = 4;

// sets TOSH_RS_STATE_TYPE for the server when it was not set by hand; returns the checkpoint size it chose
static double rs_checkpoint_mib(double left_mib, double fixed_mib, double recr_elems, int n_wanted) {
    const char * rs = getenv("TOSH_RS_STATE_TYPE");
    if (rs != nullptr || recr_elems <= 0) {
        const std::string t = rs ? rs : "f32";
        const double b = t == "f32" ? 4.0 : t == "f16" || t == "bf16" ? 2.0 : 34.0/32.0;
        return fixed_mib + recr_elems*b/MiB;
    }
    const double exact = fixed_mib + recr_elems*4.0/MiB;
    if (left_mib >= std::min(n_wanted, RS_EXACT_MIN_CHECKPOINTS)*exact) {
        return exact;
    }
    setenv("TOSH_RS_STATE_TYPE", "q8_0", 1);
    LOG_WRN("%s: %.0f MiB left for checkpoints holds fewer than %d exact ones of %.1f MiB, keeping them in 8 bits\n",
            __func__, left_mib, RS_EXACT_MIN_CHECKPOINTS, exact);
    return fixed_mib + recr_elems*34.0/32.0/MiB;
}

void tosh_plan_checkpoints(common_params & params) {
    if (params.n_ctx_checkpoints <= 0 || params.model.path.empty()) {
        return;
    }
    const moe_info mi = read_moe_info(params.model.path.c_str(), params.lazy_mode);
    if (!mi.error.empty() || mi.recr_elems <= 0) {
        return;
    }
    const double left = read_host_info().available_mib - HOST_MARGIN_MIB;
    const double ckpt = rs_checkpoint_mib(left, 0.0, mi.recr_elems, params.n_ctx_checkpoints);
    const int fit = (int) std::max(0.0, std::floor(left/ckpt));
    if (fit < params.n_ctx_checkpoints) {
        LOG_WRN("%s: %d context checkpoints of %.1f MiB fit in %.0f MiB free, down from %d\n",
                __func__, fit, ckpt, left, params.n_ctx_checkpoints);
        params.n_ctx_checkpoints = fit;
    }
}

bool tosh_plan_apply(common_params & params, tosh_plan * out) {
    // precedence: TOSH_DMOE_* set by hand, then TOSH_AUTO, then --dynamic-moe, then the standard path
    const char * mode_env = getenv("TOSH_AUTO");
    g_runtime_default = params.dynamic_moe;
    if (mode_env == nullptr && params.dynamic_moe) mode_env = "1";
    if (mode_env == nullptr || strcmp(mode_env, "0") == 0 || strcmp(mode_env, "off") == 0) {
        return false;
    }
    if (getenv("TOSH_DMOE_CACHE_MIB") != nullptr || getenv("TOSH_DMOE_RESIDENCY") != nullptr) {
        LOG_WRN("%s: TOSH_DMOE_* set by hand, the auto plan steps aside\n", __func__);
        return false;
    }
    std::string force = (strcmp(mode_env, "1") == 0 || strcmp(mode_env, "auto") == 0) ? "" : mode_env;
    // nodmoe: auto without mixed execution, the retry after the expert bank could not be locked
    const bool no_dmoe = force == "nodmoe";
    if (no_dmoe) force.clear();

    tosh_plan p;
    const moe_info mi = read_moe_info(params.model.path.c_str(), params.lazy_mode);
    p.n_split = mi.n_split; p.model_mib = mi.total_mib; p.warm_floor_mib = mi.warm_floor_mib;
    if (!mi.error.empty()) {
        p.state = TOSH_PLAN_UNSUPPORTED; p.mode = "none";
        p.plan_mode = TOSH_PLAN_MODE_UNSUPPORTED; p.code = tosh_plan_mode_name(p.plan_mode);
        p.primary_reason = "SPLIT_GGUF_METADATA_ERROR"; p.limiting_resource = "METADATA";
        p.limits = { "SPLIT_GGUF_METADATA_ERROR", "NO_SAFE_FALLBACK" };
        p.reason = mi.error;
        plan_emit(p);
        if (out) *out = p;
        return false;
    }
    if (!mi.moe) {
        if (out) *out = p;
        return false;
    }

    p.n_ctx = params.n_ctx > 0 ? params.n_ctx : std::min(mi.n_ctx_train, 32768);
    p.bank_mib = mi.bank_mib;
    const double per_expert = mi.bank_mib/std::max(1, mi.n_moe*mi.n_expert);
    // below one token's worth of experts the arena mostly misses and the host does the work
    p.min_arena_mib = std::max(256.0, env_mib("TOSH_AUTO_MIN_ARENA_MIB", mi.n_moe*mi.n_used*per_expert));
    // each chunk touches most experts when every token picks a large share of them
    const bool high = mi.n_expert > 0 && (double) mi.n_used/mi.n_expert > 1.0/16;
    p.dispersion = high ? "high" : "normal";

    const host_mem hi0 = read_host_info();
    const auto t_snapshot = std::chrono::steady_clock::now();
    p.host_ram_mib     = hi0.ram_mib;
    p.host_reserve_mib = env_mib("TOSH_AUTO_HOST_RESERVE_MIB", std::max(6144.0, 0.25*hi0.ram_mib));

    // cache types to consider
    std::vector<ggml_type> kvs;
    const std::string kv_env = getenv("TOSH_AUTO_KV") ? getenv("TOSH_AUTO_KV") : "auto";
    if (kv_env == "auto") {
        kvs = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
    } else {
        const ggml_type t = kv_env == "f16" ? GGML_TYPE_F16 : kv_env == "q8_0" ? GGML_TYPE_Q8_0 :
                            kv_env == "turbo4" ? GGML_TYPE_TURBO4_0 : GGML_TYPE_COUNT;
        if (t == GGML_TYPE_COUNT) {
            LOG_WRN("%s: unknown TOSH_AUTO_KV=%s, using auto\n", __func__, kv_env.c_str());
            kvs = { GGML_TYPE_F16, GGML_TYPE_Q8_0 };
        } else {
            kvs = { t };
        }
    }
    // turbo4 costs some quality, so auto reaches for it only where F16 and Q8_0 leave the experts on
    // the host; the caller says whether the model and the GPU support it
    const char * turbo_env = getenv("TOSH_AUTO_KV_TURBO4");
    const bool turbo_fallback = kv_env == "auto" && turbo_env != nullptr && strcmp(turbo_env, "0") != 0;

    // measured, not estimated: each configuration is loaded without allocating. The probe with the
    // experts in host memory places ops as the real run does; the full model is that plus the bank,
    // since a probe of the full model puts the expert ops on the CPU and overstates compute.
    const int ubs[3] = { 512, 1024, 2048 };
    const int ub_full = params.n_ubatch;
    probe dm[4][4];
    bool dmoe_built = false;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    dmoe_built = true;
#endif
    std::vector<std::pair<ggml_type, probe>> full;
    auto probe_kv = [&](size_t k) {
        for (int u = 0; u < 3; ++u) dm[k][u] = run_probe(params, p.n_ctx, kvs[k], ubs[u], true, 0);
        int uf = -1;
        for (int u = 0; u < 3; ++u) if (ubs[u] == ub_full) uf = u;
        const probe pf = uf >= 0 ? dm[k][uf] : (dm[k][3] = run_probe(params, p.n_ctx, kvs[k], ub_full, true, 0));
        probe f = pf;
        f.model += mi.bank_mib;
        f.host_model = std::max(0.0, f.host_model - mi.bank_mib);
        full.push_back({ kvs[k], f });
    };
    for (size_t k = 0; k < kvs.size(); ++k) probe_kv(k);
    bool turbo_probed = false;
    if (!read_vram(p.vram_free_mib, p.vram_total_mib)) {
        LOG_WRN("%s: no GPU memory reading, planning off\n", __func__);
        return false;
    }
    // every weight the probe left in host memory besides the bank and what is read lazily
    p.host_weights_mib = dm[0][0].ok ? std::max(mi.embd_mib, dm[0][0].host_model - mi.bank_mib - mi.lazy_mib) : mi.embd_mib;
    p.host_lazy_mib = mi.lazy_mib;
    const bool runtime = host_policy_runtime() && dmoe_built;
    const tosh_plan base = p;

    // the decision from one host memory snapshot; run again when memory changed before committing
    auto decide = [&](const host_mem & hi) -> const tosh_plan_candidate * {
    p = base;
    p.host_available_mib = hi.available_mib;
    // a lazy table only needs its hot rows cached; the OS drops the rest under pressure
    const double host_engine = p.host_weights_mib + std::min(p.host_lazy_mib, 1024.0) + dm[0][0].host_other + 1024.0;
    const double host_dmoe   = mi.bank_mib + host_engine;
    std::string dmoe_block;
    if (!dmoe_built) {
        dmoe_block = "engine built without Dynamic MoE";
    } else if (hi.ram_mib > 0 && host_dmoe + p.host_reserve_mib > hi.ram_mib) {
        dmoe_block = string_format("host RAM: %.1f GiB needed + %.1f GiB reserve > %.1f GiB",
                                   host_dmoe/1024, p.host_reserve_mib/1024, hi.ram_mib/1024);
    } else if (hi.available_mib > 0 && host_dmoe + HOST_MARGIN_MIB > hi.available_mib) {
        dmoe_block = string_format("host RAM: %.1f GiB needed, %.1f GiB available now", host_dmoe/1024, hi.available_mib/1024);
    } else if (hi.wire_limit_mib > 0 && mi.bank_mib + p.host_weights_mib > hi.wire_limit_mib - hi.wired_mib) {
        dmoe_block = string_format("DMOE_HOST_BANK_NOT_LOCKABLE: %.1f GiB bank and host weights > %.1f GiB lockable",
                                   (mi.bank_mib + p.host_weights_mib)/1024, (hi.wire_limit_mib - hi.wired_mib)/1024);
    }
    p.mlock = dmoe_block.rfind("DMOE_HOST_BANK_NOT_LOCKABLE", 0) == 0 ? "not_lockable" : "required";

    // runtime host policy: every cap is memory this load may still add; the whole bank only when
    // it fits all three with the machine calm, else the largest RAM cache that does
    const double staging_mib = 128.0;
    double bounded_warm = 0;
    bool full_host_ok = true;
    const bool lean = runtime && host_bank_lean();
    if (runtime) {
        p.host_policy = "runtime";
        p.host_simulated = hi.simulated; p.swap_growing = hi.swap_growing;
        p.reclaimable_mib = hi.reclaimable_mib; p.wired_mib = hi.wired_mib; p.compressed_mib = hi.compressed_mib;
        p.swap_used_mib = hi.swap_used_mib; p.self_rss_mib = hi.self_rss_mib;
        p.host_pressure = hi.pressure_level >= 4 ? "CRITICAL" : hi.pressure_level >= 2 || hi.swap_growing ? "ELEVATED" : "NORMAL";
        p.static_reserve_mib = env_mib("TOSH_AUTO_HOST_RESERVE_MIB", std::max(6144.0, 0.20*hi.ram_mib));
        p.headroom_mib   = env_mib("TOSH_AUTO_HOST_HEADROOM_MIB", std::clamp(0.15*hi.ram_mib, 4096.0, 12288.0));
        p.static_cap_mib = std::max(0.0, hi.ram_mib - p.static_reserve_mib - hi.self_rss_mib);
        p.dynamic_cap_mib = std::max(0.0, hi.reclaimable_mib - p.headroom_mib);
        p.lockable_mib   = std::max(0.0, hi.wire_limit_mib - hi.wired_mib - 1024.0);
        p.effective_mib  = std::min(p.static_cap_mib, p.dynamic_cap_mib);
        p.full_host_code = p.host_pressure != "NORMAL"           ? "HOST_MEMORY_PRESSURE_HIGH" :
                           host_dmoe > p.static_cap_mib           ? "FULL_HOST_STATIC_LIMIT" :
                           host_dmoe > p.dynamic_cap_mib          ? "FULL_HOST_CURRENT_RAM_LIMIT" :
                           mi.bank_mib + p.host_weights_mib > p.lockable_mib ? "FULL_HOST_WIRE_LIMIT" : "FULL_HOST_SAFE";
        if (lean && p.full_host_code == "FULL_HOST_SAFE") p.full_host_code = "FULL_HOST_NOT_WANTED";
        full_host_ok = p.full_host_code == "FULL_HOST_SAFE";
        dmoe_block = full_host_ok ? "" : string_format("%s: %.1f GiB to add, %.1f GiB allowed now", p.full_host_code.c_str(),
                                                        host_dmoe/1024, std::min(p.effective_mib, p.lockable_mib + host_engine)/1024);
        if (full_host_ok) {
            p.bounded_code = "NOT_NEEDED";
        } else if (p.host_pressure == "CRITICAL") {
            p.bounded_code = "HOST_MEMORY_PRESSURE_HIGH";
        } else {
            // under elevated pressure the cache takes half of what is left
            const double budget = p.effective_mib*(p.host_pressure == "ELEVATED" ? 0.5 : 1.0) - host_engine - staging_mib;
            // the weights kept in RAM and the staging are locked too
            bounded_warm = std::min({ mi.bank_mib, budget, p.lockable_mib - host_engine - staging_mib });
            p.bounded_code = bounded_warm >= mi.warm_floor_mib ? "BOUNDED_HOST_SELECTED" :
                             p.lockable_mib < mi.warm_floor_mib ? "HOST_MEMORY_LOCK_LIMIT" :
                             bounded_warm > 0 ? "STRUCTURAL_WARM_TOO_SMALL" : "BOUNDED_HOST_CURRENT_RAM_TOO_LOW";
            // the cache refuses to start below one layer's experts per size, a prefill chunk can pin them all
            if (bounded_warm < mi.warm_floor_mib) {
                if (p.bounded_code == "STRUCTURAL_WARM_TOO_SMALL") {
                    dmoe_block += string_format("; STRUCTURAL_WARM_TOO_SMALL: %.0f MiB RAM cache < %.0f MiB, one layer per expert size",
                                                bounded_warm, mi.warm_floor_mib);
                }
                bounded_warm = 0;
            }
        }
        p.mlock = "required";
    }

    const double slots    = 3.0*mi.max_tensor_mib;   // prefill uploads
    const double overhead = env_mib("TOSH_DMOE_OVERHEAD_MIB", 64);
    const double weights_full = full[0].second.ok ? full[0].second.model : mi.bank_mib;
    const bool weights_fit = weights_full + 512.0 <= p.vram_free_mib;

    // the code of what rules a host mode out, from the checks above
    auto host_limit = [&](const std::string & why) -> const char * {
        return why.find("STATIC_LIMIT") != std::string::npos || why.find("reserve >") != std::string::npos ? "STATIC_RAM_LIMIT" :
               why.find("WIRE_LIMIT") != std::string::npos || why.find("NOT_LOCKABLE") != std::string::npos ||
               why.find("LOCK_LIMIT") != std::string::npos ? "WIRE_LIMIT" :
               why.find("STRUCTURAL_WARM") != std::string::npos ? "STRUCTURAL_WARM_TOO_SMALL" :
               why.find("COVERAGE") != std::string::npos ? "COVERAGE_TOO_LOW" :
               why.find("PRESSURE") != std::string::npos ? "HOST_MEMORY_PRESSURE_HIGH" : "CURRENT_RAM_LIMIT";
    };
    auto evaluate = [&](double R) {
        p.candidates.clear();
        p.reserve_mib = R;
        for (auto & [kv, pr] : full) {
            tosh_plan_candidate c;
            c.name = std::string("full_gpu+") + kv_name(kv); c.mode = "full_gpu"; c.kv = kv; c.ubatch = ub_full;
            c.private_mib = pr.model + pr.context + pr.compute;
            c.free_mib    = p.vram_free_mib - c.private_mib;
            c.kv_mib = pr.context; c.compute_mib = pr.compute;
            c.valid  = pr.ok && c.free_mib >= R;
            c.need_vram_mib = c.private_mib + R; c.need_host_mib = host_engine;
            c.reason = !pr.ok ? "probe failed" : c.valid ? "fits with the reserve" :
                       c.free_mib >= 0 ? "VRAM_LIMIT: loads, but leaves less than the reserve for the requested context" : "VRAM_LIMIT: does not fit";
            p.candidates.push_back(c);
        }
        for (size_t k = 0; k < kvs.size() && dmoe_built; ++k) {
            for (int u = 0; u < 3; ++u) {
                const probe & pr = dm[k][u];
                tosh_plan_candidate c;
                c.name = std::string("dmoe+") + kv_name(kvs[k]); c.mode = "dmoe"; c.kv = kvs[k]; c.ubatch = ubs[u];
                const double base = pr.model + pr.context + pr.compute;
                c.arena_mib   = std::max(0.0, p.vram_free_mib - base - slots - overhead - R);
                c.private_mib = base + slots + overhead + c.arena_mib;
                c.free_mib    = p.vram_free_mib - c.private_mib;
                c.kv_mib = pr.context; c.compute_mib = pr.compute;
                c.need_vram_mib = base + slots + overhead + p.min_arena_mib + R;
                const double cover = mi.bank_mib*BOUNDED_MIN_COVERAGE - c.arena_mib;
                c.need_host_mib = !runtime || full_host_ok ? host_dmoe :
                                  host_engine + staging_mib + std::max(mi.warm_floor_mib, disk_experts() ? std::min(cover, bounded_warm) : cover);
                // a bounded cache must still cover the bank together with the arena, unless the disk may serve the rest
                std::string host_reason = dmoe_block;
                if (!host_reason.empty() && bounded_warm > 0) {
                    const double cov = (c.arena_mib + bounded_warm)/mi.bank_mib;
                    host_reason = cov >= (disk_experts() ? BOUNDED_DISK_MIN_COVERAGE : BOUNDED_MIN_COVERAGE) ? "" :
                                  string_format("BOUNDED_HOST_COVERAGE_REJECTED: %.1f GiB RAM cache, coverage %.2f", bounded_warm/1024, cov);
                }
                c.valid  = pr.ok && host_reason.empty() && c.arena_mib >= p.min_arena_mib;
                c.reason = !pr.ok ? "probe failed" : !host_reason.empty() ? host_reason :
                           c.valid ? (dmoe_block.empty() ? "context and reserve kept" :
                                      string_format("context and reserve kept, %.1f GiB of the bank in RAM", bounded_warm/1024)) :
                           string_format("VRAM_LIMIT: arena %.0f MiB under the useful minimum %.0f", c.arena_mib, p.min_arena_mib);
                p.candidates.push_back(c);
            }
        }
        // legacy: the fewest layers of experts on the host that leave the reserve
        for (auto & [kv, pr] : full) {
            tosh_plan_candidate c;
            c.name = std::string("legacy+") + kv_name(kv); c.mode = "legacy_offload"; c.kv = kv; c.ubatch = ub_full;
            const double per_layer = mi.bank_mib/std::max(1, mi.n_moe);
            const double full_priv = pr.model + pr.context + pr.compute;
            for (int n = 1; n <= mi.n_layer && pr.ok; ++n) {
                const double priv = full_priv - n*per_layer + slots;
                if (p.vram_free_mib - priv >= R) {
                    c.ncmoe = n; c.private_mib = priv; c.free_mib = p.vram_free_mib - priv; c.valid = true;
                    break;
                }
            }
            c.kv_mib = pr.context; c.compute_mib = pr.compute;
            const double host_legacy = mi.bank_mib*c.ncmoe/std::max(1, mi.n_moe) + p.host_weights_mib + 1024.0;
            c.need_vram_mib = (c.valid ? c.private_mib : full_priv - mi.bank_mib + slots) + R;
            c.need_host_mib = c.valid ? host_legacy : mi.bank_mib + p.host_weights_mib + 1024.0;
            const bool host_ok = runtime ? p.host_pressure != "CRITICAL" && host_legacy <= p.effective_mib :
                                 hi.available_mib <= 0 || host_legacy + HOST_MARGIN_MIB <= hi.available_mib;
            const double host_allowed = runtime ? p.effective_mib : hi.available_mib - HOST_MARGIN_MIB;
            c.reason = !c.valid ? "VRAM_LIMIT: does not fit" : !host_ok ?
                string_format("%s: %d layers of experts on the host need %.1f GiB, %.1f GiB allowed now",
                              runtime && p.effective_mib == p.static_cap_mib ? "STATIC_RAM_LIMIT" : "CURRENT_RAM_LIMIT",
                              c.ncmoe, host_legacy/1024, host_allowed/1024) :
                string_format("%d layers of experts on the host", c.ncmoe);
            c.valid = c.valid && host_ok;
            p.candidates.push_back(c);
        }
    };

    auto find = [&](const std::string & mode, ggml_type kv, int ub) -> tosh_plan_candidate * {
        for (auto & c : p.candidates) if (c.mode == mode && c.kv == kv && (ub == 0 || c.ubatch == ub)) return &c;
        return nullptr;
    };

    // the ubatch for a DMoE candidate: larger ones speed prefill but take compute memory from the arena
    auto pick_ub = [&](ggml_type kv) -> tosh_plan_candidate * {
        tosh_plan_candidate * c512 = find("dmoe", kv, 512);
        if (c512 == nullptr || !c512->valid) return nullptr;
        const double a = c512->arena_mib;
        tosh_plan_candidate * c1k = find("dmoe", kv, 1024);
        tosh_plan_candidate * c2k = find("dmoe", kv, 2048);
        // what a larger ubatch takes from the arena is paid in decode misses: a tenth of the bank,
        // and never more than a tenth of the arena itself, which is the hottest share of it
        const double cost = std::min(0.10*mi.bank_mib, 0.10*a);
        if (high && c2k && c2k->valid && a - c2k->arena_mib <= cost) return c2k;
        if (p.n_ctx <= 32768 && c1k && c1k->valid && a - c1k->arena_mib <= cost) return c1k;
        return c512;
    };

    const tosh_plan_candidate * sel = nullptr;
    // VRAM kept free after every known allocation, for the desktop and other apps; the plan
    // already counts weights, KV, compute, prefill slots and the driver margin
    // an MTP head adds its own context and buffers after the plan; past the reserve the driver pages
    // GPU memory into wired RAM, which slows every token and can use up the host
    const auto & spec_types = params.speculative.types;
    const bool mtp = std::find(spec_types.begin(), spec_types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP) != spec_types.end();
    // a head that comes as its own file loads its weights after the plan as well
    double mtp_mib = 0;
    bool head_file = false;
    if (mtp) {
        std::error_code ec;
        const std::string & head = params.speculative.draft.mparams.path;
        const auto head_bytes = head.empty() ? 0 : std::filesystem::file_size(head, ec);
        head_file = !head.empty() && !ec;
        mtp_mib = env_mib("TOSH_AUTO_MTP_RESERVE_MIB", 768.0 + (head_file ? head_bytes/MiB : 0.0));
    }
    // the app's VRAM reserve setting replaces the default share of the card; the head adds to either
    const double reserve_base = env_mib("GGML_METAL_VRAM_RESERVE_MB", std::clamp(RESERVE_FRACTION*p.vram_total_mib, RESERVE_MIN_MIB, RESERVE_MAX_MIB));
    const double reserve_pref = env_mib("TOSH_AUTO_RESERVE_MIB", reserve_base + mtp_mib);
    std::vector<double> reserves = { reserve_pref, env_mib("TOSH_AUTO_MIN_RESERVE_MIB", std::max(512.0, reserve_pref - 256.0)) };
    // the head is optional: where it leaves no room, the plan is made again without it
    const size_t n_with_head = reserves.size();
    if (mtp_mib > 0) {
        reserves.push_back(reserve_pref - mtp_mib);
        reserves.push_back(std::max(512.0, reserve_pref - mtp_mib - 256.0));
    }
    p.mtp_head_dropped = false;
    p.mtp_head_arena   = false;
    // experts on the host lose more than the head gains: every reserve, the head's last, is
    // tried for the GPU modes before any of them settles for whole layers on the host.
    // Pass 0 tries the GPU modes, pass 1 the same with turbo4 where allowed, pass 2 adds the host.
    for (size_t pass = 0; pass < 3 && sel == nullptr; ++pass) {
    const bool turbo = pass == 1;
    if (turbo) {
        if (!turbo_fallback || force == "legacy") continue;
        if (!turbo_probed) {
            kvs.push_back(GGML_TYPE_TURBO4_0);
            probe_kv(kvs.size() - 1);
            turbo_probed = true;
        }
    }
    const ggml_type ka = turbo ? GGML_TYPE_TURBO4_0 : kvs[0];
    const ggml_type kb = !turbo && kvs.size() > 1 ? kvs[1] : GGML_TYPE_COUNT;
    for (size_t ri = 0; ri < reserves.size(); ++ri) {
        const double R = reserves[ri];
        evaluate(R);
        const tosh_plan_candidate * f16 = find("full_gpu", ka, 0);
        const tosh_plan_candidate * q8  = kb != GGML_TYPE_COUNT ? find("full_gpu", kb, 0) : nullptr;
        if (force.empty() || force == "full") {
            if (f16 && f16->valid) { sel = f16; p.reason = "the model and the requested context fit with the reserve"; }
            else if (q8 && q8->valid) { sel = q8; p.reason = "q8_0 cache makes the full model fit with the reserve"; }
        }
        auto pick_dmoe = [&]() -> const tosh_plan_candidate * {
            tosh_plan_candidate * a = pick_ub(ka);
            tosh_plan_candidate * b = kb != GGML_TYPE_COUNT ? pick_ub(kb) : nullptr;
            // while the bank does not fit, every MiB the q8_0 cache saves holds experts the CPU would compute
            if (a && b && a->arena_mib < mi.bank_mib && b->arena_mib - a->arena_mib >= std::max(512.0, 0.08*a->arena_mib)) {
                p.reason = "q8_0 cache gives the expert arena a material share of the bank";
                return b;
            }
            if (a) {
                p.reason = "the requested context and the reserve fit; experts take what is left";
                return a;
            }
            if (b) {
                p.reason = "only a q8_0 cache leaves a useful expert arena";
            }
            return b;
        };
        bool head_off = ri >= n_with_head;
        if (sel == nullptr && !no_dmoe && (force.empty() || force == "dmoe")) {
            sel = pick_dmoe();
            // with mixed execution the head's VRAM comes out of the expert arena, and past this share
            // the experts it pushes to the host cost more than its drafts save
            static const double max_share = env_mib("TOSH_AUTO_MTP_ARENA_SHARE", 0.20);
            if (sel && !head_off && mtp_mib > 0 && max_share > 0) {
                // the probes above already hold the head's context, so the arena without it comes from
                // a probe of the model as it loads without the head
                common_params params_bare = params;
                params_bare.speculative.types = { COMMON_SPECULATIVE_TYPE_NONE };
                params_bare.speculative.draft.mparams = common_params_model();
                const probe pb = run_probe(params_bare, p.n_ctx, sel->kv, sel->ubatch, true, 0);
                const double arena_bare = pb.ok ? std::max(0.0, p.vram_free_mib - (pb.model + pb.context + pb.compute) -
                                                                 slots - overhead - reserves[n_with_head]) : 0.0;
                if (arena_bare > 0 && sel->arena_mib < (1.0 - max_share)*arena_bare) {
                    evaluate(reserves[n_with_head]);
                    sel = pick_dmoe();
                    head_off = sel != nullptr;
                    p.mtp_head_arena = head_off;
                }
            }
        }
        if (sel == nullptr && (pass == 2 || force == "legacy") && (force.empty() || force == "legacy")) {
            const tosh_plan_candidate * l16 = find("legacy_offload", kvs[0], 0);
            const tosh_plan_candidate * l8  = kvs.size() > 1 ? find("legacy_offload", kvs[1], 0) : nullptr;
            if (l16 && l16->valid) sel = l16; else if (l8 && l8->valid) sel = l8;
            if (sel) p.reason = "mixed execution is not available here; expert layers go to the host";
        }
        if (sel) {
            if (turbo) p.reason = "only a turbo4 cache keeps the experts on the GPU: faster than on the host, at some quality cost";
            p.mtp_head_dropped = head_off;
            const double R_used = p.mtp_head_arena ? reserves[n_with_head] : R;
            if (R_used < reserves[p.mtp_head_dropped ? n_with_head : 0]) p.fallback = string_format("reserve lowered to %.1f GiB", R_used/1024);
            break;
        }
    }
    }

    if (sel == nullptr) {
        p.state  = TOSH_PLAN_UNSUPPORTED;
        p.mode   = "none";
        p.code   = "PLAN_UNSUPPORTED";
        p.reason = "no configuration fits the requested context with the minimum reserve";
    } else {
        p.mode = sel->mode; p.kv = sel->kv; p.ubatch = sel->ubatch; p.ncmoe = sel->ncmoe;
        p.arena_mib = sel->arena_mib; p.private_mib = sel->private_mib; p.free_mib = sel->free_mib;
        p.code = sel->mode == "full_gpu" ? "PLAN_FULL_GPU" : sel->mode == "legacy_offload" ? "PLAN_CLASSIC_NCMOE" :
                 runtime && !full_host_ok ? "PLAN_BOUNDED_DMOE" : "PLAN_FULL_HOST_DMOE";
        if (sel->mode == "full_gpu") {
            p.state = TOSH_PLAN_FULL_GPU_OPTIMAL;
            p.host_required_mib = host_engine;
            p.mlock = "not_required";
        } else if (sel->mode == "dmoe" && runtime && !full_host_ok) {
            p.state = TOSH_PLAN_DMOE_BOUNDED_HOST;
            p.mode  = "dmoe_bounded";
            p.warm_mib = !lean ? bounded_warm :
                         std::min(bounded_warm, std::max(mi.warm_floor_mib, mi.bank_mib*LEAN_COVERAGE - sel->arena_mib));
            p.coverage = (sel->arena_mib + p.warm_mib)/mi.bank_mib;
            p.bounded_code = p.coverage >= BOUNDED_GOOD_COVERAGE ? "BOUNDED_HOST_COVERAGE_GOOD" : "BOUNDED_HOST_COVERAGE_CONSTRAINED";
            p.host_required_mib = host_engine + p.warm_mib + staging_mib;
            p.fallback = dmoe_block;
        } else if (sel->mode == "dmoe") {
            p.state = weights_fit ? TOSH_PLAN_DMOE_CONTEXT_OPTIMAL : TOSH_PLAN_DMOE_CAPACITY_REQUIRED;
            p.host_required_mib = host_dmoe;
            if (runtime) p.coverage = (sel->arena_mib + mi.bank_mib)/mi.bank_mib;
        } else {
            p.state = TOSH_PLAN_LEGACY_OFFLOAD_BETTER;
            p.host_required_mib = mi.bank_mib*sel->ncmoe/std::max(1, mi.n_moe) + host_engine;
            p.mlock = "not_required";
            if (!dmoe_block.empty()) p.fallback = dmoe_block;
        }
    }
    if (runtime) {
        if (sel == nullptr || sel->mode != "dmoe" || full_host_ok) {
            if (p.bounded_code == "BOUNDED_HOST_SELECTED") p.bounded_code = "BOUNDED_HOST_COVERAGE_REJECTED";
        }
        p.projected_rss_mib  = hi.self_rss_mib + p.host_required_mib;
        p.projected_left_mib = hi.reclaimable_mib - p.host_required_mib;
    }
    // what ruled out each mode preferred over the selected one
    const auto tier = [](const std::string & m) { return m == "full_gpu" ? 0 : m == "dmoe" ? 1 : 2; };
    const int sel_tier = sel ? tier(sel->mode) : 3;
    for (const auto & c : p.candidates) {
        if (c.valid || tier(c.mode) >= sel_tier || c.reason == "probe failed") continue;
        const char * code = c.reason.rfind("VRAM_LIMIT", 0) == 0 ? "VRAM_LIMIT" : host_limit(c.reason);
        if (std::find(p.limits.begin(), p.limits.end(), code) == p.limits.end()) p.limits.push_back(code);
    }
    if (sel == nullptr) p.limits.push_back("NO_SAFE_FALLBACK");

    // product view: the mode, why it and not the preferred one, and what to watch
    auto has_limit = [&](const char * l) { return std::find(p.limits.begin(), p.limits.end(), l) != p.limits.end(); };
    if (sel == nullptr) {
        p.plan_mode = TOSH_PLAN_MODE_UNSUPPORTED;
        p.primary_reason = "NO_SAFE_FALLBACK";
        bool vram_ok = false;
        p.required_vram_mib = p.required_host_mib = 0;
        for (const auto & c : p.candidates) {
            if (c.need_vram_mib > 0 && (p.required_vram_mib == 0 || c.need_vram_mib < p.required_vram_mib)) p.required_vram_mib = c.need_vram_mib;
            if (c.need_vram_mib > 0 && c.need_vram_mib <= p.vram_free_mib) {
                vram_ok = true;
                if (p.required_host_mib == 0 || c.need_host_mib < p.required_host_mib) p.required_host_mib = c.need_host_mib;
            }
        }
        p.available_vram_mib = p.vram_free_mib;
        p.available_host_mib = runtime ? std::min(p.effective_mib, p.lockable_mib + host_engine) : std::max(0.0, hi.available_mib - HOST_MARGIN_MIB);
        p.limiting_resource  = !vram_ok ? "VRAM" : has_limit("WIRE_LIMIT") && !has_limit("CURRENT_RAM_LIMIT") ? "WIRE" : "HOST_RAM";
    } else if (sel->mode == "full_gpu") {
        p.plan_mode = TOSH_PLAN_MODE_FULL_GPU;
        p.primary_reason = "FULL_GPU_FITS";
        p.limiting_resource = "NONE";
    } else if (sel->mode == "legacy_offload") {
        p.plan_mode = TOSH_PLAN_MODE_CLASSIC_NCMOE;
        p.primary_reason = "CLASSIC_NCMOE_FALLBACK";
        p.limiting_resource = has_limit("WIRE_LIMIT") && !has_limit("CURRENT_RAM_LIMIT") ? "WIRE" : dmoe_built ? "HOST_RAM" : "VRAM";
    } else if (runtime && !full_host_ok) {
        p.plan_mode = TOSH_PLAN_MODE_BOUNDED_DMOE;
        p.primary_reason = p.full_host_code == "FULL_HOST_NOT_WANTED"   ? "HOST_RAM_SAVING" :
                           p.full_host_code == "FULL_HOST_STATIC_LIMIT" ? "STATIC_RAM_LIMIT" :
                           p.full_host_code == "FULL_HOST_WIRE_LIMIT"   ? "WIRE_LIMIT" :
                           p.full_host_code == "HOST_MEMORY_PRESSURE_HIGH" ? "HOST_MEMORY_PRESSURE_HIGH" : "CURRENT_RAM_LIMIT";
        p.limiting_resource = p.primary_reason == "WIRE_LIMIT" ? "WIRE" : "HOST_RAM";
    } else {
        p.plan_mode = TOSH_PLAN_MODE_FULL_HOST_DMOE;
        p.primary_reason = "FULL_HOST_SAFE";
        p.limiting_resource = "VRAM";
        p.coverage = (sel->arena_mib + mi.bank_mib)/mi.bank_mib;
    }
    p.code = tosh_plan_mode_name(p.plan_mode);
    if (sel) p.kv_mib = sel->kv_mib;
    if (p.plan_mode == TOSH_PLAN_MODE_FULL_HOST_DMOE || p.plan_mode == TOSH_PLAN_MODE_BOUNDED_DMOE) {
        p.coverage_state = p.coverage >= BOUNDED_GOOD_COVERAGE ? "GOOD" : p.coverage >= BOUNDED_MIN_COVERAGE ? "CONSTRAINED" :
                           disk_experts() ? "DISK" : "REJECT";
        if (p.coverage_state == "CONSTRAINED") p.warnings.push_back("COVERAGE_CONSTRAINED");
        if (p.coverage_state == "DISK") p.warnings.push_back("EXPERTS_FROM_DISK");
    }
    if (sel && sel->mode != "full_gpu" && p.projected_rss_mib == 0) p.projected_rss_mib = hi.self_rss_mib + p.host_required_mib;
    // the server keeps context checkpoints and a prompt cache in RAM; they take what the plan leaves,
    // checkpoints first since they spare re-reading a conversation, and the model before both
    double left = runtime ? p.effective_mib - p.host_required_mib : hi.available_mib - HOST_MARGIN_MIB - p.host_required_mib;
    if (sel && (mi.swa_elems > 0 || mi.recr_elems > 0) && params.n_ctx_checkpoints > 0) {
        const double swa_mib = mi.swa_elems*(mi.n_swa + sel->ubatch)*ggml_type_size(sel->kv)/ggml_blck_size(sel->kv)/MiB;
        p.checkpoint_mib = rs_checkpoint_mib(left, swa_mib, mi.recr_elems, params.n_ctx_checkpoints);
        p.n_ctx_checkpoints = std::min(params.n_ctx_checkpoints, (int) std::max(0.0, std::floor(left/p.checkpoint_mib)));
        // the next turn restores one of the last checkpoints, and with none a hybrid model re-reads the whole
        // conversation: a few come out of what the load leaves to the system
        const int n_min = std::min(params.n_ctx_checkpoints, RS_MIN_CHECKPOINTS);
        if (p.n_ctx_checkpoints < n_min) {
            const double extra = (n_min - p.n_ctx_checkpoints)*p.checkpoint_mib;
            if (p.projected_left_mib - extra >= HOST_MARGIN_MIB) {
                p.n_ctx_checkpoints = n_min;
                p.projected_left_mib -= extra;
            }
        }
        if (p.n_ctx_checkpoints < params.n_ctx_checkpoints) p.warnings.push_back("CONTEXT_CHECKPOINTS_REDUCED");
        left -= p.n_ctx_checkpoints*p.checkpoint_mib;
        p.projected_rss_mib += p.n_ctx_checkpoints*p.checkpoint_mib;
    } else {
        p.n_ctx_checkpoints = params.n_ctx_checkpoints;
    }
    if (sel && params.cache_ram_mib != 0) {
        const double want = params.cache_ram_mib > 0 ? params.cache_ram_mib : std::max(0.0, left);
        p.prompt_cache_mib = std::clamp(std::floor(left), 0.0, want);
        if (p.prompt_cache_mib < want) p.warnings.push_back("PROMPT_CACHE_REDUCED");
        p.projected_rss_mib += p.prompt_cache_mib;
    }
    if (p.fallback.rfind("reserve lowered", 0) == 0) p.warnings.push_back("RESERVE_LOWERED");
    if (sel && turbo_fallback && sel->kv == GGML_TYPE_TURBO4_0) p.warnings.push_back("KV_TURBO4_AUTO");
    if (runtime && p.host_pressure == "ELEVATED") p.warnings.push_back("HOST_MEMORY_PRESSURE_ELEVATED");
    if (runtime && hi.swap_growing) p.warnings.push_back("SWAP_GROWING");
    if (hi.simulated) p.warnings.push_back("SIMULATED_HOST");
    return sel;
    };

    const tosh_plan_candidate * sel = decide(hi0);
    // the probes take seconds: if memory went elsewhere meanwhile, the plan is made again, once
    if (runtime && sel != nullptr && sel->mode != "full_gpu" && !hi0.simulated) {
        const host_mem hi1 = read_host_info();
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t_snapshot).count();
        const bool changed = hi0.reclaimable_mib - hi1.reclaimable_mib > STALE_DROP_MIB ||
                             hi1.pressure_level > hi0.pressure_level || (hi1.swap_growing && !hi0.swap_growing);
        if (changed) {
            LOG_WRN("%s: memory changed while planning (reclaimable %.1f -> %.1f GiB), planning again\n",
                    __func__, hi0.reclaimable_mib/1024, hi1.reclaimable_mib/1024);
            sel = decide(hi1);
            p.warnings.push_back("REPLANNED_AFTER_MEMORY_CHANGE");
        }
        p.recheck_ms = ms;
    }
    if (p.mtp_head_dropped) p.warnings.push_back("MTP_HEAD_DROPPED");
    plan_emit(p);
    if (out) *out = p;
    if (getenv("TOSH_AUTO_DRY_RUN") != nullptr || sel == nullptr) {
        return false;
    }

    // apply: the plan owns memory from here
    if (p.mtp_head_dropped) {
        LOG_WRN("%s: %s, starting without it\n", __func__, p.mtp_head_arena ?
                "the MTP head would take too much of the expert arena" : "the MTP head does not fit beside the model");
        auto & types = params.speculative.types;
        types.erase(std::remove(types.begin(), types.end(), COMMON_SPECULATIVE_TYPE_DRAFT_MTP), types.end());
        if (types.empty()) types = { COMMON_SPECULATIVE_TYPE_NONE };
        params.speculative.draft.mparams = common_params_model();
    }
    params.fit_params = false;
    params.n_gpu_layers = 999;
    params.cache_type_k = p.kv;
    params.cache_type_v = p.kv;
    params.n_ctx = p.n_ctx;
    params.n_ubatch = p.ubatch;
    params.n_batch  = std::max(params.n_batch, p.ubatch);
    if (params.cache_ram_mib != 0) params.cache_ram_mib = (int32_t) p.prompt_cache_mib;
    params.n_ctx_checkpoints = p.n_ctx_checkpoints;
    if (p.mode != "full_gpu") {
        params.no_extra_bufts = true;
    }
    if (p.mode == "dmoe" || p.mode == "dmoe_bounded") {
        add_override(params, experts_pattern());
        params.load_mode = LLAMA_LOAD_MODE_MLOCK;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        if (p.mode == "dmoe_bounded") {
            // the bank stays in the file; the locked RAM cache is the host cache's own allocation
            params.load_mode = LLAMA_LOAD_MODE_NONE;
            setenv("TOSH_DMOE_HOST_CACHE_MIB", string_format("%.0f", p.warm_mib).c_str(), 1);
            setenv("GGML_OP_OFFLOAD_MIN_BATCH", std::to_string(TOSH_DMOE_MAX_TOKENS + 1).c_str(), 1);
            // fill the cache from the file while it is locked, until the server is ready
            setenv("TOSH_DMOE_HOST_WARMUP_FILL_MIB", "max", 0);
        }
#endif
        setenv("TOSH_DMOE_CACHE_MIB", "auto", 1);
        setenv("TOSH_DMOE_RESERVE_MIB", string_format("%.0f", p.reserve_mib).c_str(), 1);
        setenv("TOSH_DMOE_MIN_ARENA_MIB", string_format("%.0f", p.min_arena_mib).c_str(), 1);
        setenv("TOSH_DMOE_RARE_ROWS", "auto", 0);
        setenv("TOSH_DMOE_RARE_TAIL", "0", 0);
        setenv("GGML_SCHED_PREFETCH_EXPERTS", "1", 0);
        setenv("GGML_CPU_NO_REPACK", "1", 0);
        // the host executor computes the missed experts on every core; the backend's own threads find little
        // CPU work here (verify windows and prompts included) and only compete with it for the cores
        const int cpu_threads = (int) env_mib("TOSH_DMOE_LLAMA_THREADS", 2);
        if (cpu_threads > 0) {
            for (auto * cp : { &params.cpuparams, &params.cpuparams_batch }) {
                if (cp->n_threads <= 0 || cp->n_threads > cpu_threads) {
                    cp->n_threads = cpu_threads;
                }
            }
        }
#ifdef TOSH_ENABLE_DYNAMIC_MOE
        tosh_dmoe_enable();
#endif
    } else if (p.mode == "legacy_offload") {
        // the host share is sized as anonymous memory, as with the expert bank
        if (params.load_mode == LLAMA_LOAD_MODE_AUTO) params.load_mode = LLAMA_LOAD_MODE_NONE;
        for (int i = 0; i < p.ncmoe; ++i) {
            pattern_store().push_back(llm_ffn_block_regex(i, LLM_FFN_EXPS_REGEX));
            add_override(params, pattern_store().back().c_str());
        }
        setenv("GGML_SCHED_PREFETCH_EXPERTS", "1", 0);
        setenv("GGML_CPU_NO_REPACK", "1", 0);
    }
    return true;
}

void tosh_plan_report_load(const char * result, const char * detail) {
    const char * f = getenv("TOSH_AUTO_PLAN_FILE");
    if (!g_planned || f == nullptr) return;
    const tosh_plan & p = g_last_plan;
    double rss_mib = 0, footprint_mib = 0, vram_free = 0, vram_total = 0;
    GGML_UNUSED(vram_total);
#ifdef __APPLE__
    mach_task_basic_info_data_t ti; mach_msg_type_number_t tn = MACH_TASK_BASIC_INFO_COUNT;
    if (task_info(mach_task_self(), MACH_TASK_BASIC_INFO, (task_info_t) &ti, &tn) == KERN_SUCCESS) rss_mib = ti.resident_size/MiB;
    // resident also counts clean file pages the system can drop; the footprint is what the plan commits
    task_vm_info_data_t vi; mach_msg_type_number_t vn = TASK_VM_INFO_COUNT;
    if (task_info(mach_task_self(), TASK_VM_INFO, (task_info_t) &vi, &vn) == KERN_SUCCESS) footprint_mib = vi.phys_footprint/MiB;
#endif
    // this process's private Metal memory; else the card's live free memory against the plan's reading
    double vram_mib = -1;
#ifdef TOSH_ENABLE_DYNAMIC_MOE
    if (const size_t b = tosh_dmoe_private_bytes()) vram_mib = b/MiB;
    size_t vf = 0, vt = 0;
    if (vram_mib < 0 && tosh_vram_read(&vf, &vt)) { vram_free = vf/MiB; vram_total = vt/MiB; vram_mib = std::max(0.0, p.vram_free_mib - vram_free); }
#endif
    // what mixed execution ended up doing, e.g. DMOE_HOST_BANK_NOT_LOCKABLE when it had to stay off
    std::string runtime_code;
    if (const char * sf = getenv("TOSH_DMOE_STATUS_FILE")) {
        if (FILE * rf = fopen(sf, "r")) {
            char buf[1024] = {};
            const size_t n = fread(buf, 1, sizeof(buf) - 1, rf);
            fclose(rf);
            const std::string st(buf, n);
            const size_t k = st.find("\"code\": \"");
            if (k != std::string::npos) runtime_code = st.substr(k + 9, st.find('"', k + 9) - (k + 9));
        }
    }
    if (FILE * fp = fopen((std::string(f) + ".actual").c_str(), "w")) {
        fprintf(fp, "{\"plan_schema_version\": %d, \"load_result\": %s, \"detail\": %s, \"runtime_code\": %s, \"mode\": %s, "
                    "\"projected_rss_bytes\": %lld, \"actual_rss_bytes\": %lld, \"actual_footprint_bytes\": %lld, \"projected_vram_bytes\": %lld, \"actual_vram_bytes\": %lld}\n",
                TOSH_PLAN_SCHEMA_VERSION, jstr(result).c_str(), jstr(detail ? detail : "").c_str(), jstr(runtime_code).c_str(), jstr(p.code).c_str(),
                bytes(p.projected_rss_mib), bytes(rss_mib), bytes(footprint_mib), bytes(p.private_mib), vram_mib >= 0 ? bytes(vram_mib) : -1LL);
        fclose(fp);
    }
}
