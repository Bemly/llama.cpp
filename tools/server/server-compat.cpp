#include "server-compat.h"

#include "server-common.h"
#include "common.h"
#include "log.h"

#include <algorithm>
#include <cctype>
#include <cerrno>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <fstream>
#include <functional>
#include <mutex>
#include <random>
#include <sstream>
#include <sys/stat.h>
#include <unistd.h>

#if defined(__APPLE__)
#    include <mach/mach.h>
#    include <sys/mount.h>
#    include <sys/sysctl.h>
#elif defined(__linux__)
#    include <sys/statvfs.h>
#    include <sys/sysinfo.h>
#endif

namespace {

int64_t now_ms() {
    return std::chrono::duration_cast<std::chrono::milliseconds>(
                   std::chrono::system_clock::now().time_since_epoch())
            .count();
}

std::string random_token() {
    static thread_local std::mt19937_64 rng{std::random_device{}()};
    std::uniform_int_distribution<uint64_t> dist;
    char buf[33];
    std::snprintf(buf, sizeof(buf), "%016llx%016llx",
                  (unsigned long long) dist(rng), (unsigned long long) dist(rng));
    return std::string(buf);
}

std::string header_get(const server_http_req & req, const std::string & key) {
    for (const auto & [k, v] : req.headers) {
        if (k.size() == key.size()) {
            bool eq = true;
            for (size_t i = 0; i < k.size(); ++i) {
                if (std::tolower((unsigned char) k[i]) != std::tolower((unsigned char) key[i])) {
                    eq = false;
                    break;
                }
            }
            if (eq) {
                return v;
            }
        }
    }
    return {};
}

server_http_res_ptr json_res(int status, const json & body) {
    auto res = std::make_unique<server_http_res>();
    res->status = status;
    res->content_type = "application/json; charset=utf-8";
    res->headers["Cache-Control"] = "no-store";
    res->data = safe_json_to_str(body);
    return res;
}

server_http_res_ptr stub_unsupported(const std::string & what) {
    return json_res(501, {
            {"ok", false},
            {"error", "unsupported_on_llama_server"},
            {"reason", what},
    });
}

bool ensure_dir(const std::string & path) {
    if (path.empty()) {
        return false;
    }
#if defined(_WIN32)
    return _mkdir(path.c_str()) == 0 || errno == EEXIST;
#else
    return mkdir(path.c_str(), 0755) == 0 || errno == EEXIST;
#endif
}

json host_hardware_sample() {
    json out = {
            {"schema_version", 1},
            {"scope", "system"},
            {"sampled_at_ms", now_ms()},
            {"status", "ok"},
            {"cpu_percent", nullptr},
            {"ram", json::object()},
            {"disk", json::object()},
    };

#if defined(__APPLE__)
    {
        host_cpu_load_info_data_t load;
        mach_msg_type_number_t count = HOST_CPU_LOAD_INFO_COUNT;
        if (host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, (host_info_t) &load, &count) == KERN_SUCCESS) {
            static host_cpu_load_info_data_t prev{};
            static bool have_prev = false;
            if (have_prev) {
                uint64_t user = load.cpu_ticks[CPU_STATE_USER] - prev.cpu_ticks[CPU_STATE_USER];
                uint64_t sys  = load.cpu_ticks[CPU_STATE_SYSTEM] - prev.cpu_ticks[CPU_STATE_SYSTEM];
                uint64_t idle = load.cpu_ticks[CPU_STATE_IDLE] - prev.cpu_ticks[CPU_STATE_IDLE];
                uint64_t nice = load.cpu_ticks[CPU_STATE_NICE] - prev.cpu_ticks[CPU_STATE_NICE];
                uint64_t total = user + sys + idle + nice;
                if (total > 0) {
                    out["cpu_percent"] = 100.0 * double(user + sys + nice) / double(total);
                }
            }
            prev = load;
            have_prev = true;
        }
    }
    {
        int64_t memsize = 0;
        size_t len = sizeof(memsize);
        sysctlbyname("hw.memsize", &memsize, &len, nullptr, 0);
        mach_msg_type_number_t count = HOST_VM_INFO64_COUNT;
        vm_statistics64_data_t vm;
        if (host_statistics64(mach_host_self(), HOST_VM_INFO64, (host_info64_t) &vm, &count) == KERN_SUCCESS && memsize > 0) {
            const int64_t page = sysconf(_SC_PAGESIZE);
            const int64_t used = page * int64_t(vm.active_count + vm.wire_count + vm.compressor_page_count);
            out["ram"] = {
                    {"total_bytes", memsize},
                    {"used_bytes", used},
                    {"available_bytes", memsize - used},
            };
        }
    }
    {
        struct statfs sfs {};
        if (statfs("/", &sfs) == 0) {
            const uint64_t total = uint64_t(sfs.f_blocks) * uint64_t(sfs.f_bsize);
            const uint64_t freeb = uint64_t(sfs.f_bavail) * uint64_t(sfs.f_bsize);
            out["disk"] = {
                    {"read_bytes_per_s", nullptr},
                    {"write_bytes_per_s", nullptr},
                    {"total_bytes", total},
                    {"available_bytes", freeb},
            };
        }
    }
#elif defined(__linux__)
    {
        std::ifstream f("/proc/stat");
        std::string cpu;
        uint64_t user = 0, nice = 0, system = 0, idle = 0;
        if (f >> cpu >> user >> nice >> system >> idle) {
            static uint64_t pu = 0, pn = 0, ps = 0, pi = 0;
            static bool have = false;
            if (have) {
                const uint64_t du = user - pu, dn = nice - pn, ds = system - ps, di = idle - pi;
                const uint64_t tot = du + dn + ds + di;
                if (tot > 0) {
                    out["cpu_percent"] = 100.0 * double(du + dn + ds) / double(tot);
                }
            }
            pu = user; pn = nice; ps = system; pi = idle;
            have = true;
        }
    }
    {
        struct sysinfo si {};
        if (sysinfo(&si) == 0) {
            const uint64_t total = uint64_t(si.totalram) * si.mem_unit;
            const uint64_t freeb = uint64_t(si.freeram) * si.mem_unit;
            out["ram"] = {
                    {"total_bytes", total},
                    {"used_bytes", total - freeb},
                    {"available_bytes", freeb},
            };
        }
    }
    {
        struct statvfs sfs {};
        if (statvfs("/", &sfs) == 0) {
            const uint64_t total = uint64_t(sfs.f_blocks) * sfs.f_frsize;
            const uint64_t freeb = uint64_t(sfs.f_bavail) * sfs.f_frsize;
            out["disk"] = {
                    {"read_bytes_per_s", nullptr},
                    {"write_bytes_per_s", nullptr},
                    {"total_bytes", total},
                    {"available_bytes", freeb},
            };
        }
    }
#else
    out["status"] = "unavailable";
#endif
    return out;
}

struct wrapping_res : server_http_res {
    server_http_res_ptr inner;
    std::function<void()> on_done;
    bool finished = false;

    wrapping_res(server_http_res_ptr i, std::function<void()> done)
        : inner(std::move(i)), on_done(std::move(done)) {
        content_type = inner->content_type;
        status = inner->status;
        data = std::move(inner->data);
        headers = inner->headers;
        if (inner->is_stream()) {
            next = [this](std::string & out) {
                const bool more = inner->next(out);
                if (!more) {
                    finish();
                }
                return more;
            };
        } else {
            finish();
        }
    }

    void finish() {
        if (finished) {
            return;
        }
        finished = true;
        if (on_done) {
            on_done();
        }
    }

    void on_complete() override {
        finish();
        inner->on_complete();
    }

    ~wrapping_res() override {
        finish();
    }
};

} // namespace

struct server_compat::impl {
    const common_params & params;

    std::mutex mutex;
    std::string instance_id;
    std::string console_token;
    int64_t started_at_ms = 0;
    std::string data_dir;
    bool engine_stopped = false;

    struct req_row {
        std::string id;
        std::string instance;
        std::string endpoint = "/v1/chat/completions";
        std::string status = "generating"; // generating | completed | cancelled | error
        std::string phase;
        int64_t started_at_ms = 0;
        int64_t finished_at_ms = 0;
        int prompt_tokens = 0;
        int cached_tokens = 0;
        int generated_tokens = 0;
        int limit = 0;
        double decode_tps = 0;
        double prefill_tps = 0;
        int64_t elapsed_ms = 0;
        bool has_media = false;
    };

    std::deque<req_row> active;
    std::deque<req_row> recent; // newest at front
    static constexpr size_t k_recent_max = 100;
    static constexpr size_t k_active_max = 512;

    std::string log_text;
    uint64_t log_revision = 1;
    std::string log_session;
    static constexpr size_t k_log_max = 256 * 1024;

    json settings = json::object();
    json config = json::object();
    json prompts = json::object({{"system_prompt", ""}});

    json assist_log = json::array();

    server_http_context::handler_t prometheus_metrics;

    explicit impl(const common_params & p) : params(p) {
        instance_id = "llama-" + random_token().substr(0, 12);
        console_token = params.api_keys.empty() ? random_token() : params.api_keys.front();
        started_at_ms = now_ms();
        log_session = instance_id;

        if (const char * env = std::getenv("TOSH_COMPAT_DIR")) {
            data_dir = env;
        } else {
            data_dir = ".tosh-compat";
        }
        ensure_dir(data_dir);
        ensure_dir(data_dir + "/_old_chats");

        load_json_file(data_dir + "/settings.json", settings);
        load_json_file(data_dir + "/config.json", config);

        const char * prompts_path = std::getenv("TOSH_PROMPTS_FILE");
        if (prompts_path && *prompts_path) {
            load_json_file(prompts_path, prompts);
        } else {
            load_json_file(data_dir + "/prompt.json", prompts);
        }

        append_log_unlocked("compat: ready instance=" + instance_id);
    }

    void load_json_file(const std::string & path, json & dest) {
        std::ifstream in(path);
        if (!in) {
            return;
        }
        try {
            std::ostringstream ss;
            ss << in.rdbuf();
            json j = json::parse(ss.str());
            if (j.is_object()) {
                dest = std::move(j);
            }
        } catch (...) {
        }
    }

    static void merge_object(json & dest, const json & patch) {
        if (!patch.is_object()) {
            return;
        }
        if (!dest.is_object()) {
            dest = json::object();
        }
        for (const auto & [key, val] : patch.items()) {
            dest[key] = val;
        }
    }

    void save_json_file(const std::string & path, const json & j) {
        std::ofstream out(path);
        if (!out) {
            return;
        }
        out << j.dump(2);
    }

    void append_log_unlocked(const std::string & line) {
        const std::string stamped = std::to_string(now_ms()) + " " + line + "\n";
        log_text += stamped;
        if (log_text.size() > k_log_max) {
            log_text.erase(0, log_text.size() - k_log_max);
        }
        ++log_revision;
    }

    void append_log(const std::string & line) {
        std::lock_guard<std::mutex> lock(mutex);
        append_log_unlocked(line);
    }

    bool token_ok(const server_http_req & req) const {
        if (params.api_keys.empty()) {
            // still accept our console token or missing key for local use
            const std::string t = header_get(req, "X-KVMem-Token");
            return t.empty() || t == console_token;
        }
        const std::string t = header_get(req, "X-KVMem-Token");
        if (t == console_token) {
            return true;
        }
        for (const auto & k : params.api_keys) {
            if (t == k) {
                return true;
            }
        }
        const std::string auth = header_get(req, "Authorization");
        for (const auto & k : params.api_keys) {
            if (auth == "Bearer " + k) {
                return true;
            }
        }
        return false;
    }

    json enrich_health(json base) {
        if (!base.is_object()) {
            base = json::object();
        }
        const bool ok = json_value(base, "status", std::string("ok")) == "ok" && !engine_stopped;
        base["status"] = ok ? "ok" : "offline";
        base["loaded"] = ok;
        base["engine"] = {
                {"state", ok ? "ready" : (engine_stopped ? "stopped" : "offline")},
                {"code", engine_stopped ? "stopped" : json(nullptr)},
                {"retry", {{"state", "idle"}}},
        };
        return base;
    }

    json enrich_props(json base) {
        if (!base.is_object()) {
            base = json::object();
        }
        base["role"] = "model";
        if (!base.contains("total_slots")) {
            base["total_slots"] = params.n_parallel > 0 ? params.n_parallel : 1;
        }

        json kvmem = json::object();
        int gen_limit = params.n_predict;
        if (gen_limit < 0) {
            gen_limit = 8192;
        }
        kvmem["generation_limit"] = gen_limit;
        kvmem["defaults"] = {
                {"enable_thinking", true},
                {"reasoning_budget_tokens", nullptr},
        };
        base["kvmem"] = kvmem;

        json devices = json::array();
        devices.push_back({
                {"name", "Metal"},
                {"uuid", instance_id},
                {"pci", nullptr},
        });
        base["execution"] = {
                {"monitor_devices", devices},
                {"backend", "llamacpp"},
        };
        return base;
    }

    json row_to_json(const req_row & r) const {
        return {
                {"id", r.id},
                {"instance", r.instance},
                {"endpoint", r.endpoint},
                {"status", r.status},
                {"phase", r.phase},
                {"started_at_ms", r.started_at_ms},
                {"finished_at_ms", r.finished_at_ms ? json(r.finished_at_ms) : json(nullptr)},
                {"prompt_tokens", r.prompt_tokens},
                {"cached_tokens", r.cached_tokens},
                {"generated_tokens", r.generated_tokens},
                {"limit", r.limit},
                {"decode_tps", r.decode_tps},
                {"prefill_tps", r.prefill_tps},
                {"elapsed_ms", r.elapsed_ms},
                {"has_media", r.has_media},
        };
    }

    json build_snapshot() {
        std::lock_guard<std::mutex> lock(mutex);
        json active_j = json::array();
        json recent_j = json::array();
        for (const auto & r : active) {
            active_j.push_back(row_to_json(r));
        }
        for (const auto & r : recent) {
            recent_j.push_back(row_to_json(r));
        }

        json gpus = json::array();
        gpus.push_back({
                {"name", "Metal"},
                {"uuid", instance_id},
                {"util", nullptr},
                {"vram_used_bytes", nullptr},
                {"vram_total_bytes", nullptr},
                {"temp_c", nullptr},
                {"power_w", nullptr},
        });

        return {
                {"schema_version", 1},
                {"backend", "llamacpp"},
                {"server_instance", instance_id},
                {"sampled_at_ms", now_ms()},
                {"active", active_j},
                {"recent", recent_j},
                {"hardware", {
                        {"sampled_at_ms", now_ms()},
                        {"gpus", gpus},
                }},
                {"kvmem", {
                        {"host_payload_bytes", nullptr},
                        {"ssd_payload_bytes", nullptr},
                }},
        };
    }

    json build_metrics(bool all_requests) {
        std::lock_guard<std::mutex> lock(mutex);
        json live = {
                {"state", active.empty() ? (engine_stopped ? "unloaded" : "idle") : "generating"},
                {"queued", 0},
                {"phase", active.empty() ? json(nullptr) : json("answering")},
                {"prompt_tokens", json(nullptr)},
                {"prompt_read", json(nullptr)},
                {"prompt_total", json(nullptr)},
                {"generated", active.empty() ? json(nullptr) : json(active.front().generated_tokens)},
                {"max_tokens", json(nullptr)},
                {"elapsed_s", json(nullptr)},
                {"tok_s", active.empty() ? json(nullptr) : json(active.front().decode_tps)},
                {"tok_s_mean", json(nullptr)},
                {"prefill_tok_s_mean", json(nullptr)},
                {"tok_s_window_s", json(nullptr)},
        };

        json requests = json::array();
        size_t n = 0;
        const size_t limit = all_requests ? recent.size() : std::min(recent.size(), size_t(12));
        for (const auto & r : recent) {
            if (n >= limit) {
                break;
            }
            requests.push_back({
                    {"id", r.id},
                    {"status", r.status},
                    {"prompt_tokens", r.prompt_tokens},
                    {"reused", r.cached_tokens},
                    {"output_tokens", r.generated_tokens},
                    {"decode_tok_s", r.decode_tps},
                    {"prompt_ms", nullptr},
                    {"decode_ms", r.decode_tps > 0 && r.generated_tokens > 0
                                         ? json(1000.0 * r.generated_tokens / r.decode_tps)
                                         : json(nullptr)},
                    {"duration_s", r.elapsed_ms / 1000.0},
            });
            ++n;
        }

        const auto hw = host_hardware_sample();
        return {
                {"engine", {
                        {"model", params.model_alias.empty() ? params.model.path : *params.model_alias.begin()},
                        {"max_context", params.n_ctx},
                        {"images", false},
                        {"version", "llama.cpp"},
                }},
                {"live", live},
                {"requests", requests},
                {"conversation_cache", {
                        {"requests", 0},
                        {"requests_reused", 0},
                        {"prompt_tokens", 0},
                        {"reused_tokens", 0},
                }},
                {"requests_kept", recent.size()},
                {"totals", {
                        {"requests", recent.size()},
                        {"prompt_tokens", 0},
                        {"output_tokens", 0},
                        {"reused", 0},
                        {"prompt_ms", 0},
                        {"decode_ms", 0},
                }},
                {"hardware", {
                        {"gpu_util", nullptr},
                        {"gpu_mem_used", nullptr},
                        {"gpu_temp", nullptr},
                        {"gpu_power", nullptr},
                        {"gpu_pcie_rx_mb", nullptr},
                        {"tok_s", live["tok_s"]},
                        {"cpu", hw.contains("cpu_percent") ? hw.at("cpu_percent") : json(nullptr)},
                }},
                {"hardware_static", json::object()},
                {"history", json::object()},
                {"time", now_ms() / 1000.0},
                {"mcp", json::array()},
                {"assist", json::object()},
        };
    }

    void begin_request(const std::string & id, const std::string & instance, bool has_media) {
        std::lock_guard<std::mutex> lock(mutex);
        req_row r;
        r.id = id.empty() ? random_token().substr(0, 16) : id;
        r.instance = instance.empty() ? instance_id : instance;
        r.started_at_ms = now_ms();
        r.status = "generating";
        r.phase = "answering";
        r.has_media = has_media;
        active.push_back(std::move(r));
        while (active.size() > k_active_max) {
            active.pop_front();
        }
    }

    void finish_request(const std::string & id, const std::string & status) {
        std::lock_guard<std::mutex> lock(mutex);
        for (auto it = active.begin(); it != active.end(); ++it) {
            if (it->id == id) {
                it->status = status;
                it->finished_at_ms = now_ms();
                it->elapsed_ms = it->finished_at_ms - it->started_at_ms;
                if (it->elapsed_ms > 0 && it->generated_tokens > 0) {
                    it->decode_tps = 1000.0 * it->generated_tokens / double(it->elapsed_ms);
                }
                recent.push_front(*it);
                while (recent.size() > k_recent_max) {
                    recent.pop_back();
                }
                active.erase(it);
                return;
            }
        }
    }

    void cancel_request(const std::string & id) {
        finish_request(id, "cancelled");
    }

    json find_request(const std::string & id) {
        std::lock_guard<std::mutex> lock(mutex);
        for (const auto & r : active) {
            if (r.id == id) {
                return row_to_json(r);
            }
        }
        for (const auto & r : recent) {
            if (r.id == id) {
                return row_to_json(r);
            }
        }
        return nullptr;
    }

    json launcher_info() {
        std::lock_guard<std::mutex> lock(mutex);
        std::string model = params.model.path;
        if (!params.model_alias.empty()) {
            model = *params.model_alias.begin();
        }
        return {
                {"token", console_token},
                {"backends", json::array({"llamacpp"})},
                {"gpus", json::array({{{"uuid", instance_id}, {"name", "Metal"}}})},
                {"native_picker", false},
                {"prefer_bundled_worker", false},
                {"worker_capabilities_version", 0},
                {"model_inspection_version", 0},
                {"model_library_version", 0},
                {"query_scoring_version", 2},
                {"kv_cache_version", 1},
                {"recall_strategy_version", 0},
                {"status", {
                        {"state", engine_stopped ? "stopped" : "ready"},
                        {"instance", instance_id},
                        {"pid", nullptr},
                        {"active_config", {
                                {"parameters", {{"Model", model}, {"Backend", "llamacpp"}}},
                        }},
                }},
        };
    }

    json chat_context() {
        std::lock_guard<std::mutex> lock(mutex);
        std::string model = params.model.path;
        if (!params.model_alias.empty()) {
            model = *params.model_alias.begin();
        }
        const auto slash = model.find_last_of("/\\");
        const std::string display = slash == std::string::npos ? model : model.substr(slash + 1);
        return {
                {"instance", instance_id},
                {"model", model},
                {"display_model", display},
                {"n_ctx", params.n_ctx},
                {"n_predict", params.n_predict},
                {"enable_thinking", true},
                {"modalities", {{"vision", false}, {"audio", false}, {"video", false}}},
                {"defaults", {
                        {"temperature", params.sampling.temp},
                        {"top_p", params.sampling.top_p},
                        {"top_k", params.sampling.top_k},
                        {"min_p", params.sampling.min_p},
                }},
        };
    }

    json logs_response(const std::string & revision) {
        std::lock_guard<std::mutex> lock(mutex);
        json out = {
                {"running", true},
                {"backend", "llamacpp"},
                {"session", log_session},
                {"revision", std::to_string(log_revision)},
                {"text", log_text},
        };
        if (!revision.empty() && revision == std::to_string(log_revision)) {
            out["text"] = nullptr;
        }
        return out;
    }
};

server_compat::server_compat(const common_params & params) : pimpl(std::make_unique<impl>(params)) {}
server_compat::~server_compat() = default;

void server_compat::append_log(const std::string & line) {
    pimpl->append_log(line);
}

void server_compat::wrap_routes(server_routes & routes) {
    auto & c = *pimpl;

    c.prometheus_metrics = routes.get_metrics;

    {
        auto orig = routes.get_health;
        routes.get_health = [orig, &c](const server_http_req & req) {
            auto res = orig(req);
            try {
                json body = json::parse(res->data);
                body = c.enrich_health(std::move(body));
                res->data = safe_json_to_str(body);
            } catch (...) {
            }
            return res;
        };
    }

    {
        auto orig = routes.get_props;
        routes.get_props = [orig, &c](const server_http_req & req) {
            auto res = orig(req);
            try {
                json body = json::parse(res->data);
                body = c.enrich_props(std::move(body));
                res->data = safe_json_to_str(body);
            } catch (...) {
            }
            return res;
        };
    }

    routes.get_metrics = [&c](const server_http_req & req) {
        const bool all = req.query_string.find("requests=all") != std::string::npos;
        return json_res(200, c.build_metrics(all));
    };

    {
        auto orig = routes.get_slots;
        routes.get_slots = [orig, &c](const server_http_req & req) {
            if (!c.params.endpoint_slots) {
                return json_res(200, json::array());
            }
            return orig(req);
        };
    }
}

void server_compat::register_routes(
        server_http_context & ctx_http,
        server_routes & routes,
        const server_http_context::handler_t & chat_completions,
        const route_wrapper_t & wrap) {
    auto & c = *pimpl;
    (void) routes;

    auto W = [&wrap](server_http_context::handler_t h) {
        return wrap(std::move(h));
    };

    // prometheus moved off /metrics
    if (c.prometheus_metrics) {
        ctx_http.get("/metrics.prom", W(c.prometheus_metrics));
        ctx_http.get("/metrics/prometheus", W(c.prometheus_metrics));
    }

    ctx_http.get("/api/monitor/snapshot", W([&c](const server_http_req &) {
        return json_res(200, c.build_snapshot());
    }));

    ctx_http.get("/api/requests", W([&c](const server_http_req & req) {
        const std::string id = req.get_param("id");
        if (!id.empty()) {
            json row = c.find_request(id);
            if (row.is_null()) {
                return json_res(404, {{"error", {{"message", "request no longer retained"}}}});
            }
            return json_res(200, row);
        }
        auto snap = c.build_snapshot();
        json list = json::array();
        for (const auto & r : snap["recent"]) {
            list.push_back(r);
        }
        return json_res(200, {
                {"requests", list},
                {"retention", 100},
                {"persistent", false},
                {"loaded", !c.engine_stopped},
                {"auto_load", false},
        });
    }));

    ctx_http.get("/api/launcher/info", W([&c](const server_http_req &) {
        return json_res(200, c.launcher_info());
    }));

    ctx_http.get("/api/launcher/hardware", W([](const server_http_req &) {
        return json_res(200, host_hardware_sample());
    }));

    ctx_http.get("/api/launcher/logs", W([&c](const server_http_req & req) {
        return json_res(200, c.logs_response(req.get_param("revision")));
    }));

    ctx_http.get("/api/launcher/log", W([&c](const server_http_req &) {
        auto full = c.logs_response("");
        std::string text = full.value("text", std::string());
        if (text.size() > 32 * 1024) {
            text = text.substr(text.size() - 32 * 1024);
        }
        full["text"] = text;
        return json_res(200, full);
    }));

    // launcher write stubs
    const char * launcher_stubs[] = {
            "status", "service", "services", "start-revision", "export", "preview",
            "retry-memory", "adjust-budget", "browse-model", "browse-model-folder",
            "browse-mmproj", "browse-template", "scan-models", "inspect-model",
            "model-directories", "worker-capabilities",
    };
    for (const char * name : launcher_stubs) {
        const std::string path = std::string("/api/launcher/") + name;
        ctx_http.get(path, W([name](const server_http_req &) {
            return stub_unsupported(std::string("launcher ") + name);
        }));
        ctx_http.post(path, W([name](const server_http_req &) {
            return stub_unsupported(std::string("launcher ") + name);
        }));
    }

    ctx_http.get("/api/chat/context", W([&c](const server_http_req &) {
        return json_res(200, c.chat_context());
    }));

    ctx_http.get("/api/chat/requests/:id", W([&c](const server_http_req & req) {
        const std::string id = req.get_param("id");
        json row = c.find_request(id);
        if (row.is_null()) {
            return json_res(404, {{"error", "not found"}});
        }
        return json_res(200, row);
    }));

    ctx_http.post("/api/chat/cancel", W([&c](const server_http_req & req) {
        try {
            json body = json::parse(req.body.empty() ? "{}" : req.body);
            const std::string id = json_value(body, "id", std::string());
            if (!id.empty()) {
                c.cancel_request(id);
            }
        } catch (...) {
        }
        return json_res(200, {{"ok", true}});
    }));

    ctx_http.post("/api/chat/completions", W([&c, chat_completions](const server_http_req & req) -> server_http_res_ptr {
        if (!c.token_ok(req)) {
            return json_res(401, {{"error", {{"message", "unauthorized"}}}});
        }
        if (c.engine_stopped) {
            return json_res(409, {{"error", {{"message", "engine stopped"}}}});
        }
        std::string id = header_get(req, "X-KVMem-Request");
        if (id.empty()) {
            id = random_token().substr(0, 16);
        }
        const std::string instance = header_get(req, "X-KVMem-Instance");
        bool has_media = false;
        try {
            const json body = json::parse(req.body.empty() ? "{}" : req.body);
            if (body.contains("messages")) {
                for (const auto & m : body.at("messages")) {
                    if (m.contains("content") && m.at("content").is_array()) {
                        has_media = true;
                        break;
                    }
                }
            }
        } catch (...) {
        }
        c.begin_request(id, instance, has_media);
        auto inner = chat_completions(req);
        return server_http_res_ptr(std::make_unique<wrapping_res>(std::move(inner), [&c, id]() {
            c.finish_request(id, "completed");
        }));
    }));

    // HyperCore surface
    ctx_http.get("/prompts", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        return json_res(200, c.prompts);
    }));

    ctx_http.get("/history/status", W([&c](const server_http_req &) {
        int count = 0;
        // lightweight: report directory presence only
        std::ifstream probe(c.data_dir + "/_old_chats/.keep");
        (void) probe;
        return json_res(200, {
                {"archived", count},
                {"distilled", 0},
                {"last_archive_at", nullptr},
                {"last_rebuild_at", nullptr},
                {"rebuild_running", false},
        });
    }));

    ctx_http.post("/history/archive", W([&c](const server_http_req & req) {
        ensure_dir(c.data_dir + "/_old_chats");
        const std::string name = "chat-" + std::to_string(now_ms()) + ".json";
        const std::string path = c.data_dir + "/_old_chats/" + name;
        std::ofstream out(path);
        if (out) {
            out << (req.body.empty() ? "{}" : req.body);
        }
        c.append_log("history archive " + name);
        return json_res(200, {{"ok", true}, {"path", path}});
    }));

    ctx_http.get("/history/distil", W([](const server_http_req &) {
        return json_res(200, {{"running", false}, {"error", "no_distil_pipeline"}});
    }));
    ctx_http.post("/history/distil", W([](const server_http_req &) {
        return json_res(200, {{"running", false}, {"error", "no_distil_pipeline"}});
    }));

    ctx_http.get("/settings", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        return json_res(200, c.settings);
    }));
    ctx_http.post("/settings", W([&c](const server_http_req & req) {
        try {
            json body = json::parse(req.body.empty() ? "{}" : req.body);
            std::lock_guard<std::mutex> lock(c.mutex);
            if (body.is_object()) {
                impl::merge_object(c.settings, body);
                c.save_json_file(c.data_dir + "/settings.json", c.settings);
            }
            return json_res(200, c.settings);
        } catch (const std::exception & e) {
            return json_res(400, {{"error", e.what()}});
        }
    }));

    ctx_http.get("/config", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        json out = c.config;
        out["model"] = c.params.model.path;
        out["n_ctx"] = c.params.n_ctx;
        out["n_predict"] = c.params.n_predict;
        return json_res(200, out);
    }));
    ctx_http.post("/config", W([&c](const server_http_req & req) {
        try {
            json body = json::parse(req.body.empty() ? "{}" : req.body);
            std::lock_guard<std::mutex> lock(c.mutex);
            if (body.contains("set") && body.at("set").is_object()) {
                impl::merge_object(c.config, body.at("set"));
            } else if (body.is_object()) {
                impl::merge_object(c.config, body);
            }
            c.save_json_file(c.data_dir + "/config.json", c.config);
            return json_res(200, c.config);
        } catch (const std::exception & e) {
            return json_res(400, {{"error", e.what()}});
        }
    }));

    ctx_http.get("/config/strata-hook", W([](const server_http_req &) {
        return json_res(200, {{"running", false}, {"log", ""}, {"offset", 0}});
    }));
    ctx_http.post("/config/strata-hook", W([](const server_http_req &) {
        return stub_unsupported("strata-hook");
    }));
    ctx_http.post("/config/strata-hook/stop", W([](const server_http_req &) {
        return json_res(200, {{"ok", true}});
    }));

    ctx_http.get("/versions", W([&c](const server_http_req &) {
        return json_res(200, {
                {"active", "llama"},
                {"versions", {{"llama", {{"engine", "llama.cpp"}, {"instance", c.instance_id}}}}},
        });
    }));
    ctx_http.post("/versions/select", W([](const server_http_req &) {
        return stub_unsupported("versions/select");
    }));

    ctx_http.get("/mcp", W([](const server_http_req &) {
        return json_res(200, {{"servers", json::array()}});
    }));
    ctx_http.post("/mcp", W([](const server_http_req &) {
        return json_res(200, {{"ok", true}, {"servers", json::array()}});
    }));

    // Assist stubs / minimal
    ctx_http.get("/assist/models", W([](const server_http_req &) {
        return json_res(200, {{"models", json::array()}, {"configured", false}});
    }));
    ctx_http.get("/assist/log", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        return json_res(200, {{"calls", c.assist_log}});
    }));
    ctx_http.get("/assist/probe/status", W([](const server_http_req &) {
        return json_res(200, {{"running", false}});
    }));
    ctx_http.post("/assist/models/probe", W([](const server_http_req &) {
        return stub_unsupported("assist probe (set external endpoint later)");
    }));
    ctx_http.post("/assist/chat", W([](const server_http_req &) {
        return json_res(503, {{"error", {{"message", "assist endpoint not configured"}}}});
    }));

    // load/unload / lifecycle
    auto refuse_unload = W([](const server_http_req &) {
        return json_res(409, {{"error", {{"message", "unload refused: single-model llama-server keeps weights resident"}}}});
    });
    ctx_http.post("/unload", refuse_unload);
    ctx_http.post("/v1/unload", refuse_unload);
    ctx_http.post("/load", W([](const server_http_req &) {
        return json_res(200, {{"ok", true}, {"loaded", true}});
    }));
    ctx_http.post("/v1/load", W([](const server_http_req &) {
        return json_res(200, {{"ok", true}, {"loaded", true}});
    }));

    ctx_http.post("/v1/stop", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        c.engine_stopped = true;
        return json_res(200, {{"ok", true}, {"engine_stopped", true}});
    }));
    ctx_http.post("/v1/start", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        c.engine_stopped = false;
        return json_res(200, {{"ok", true}, {"engine_stopped", false}});
    }));
    ctx_http.post("/v1/shutdown", W([](const server_http_req &) {
        return json_res(200, {{"ok", false}, {"error", "shutdown ignored; stop the process externally"}});
    }));
    ctx_http.post("/v1/vram", W([](const server_http_req &) {
        return stub_unsupported("v1/vram");
    }));

    ctx_http.get("/v1/status", W([&c](const server_http_req &) {
        std::lock_guard<std::mutex> lock(c.mutex);
        return json_res(200, {
                {"service", "tosh-llama-compat"},
                {"model", c.params.model.path},
                {"loaded", !c.engine_stopped},
                {"engine_stopped", c.engine_stopped},
                {"started", c.started_at_ms / 1000},
                {"uptime_s", (now_ms() - c.started_at_ms) / 1000},
                {"dialects", json::array({"/v1/chat/completions", "/v1/messages", "/v1/responses"})},
        });
    }));

    // Tune / Build stubs
    auto tune_stub = W([](const server_http_req &) {
        return json_res(200, {{"running", false}, {"enabled", false}, {"error", "tune unavailable on llama-server"}});
    });
    ctx_http.get("/tune/status", tune_stub);
    ctx_http.get("/tune/params", W([](const server_http_req &) {
        return json_res(200, {{"params", json::array()}});
    }));
    ctx_http.get("/tune/results", W([](const server_http_req &) {
        return json_res(200, {{"results", json::array()}});
    }));
    ctx_http.post("/tune/run", W([](const server_http_req &) { return stub_unsupported("tune/run"); }));
    ctx_http.post("/tune/stop", W([](const server_http_req &) { return json_res(200, {{"ok", true}}); }));
    ctx_http.post("/tune/apply", W([](const server_http_req &) { return stub_unsupported("tune/apply"); }));

    auto build_stub = W([](const server_http_req &) {
        return json_res(200, {{"running", false}, {"enabled", false}, {"error", "build unavailable on llama-server"}});
    });
    ctx_http.get("/build/status", build_stub);
    ctx_http.get("/build/options", W([](const server_http_req &) {
        return json_res(200, {{"options", json::array()}});
    }));
    ctx_http.get("/build/log", W([](const server_http_req &) {
        return json_res(200, {{"text", ""}, {"offset", 0}});
    }));
    ctx_http.post("/build/check", W([](const server_http_req &) { return stub_unsupported("build/check"); }));
    ctx_http.post("/build/fetch-prebuilt", W([](const server_http_req &) { return stub_unsupported("build/fetch-prebuilt"); }));
    ctx_http.post("/build/start", W([](const server_http_req &) { return stub_unsupported("build/start"); }));
    ctx_http.post("/build/cancel", W([](const server_http_req &) { return json_res(200, {{"ok", true}}); }));
    ctx_http.post("/build/apply", W([](const server_http_req &) { return stub_unsupported("build/apply"); }));

    c.append_log("compat routes registered");
}
