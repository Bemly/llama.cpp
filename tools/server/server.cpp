#include "server-context.h"
#include "server-http.h"
#include "server-models.h"
#include "server-cors-proxy.h"
#include "server-stream.h"
#include "server-tools.h"

#include "arg.h"
#include "build-info.h"
#include "common.h"
#include "fit.h"
#include "llama.h"
#include "log.h"

#include <algorithm>
#include <atomic>
#include <clocale>
#include <condition_variable>
#include <deque>
#include <mutex>
#include <random>
#include <exception>
#include <signal.h>
#include <thread> // for std::thread::hardware_concurrency

#if defined(_WIN32)
#include <winsock2.h>
#include <ws2tcpip.h>
#include <windows.h>
#else
#include <arpa/inet.h>
#endif

static std::function<void(int)> shutdown_handler;

// an address only this computer reaches: loopback in IPv4 or IPv6, or a UNIX socket; a host name may resolve elsewhere
static bool tosh_local_address(std::string host) {
    if (host == "localhost" || string_ends_with(host, ".sock")) {
        return true;
    }
    if (host.size() > 2 && host.front() == '[' && host.back() == ']') {
        host = host.substr(1, host.size() - 2);
    }
    unsigned char v4[4];
    if (inet_pton(AF_INET, host.c_str(), v4) == 1) {
        return v4[0] == 127;
    }
    unsigned char v6[16];
    if (inet_pton(AF_INET6, host.c_str(), v6) != 1) {
        return false;
    }
    const unsigned char loopback[16] = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1};
    const unsigned char mapped[12]   = {0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff};
    return std::equal(v6, v6 + 16, loopback) || (std::equal(v6, v6 + 12, mapped) && v6[12] == 127);
}
static std::atomic_flag is_terminating = ATOMIC_FLAG_INIT;

static inline void signal_handler(int signal) {
    if (is_terminating.test_and_set()) {
        // in case it hangs, we can force terminate the server by hitting Ctrl+C twice
        // this is for better developer experience, we can remove when the server is stable enough
        fprintf(stderr, "Received second interrupt, terminating immediately.\n");
        exit(1);
    }

    shutdown_handler(signal);
}

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
int llama_server(common_params & params, int argc, char ** argv, server_child & child);
void llama_server_terminate();
void llama_server_terminate() {
    if (shutdown_handler) {
        shutdown_handler(0);
    }
}


// The Tosh agent (--mcp-agent NAME) answers /v1/chat/completions with the "run" tool of the MCP server
// NAME, which calls this server back for raw inference and /tools. The contract, per request:
//   X-Tosh-Agent: off (or raw)  raw model inference, always
//   X-Tosh-Agent: on            the agent; refused when the server runs none or the request brings tools
//                               (with --mcp-agent-client-tools, tools of the client's own go to the agent too)
//   no header                   the agent, unless the server takes it on request only (--mcp-agent-explicit)
//                               and the request does not come from a page in a browser (the bundled web chat),
//                               or the request brings tools, functions, a grammar, json_schema or
//                               response_format: then raw, so clients that run their own tools are unaffected
static std::string tosh_header(const server_http_req & req, const char * name) {
    for (const auto & [key, value] : req.headers) {
        std::string k = key, v = value;
        std::transform(k.begin(), k.end(), k.begin(), ::tolower);
        std::transform(v.begin(), v.end(), v.begin(), ::tolower);
        if (k == name) {
            return v;
        }
    }
    return "";
}

static bool tosh_agent_constrained(const json & body, bool client_tools) {
    for (const char * key : {"tools", "functions", "grammar", "json_schema", "response_format"}) {
        if (client_tools && (std::string(key) == "tools" || std::string(key) == "functions")) {
            continue;
        }
        if (body.contains(key) && !body.at(key).is_null() && !(body.at(key).is_array() && body.at(key).empty())) {
            return true;
        }
    }
    return false;
}

static json tosh_agent_answer(const json & out, int & status) {
    status = 502;
    auto failure = [](const std::string & message) {
        return json{{"error", {{"message", message}, {"type", "agent_error"}}}};
    };
    if (out.contains("error")) {
        return failure(out.at("error").is_string() ? out.at("error").get<std::string>() : out.at("error").dump());
    }
    if (!out.contains("plain_text_response") || !out.at("plain_text_response").is_string()) {
        return failure("the agent gave no answer");
    }
    json answer;
    try {
        answer = json::parse(out.at("plain_text_response").get<std::string>());
    } catch (const std::exception &) {
        return failure("the agent gave no answer");
    }
    if (!answer.is_object()) {
        return failure("the agent gave no answer");
    }
    if (answer.contains("error")) {
        status = answer.value("status", 500);
        answer.erase("status");
        return answer;
    }
    status = 200;
    return answer;
}

// the answer goes out once the agent has checked it, except the text of a pass it streamed live; before that
// only progress events and keep-alives. `streamed` is the text already sent since the last retracted pass
static std::vector<std::string> tosh_agent_chunks(const json & answer, int status, const json & head,
                                                  const std::string & streamed, bool retracted, bool started) {
    if (status != 200) {
        return {"data: " + answer.dump() + "\n\n", "data: [DONE]\n\n"};
    }
    std::string content;
    if (answer.contains("choices") && answer.at("choices").is_array() && !answer.at("choices").empty()) {
        content = answer.at("choices").at(0).value("message", json::object()).value("content", "");
    }
    if (!streamed.empty() && content.compare(0, streamed.size(), streamed) == 0) {
        content.erase(0, streamed.size());
    } else if (retracted || !streamed.empty()) {
        // a client that cannot drop the text it was shown reads the answer after it
        content = "\n\n" + content;
    }
    json delta = {{"content", content}};
    if (!started) {
        delta["role"] = "assistant";
    }
    // calls to the client's own tools, which the client runs and sends back
    json calls = json::array();
    if (answer.contains("choices") && answer.at("choices").is_array() && !answer.at("choices").empty()) {
        json message = answer.at("choices").at(0).value("message", json::object());
        if (message.contains("tool_calls") && message.at("tool_calls").is_array()) {
            calls = message.at("tool_calls");
        }
    }
    for (size_t i = 0; i < calls.size(); i++) {
        calls[i]["index"] = (int) i;
    }
    if (!calls.empty()) {
        delta["tool_calls"] = calls;
    }
    // an answer the agent says was cut at its token limit keeps that reason, so a client can say so
    std::string finish = calls.empty() ? "stop" : "tool_calls";
    if (calls.empty() && answer.contains("choices") && answer.at("choices").is_array() && !answer.at("choices").empty() &&
        answer.at("choices").at(0).value("finish_reason", "") == "length") {
        finish = "length";
    }
    json first = head;
    first["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
    json last = head;
    last["choices"] = json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", finish}}});
    for (const char * key : {"usage", "tosh"}) {
        if (answer.contains(key)) {
            last[key] = answer.at(key);
        }
    }
    return {"data: " + first.dump() + "\n\n", "data: " + last.dump() + "\n\n", "data: [DONE]\n\n"};
}

struct tosh_agent_state {
    std::mutex mutex;
    std::condition_variable cv;
    std::deque<json> events;
    std::atomic<bool> cancelled{false};
    bool done = false;
    json out;
};

// a client that goes away stops the agent, its model passes and its tool calls, unless the request names its
// conversation: then the agent runs to its end into the replay buffer, where the client picks the answer up again
struct tosh_agent_res : server_res_spipe {
    std::shared_ptr<tosh_agent_state> state;
    void on_complete() override {
        server_res_spipe::on_complete();
        state->cancelled = true;
        state->cv.notify_all();
    }
};

static server_http_res_ptr tosh_agent_refuse(const std::string & message) {
    auto res = std::make_unique<server_http_res>();
    res->status = 400;
    res->data = json{{"error", {{"message", message}, {"type", "invalid_request_error"}}}}.dump();
    return res;
}

// 256 bits from the system's generator
static std::string tosh_agent_key() {
    std::random_device rd;
    std::string key;
    for (int i = 0; i < 8; i++) {
        key += string_format("%08x", (unsigned) rd());
    }
    return key;
}

static server_http_context::handler_t tosh_agent_handler(server_http_context::handler_t raw,
                                                         server_mcp & mcp_mgr,
                                                         const common_params & params,
                                                         const std::string & agent_key) {
    return [raw, &mcp_mgr, &params, agent_key](const server_http_req & req) -> server_http_res_ptr {
        const std::string mode = tosh_header(req, "x-tosh-agent");
        if (mode == "off" || mode == "raw") {
            return raw(req);
        }
        const bool asked = mode == "on";
        json body;
        try {
            body = json::parse(req.body);
        } catch (const std::exception &) {
            return raw(req);
        }
        if (params.mcp_agent.empty()) {
            return asked ? tosh_agent_refuse("this server runs no Tosh agent: start it with --mcp-agent") : raw(req);
        }
        if (!body.is_object() || tosh_agent_constrained(body, params.mcp_agent_client_tools)) {
            return asked ? tosh_agent_refuse("a request for the Tosh agent cannot bring tools, functions, a grammar or a response format")
                         : raw(req);
        }
        // browsers mark every request with one of these; a page never gets the MCP tools for its own loop
        const bool browser = !tosh_header(req, "sec-fetch-site").empty() || !tosh_header(req, "origin").empty();
        if (!asked && params.mcp_agent_explicit && !browser) {
            return raw(req);
        }
        std::string host = params.hostnames.empty() ? "127.0.0.1" : params.hostnames.front();
        if (host.empty() || host == "0.0.0.0" || host == "::") {
            host = "127.0.0.1";
        }
        // an IPv6 literal goes in brackets, the % of its zone written %25 (RFC 6874)
        if (host.find(':') != std::string::npos) {
            std::string escaped;
            for (char c : host) {
                escaped += c == '%' ? std::string("%25") : std::string(1, c);
            }
            host = "[" + escaped + "]";
        }
        json args = {{"request", body},
                     {"base_url", "http://" + host + ":" + std::to_string(params.port)},
                     {"api_key", params.api_keys.empty() ? "" : params.api_keys.front()},
                     {"agent_key", agent_key}};
        if (!body.value("stream", false)) {
            auto res = std::make_unique<server_http_res>();
            int status = 502;
            json answer = tosh_agent_answer(mcp_mgr.call_tool(params.mcp_agent, "run", args, req.should_stop), status);
            res->status = status;
            res->data = answer.dump();
            return res;
        }
        auto state = std::make_shared<tosh_agent_state>();
        std::thread([state, args, &mcp_mgr, name = params.mcp_agent]() {
            json out = mcp_mgr.call_tool(name, "run", args,
                [state]() { return state->cancelled.load(); },
                [state](const json & progress) {
                    std::lock_guard<std::mutex> lock(state->mutex);
                    state->events.push_back(progress);
                    state->cv.notify_all();
                });
            std::lock_guard<std::mutex> lock(state->mutex);
            state->out = std::move(out);
            state->done = true;
            state->cv.notify_all();
        }).detach();
        auto res = std::make_unique<tosh_agent_res>();
        res->set_req(&req);
        res->state = state;
        res->content_type = "text/event-stream";
        res->status = 200;
        json head = {{"id", "chatcmpl-tosh-" + tosh_agent_key().substr(0, 32)}, {"object", "chat.completion.chunk"},
                     {"created", (int64_t) time(nullptr)},
                     {"model", body.contains("model") && body.at("model").is_string() ? body.at("model").get<std::string>() : "tosh-agent"}};
        auto gone = [self = res.get()]() { return self->should_stop(); };
        res->set_next([state, gone, head, finals = std::vector<std::string>(), sent = size_t(0),
                     streamed = std::string(), retracted = false, started = false](std::string & output) mutable -> bool {
            if (finals.empty()) {
                std::unique_lock<std::mutex> lock(state->mutex);
                // a keep-alive every 5 s, but a client that went away is noticed within a second
                for (int i = 0; i < 5 && !state->done && state->events.empty(); i++) {
                    state->cv.wait_for(lock, std::chrono::seconds(1), [&] { return state->done || !state->events.empty(); });
                    if (gone()) {
                        state->cancelled = true;
                        state->cv.notify_all();
                        return false;
                    }
                }
                if (!state->events.empty()) {
                    json progress = state->events.front();
                    state->events.pop_front();
                    json event = progress.value("message", json());
                    if (event.is_string()) {
                        try {
                            event = json::parse(event.get<std::string>());
                        } catch (const std::exception &) {
                        }
                    }
                    const std::string type = event.is_object() ? event.value("type", "") : "";
                    // text of a pass the agent streams live goes out as the model writes it
                    if (type == "delta") {
                        json delta = json::object();
                        for (const char * field : {"content", "reasoning_content"}) {
                            if (event.contains(field) && event.at(field).is_string()) {
                                std::string text = event.at(field).get<std::string>();
                                if (std::string(field) == "content") {
                                    streamed += text;
                                    if (retracted && streamed.size() == text.size()) {
                                        text = "\n\n" + text;
                                    }
                                }
                                delta[field] = text;
                            }
                        }
                        if (!started) {
                            delta["role"] = "assistant";
                            started = true;
                        }
                        json chunk = head;
                        chunk["choices"] = json::array({{{"index", 0}, {"delta", delta}, {"finish_reason", nullptr}}});
                        output = "data: " + chunk.dump() + "\n\n";
                        return true;
                    }
                    // that pass ended in tool calls: its text was not the answer
                    if (type == "retract") {
                        retracted = retracted || !streamed.empty();
                        streamed.clear();
                    }
                    json chunk = {{"object", "chat.completion.chunk"},
                                  {"choices", json::array({{{"index", 0}, {"delta", json::object()}, {"finish_reason", nullptr}}})},
                                  {"tosh", {{"version", 1}, {"event", event}}}};
                    output = "data: " + chunk.dump() + "\n\n";
                    return true;
                }
                if (!state->done) {
                    output = ": keep-alive\n\n";
                    return true;
                }
                int status = 502;
                finals = tosh_agent_chunks(tosh_agent_answer(state->out, status), status, head, streamed, retracted, started);
            }
            output = finals[sent++];
            return sent < finals.size();
        });
        return res;
    };
}

// wrapper function that handles exceptions and logs errors
// this is to make sure handler_t never throws exceptions; instead, it returns an error response
static server_http_context::handler_t ex_wrapper(server_http_context::handler_t func) {
    return [func = std::move(func)](const server_http_req & req) -> server_http_res_ptr {
        std::string message;
        error_type error;
        try {
            return func(req);
        } catch (const std::invalid_argument & e) {
            // treat invalid_argument as invalid request (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const common_json_error & e) {
            // JSON parse and type errors are invalid requests (400)
            error = ERROR_TYPE_INVALID_REQUEST;
            message = e.what();
        } catch (const std::exception & e) {
            // treat other exceptions as server error (500)
            error = ERROR_TYPE_SERVER;
            message = e.what();
        } catch (...) {
            error = ERROR_TYPE_SERVER;
            message = "unknown error";
        }

        auto res = std::make_unique<server_http_res>();
        res->status = 500;
        try {
            json error_data = format_error_response(message, error);
            res->status = json_value(error_data, "code", 500);
            res->data = safe_json_to_str({{ "error", error_data }});
            SRV_WRN("got exception: %s\n", res->data.c_str());
        } catch (const std::exception & e) {
            SRV_ERR("got another exception: %s | while handling exception: %s\n", e.what(), message.c_str());
            res->data = "Internal Server Error";
        }
        return res;
    };
}

int llama_server(int argc, char ** argv) {
    server_child child;

    std::setlocale(LC_NUMERIC, "C");

#ifndef _WIN32
    // Ignore SIGPIPE so the server does not crash if a child (MCP server, tools runtime) exits while we are writing to its stdin
    signal(SIGPIPE, SIG_IGN);
#endif

    // own arguments required by this example
    common_params params;

    common_init();

    // start the stream session manager GC right after common init, before any HTTP route can
    // touch it. lifecycle is symmetric, stop_gc() runs in clean_up() before backend free
    server_stream_session_manager_start();

    if (!common_params_parse(argc, argv, params, LLAMA_EXAMPLE_SERVER)) {
        return 1;
    }

    SRV_INF("%s", "initializing ...\n");

    llama_backend_init();
    llama_numa_init(params.numa);

    const int result = llama_server(params, argc, argv, child);
    common_log_flush(common_log_main());
    return result;
}

int llama_server(common_params & params, int argc, char ** argv) {
    server_child child;
    return llama_server(params, argc, argv, child);
}

int llama_server(common_params & params, int argc, char ** argv, server_child & child) {
    bool is_run_by_cli = (argv == nullptr);

    common_models_handler models_handler;

    // note: router mode also accepts -hf remote-preset, so we need to check that first
    if (!is_run_by_cli && !params.model.hf_repo.empty()) {
        try {
            models_handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            if (common_models_handler_is_preset_repo(models_handler)) {
                // apply the preset and start the server in router mode
                common_models_handler_apply(models_handler, params);
            }
        } catch (const std::exception & e) {
            SRV_ERR("failed to fetch model metadata: %s\n", e.what());
            return 1;
        }
    }

    // router server never loads a model and must not touch the GPU
    const bool is_router_server = params.model.path.empty()
                               && params.model.hf_repo.empty()
                               && params.model.docker_repo.empty();

    // skip device enumeration so the CUDA primary context stays uncreated
    common_params_print_info(params, !is_router_server);

    if (!is_router_server) {
        // validate batch size for embeddings
        // embeddings require all tokens to be processed in a single ubatch
        // see https://github.com/ggml-org/llama.cpp/issues/12836
        if (params.embedding && params.n_batch > params.n_ubatch) {
            SRV_WRN("embeddings enabled with n_batch (%d) > n_ubatch (%d)\n", params.n_batch, params.n_ubatch);
            SRV_WRN("setting n_batch = n_ubatch = %d to avoid assertion failure\n", params.n_ubatch);
            params.n_batch = params.n_ubatch;
        }

        if (params.n_parallel < 0) {
            SRV_TRC("%s", "n_parallel is set to auto, using n_parallel = 4 and kv_unified = true\n");

            params.n_parallel = 4;
            params.kv_unified = true;
        }
    }

    // size the KV pool from --kv-unified-per-slot, unless the user pinned it with -c
    // or with -c 0 for max context
    const bool ctx_pool_auto_sized = params.kv_unified_per_slot > 0 &&
                                     params.n_ctx == 0 &&
                                     (uint32_t) params.fit_params_min_ctx != UINT32_MAX;

    if (ctx_pool_auto_sized) {
        params.n_ctx = params.n_parallel * params.kv_unified_per_slot;
        SRV_INF("--kv-unified-per-slot: sizing KV pool to n_parallel * kv_unified_per_slot = %d * %d = %d\n", params.n_parallel,
                params.kv_unified_per_slot, params.n_ctx);
    }

    // for consistency between server router mode and single-model mode, we set the same model name as alias
    auto model_name = params.model.get_name();
    if (params.model_alias.empty() && !model_name.empty()) {
        params.model_alias.insert(model_name);
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;

    //
    // Router
    //

    // register API routes
    server_routes routes(params, ctx_server);
    server_tools tools;

    std::optional<server_models_routes> models_routes{};

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    if (is_router_server) {
        // setup server instances manager
        try {
            models_routes.emplace(params, argc, argv);
        } catch (const std::exception & e) {
            SRV_ERR("failed to initialize router models: %s\n", e.what());
            return 1;
        }

        // proxy handlers
        // note: routes.get_health stays the same
        routes.get_metrics                 = models_routes->proxy_get;
        routes.post_props                  = models_routes->proxy_post;
        routes.post_completions            = models_routes->proxy_post;
        routes.post_completions_oai        = models_routes->proxy_post;
        routes.post_chat_completions       = models_routes->proxy_post;
        routes.post_control                = models_routes->proxy_post;
        routes.post_responses_oai          = models_routes->proxy_post;
        routes.post_transcriptions_oai     = models_routes->proxy_post;
        routes.post_anthropic_messages     = models_routes->proxy_post;
        routes.post_anthropic_count_tokens = models_routes->proxy_post;
        routes.post_infill                 = models_routes->proxy_post;
        routes.post_embeddings             = models_routes->proxy_post;
        routes.post_embeddings_oai         = models_routes->proxy_post;
        routes.post_rerank                 = models_routes->proxy_post;
        routes.post_systemone              = models_routes->proxy_post;
        routes.post_tokenize               = models_routes->proxy_post;
        routes.post_detokenize             = models_routes->proxy_post;
        routes.post_apply_template         = models_routes->proxy_post;
        routes.post_chat_completions_tok   = models_routes->proxy_post;
        routes.post_responses_tok_oai      = models_routes->proxy_post;
        routes.get_lora_adapters           = models_routes->proxy_get;
        routes.post_lora_adapters          = models_routes->proxy_post;
        routes.get_slots                   = models_routes->proxy_get;
        routes.post_slots                  = models_routes->proxy_post;

        // custom routes for router
        routes.get_props                   = models_routes->get_router_props;
        routes.get_models                  = models_routes->get_router_models;

        ctx_http.post("/models",               ex_wrapper(models_routes->post_router_models));
        ctx_http.post("/models/load",          ex_wrapper(models_routes->post_router_models_load));
        ctx_http.post("/models/unload",        ex_wrapper(models_routes->post_router_models_unload));
        ctx_http.get ("/models/sse",           ex_wrapper(models_routes->get_router_models_sse));
        ctx_http.del ("/models",               ex_wrapper(models_routes->del_router_models));
    }

    ctx_http.get ("/health",                   ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/v1/health",                ex_wrapper(routes.get_health)); // public endpoint (no API key check)
    ctx_http.get ("/metrics",                  ex_wrapper(routes.get_metrics));
    ctx_http.get ("/props",                    ex_wrapper(routes.get_props));
    ctx_http.post("/props",                    ex_wrapper(routes.post_props));
    ctx_http.get ("/models",                   ex_wrapper(routes.get_models));
    ctx_http.get ("/v1/models",                ex_wrapper(routes.get_models));
    ctx_http.post("/completion",               ex_wrapper(routes.post_completions)); // legacy
    ctx_http.post("/completions",              ex_wrapper(routes.post_completions));
    ctx_http.post("/v1/completions",           ex_wrapper(routes.post_completions_oai));
    // the agent shows this key to /tools for the servers only it may use; new on every start
    const std::string agent_key = params.mcp_agent.empty() ? "" : tosh_agent_key();
    // also without an agent, so that X-Tosh-Agent: on is refused instead of silently answered raw
    auto chat_completions = tosh_agent_handler(routes.post_chat_completions, mcp_mgr, params, agent_key);
    ctx_http.post("/chat/completions",         ex_wrapper(chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/v1/messages",              ex_wrapper(routes.post_anthropic_messages)); // anthropic messages API
    ctx_http.post("/infill",                   ex_wrapper(routes.post_infill));
    ctx_http.post("/embedding",                ex_wrapper(routes.post_embeddings)); // legacy
    ctx_http.post("/embeddings",               ex_wrapper(routes.post_embeddings));
    ctx_http.post("/v1/embeddings",            ex_wrapper(routes.post_embeddings_oai));
    ctx_http.post("/rerank",                   ex_wrapper(routes.post_rerank));
    ctx_http.post("/reranking",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/rerank",                ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/reranking",             ex_wrapper(routes.post_rerank));
    ctx_http.post("/v1/systemone",             ex_wrapper(routes.post_systemone));
    ctx_http.post("/tokenize",                 ex_wrapper(routes.post_tokenize));
    ctx_http.post("/detokenize",               ex_wrapper(routes.post_detokenize));
    ctx_http.post("/apply-template",           ex_wrapper(routes.post_apply_template));
    // token counting
    ctx_http.post("/chat/completions/input_tokens",    ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/v1/chat/completions/input_tokens", ex_wrapper(routes.post_chat_completions_tok));
    ctx_http.post("/responses/input_tokens",           ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/responses/input_tokens",        ex_wrapper(routes.post_responses_tok_oai));
    ctx_http.post("/v1/messages/count_tokens",         ex_wrapper(routes.post_anthropic_count_tokens)); // anthropic token counting
    // LoRA adapters hotswap
    ctx_http.get ("/lora-adapters",            ex_wrapper(routes.get_lora_adapters));
    ctx_http.post("/lora-adapters",            ex_wrapper(routes.post_lora_adapters));
    // Save & load slots
    ctx_http.get ("/slots",                    ex_wrapper(routes.get_slots));
    ctx_http.post("/slots/:id_slot",           ex_wrapper(routes.post_slots));

    // resumable streaming: a child binds the local session factories, the router binds
    // proxies that resolve the owning child, see server-stream.h
    server_http_context::handler_t stream_get_h;
    server_http_context::handler_t streams_lookup_h;
    server_http_context::handler_t stream_delete_h;
    if (is_router_server) {
        stream_get_h     = models_routes->router_stream_get;
        streams_lookup_h = models_routes->router_streams_lookup;
        stream_delete_h  = models_routes->router_stream_delete;
    } else {
        stream_get_h     = server_stream_make_get_handler();
        streams_lookup_h = server_stream_make_lookup_handler();
        stream_delete_h  = server_stream_make_delete_handler();
    }
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get_h));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup_h));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete_h));

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();

    // return 403 for disabled features
    server_http_context::handler_t res_403 = [](const server_http_req &) {
        auto res = std::make_unique<server_http_res>();
        res->status = 403;
        res->data = safe_json_to_str({
            {"error", {
                {"message", "this feature is disabled"},
                {"type", "feature_disabled"},
            }}
        });
        return res;
    };

    if (params.cors_origins == "*" && params.api_keys.empty()) {
        SRV_WRN("%s", "security: no API key is set and CORS allows all origins (see https://github.com/ggml-org/llama.cpp/pull/25655)\n");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
    std::vector<std::string> warn_names;
    if (is_router_server) {
        warn_names.push_back("router mode");
    }

    if (params.ui_mcp_proxy) {
        ctx_http.get ("/cors-proxy",      ex_wrapper(proxy_handler_get));
        ctx_http.post("/cors-proxy",      ex_wrapper(proxy_handler_post));
        warn_names.push_back("MCP proxy (experimental)");
    } else {
        ctx_http.get ("/cors-proxy",      ex_wrapper(res_403));
        ctx_http.post("/cors-proxy",      ex_wrapper(res_403));
    }

    try {
        mcp_mgr.start(params);
    } catch (const std::exception & e) {
        SRV_ERR("MCP starting failed: %s\n", e.what());
        return 1;
    }

    if (!params.server_tools.empty() || !mcp_mgr.empty()) {
        try {
            std::set<std::string> agent_only;
            for (const auto & name : string_split<std::string>(params.mcp_agent_only, ',')) {
                if (!name.empty()) {
                    agent_only.insert(name);
                }
            }
            tools.setup(params.server_tools, mcp_mgr, params.server_tools_runtime, params.mcp_agent, agent_only, agent_key,
                        !params.api_keys.empty());
        } catch (const std::exception & e) {
            SRV_ERR("tools setup failed: %s\n", e.what());
            return 1;
        }
        ctx_http.get ("/tools",           ex_wrapper(tools.handle_get));
        ctx_http.post("/tools",           ex_wrapper(tools.handle_post));
        if (!params.server_tools.empty()) {
            warn_names.push_back("server tools (experimental)");
        }
        if (!params.server_tools_runtime.empty()) {
            warn_names.push_back("tools runtime (experimental)");
        }
        if (!mcp_mgr.empty()) {
            warn_names.push_back("MCP servers (experimental)");
        }
    } else {
        ctx_http.get ("/tools",           ex_wrapper(res_403));
        ctx_http.post("/tools",           ex_wrapper(res_403));
    }

    if (!warn_names.empty()) {
        std::string features;
        for (const auto & name : warn_names) {
            if (!features.empty()) features += ", ";
            features += name;
        }
        SRV_WRN("security: %s enabled - do not expose to untrusted environments\n", features.c_str());
    }

    // serving others without a key is the administrator's choice, made knowing what they reach
    if (params.api_keys.empty()) {
        std::vector<std::string> open;
        for (const auto & host : params.hostnames) {
            if (!host.empty() && !tosh_local_address(host)) {
                open.push_back(host);
            }
        }
        if (!open.empty()) {
            std::string what = is_router_server ? "the API and its models" : "the API and the model";
            if (!mcp_mgr.empty()) {
                what += ", the MCP servers " + string_join(mcp_mgr.server_names(), ", ");
            }
            if (params.ui_mcp_proxy) {
                what += ", the web UI's MCP proxy";
            }
            if (!params.server_tools.empty()) {
                what += " (built-in tools stay for this computer)";
            }
            SRV_WRN("security: listening on %s without an API key: other computers that can reach %s can use %s; "
                    "--api-key makes clients send a key\n",
                    string_join(open, ", ").c_str(), open.size() == 1 ? "it" : "them", what.c_str());
        }
    }

    //
    // Handle downloading model
    //

    if (child.is_child() && child.get_mode() == SERVER_CHILD_MODE_DOWNLOAD) {
        return child.run_download(params);
    } else if (!is_router_server && !is_run_by_cli) {
        // single-model mode (NOT spawned by router)
        // if this is invoked by CLI, model downloading should be already handled
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    //
    // Start the server
    //

    std::function<void()> clean_up;

    if (is_router_server) {
        SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

        clean_up = [&models_routes, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            if (models_routes.has_value()) {
                models_routes->stopping.store(true); // maybe redundant, but just to be safe
                models_routes->models.unload_all();
            }
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }
        ctx_http.is_ready.store(true);

        shutdown_handler = [&](int) {
            if (models_routes.has_value()) {
                // important to disconnect any SSE clients
                models_routes->stopping.store(true);
            }
            mcp_mgr.shutdown();
            ctx_http.stop();
        };

        try {
            models_routes->models.load_startup_models();
        } catch (const std::exception & e) {
            SRV_ERR("failed to load models on startup: %s\n", e.what());
            ctx_http.stop();
            ctx_http.join();
            clean_up();
            return 1;
        }

    } else {
        // setup clean up function, to be called before exit
        clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
            SRV_INF("%s: cleaning up before exit...\n", __func__);
            // stop the session GC first, it finalizes live sessions and wakes pending readers
            server_stream_session_manager_stop();
            ctx_http.stop();
            ctx_server.terminate();
            mcp_mgr.shutdown();
            llama_backend_free();
        };

        // start the HTTP server before loading the model to be able to serve /health requests
        if (!ctx_http.start()) {
            clean_up();
            SRV_ERR("%s", "exiting due to HTTP server error\n");
            return 1;
        }

        // setup communication child --> router if necessary
        if (child.is_child()) {
            ctx_server.set_state_callback([&](server_state state, json payload) {
                child.notify_to_router(server_state_to_str(state), payload);
            });
        }

        if (!ctx_server.load_model(params)) {
            clean_up();
            ctx_http.join();
            SRV_ERR("%s", "exiting due to model loading error\n");
            return 1;
        }

        routes.update_meta(ctx_server);
        ctx_http.is_ready.store(true);

        SRV_INF("%s", "model loaded\n");

        shutdown_handler = [&](int) {
            mcp_mgr.shutdown();
            // this will unblock start_loop()
            ctx_server.terminate();
        };
    }

    // register signal handler if not running by CLI
    if (!is_run_by_cli) {
#if defined (__unix__) || (defined (__APPLE__) && defined (__MACH__))
        struct sigaction sigint_action;
        sigint_action.sa_handler = signal_handler;
        sigemptyset (&sigint_action.sa_mask);
        sigint_action.sa_flags = 0;
        sigaction(SIGINT, &sigint_action, NULL);
        sigaction(SIGTERM, &sigint_action, NULL);
#elif defined (_WIN32)
        auto console_ctrl_handler = +[](DWORD ctrl_type) -> BOOL {
            return (ctrl_type == CTRL_C_EVENT) ? (signal_handler(SIGINT), true) : false;
        };
        SetConsoleCtrlHandler(reinterpret_cast<PHANDLER_ROUTINE>(console_ctrl_handler), true);
#endif
    }

    bool uses_default_port = false;
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
        uses_default_port |= string_ends_with(address, ":8080");
    }

    // TODO: remove this in the future
    // check the string to also handle the .sock case
    if (uses_default_port) {
        SRV_WRN("%s", "notice: server default port will be changed to :9931 in a future release (ref: https://github.com/ggml-org/llama.cpp/pull/26508)\n");
    }

    if (is_router_server) {
        if (!params.models_preset_hf.empty()) {
            SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
            SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
        }

        ctx_http.join(); // keep the main thread alive

        // when the HTTP server stops, clean up and exit
        clean_up();
    } else {
        // optionally, notify router server that this instance is ready
        std::thread monitor_thread;
        if (child.is_child()) {
            monitor_thread = child.setup(shutdown_handler);
            child.notify_to_router(server_state_to_str(SERVER_STATE_READY), routes.get_model_info());
        }

        // this call blocks the main thread until queue_tasks.terminate() is called
        ctx_server.start_loop();

        clean_up();
        ctx_http.join();
        if (monitor_thread.joinable()) {
            monitor_thread.join();
        }

        auto * ll_ctx = ctx_server.get_llama_context();
        if (ll_ctx != nullptr) {
            common_memory_breakdown_print(ll_ctx);
        }
    }

    return 0;
}
