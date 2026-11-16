#include "server-connect.h"
#include "server-context.h"
#include "server-compat.h"
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
#include <unordered_map>

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

// how the server is run; everything is on when it is the llama-server binary
struct server_run_opts {
    bool install_signal_handlers = true; // off when embedded in llama-cli, which handles them itself
    bool resolve_model           = true; // download / resolve -hf and docker models; off when the caller did it already
};

// satisfies -Wmissing-declarations (used by llama command)
int llama_server(int argc, char ** argv);

// to be used via CLI (argc / argv are used by router mode only)
int llama_server(common_params & params, int argc, char ** argv);
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

//
// shared by every role
//

static server_role_type server_role_detect(const common_params & params, server_role & role) {
    if (role.is_managed_by_router()) {
        return role.get_role();
    }
    // a server given no model is a router
    const bool has_model = !params.model.path.empty()
                        || !params.model.hf_repo.empty()
                        || !params.model.docker_repo.empty();
    return has_model ? SERVER_ROLE_INSTANCE : SERVER_ROLE_ROUTER;
}

// adjust the params for the role; returns false on a configuration error
static bool server_prepare_params(common_params & params, server_role_type role_type) {
    if (role_type != SERVER_ROLE_ROUTER) {
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

    // check early, so a missing llama-connect fails-fast
    if (params.server_connect) {
        const std::string reason = server_connect::unavailable_reason(params);
        if (!reason.empty()) {
            SRV_ERR("--connect is not available: %s\n", reason.c_str());
            return false;
        }
        params.server_connect_code = server_connect::resolve_code(params.server_connect_code);
    } else if (!params.server_connect_code.empty()) {
        // the code is only meaningful while the tunnel runs, and /props keys off it being set
        SRV_WRN("%s", "--connect-code is ignored without --connect\n");
        params.server_connect_code.clear();
    }

    return true;
}

// the API of every server; a router points the handlers in routes at its proxies before calling this
// agent_key: shown by the Tosh agent to /tools for the servers only it may use, empty without --mcp-agent
static void server_register_routes(server_http_context & ctx_http, server_routes & routes,
                                   const common_params & params, server_mcp & mcp_mgr, const std::string & agent_key,
                                   server_compat & compat) {
    // kvmem / HyperCore: enrich health/props, JSON /metrics, empty /slots when disabled
    compat.wrap_routes(routes);

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
    // also without an agent, so that X-Tosh-Agent: on is refused instead of silently answered raw
    auto chat_completions = tosh_agent_handler(routes.post_chat_completions, mcp_mgr, params, agent_key);
    // kvmem/Strata monitors: record every chat completion (ToshLLM + stock webui use /v1)
    chat_completions = compat.wrap_chat_completions(std::move(chat_completions));
    ctx_http.post("/chat/completions",         ex_wrapper(chat_completions));
    ctx_http.post("/v1/chat/completions",      ex_wrapper(chat_completions));
    ctx_http.post("/v1/chat/completions/control", ex_wrapper(routes.post_control));
    ctx_http.post("/v1/responses",             ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/responses",                ex_wrapper(routes.post_responses_oai));
    ctx_http.post("/v1/audio/transcriptions",  ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/audio/transcriptions",     ex_wrapper(routes.post_transcriptions_oai));
    ctx_http.post("/tts",                      ex_wrapper(routes.post_tts));
    ctx_http.post("/v1/audio/speech",          ex_wrapper(routes.post_speech_oai));
    ctx_http.post("/audio/speech",             ex_wrapper(routes.post_speech_oai));
    ctx_http.ws  ("/v1/realtime",              routes.ws_realtime);
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

    // kvmem console + HyperCore WebUI compatibility routes
    compat.register_routes(ctx_http, routes, chat_completions, ex_wrapper);

    // Google Cloud Platform (Vertex AI) compat
    ctx_http.register_gcp_compat();
}

// resumable streaming, see server-stream.h: an instance binds the local session factories,
// a router binds proxies that resolve the owning child
static void server_register_stream_routes(
        server_http_context & ctx_http,
        const server_http_context::handler_t & stream_get,
        const server_http_context::handler_t & streams_lookup,
        const server_http_context::handler_t & stream_delete) {
    ctx_http.get ("/v1/stream",                ex_wrapper(stream_get));
    ctx_http.post("/v1/streams/lookup",        ex_wrapper(streams_lookup));
    ctx_http.del ("/v1/stream",                ex_wrapper(stream_delete));
}

// the features behind a flag, and the warning about the ones that must not face untrusted clients
// returns false when one of them fails to start
static bool server_setup_features(
        server_http_context & ctx_http,
        const common_params & params,
        server_mcp & mcp_mgr,
        server_tools & tools,
        server_role_type role_type,
        const std::string & agent_key) {
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

    // tools can run shell commands: do not expose them to the network without auth
    const bool tools_enabled = !params.server_tools.empty() || !params.mcp_servers_config.empty() || !params.mcp_servers_json.empty() || params.ui_mcp_proxy;
    const bool loopback_only = std::all_of(params.hostnames.begin(), params.hostnames.end(), [](const std::string & h) {
        return string_ends_with(h, ".sock") || h == "localhost" || h == "::1" || h == "[::1]" || string_starts_with(h, "127.");
    });
    // ToshLLM serves the LAN without a key on request: built-in tools then stay for this computer
    // (server_tools local_only) and the warning below names what other computers reach, so this only warns
    if (tools_enabled && !loopback_only && params.api_keys.empty() && !params.server_tools_allow_unauth) {
        SRV_WRN("%s", "security: server tools, MCP servers or MCP proxy are enabled on a non-loopback address without an API key\n");
        SRV_WRN("%s", "set --api-key, bind to 127.0.0.1, or pass --allow-unauthenticated-tools on a trusted network\n");
    }

    std::vector<std::string> warn_names;
    if (role_type == SERVER_ROLE_ROUTER) {
        warn_names.push_back("router mode");
    }

    if (params.server_connect) {
        warn_names.push_back("peer-to-peer tunnel (llama-connect, experimental)");
    }

    // CORS proxy (EXPERIMENTAL, only used by the Web UI for MCP)
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
        return false;
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
            return false;
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
            std::string what = role_type == SERVER_ROLE_ROUTER ? "the API and its models" : "the API and the model";
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

    return true;
}

static void server_install_signal_handlers() {
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

// once the HTTP server listens: log the addresses and spawn llama-connect if asked, so the port is
// bound and the child startup health check reaches us. the destructor of connect_proc stops it
static bool server_on_listening(server_http_context & ctx_http, const common_params & params, server_connect & connect_proc) {
    for (const auto & address : ctx_http.listening_addresses) {
        SRV_INF("listening on %s\n", address.c_str());
    }

    if (params.server_connect && !connect_proc.start(params, ctx_http.port)) {
        SRV_ERR("%s", "exiting due to llama-connect error\n");
        return false;
    }
    return true;
}

//
// download worker: a child of the router that pulls a model into the cache and reports the progress
//

struct server_download_state : public common_download_callback {
    server_role * role;
    std::function<bool()> should_stop;
    // files download in parallel threads: throttle each file on its own, a shared
    // timer lets a small, fast file (e.g. a draft) starve the updates of the others
    std::mutex progress_mutex;
    std::unordered_map<std::string, int64_t> last_progress_time;
    bool is_ok = false;

    server_download_state(server_role * r) : role(r) {}

    bool run(common_params & params) {
        try {
            common_models_handler handler = common_models_handler_init(params, LLAMA_EXAMPLE_SERVER);
            common_models_handler_apply(handler, params, this);
            is_ok = true;
        } catch (const std::exception & e) {
            auto model_name = params.model.get_name();
            SRV_ERR("download failed for model name=%s: %s\n", model_name.c_str(), e.what());
            is_ok = false;
        }
        return is_ok;
    }
    void on_progress(const common_download_progress & p) {
        json data = {
            {"url", p.url},
            {"downloaded", p.downloaded},
            {"total", p.total},
        };
        role->notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), data);
    }
    void on_start(const common_download_progress & p) override {
        on_progress(p);
    }
    void on_update(const common_download_progress & p) override {
        int64_t now = ggml_time_ms();
        // throttle progress updates to avoid flooding logs; the first update of a file
        // always passes, it carries the file's total size
        {
            std::lock_guard<std::mutex> lk(progress_mutex);
            auto it = last_progress_time.find(p.url);
            if (it != last_progress_time.end() && now - it->second < 100) {
                return;
            }
            last_progress_time[p.url] = now;
        }
        on_progress(p);
    }
    void on_done(const common_download_progress & p, bool) override {
        on_progress(p);
    }
    bool is_cancelled() const override {
        return should_stop ? should_stop() : false;
    }
};

static int server_run_download(common_params & params, server_role & role) {
    auto cancelled = std::make_shared<std::atomic<bool>>(false);

    // monitor stdin for cancellation command from the router
    std::thread signal_thread = role.setup([cancelled](int) {
        cancelled->store(true, std::memory_order_relaxed);
    });

    server_download_state dl(&role);
    dl.should_stop = [cancelled]() {
        return cancelled->load(std::memory_order_relaxed);
    };

    bool ok = dl.run(params);

    role.notify_to_router(server_state_to_str(SERVER_STATE_DOWNLOADING), {
        {"result", ok ? "download_finished" : "download_failed"},
    });

    // router should send CMD_ROUTER_TO_CHILD_EXIT after receiving the result
    if (signal_thread.joinable()) {
        signal_thread.join();
    }

    SRV_INF("download completed %s\n", ok ? "successfully" : "with errors");
    return 0;
}

//
// measure worker: a child of the router, reports the memory a model needs without loading it
//

static int server_run_measure(common_params & params, server_role & role, common_models_handler & models_handler) {
    // the router sets --offline: only resolve the local path, never download
    try {
        common_models_handler_apply(models_handler, params);
    } catch (const std::exception & e) {
        SRV_ERR("failed to resolve model: %s\n", e.what());
        return 1;
    }

    const std::string model_name = params.model.get_name();

    // TODO: mmproj is not in llama_get_memory_breakdown, a multimodal model is underestimated
    json payload;
    try {
        common_memory_reports m = common_measure_memory(params);
        payload = {
            {"result",      "measured"},
            {"n_ctx_train", m.n_ctx_train},
            {"min",         server_model_memory(m.n_ctx_min, m.info_min).to_json()},
            {"max",         server_model_memory(m.n_ctx_max, m.info_max).to_json()},
        };
        SRV_INF("memory measured for model name=%s at n_ctx=%u (min) and n_ctx=%u (max)\n",
            model_name.c_str(), m.n_ctx_min, m.n_ctx_max);
    } catch (const std::exception & e) {
        SRV_ERR("failed to measure memory for model name=%s: %s\n", model_name.c_str(), e.what());
        payload = {
            {"result", "failed"},
            {"error",  e.what()},
        };
    }

    role.notify_to_router(server_state_to_str(SERVER_STATE_MEASURING), payload);
    return 0;
}

//
// router
//

static int server_run_router(common_params & params, int argc, char ** argv, const server_run_opts & opts) {
    SRV_INF("%s", "starting server in router mode. models will be automatically loaded on-demand\n");

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // the router loads no model, its context stays empty; the routes it does not proxy still need one
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // setup server instances manager
    std::optional<server_models_routes> models_routes;
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
    routes.post_tts                    = models_routes->proxy_post;
    routes.post_speech_oai             = models_routes->proxy_post;
    routes.ws_realtime                 = models_routes->proxy_ws;
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

    // new on every start
    const std::string agent_key = params.mcp_agent.empty() ? "" : tosh_agent_key();

    server_compat compat(params);
    server_register_routes(ctx_http, routes, params, mcp_mgr, agent_key, compat);
    server_register_stream_routes(ctx_http,
        models_routes->router_stream_get,
        models_routes->router_streams_lookup,
        models_routes->router_stream_delete);

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_ROUTER, agent_key)) {
        return 1;
    }

    auto clean_up = [&models_routes, &mcp_mgr]() {
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

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    if (!params.models_preset_hf.empty()) {
        SRV_WRN(      "NOTE: using preset.ini from HF repo '%s'\n", params.models_preset_hf.c_str());
        SRV_WRN("%s", "      please only use presets that you can trust! Unknown presets may be unsafe\n");
    }

    ctx_http.join(); // keep the main thread alive

    // when the HTTP server stops, clean up and exit
    clean_up();

    return 0;
}

//
// instance
//

static int server_run_instance(common_params & params, server_role & role, common_models_handler & models_handler, const server_run_opts & opts) {
    // -hf / docker: download if needed and resolve the local path. a child resolves what the router downloaded
    if (opts.resolve_model) {
        try {
            common_models_handler_apply(models_handler, params);
        } catch (const std::exception & e) {
            SRV_ERR("failed to download model: %s\n", e.what());
            return 1;
        }
    }

    // note: this is guaranteed to out-live ctx_http and tools
    server_mcp mcp_mgr;

    // struct that contains llama context and inference
    server_context ctx_server;
    server_routes  routes(params, ctx_server);
    server_tools   tools;

    server_http_context ctx_http;
    if (!ctx_http.init(params)) {
        SRV_ERR("%s", "failed to initialize HTTP server\n");
        return 1;
    }

    // new on every start
    const std::string agent_key = params.mcp_agent.empty() ? "" : tosh_agent_key();

    server_compat compat(params);
    server_register_routes(ctx_http, routes, params, mcp_mgr, agent_key, compat);
    server_register_stream_routes(ctx_http,
        server_stream_make_get_handler(),
        server_stream_make_lookup_handler(),
        server_stream_make_delete_handler());

    if (!server_setup_features(ctx_http, params, mcp_mgr, tools, SERVER_ROLE_INSTANCE, agent_key)) {
        return 1;
    }

    // setup clean up function, to be called before exit
    auto clean_up = [&ctx_http, &ctx_server, &mcp_mgr]() {
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
    if (role.is_managed_by_router()) {
        ctx_server.set_state_callback([&](server_state state, json payload) {
            role.notify_to_router(server_state_to_str(state), payload);
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

    if (opts.install_signal_handlers) {
        server_install_signal_handlers();
    }

    server_connect connect_proc;
    if (!server_on_listening(ctx_http, params, connect_proc)) {
        ctx_http.stop();
        ctx_http.join();
        clean_up();
        return 1;
    }

    // optionally, notify router server that this instance is ready
    std::thread monitor_thread;
    if (role.is_managed_by_router()) {
        monitor_thread = role.setup(shutdown_handler);
        json info = routes.get_model_info();
        // the router tracks device memory from these reports
        if (const auto * ll_ctx = ctx_server.get_llama_context()) {
            info["memory"] = server_model_memory(llama_n_ctx(ll_ctx), common_memory_breakdown_rows(ll_ctx)).to_json();
        }
        role.notify_to_router(server_state_to_str(SERVER_STATE_READY), info);
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

    return 0;
}

//
// entry points
//

static int server_run(common_params & params, int argc, char ** argv, server_role & role, const server_run_opts & opts) {
    common_models_handler models_handler;

    // -hf may name a preset repo, which makes this a router: settle that before deciding the role
    if (opts.resolve_model && !params.model.hf_repo.empty()) {
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

    const server_role_type role_type = server_role_detect(params, role);

    // only an instance loads a model; the others must not touch the GPU, so skip the
    // device enumeration for them and the CUDA primary context stays uncreated
    common_params_print_info(params, role_type == SERVER_ROLE_INSTANCE);

    if (!server_prepare_params(params, role_type)) {
        return 1;
    }

    switch (role_type) {
        case SERVER_ROLE_DOWNLOAD: return server_run_download(params, role);
        case SERVER_ROLE_MEASURE:  return server_run_measure(params, role, models_handler);
        case SERVER_ROLE_ROUTER:   return server_run_router(params, argc, argv, opts);
        case SERVER_ROLE_INSTANCE: return server_run_instance(params, role, models_handler, opts);
    }

    GGML_ABORT("invalid server role");
}

int llama_server(int argc, char ** argv) {
    server_role role;

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

    const int result = server_run(params, argc, argv, role, server_run_opts{});
    common_log_flush(common_log_main());
    return result;
}

int llama_server(common_params & params, int argc, char ** argv) {
    // embedded in llama-cli: it handles the signals, and the model is resolved already
    server_run_opts opts;
    opts.install_signal_handlers = false;
    opts.resolve_model           = false;

    server_role role;
    return server_run(params, argc, argv, role, opts);
}
