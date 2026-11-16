#pragma once

#include "server-http.h"
#include "server-context.h"

#include <functional>
#include <memory>
#include <string>

struct common_params;

// kvmem + HyperCore compatibility surface for Bemly llama-server
struct server_compat {
    using route_wrapper_t = std::function<server_http_context::handler_t(server_http_context::handler_t)>;

    explicit server_compat(const common_params & params);
    ~server_compat();

    server_compat(const server_compat &) = delete;
    server_compat & operator=(const server_compat &) = delete;

    // wrap existing route handlers before they are registered
    void wrap_routes(server_routes & routes);

    // record request rows for kvmem/Strata monitors (usage/timings when present)
    server_http_context::handler_t wrap_chat_completions(server_http_context::handler_t inner);

    // register extra paths; chat_completions is the same handler used for /v1/chat/completions
    void register_routes(
            server_http_context & ctx_http,
            server_routes & routes,
            const server_http_context::handler_t & chat_completions,
            const route_wrapper_t & wrap);

    // optional: append a line to the launcher log ring
    void append_log(const std::string & line);

private:
    struct impl;
    std::unique_ptr<impl> pimpl;
};
