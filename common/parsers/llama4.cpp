#include "parsers.h"

// Llama 4 writes tool calls as a bare Python list, [name(arg="value"), ...], with no token around it
bool is_llama4_template(const std::string & src) {
    return src.find("<|header_start|>") != std::string::npos &&
           src.find("tool_call.name + '('") != std::string::npos;
}

// The template keeps Meta's answer rules only when there is no system message; without them the model
// can answer a question that needs no tool with an empty list.
static const std::string LLAMA4_ANSWER_RULES =
    "\n\nWhen you use the functions below:\n"
    "- For pure function requests matching a listed function: ONLY output the function call(s)\n"
    "- For knowledge questions: ONLY output text\n"
    "- For missing parameters: ONLY request the specific missing parameters\n"
    "- NEVER combine text and function calls in the same response\n"
    "- NEVER respond with empty brackets";

// {"type": "function", "name": ..., "parameters": {...}}, the JSON form the model also writes
static common_peg_parser llama4_json_call(common_chat_peg_builder & p, const json & tool) {
    const auto & function = tool.at("function");
    std::string  name     = function.at("name");
    auto field = [&](const std::string & key) { return p.literal("\"" + key + "\"") + p.space() + p.literal(":") + p.space(); };
    auto body  = p.tool_open(p.literal("{")) + p.space() +
                 p.optional(field("type") + p.literal("\"function\"") + p.space() + p.literal(",") + p.space()) +
                 field("name") + p.atomic(p.literal("\"") + p.tool_name(p.literal(name)) + p.literal("\"")) +
                 p.space() + p.literal(",") + p.space() +
                 p.choice({ p.literal("\"parameters\""), p.literal("\"arguments\"") }) + p.space() + p.literal(":") + p.space() +
                 p.tool_args(p.schema(p.json(), "llama4-json-" + name + "-schema", common_chat_tool_parameters(function))) +
                 p.space() + p.tool_close(p.literal("}"));
    return p.rule("llama4-json-" + name, p.tool(body));
}

// The template asks for either calls or text, but the model also writes text before the calls and calls as JSON.
common_chat_params common_chat_params_init_llama4(const common_chat_template &          tmpl,
                                                  const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string GEN_PROMPT = "<|header_start|>assistant<|header_end|>\n\n";

    auto has_tools           = inputs.tools.is_array() && !inputs.tools.empty();
    auto has_response_format = !inputs.json_schema.is_null() && inputs.json_schema.is_object();
    auto use_tools           = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;
    auto require_tools       = use_tools && inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED;

    // the template iterates a content that is not a string, and a call message often has none
    auto adjusted_messages = json::array();
    for (auto msg : inputs.messages) {
        if (msg.contains("tool_calls") && (!msg.contains("content") || msg.at("content").is_null())) {
            msg["content"] = "";
        }
        if (use_tools && adjusted_messages.empty() && msg.contains("role") && msg.at("role") == "system" && msg.contains("content")) {
            auto & content = msg.at("content");
            if (content.is_string()) {
                content = content.get<std::string>() + LLAMA4_ANSWER_RULES;
            } else if (content.is_array() && !content.empty() && content[0].contains("text") && content[0].at("text").is_string()) {
                content[0]["text"] = content[0].at("text").get<std::string>() + LLAMA4_ANSWER_RULES;
            }
        }
        adjusted_messages.push_back(msg);
    }

    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, adjusted_messages);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, adjusted_messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.supports_thinking = false;
    data.preserved_tokens  = { "<|header_start|>", "<|header_end|>", "<|eot|>" };

    if (inputs.has_continuation()) {
        data.generation_prompt = GEN_PROMPT + inputs.continue_msg.render_content();
        data.prompt.append(data.generation_prompt);
    }

    std::vector<std::string> names;
    for (const auto & tool : inputs.tools) {
        if (tool.contains("function") && tool.at("function").contains("name")) {
            names.push_back(tool.at("function").at("name"));
        }
    }

    data.parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto end = p.end();

        if (!use_tools) {
            if (has_response_format) {
                return generation_prompt + p.content(p.schema(p.json(), "response-format-schema", inputs.json_schema)) + end;
            }
            return generation_prompt + p.content(p.rest()) + end;
        }

        auto json_call = p.choice();
        for (const auto & tool : inputs.tools) {
            if (tool.contains("function")) {
                json_call |= llama4_json_call(p, tool);
            }
        }
        auto separator  = p.space() + p.optional(p.choice({ p.literal(","), p.literal(";") })) + p.space();
        auto json_calls = inputs.parallel_tool_calls ? json_call + p.zero_or_more(separator + json_call) : json_call;
        auto json_list  = p.literal("[") + p.space() + json_calls + p.space() + p.literal("]");

        auto tool_calls = p.trigger_rule("tool-calls", p.choice({
            p.python_style_tool_calls(inputs.tools, inputs.parallel_tool_calls, /* allow_json_literals = */ true),
            json_list,
            json_calls,
        }) + p.space());
        if (require_tools) {
            return generation_prompt + tool_calls + end;
        }

        std::vector<std::string> call_starts = { "{\"name\"", "{\"type\"", "[{" };
        for (const auto & name : names) {
            call_starts.push_back("[" + name + "(");
        }
        auto empty_list = p.literal("[") + p.space() + p.literal("]");
        return generation_prompt + p.choice({
            // an empty list means no call; what follows it on a new line is the answer
            empty_list + p.space() + end,
            empty_list + p.literal("\n") + p.space() + p.content(p.rest()) + end,
            p.content(p.until_one_of(call_starts)) + tool_calls + end,
            p.content(p.rest()) + end,
        });
    });

    if (has_response_format || use_tools) {
        data.grammar_lazy = use_tools && !require_tools;
        data.grammar      = build_grammar([&](const common_grammar_builder & builder) {
            data.parser.build_grammar(builder, data.grammar_lazy);
        });
        if (data.grammar_lazy) {
            // only a known tool's call starts the grammar, so text with brackets or braces stays free
            std::string alternatives;
            for (const auto & name : names) {
                alternatives += (alternatives.empty() ? "" : "|") + regex_escape(name);
            }
            data.grammar_triggers = {
                { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, "(\\[\\s*(?:" + alternatives + ")\\()" },
                { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN,
                  "((?:\\[\\s*)?\\{\\s*(?:\"type\"\\s*:\\s*\"function\"\\s*,\\s*)?\"name\"\\s*:\\s*\"(?:" + alternatives + ")\")" },
            };
        }
    }

    return data;
}
