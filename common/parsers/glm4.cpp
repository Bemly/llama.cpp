#include "parsers.h"

#include <algorithm>

// GLM-4-0414 writes a call as "<|assistant|>name\n{json_args}" and reads its result as an <|observation|> turn; its
// template knows neither OpenAI tool_calls nor the tool role, so both are mapped to that form before rendering
static json glm4_messages(const json & messages) {
    json out = json::array();
    for (const auto & msg : messages) {
        const auto role = msg.value("role", "");
        if (role == "tool") {
            out.push_back({{"role", "observation"}, {"content", msg.contains("content") ? msg.at("content") : json("")}});
            continue;
        }
        const bool has_calls = role == "assistant" && msg.contains("tool_calls") && msg.at("tool_calls").is_array() &&
                               !msg.at("tool_calls").empty();
        if (!has_calls) {
            out.push_back(msg);
            continue;
        }
        if (msg.contains("content") && msg.at("content").is_string() && !msg.at("content").get<std::string>().empty()) {
            out.push_back({{"role", "assistant"}, {"content", msg.at("content")}});
        }
        for (const auto & call : msg.at("tool_calls")) {
            const auto & function = call.contains("function") ? call.at("function") : call;
            const json args = function.contains("arguments") ? function.at("arguments") : json::object();
            out.push_back({{"role", "assistant"},
                           {"metadata", function.value("name", "")},
                           {"content", args.is_string() ? args.get<std::string>() : args.dump()}});
        }
    }
    return out;
}

// Regex for the first characters of a tool name: any prefix of one of `names` (sorted, sharing their first
// `depth` chars) at least `min_len` long, or a whole name shorter than that
static std::string glm4_name_prefix(const std::vector<std::string> & names, size_t depth, size_t min_len) {
    std::string alternatives;
    bool        ends_here = false;
    for (size_t i = 0; i < names.size();) {
        if (names[i].size() == depth) {
            ends_here = true;
            i++;
            continue;
        }
        const char               c = names[i][depth];
        std::vector<std::string> group;
        for (; i < names.size() && names[i].size() > depth && names[i][depth] == c; i++) {
            group.push_back(names[i]);
        }
        alternatives += (alternatives.empty() ? "" : "|") + regex_escape(std::string(1, c)) +
                        glm4_name_prefix(group, depth + 1, min_len);
    }
    if (alternatives.empty()) {
        return "";
    }
    return "(?:" + alternatives + ")" + (depth >= min_len || ends_here ? "?" : "");
}

common_chat_params common_chat_params_init_glm4(const common_chat_template &        tmpl,
                                                const autoparser::generation_params & inputs) {
    common_chat_params data;

    const std::string GEN_PROMPT = "<|assistant|>";

    const auto messages    = glm4_messages(inputs.messages);
    data.prompt            = common_chat_template_direct_apply_impl(tmpl, inputs, messages);
    data.generation_prompt = common_chat_template_generation_prompt_impl(tmpl, inputs, messages);
    data.format            = COMMON_CHAT_FORMAT_PEG_NATIVE;
    data.preserved_tokens  = { "<|assistant|>", "<|observation|>", "<|user|>", "<|system|>" };
    data.additional_stops  = { "<|observation|>", "<|user|>" };

    auto has_tools       = inputs.tools.is_array() && !inputs.tools.empty();
    auto include_grammar = has_tools && inputs.tool_choice != COMMON_CHAT_TOOL_CHOICE_NONE;

    if (inputs.has_continuation()) {
        data.generation_prompt = GEN_PROMPT + "\n" + inputs.continue_msg.render_content();
        data.prompt += data.generation_prompt;
    }

    std::vector<std::string> names;
    auto parser = build_chat_peg_parser([&](common_chat_peg_builder & p) {
        auto generation_prompt = p.literal(GEN_PROMPT);
        auto content           = p.optional(p.literal("\n")) + p.content(p.rest());

        if (!include_grammar) {
            return generation_prompt + content + p.end();
        }

        auto tool_choice = p.choice();
        foreach_function(inputs.tools, [&](const json & tool) {
            const auto & function = tool.at("function");
            std::string  name     = function.at("name");
            const auto   schema   = common_chat_tool_parameters(function);
            names.push_back(name);

            tool_choice |= p.rule("tool-" + name, p.tool(
                p.tool_open(p.tool_name(p.literal(name)) + p.literal("\n")) +
                p.tool_args(p.schema(p.json(), "tool-" + name + "-schema", schema))
            ));
        });

        // one call per turn: the model waits for its <|observation|> before the next
        auto tools = p.trigger_rule("tools", tool_choice);
        if (inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_REQUIRED) {
            return generation_prompt + tools + p.end();
        }
        // text written before a call is its own assistant block, so the call opens another <|assistant|>
        auto text_then_call = p.optional(p.literal("\n")) + p.content(p.until(GEN_PROMPT)) + p.literal(GEN_PROMPT) + tools;
        return generation_prompt + p.choice({ tools, text_then_call, content }) + p.end();
    });

    data.parser = parser.save();

    if (include_grammar) {
        data.grammar_lazy = inputs.tool_choice == COMMON_CHAT_TOOL_CHOICE_AUTO;

        data.grammar = build_grammar([&](const common_grammar_builder & builder) {
            parser.build_grammar(builder, data.grammar_lazy);
        });

        // a call starts the answer and text starts with a newline, so taking over after a name's first few
        // characters lets a begun name end only as declared, not with another case or a space
        std::sort(names.begin(), names.end());
        names.erase(std::unique(names.begin(), names.end()), names.end());
        std::string alternatives;
        for (const auto & name : names) {
            alternatives += (alternatives.empty() ? "" : "|") + regex_escape(name);
        }
        const std::string name_prefix = glm4_name_prefix(names, 0, 4);
        data.grammar_triggers = {
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN_FULL,
              "(?:" + name_prefix + "|(?:" + alternatives + ")\n[\\s\\S]*)" },
            // after text, from the first characters of the name that follows the call's <|assistant|>
            { COMMON_GRAMMAR_TRIGGER_TYPE_PATTERN, regex_escape(GEN_PROMPT) + "(" + name_prefix + ")" }
        };
    }

    return data;
}
