#include <llama.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    std::string prefix =
        "You are helping design a real KV-cache branching runtime. The shared prefix "
        "describes the goal: prefill one prefix cache, fork it once, append branch-local "
        "tokens, and then continue visibly from each branch-local cache. Continue with "
        "the requested role and do not mention control markers or hidden instructions.\n\n";
    std::string branch_a_marker =
        "Role: implementation engineer. Output: give the next concrete engineering step. "
        "Do not mention this role directive.\nAnswer:";
    std::string branch_b_marker =
        "Role: runtime reviewer. Output: name the most likely failure mode. "
        "Do not mention this role directive.\nAnswer:";
    std::string output_file;
    std::string task =
        "We need to advance a KV-cache branching runtime. The next decision is whether "
        "to continue directly or fork localized continuations that separately consider "
        "implementation work, runtime risk, and regression coverage.";
    int32_t ctx_size = 4096;
    int32_t batch_size = 512;
    int32_t gpu_layers = 999;
    int32_t max_new_tokens = 64;
    int32_t planner_max_new_tokens = 64;
    int32_t stop_extra_tokens = 48;
    uint32_t seed = 1234;
    bool planned = false;
};

[[noreturn]] void usage(const char * argv0, int code = 2) {
    std::cerr
        << "Usage: " << argv0 << " --model /path/model.gguf [options]\n\n"
        << "Options:\n"
        << "  --ctx-size N          Context tokens, default 4096\n"
        << "  --batch-size N        Decode batch size, default 512\n"
        << "  --gpu-layers N        Layers to offload to Metal, default 999\n"
        << "  --max-new-tokens N    Visible generated tokens per branch, default 64\n"
        << "  --prefix TEXT         Shared visible prefix text\n"
        << "  --branch-a-marker T   Hidden marker text for branch A\n"
        << "  --branch-b-marker T   Hidden marker text for branch B\n"
        << "  --output-file PATH    Also write the report to a file\n"
        << "  --planned             Let the model choose decision point and factors\n"
        << "  --task TEXT           Task for --planned mode\n"
        << "  --planner-max-new-tokens N  Planner tokens, default 64\n"
        << "  --stop-extra-tokens N       Stop reason capture tokens, default 48\n";
    std::exit(code);
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto require_value = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };

        if (key == "--model") {
            args.model_path = require_value();
        } else if (key == "--ctx-size") {
            args.ctx_size = std::stoi(require_value());
        } else if (key == "--batch-size") {
            args.batch_size = std::stoi(require_value());
        } else if (key == "--gpu-layers") {
            args.gpu_layers = std::stoi(require_value());
        } else if (key == "--max-new-tokens") {
            args.max_new_tokens = std::stoi(require_value());
        } else if (key == "--prefix") {
            args.prefix = require_value();
        } else if (key == "--branch-a-marker") {
            args.branch_a_marker = require_value();
        } else if (key == "--branch-b-marker") {
            args.branch_b_marker = require_value();
        } else if (key == "--output-file") {
            args.output_file = require_value();
        } else if (key == "--planned") {
            args.planned = true;
        } else if (key == "--task") {
            args.task = require_value();
        } else if (key == "--planner-max-new-tokens") {
            args.planner_max_new_tokens = std::stoi(require_value());
        } else if (key == "--stop-extra-tokens") {
            args.stop_extra_tokens = std::stoi(require_value());
        } else if (key == "--help" || key == "-h") {
            usage(argv[0], 0);
        } else {
            std::cerr << "Unknown argument: " << key << "\n";
            usage(argv[0]);
        }
    }

    if (args.model_path.empty()) {
        usage(argv[0]);
    }
    return args;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t cap = std::max<int32_t>(32, static_cast<int32_t>(text.size()) + 8);
    std::vector<llama_token> tokens(cap);
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()),
                               add_special, true);
    if (n < 0) {
        tokens.assign(static_cast<size_t>(-n), 0);
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()),
                           add_special, true);
    }
    if (n < 0) {
        throw std::runtime_error("tokenization failed");
    }
    tokens.resize(static_cast<size_t>(n));
    return tokens;
}

std::string token_piece(const llama_vocab * vocab, llama_token token) {
    std::vector<char> buf(64);
    int32_t n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, false);
    if (n < 0) {
        buf.assign(static_cast<size_t>(-n), 0);
        n = llama_token_to_piece(vocab, token, buf.data(), static_cast<int32_t>(buf.size()), 0, false);
    }
    if (n < 0) {
        return "";
    }
    return std::string(buf.data(), static_cast<size_t>(n));
}

void decode_tokens(llama_context * ctx, const std::vector<llama_token> & tokens, llama_seq_id seq_id,
                   llama_pos start_pos, int32_t batch_size, bool logits_last) {
    if (tokens.empty()) {
        return;
    }

    for (size_t offset = 0; offset < tokens.size(); offset += static_cast<size_t>(batch_size)) {
        const int32_t n = static_cast<int32_t>(
            std::min<size_t>(static_cast<size_t>(batch_size), tokens.size() - offset));
        llama_batch batch = llama_batch_init(n, 0, 1);
        batch.n_tokens = n;

        for (int32_t i = 0; i < n; ++i) {
            const size_t token_index = offset + static_cast<size_t>(i);
            batch.token[i] = tokens[token_index];
            batch.pos[i] = start_pos + static_cast<llama_pos>(token_index);
            batch.n_seq_id[i] = 1;
            batch.seq_id[i][0] = seq_id;
            batch.logits[i] = logits_last && token_index + 1 == tokens.size();
        }

        const int32_t rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
        }
    }
}

llama_token sample_next(llama_sampler * sampler, llama_context * ctx) {
    llama_token token = llama_sampler_sample(sampler, ctx, -1);
    llama_sampler_accept(sampler, token);
    return token;
}

llama_sampler * make_greedy_sampler() {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_greedy());
    return sampler;
}

llama_sampler * make_sample_sampler(uint32_t seed) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.65f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    return sampler;
}

std::string generate_visible(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq_id,
                             llama_pos start_pos, int32_t max_new_tokens, uint32_t seed) {
    llama_sampler * sampler = make_sample_sampler(seed);

    std::string text;
    llama_pos pos = start_pos;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const llama_token token = sample_next(sampler, ctx);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        text += token_piece(vocab, token);
        const std::vector<llama_token> one{token};
        decode_tokens(ctx, one, seq_id, pos, 1, true);
        ++pos;
    }

    llama_sampler_free(sampler);
    return text;
}

struct GeneratedText {
    std::string text;
    bool stop_detected = false;
    std::string stop_point;
};

std::string lowercase(std::string value);

std::string trim(const std::string & value) {
    size_t start = 0;
    while (start < value.size() && std::isspace(static_cast<unsigned char>(value[start]))) {
        ++start;
    }
    size_t end = value.size();
    while (end > start && std::isspace(static_cast<unsigned char>(value[end - 1]))) {
        --end;
    }
    return value.substr(start, end - start);
}

std::vector<std::string> split(const std::string & value, char delim) {
    std::vector<std::string> parts;
    std::stringstream stream(value);
    std::string item;
    while (std::getline(stream, item, delim)) {
        parts.push_back(trim(item));
    }
    return parts;
}

std::string extract_stop_point(const std::string & text) {
    const std::string stop = "stop_point:";
    const std::string lowered = lowercase(text);
    const size_t pos = lowered.find(stop);
    if (pos == std::string::npos) {
        return "";
    }
    std::string after = text.substr(pos + stop.size());
    const size_t newline = after.find('\n');
    if (newline != std::string::npos) {
        after = after.substr(0, newline);
    }
    return trim(after);
}

std::string strip_empty_think_blocks(std::string text) {
    const std::string open = "<think>";
    const std::string close = "</think>";
    size_t start = 0;
    while ((start = lowercase(text).find(open, start)) != std::string::npos) {
        const size_t body_start = start + open.size();
        const size_t end = lowercase(text).find(close, body_start);
        if (end == std::string::npos) {
            break;
        }
        const std::string body = text.substr(body_start, end - body_start);
        if (!trim(body).empty()) {
            start = end + close.size();
            continue;
        }
        size_t erase_end = end + close.size();
        while (erase_end < text.size() && std::isspace(static_cast<unsigned char>(text[erase_end]))) {
            ++erase_end;
        }
        text.erase(start, erase_end - start);
    }
    return text;
}

GeneratedText generate_greedy_until_stop(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq_id,
                                         llama_pos start_pos, int32_t max_new_tokens,
                                         const std::string & stop_phrase, int32_t stop_extra_tokens) {
    llama_sampler * sampler = make_greedy_sampler();

    GeneratedText generated;
    llama_pos pos = start_pos;
    int32_t stop_seen_at = -1;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const llama_token token = sample_next(sampler, ctx);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }

        generated.text += token_piece(vocab, token);
        const std::vector<llama_token> one{token};
        decode_tokens(ctx, one, seq_id, pos, 1, true);
        ++pos;

        const std::string lowered = lowercase(generated.text);
        const std::string stop_lower = lowercase(stop_phrase);
        if (stop_seen_at < 0 && lowered.find(stop_lower) != std::string::npos) {
            stop_seen_at = i + 1;
        }
        if (stop_seen_at >= 0) {
            generated.stop_point = extract_stop_point(generated.text);
            const int32_t extra = i + 1 - stop_seen_at;
            const std::string after = lowered.substr(lowered.find(stop_lower) + stop_lower.size());
            const bool newline_after_stop = after.find('\n') != std::string::npos;
            const bool sentence_end = !generated.stop_point.empty()
                && (generated.stop_point.back() == '.' || generated.stop_point.back() == '!' || generated.stop_point.back() == '?')
                && extra >= 3;
            if (newline_after_stop || sentence_end || extra >= stop_extra_tokens) {
                generated.stop_detected = !generated.stop_point.empty();
                break;
            }
        }
    }

    llama_sampler_free(sampler);
    if (!generated.stop_point.empty()) {
        generated.stop_detected = true;
    }
    generated.text = strip_empty_think_blocks(generated.text);
    generated.stop_point = extract_stop_point(generated.text);
    generated.stop_detected = !generated.stop_point.empty();
    return generated;
}

std::string generate_greedy(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq_id,
                            llama_pos start_pos, int32_t max_new_tokens) {
    llama_sampler * sampler = make_greedy_sampler();
    std::string text;
    llama_pos pos = start_pos;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const llama_token token = sample_next(sampler, ctx);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        text += token_piece(vocab, token);
        const std::vector<llama_token> one{token};
        decode_tokens(ctx, one, seq_id, pos, 1, true);
        ++pos;
    }
    llama_sampler_free(sampler);
    return text;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

std::vector<std::string> visible_control_leaks(const std::string & text) {
    static const std::vector<std::string> phrases = {
        "hidden branch control",
        "branch control marker",
        "implementation branch",
        "runtime reviewer branch",
        "answer as",
        "role directive",
        "control marker",
        "hidden instruction",
        "decision_point",
        "factors:",
        "selected factors",
        "branch marker",
        "do not repeat these instructions",
        "short reason this branch is complete",
        "implementation factor",
        "runtime-review factor",
        "regression-test factor",
        "factor has enough output",
        "no preamble",
        "no analysis",
        "instruction text",
    };

    const std::string haystack = lowercase(text);
    std::vector<std::string> leaks;
    for (const std::string & phrase : phrases) {
        if (haystack.find(phrase) != std::string::npos) {
            leaks.push_back(phrase);
        }
    }
    return leaks;
}

std::string field_value(const std::string & raw, const std::string & field) {
    std::stringstream stream(raw);
    std::string line;
    const std::string field_lower = lowercase(field) + ":";
    while (std::getline(stream, line)) {
        const std::string trimmed = trim(line);
        const std::string lowered = lowercase(trimmed);
        if (lowered.rfind(field_lower, 0) == 0) {
            return trim(trimmed.substr(field_lower.size()));
        }
    }
    return "";
}

struct FactorSpec {
    std::string label;
    std::string title;
    std::string marker;
};

const std::vector<FactorSpec> & factor_catalog() {
    static const std::vector<FactorSpec> catalog = {
        {
            "build",
            "Implementation engineer",
            "<|im_start|>user\n"
            "Focus only on implementation work. Output exactly two lines. "
            "Line 1: one concrete next implementation step. "
            "Line 2: STOP_POINT: followed by why the answer is complete. "
            "No preamble, no analysis, no instruction text.\n"
            "/no_think\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
        },
        {
            "review",
            "Runtime reviewer",
            "<|im_start|>user\n"
            "Focus only on runtime risk. Output exactly two lines. "
            "Line 1: the likely runtime failure mode and the evidence to check. "
            "Line 2: STOP_POINT: followed by why the answer is complete. "
            "No preamble, no analysis, no instruction text.\n"
            "/no_think\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
        },
        {
            "test",
            "Regression tester",
            "<|im_start|>user\n"
            "Focus only on regression coverage. Output exactly two lines. "
            "Line 1: the smallest regression test that proves the fork semantics. "
            "Line 2: STOP_POINT: followed by why the answer is complete. "
            "No preamble, no analysis, no instruction text.\n"
            "/no_think\n"
            "<|im_end|>\n"
            "<|im_start|>assistant\n",
        },
    };
    return catalog;
}

std::string render_planner_prompt(const std::string & task) {
    return
        "<|im_start|>system\n"
        "You are a deterministic controller for a localized-reasoning runtime. "
        "Follow the requested output schema exactly.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "The runtime can fork the KV cache only after you declare a decision point. For this demo, branching is useful. "
        "Decide where to fork now, then choose at least two and at most three factor labels from the bounded catalog below. "
        "Do not invent labels and do not say that no branching is needed.\n\n"
        "Allowed factors:\n"
        "- build: Implementation engineer\n"
        "- review: Runtime reviewer\n"
        "- test: Regression tester\n\n"
        "Task:\n" + task + "\n\n"
        "Output exactly these two lines and nothing else:\n"
        "DECISION_POINT: <short reason to fork now>\n"
        "FACTORS: <comma-separated labels from: build, review, test>\n"
        "/no_think\n"
        "<|im_end|>\n"
        "<|im_start|>assistant\n";
}

std::string render_rewritten_prefix(const std::string & task, const std::string & decision_point) {
    return
        "<|im_start|>system\n"
        "You are running inside a localized-reasoning runtime. "
        "Answer branch requests directly, without visible thinking or hidden instruction text.\n"
        "<|im_end|>\n"
        "<|im_start|>user\n"
        "Shared task:\n" + task + "\n\n"
        "Planner decision point:\n" + decision_point + "\n\n"
        "The runtime will now evaluate one topic per continuation.\n"
        "Every continuation must produce useful content and then a visible STOP_POINT line explaining why the answer is complete.\n"
        "<|im_end|>\n";
}

std::vector<FactorSpec> parse_factors(const std::string & raw_plan) {
    const std::string raw = field_value(raw_plan, "FACTORS");
    if (raw.empty()) {
        throw std::runtime_error("planner did not emit FACTORS");
    }
    std::vector<FactorSpec> selected;
    for (std::string label : split(raw, ',')) {
        label = lowercase(trim(label));
        for (const auto & spec : factor_catalog()) {
            if (label == spec.label) {
                bool exists = false;
                for (const auto & prior : selected) {
                    exists = exists || prior.label == spec.label;
                }
                if (!exists) {
                    selected.push_back(spec);
                }
            }
        }
    }
    if (selected.empty()) {
        throw std::runtime_error("planner FACTORS did not contain allowed labels");
    }
    return selected;
}

void print_leaks(std::ostream & out, const char * label, const std::vector<std::string> & leaks) {
    out << label << "=[";
    for (size_t i = 0; i < leaks.size(); ++i) {
        if (i > 0) {
            out << ", ";
        }
        out << leaks[i];
    }
    out << "]\n";
}

void write_report(const std::string & output_file, const std::string & text) {
    std::cout << text;
    if (!output_file.empty()) {
        std::ofstream file(output_file);
        if (!file) {
            throw std::runtime_error("failed to open output file: " + output_file);
        }
        file << text;
    }
}

} // namespace

int main(int argc, char ** argv) {
    const Args args = parse_args(argc, argv);

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.gpu_layers;

    llama_model * model = llama_model_load_from_file(args.model_path.c_str(), mparams);
    if (model == nullptr) {
        std::cerr << "failed to load model: " << args.model_path << "\n";
        llama_backend_free();
        return 1;
    }

    llama_context_params cparams = llama_context_default_params();
    cparams.n_ctx = static_cast<uint32_t>(args.ctx_size);
    cparams.n_batch = static_cast<uint32_t>(args.batch_size);
    cparams.n_ubatch = static_cast<uint32_t>(std::min(args.batch_size, 512));
    cparams.n_seq_max = 8;
    cparams.offload_kqv = true;
    cparams.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    try {
        const llama_vocab * vocab = llama_model_get_vocab(model);
        llama_memory_t mem = llama_get_memory(ctx);

        if (args.planned) {
            const std::string planner_prompt = render_planner_prompt(args.task);
            const auto planner_tokens = tokenize(vocab, planner_prompt, true);
            if (static_cast<int32_t>(planner_tokens.size()) + args.planner_max_new_tokens >= args.ctx_size) {
                throw std::runtime_error("ctx-size too small for planner prompt");
            }
            decode_tokens(ctx, planner_tokens, 0, 0, args.batch_size, true);
            const std::string raw_plan = generate_greedy(
                ctx,
                vocab,
                0,
                static_cast<llama_pos>(planner_tokens.size()),
                args.planner_max_new_tokens);
            const std::string decision_point = field_value(raw_plan, "DECISION_POINT");
            if (decision_point.empty()) {
                std::ostringstream report;
                report << "backend=llama.cpp\n";
                report << "mode=planned\n";
                report << "status=planner_parse_failed\n";
                report << "parse_error=planner did not emit DECISION_POINT\n";
                report << "=== planner_prompt ===\n" << planner_prompt;
                report << "=== raw_planner_output ===\n" << raw_plan << "\n";
                write_report(args.output_file, report.str());
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 4;
            }
            std::vector<FactorSpec> selected_factors;
            try {
                selected_factors = parse_factors(raw_plan);
            } catch (const std::exception & exc) {
                std::ostringstream report;
                report << "backend=llama.cpp\n";
                report << "mode=planned\n";
                report << "status=planner_parse_failed\n";
                report << "parse_error=" << exc.what() << "\n";
                report << "=== planner_prompt ===\n" << planner_prompt;
                report << "=== raw_planner_output ===\n" << raw_plan << "\n";
                write_report(args.output_file, report.str());
                llama_free(ctx);
                llama_model_free(model);
                llama_backend_free();
                return 4;
            }

            llama_memory_clear(mem, true);

            const std::string rewritten_prefix = render_rewritten_prefix(args.task, decision_point);
            const auto prefix_tokens = tokenize(vocab, rewritten_prefix, true);
            const llama_pos prefix_len = static_cast<llama_pos>(prefix_tokens.size());
            decode_tokens(ctx, prefix_tokens, 0, 0, args.batch_size, true);
            const llama_pos prefix_min = llama_memory_seq_pos_min(mem, 0);
            const llama_pos prefix_max = llama_memory_seq_pos_max(mem, 0);

            struct PlannedBranchResult {
                FactorSpec factor;
                int32_t marker_tokens = 0;
                llama_pos copied_pos_max = -1;
                llama_pos final_pos_max = -1;
                GeneratedText generated;
                std::vector<std::string> leaks;
            };

            std::vector<PlannedBranchResult> branch_results;
            for (size_t i = 0; i < selected_factors.size(); ++i) {
                const llama_seq_id seq_id = static_cast<llama_seq_id>(i + 1);
                llama_memory_seq_cp(mem, 0, seq_id, 0, prefix_len);
                const llama_pos copied_pos_max = llama_memory_seq_pos_max(mem, seq_id);
                const auto marker_tokens = tokenize(vocab, selected_factors[i].marker, false);
                if (prefix_len + marker_tokens.size() + args.max_new_tokens >= static_cast<size_t>(args.ctx_size)) {
                    throw std::runtime_error("ctx-size too small for planned branch");
                }
                decode_tokens(ctx, marker_tokens, seq_id, prefix_len, args.batch_size, true);
                GeneratedText generated = generate_greedy_until_stop(
                    ctx,
                    vocab,
                    seq_id,
                    prefix_len + static_cast<llama_pos>(marker_tokens.size()),
                    args.max_new_tokens,
                    "STOP_POINT:",
                    args.stop_extra_tokens);
                branch_results.push_back(
                    PlannedBranchResult{
                        selected_factors[i],
                        static_cast<int32_t>(marker_tokens.size()),
                        copied_pos_max,
                        llama_memory_seq_pos_max(mem, seq_id),
                        generated,
                        visible_control_leaks(generated.text),
                    });
            }

            char desc[256] = {};
            llama_model_desc(model, desc, sizeof(desc));

            std::ostringstream report;
            report << "backend=llama.cpp\n";
            report << "mode=planned\n";
            report << "model=" << desc << "\n";
            report << "model_params=" << llama_model_n_params(model) << "\n";
            report << "model_size_bytes=" << llama_model_size(model) << "\n";
            report << "planner_selected_factors=[";
            for (size_t i = 0; i < selected_factors.size(); ++i) {
                if (i > 0) {
                    report << ", ";
                }
                report << selected_factors[i].label;
            }
            report << "]\n";
            report << "parallel_continuation_count=" << selected_factors.size() << "\n";
            report << "=== planner_prompt ===\n" << planner_prompt;
            report << "=== raw_planner_output ===\n" << raw_plan << "\n";
            report << "=== rewritten_shared_prefix ===\n" << rewritten_prefix;
            report << "=== rewritten_branch_markers ===\n";
            for (const auto & factor : selected_factors) {
                report << "## branch_marker=" << factor.label << "\n" << factor.marker << "\n";
            }
            report << "=== kv_fork_result ===\n";
            report << "prefix_forward_passes=1\n";
            report << "sequence_copy_api=llama_memory_seq_cp\n";
            report << "prefix_tokens=" << prefix_tokens.size() << "\n";
            report << "seq0_prefix_pos_range=[" << prefix_min << ", " << prefix_max << "]\n";
            report << "seq0_final_pos_max=" << llama_memory_seq_pos_max(mem, 0) << "\n";
            bool ok = prefix_min == 0 && prefix_max == prefix_len - 1;
            for (const auto & branch : branch_results) {
                report << "## branch=" << branch.factor.label << "\n";
                report << "marker_token_count=" << branch.marker_tokens << "\n";
                report << "seq_after_copy_pos_max=" << branch.copied_pos_max << "\n";
                report << "seq_final_pos_max=" << branch.final_pos_max << "\n";
                report << "model_stop_detected=" << (branch.generated.stop_detected ? "true" : "false") << "\n";
                report << "model_stop_point='" << branch.generated.stop_point << "'\n";
                print_leaks(report, "visible_control_leaks", branch.leaks);
                report << "visible_text='" << branch.generated.text << "'\n";
                ok = ok && branch.copied_pos_max == prefix_len - 1;
                ok = ok && branch.generated.stop_detected;
                ok = ok && branch.leaks.empty();
            }

            write_report(args.output_file, report.str());

            llama_free(ctx);
            llama_model_free(model);
            llama_backend_free();
            return ok ? 0 : 3;
        }

        const auto prefix_tokens = tokenize(vocab, args.prefix, true);
        const auto marker_a_tokens = tokenize(vocab, args.branch_a_marker, false);
        const auto marker_b_tokens = tokenize(vocab, args.branch_b_marker, false);

        const llama_pos prefix_len = static_cast<llama_pos>(prefix_tokens.size());
        if (prefix_len + marker_a_tokens.size() + args.max_new_tokens >= args.ctx_size ||
            prefix_len + marker_b_tokens.size() + args.max_new_tokens >= args.ctx_size) {
            throw std::runtime_error("ctx-size too small for prefix, markers, and generated tokens");
        }

        decode_tokens(ctx, prefix_tokens, 0, 0, args.batch_size, true);

        const llama_pos prefix_min = llama_memory_seq_pos_min(mem, 0);
        const llama_pos prefix_max = llama_memory_seq_pos_max(mem, 0);

        llama_memory_seq_cp(mem, 0, 1, 0, prefix_len);
        llama_memory_seq_cp(mem, 0, 2, 0, prefix_len);

        const llama_pos branch_a_copied_max = llama_memory_seq_pos_max(mem, 1);
        const llama_pos branch_b_copied_max = llama_memory_seq_pos_max(mem, 2);

        decode_tokens(ctx, marker_a_tokens, 1, prefix_len, args.batch_size, true);
        const std::string branch_a_text = generate_visible(
            ctx, vocab, 1, prefix_len + static_cast<llama_pos>(marker_a_tokens.size()),
            args.max_new_tokens, args.seed);

        decode_tokens(ctx, marker_b_tokens, 2, prefix_len, args.batch_size, true);
        const std::string branch_b_text = generate_visible(
            ctx, vocab, 2, prefix_len + static_cast<llama_pos>(marker_b_tokens.size()),
            args.max_new_tokens, args.seed + 1);

        const auto branch_a_leaks = visible_control_leaks(branch_a_text);
        const auto branch_b_leaks = visible_control_leaks(branch_b_text);

        char desc[256] = {};
        llama_model_desc(model, desc, sizeof(desc));

        std::ostringstream report;
        report << "backend=llama.cpp\n";
        report << "model=" << desc << "\n";
        report << "model_params=" << llama_model_n_params(model) << "\n";
        report << "model_size_bytes=" << llama_model_size(model) << "\n";
        report << "prefix_forward_passes=1\n";
        report << "sequence_copy_api=llama_memory_seq_cp\n";
        report << "prefix_tokens=" << prefix_tokens.size() << "\n";
        report << "branch_a_hidden_marker_tokens=" << marker_a_tokens.size() << "\n";
        report << "branch_b_hidden_marker_tokens=" << marker_b_tokens.size() << "\n";
        report << "seq0_prefix_pos_range=[" << prefix_min << ", " << prefix_max << "]\n";
        report << "seq1_after_copy_pos_max=" << branch_a_copied_max << "\n";
        report << "seq2_after_copy_pos_max=" << branch_b_copied_max << "\n";
        report << "seq0_final_pos_max=" << llama_memory_seq_pos_max(mem, 0) << "\n";
        report << "seq1_final_pos_max=" << llama_memory_seq_pos_max(mem, 1) << "\n";
        report << "seq2_final_pos_max=" << llama_memory_seq_pos_max(mem, 2) << "\n";
        print_leaks(report, "branch_a_visible_control_leaks", branch_a_leaks);
        print_leaks(report, "branch_b_visible_control_leaks", branch_b_leaks);
        report << "\n=== branch_a_visible ===\n" << branch_a_text << "\n";
        report << "\n=== branch_b_visible ===\n" << branch_b_text << "\n";

        write_report(args.output_file, report.str());

        bool ok = true;
        ok = ok && prefix_min == 0;
        ok = ok && prefix_max == prefix_len - 1;
        ok = ok && branch_a_copied_max == prefix_len - 1;
        ok = ok && branch_b_copied_max == prefix_len - 1;
        ok = ok && branch_a_text != branch_b_text;
        ok = ok && branch_a_leaks.empty() && branch_b_leaks.empty();

        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return ok ? 0 : 3;
    } catch (const std::exception & exc) {
        std::cerr << "error: " << exc.what() << "\n";
        llama_free(ctx);
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }
}
