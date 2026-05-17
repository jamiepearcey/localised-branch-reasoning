#include <llama.h>

#include <algorithm>
#include <cctype>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace {

struct Args {
    std::string model_path;
    int32_t ctx_size = 4096;
    int32_t batch_size = 512;
    int32_t gpu_layers = 999;
    int32_t max_seqs = 16;
    uint32_t seed = 1234;
};

struct BranchInput {
    std::string label;
    std::string marker;
};

struct BranchOutput {
    std::string label;
    std::string text;
    bool stop_detected = false;
    int32_t tokens_generated = 0;
    llama_pos copied_pos_max = -1;
    llama_pos final_pos_max = -1;
};

struct PrefixCacheEntry {
    std::string id;
    llama_seq_id seq_id = -1;
    llama_pos token_count = 0;
};

[[noreturn]] void usage(const char * argv0, int code = 2) {
    std::cerr
        << "Usage: " << argv0 << " --model /path/model.gguf [options]\n\n"
        << "Options:\n"
        << "  --ctx-size N      Context tokens, default 4096\n"
        << "  --batch-size N    Decode batch size, default 512\n"
        << "  --gpu-layers N    Layers to offload to Metal/CUDA, default 999\n"
        << "  --max-seqs N      Maximum simultaneous sequences, default 16\n"
        << "  --seed N          Sampler seed, default 1234\n";
    std::exit(code);
}

Args parse_args(int argc, char ** argv) {
    Args args;
    for (int i = 1; i < argc; ++i) {
        const std::string key = argv[i];
        auto value = [&]() -> std::string {
            if (i + 1 >= argc) {
                usage(argv[0]);
            }
            return argv[++i];
        };
        if (key == "--model") {
            args.model_path = value();
        } else if (key == "--ctx-size") {
            args.ctx_size = std::stoi(value());
        } else if (key == "--batch-size") {
            args.batch_size = std::stoi(value());
        } else if (key == "--gpu-layers") {
            args.gpu_layers = std::stoi(value());
        } else if (key == "--max-seqs") {
            args.max_seqs = std::stoi(value());
        } else if (key == "--seed") {
            args.seed = static_cast<uint32_t>(std::stoul(value()));
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
    if (args.max_seqs < 2) {
        throw std::runtime_error("--max-seqs must be at least 2");
    }
    return args;
}

std::string json_escape(const std::string & value) {
    std::ostringstream out;
    for (unsigned char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\b': out << "\\b"; break;
        case '\f': out << "\\f"; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default:
            if (ch < 0x20) {
                const char * hex = "0123456789abcdef";
                out << "\\u00" << hex[ch >> 4] << hex[ch & 0xf];
            } else {
                out << static_cast<char>(ch);
            }
        }
    }
    return out.str();
}

std::string json_unescape(const std::string & value) {
    std::string out;
    out.reserve(value.size());
    for (size_t i = 0; i < value.size(); ++i) {
        const char ch = value[i];
        if (ch != '\\' || i + 1 >= value.size()) {
            out += ch;
            continue;
        }
        const char esc = value[++i];
        switch (esc) {
        case '"': out += '"'; break;
        case '\\': out += '\\'; break;
        case '/': out += '/'; break;
        case 'b': out += '\b'; break;
        case 'f': out += '\f'; break;
        case 'n': out += '\n'; break;
        case 'r': out += '\r'; break;
        case 't': out += '\t'; break;
        case 'u':
            // The Python client sends UTF-8 with ensure_ascii=false, so the
            // worker only needs to preserve rare escaped code points.
            if (i + 4 < value.size()) {
                out += '?';
                i += 4;
            }
            break;
        default:
            out += esc;
        }
    }
    return out;
}

size_t find_string_start(const std::string & json, const std::string & field) {
    const std::string needle = "\"" + field + "\"";
    const size_t key = json.find(needle);
    if (key == std::string::npos) {
        return std::string::npos;
    }
    const size_t colon = json.find(':', key + needle.size());
    if (colon == std::string::npos) {
        return std::string::npos;
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    if (pos >= json.size() || json[pos] != '"') {
        return std::string::npos;
    }
    return pos + 1;
}

std::string json_string_field(const std::string & json, const std::string & field, const std::string & fallback = "") {
    const size_t start = find_string_start(json, field);
    if (start == std::string::npos) {
        return fallback;
    }
    std::string raw;
    bool escaped = false;
    for (size_t i = start; i < json.size(); ++i) {
        const char ch = json[i];
        if (escaped) {
            raw += '\\';
            raw += ch;
            escaped = false;
            continue;
        }
        if (ch == '\\') {
            escaped = true;
            continue;
        }
        if (ch == '"') {
            return json_unescape(raw);
        }
        raw += ch;
    }
    throw std::runtime_error("unterminated JSON string field: " + field);
}

int32_t json_int_field(const std::string & json, const std::string & field, int32_t fallback) {
    const std::string needle = "\"" + field + "\"";
    const size_t key = json.find(needle);
    if (key == std::string::npos) {
        return fallback;
    }
    const size_t colon = json.find(':', key + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    size_t end = pos;
    while (end < json.size() && (std::isdigit(static_cast<unsigned char>(json[end])) || json[end] == '-')) {
        ++end;
    }
    if (end == pos) {
        return fallback;
    }
    return std::stoi(json.substr(pos, end - pos));
}

bool json_bool_field(const std::string & json, const std::string & field, bool fallback) {
    const std::string needle = "\"" + field + "\"";
    const size_t key = json.find(needle);
    if (key == std::string::npos) {
        return fallback;
    }
    const size_t colon = json.find(':', key + needle.size());
    if (colon == std::string::npos) {
        return fallback;
    }
    size_t pos = colon + 1;
    while (pos < json.size() && std::isspace(static_cast<unsigned char>(json[pos]))) {
        ++pos;
    }
    if (json.compare(pos, 4, "true") == 0) {
        return true;
    }
    if (json.compare(pos, 5, "false") == 0) {
        return false;
    }
    return fallback;
}

std::string extract_array(const std::string & json, const std::string & field) {
    const std::string needle = "\"" + field + "\"";
    const size_t key = json.find(needle);
    if (key == std::string::npos) {
        return "";
    }
    const size_t colon = json.find(':', key + needle.size());
    const size_t start = json.find('[', colon);
    if (colon == std::string::npos || start == std::string::npos) {
        return "";
    }
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    for (size_t i = start; i < json.size(); ++i) {
        const char ch = json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '[') {
            ++depth;
        } else if (ch == ']') {
            --depth;
            if (depth == 0) {
                return json.substr(start, i - start + 1);
            }
        }
    }
    throw std::runtime_error("unterminated JSON array field: " + field);
}

std::vector<std::string> object_slices(const std::string & array_json) {
    std::vector<std::string> objects;
    int depth = 0;
    bool in_string = false;
    bool escaped = false;
    size_t start = std::string::npos;
    for (size_t i = 0; i < array_json.size(); ++i) {
        const char ch = array_json[i];
        if (in_string) {
            if (escaped) {
                escaped = false;
            } else if (ch == '\\') {
                escaped = true;
            } else if (ch == '"') {
                in_string = false;
            }
            continue;
        }
        if (ch == '"') {
            in_string = true;
        } else if (ch == '{') {
            if (depth == 0) {
                start = i;
            }
            ++depth;
        } else if (ch == '}') {
            --depth;
            if (depth == 0 && start != std::string::npos) {
                objects.push_back(array_json.substr(start, i - start + 1));
                start = std::string::npos;
            }
        }
    }
    return objects;
}

std::vector<BranchInput> json_branches(const std::string & json) {
    const std::string array = extract_array(json, "branches");
    std::vector<BranchInput> branches;
    for (const std::string & object : object_slices(array)) {
        BranchInput branch;
        branch.label = json_string_field(object, "label");
        branch.marker = json_string_field(object, "marker");
        if (branch.label.empty()) {
            throw std::runtime_error("branch label is required");
        }
        if (branch.marker.empty()) {
            throw std::runtime_error("branch marker is required");
        }
        branches.push_back(branch);
    }
    return branches;
}

std::string lowercase(std::string value) {
    std::transform(value.begin(), value.end(), value.begin(), [](unsigned char ch) {
        return static_cast<char>(std::tolower(ch));
    });
    return value;
}

bool contains_stop(const std::string & text, const std::string & stop) {
    if (stop.empty()) {
        return false;
    }
    return lowercase(text).find(lowercase(stop)) != std::string::npos;
}

std::vector<llama_token> tokenize(const llama_vocab * vocab, const std::string & text, bool add_special) {
    int32_t cap = std::max<int32_t>(32, static_cast<int32_t>(text.size()) + 8);
    std::vector<llama_token> tokens(cap);
    int32_t n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                               tokens.data(), static_cast<int32_t>(tokens.size()), add_special, true);
    if (n < 0) {
        tokens.assign(static_cast<size_t>(-n), 0);
        n = llama_tokenize(vocab, text.c_str(), static_cast<int32_t>(text.size()),
                           tokens.data(), static_cast<int32_t>(tokens.size()), add_special, true);
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

llama_pos decode_text(llama_context * ctx, const llama_vocab * vocab, const std::string & text,
                      llama_seq_id seq_id, llama_pos start_pos, int32_t batch_size,
                      bool add_special, bool logits_last) {
    const auto tokens = tokenize(vocab, text, add_special);
    decode_tokens(ctx, tokens, seq_id, start_pos, batch_size, logits_last);
    return static_cast<llama_pos>(tokens.size());
}

void decode_rows(llama_context * ctx, const std::vector<llama_token> & tokens,
                 const std::vector<llama_seq_id> & seq_ids, const std::vector<llama_pos> & positions,
                 bool logits) {
    if (tokens.empty()) {
        return;
    }
    if (tokens.size() != seq_ids.size() || tokens.size() != positions.size()) {
        throw std::runtime_error("decode_rows called with mismatched vectors");
    }
    llama_batch batch = llama_batch_init(static_cast<int32_t>(tokens.size()), 0, 1);
    batch.n_tokens = static_cast<int32_t>(tokens.size());
    for (int32_t i = 0; i < batch.n_tokens; ++i) {
        batch.token[i] = tokens[static_cast<size_t>(i)];
        batch.pos[i] = positions[static_cast<size_t>(i)];
        batch.n_seq_id[i] = 1;
        batch.seq_id[i][0] = seq_ids[static_cast<size_t>(i)];
        batch.logits[i] = logits;
    }
    const int32_t rc = llama_decode(ctx, batch);
    llama_batch_free(batch);
    if (rc != 0) {
        throw std::runtime_error("llama_decode failed with code " + std::to_string(rc));
    }
}

llama_sampler * make_sampler(uint32_t seed) {
    llama_sampler * sampler = llama_sampler_chain_init(llama_sampler_chain_default_params());
    llama_sampler_chain_add(sampler, llama_sampler_init_top_k(40));
    llama_sampler_chain_add(sampler, llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(sampler, llama_sampler_init_temp(0.35f));
    llama_sampler_chain_add(sampler, llama_sampler_init_dist(seed));
    return sampler;
}

std::string generate_one(llama_context * ctx, const llama_vocab * vocab, llama_seq_id seq_id,
                         llama_pos start_pos, int32_t max_new_tokens, const std::string & stop,
                         uint32_t seed, int32_t & tokens_generated, bool & stop_detected) {
    llama_sampler * sampler = make_sampler(seed);
    std::string text;
    llama_pos pos = start_pos;
    for (int32_t i = 0; i < max_new_tokens; ++i) {
        const llama_token token = llama_sampler_sample(sampler, ctx, -1);
        llama_sampler_accept(sampler, token);
        if (llama_vocab_is_eog(vocab, token)) {
            break;
        }
        text += token_piece(vocab, token);
        ++tokens_generated;
        if (contains_stop(text, stop)) {
            stop_detected = true;
            break;
        }
        decode_tokens(ctx, std::vector<llama_token>{token}, seq_id, pos, 1, true);
        ++pos;
    }
    llama_sampler_free(sampler);
    return text;
}

PrefixCacheEntry * find_prefix_cache(std::vector<PrefixCacheEntry> & caches, const std::string & id) {
    for (PrefixCacheEntry & cache : caches) {
        if (cache.id == id) {
            return &cache;
        }
    }
    return nullptr;
}

const PrefixCacheEntry * find_prefix_cache(const std::vector<PrefixCacheEntry> & caches, const std::string & id) {
    for (const PrefixCacheEntry & cache : caches) {
        if (cache.id == id) {
            return &cache;
        }
    }
    return nullptr;
}

std::string handle_cache_prefix(llama_context * ctx, const llama_vocab * vocab, const std::string & line,
                                int32_t batch_size, size_t max_seqs,
                                std::vector<PrefixCacheEntry> & caches) {
    const std::string request_id = json_string_field(line, "request_id");
    const std::string prefix_id = json_string_field(line, "prefix_id");
    const std::string prefix = json_string_field(line, "prefix");
    if (prefix_id.empty()) {
        throw std::runtime_error("prefix_id is required");
    }
    if (prefix.empty()) {
        throw std::runtime_error("prefix is required");
    }

    PrefixCacheEntry * entry = find_prefix_cache(caches, prefix_id);
    if (entry == nullptr) {
        if (caches.size() + 2 >= max_seqs) {
            throw std::runtime_error("not enough sequence slots for another cached prefix");
        }
        PrefixCacheEntry new_entry;
        new_entry.id = prefix_id;
        new_entry.seq_id = static_cast<llama_seq_id>(caches.size());
        caches.push_back(new_entry);
        entry = &caches.back();
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, entry->seq_id, 0, -1);
    const auto tokens = tokenize(vocab, prefix, true);
    decode_tokens(ctx, tokens, entry->seq_id, 0, batch_size, false);
    entry->token_count = static_cast<llama_pos>(tokens.size());

    std::ostringstream out;
    out << "{\"ok\":true,\"cmd\":\"cache_prefix\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"prefix_id\":\"" << json_escape(prefix_id)
        << "\",\"seq_id\":" << entry->seq_id
        << ",\"prefix_tokens\":" << entry->token_count
        << "}";
    return out.str();
}

std::string handle_cached_generate(llama_context * ctx, const llama_vocab * vocab, const std::string & line,
                                   int32_t batch_size, uint32_t seed, size_t max_seqs,
                                   const std::vector<PrefixCacheEntry> & caches) {
    const std::string request_id = json_string_field(line, "request_id");
    const std::string prefix_id = json_string_field(line, "prefix_id");
    const std::string suffix = json_string_field(line, "suffix");
    const int32_t max_new_tokens = json_int_field(line, "max_new_tokens", 128);
    const std::string stop = json_string_field(line, "stop", "");

    const PrefixCacheEntry * cache = find_prefix_cache(caches, prefix_id);
    if (cache == nullptr) {
        throw std::runtime_error("unknown cached prefix: " + prefix_id);
    }
    const llama_seq_id work_seq = static_cast<llama_seq_id>(caches.size());
    if (static_cast<size_t>(work_seq) >= max_seqs) {
        throw std::runtime_error("not enough sequence slots for cached generate");
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, work_seq, 0, -1);
    llama_memory_seq_cp(mem, cache->seq_id, work_seq, 0, cache->token_count);

    const llama_pos suffix_tokens = decode_text(
        ctx, vocab, suffix, work_seq, cache->token_count, batch_size, false, true);
    if (suffix_tokens == 0) {
        throw std::runtime_error("cached generate suffix tokenized to empty sequence");
    }

    int32_t generated = 0;
    bool stop_detected = false;
    const std::string text = generate_one(
        ctx, vocab, work_seq, cache->token_count + suffix_tokens, max_new_tokens, stop, seed, generated, stop_detected);

    llama_memory_seq_rm(mem, work_seq, 0, -1);

    std::ostringstream out;
    out << "{\"ok\":true,\"cmd\":\"cached_generate\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"prefix_id\":\"" << json_escape(prefix_id)
        << "\",\"cache_prefix_tokens\":" << cache->token_count
        << ",\"suffix_tokens\":" << suffix_tokens
        << ",\"prompt_tokens\":" << (cache->token_count + suffix_tokens)
        << ",\"tokens_generated\":" << generated
        << ",\"stop_detected\":" << (stop_detected ? "true" : "false")
        << ",\"text\":\"" << json_escape(text) << "\"}";
    return out.str();
}

std::string handle_generate(llama_context * ctx, const llama_vocab * vocab, const std::string & line,
                            int32_t batch_size, uint32_t seed, size_t cache_count, size_t max_seqs) {
    const std::string request_id = json_string_field(line, "request_id");
    const std::string prompt = json_string_field(line, "prompt");
    const int32_t max_new_tokens = json_int_field(line, "max_new_tokens", 128);
    const std::string stop = json_string_field(line, "stop", "");

    const llama_seq_id work_seq = static_cast<llama_seq_id>(cache_count == 0 ? 0 : cache_count);
    if (static_cast<size_t>(work_seq) >= max_seqs) {
        throw std::runtime_error("not enough sequence slots for generate");
    }
    llama_memory_t mem = llama_get_memory(ctx);
    if (cache_count == 0) {
        llama_memory_clear(mem, true);
    } else {
        llama_memory_seq_rm(mem, work_seq, 0, -1);
    }
    const auto tokens = tokenize(vocab, prompt, true);
    decode_tokens(ctx, tokens, work_seq, 0, batch_size, true);

    int32_t generated = 0;
    bool stop_detected = false;
    const std::string text = generate_one(
        ctx, vocab, work_seq, static_cast<llama_pos>(tokens.size()), max_new_tokens, stop, seed, generated, stop_detected);
    if (cache_count != 0) {
        llama_memory_seq_rm(mem, work_seq, 0, -1);
    }

    std::ostringstream out;
    out << "{\"ok\":true,\"cmd\":\"generate\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"prompt_tokens\":" << tokens.size()
        << ",\"tokens_generated\":" << generated
        << ",\"stop_detected\":" << (stop_detected ? "true" : "false")
        << ",\"text\":\"" << json_escape(text) << "\"}";
    return out.str();
}

std::vector<BranchOutput> decode_branches_from_base(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::vector<BranchInput> & branches,
    int32_t max_new_tokens,
    const std::string & stop,
    int32_t batch_size,
    uint32_t seed,
    llama_seq_id base_seq_id,
    llama_seq_id first_branch_seq_id,
    llama_pos base_token_count,
    size_t max_seqs) {

    if (branches.empty()) {
        throw std::runtime_error("at least one branch is required");
    }
    if (static_cast<size_t>(first_branch_seq_id) + branches.size() > max_seqs) {
        throw std::runtime_error("branch count exceeds available sequence slots");
    }

    llama_memory_t mem = llama_get_memory(ctx);

    std::vector<std::vector<llama_token>> marker_tokens;
    marker_tokens.reserve(branches.size());
    std::vector<BranchOutput> outputs;
    outputs.reserve(branches.size());
    std::vector<llama_pos> next_pos(branches.size(), base_token_count);

    for (size_t i = 0; i < branches.size(); ++i) {
        const llama_seq_id seq_id = static_cast<llama_seq_id>(first_branch_seq_id + static_cast<llama_seq_id>(i));
        llama_memory_seq_rm(mem, seq_id, 0, -1);
        llama_memory_seq_cp(mem, base_seq_id, seq_id, 0, base_token_count);
        const llama_pos copied_pos_max = llama_memory_seq_pos_max(mem, seq_id);

        auto tokens = tokenize(vocab, branches[i].marker, false);
        if (tokens.empty()) {
            throw std::runtime_error("branch marker tokenized to empty sequence");
        }
        if (tokens.size() > 1) {
            std::vector<llama_token> head(tokens.begin(), tokens.end() - 1);
            decode_tokens(ctx, head, seq_id, base_token_count, batch_size, false);
        }
        next_pos[i] += static_cast<llama_pos>(tokens.size());
        marker_tokens.push_back(std::move(tokens));

        BranchOutput output;
        output.label = branches[i].label;
        output.copied_pos_max = copied_pos_max;
        outputs.push_back(output);
    }

    std::vector<llama_token> final_marker_tokens;
    std::vector<llama_seq_id> final_marker_seqs;
    std::vector<llama_pos> final_marker_positions;
    for (size_t i = 0; i < branches.size(); ++i) {
        const auto & tokens = marker_tokens[i];
        final_marker_tokens.push_back(tokens.back());
        final_marker_seqs.push_back(static_cast<llama_seq_id>(first_branch_seq_id + static_cast<llama_seq_id>(i)));
        final_marker_positions.push_back(static_cast<llama_pos>(base_token_count + tokens.size() - 1));
    }
    decode_rows(ctx, final_marker_tokens, final_marker_seqs, final_marker_positions, true);

    std::vector<llama_sampler *> samplers;
    samplers.reserve(branches.size());
    std::vector<bool> active(branches.size(), true);
    std::vector<int32_t> logits_index(branches.size(), -1);
    for (size_t i = 0; i < branches.size(); ++i) {
        samplers.push_back(make_sampler(seed + static_cast<uint32_t>(i)));
        logits_index[i] = static_cast<int32_t>(i);
    }

    for (int32_t step = 0; step < max_new_tokens; ++step) {
        std::vector<llama_token> sampled_tokens;
        std::vector<llama_seq_id> sampled_seqs;
        std::vector<llama_pos> sampled_positions;
        std::vector<size_t> sampled_branch_indexes;

        for (size_t i = 0; i < branches.size(); ++i) {
            if (!active[i]) {
                continue;
            }
            const llama_token token = llama_sampler_sample(samplers[i], ctx, logits_index[i]);
            llama_sampler_accept(samplers[i], token);
            if (llama_vocab_is_eog(vocab, token)) {
                active[i] = false;
                continue;
            }

            outputs[i].text += token_piece(vocab, token);
            ++outputs[i].tokens_generated;
            if (contains_stop(outputs[i].text, stop)) {
                outputs[i].stop_detected = true;
                active[i] = false;
                continue;
            }

            sampled_tokens.push_back(token);
            sampled_seqs.push_back(static_cast<llama_seq_id>(first_branch_seq_id + static_cast<llama_seq_id>(i)));
            sampled_positions.push_back(next_pos[i]);
            sampled_branch_indexes.push_back(i);
            ++next_pos[i];
        }

        if (sampled_tokens.empty()) {
            break;
        }

        decode_rows(ctx, sampled_tokens, sampled_seqs, sampled_positions, true);
        for (size_t row = 0; row < sampled_branch_indexes.size(); ++row) {
            logits_index[sampled_branch_indexes[row]] = static_cast<int32_t>(row);
        }
    }

    for (size_t i = 0; i < samplers.size(); ++i) {
        llama_sampler_free(samplers[i]);
        const llama_seq_id seq_id = static_cast<llama_seq_id>(first_branch_seq_id + static_cast<llama_seq_id>(i));
        outputs[i].final_pos_max = llama_memory_seq_pos_max(mem, seq_id);
    }
    return outputs;
}

std::vector<BranchOutput> handle_branch_parallel(
    llama_context * ctx,
    const llama_vocab * vocab,
    const std::string & prefix,
    const std::vector<BranchInput> & branches,
    int32_t max_new_tokens,
    const std::string & stop,
    int32_t batch_size,
    uint32_t seed,
    size_t max_seqs,
    size_t & prefix_tokens_out) {

    if (branches.size() + 1 > max_seqs) {
        throw std::runtime_error("branch count exceeds --max-seqs");
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_clear(mem, true);

    const auto prefix_tokens = tokenize(vocab, prefix, true);
    prefix_tokens_out = prefix_tokens.size();
    decode_tokens(ctx, prefix_tokens, 0, 0, batch_size, false);

    return decode_branches_from_base(
        ctx, vocab, branches, max_new_tokens, stop, batch_size, seed, 0, 1,
        static_cast<llama_pos>(prefix_tokens.size()), max_seqs);
}

std::string handle_branch(llama_context * ctx, const llama_vocab * vocab, const std::string & line,
                          int32_t batch_size, uint32_t seed, size_t max_seqs) {
    const std::string request_id = json_string_field(line, "request_id");
    const std::string prefix = json_string_field(line, "prefix");
    const auto branches = json_branches(line);
    const int32_t max_new_tokens = json_int_field(line, "max_new_tokens", 96);
    const std::string stop = json_string_field(line, "stop", "STOP_POINT:");
    const bool parallel_requested = json_bool_field(line, "parallel", true);

    size_t prefix_tokens = 0;
    const std::vector<BranchOutput> outputs = handle_branch_parallel(
        ctx, vocab, prefix, branches, max_new_tokens, stop, batch_size, seed, max_seqs, prefix_tokens);

    std::ostringstream out;
    out << "{\"ok\":true,\"cmd\":\"branch\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"parallel_requested\":" << (parallel_requested ? "true" : "false")
        << ",\"parallel_decode\":true"
        << ",\"prefix_forward_passes\":1"
        << ",\"sequence_copy_api\":\"llama_memory_seq_cp\""
        << ",\"prefix_tokens\":" << prefix_tokens
        << ",\"branches\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "{\"label\":\"" << json_escape(outputs[i].label)
            << "\",\"text\":\"" << json_escape(outputs[i].text)
            << "\",\"stop_detected\":" << (outputs[i].stop_detected ? "true" : "false")
            << ",\"tokens_generated\":" << outputs[i].tokens_generated
            << ",\"seq_after_copy_pos_max\":" << outputs[i].copied_pos_max
            << ",\"seq_final_pos_max\":" << outputs[i].final_pos_max
            << "}";
    }
    out << "]}";
    return out.str();
}

std::string handle_cached_branch(llama_context * ctx, const llama_vocab * vocab, const std::string & line,
                                 int32_t batch_size, uint32_t seed, size_t max_seqs,
                                 const std::vector<PrefixCacheEntry> & caches) {
    const std::string request_id = json_string_field(line, "request_id");
    const std::string prefix_id = json_string_field(line, "prefix_id");
    const std::string suffix = json_string_field(line, "suffix");
    const auto branches = json_branches(line);
    const int32_t max_new_tokens = json_int_field(line, "max_new_tokens", 96);
    const std::string stop = json_string_field(line, "stop", "STOP_POINT:");
    const bool parallel_requested = json_bool_field(line, "parallel", true);

    const PrefixCacheEntry * cache = find_prefix_cache(caches, prefix_id);
    if (cache == nullptr) {
        throw std::runtime_error("unknown cached prefix: " + prefix_id);
    }
    const llama_seq_id base_seq = static_cast<llama_seq_id>(caches.size());
    const llama_seq_id first_branch_seq = static_cast<llama_seq_id>(base_seq + 1);
    if (static_cast<size_t>(first_branch_seq) + branches.size() > max_seqs) {
        throw std::runtime_error("branch count exceeds available sequence slots after cached prefixes");
    }

    llama_memory_t mem = llama_get_memory(ctx);
    llama_memory_seq_rm(mem, base_seq, 0, -1);
    llama_memory_seq_cp(mem, cache->seq_id, base_seq, 0, cache->token_count);
    const llama_pos suffix_tokens = decode_text(
        ctx, vocab, suffix, base_seq, cache->token_count, batch_size, false, false);
    const llama_pos base_tokens = cache->token_count + suffix_tokens;

    const std::vector<BranchOutput> outputs = decode_branches_from_base(
        ctx, vocab, branches, max_new_tokens, stop, batch_size, seed, base_seq, first_branch_seq, base_tokens, max_seqs);

    llama_memory_seq_rm(mem, base_seq, 0, -1);

    std::ostringstream out;
    out << "{\"ok\":true,\"cmd\":\"cached_branch\",\"request_id\":\"" << json_escape(request_id)
        << "\",\"prefix_id\":\"" << json_escape(prefix_id)
        << "\",\"parallel_requested\":" << (parallel_requested ? "true" : "false")
        << ",\"parallel_decode\":true"
        << ",\"prefix_forward_passes\":0"
        << ",\"sequence_copy_api\":\"llama_memory_seq_cp\""
        << ",\"cache_prefix_tokens\":" << cache->token_count
        << ",\"suffix_tokens\":" << suffix_tokens
        << ",\"prefix_tokens\":" << base_tokens
        << ",\"branches\":[";
    for (size_t i = 0; i < outputs.size(); ++i) {
        if (i > 0) {
            out << ",";
        }
        out << "{\"label\":\"" << json_escape(outputs[i].label)
            << "\",\"text\":\"" << json_escape(outputs[i].text)
            << "\",\"stop_detected\":" << (outputs[i].stop_detected ? "true" : "false")
            << ",\"tokens_generated\":" << outputs[i].tokens_generated
            << ",\"seq_after_copy_pos_max\":" << outputs[i].copied_pos_max
            << ",\"seq_final_pos_max\":" << outputs[i].final_pos_max
            << "}";
    }
    out << "]}";
    return out.str();
}

std::string error_json(const std::string & request_id, const std::string & message) {
    return "{\"ok\":false,\"request_id\":\"" + json_escape(request_id) + "\",\"error\":\"" + json_escape(message) + "\"}";
}

} // namespace

int main(int argc, char ** argv) {
    Args args;
    try {
        args = parse_args(argc, argv);
    } catch (const std::exception & exc) {
        std::cerr << "error: " << exc.what() << "\n";
        return 2;
    }

    llama_backend_init();

    llama_model_params mparams = llama_model_default_params();
    mparams.n_gpu_layers = args.gpu_layers;

    std::cerr << "loading model: " << args.model_path << "\n";
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
    cparams.n_seq_max = static_cast<uint32_t>(args.max_seqs);
    cparams.offload_kqv = true;
    cparams.kv_unified = true;

    llama_context * ctx = llama_init_from_model(model, cparams);
    if (ctx == nullptr) {
        std::cerr << "failed to create context\n";
        llama_model_free(model);
        llama_backend_free();
        return 1;
    }

    const llama_vocab * vocab = llama_model_get_vocab(model);
    char desc[256] = {};
    llama_model_desc(model, desc, sizeof(desc));
    std::cerr << "ready: " << desc << "\n";

    std::vector<PrefixCacheEntry> prefix_caches;

    std::string line;
    while (std::getline(std::cin, line)) {
        if (line.empty()) {
            continue;
        }
        const std::string request_id = json_string_field(line, "request_id", "");
        try {
            const std::string cmd = json_string_field(line, "cmd");
            if (cmd == "status") {
                std::cout
                    << "{\"ok\":true,\"cmd\":\"status\",\"request_id\":\"" << json_escape(request_id)
                    << "\",\"model\":\"" << json_escape(desc)
                    << "\",\"ctx_size\":" << args.ctx_size
                    << ",\"max_seqs\":" << args.max_seqs
                    << ",\"kv_unified\":true"
                    << ",\"prefix_cache_count\":" << prefix_caches.size()
                    << "}\n"
                    << std::flush;
            } else if (cmd == "cache_prefix") {
                std::cout << handle_cache_prefix(
                    ctx, vocab, line, args.batch_size, static_cast<size_t>(args.max_seqs), prefix_caches)
                          << "\n" << std::flush;
            } else if (cmd == "cached_generate") {
                std::cout << handle_cached_generate(
                    ctx, vocab, line, args.batch_size, args.seed, static_cast<size_t>(args.max_seqs), prefix_caches)
                          << "\n" << std::flush;
            } else if (cmd == "cached_branch") {
                std::cout << handle_cached_branch(
                    ctx, vocab, line, args.batch_size, args.seed, static_cast<size_t>(args.max_seqs), prefix_caches)
                          << "\n" << std::flush;
            } else if (cmd == "generate") {
                std::cout << handle_generate(
                    ctx, vocab, line, args.batch_size, args.seed, prefix_caches.size(), static_cast<size_t>(args.max_seqs))
                          << "\n" << std::flush;
            } else if (cmd == "branch") {
                std::cout << handle_branch(ctx, vocab, line, args.batch_size, args.seed, static_cast<size_t>(args.max_seqs))
                          << "\n" << std::flush;
            } else if (cmd == "shutdown") {
                std::cout << "{\"ok\":true,\"cmd\":\"shutdown\",\"request_id\":\"" << json_escape(request_id) << "\"}\n"
                          << std::flush;
                break;
            } else {
                std::cout << error_json(request_id, "unknown command: " + cmd) << "\n" << std::flush;
            }
        } catch (const std::exception & exc) {
            std::cout << error_json(request_id, exc.what()) << "\n" << std::flush;
        }
    }

    llama_free(ctx);
    llama_model_free(model);
    llama_backend_free();
    return 0;
}
