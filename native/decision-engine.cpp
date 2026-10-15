// Vendored from thecodacus/llama.cpp, branch parallel-decision, commit 14d04e755
// (tools/parallel-decision/decision-engine.{h,cpp}; MIT, Copyright (c) 2023-2026
// The ggml authors), until llama.cpp takes it (ggml-org/llama.cpp#29752).
// Kept here: the engine lev_decision.cpp drives. Left out: the schema compiler,
// the prompt renderer and assemble, the fork's CLI and server use them.
// Changed: common_batch_add / common_batch_clear, which left llama.cpp's
// common/ after the fork's base, are local (de_batch_add / de_batch_clear).

#include "decision-engine.h"

#include "common.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace llama_decision {

namespace {

void de_batch_clear(llama_batch & batch) { batch.n_tokens = 0; }

void de_batch_add(llama_batch & batch, llama_token id, llama_pos pos,
                  const std::vector<llama_seq_id> & seq_ids, bool logits) {
    batch.token   [batch.n_tokens] = id;
    batch.pos     [batch.n_tokens] = pos;
    batch.n_seq_id[batch.n_tokens] = (int32_t) seq_ids.size();
    for (size_t i = 0; i < seq_ids.size(); ++i) {
        batch.seq_id[batch.n_tokens][i] = seq_ids[i];
    }
    batch.logits  [batch.n_tokens] = logits;
    batch.n_tokens++;
}

} // namespace

namespace {

double ms_since(std::chrono::steady_clock::time_point t) {
    return std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t).count();
}

struct decision_field {
    tokens_t              suffix;
    std::vector<tokens_t> paths;

    std::vector<int> active;  // greedy walk
    tokens_t         chosen;

    std::vector<tokens_t> node_prefix;   // tree: trie nodes with more than one allowed next token
    std::vector<tokens_t> node_options;
    std::vector<float>    probs;

    bool  use_tree     = false;
    int   winner       = -1;
    float path_score   = 1.0f;
    int   scored_nodes = 0;

    decision_field(tokens_t s, std::vector<tokens_t> p) : suffix(std::move(s)), paths(std::move(p)) {
        for (int i = 0; i < (int) paths.size(); ++i) {
            active.push_back(i);
        }
    }

    // Greedy: follow single-path edges for free; return the options at the next divergence.
    tokens_t options() {
        while (active.size() > 1) {
            const size_t depth = chosen.size();
            tokens_t opts;
            for (int i : active) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                return opts;
            }
            chosen.push_back(opts[0]);
        }
        winner = active.empty() ? -1 : active[0];
        return {};
    }

    void select(llama_token tok, float p) {
        const size_t depth = chosen.size();
        std::vector<int> remaining;
        for (int i : active) {
            if (paths[i][depth] == tok) {
                remaining.push_back(i);
            }
        }
        active = remaining;
        chosen.push_back(tok);
        path_score *= p;
        scored_nodes += 1;
    }

    void build_nodes() {
        std::vector<std::pair<std::vector<int>, size_t>> stack;
        std::vector<int> root(paths.size());
        for (int i = 0; i < (int) paths.size(); ++i) {
            root[i] = i;
        }
        stack.push_back({ root, 0 });
        while (!stack.empty()) {
            auto [act, depth] = stack.back();
            stack.pop_back();
            if (act.size() <= 1) {
                continue;
            }
            tokens_t opts;
            for (int i : act) {
                if (depth >= paths[i].size()) {
                    throw std::runtime_error("candidate paths must be distinct and terminated");
                }
                if (std::find(opts.begin(), opts.end(), paths[i][depth]) == opts.end()) {
                    opts.push_back(paths[i][depth]);
                }
            }
            if (opts.size() > 1) {
                node_prefix.emplace_back(paths[act[0]].begin(), paths[act[0]].begin() + depth);
                node_options.push_back(opts);
            }
            for (llama_token tok : opts) {
                std::vector<int> sub;
                for (int i : act) {
                    if (paths[i][depth] == tok) {
                        sub.push_back(i);
                    }
                }
                stack.push_back({ sub, depth + 1 });
            }
        }
    }

    int tree_rows() const {
        int n = 0;
        for (const auto & p : node_prefix) {
            n += (int) (suffix.size() + p.size());
        }
        return n;
    }

    // Exact constrained distribution: log-softmax at each node over its allowed tokens,
    // summed along every candidate path, normalised over candidates.
    void finish_tree(const std::vector<std::vector<float>> & node_scores) {
        std::vector<std::vector<float>> node_logp;
        for (const auto & s : node_scores) {
            const float mx = *std::max_element(s.begin(), s.end());
            double z = 0;
            for (float x : s) {
                z += std::exp(x - mx);
            }
            const float lz = mx + (float) std::log(z);
            std::vector<float> lp;
            for (float x : s) {
                lp.push_back(x - lz);
            }
            node_logp.push_back(lp);
        }
        std::vector<float> path_lp(paths.size(), 0.0f);
        for (size_t i = 0; i < paths.size(); ++i) {
            for (size_t n = 0; n < node_prefix.size(); ++n) {
                const auto & pre = node_prefix[n];
                if (pre.size() >= paths[i].size() || !std::equal(pre.begin(), pre.end(), paths[i].begin())) {
                    continue;
                }
                const auto & opts = node_options[n];
                const auto   it   = std::find(opts.begin(), opts.end(), paths[i][pre.size()]);
                if (it == opts.end()) {
                    throw std::runtime_error("candidate token missing from its trie node");
                }
                path_lp[i] += node_logp[n][it - opts.begin()];
            }
        }
        const int best = (int) (std::max_element(path_lp.begin(), path_lp.end()) - path_lp.begin());
        double z = 0;
        for (float x : path_lp) {
            z += std::exp(x - path_lp[best]);
        }
        probs.clear();
        for (float x : path_lp) {
            probs.push_back((float) (std::exp(x - path_lp[best]) / z));
        }
        winner       = best;
        path_score   = probs[best];
        scored_nodes = (int) node_prefix.size();
    }
};

} // namespace

// ---------------------------------------------------------------- engine

engine::engine(llama_context * ctx, llama_seq_id seq_base, int n_seqs)
    : ctx(ctx), vocab(llama_model_get_vocab(llama_get_model(ctx))), mem(llama_get_memory(ctx)),
      seq_snap(seq_base), seq_pool(seq_base + 1), n_pool(n_seqs - 1) {
    if (n_seqs < 3) {
        throw std::invalid_argument("a decision engine needs at least 3 sequences");
    }
}

tokens_t engine::tokenize(const std::string & text, bool add_special) const {
    tokens_t toks = common_tokenize(vocab, text, add_special, /*parse_special=*/ true);
    // a chat template may already start with the BOS text; keep a single BOS
    const llama_token bos = llama_vocab_bos(vocab);
    if (toks.size() >= 2 && toks[0] == bos && toks[1] == bos) {
        toks.erase(toks.begin());
    }
    return toks;
}

// Decode several prompts, each on its own sequence, packed into as few batches as n_batch allows.
void engine::decode_parts(const std::vector<prompt_part> & parts) {
    const int n_batch = (int) llama_n_batch(ctx);
    llama_batch batch = llama_batch_init(n_batch, 0, 1);
    auto flush = [&]() {
        const int rc = batch.n_tokens > 0 ? llama_decode(ctx, batch) : 0;
        de_batch_clear(batch);
        if (rc != 0) {
            llama_batch_free(batch);
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision prompt"
                                             : "llama_decode failed on the decision prompt (" + std::to_string(rc) + ")");
        }
    };
    for (const auto & p : parts) {
        for (size_t i = 0; i < p.toks->size(); ++i) {
            if (batch.n_tokens == n_batch) {
                flush();
            }
            de_batch_add(batch, (*p.toks)[i], p.pos0 + (llama_pos) i, { p.seq }, false);
        }
    }
    flush();
    llama_batch_free(batch);
}

// Restore (or build) the cached static prefix on seq_snap. Only this engine's own sequences are
// touched, so it can share a context with other users (e.g. server slots).
bool engine::prepare_prefix(const tokens_t & shared, bool allow_cache) {
    if (allow_cache && !shared.empty() && shared == cached &&
        llama_memory_seq_pos_max(mem, seq_snap) == (llama_pos) cached.size() - 1) {
        return true;
    }
    for (llama_seq_id s = seq_snap; s < seq_pool + n_pool; ++s) {
        llama_memory_seq_rm(mem, s, -1, -1);
    }
    cached.clear();
    if (!shared.empty()) {
        decode_parts({ { &shared, 0, seq_snap } });
        cached = shared;
    }
    return false;
}

// Score each branch as its own sequence forked from its trunk; return each branch's last-token
// logits restricted to its candidate tokens. Groups are bounded by free sequences and batch rows.
std::vector<std::vector<float>> engine::score_branches(const std::vector<branch> & branches, llama_seq_id first, int n_free) {
    std::vector<std::vector<float>> result(branches.size());
    const int max_rows = (int) llama_n_batch(ctx);
    size_t start = 0;
    while (start < branches.size()) {
        size_t end  = start;
        int    rows = 0;
        while (end < branches.size() && (int) (end - start) < n_free && rows + (int) branches[end].toks.size() <= max_rows) {
            rows += (int) branches[end].toks.size();
            ++end;
        }
        if (end == start) {
            throw std::runtime_error("a decision suffix exceeds the batch size");
        }
        llama_batch batch = llama_batch_init(rows, 0, 1);
        std::vector<int> out_idx;
        for (size_t b = start; b < end; ++b) {
            const llama_seq_id seq = first + (llama_seq_id) (b - start);
            llama_memory_seq_rm(mem, seq, -1, -1);
            llama_memory_seq_cp(mem, branches[b].trunk, seq, -1, -1);
            const auto & toks = branches[b].toks;
            for (size_t i = 0; i < toks.size(); ++i) {
                const bool last = i + 1 == toks.size();
                if (last) {
                    out_idx.push_back(batch.n_tokens);
                }
                de_batch_add(batch, toks[i], branches[b].pos0 + (llama_pos) i, { seq }, last);
            }
        }
        const int rc = llama_decode(ctx, batch);
        llama_batch_free(batch);
        if (rc != 0) {
            throw std::runtime_error(rc == 1 ? "no free KV cache space for the decision branches"
                                             : "llama_decode failed on the decision branches (" + std::to_string(rc) + ")");
        }
        for (size_t b = start; b < end; ++b) {
            const float * logits = llama_get_logits_ith(ctx, out_idx[b - start]);
            for (llama_token t : branches[b].cands) {
                result[b].push_back(logits[t]);
            }
            llama_memory_seq_rm(mem, first + (llama_seq_id) (b - start), -1, -1);
        }
        start = end;
    }
    return result;
}

result engine::decide(const std::string & shared_text, const std::string & context_text,
                      const std::vector<field_input> & inputs, const options & opt) {
    batch_result b = decide_batch(shared_text, { context_text }, inputs, opt);
    result r = std::move(b.items[0]);
    r.cache_hit     = b.cache_hit;
    r.shared_tokens = b.shared_tokens;
    r.rounds        = b.rounds;
    r.prefill_ms    = b.prefill_ms;
    r.scoring_ms    = b.scoring_ms;
    return r;
}

batch_result engine::decide_batch(const std::string & shared_text, const std::vector<std::string> & contexts,
                                  const std::vector<field_input> & inputs, const options & opt) {
    if (opt.mode != "auto" && opt.mode != "tree" && opt.mode != "greedy") {
        throw std::invalid_argument("mode must be auto, tree or greedy");
    }
    if (contexts.empty()) {
        throw std::invalid_argument("a decision needs at least one context");
    }
    const tokens_t shared = tokenize(shared_text, true);
    std::vector<tokens_t> prefixes;
    for (const auto & text : contexts) {
        prefixes.push_back(tokenize(text, shared.empty()));
        if (prefixes.back().empty()) {
            throw std::invalid_argument("the decision context must not be empty");
        }
    }

    std::vector<decision_field> fields;
    int total    = 0;
    int branches = 0; // round-1 branches of one context
    for (const auto & in : inputs) {
        const size_t n = in.candidates.size();
        if (n < 1 || n > 255) {
            throw std::invalid_argument("each field needs 1-255 allowed values");
        }
        tokens_t suffix;
        std::vector<tokens_t> paths;
        if (opt.split_boundary) {
            for (const auto & c : in.candidates) {
                paths.push_back(tokenize(c + "\n", false));
            }
            suffix = tokenize(in.suffix, false);
        } else {
            // Tokenise each complete "suffix + value + terminator" and split at the longest token
            // prefix shared by every candidate: the value's first token is exactly what the model
            // would write itself (e.g. `":` then ` true`, not `": ` then `true`).
            std::vector<tokens_t> seqs;
            for (const auto & c : in.candidates) {
                seqs.push_back(tokenize(in.suffix + c + "\n", false));
            }
            size_t common = seqs[0].size();
            for (const auto & s : seqs) {
                size_t c = 0;
                while (c < std::min(common, s.size()) && s[c] == seqs[0][c]) {
                    ++c;
                }
                common = std::min({ common, c, s.size() - 1 });
            }
            suffix.assign(seqs[0].begin(), seqs[0].begin() + common);
            for (const auto & s : seqs) {
                paths.emplace_back(s.begin() + common, s.end());
            }
        }
        if (suffix.empty()) {
            throw std::invalid_argument("a field suffix must not be empty");
        }
        size_t max_path = 0;
        for (size_t a = 0; a < paths.size(); ++a) {
            max_path = std::max(max_path, paths[a].size());
            for (size_t b = 0; b < a; ++b) {
                const size_t m = std::min(paths[a].size(), paths[b].size());
                if (std::equal(paths[a].begin(), paths[a].begin() + m, paths[b].begin())) {
                    throw std::invalid_argument("two allowed values tokenise to colliding paths");
                }
            }
        }
        decision_field field(suffix, paths);
        field.build_nodes();
        field.use_tree = opt.mode == "tree" ? true : opt.mode == "greedy" ? false : n <= opt.tree_max;
        total    += field.use_tree ? field.tree_rows() : (int) (suffix.size() + max_path);
        branches += field.use_tree ? (int) field.node_prefix.size() : 1;
        fields.push_back(std::move(field));
    }

    batch_result out;
    out.shared_tokens = shared.size();
    out.rows          = total * (int) contexts.size();
    out.items.resize(contexts.size());

    const auto t0 = std::chrono::steady_clock::now();
    out.cache_hit = prepare_prefix(shared, opt.allow_cache);
    out.prefill_ms += ms_since(t0);

    // each context in a group holds one trunk sequence; the rest of the pool scores branches
    const size_t per_group = std::clamp<size_t>(n_pool / (1 + branches), 1, contexts.size());
    for (size_t g0 = 0; g0 < contexts.size(); g0 += per_group) {
        const size_t n_group = std::min(per_group, contexts.size() - g0);

        const auto tp = std::chrono::steady_clock::now();
        std::vector<prompt_part> parts;
        for (size_t i = 0; i < n_group; ++i) {
            const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
            llama_memory_seq_rm(mem, trunk, -1, -1);
            if (!shared.empty()) {
                llama_memory_seq_cp(mem, seq_snap, trunk, -1, -1);
            }
            parts.push_back({ &prefixes[g0 + i], (llama_pos) shared.size(), trunk });
        }
        decode_parts(parts);
        llama_synchronize(ctx); // llama_decode is asynchronous: wait for the prefill so its time isn't billed to scoring
        out.prefill_ms += ms_since(tp);

        // round 1 carries every tree node and each greedy field's first step; later rounds only
        // continue greedy fields that are still open
        const auto ts = std::chrono::steady_clock::now();
        std::vector<std::vector<decision_field>> state(n_group, fields);
        bool first = true;
        while (true) {
            std::vector<branch> todo;
            std::vector<std::pair<size_t, size_t>> owner; // (context in group, field)
            for (size_t i = 0; i < n_group; ++i) {
                const llama_seq_id trunk = seq_pool + (llama_seq_id) i;
                const llama_pos    pos0  = (llama_pos) (shared.size() + prefixes[g0 + i].size());
                for (size_t f = 0; f < state[i].size(); ++f) {
                    auto & fd = state[i][f];
                    if (fd.use_tree) {
                        if (first) {
                            for (size_t n = 0; n < fd.node_prefix.size(); ++n) {
                                tokens_t ids = fd.suffix;
                                ids.insert(ids.end(), fd.node_prefix[n].begin(), fd.node_prefix[n].end());
                                todo.push_back({ trunk, pos0, ids, fd.node_options[n] });
                                owner.push_back({ i, f });
                            }
                        }
                    } else {
                        tokens_t opts = fd.options();
                        if (!opts.empty()) {
                            tokens_t ids = fd.suffix;
                            ids.insert(ids.end(), fd.chosen.begin(), fd.chosen.end());
                            todo.push_back({ trunk, pos0, ids, opts });
                            owner.push_back({ i, f });
                        }
                    }
                }
            }
            if (todo.empty()) {
                break;
            }
            const auto scores = score_branches(todo, seq_pool + (llama_seq_id) n_group, n_pool - (int) n_group);
            out.rounds += 1;
            std::vector<std::vector<std::vector<std::vector<float>>>> tree_scores(n_group, std::vector<std::vector<std::vector<float>>>(fields.size()));
            for (size_t row = 0; row < owner.size(); ++row) {
                const auto [i, f] = owner[row];
                auto & fd = state[i][f];
                if (fd.use_tree) {
                    tree_scores[i][f].push_back(scores[row]);
                } else {
                    const auto & s    = scores[row];
                    const int    best = (int) (std::max_element(s.begin(), s.end()) - s.begin());
                    double denom = 0;
                    for (float x : s) {
                        denom += std::exp(x - s[best]);
                    }
                    fd.select(todo[row].cands[best], (float) (1.0 / denom));
                }
            }
            if (first) {
                for (size_t i = 0; i < n_group; ++i) {
                    for (size_t f = 0; f < fields.size(); ++f) {
                        if (state[i][f].use_tree) {
                            state[i][f].finish_tree(tree_scores[i][f]);
                        }
                    }
                }
            }
            first = false;
        }
        for (size_t i = 0; i < n_group; ++i) {
            llama_memory_seq_rm(mem, seq_pool + (llama_seq_id) i, -1, -1);
            result & r = out.items[g0 + i];
            r.context_tokens = prefixes[g0 + i].size();
            r.rows           = total;
            for (auto & fd : state[i]) {
                if (fd.use_tree && fd.probs.empty()) {
                    fd.finish_tree({});
                }
                r.fields.push_back({ fd.winner, fd.path_score, fd.scored_nodes, fd.use_tree, fd.probs });
            }
        }
        out.scoring_ms += ms_since(ts);
    }
    return out;
}

} // namespace llama_decision
