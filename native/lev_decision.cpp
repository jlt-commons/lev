/* lev_decision.cpp — Jev mode: the decision engine (native/decision-engine.cpp,
 * vendored from thecodacus/llama.cpp's parallel-decision branch) behind one
 * flat C entry point for jolt.ffi.
 *
 * A call answers a set of fields for one or more contexts in one pass: the
 * shared text (the part of the prompt every call repeats) is decoded once
 * and kept on the engine's first sequence across calls; each context
 * continues it on a trunk sequence; each field is a branch forked from its
 * trunk (llama_memory_seq_cp on the unified KV cache), its suffix's tokens
 * decoded together with every other branch in the same llama_decode, and
 * its allowed values scored at every divergence node of their token trie.
 * The fields cannot see each other; what comes back per field is the exact
 * distribution over its allowed values.
 *
 * The engine owns the handle's sequences [0, n_seq_max): lev_llm_generate
 * and lev_llm_decide clear the whole memory, so they drop it
 * (lev_jev_forget) and the next call here decodes the shared text again.
 */
#include "lev_llm.h"

#include "decision-engine.h"

#include <cstdio>
#include <exception>
#include <string>
#include <vector>

extern "C" {

void lev_jev_forget(struct lev_llm *h) {
    if (h && h->jev) {
        delete static_cast<llama_decision::engine *>(h->jev);
        h->jev = nullptr;
    }
}

/* Decide n_fields fields for each of n_contexts contexts.
 *
 *   shared       the text before every context (cached across calls while
 *                it stays the same); "" for none
 *   contexts     n_contexts non-empty texts, each continuing `shared`
 *   suffixes     per field, the text after the context that the field's
 *                value follows
 *   n_values     per field, how many allowed values it has (1-255)
 *   values       every field's values, flattened in field order; each is
 *                tokenized together with its suffix and the split falls
 *                where the values' tokens start to differ
 *   allow_cache  reuse the kept shared text when it matches
 *   split_boundary  tokenize each suffix and value apart (the values
 *                start a token) instead of together, split where they differ
 *   probs        out, n_contexts x sum(n_values): per context, per field,
 *                the probability of each value
 *   stats        out, 6 doubles: prefill ms, scoring ms, rounds, scored
 *                rows, cache hit (0/1), shared tokens
 *   ctx_tokens   out, n_contexts token counts of the contexts
 *
 * Answers 0, or -1 with the reason in lev_llm_error. */
int lev_llm_jev(struct lev_llm *h, const char *shared, const char **contexts, int n_contexts,
                const char **suffixes, const int *n_values, const char **values, int n_fields,
                int allow_cache, int split_boundary, double *probs, double *stats, int *ctx_tokens) {
    if (!h || !h->ctx) return -1;
    try {
        if (!h->jev) h->jev = new llama_decision::engine(h->ctx, 0, h->n_seq_max);
        auto *eng = static_cast<llama_decision::engine *>(h->jev);

        std::vector<llama_decision::field_input> fields((size_t)n_fields);
        int k = 0;
        for (int f = 0; f < n_fields; f++) {
            fields[f].suffix = suffixes[f];
            for (int v = 0; v < n_values[f]; v++) fields[f].candidates.emplace_back(values[k++]);
        }
        std::vector<std::string> ctxs;
        for (int i = 0; i < n_contexts; i++) ctxs.emplace_back(contexts[i]);

        llama_decision::options opt;
        opt.mode = "tree"; /* every divergence node scored: the exact distribution, in one round */
        opt.allow_cache = allow_cache != 0;
        opt.split_boundary = split_boundary != 0;

        llama_decision::batch_result r = eng->decide_batch(shared, ctxs, fields, opt);

        int p = 0;
        for (int i = 0; i < n_contexts; i++) {
            const auto &item = r.items[(size_t)i];
            for (int f = 0; f < n_fields; f++) {
                const auto &fr = item.fields[(size_t)f];
                for (int v = 0; v < n_values[f]; v++)
                    probs[p++] = fr.probs.size() == (size_t)n_values[f] ? fr.probs[(size_t)v] : (v == fr.winner ? 1.0 : 0.0);
            }
            ctx_tokens[i] = (int)item.context_tokens;
        }
        stats[0] = r.prefill_ms;
        stats[1] = r.scoring_ms;
        stats[2] = r.rounds;
        stats[3] = r.rows;
        stats[4] = r.cache_hit ? 1 : 0;
        stats[5] = (double)r.shared_tokens;
        return 0;
    } catch (const std::exception &e) {
        snprintf(h->error, sizeof h->error, "decision: %s", e.what());
    } catch (...) {
        snprintf(h->error, sizeof h->error, "decision: unknown failure");
    }
    /* a throw can leave branch sequences behind: start over next time */
    lev_jev_forget(h);
    llama_memory_clear(llama_get_memory(h->ctx), 0);
    return -1;
}

} // extern "C"
