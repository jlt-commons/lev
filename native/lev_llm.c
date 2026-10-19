/* lev_llm.c — a flat C face on llama.cpp for jolt.ffi.
 *
 * llama.h is a struct-by-value API (params, batches, token arrays); this
 * file keeps all of that in C and exposes pointer-and-scalar entry points
 * the FFI binds in lev.llm: load a GGUF, free it, generate text, and
 * `decide`, which is what the thinker engine runs: decode a chat prompt,
 * let the model think until its closing tag, force the answer prefix, then
 * score every candidate answer by teacher-forcing its tokens on a copy of
 * the KV state (one sequence per option, one batch), answering log
 * probabilities the caller softmaxes into the answer's probabilities.
 *
 * One handle is one model + one context; calls on it are not thread-safe
 * (the server serializes inference on a lock). llama.cpp's own logging is
 * dropped unless LEV_LLM_LOG is set.
 *
 * Built by `jolt llama` together with llama.cpp's static libraries into
 * liblev_llm.{dylib,so} for jolt run/test and liblev_llm.a for jolt build.
 */
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "lev_llm.h"

static int lev_logging = -1;

/* Every live handle, so a process that exits without freeing them (a
 * server on SIGTERM, a test runner) still frees them before ggml's own
 * static destructors run: ggml-metal asserts that no residency set is
 * alive when its device goes away. atexit handlers registered after the
 * library loaded run before its static destructors, which were
 * registered at load. */
#define LEV_MAX_HANDLES 16
static struct lev_llm *live[LEV_MAX_HANDLES];
static int live_registered = 0;

static void free_handle(struct lev_llm *h) {
    lev_jev_forget(h);
    free(h->specials);
    h->specials = NULL;
    h->n_specials = 0;
    if (h->ctx) llama_free(h->ctx);
    if (h->model) llama_model_free(h->model);
    h->ctx = NULL;
    h->model = NULL;
}

static void free_all_live(void) {
    for (int i = 0; i < LEV_MAX_HANDLES; i++)
        if (live[i]) {
            free_handle(live[i]);
            live[i] = NULL;
        }
}

static void track(struct lev_llm *h) {
    if (!live_registered) {
        atexit(free_all_live);
        live_registered = 1;
    }
    for (int i = 0; i < LEV_MAX_HANDLES; i++)
        if (!live[i]) {
            live[i] = h;
            return;
        }
}

static void untrack(struct lev_llm *h) {
    for (int i = 0; i < LEV_MAX_HANDLES; i++)
        if (live[i] == h) live[i] = NULL;
}

static void lev_log(enum ggml_log_level level, const char *text, void *user) {
    (void)level;
    (void)user;
    if (lev_logging < 0) lev_logging = getenv("LEV_LLM_LOG") != NULL;
    if (lev_logging) fputs(text, stderr);
}

static void set_error(struct lev_llm *h, const char *fmt, ...) {
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(h->error, sizeof h->error, fmt, ap);
    va_end(ap);
}

const char *lev_llm_error(struct lev_llm *h) { return h ? h->error : "no handle"; }

#ifndef LEV_LLAMA_BUILD
#define LEV_LLAMA_BUILD "unknown"
#endif

/* the llama.cpp tag this was built against (jolt llama passes it), so a
 * binary can say what it carries */
const char *lev_llm_version(void) { return "llama.cpp " LEV_LLAMA_BUILD; }

/* Load a GGUF. n_ctx 0 = the model's; n_gpu_layers < 0 = all (Metal on
 * mac, nothing on a CPU-only build); n_threads <= 0 = llama.cpp's
 * default; n_seq_max bounds the options `decide` can score at once. */
struct lev_llm *lev_llm_load(const char *path, int n_ctx, int n_gpu_layers, int n_threads, int n_seq_max) {
    static int backend_ready = 0;
    if (!backend_ready) {
        llama_log_set(lev_log, NULL);
        llama_backend_init();
        backend_ready = 1;
    }
    struct lev_llm *h = calloc(1, sizeof *h);
    if (!h) return NULL;
    struct llama_model_params mp = llama_model_default_params();
    mp.n_gpu_layers = n_gpu_layers < 0 ? 999 : n_gpu_layers;
    h->model = llama_model_load_from_file(path, mp);
    if (!h->model) {
        set_error(h, "cannot load %s", path);
        return h;
    }
    h->vocab = llama_model_get_vocab(h->model);
    h->n_vocab = llama_vocab_n_tokens(h->vocab);
    struct llama_context_params cp = llama_context_default_params();
    if (n_ctx > 0) cp.n_ctx = (uint32_t)n_ctx;
    cp.n_batch = cp.n_ctx > 0 && cp.n_ctx < 2048 ? cp.n_ctx : 2048;
    cp.n_ubatch = 512;
    cp.n_seq_max = (uint32_t)(n_seq_max < 2 ? 2 : n_seq_max);
    /* one KV buffer for every sequence: the option sequences are copies of
     * the prompt sequence, so they share its cells; split per sequence
     * (the default past n_seq_max 1) each would get n_ctx / n_seq_max
     * tokens and a 300-token thought would not fit */
    cp.kv_unified = true;
    if (n_threads > 0) {
        cp.n_threads = n_threads;
        cp.n_threads_batch = n_threads;
    }
    cp.no_perf = true;
    h->ctx = llama_init_from_model(h->model, cp);
    if (!h->ctx) {
        set_error(h, "cannot create a context for %s", path);
        llama_model_free(h->model);
        h->model = NULL;
        return h;
    }
    h->n_ctx = (int)llama_n_ctx(h->ctx);
    h->n_batch = (int)cp.n_batch;
    h->n_seq_max = (int)cp.n_seq_max;
    track(h);
    return h;
}

int lev_llm_ok(struct lev_llm *h) { return h && h->ctx != NULL; }
int lev_llm_n_ctx(struct lev_llm *h) { return h ? h->n_ctx : 0; }
int lev_llm_n_seq_max(struct lev_llm *h) { return h ? h->n_seq_max : 0; }
int lev_llm_n_vocab(struct lev_llm *h) { return h ? h->n_vocab : 0; }

void lev_llm_free(struct lev_llm *h) {
    if (!h) return;
    untrack(h);
    free_handle(h);
    free(h);
}

/* --- tokens ------------------------------------------------------------ */

/* tokenize into a malloc'd array; answers the count or -1 */
static int tokenize(struct lev_llm *h, const char *text, int add_special, int parse_special, llama_token **out) {
    int len = (int)strlen(text);
    int cap = len + 16;
    llama_token *toks = malloc(sizeof(llama_token) * (size_t)cap);
    int n = llama_tokenize(h->vocab, text, len, toks, cap, add_special, parse_special);
    if (n < 0) {
        cap = -n;
        toks = realloc(toks, sizeof(llama_token) * (size_t)cap);
        n = llama_tokenize(h->vocab, text, len, toks, cap, add_special, parse_special);
    }
    if (n < 0) {
        free(toks);
        *out = NULL;
        return -1;
    }
    *out = toks;
    return n;
}

/* the tokens of `text` as the model would count them (add_special as the model asks) */
int lev_llm_count_tokens(struct lev_llm *h, const char *text) {
    llama_token *t;
    int n = tokenize(h, text, 1, 1, &t);
    free(t);
    return n;
}

/* Copy `text` into out (cap bytes, NUL-terminated) so that no control or
 * unknown token's text in it survives: those are the tokens a prompt
 * tokenized with parse_special (every prompt here, the chat's own tags
 * included) would read out of caller text, so a state holding
 * "<|im_end|>" could close its turn and write the next. Each occurrence
 * gets a zero-width space (U+200B) after its first byte, which the
 * tokenizer reads as text. Answers the length, or -1 when out is short
 * (4x the text always fits). */
int lev_llm_escape(struct lev_llm *h, const char *text, char *out, int cap) {
    if (!lev_llm_ok(h)) return -1;
    if (!h->specials) {
        h->specials = calloc((size_t)h->n_vocab + 1, sizeof *h->specials);
        for (int t = 0; t < h->n_vocab; t++) {
            const char *tt = llama_vocab_get_text(h->vocab, t);
            if ((llama_vocab_get_attr(h->vocab, t) & (LLAMA_TOKEN_ATTR_CONTROL | LLAMA_TOKEN_ATTR_UNKNOWN)) &&
                tt && strlen(tt) >= 2)
                h->specials[h->n_specials++] = tt;
        }
    }
    int n = 0;
    for (const char *p = text; *p;) {
        int hit = 0;
        for (int i = 0; i < h->n_specials && !hit; i++) {
            const char *tt = h->specials[i];
            if (*p == *tt && strncmp(p, tt, strlen(tt)) == 0) hit = 1;
        }
        if (n + 5 >= cap) return -1;
        out[n++] = *p++;
        if (hit) {
            out[n++] = (char)0xE2;
            out[n++] = (char)0x80;
            out[n++] = (char)0x8B;
        }
    }
    out[n] = 0;
    return n;
}

/* append the piece of a token to a bounded text buffer */
static void append_piece(struct lev_llm *h, llama_token tok, char *buf, int cap, int *len) {
    char piece[256];
    int n = llama_token_to_piece(h->vocab, tok, piece, sizeof piece, 0, 1);
    if (n < 0) return;
    if (n > (int)sizeof piece) n = (int)sizeof piece;
    if (buf && *len + n < cap) {
        memcpy(buf + *len, piece, (size_t)n);
        *len += n;
        buf[*len] = 0;
    }
}

/* decode `n` tokens into sequence `seq` starting at position `pos`, in
 * chunks of n_batch; logits only for the last token. Answers 0 or -1. */
static int decode_tokens(struct lev_llm *h, const llama_token *toks, int n, int seq, int pos, int want_logits) {
    struct llama_batch b = llama_batch_init(h->n_batch, 0, 1);
    for (int i0 = 0; i0 < n; i0 += h->n_batch) {
        int k = n - i0 < h->n_batch ? n - i0 : h->n_batch;
        b.n_tokens = k;
        for (int j = 0; j < k; j++) {
            b.token[j] = toks[i0 + j];
            b.pos[j] = pos + i0 + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = seq;
            b.logits[j] = (int8_t)(want_logits && i0 + j == n - 1);
        }
        if (llama_decode(h->ctx, b) != 0) {
            llama_batch_free(b);
            set_error(h, "llama_decode failed at position %d (context %d)", pos + i0, h->n_ctx);
            return -1;
        }
    }
    llama_batch_free(b);
    return 0;
}

static struct llama_sampler *make_sampler(float temperature, float top_p, float min_p, unsigned seed) {
    struct llama_sampler_chain_params sp = llama_sampler_chain_default_params();
    sp.no_perf = true;
    struct llama_sampler *s = llama_sampler_chain_init(sp);
    if (temperature <= 0.0f) {
        llama_sampler_chain_add(s, llama_sampler_init_greedy());
    } else {
        if (top_p < 1.0f) llama_sampler_chain_add(s, llama_sampler_init_top_p(top_p, 1));
        if (min_p > 0.0f) llama_sampler_chain_add(s, llama_sampler_init_min_p(min_p, 1));
        llama_sampler_chain_add(s, llama_sampler_init_temp(temperature));
        llama_sampler_chain_add(s, llama_sampler_init_dist(seed));
    }
    return s;
}

static int ends_with(const char *buf, int len, const char *suffix) {
    int n = (int)strlen(suffix);
    return n > 0 && len >= n && memcmp(buf + len - n, suffix, (size_t)n) == 0;
}

/* Sample up to max_tokens tokens into seq 0 from position *pos (which
 * advances), appending their text to out; stops at end-of-generation or
 * when the text ends with `stop` (kept in the text). Answers the number
 * of tokens sampled, -1 on a decode error. The context's logits must be
 * those of the last decoded token on entry, and are those of the last
 * sampled token on exit (when any was sampled). */
static int sample_until(struct lev_llm *h, struct llama_sampler *s, int max_tokens, const char *stop,
                        int *pos, char *out, int cap, int *len, int *stopped) {
    int n = 0;
    *stopped = 0;
    while (n < max_tokens) {
        llama_token tok = llama_sampler_sample(s, h->ctx, -1);
        if (llama_vocab_is_eog(h->vocab, tok)) {
            *stopped = 1;
            break;
        }
        llama_sampler_accept(s, tok);
        if (decode_tokens(h, &tok, 1, 0, *pos, 1) < 0) return -1;
        (*pos)++;
        n++;
        append_piece(h, tok, out, cap, len);
        if (stop && *stop && ends_with(out, *len, stop)) {
            *stopped = 1;
            break;
        }
    }
    return n;
}

/* --- generate ---------------------------------------------------------- */

/* Plain completion of `prompt` (BOS added if the model wants one, special
 * tokens parsed): up to max_tokens tokens, stopping at end-of-generation
 * or when the text ends with `stop` (may be NULL). The text lands in
 * out (NUL-terminated, cut at cap-1 bytes); answers the token count or -1. */
int lev_llm_generate(struct lev_llm *h, const char *prompt, int max_tokens, const char *stop,
                     float temperature, float top_p, float min_p, unsigned seed, char *out, int cap) {
    if (!lev_llm_ok(h)) return -1;
    lev_jev_forget(h);
    llama_memory_clear(llama_get_memory(h->ctx), 0);
    llama_token *toks;
    int n = tokenize(h, prompt, 1, 1, &toks);
    if (n < 0) {
        set_error(h, "cannot tokenize the prompt");
        return -1;
    }
    if (n + max_tokens > h->n_ctx) {
        free(toks);
        set_error(h, "prompt of %d tokens + %d to generate exceed the context of %d", n, max_tokens, h->n_ctx);
        return -1;
    }
    int rc = decode_tokens(h, toks, n, 0, 0, 1);
    free(toks);
    if (rc < 0) return -1;
    int pos = n, len = 0, stopped = 0;
    if (cap > 0) out[0] = 0;
    struct llama_sampler *s = make_sampler(temperature, top_p, min_p, seed);
    int made = sample_until(h, s, max_tokens, stop, &pos, out, cap, &len, &stopped);
    llama_sampler_free(s);
    return made;
}

/* --- decide ------------------------------------------------------------ */

static void log_softmax_into(const float *logits, int n, float *out) {
    float m = logits[0];
    for (int i = 1; i < n; i++) if (logits[i] > m) m = logits[i];
    double z = 0.0;
    for (int i = 0; i < n; i++) z += exp((double)(logits[i] - m));
    float lz = (float)log(z);
    for (int i = 0; i < n; i++) out[i] = logits[i] - m - lz;
}

static double now_ms(void) {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec * 1e3 + (double)ts.tv_nsec / 1e6;
}

/* Sample a token from a saved logits vector, as llama_sampler_sample does
 * from the context's: the thought keeps its own last logits so a probe on
 * another sequence can overwrite the context's in between. */
static llama_token sample_logits(struct llama_sampler *s, const float *logits, llama_token_data *cand, int n_vocab) {
    for (int i = 0; i < n_vocab; i++) {
        cand[i].id = i;
        cand[i].logit = logits[i];
        cand[i].p = 0.0f;
    }
    llama_token_data_array arr = {cand, (size_t)n_vocab, -1, false};
    llama_sampler_apply(s, &arr);
    return arr.data[arr.selected].id;
}

/* decode `text` (special tokens parsed) into `seq` at *pos, which advances.
 * Answers how many tokens were decoded (0 for empty text) or -1. */
static int decode_text(struct lev_llm *h, const char *text, int seq, int *pos) {
    if (!text || !*text) return 0;
    llama_token *t;
    int n = tokenize(h, text, 0, 1, &t);
    if (n < 0 || decode_tokens(h, t, n, seq, *pos, 1) < 0) {
        if (n >= 0) free(t);
        return -1;
    }
    *pos += n;
    free(t);
    return n;
}

/* The answer read off sequence `src` at `pos`: think_end when `close`,
 * then the answer prefix, then every option's tokens (+ answer_end), each
 * on its own copy of `src` (sequences 1..n_options), in one batch.
 * `last` holds the raw logits of src's last token, used when nothing is
 * decoded here. Writes n_options log probabilities; answers 0 or -1. */
static int readout(struct lev_llm *h, int src, int pos, int close, const char *think_end, const char *answer_prefix,
                   const char **options, int n_options, const char *answer_end, const float *last, double *logp) {
    llama_memory_t mem = llama_get_memory(h->ctx);
    int decoded = 0, d;
    if (close) {
        if ((d = decode_text(h, think_end, src, &pos)) < 0) return -1;
        decoded += d;
    }
    if ((d = decode_text(h, answer_prefix, src, &pos)) < 0) return -1;
    decoded += d;

    /* the distribution over the first answer token, before the options overwrite the logits */
    float *first = malloc(sizeof(float) * (size_t)h->n_vocab);
    log_softmax_into(decoded ? llama_get_logits_ith(h->ctx, -1) : last, h->n_vocab, first);

    llama_token **ot = calloc((size_t)n_options, sizeof *ot);
    int *on = calloc((size_t)n_options, sizeof *on);
    int total = 0, rc = 0;
    size_t endlen = answer_end ? strlen(answer_end) : 0;
    for (int i = 0; i < n_options; i++) {
        size_t olen = strlen(options[i]);
        char *text = malloc(olen + endlen + 1);
        memcpy(text, options[i], olen);
        if (endlen) memcpy(text + olen, answer_end, endlen);
        text[olen + endlen] = 0;
        on[i] = tokenize(h, text, 0, 1, &ot[i]);
        free(text);
        if (on[i] < 1) {
            set_error(h, "cannot tokenize option %d", i);
            rc = -1;
        }
        total += on[i] > 0 ? on[i] : 0;
        llama_memory_seq_cp(mem, src, i + 1, -1, -1);
    }
    float *scratch = malloc(sizeof(float) * (size_t)h->n_vocab);
    if (rc == 0) {
        struct llama_batch b = llama_batch_init(total, 0, 1);
        int k = 0;
        for (int i = 0; i < n_options; i++)
            for (int j = 0; j < on[i]; j++) {
                b.token[k] = ot[i][j];
                b.pos[k] = pos + j;
                b.n_seq_id[k] = 1;
                b.seq_id[k][0] = i + 1;
                b.logits[k] = (int8_t)(j + 1 < on[i]); /* the last token's logits are not needed */
                k++;
            }
        b.n_tokens = total;
        if (total > h->n_batch) {
            set_error(h, "%d option tokens exceed the batch of %d", total, h->n_batch);
            rc = -1;
        } else if (llama_decode(h->ctx, b) != 0) {
            set_error(h, "llama_decode failed scoring the options");
            rc = -1;
        } else {
            k = 0;
            for (int i = 0; i < n_options; i++) {
                double lp = first[ot[i][0]];
                for (int j = 0; j + 1 < on[i]; j++) {
                    log_softmax_into(llama_get_logits_ith(h->ctx, k + j), h->n_vocab, scratch);
                    lp += scratch[ot[i][j + 1]];
                }
                logp[i] = lp;
                k += on[i];
            }
        }
        llama_batch_free(b);
    }
    for (int i = 0; i < n_options; i++) {
        llama_memory_seq_rm(mem, i + 1, -1, -1);
        free(ot[i]);
    }
    free(ot);
    free(on);
    free(first);
    free(scratch);
    return rc;
}

/* The thinker's one call per question.
 *
 *   prompt        the chat prompt through the assistant turn: either ending
 *                 with the model's open-thought tag (think_max > 0) or with
 *                 the closed empty thought (think_max = 0)
 *   think_max     tokens of thought allowed; the thought ends at think_end
 *                 or end-of-generation, and if it runs out think_end is
 *                 forced so the model still answers
 *   think_end     e.g. "</think>"
 *   answer_prefix text forced after the thought, e.g. "\n\nANSWER: "
 *   options       the candidate answers, each scored as its tokens followed
 *                 by answer_end (e.g. "<|im_end|>"), so an option that is a
 *                 prefix of another is not favoured
 *   deadline_ms   0, or the wall time this call may take (from entry): the
 *                 thought is closed once deadline_ms - reserve_ms has passed,
 *                 reserve_ms being what the readout is expected to need
 *   cuts          n_cuts ascending thought lengths (0 < cut < think_max) at
 *                 which the answer is also read, on a copy of the state with
 *                 the thought closed there, before the thought goes on: what
 *                 every smaller budget would have answered, from one
 *                 generation (exact under greedy decoding)
 *   logp          out, (n_cuts + 1) x n_options log probabilities of the
 *                 options given everything before them (unnormalised over
 *                 options): one row per cut, then the final answer's; a cut
 *                 the thought closed before reads the final answer
 *   stats         out (may be NULL), 4 + 2 n_cuts: how the thought closed
 *                 (-1 none, 0 its own tag, 1 think_max, 2 end-of-generation,
 *                 3 the deadline), prefill ms, thought ms, final readout ms,
 *                 then per cut the thought ms when it was reached and its
 *                 readout ms (-1 for a cut the thought closed before)
 *   thought       out, the text of the thought (may be NULL)
 *
 * Answers the number of thought tokens, or -1 (see lev_llm_error). */
int lev_llm_decide(struct lev_llm *h, const char *prompt, int think_max, const char *think_end,
                   const char *answer_prefix, const char **options, int n_options, const char *answer_end,
                   float temperature, float top_p, float min_p, unsigned seed,
                   double deadline_ms, double reserve_ms, const int *cuts, int n_cuts,
                   double *logp, double *stats, char *thought, int thought_cap) {
    if (!lev_llm_ok(h)) return -1;
    int probe = n_options + 1; /* a cut's readout state */
    if (n_options < 1 || n_options + 1 + (n_cuts > 0) > h->n_seq_max) {
        set_error(h, "%d options%s, but the context holds %d sequences (n_seq_max)", n_options,
                  n_cuts > 0 ? " and a cut's state" : "", h->n_seq_max);
        return -1;
    }
    for (int c = 0; c < n_cuts; c++)
        if (cuts[c] <= 0 || cuts[c] >= think_max || (c > 0 && cuts[c] <= cuts[c - 1])) {
            set_error(h, "cuts must ascend strictly within (0, think_max = %d)", think_max);
            return -1;
        }
    double t0 = now_ms();
    lev_jev_forget(h);
    llama_memory_t mem = llama_get_memory(h->ctx);
    llama_memory_clear(mem, 0);
    if (thought && thought_cap > 0) thought[0] = 0;
    if (stats) {
        stats[0] = -1;
        for (int i = 1; i < 4 + 2 * n_cuts; i++) stats[i] = i < 4 ? 0 : -1;
    }

    llama_token *toks;
    int n = tokenize(h, prompt, 1, 1, &toks);
    if (n < 0) {
        set_error(h, "cannot tokenize the prompt");
        return -1;
    }
    if (n + think_max + 64 > h->n_ctx) {
        free(toks);
        set_error(h, "prompt of %d tokens + %d of thought exceed the context of %d", n, think_max, h->n_ctx);
        return -1;
    }
    int rc = decode_tokens(h, toks, n, 0, 0, 1);
    free(toks);
    if (rc < 0) return -1;
    int pos = n;
    float *cur = malloc(sizeof(float) * (size_t)h->n_vocab);
    memcpy(cur, llama_get_logits_ith(h->ctx, -1), sizeof(float) * (size_t)h->n_vocab);
    double t_think = now_ms();
    if (stats) stats[1] = t_think - t0;

    /* the thought */
    int made = 0, closed = -1, ci = 0;
    if (think_max > 0) {
        int len = 0;
        llama_token_data *cand = malloc(sizeof *cand * (size_t)h->n_vocab);
        struct llama_sampler *s = make_sampler(temperature, top_p, min_p, seed);
        closed = 1;
        while (made < think_max) {
            if (ci < n_cuts && made == cuts[ci]) {
                double tc = now_ms();
                llama_memory_seq_cp(mem, 0, probe, -1, -1);
                rc = readout(h, probe, pos, 1, think_end, answer_prefix, options, n_options, answer_end, cur,
                             logp + (size_t)ci * (size_t)n_options);
                llama_memory_seq_rm(mem, probe, -1, -1);
                if (rc < 0) break;
                if (stats) {
                    stats[4 + ci] = tc - t_think;
                    stats[4 + n_cuts + ci] = now_ms() - tc;
                }
                ci++;
                continue;
            }
            if (deadline_ms > 0 && now_ms() - t0 >= deadline_ms - reserve_ms) {
                closed = 3;
                break;
            }
            llama_token tok = sample_logits(s, cur, cand, h->n_vocab);
            if (llama_vocab_is_eog(h->vocab, tok)) {
                closed = 2;
                break;
            }
            llama_sampler_accept(s, tok);
            if ((rc = decode_tokens(h, &tok, 1, 0, pos, 1)) < 0) break;
            memcpy(cur, llama_get_logits_ith(h->ctx, -1), sizeof(float) * (size_t)h->n_vocab);
            pos++;
            made++;
            append_piece(h, tok, thought, thought_cap, &len);
            if (think_end && *think_end && ends_with(thought ? thought : "", len, think_end)) {
                closed = 0;
                break;
            }
        }
        llama_sampler_free(s);
        free(cand);
    }
    double t_read = now_ms();
    if (stats) {
        stats[0] = closed;
        stats[2] = t_read - t_think;
    }

    /* the answer: out of budget, at the deadline or on EOG mid-thought, close it ourselves */
    double *final = logp + (size_t)n_cuts * (size_t)n_options;
    if (rc == 0)
        rc = readout(h, 0, pos, think_max > 0 && closed != 0, think_end, answer_prefix, options, n_options,
                     answer_end, cur, final);
    if (stats) stats[3] = now_ms() - t_read;
    /* the cuts the thought never reached read what the whole thought did */
    for (; rc == 0 && ci < n_cuts; ci++)
        memcpy(logp + (size_t)ci * (size_t)n_options, final, sizeof(double) * (size_t)n_options);
    free(cur);
    return rc < 0 ? -1 : made;
}
