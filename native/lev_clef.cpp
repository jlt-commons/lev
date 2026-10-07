/* lev_clef.cpp — what a Clef agent (lev.clef) needs from llama.cpp beyond
 * lev_llm.c: the token ids of a text, the backbone's last hidden state at
 * every token, and rows of the output embedding (the lm_head) out of the
 * GGUF file.
 *
 * Clef (Cloudflare/clef) is a Qwen3.5-architecture backbone with a joint
 * schema head that reads the final-normed hidden state of every token of
 * the prompt, so the hidden states come from the fork's staging API
 * (llama-ext.h, C++ only): nextn embeddings, unmasked, are exactly the
 * output norm's result for every token of a batch, without running the
 * lm_head on each of them.
 */
#include "lev_llm.h"

#include "gguf.h"
#include "llama-ext.h"

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" {

int lev_llm_ok(struct lev_llm *h); /* lev_llm.c */

int lev_llm_n_embd(struct lev_llm *h) { return h && h->model ? llama_model_n_embd(h->model) : 0; }

/* The token ids of `text` into out (cap of them); answers the count, -1
 * when it does not fit or cannot be tokenized. add_special: BOS as the
 * model asks; parse_special: control tokens' texts read as those tokens
 * (0 for caller text, which then cannot forge a chat tag). */
int lev_llm_tokenize(struct lev_llm *h, const char *text, int add_special, int parse_special, int32_t *out, int cap) {
    if (!h || !h->vocab) return -1;
    int n = llama_tokenize(h->vocab, text, (int32_t)strlen(text), out, cap, add_special != 0, parse_special != 0);
    if (n < 0) snprintf(h->error, sizeof h->error, "%d tokens do not fit in %d", -n, cap);
    return n;
}

/* The last hidden state (after the output norm) of each of the n tokens,
 * decoded as one sequence from position 0: out holds n * n_embd floats,
 * token-major. Clears the context's memory first. Answers 0 or -1. */
int lev_llm_hidden(struct lev_llm *h, const int32_t *toks, int n, float *out) {
    if (!lev_llm_ok(h)) return -1;
    if (n < 1 || n > h->n_ctx) {
        snprintf(h->error, sizeof h->error, "%d tokens for a context of %d", n, h->n_ctx);
        return -1;
    }
    lev_jev_forget(h);
    llama_memory_clear(llama_get_memory(h->ctx), false);
    llama_set_embeddings_nextn(h->ctx, true, false);
    const int d = llama_model_n_embd(h->model);
    struct llama_batch b = llama_batch_init(h->n_batch, 0, 1);
    int rc = 0;
    for (int i0 = 0; i0 < n && rc == 0; i0 += h->n_batch) {
        int k = n - i0 < h->n_batch ? n - i0 : h->n_batch;
        b.n_tokens = k;
        for (int j = 0; j < k; j++) {
            b.token[j] = toks[i0 + j];
            b.pos[j] = i0 + j;
            b.n_seq_id[j] = 1;
            b.seq_id[j][0] = 0;
            /* one output per batch: the lm_head runs on it alone */
            b.logits[j] = (int8_t)(j == k - 1);
        }
        if (llama_decode(h->ctx, b) != 0) {
            snprintf(h->error, sizeof h->error, "llama_decode failed at position %d (context %d)", i0, h->n_ctx);
            rc = -1;
            break;
        }
        const float *e = llama_get_embeddings_nextn(h->ctx);
        if (!e) {
            snprintf(h->error, sizeof h->error, "the model gives no hidden states");
            rc = -1;
            break;
        }
        memcpy(out + (size_t)i0 * d, e, sizeof(float) * (size_t)k * d);
    }
    llama_batch_free(b);
    llama_set_embeddings_nextn(h->ctx, false, false);
    llama_memory_clear(llama_get_memory(h->ctx), false);
    return rc;
}

/* --- rows of a GGUF tensor ---------------------------------------------- */

struct lev_rows {
    FILE *f;
    size_t base;      /* file offset of the tensor's data */
    size_t row_bytes;
    int64_t n_rows;
    int64_t n_cols;
    ggml_to_float_t to_float;
    std::vector<unsigned char> buf;
};

/* Open tensor `name` (2-D, ne[1] rows of ne[0] columns) of the GGUF at
 * path for row reads; NULL when it is not there or its type has no
 * dequantizer. */
struct lev_rows *lev_rows_open(const char *path, const char *name) {
    struct ggml_context *meta = nullptr;
    struct gguf_init_params p = {true, &meta};
    struct gguf_context *g = gguf_init_from_file(path, p);
    if (!g) return nullptr;
    int64_t id = gguf_find_tensor(g, name);
    struct ggml_tensor *t = id < 0 ? nullptr : ggml_get_tensor(meta, name);
    struct lev_rows *r = nullptr;
    if (t && ggml_n_dims(t) == 2) {
        const struct ggml_type_traits *tr = ggml_get_type_traits(t->type);
        FILE *f = (t->type == GGML_TYPE_F32 || (tr && tr->to_float)) ? fopen(path, "rb") : nullptr;
        if (f) {
            r = new lev_rows();
            r->f = f;
            r->base = gguf_get_data_offset(g) + gguf_get_tensor_offset(g, id);
            r->n_cols = t->ne[0];
            r->n_rows = t->ne[1];
            r->row_bytes = ggml_row_size(t->type, t->ne[0]);
            r->to_float = t->type == GGML_TYPE_F32 ? nullptr : tr->to_float;
            r->buf.resize(r->row_bytes);
        }
    }
    gguf_free(g);
    ggml_free(meta);
    return r;
}

int64_t lev_rows_n_rows(struct lev_rows *r) { return r ? r->n_rows : 0; }
int64_t lev_rows_n_cols(struct lev_rows *r) { return r ? r->n_cols : 0; }

/* Rows ids[0..n) of the tensor as f32 into out (n * n_cols floats).
 * Answers 0 or -1 (a bad id, a short read). */
int lev_rows_get(struct lev_rows *r, const int32_t *ids, int n, float *out) {
    if (!r || r->row_bytes == 0) return -1;
    for (int i = 0; i < n; i++) {
        if (ids[i] < 0 || ids[i] >= r->n_rows) return -1;
        if (fseeko(r->f, (off_t)(r->base + (size_t)ids[i] * r->row_bytes), SEEK_SET) != 0) return -1;
        float *dst = out + (size_t)i * r->n_cols;
        if (r->to_float) {
            if (fread(r->buf.data(), 1, r->row_bytes, r->f) != r->row_bytes) return -1;
            r->to_float(r->buf.data(), dst, r->n_cols);
        } else if (fread(dst, 1, r->row_bytes, r->f) != r->row_bytes) {
            return -1;
        }
    }
    return 0;
}

void lev_rows_close(struct lev_rows *r) {
    if (!r) return;
    fclose(r->f);
    delete r;
}

/* --- the joint schema head's small ops (lev.clef; the gemms are cblas) -- */

/* count bf16 values at offset of the file at path, as f32 into dst.
 * Answers count, or -1. */
int64_t lev_clef_bf16_read(const char *path, int64_t offset, int64_t count, float *dst) {
    FILE *f = fopen(path, "rb");
    if (!f) return -1;
    if (fseeko(f, (off_t)offset, SEEK_SET) != 0) {
        fclose(f);
        return -1;
    }
    std::vector<uint16_t> buf(1 << 16);
    int64_t done = 0;
    while (done < count) {
        size_t want = (size_t)(count - done < (int64_t)buf.size() ? count - done : (int64_t)buf.size());
        if (fread(buf.data(), 2, want, f) != want) {
            done = -1;
            break;
        }
        for (size_t i = 0; i < want; i++) {
            uint32_t bits = (uint32_t)buf[i] << 16;
            memcpy(dst + done + (int64_t)i, &bits, 4);
        }
        done += (int64_t)want;
    }
    fclose(f);
    return done;
}

/* x[i, :] += b for each of n rows of d */
void lev_clef_add_bias(float *x, const float *b, int64_t n, int64_t d) {
    for (int64_t i = 0; i < n; i++)
        for (int64_t j = 0; j < d; j++) x[i * d + j] += b[j];
}

/* dst = the mean of rows [s, e) of src (rows of d); an empty span is zero */
void lev_clef_span_mean(const float *src, int64_t d, int64_t s, int64_t e, float *dst) {
    std::vector<double> acc((size_t)d, 0.0);
    for (int64_t i = s; i < e; i++)
        for (int64_t j = 0; j < d; j++) acc[(size_t)j] += src[i * d + j];
    double n = e > s ? (double)(e - s) : 1.0;
    for (int64_t j = 0; j < d; j++) dst[j] = (float)(acc[(size_t)j] / n);
}

/* dst = sum_o softmax_o(opts[o] . field / sqrt(w)) opts[o], over k rows
 * of w: a question's summary of its routed options */
void lev_clef_attend_pool(const float *field, const float *opts, int64_t k, int64_t w, float *dst) {
    std::vector<double> sc((size_t)k);
    double mx = -INFINITY;
    for (int64_t o = 0; o < k; o++) {
        double dot = 0.0;
        for (int64_t j = 0; j < w; j++) dot += (double)opts[o * w + j] * field[j];
        sc[(size_t)o] = dot / sqrt((double)w);
        if (sc[(size_t)o] > mx) mx = sc[(size_t)o];
    }
    double z = 0.0;
    for (int64_t o = 0; o < k; o++) z += sc[(size_t)o] = exp(sc[(size_t)o] - mx);
    for (int64_t j = 0; j < w; j++) {
        double v = 0.0;
        for (int64_t o = 0; o < k; o++) v += sc[(size_t)o] / z * opts[o * w + j];
        dst[j] = (float)v;
    }
}

/* the residual scorer's input for one question: per option row o,
 * [field, opt_o, field * opt_o, |field - opt_o|] into out [k x 4w] */
void lev_clef_pair_features(const float *field, const float *opts, int64_t k, int64_t w, float *out) {
    for (int64_t o = 0; o < k; o++) {
        float *r = out + o * 4 * w;
        const float *x = opts + o * w;
        for (int64_t j = 0; j < w; j++) {
            r[j] = field[j];
            r[w + j] = x[j];
            r[2 * w + j] = field[j] * x[j];
            r[3 * w + j] = fabsf(field[j] - x[j]);
        }
    }
}

static double norm_of(const float *x, int64_t n) {
    double s = 0.0;
    for (int64_t i = 0; i < n; i++) s += (double)x[i] * x[i];
    return sqrt(s);
}

/* One question's option logits:
 *   prior_o = prior_scale * cos(lex_o, anchor)          (anchor: D)
 *   joint_o = joint_scale * cos(field, opts_o) + residual_o
 *   out_o   = prior_o + gate * joint_o
 * with torch's eps (normalize: 1e-12 on the norm; cosine_similarity: 1e-8
 * on each norm). The scales and gate arrive already clamped / squashed. */
void lev_clef_logits(const float *field, const float *opts, int64_t k, int64_t w,
                     const float *lex, const float *anchor, int64_t D, const float *residual,
                     float prior_scale, float joint_scale, float gate, float *out) {
    double an = norm_of(anchor, D);
    if (an < 1e-12) an = 1e-12;
    double fn = norm_of(field, w);
    if (fn < 1e-8) fn = 1e-8;
    for (int64_t o = 0; o < k; o++) {
        const float *l = lex + o * D;
        double ln = norm_of(l, D);
        if (ln < 1e-12) ln = 1e-12;
        double dot = 0.0;
        for (int64_t j = 0; j < D; j++) dot += (double)l[j] * anchor[j];
        double prior = prior_scale * dot / (ln * an);
        const float *x = opts + o * w;
        double xn = norm_of(x, w);
        if (xn < 1e-8) xn = 1e-8;
        double fx = 0.0;
        for (int64_t j = 0; j < w; j++) fx += (double)field[j] * x[j];
        double joint = joint_scale * fx / (fn * xn) + residual[o];
        out[o] = (float)(prior + gate * joint);
    }
}

}
