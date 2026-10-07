# Benchmarks

`data/authored144.jsonl` is von's authored144 set
([github.com/wfzyx/von](https://github.com/wfzyx/von), Apache-2.0; project-authored,
no copied text): 144 three-way decisions over one- or two-sentence states,
48 each of evidence interpretation (supported / insufficient / contradicted),
rule application (permitted / prohibited / insufficient) and candidate
selection (A / B / neither), labels balanced. `data/perturbations108.jsonl`
re-asks 108 of them with the options reversed or the question rewrapped. The
set is adversarial by design: every case turns on a negation, a scope word or
an abstention, and half the trick is knowing when the evidence does not
settle the question.

```
jolt -M bench/authored144.clj                              # english
jolt -M bench/authored144.clj --model typed-decisions
jolt -M bench/authored144.clj --model minicpm5 --thinking false   # a configured thinker (config.edn :thinkers, or --thinker PATH as `thinker`)
jolt -M bench/authored144.clj --model minicpm5                    # thinking
jolt -M bench/authored144.clj --file bench/data/perturbations108.jsonl --out results.jsonl
```

Each row is one `choice` question: the row's question as the instructions,
its options as the criteria, its state as the state.

## Results, 2026-09-19

10-core M-series laptop. lev through `bench/authored144.clj`; von-1.0
through its own Python backend (`BertaBackend`, torch 2.14 on the CPU, the
weights from `wfzyx/von-1.0`); MiniCPM5-2B (`openbmb/MiniCPM5-2B-GGUF`,
Q8_0) through `llama-server`, one chat completion per case with the answer
constrained by a grammar to the option ids, thinking on or off through the
chat template.

| model | params | authored144 | balanced | perturbations108 | ms / case |
|---|---|---|---|---|---|
| **lev encoder** `english` | 395M | 61.1% (88) | 64.8% | 67.6% (73) | 117 (CPU) |
| **lev encoder** `typed-decisions` | 395M | 66.7% (96) | 68.1% | | 116 (CPU) |
| **lev encoder** `multilingual` | 307M | 59.0% (85) | 56.4% | | 48 (CPU) |
| von-1.0, NLI zero-shot | 395M | 76.4% (110) | 76.4% | 73.1% (79) | 160 (CPU, torch) |
| MiniCPM5-2B Q8 through llama-server, direct answer (grammar) | 2.5B | 73.6% (106) | 71.8% | | 566 (CPU) / 89 (Metal) |
| MiniCPM5-2B Q8 through llama-server, thinking, free-form answer | 2.5B | **97.2% (140)** | 97.6% | | 3,938 mean, 2,759 median (Metal); ~9,800 mean, 8,050 median (CPU, 24-case sample); ~300 tokens of thought |
| **lev thinker** (MiniCPM5-2B Q8 in the binary), thinking off | 2.5B | 74.3% (107) | 71.4% | | 146 (Metal) |
| **lev thinker**, thinking, sampled (temperature 1.0) | 2.5B | 91.0% (131) | 91.3% | | 5,388 mean, 3,926 median (Metal); 342 tokens |
| **lev thinker**, thinking, greedy (the default) | 2.5B | **95.1% (137)** | 95.5% | | 4,716 mean, 3,084 median (Metal); 369 tokens |
| [localjev](https://github.com/githubnext/localjev) + DiffusionGemma 26B-A4B 4-bit on oMLX 0.6.4 | 26B (4B active), 16.6 GB | 91.0% (131) | 90.0% | | 5,192 mean, 5,262 median (Metal) |
| lev thinker, Qwen3-4B-Thinking-2507 Q8 (unsloth GGUF), greedy, 4,096-token budget | 4B | 94.4% (136) | 95.3% | | 42,543 mean, 41,216 median (Metal); 1,883 tokens |

The lev thinker is the same model and the same llama.cpp inside
`lev-server` (`lev.think`): the prompt is built by lev, the thought is
decoded greedily until `</think>`, then `\n\nANSWER: ` is forced and each
option is scored by the log probability of its tokens (plus the end-of-turn
token), a softmax over those being the answer's probabilities — the same
calibrated shape the encoders give, where the llama-server harness parsed
a free-form final line. Greedy beats a sampled thought by four points here
and is reproducible; the two points to the free-form run are prompt and
decoding noise on seven cases.

localjev is a Bun bridge that turns the Jev request into a prompt asking
an MLX-served chat model for a JSON probability vector (no thinking,
temperature 0, retries on malformed JSON). Its probabilities are
self-reported: 143 of the 144 answers here were one-hot, and all 13 wrong
answers came with a probability of 0.9 or more, so its `confidence` says
nothing. It needs oMLX and bun on Apple Silicon and a 16.6 GB model; it
was slower than the 2.5B thinker in the lev binary and four points behind it.

## In-distribution: AG News, BoolQ, SST-5

The trio localjev's bake-off uses, 40 balanced cases each
(`bench/triad120.py` builds the set from the public datasets; `jolt -M
bench/triad.clj <model> [thinking]` runs it). Routing-style traffic, the
kind the encoder's presets exist for:

| model | AG News | BoolQ | SST-5 (acc / MAE) | all | ms per question |
|---|---|---|---|---|---|
| lev encoder `english` | **97.5%** | 72.5% | 27.5% / 1.08 | **65.8%** | **125** (CPU) |
| lev thinker, thinking off | 82.5% | 70.0% | 27.5% / 1.25 | 60.0% | 152 (Metal) |
| lev thinker, thinking (greedy) | 85.0% | **90.0%** | **37.5% / 0.80** | **70.8%** | 3,001 (Metal) |

Localjev's own numbers on the same three tasks (different samples, M5
Max): Gemma 4 26B-A4B 75.0% macro at 0.68 s, DiffusionGemma 26B-A4B
74.2% at 1.2 s, with a 2,048-word distraction dropping everything.

The encoder is the fast path: it beats the 2.5B model answering at once on
the routing task by 15 points, at 125 ms on a CPU against 150 ms on a GPU
(~560 ms on a CPU), one forward for a whole workflow instead of one prompt
per question, with calibrated confidence. The thinker's thinking earns
its seconds on the reading-comprehension and reasoning cases.

## Candidates from RESEARCH-2026-09.md, measured

- **Option-permutation averaging** (the note's #3; the encoder's answer
  averaged over every rotation of the option list): 20 of 144 cases change
  their argmax under rotation (14%, the same share nibzard measured on Jev).
  Averaging moves `english` 61.1% → 63.9% and its ECE 0.101 → 0.082;
  `typed-decisions` 66.7% → 66.0%. Three forwards per question, one batch.
  Small and inconsistent.
- **Neutral-state prior subtraction** (jevmlx's correction; score the
  options against an empty state, subtract the log-odds): `english` 61.1%
  → 54.9%, and combined with averaging 62.5%. An empty state is out of the
  encoder's distribution; the correction hurts. Not worth it.
- **Qwen3-4B-Thinking-2507 as the thinker** (#5): 94.4%, the same as
  MiniCPM5-2B's 95.1%, at 13× the time (41 s a case, 1,883 thought tokens
  against 369). The 2.5B model is the escalation tier.
- **The gate on in-distribution traffic** (#4): the encoder's noul
  confidence is `max(p, 1 − p)`, never below 0.5, and its BoolQ answers are
  over-confident — at a 0.5 threshold every BoolQ case stays on the
  encoder (72.5% right) where the thinker gets 90%; at 0.7 it still keeps
  37 of 40. Choice and score confidences (1 − normalised entropy) behave:
  at 0.5 the gate keeps 36/40 AG News (the encoder's best task) and 13/40
  SST-5 (its worst). The gate needs a per-type threshold or a refit noul
  temperature before it can be trusted on yes/no questions.
- **Calibration** on authored144: encoder ECE 0.101 (typed-decisions
  0.133), von 0.124, the thinker 0.039 — but 97% of the thinker's answers
  come with confidence above 0.5, so its confidence cannot gate anything;
  the encoder's can (88% of these adversarial cases sit below 0.5, 12% of
  in-distribution AG News).

## The fixes, measured (`lev.calibrate`, per-type thresholds, `debias`)

`jolt -M:calibrate --labels bench/data/calib.jsonl` (1,240 public cases
from `bench/calib_data.py`: SST-2, AG News, dair-ai/emotion, 20
newsgroups, BoolQ, SST-5 — one per temperature bucket), fitted on the even
cases, reported on the odd ones, `english`:

| bucket | n | accuracy | T shipped → refit | NLL before → after | ECE before → after |
|---|---|---|---|---|---|
| choice:2 (SST-2) | 87 | 95.4% | 1.91 → 1.24 | 0.169 → 0.159 | 0.060 → 0.033 |
| choice:3-5 (AG News) | 97 | 95.9% | 1.76 → 1.47 | 0.161 → 0.145 | 0.047 → 0.021 |
| choice:6-10 (emotion) | 127 | 39.4% | 1.00 → 3.00 | 2.929 → 1.539 | 0.468 → 0.172 |
| choice:11+ (20 newsgroups) | 104 | 39.4% | 0.10 → 1.48 | 11.02 → 2.06 | 0.567 → 0.103 |
| noul:2 (BoolQ) | 104 | 77.9% | 1.98 → 2.70 | 0.534 → 0.476 | 0.114 → 0.090 |
| score:3-5 (SST-5) | 101 | 35.6% | 1.25 → 4.02 | 2.002 → 1.458 | 0.290 → 0.101 |

The shipped temperatures are sharp where the encoder is weakest (a 20-way
choice at 35% accuracy reported ~1.0 confidence). On the trio the refit
takes ECE 0.161 → 0.106 and the gate at 0.5 now escalates every SST-5 case
(13/40 were kept before) while AG News keeps 38/40 at 97%. BoolQ does not
move: the noul confidence floor is 0.5 and within the task confidence
barely tracks correctness (kept accuracy 72% at ≥0.5, 79% at ≥0.9), so
yes/no questions need their own, higher threshold — `threshold`
{"noul": 0.9} — or, on traffic like this, escalating them outright (the
thinker gets 90%).

`--debias` through the runner: `english` 63.9%, ECE 0.082, 199 ms a case
(the rotations share one batched forward).

What the numbers say:

- The gap to a hosted generative decision API on hard cases is the
  thinking, not the encoder. Answering directly, the 2.5B decoder is no
  better than von (73.6% vs 76.4%) at 3.5x the CPU time; with ~300 tokens
  of reasoning it gets 140/144, at 3 s a case on the GPU and ~10 s on the
  CPU.
- von-1.0 is the same ModernBERT-large encoder this port runs, with an NLI
  head instead of the encoders' marker head, one pair sequence per option
  instead of one sequence per question. +15 points over the `english`
  encoder on this set for ~1.4x the time. The encoder under-predicts the abstaining option
  (`insufficient` 18 times against 36 expected); von's failures are
  concentrated in rule application under perturbation (10/36).
- The two encoders fail on different cases: 41 von-right/english-wrong, 19 the
  other way, 15 both wrong, and the thinker gets all 15 of those. A
  confidence gate over the encoder's top-2 margin at 0.5 sends 92 of 144 cases up
  to the thinker and lands at 88.9%; over von's at 0.5, 92 up and 93.1%.
  On this set almost everything is low-confidence; on routine traffic the
  gate would pass most cases through at encoder speed.

The Jev benchmark from the other session (24/64 for the encoder vs 60/64 hosted Jev)
is the same shape: a generative model with a reasoning budget against a
single-pass encoder.

## Multi-question calls: the state tokenized once, the prefixes cached

`bench/workflow.clj` times the README's four email questions as one call
(choice, score, noul, noul) on a short state (53 tokens) and a long one
(1,631 tokens, cut to max_len 512 for every question), 30–40 iterations
after a warmup, `english` on the M-series CPU. Before, `encoder-forward`
tokenized the state once per question; after (laya-mlx's PrefixCache
ported: `lev.sequence/encode-state` once a call, `cached-prefix` in a
128-entry LRU per agent), once per call, and a question asked before
costs no tokenization at all. Answers are unchanged (golden/ pins them:
`prefix-and-state-assemble-into-build-sequence` proves the ids byte for
byte).

| call | before, p50 (two runs) | after, p50 (two runs) |
|---|---|---|
| tokenize the 1.6k-token state, alone | 57 ms | 58 ms |
| 4 questions, short state | 288 / 304 ms | 296 / 302 ms |
| 4 questions, long state | 1,392 / 1,381 ms | 1,247 / 1,239 ms |
| 1 question, long state | 400 / 374 ms | 391 / 398 ms |

The long-state call drops 10%: three of its four 48 ms state
tokenizations were redundant. The short-state case saves ~5 ms of a 300
ms call, under the run-to-run drift of the same code (288 vs 304 ms), so
no claim there. The tokenizer itself runs at ~33 µs a token, slow in
absolute terms (laya-jolt-edf); this removes the repetition, not the rate.

## Jev mode: the thinker's questions in one pass (2026-09-22)

`native/llama.cpp` is now thecodacus/llama.cpp's `parallel-decision`
branch (upstream b10435 + 617 commits, plus
`tools/parallel-decision/decision-engine.cpp`, linked into lev_llm by
`native/lev_decision.cpp`). With thinking off, `lev.think` hands every
question of a call to that engine (`lev.llm/jev`): the chat up to the
state is decoded once and kept across calls, the state continues it, and
each question is a branch sequence forked from the state (unified KV
cache, `llama_memory_seq_cp`), all of them in one `llama_decode`. Each
branch is the rest of that question's direct prompt through `ANSWER: `,
and its option ids (closed by `<|im_end|>`) are scored at every node where
their tokens diverge. So the prompts are byte for byte the ones the
per-question path scores (`think-test` proves it), the questions cannot
see each other there either, and what changes is the time and the
scoring. MiniCPM5-2B Q8, Metal, same machine:

| thinking off | authored144 | ECE | perturbations108 | trio (AG News / BoolQ / SST-5) | ms / case (authored144) |
|---|---|---|---|---|---|
| per-question path (`:jev false`), on the fork | 73.6% (106) | 0.193 | 60.2% (65) | 60.0% (72), ECE 0.281 | 139 |
| **Jev mode**, the ids start their own token (`:split-boundary true`, the default) | **75.0% (108)** | **0.174** | 61.1% (66) | 60.0% (72), ECE 0.279 | **100** |
| Jev mode, the engine's own cut (the id merged with the prefix's space) | 68.1% (98) | 0.226 | | | 100 |
| Jev mode, kev's SemIf prompt (JSON payload, option letters), either cut | 65.3% (94) | | | | |

Against the per-question path, Jev mode with lev's cut moves four cases
(+3 -1) and a probability by 0.011 at the median: the same answers,
normalised at the divergence nodes instead of over the whole vocabulary.
The per-question path read 74.3% on llama.cpp b10435 and 73.6% on the
fork: one case, the newer kernels. The engine's own cut, which tokenizes
`ANSWER: supported` whole so the id is scored as ` supported` (the
token the model would write), moves 38 cases the wrong way on balance
(+15 -23); on this model the id after a decoded space is better, so lev
passes the engine's `split_boundary`. kev's zero-shot SemIf prompt
(which took an instruct Qwen from 0.726 to 0.812 on kev's transfer suite)
loses ten points here: MiniCPM5 reads lev's prompt better.

Where it pays is a call with several questions (`bench/workflow.clj
--model <thinker>`, the four email questions, 20 iterations, p50):

| call | per-question | Jev mode |
|---|---|---|
| 4 questions, short state | 539 ms | **207 ms** |
| 4 questions, long state (1.6k-token email) | 4,317 ms | **1,177 ms** |
| 1 question, long state | 1,096 ms | 1,076 ms |

The per-question path decodes the whole prompt four times; Jev mode
decodes the system turn once for the server's life, the state once a
call, and the four questions' ~40-token tails together. One question on a
long state is the state's prefill either way. Through `lev-server`, three
questions on a short ticket answer in 170 ms warm.

Several states against the same questions (`/v1/systemone/batch`,
`lev.agent/system-one-batch`) go to the engine as one call. With this
layout it saves little: 8 short states take 1,671 ms against 1,681 ms one
call at a time. Every question's ~50-token tail is decoded again for every
state, so the work is compute-bound either way. The catalog layout
(`:layout "catalog"`: the questions go in the cached text before the
state, and each field is a short `ANSWER <n>: `) is what batching is for.
It takes 565 ms for the 8 states and 104 ms for four questions on one,
but authored144 falls to 48.6% (balanced 43.3%). A batched state's
probabilities match the same state answered alone to within 0.007
(`llm-test`): llama.cpp is not batch-invariant.

## Other models in Jev mode, and SemIf (2026-09-22)

Same benches, thinking off, Metal. Qwen2.5-1.5B-Instruct is what
harshatheg/Qwen-2.5-1B-RLCD runs: that repo holds no weights, only the
parallel-decoding code over the stock model. Qwen3.5-4B is the model SemIf
(github.com/TheoLeeCJ/SemIf, #2 on JevBench v1.3.0) uses. Both run as
thinkers: Qwen2.5 with `:thinks false`, since it has no thinking mode,
and Qwen3.5 with lev's ChatML (its template's no-thinking turn is the same
`<think>\n\n</think>\n\n`).

| model, prompt | authored144 | balanced | ECE | perturbations108 | trio | 4 questions, short state |
|---|---|---|---|---|---|---|
| MiniCPM5-2B Q8, lev prompt (above) | 75.0% (108) | 71.5% | 0.174 | 61.1% | 60.0% (72) | 207 ms |
| MiniCPM5-2B Q8, SemIf prompt (`:prompt "semif"`) | 64.6% (93) | 63.1% | 0.234 | | | |
| Qwen2.5-1.5B-Instruct Q8, lev prompt, the engine's own cut | 59.7% (86) | 65.5% | 0.305 | 69.4% | 60.0% (72) | 131 ms |
| Qwen2.5-1.5B-Instruct Q8, lev prompt, lev's cut | 56.3% (81) | 61.7% | 0.418 | | | |
| Qwen2.5-1.5B-Instruct Q8, the RLCD repo's prompt | 54.2% (78) | | | | | |
| **Qwen3.5-4B Q8, lev prompt** | **95.1% (137)** | **93.3%** | **0.034** | **79.6%** | **74.2% (89)** | 533 ms |
| Qwen3.5-4B Q8, lev prompt, per question (`:jev false`) | 95.1% (137) | 93.3% | 0.034 | | | |
| Qwen3.5-4B Q8, SemIf prompt | 79.2% (114) | 78.7% | 0.069 | | | |
| Qwen3.5-4B Q8, SemIf prompt, per question | 80.6% (116) | 80.1% | 0.058 | | | |
| SemIf's own scorer, its llama.cpp backend, the same Q8 GGUF | 80.6% (116) | 79.8% | 0.062 | | | 1.9 s a case (CPU) |
| SemIf's committed BF16 predictions (RTX 3090) | 80.6% (116) | 80.0% | 0.068 | | | |

The last three rows answer whether lev or the model was the limit. SemIf's
own code on the same GGUF gets 116/144, what lev gets with its prompt, and
what SemIf published from BF16. The SemIf row's balanced accuracy uses
lev's metric (mean recall per label); SemIf reports 0.813 by its own. lev's
prompt adds 14.5 points on the same model. Most of that is
candidate_selection: 48/48 with lev's prompt, 32/48 with SemIf's. There
the options are candidates named A and B, which SemIf relabels with its
own letters. Qwen3.5-4B with thinking off matches MiniCPM5 with thinking
(95.1%) at 174 ms a question instead of 3 s.

Jev mode and the per-question path agree case for case on Qwen3.5 (the
lev-prompt rows), so `llama_memory_seq_cp` on its hybrid memory holds up.
SemIf's llama.cpp backend avoids it and serializes whole states. What
the hybrid costs is batching. Its recurrent layers need every sequence in a
ubatch to be the same length, so llama.cpp splits the branches into
separate passes: four questions take 533 ms against 174 ms for one. The
catalog layout (short, equal branches) runs 8 states in 1,244 ms instead
of 4,197 ms, but scores 72.9% on authored144, so it stays opt-in.

Escalation (`lev.patterns/escalate`), simulated from per-case results:
the `english` encoder answers first, and the cases below a threshold on
its confidence take the thinker's answer instead.

| gate | escalated | to Qwen3.5-4B (answering at once) | to MiniCPM5-2B (answering at once) |
|---|---|---|---|
| 0.3 | 99 / 144 | 89.6%, 234 ms a case on average | 75.7%, 178 ms |
| 0.5 | 126 / 144 | **92.4%**, 270 ms | 74.3%, 196 ms |
| 0.7 | 141 / 144 | 95.1%, 289 ms | 75.0%, 207 ms |

Qwen3.5-4B is the escalation model from here on (README, config
examples). MiniCPM5-2B escalated with thinking gets 88.9% at 0.5, at
seconds a case (above). Qwen3.5-4B with its own thinking on has not been
measured.

Not ported from SemIf: its per-workload temperature scaling. `lev.calibrate`
refits the encoders only. At ECE 0.034 Qwen3.5-4B needs it least.

## typed-decisions (2026-09-29)

`bench/typed_decisions.clj` on the test split of
[LocalLLaMA/typed-decisions](https://huggingface.co/datasets/LocalLLaMA/typed-decisions):
400 JSON states from four workflows, five questions each, 2,000 decisions.
Build the data with `uv run --with pyarrow python bench/typed_decisions_data.py`.
The gold is the mean of three teacher samples, so accuracy (argmax against
the gold's label) measures agreement with the teacher. The teacher agrees
with itself 0.735 of the time, and the prior scores 0.470. Jev scores 0.727
on the dataset's card and 0.738 in Winnow's report. ECE uses 15 bins, as
ollaya does.

| model | accuracy | choice | score | noul | KL | Brier | ECE | noul true (gold 50.8%) |
|---|---|---|---|---|---|---|---|---|
| encoder `typed-decisions` (trained on this train split) | **0.766** | 0.733 | 0.723 | 0.857 | **0.117** | **0.061** | 0.213 | 51.5% |
| thinker Winnow-E4B Q8, `winnow` prompt, `gemma4`, T 1.2574 (the author's) | **0.724** | 0.697 | 0.686 | 0.802 | 0.294 | 0.132 | **0.025** | 60.7% |
| Clef Q8 (lev.clef: Qwen3.8-27B + joint schema head), T 1 | 0.722 | 0.643 | 0.688 | 0.847 | 0.191 | 0.103 | **0.024** | 57.8% |
| thinker Qwen3.8-27B Q8, lev prompt, Jev mode, T 1 | 0.695 | 0.608 | 0.699 | 0.777 | 0.844 | 0.233 | 0.149 | 52.5% |
| thinker JevK5 v0.3 4B Q8, `jevk5` prompt, T 1.22 (the author's) | 0.626 | 0.567 | 0.578 | 0.750 | 0.406 | 0.198 | 0.096 | 58.2% |
| thinker Qwen3.5-4B Q8, lev prompt, Jev mode, fitted T (5.07 / 3.82 / 2.88) | 0.588 | 0.560 | 0.526 | 0.698 | 0.287 | 0.152 | 0.058 | 52.7% |
| thinker Qwen3.5-4B Q8, lev prompt, Jev mode, T 1 | 0.588 | 0.560 | 0.526 | 0.698 | 1.045 | 0.322 | 0.232 | 52.7% |
| encoder `english` | 0.362 | 0.288 | 0.323 | 0.487 | 0.569 | 0.316 | **0.175** | 67.8% |
| encoder `multilingual` | 0.352 | 0.295 | 0.286 | 0.497 | 1.089 | 0.463 | 0.314 | 77.2% |

The encoders reproduce ollaya's numbers: 0.766 for `typed-decisions` and
0.361 for `english` in Winnow's report. `typed-decisions` beats the
teacher's own self-agreement, so it has learned the teacher's quirks
along with the task.

The two prompts ported from ollaya's reference code reproduce its
numbers on the fine-tunes they belong to. JevK5 scores 0.626 (ollaya
0.625, ECE 0.093). Winnow-E4B scores 0.724 (ollaya 0.722, ECE 0.022).
Winnow-E4B, a Gemma 4 E4B fine-tune, is the best general model here,
within 0.014 of Jev, and it is also the best calibrated.

Qwen3.5-4B, the escalation model, is well behind on this set. That is in
line with ollaya's generic `llm-logits` reads of 4B instruct models
(Qwen3-4B 0.559). At T 1 it puts nearly all its probability on one
answer (KL 1.045, ECE 0.232). Temperatures fitted by `jolt -M:calibrate
--model qwen3.5-4b` take ECE to 0.058 and KL to 0.287 without moving an
answer. The fit used 200 of the train split's cases, fitted per type,
with the other 200 held out: ECE went 0.363 → 0.129 for choice, 0.157 →
0.055 for noul and 0.238 → 0.076 for score.

The two general encoders answer yes to 68–77% of the noul questions,
against a gold 51%. The two fine-tuned thinkers lean the same way less
(58–61%), and Qwen3.5-4B does not.

Not recorded: latency. The GPU was shared with another session's bench
during these runs.

## Winnow-E4B on lev's benches (2026-09-29)

Winnow-E4B Q8 (the `winnow` prompt, `gemma4` template, T 1.2574, Jev
mode, Metal) against Qwen3.5-4B Q8, the escalation model (lev prompt,
numbers from the sections above).

| | authored144 | balanced | ECE | perturbations108 | trio (AG News / BoolQ / SST-5) | typed-decisions |
|---|---|---|---|---|---|---|
| Qwen3.5-4B | **95.1% (137)** | 93.3% | 0.034 | 79.6% | **74.2% (89)**: 87.5 / 87.5 / 47.5 | 0.588 |
| Winnow-E4B | 93.1% (134) | **93.8%** | 0.034 | **97.2% (105)** | 71.7% (86): 85.0 / 80.0 / 50.0 | **0.724** |

The two models' authored144 escalation, simulated from per-case
results: `english` answers first, and the cases below the gate take the
thinker's answer.

| gate | escalated | to Qwen3.5-4B | to Winnow-E4B |
|---|---|---|---|
| 0.3 | 99 / 144 | 89.6% | 86.1% |
| 0.5 | 126 / 144 | 92.4% | 90.3% |
| 0.7 | 141 / 144 | 95.1% | 92.4% |

Winnow-E4B gives up three authored144 cases and three trio cases to
Qwen3.5-4B. It holds up far better on the output-blind perturbations of
authored144 (+17.6 points) and on typed-decisions (+13.6). It is
calibrated as shipped (ECE 0.025 on typed-decisions against Qwen's 0.232
before a fit). It is also nearly twice the file (8.0 GB against 4.5 GB),
and its Gemma 4 sliding-window layers want a small `:n-seq-max` (8
here).

Latency is not compared: the GPU was shared with another session's
bench (Winnow-E4B read 470 ms a case on authored144 under that load). So
the escalation default stays Qwen3.5-4B until both are timed on a quiet
machine, with Winnow-E4B the choice where robustness to rephrasing or
structured states matters more than authored144.

## Clef on lev's benches (2026-10-07)

[Clef](https://huggingface.co/Cloudflare/clef) is Cloudflare's decision
model: Qwen3.8-27B post-trained, with a joint schema head (a 128M-parameter
transformer) in place of generation. The state and every question with its
options go into one prompt. The head reads the backbone's final hidden
state at every token, routes evidence to each option, lets the questions
attend to each other and answers one logit per option of every question
at once. No text is generated. In lev it is `lev.clef`, a thinker entry
with `:engine "clef"`:

```
python native/llama.cpp/convert_hf_to_gguf.py ~/src/models/clef --no-mtp --outtype q8_0 \
  --outfile ~/src/models/clef-backbone-Q8_0.gguf
;; config.edn :thinkers
"clef" {:engine "clef" :model ".../clef-backbone-Q8_0.gguf" :head ".../clef"}
```

The hidden states come from the fork's unmasked nextn embeddings
(`native/lev_clef.cpp`). The head runs in f32 over cblas and is held to
the release's torch head at 1e-4 (`bench/clef_golden.py` ->
`golden/clef`). The prompt matches the release tokenizer's token for
token. Answers are T 1, as shipped.

Against the escalation model, Winnow-E4B (sections above) and Clef's own
base as a thinker answering at once (lev prompt, Jev mode, T 1). M1 Max,
Metal, Q8 throughout.

| | size | authored144 | balanced | ECE | perturbations108 | trio (AG News / BoolQ / SST-5) | typed-decisions (ECE) | ms a case (authored144) |
|---|---|---|---|---|---|---|---|---|
| Qwen3.5-4B | 4.5 GB | **95.1% (137)** | 93.3% | 0.034 | 79.6% | 74.2% (89): 87.5 / 87.5 / 47.5 | 0.588 (0.232) | **184** |
| Winnow-E4B | 8.0 GB | 93.1% (134) | 93.8% | 0.034 | 97.2% (105) | 71.7% (86): 85.0 / 80.0 / 50.0 | **0.724** (0.025) | |
| Qwen3.8-27B | 28.6 GB | 89.6% (129) | 88.6% | 0.037 | 92.6% (100) | **77.5% (93)**: 87.5 / 92.5 / 52.5 | 0.695 (0.149) | 1,506 |
| Clef | 28.6 GB | 93.1% (134) | **95.7%** | 0.053 | **100% (108)** | 75.8% (91): 87.5 / 92.5 / 47.5 | 0.722 (**0.024**) | 1,612 |

authored144 escalation, simulated from per-case results as above
(`english` first, the cases below the gate take the other model's answer):

| gate | escalated | to Qwen3.5-4B | to Winnow-E4B | to Qwen3.8-27B | to Clef |
|---|---|---|---|---|---|
| 0.3 | 99 / 144 | 89.6% | 86.1% | 83.3% | 86.1% |
| 0.5 | 126 / 144 | 92.4% | 90.3% | 86.8% | 90.3% |
| 0.7 | 141 / 144 | 95.1% | 92.4% | 88.9% | 92.4% |

Clef's post-training and head are worth a lot over its base. Against
Qwen3.8-27B it gains 5 authored144 cases (rule application 75.0% ->
91.7%), 8 perturbations (to every one of them) and 0.027 typed-decisions
accuracy. Its typed-decisions KL drops from 0.844 to 0.191 and its ECE
from 0.149 to 0.024, with no refit. It gives two SST-5 cases back on the
trio. Its confidence is informative: at >= 0.7 on authored144 it keeps
113 of 144 cases at 100%.

Against the 4B models, Clef ties Winnow-E4B on authored144 (93.1%) and
on typed-decisions (0.722 / 0.724). It is ahead on the perturbations and
on the trio, and it is the best-calibrated general model here. Qwen3.5-4B
keeps two authored144 cases on it, and Clef is behind it in every
escalation row.

Clef does not become the escalation model. It is 6x the memory of
Qwen3.5-4B and 9x its time a case on authored144 (1.6 s against 184 ms,
timed back to back on the same machine; 8.8 s a five-question
typed-decisions state), and it loses authored144. Where it earns its cost is a decision model on its own:
rephrased or structured inputs, several questions decided together, and
probabilities that can be trusted as shipped. The README's Decision Index
lead over Jev does not carry over at this scale of test: on
typed-decisions Clef and Jev's published 0.727 are even.

## Measuring a change: `bench/paired.clj`

A sequential before/after (two processes, one after the other) confounds
the change with the machine's drift: above, the same code moved 288 →
304 ms between two runs, and laya-mlx's sequential pilot put `mx.compile`
at 1.24x where its interleaved run found 1.03x. `bench/paired.clj` is
that interleaved run, ported: every round takes one input (rotating over
8 state variants) and times every candidate on it in an order that
rotates by one each round; the report is the median of the per-round
`baseline / candidate` ratios with a percentile-bootstrap 95% interval
(2,000 resamples), next to the p50s. An interval that includes 1 is no
win. `cpu,cpu` is the noise floor, the same agent against itself:

```
jolt -M bench/paired.clj --candidates cpu,cpu --rounds 20
  short4 (4 questions)   cpu/cpu#2  paired speedup 1.000x  [0.998, 1.007]   per-round 0.989..1.207
  long4  (4 questions)   cpu/cpu#2  paired speedup 0.990x  [0.979, 1.016]   per-round 0.917..1.175
  long1  (1 question)    cpu/cpu#2  paired speedup 1.007x  [0.997, 1.021]   per-round 0.862..1.125
```

Single rounds swing by 20%; twenty paired rounds pin the ratio to ±2%.
Anything that changes the forward (a backend, a pruned layer) is measured
here before it lands.

## The MLX backend, measured (`--backend mlx`, mac)

`jolt mlx` builds the encoders' GPU engine (lev.mlx over mlx-c, the
same forward as the C kernels as one MLX graph). Interleaved against the
C kernels with `bench/paired.clj --candidates cpu,mlx,mlx16 --rounds 20`
on the M-series this tree is developed on, `english`:

| call | cpu p50 | mlx f32 p50 | paired speedup | mlx f16 p50 | paired speedup |
|---|---|---|---|---|---|
| 4 questions, short state (53 tokens) | 270 ms | 68 ms | **3.98x** [3.85, 4.02] | 54 ms | **5.05x** [4.97, 5.07] |
| 4 questions, long state (512 tokens a row) | 1,126 ms | 344 ms | **3.28x** [3.21, 3.46] | 276 ms | **4.07x** [3.97, 4.12] |
| 1 question, long state | 373 ms | 142 ms | **2.62x** [2.53, 2.77] | 121 ms | **3.08x** [2.94, 3.15] |

Accuracy, the same runners as above:

| | authored144 | ECE | ms/case | trio | ECE | ms/case |
|---|---|---|---|---|---|---|
| cpu f32 | 61.1% (88) | 0.101 | 101 | 65.8% (79/120) | 0.161 | 118 |
| mlx f32 | 61.1% (88) | 0.101 | 24 | 65.8% (79/120) | 0.161 | 29 |
| mlx f16 | 61.8% (89) | 0.105 | 22 | 65.8% (79/120) | 0.161 | 25 |

f32 is the C kernels' answer to the fourth decimal (lev.mlx-test holds
the golden batch to 1e-4 and the README answers to the oracle); f16
moves one authored144 case (in its favour, by luck) and probabilities by
up to 1e-2. laya-mlx's own M3 Max numbers for the same architecture at
f16 (13 ms a short single question, batch of one) are of the same order
once its lighter per-call path is accounted for; lev's call carries the
tokenizer (~1 ms a question), the calibration and the answer shaping.
Not ported from laya-mlx, with its evidence: `mx.compile` (1.03x
interleaved), a custom Metal GELU kernel (no consistent win over the
compiled graph), 8/4-bit weights (no speedup at these shapes and 62/63,
50/63 fixture agreement), low-rank weights (84% Frobenius error at the
rank a 10x needs).

## The last head layer, pruned to the rows that are read

Only each row's CLS and marker rows of the last head layer reach the
scorer and the act head, so its out-projection, second norm and FFN (9
of the layer's 12 d² a row) run on those 1+k rows instead of all L
(laya-mlx's `selected_head`; exact dependency pruning, the golden batch
and the oracle answers unchanged). Interleaved against the whole layer,
20 rounds:

| call | C kernels: whole → pruned | paired speedup | MLX f32: whole → pruned | paired speedup |
|---|---|---|---|---|
| 4 questions, short state | 264 → 261 ms | 1.015x [1.012, 1.016] | 68 → 65 ms | 1.047x [1.017, 1.062] |
| 4 questions, long state | 1,105 → 1,078 ms | 1.025x [1.023, 1.027] | 314 → 309 ms | 1.014x [1.014, 1.016] |
| 1 question, long state | 365 → 359 ms | 1.018x [1.014, 1.021] | 135 → 132 ms | 1.012x [1.007, 1.034] |

Small, every interval above 1, and it costs nothing: on by default on
both backends (`:selected-head false` in the prepared config or, for
lev.mlx, the loader's limits keeps the whole layer; `bench/paired.clj
--candidates fullhead,cpu` / `mlxfull,mlx` is the comparison).
