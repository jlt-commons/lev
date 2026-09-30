# Lev

Lev is a System One decision engine which accepts typed questions
over a state, and answers with calibrated probabilities. It
runs on [jolt](https://github.com/jolt-lang/jolt), which hosts Clojure on
Chez Scheme.

Two kinds of model sit behind Lev API:

- **the encoders**: ModernBERT-large / mmBERT-base with a decision head,
  from the `convaiinnovations/laya` checkpoints on the Hub.
  These use one forward pass covers all of a call's questions, at roughly 100 ms on
  a laptop CPU, and the probabilities are calibrated.
- **a thinker**: any GGUF chat model through a linked llama.cpp.
  It reads the state and the question, thinks, then has its
  candidate answers scored by their token log probabilities. On the
  adversarial authored144 set in [bench/](bench/README.md), Qwen3.5-4B
  answers 95% without thinking, in Jev mode at ~170 ms a question, against
  the encoders' 61 to 67%. MiniCPM5-2B reaches the same 95% by thinking
  first, in seconds.

A request either names its model, like `"model": "english"` or
`"model": "qwen3.5-4b"`, or gets routed by content to an encoder. A
confidence gate can answer on the encoder first, then escalate only what
it is unsure about to the thinker. Qwen3.5-4B is the escalation model:
behind the `english` encoder, a gate at 0.5 answers authored144 at 92.4%
in 270 ms a case on average. Either kind can be configured individually
in the configuration. You can configure just the encoders, or only a thinking model.

Which one should answer, and when? The numbers below come from a
120-case in-distribution set made of AG News, BoolQ and SST-5, the trio
localjev's bake-off uses, plus von's adversarial authored144. Details are
in [bench/](bench/README.md):

| | AG News | BoolQ | SST-5 | authored144 | ms per question |
|---|---|---|---|---|---|
| encoder `english` | **97.5%** | 72.5% | 27.5% | 61% | **125** (CPU) |
| encoder `english`, `debias` | | | | 64% | 199 (CPU) |
| thinker Qwen3.5-4B, answering at once (the escalation model) | 87.5% | 87.5% | **47.5%** | **95%** | 174 (GPU) |
| thinker MiniCPM5-2B, answering at once | 80.0% | 75.0% | 25.0% | 75% | 100 (GPU) |
| thinker MiniCPM5-2B, thinking | 85.0% | **90.0%** | 37.5% | **95%** | 3,000 (GPU) |

The encoder is the fast path. On routing-style traffic such as AG News it
beats the generative models answering at once, at a fraction of the cost
on a CPU, batching a whole workflow into one forward pass. Its confidence means is based on
the calibration which the gate relies on. The thinker is used for
the cases that need a deduction or an abstention.

The encoders are f32 end to end. F16 checkpoint weights are widened once,
during `prepare`, so the numerics match the torch CPU oracle.

## Getting the checkpoints

The weights are not in this repo, and need to be downloaded from the [Laya Hugging Face repo](https://huggingface.co/convaiinnovations/laya). The three
checkpoints are:

| name | where in the repo | encoder | context | for |
|---|---|---|---|---|
| `english` | the root | ModernBERT-large, 421M | 512 | English text |
| `typed-decisions` | `typed-decisions/` | ModernBERT-large, 421M | 1024 | the four typed-decisions workflows (invoice processing, security incidents, customer service, agent-trace observability) |
| `multilingual` | `multilingual/` | mmBERT-base, 322M | 1024 | 100+ languages (Gemma sentencepiece tokenizer, 256k vocab) |

`jolt prepare` reads five files per checkpoint from a directory laid out
the same way as the repo. The root is english, and the subfolders are optional.

```
../laya/
  model.safetensors          # ~800 MB, F16
  tokenizer/tokenizer.json
  tokenizer/tokenizer_config.json
  encoder/config.json
  rl_agent_config.json
  typed-decisions/           # same five files, optional (~800 MB)
  multilingual/              # same five files, optional (~640 MB)
```

Trim the first loop to the checkpoints
you want, where the `""` entry is english:

```bash
for sub in "" typed-decisions/ multilingual/; do
  mkdir -p ../laya/${sub}tokenizer ../laya/${sub}encoder
  for f in model.safetensors tokenizer/tokenizer.json tokenizer/tokenizer_config.json encoder/config.json rl_agent_config.json; do
    curl -fL -o ../laya/$sub$f https://huggingface.co/convaiinnovations/laya/resolve/main/$sub$f
  done
done
```

You can also clone the whole model repo with git-lfs, or use the Hub CLI:

```
git lfs install && git clone https://huggingface.co/convaiinnovations/laya ../laya
hf download convaiinnovations/laya --local-dir ../laya
```

Put it anywhere and point `LEV_CHECKPOINTS` at it, or run
`jolt -M:prepare --checkpoints DIR --out data`. `jolt prepare` converts
every checkpoint it finds into the same layout under the data root, so
`data/`, `data/typed-decisions`, `data/multilingual`. Add `--model NAME`
to convert one. A root missing any of the five files is refused, with an
error message.

### A thinker's model

Any chat GGUF that llama.cpp can load works. The escalation model is
Qwen3.5-4B
([bartowski/Qwen_Qwen3.5-4B-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3.5-4B-GGUF),
Apache-2.0, `Q8_0` 4.5 GB). It scores 95% on authored144 answering at
once, at 174 ms a question, and is the best Jev-mode model on every
bench (`bench/README.md`). Its hybrid layers make a call with several
questions slower: 533 ms for four. It can also think, but that has not
been measured, so configure it with `:thinking false`.
[openbmb/MiniCPM5-2B-GGUF](https://huggingface.co/openbmb/MiniCPM5-2B-GGUF)
(2.5B, Apache-2.0, `Q8_0` 2.7 GB) is the measured thinking model: 95%
with a thought, in seconds, and 75% answering at once. For a model
without a thinking mode, such as Qwen2.5-Instruct, set `:thinks false`.
Where robustness matters more than speed, [EldanRing/Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B)
(`gguf/Winnow-E4B-Q8_0.gguf`, 8.0 GB) with the `winnow` prompt scores
97.2% on authored144's rephrasings against Qwen3.5-4B's 79.6%, and 0.724
on typed-decisions against 0.588. It gives up 2 points on authored144
itself and is larger and slower (Thinkers, below).

```
hf download bartowski/Qwen_Qwen3.5-4B-GGUF Qwen_Qwen3.5-4B-Q8_0.gguf --local-dir ~/models
hf download openbmb/MiniCPM5-2B-GGUF MiniCPM5-2B-Q8_0.gguf --local-dir ~/models   # optional: the thinking model
```

Then name it in `~/.config/lev/config.edn` under `:thinkers`, covered
below, or pass `--thinker ~/models/Qwen_Qwen3.5-4B-Q8_0.gguf`, or set
`LEV_THINKER`. The prompt format is ChatML with the `<think>` switch both
models' templates use. Another model family needs its own template, which lives in
`lev.think/defaults`.

## Build and run

```
jolt kernels             # compile native/lev_kernels.c
jolt llama               # fetch + build llama.cpp (the parallel-decision fork, pinned commit) into native/liblev_llm.*, the thinker's native
jolt mlx                 # mac: clone + build mlx-c (pinned tag) into native/liblev_mlx.*, the encoders' GPU backend
jolt prepare             # every checkpoint under ../laya -> data/, data/typed-decisions, ...
jolt -M:test             # parity suites vs golden/
jolt -M:run demo         # README quickstart through the workflow runner
jolt -M:serve            # HTTP API on http://127.0.0.1:8080
jolt -M:calibrate --labels cases.jsonl --out calibration.edn   # refit an encoder's temperatures on your labeled traffic
jolt binary              # standalone ./lev-server (kernels + llama.cpp linked in, mlx too when built), self-tested against golden/
```

`jolt -M:test` runs everything against whatever is prepared under `data/`.
Use `jolt -M:test lev.checkpoints-test` for one namespace.
`LEV_TEST_CHECKPOINTS=typed-decisions` restricts the extra-checkpoint
parity suite. The variable is comma-separated, and empty means none, which
is how CI tests one checkpoint per process.

`jolt -M:run demo` prints the quickstart answer JSON. The `:system-one`
value should match `golden/readme.edn` to the fourth decimal, see Status.

Answers come back as ordered maps with string keys.
Option order and question order are part of the model input,
so pass `:criteria` and the questions map as ordered maps, either an
`array-map` or a literal with at most 8 entries to ensure ordering.

```clojure
(require '[lev.agent :as ag] '[lev.workflows :as wf])
(def agent (ag/load-agent "data"))
(def email (wf/load-workflow "workflows/email.clj"))          ; or (wf/load-workflows dirs)
(ag/system-one agent
               (wf/state email {"subject" "Duplicate billing" "body" raw-body "from" "customer@acme.com"})
               (wf/questions email))
```

## Configuration: `~/.config/lev`

Every entry point, whether `jolt prepare`, `jolt -M:run`, `jolt -M:serve`,
`jolt -M:calibrate`, the bench runners or the binary, resolves its
settings the same way. CLI flag beats environment variable, environment
variable beats `config.edn`, and `config.edn` beats the default. The file
lives in `$LEV_CONFIG_DIR`, else `$XDG_CONFIG_HOME/lev`, else
`~/.config/lev`:

```clojure
{:data "/Users/me/models/lev-data"       ; prepared data root: what jolt prepare writes and everything else loads
 :checkpoints-home "/Users/me/models/laya"  ; the Hub checkpoints jolt prepare reads
 :encoders {"english" "/Users/me/models/lev-data"}   ; prepared encoders by name (else the :data layout); leave one out to not serve it
 :calibration {"english" "/Users/me/models/calibration-english.edn"}   ; refit temperatures per encoder (see Calibration)
 :thinkers {"qwen3.5-4b" {:model "/Users/me/models/Qwen_Qwen3.5-4B-Q8_0.gguf" :thinking false}}   ; generative models (see Thinkers); none = encoders only
 :workflow-dirs ["/Users/me/src/decisions/workflows"]   ; extra workflow directories
 :port 8080 :host "127.0.0.1" :api-key "s3cret"        ; server defaults
 :max-loaded 2 :max-thinkers 1                          ; encoders / thinkers kept resident
 :default-model "english"                               ; loaded at startup; content routing's fallback (an encoder or a thinker)
 :auto-task-detection false                             ; route typed-decisions question sets to that checkpoint
 :max-len 768 :head-max-len 192                         ; sequence limits for every checkpoint (see Context)
 :backend "mlx" :dtype "f16"                            ; the encoders' engine: cpu (the C kernels) or mlx (Apple's GPU, jolt mlx), f32 or f16 (see Speed or accuracy)
 :checkpoints {"multilingual" {:max-len 2048 :backend "cpu"}}}   ; ... and per checkpoint, which wins
```

A server needs at least one model of either kind, and serves whatever is
available. With no thinker, `model` names only encoders. With no prepared
encoder and a thinker as `:default-model`, every request including the
content-routed ones goes to the thinker. A request for a model that is
configured but unavailable, from a missing data directory, a missing GGUF
or an unbuilt llm native, is a 503. `GET /v1/models` says which is which.

| setting | flag | environment | `config.edn` | default |
|---|---|---|---|---|
| prepared data root | `--data DIR` | `LEV_DATA` | `:data` | `data` |
| checkpoints (prepare) | `--checkpoints DIR` | `LEV_CHECKPOINTS` | `:checkpoints-home` | `../laya` |
| encoders served | n/a | n/a | `:encoders {"name" dir}` | the `:data` layout |
| calibration | `--calibration FILE` (every encoder) | `LEV_CALIBRATION` | `:calibration {"name" file}` | the checkpoint's own temperatures |
| thinkers served | `--thinker PATH.gguf` (as `thinker`) | `LEV_THINKER` | `:thinkers {"name" {...}}` | none |
| workflow dirs | `--workflows DIR[:DIR]` | `LEV_WORKFLOWS` | `:workflow-dirs` (adds) | see below |
| server | `--port` `--host` `--api-key` | `PORT` `LEV_HOST` `LEV_API_KEY` | `:port` `:host` `:api-key` | `8080` `127.0.0.1` none |
| resident encoders / thinkers | `--max-loaded N` / `--max-thinkers N` | `LEV_MAX_LOADED` / `LEV_MAX_THINKERS` | `:max-loaded` / `:max-thinkers` | `1` / `1` |
| startup / fallback model | `--default-model NAME` | `LEV_DEFAULT_MODEL` | `:default-model` | `english` |
| typed-decisions by question ids | `--auto-task-detection` | n/a | `:auto-task-detection` | off |
| sequence limits | `--max-len N` `--head-max-len N` | `LEV_MAX_LEN` `LEV_HEAD_MAX_LEN` | `:max-len` `:head-max-len`, `:checkpoints {"name" {…}}` | the checkpoint's own (`rl_agent_config.json`) |
| encoder engine | `--backend cpu\|mlx` `--dtype f32\|f16` | `LEV_BACKEND` `LEV_DTYPE` | `:backend` `:dtype`, `:checkpoints {"name" {…}}` | `cpu`, `f32` |
| goldens (`--self-test`) | `--golden DIR` | `LEV_GOLDEN` | `:golden` | `golden` |

`--workflows` and `LEV_WORKFLOWS` are the exception to "adds". They name
 the directories to scan, replacing the defaults, so a test or a
one-off run stays isolated from whatever is in `~/.config/lev`.

### Calibration

The checkpoints' temperatures were fitted on their training data. On
other traffic the probabilities can be far off. On BoolQ the english
encoder's yes/no answers average 0.92 confidence at 72.5% accuracy, and
its 20-way choices sit at 0.99 confidence with 35% accuracy.

`jolt -M:calibrate` refits one temperature per question type and
option-count bucket on labeled cases, by minimising NLL. It reports NLL
and ECE per bucket on a held-out half, then writes a file that
`config.edn` under `:calibration {"english" "calibration.edn"}` loads
over the checkpoint's own. `--calibration FILE` and `LEV_CALIBRATION` do
the same for every encoder.

A refit changes no answer. Temperature scaling keeps every argmax, so
only what `probabilities` and `confidence` mean which the gate reads moves.
Labeled cases are JSON lines:

```json
{"state": "...", "questions": {"q": {"type": "noul", "instructions": "..."}}, "labels": {"q": true}}
```

A choice's label is its option, a score's is the level index, and a
noul's is the boolean. `bench/calib_data.py` builds 1,240 such cases from
public datasets covering every bucket. On their held-out half the refit
takes the 20-way choice bucket's ECE from 0.57 to 0.10 and the 6-way from
0.47 to 0.17, and the score bucket from 0.29 to 0.10. On the
in-distribution trio the encoder's ECE drops from 0.16 to 0.11. Full
numbers are in `bench/README.md`.

A case can also carry `"targets": {"q": [p, ...]}`, a gold distribution
in answer order (a noul's is `[false, true]`). The fit then minimises the
cross-entropy against it instead of the label's NLL.
`bench/typed_decisions_data.py` writes the typed-decisions train split
this way, as `bench/data/typed-decisions-calib.jsonl`.

A thinker is refit the same way, with `--model qwen3.5-4b`. Its raw
option scores are much more overconfident than an encoder's. ollaya
measured ECE around 0.4 at T=1 for 4B instruct models on typed-decisions,
and 0.08 after fitting one temperature per type. The file goes under the
thinker's name in `:calibration`, or inline as the thinker's
`:calibration`, which can be one temperature, three (choice, score,
noul), a calibration map or its file's path.

### Thinkers

```clojure
{:thinkers {"qwen3.5-4b" {:model "/Users/me/models/Qwen_Qwen3.5-4B-Q8_0.gguf" :thinking false}   ; the escalation model, answering at once
            "minicpm5" {:model "/Users/me/models/MiniCPM5-2B-Q8_0.gguf"   ; a thinking model, every key at its default:
                        :thinking true          ; think before answering (a request can override)
                        :max-think-tokens 2048  ; the budget; the thought is closed when it runs out
                        :n-ctx 4096 :n-gpu-layers -1 :threads 0   ; llama.cpp: context, layers on the GPU (-1 all), threads (0 = its default)
                        :temperature 0.0 :top-p 0.95 :min-p 0.0 :seed 42     ; the thought: greedy by default (95% vs 91% sampled on authored144)
                        :jev true               ; thinking off: every question in one pass (Jev mode, below)
                        :split-boundary true    ; Jev mode: option ids start their own token (75% vs 68% on authored144)
                        :thinks true            ; false: no thinking mode (no <think> tags; thinking stays off)
                        :prompt "lev"           ; or "semif", "jevk5", "winnow" (below)
                        :template "chatml"      ; the chat format: chatml (Qwen, MiniCPM) or gemma4 (Gemma 4, Winnow)
                        :calibration 1.0        ; temperature(s) over the option scores (Calibration)
                        :layout "question"}}    ; or "catalog": questions before the state (fast batches, much less accurate)
 :max-thinkers 1}                                ; resident at once (each is GBs)
```

The `minicpm5` entry spells out the defaults from `lev.think/defaults`,
so `{:model path}` alone is a complete entry.

`:prompt` picks how a question is put to the model. `lev` is the state,
the question, and the options by id, with `ANSWER: <id>` scored.
`semif` is SemIf's JSON payload with lettered options. The other two are
the prompts of models fine-tuned on them, and they read the letter's next
token only:

- `jevk5` is
  [alibiserikbay/JevK5](https://huggingface.co/alibiserikbay/JevK5-GGUF)'s
  prompt (a Qwen3.5-4B fine-tune): SemIf's payload with the state as JSON
  and `id: description` options, at most 16. The author's temperature is
  1.22.
- `winnow` is [EldanRing/Winnow-E4B](https://huggingface.co/EldanRing/Winnow-E4B)'s
  prompt (a Gemma 4 fine-tune): compact JSON with `<` escaped, lettered
  options, `Answer:\n`. It needs `:template "gemma4"`, and `:thinks false`
  for E4B, whose template has no empty-thought marker. The author's
  fitted temperature is 1.2574.

```clojure
{:thinkers {"jevk5" {:model "/Users/me/models/jevk5-4b-v0.3-Q8_0.gguf" :prompt "jevk5" :thinking false :calibration 1.22}
            "winnow-e4b" {:model "/Users/me/models/Winnow-E4B-Q8_0.gguf" :prompt "winnow" :template "gemma4"
                          :thinks false :n-ctx 8192 :n-seq-max 8 :calibration 1.2574}}}
```

On typed-decisions (`bench/README.md`), Winnow-E4B scores 0.724 at ECE
0.025, the best general model measured there, and JevK5 scores 0.626.
Both match ollaya's numbers. Qwen3.5-4B scores 0.588 with its
temperatures fitted. authored144 has not been run on either.

Each entry defines a model name a request can ask for. `--thinker PATH`
or `LEV_THINKER` adds one named `thinker`. Thinkers are loaded on first
use, are never chosen by content routing, and are listed by
`GET /v1/models`. Without the llm native from `jolt llama`, or without the
GGUF on disk, a thinker is listed as unavailable and a request for it is
a 503.

### Jev mode

With thinking off, a thinker answers every question of a call in one
pass. `native/llama.cpp` is [thecodacus/llama.cpp's](https://github.com/thecodacus/llama.cpp/tree/parallel-decision) `parallel-decision`
branch, and its decision engine does the work. The chat up to the state
is decoded once and kept for later calls. The state is decoded once per
call. Each question is then a branch forked from the state in the KV
cache, and every branch is decoded in the same batch. Each branch holds
the rest of that question's prompt, and its option ids are scored where
their tokens diverge. The questions can't see each other, and the
prompts are the same ones the per-question path uses. On MiniCPM5-2B,
four questions over a short email take 207 ms instead of 539 ms, and
1.2 s instead of 4.3 s over a 1.6k-token one. Accuracy is unchanged:
75.0% on authored144 against 73.6% for the per-question path, and the
same on the AG News / BoolQ / SST-5 trio (`bench/README.md`). Set
`:jev false` for the per-question path.

The engine runs on the handle's `:n-ctx` / `:n-seq-max` context (one
sequence keeps the chat's opening, one the state, the rest are
branches). A question whose options diverge at several token positions
takes a branch per position, and branches beyond the pool are decoded in
more passes. Hybrid models with recurrent layers work, but llama.cpp
splits their batches per sequence length, so they gain less.
Sliding-window models (Gemma) allocate their window per sequence, so
keep `:n-seq-max` low for them.

Text from the caller (the state, instructions and option descriptions)
is escaped before it reaches either path: a control token's text such as
`<|im_end|>` (or Gemma's `<turn|>`) gets a zero-width space, so a state
can't close its turn and write the next one (`lev.llm/escape`). An empty
state goes the per-question path, since the engine's contexts can't be
empty.

## Context: what the model sees

The encoders have no sessions, turns or memory. Every call is one
stateless forward pass, and the context is the `state` you pass in.
The server only caches loaded weights. For each question,
`build-sequence` lays out

```
[CLS] <type> question: <instructions> [SEP] [MASK] option 0 [MASK] option 1 … [SEP] <state> [SEP]
```

and the encoder reads it bidirectionally in one pass. The head scores the
`[MASK]` marker of each option. The questions in a call share the state
but run as independent rows with nothing passing between them, or
surviving the call.

**Budget.** `max_len` is 512 tokens on `english`, and 1024 on
`typed-decisions` and `multilingual`, as trained. Instructions and
options get up to `head_max_len`, 192 or 256, where long option lists
shrink evenly first and the instructions after. The state gets the rest.
Measured on the bundled workflows, `english` sees 427 to 478 state tokens
per question, roughly 1,700 to 1,900 characters of English prose and less
for JSON. The other two checkpoints see 940 to 990.

Whatever does not fit is dropped from the end. The first tokens of the
serialized state are kept, and the answer says so under `"truncated"`
with a `{qid: tokens dropped}` map that appears only when something was
cut. This is why the `email` workflow strips quoted history, signatures
and disclaimers, and caps the body at 3,000 characters before the model
sees anything. `usage.input_tokens` in every answer is the total over all
questions.

Both limits are yours to change, with `:max-len` and `:head-max-len` in
`config.edn` for every checkpoint or per name under `:checkpoints`, or
`--max-len` and `--head-max-len`, or `LEV_MAX_LEN` and
`LEV_HEAD_MAX_LEN`. See Configuration. The limits apply when a checkpoint
loads. `GET /v1/models` reports the effective values, and the server log
prints `max_len 768 (trained 512)`.

RoPE has no position table, so a longer sequence runs fine mechanically
and simply reads more of the state. The checkpoints were trained at 512
and 1024 though, and answer quality past that is unmeasured. Raise it
deliberately, and check on your own data. Lowering `head_max_len` buys
state room at the cost of shrinking long option texts sooner.

**Shaping the state.** A string is tokenized as is. A map or vector is
serialized with key order kept and
`ensure_ascii` off, and the keys are tokens too. Name them, and refer to
them in the instructions with backticks, the way the presets do:

```clojure
(ag/system-one agent
  {"ticket" "Charged twice, want my money back" "plan" "pro" "account_age_days" 412}
  {"churn_risk" {"type" "noul" "instructions" "Does `ticket` suggest the customer may cancel?"}})
```

Put the facts the decision needs in the state, and only those. A short,
structured state beats a long raw one, both for the budget and for the
answers.

**Conversations.** The state can be the trajectory so far, a vector of
turns:

```clojure
(def turns [{"role" "user" "text" "My payouts have failed for three days."}
            {"role" "agent" "text" "I see two failed transfers. Can you confirm the account ending 4411?"}
            {"role" "user" "text" "Yes. If this isn't fixed today I'm moving to Stripe."}])
(ag/system-one agent turns
  {"churn_risk" {"type" "noul" "instructions" "Will this customer leave?"}
   "needs_human" {"type" "noul" "instructions" "Should a person take over this conversation?"}})
```

The checkpoints were trained on conversation prefixes with TD(λ = 1)
targets, so re-asking the same questions on the growing prefix after each
turn is the intended use. `rl_agent_config.json` shows the depth the
training used in its `max_prefixes 6`. Your application holds the turns.
When they outgrow the budget, pass the last few, or a summary you produce
elsewhere.

**Chaining.** Multi-step decisions are successive calls with state you
assemble. Run `guard` before anything else, `llm-router` to pick a model,
then the workflow for the request. `action.act_probability` in every
answer is the act head's estimate that the system should act rather than
escalate, which is the natural input to gating between calls. A
server-side session store that appends a turn and re-runs a workflow on
the trajectory would be an application-layer feature, one the upstream
package does not have either. It would go in the workflow contract.

## Workflows

A workflow packages a use case: how to turn raw input into the model's
state, and which typed questions to ask. Workflows are ordinary Clojure
files, not part of `src/`. The bundled ones live in
[`workflows/`](workflows), and yours go in `~/.config/lev/workflows/` or
any directory listed in `config.edn :workflow-dirs`. Directories load in
that order and a later one wins on a name clash, so a
`~/.config/lev/workflows/email.clj` replaces the bundled `email`.

A file `<dir>/<name>.clj` defines the namespace `workflows.<name>`, with
underscores in the file name becoming dashes, holding:

```clojure
(ns workflows.refund-risk
  (:require [clojure.string :as str]))

(defn questions
  "Refund risk on a support ticket."                 ; the docstring is the description
  ([] (questions {}))
  ([opts]                                            ; optional 1-arity: the caller's options
   {"wants_refund" {"type" "noul" "instructions" "Does the customer ask for money back?"}
    "tone" {"type" "score" "instructions" "How angry is the ticket?"
            "criteria" ["calm" "annoyed" "furious"]}
    "team" {"type" "choice" "instructions" "Who should own this?"
            "criteria" (get opts "teams" {"billing" "money" "support" "everything else"})}}))

(defn state                                          ; optional; without it the input is the state
  [input]
  {"ticket" (str/trim (get input "text" ""))})
```

`questions` is required, `state` is optional. Question maps use string
keys, the shape the HTTP API receives. An `array-map` keeps option order,
which is model input. Run it:

```
jolt -M:run --list                                          # what is loaded, from where
jolt -M:run refund-risk '{"text": "Charged twice, want my money back"}'
jolt -M:run refund-risk @ticket.json --options '{"teams": {"billing": "money", "fraud": "chargebacks"}}'
jolt -M:run refund-risk @ticket.json --constraints '[["implies", ["wants_refund", true], ["team", "billing"]]]'
jolt -M:run refund-risk @ticket.json --model qwen3.5-4b              # on a thinker (--thinking true to think first)
LEV_WORKFLOWS=./my-workflows jolt -M:run refund-risk @ticket.json   # only that directory
```

One-off decisions without a workflow work too, with von's `decide`,
`judge` and `rate`, or a whole request from a file:

```
jolt -M:run decide "Charged twice, want my money back" --choices refund,help,other --instructions "What does the customer want?"
jolt -M:run judge "Refund me before Friday or we cancel." --instructions "Does the customer threaten to leave?"
jolt -M:run rate "Refund me before Friday or we cancel." --levels calm,annoyed,furious
jolt -M:run ask @request.json     # {"state": ..., "questions": {...}, "constraints"?: [...], "thinking"?: ...}
```

### Constraints

The model answers each question on its own. A workflow or a request can
tie them together with constraints, decided jointly after the forward
pass. The implementation is `lev.constraints`, a port of
constrained classification:

```clojure
(defn constraints                                    ; optional; same arities as questions
  []
  [[:implies ["wants_refund" true] ["team" "billing"]]
   [:at-most 1 ["tone" 2] ["team" "support"]]
   [:min-level "tone" 1]])
```

A ref `[question label]` names a choice option, a score level by index or
legend text, or a noul boolean. Operators are `not`, `all-of`, `any-of`,
`implies`, `iff`, `excludes`, `exactly-one-of`, `at-least k`, `at-most k`
and `exactly k`, over refs or nested constraints. Score questions add
`at-level`, `min-level`, `max-level` and `between-level`. Write them as
strings, keywords, dashes or underscores. JSON requests use the same
shape.

The decision maximizes the joint probability, the sum of the calibrated
log probabilities, subject to the constraints. Questions that nothing
couples stay independent. Coupled ones go through exact search with
branch and bound, which falls back to beam search past a node budget.
Whenever constraints are given, even an empty list, every answer carries
`decided` next to its own field: the option for a choice, the level index
for a score, the boolean for a noul. The result also carries a
`constraints` report with `feasible`, `decoder`, `exact` and
`violations`. The model's own `choice`, `score`, `noul` and
`probabilities` never change.

A contradictory set falls to the assignment with the fewest violated
constraints, listed in `violations`. Set `on_infeasible` to `raise` to
fail instead. The bundled `email` workflow ties `needs_reply` to
`is_spam` and `is_phishing`.

Bundled are five with questions pinned by `golden/presets.edn`:

| workflow | input | asks |
|---|---|---|
| `demo` | nothing (the quickstart email, pinned by `golden/readme.edn`) | department, urgency, churn risk, phishing |
| `email` | `{"subject", "body", "from"}` with quoted history, signatures and disclaimers cleaned | category (option `{"categories" {key description}}` swaps the teams), spam, phishing, urgency, needs reply |
| `triage` | `{"message"}` or a string | intent, urgency, frustration, refund requested, churn risk |
| `guard` | `{"prompt"}` or a string | jailbreak, prompt injection, sensitive data, harm severity, topic |
| `moderation` | `{"post"}` or a string | toxic, harassment, threat, spam, severity |
| `llm-router` | `{"request"}` or a string | difficulty, domain, needs tools, is sensitive (routing *your* LLM traffic, `lev.router` picks Lev models) |
| `security` | `{"event"}` or a string | event type, active threat, severity (von's security preset), with constraints: benign is no threat, a threat is at least elevated |

## HTTP API

`lev.server` mirrors the [TypeSafe Jev API](https://docs.typesafe.ai/api)
and adds `Router` along with presets.

```
POST /v1/systemone        Authorization: Bearer <key>   (only if a key is configured)
{"state": <string|object|array>, "questions": {"<id>": {...}}, "constraints"?: [...], "on_infeasible"?: "min_violations"|"raise",
 "model"?: ..., "lang"?: ..., "task"?: ..., "thinking"?: bool, "thought"?: bool, "debias"?: bool, "escalate"?: {...}}
-> {"model": "english", "answers": {"<id>": {...}}, "usage": {"input_tokens": n, "output_tokens": 0},
    "truncated"?: {"<id>": tokens}, "debias"?: {"<id>": rotations},
    "constraints"?: {"feasible": bool, "decoder": ..., "exact": bool, "violations": [...]},
    "routing": {"model": "english", "repo": "convaiinnovations/laya", "reason": "English Latin text",
                "detection": {...}, "workflow": null}}

POST /v1/systemone/batch  the same body with "states": [1-256 states] instead of "state", no "escalate"
                          -> {"results": [what /v1/systemone answers for each state, in order]}
POST /v1/route            same body, questions optional -> the routing decision alone (nothing loaded or run)
POST /v1/workflows/<name> {"input": <anything the workflow's state fn takes>, "options"?: {...}, "constraints"?: [...],
                           "on_infeasible"?: ..., "model"?/"lang"?/"task"?}
                          -> the systemone answer + "workflow" + the built "state"; the request's constraints
                             are added to the workflow's own
GET  /v1/models           -> {"default": ..., "max_loaded": n, "thinkers": {"minicpm5": {"model", "available", "loaded", "thinking"}},
                              "models": {"english": {"repo", "data", "available", "loaded", "limits"}, ...}}
GET  /v1/workflows        -> {"workflows": {"email": {"description", "file", "questions": [ids], "constraints": [...], "options": bool}, ...}}
GET  /health              -> {"status": "ok", "model": "lev", "loaded": [...], "thinkers": [...], "workflows": [...]}
```

Questions and answers score and noul, plus the `action.act_probability` extension on
the encoders' answers. A thinker answers in the same shapes without
`action`, adds a `"thinking": {"enabled", "tokens", "max_tokens"}`
report, and on request, with `"thought": true`, each answer's reasoning
under `"thought"`. `"thinking": false` asks it to answer at once (Jev
mode). On a GPU Qwen3.5-4B does that in about 170 ms a question at 95% on
authored144. MiniCPM5-2B takes about 100 ms but scores 75%, and needs
its thinking to reach 95%.

`/v1/systemone/batch` answers several states against the same questions.
Each state gets its own content routing on the encoders. A thinker with
thinking off hands every state to one engine call, which decodes the
states together and scores every question on every state in the same
passes. Grouping only saves time when each question's own tokens are few,
which is the `:layout "catalog"` thinker option. On MiniCPM5 that layout
costs 26 points on authored144, so the default layout gains little from a
batch (`bench/README.md`). With `"on_infeasible": "raise"`, one
infeasible state fails the whole batch.

```
POST /v1/systemone {"state": ..., "questions": {...}, "model": "minicpm5", "thinking": true, "thought": false}
POST /v1/systemone {"state": ..., "questions": {...}, "escalate": {"model": "qwen3.5-4b", "threshold": 0.8}}
                   -> the encoder's answers, the ones below the threshold replaced by the thinker's, plus
                      "escalation": {"threshold", "model", "escalated": [ids], "usage": the thinker's}
POST /v1/patterns/confidence-gate   systemone body + "threshold"          -> {"automatic": {...}, "escalate": {...}, "response": {...}}
POST /v1/patterns/composite-score   systemone body + "weights" {id: w}    -> {"score": 0..1, "breakdown": {...}, "response": {...}}
POST /v1/patterns/two-stage-choice  {"state", "taxonomy": {category: {option: description}}} -> {"category", "choice", "combined_confidence", ...}
```

On authored144, escalating the `english` encoder's answers to Qwen3.5-4B
lands at 89.6% with a threshold of 0.3 (99 of 144 up, 234 ms a case on
average), 92.4% at 0.5 (126 up, 270 ms) and 95.1% from 0.7. This is a set
where the encoder is rarely sure, so on routine traffic far fewer cases
go up. `escalate` also works on a workflow request, and constraints then decide
over the merged answers. `threshold` is one number for every question
type, or per type as `{"choice": 0.8, "score": 0.8, "noul": 0.9}`. A
noul's confidence is `max(p, 1 − p)`, never below 0.5. On yes/no reading
comprehension the encoder's confidence is uninformative below roughly
0.9, see `bench/README.md`, so nouls want a higher bar.

`"debias": true` asks every choice with three or more options once per
rotation of its options, in the same batch, and averages the
probabilities. Around 14% of the encoder's choices move under rotation.
Averaging is +2.8 points and a lower ECE on `english`, unchanged on
`typed-decisions`, at roughly 1.7× the time. When a state did not fit a
question's sequence, the body carries `"truncated"` with the tokens
dropped per question.

`model` routes by content when absent, or `lev`, or a TypeSafe SDK's
default `jev-latest` / `jev-preview`. Otherwise it is a checkpoint name
or alias, like `english`, `multilingual`, `typed-decisions`, `en` or
`ml`, or a thinker's name to pick one. `lang`, like `"de"` or `"en-GB"`,
and `task`, like `"typed_decisions"`, are the Router's other hints.
Precedence matches upstream: model, then task, then detected workflow
when opted in, then lang, then detected script and language, then the
default. Every answer says what was chosen and why under `routing`.

Errors are `401` for a missing or wrong key, `404` for unknown routes and
workflows, `405` for the wrong method, `503` when the chosen model is not
available, so an encoder with no prepared data or a thinker with no GGUF
or no llm native, and `413` past `:max-request-bytes`, 4 MiB. Anything
wrong with the body is a `422` carrying
`{"detail": [{"loc": ..., "msg": ..., "type": ...}]}`: malformed JSON,
missing state or questions, unknown type or model, criteria that don't
fit the type or the head, or a constraint naming no question or label at
`["body", "constraints", i]`. With `on_infeasible: raise`, a set nothing
satisfies is also a 422, of type `infeasible`, carrying the
`violations`. Inference and model loading are serialized on one lock, and
the adapter's workers overlap only on I/O.

```
jolt -M:serve --port 8080 --host 0.0.0.0 --api-key s3cret   # or PORT / LEV_HOST / LEV_API_KEY / LEV_DATA, or config.edn
curl -s -H 'Authorization: Bearer s3cret' -H 'Content-Type: application/json' \
  -d '{"state": "Help! My payouts have been failing for 3 days.",
       "questions": {"is_urgent": {"type": "noul", "instructions": "Does this convey urgency?"}}}' \
  http://127.0.0.1:8080/v1/systemone
curl -s -H 'Authorization: Bearer s3cret' -d '{"input": {"subject": "Refund", "body": "Charged twice.\n\nThanks,\nBob"}}' \
  http://127.0.0.1:8080/v1/workflows/email
```

The server holds one data root, laid out the way `jolt prepare` writes
it, so `DIR/` english, `DIR/multilingual`, `DIR/typed-decisions`, plus
the configured thinkers. The default model loads at startup and the
others on first use. The server keeps `--max-loaded` encoders resident,
default 1, with all three together around 4.6 GB of f32, and
`--max-thinkers` thinkers, evicting the least recently used when full. A
request for a model that is not available gets a `503` saying so.

### As a library

Add this repo as a `:git/url` dep, run `jolt kernels` and `jolt prepare`
for the native library and `data/`, then:

```clojure
(require '[lev.agent :as ag] '[lev.server :as server] '[lev.config :as cfg])
(def agent (ag/load-agent (cfg/setting (cfg/context {}) "--data" "LEV_DATA" :data "data")))
(ag/system-one agent state questions)                  
(def h (server/handler agent {:api-key nil}))          ; a ring handler to mount anywhere
(def s (server/start agent {:port 8080}))              ; or run it on ring-chez-adapter
(server/stop s)
```

Or route between the checkpoints, loading each
on first use and keeping `:max-loaded` resident:

```clojure
(require '[lev.router :as router])
(def rt (router/make-router {:data "data" :max-loaded 2}))          ; data/, data/multilingual, data/typed-decisions
(router/route rt {"body" "Der Kunde wurde zweimal belastet"} questions)   ; the decision, nothing loaded
;; => {"model" "multilingual", "repo" "convaiinnovations/laya/multilingual",
;;     "reason" "Latin script but language looks like 'de', not English", ...}
(router/predict rt state questions)                                  ; system-one + "routing"
(router/predict rt state questions :model "typed-decisions")         ; or :lang "de", :task "typed_decisions"
(router/loaded rt)                                                   ; ["multilingual" "typed-decisions"]
(def h (server/handler rt {:workflows (lev.workflows/load-workflows ["workflows"])}))
```

The one-question conveniences and the patterns, as a library (`lev.api`,
`lev.patterns`. `agent` is an agent or a router):

```clojure
(require '[lev.api :as api] '[lev.patterns :as pat] '[lev.router :as router])
(def rt (router/make-router {:data "data" :thinkers {"qwen3.5-4b" {:model "/Users/me/models/Qwen_Qwen3.5-4B-Q8_0.gguf" :thinking false}}}))

(api/decide rt "Database replication lag exceeded 45 seconds." {"infrastructure" "servers, network" "billing" "invoices"})
;; => {"type" "choice" "choice" "infrastructure" "probabilities" {...} "confidence" 0.83 "action" {...}}
(api/judge rt "Connection pool exhausted; handshakes timing out." "Is this blocking customers?")   ; => 0.9412
(api/rate rt "Memory at 98%, OOM killer active." ["nominal" "degraded" "critical"])                ; => the score answer
(api/decide rt state choices instructions {:model "qwen3.5-4b"})                                  ; the thinker

(pat/confidence-gate rt state questions {:threshold 0.85})           ; {"automatic" .. "escalate" .. "response" ..}
(pat/escalate rt state questions {:threshold 0.8 :model "qwen3.5-4b"}) ; the gate, with the escalated questions re-asked
(pat/route rt event (api/choice "Dispute action?" {"refund" "" "escalate" ""}) {"refund" process-refund "escalate" notify-fraud} {})
(pat/composite-score rt telemetry questions {:weights {"severity" 2.0 "is_threat" 3.0}})   ; {"score" 0.91 ...}
(pat/two-stage-choice rt "Postgres replica lag" {"cloud" {"aws" "..." "gcp" "..."} "database" {"postgres" "..." "redis" "..."}} {})
```

### As a binary

`jolt binary` runs `jolt build -m lev.server -o lev-server` with the C
kernels and llama.cpp from `jolt llama`, the pinned commit, static, Metal on
mac, linked in. It then runs `./lev-server --self-test` against
`golden/`. With a thinker configured, the self-test also asks it one
question, which proves the link.

Why the self-test? The suite runs interpreted, and a compiler release
can build the tree wrong where the interpreter runs it right.

The binary still needs the prepared data root next to it, or `--data
DIR`, or `config.edn`. It loads the workflow files from source at
startup, so they need no rebuild, from `./workflows` relative to where it
runs, `--workflows`, `LEV_WORKFLOWS` or `config.edn`, plus
`~/.config/lev/workflows`. From the OS it needs ICU and BLAS, and
libssl and libcrypto for the adapter.

```
./lev-server --data data --workflows workflows --port 8080 --api-key s3cret
./lev-server --self-test --data data --golden golden
./lev-server --mcp --data data          # MCP on stdio instead of HTTP (below)
```

### As MCP tools

`./lev-server --mcp`, or `jolt -M:mcp`, serves the same models and
workflows to an agent over the Model Context Protocol, on stdio. It takes
every flag and `config.edn` setting the HTTP server does:

```
claude mcp add lev -- /path/to/lev-server --mcp --data /path/to/data
```

The tools are `decide`, `run_workflow`, `list_models` and
`list_workflows`. `decide` answers exactly what `POST /v1/systemone`
answers, because it calls the same handler in process, and a bad request
comes back as a tool error carrying the 422's body. The resources
`lev://models`, `lev://workflows` and `lev://workflows/<name>` show what is
configured and each workflow's questions. `skills/lev-decisions/SKILL.md`
is an agent skill for it: when a typed decision beats reasoning in text,
how to write the questions, and when to act on an answer or escalate it.

Tagged releases, the `v*` tags, carry this binary prebuilt for macOS
arm64, with `golden/` and `workflows/` alongside, built and self-tested
by `.github/workflows/release.yml`.

## Speed or accuracy

`bench/` runs any configured model on von's authored144 set, 144
adversarial three-way decisions:

```
jolt -M bench/authored144.clj                                   # english: 61%, ~120 ms a case
jolt -M bench/authored144.clj --model typed-decisions            # 67%
jolt -M bench/authored144.clj --model qwen3.5-4b                 # 95%, ~175 ms a case (Metal), answering at once
jolt -M bench/authored144.clj --model minicpm5 --thinking false  # 75%, ~100 ms a case (Metal)
jolt -M bench/authored144.clj --model minicpm5                   # 95%, seconds a case
jolt -M bench/authored144.clj --debias                            # english with option-rotation averaging: 64%
jolt -M bench/triad.clj english                                   # AG News / BoolQ / SST-5, with ECE and what a gate keeps
jolt -M bench/workflow.clj                                        # a 4-question call on a short and a long state, p50 / p95
jolt -M bench/paired.clj --candidates cpu,mlx                     # two agents interleaved on the same inputs: paired speedup with a 95% interval
```

The last layer of the decision head computes only the rows the scorer
reads (each question's CLS and option markers), on either backend: exact,
1.5–2.5% of a call on the C kernels, 1–5% on MLX (bench/README.md).
A call tokenizes its state once for all its questions, and keeps every
question's own prefix (`[CLS] question [SEP] [MASK] options [SEP]`) in a
128-entry cache per loaded model, so a workflow's fixed questions on a
fresh state, or a game's questions every tick, cost no tokenization
beyond the state's. On a 1.6k-token state that is 10% of a four-question
call (bench/README.md); the answers do not change.

### The MLX backend (mac)

On Apple silicon the encoders can run on the GPU through MLX: `jolt mlx`
builds the native, `--backend mlx` (or `LEV_BACKEND=mlx`, or `:backend
"mlx"` in config.edn, per checkpoint under `:checkpoints`) runs every
encoder forward on it, and everything else — sequence building,
calibration, constraints, `debias`, the thinkers, the API — is the same
code. The router keeps one engine per checkpoint, so `:checkpoints
{"multilingual" {:backend "cpu"}}` mixes them.

- `--dtype f32` (the default) holds the weights on the device at full
  precision: the answers are the C kernels' to the fourth decimal, the
  golden tests hold it there, and authored144 and the trio reproduce
  their CPU accuracy and ECE exactly. Measured interleaved against the C
  kernels on an M-series (bench/paired.clj): **4.0x** on a short
  four-question call (270 → 68 ms), 3.3x on a long one (1,130 → 340 ms),
  2.6x on one long question; authored144 101 → 24 ms a case.
- `--dtype f16` halves the memory (0.8 GB a checkpoint) and is another
  20–25% faster (54 ms, 276 ms, 22 ms a case). The chosen answers agree
  (authored144 89/144 against f32's 88, the trio identical), the
  probabilities drift by up to 1e-2, so it is off the golden path: use it
  where the label matters and the fourth decimal does not.
- Without the native (linux, or `jolt mlx` not run) a request for the
  mlx backend is a clear `:model-unavailable` error, not a fallback.
  MLX loads its Metal kernels from `mlx.metallib` next to the binary
  holding it: `native/mlx.metallib` for `jolt run` / `test` (the build
  puts it there), and beside `lev-server` for a `jolt build`.

A hosted generative decision API wins through its reasoning budget. A
generative model answering without thinking does no better
than an NLI encoder, `thinking` sets the budget, and `escalate`
gates it so only the unsure answers pay for it. Numbers, alternatives
and the method are in [bench/README.md](bench/README.md).

## Native dependencies

Both platforms are supported. `deps.edn` carries darwin and linux
entries, and the build task branches on OS.

- **kernels**: `native/liblev_kernels.dylib` on mac and `.so` on linux,
  built by `jolt kernels`. These are the elementwise and reduction
  kernels and attention, running on a pthread pool of their own sized by
  `LEV_THREADS`. Attention calls `cblas_sgemm` through a pointer the
  Clojure side hands it, so the library links against no BLAS.
- **llama.cpp**: `native/liblev_llm.dylib` / `.so` plus `liblev_llm.a`,
  built by `jolt llama`. It fetches thecodacus/llama.cpp's
  `parallel-decision` branch at its pinned commit (upstream b10435 plus
  the decision engine) into `native/llama.cpp` and builds `llama` and
  `llama-common` static with cmake, with no https or subprocess support,
  Metal with the shader library embedded on mac and CPU elsewhere. They
  sit behind `native/lev_llm.c` and `native/lev_decision.cpp` (the
  engine's C face), which jolt.ffi binds as `lev.llm`. A checkout at
  another commit is replaced. `LLAMA_CPP_SRC=path` fetches from a local
  clone that has the commit. It is
  optional, since without it the encoders run and thinkers are
  unavailable. For `jolt build`, the archive is force-loaded and libc++
  plus the Metal, Foundation, MetalKit and Accelerate frameworks are
  linked through `lib<Name>.tbd` symlinks the build script makes under
  `native/frameworks/`. That is how a `deps.edn` `:static {:lib}` can
  name a framework, and the binary ends up depending only on system
  frameworks.
- **MLX** (mac only): `native/liblev_mlx.dylib` plus `liblev_mlx.a` and
  `native/mlx.metallib`, built by `jolt mlx`. It clones
  [mlx-c](https://github.com/ml-explore/mlx-c), Apple's C binding for
  MLX, at its pinned tag into `native/mlx-c` (its cmake fetches MLX
  itself at the matching tag) and builds both static behind
  `native/lev_mlx.c`, a flat C face jolt.ffi binds as `lev.mlx`: the
  whole encoder forward is built there as one lazy MLX graph and
  evaluated once per batch, the weights read from `data/`'s own f32
  files. Optional: without it the encoders run on the C kernels. The
  static link adds the QuartzCore framework to llama's set, and the
  binary needs `mlx.metallib` (the precompiled Metal kernels, 100 MB)
  beside it at runtime.
- **ICU**: `libicucore.dylib` on mac, whose system dylib leaves the
  symbols unguarded, and `libicuuc.so.<ver>` on linux. The tokenizers
  call `unorm2` for NFC and the `u_charType` / `u_isUWhiteSpace`
  classifiers. Linux ICU builds append the major version to every symbol, like `u_charType_76`, and the bindings resolve the first spelling that exists, for versions
  60 to 90.
- **JSON**: `org.clojure/data.json` from Maven, plus `jolt-lang/time`,
  which provides the `java.time` classes data.json needs to load. Only
  `prepare` uses them. Requests are read by `lev.json`, which keeps key
  order.
- **HTTP**: `jolt-lang/ring-chez-adapter` serves the API, and
  `org.clojars.askonomm/ruuter` from Clojars dispatches its routes.
- **BLAS**: `cblas_sgemm` from the Accelerate framework on mac, OpenBLAS
  on linux.

On linux, install the ICU and OpenBLAS runtime packages. Adjust the
version suffixes in `deps.edn` if your distro's `libicuuc.so` version is
not listed.

## Status

Encoder, head, both tokenizers, sequence, agent, the workflows and the
checkpoint conversion all match their golden traces on all three
checkpoints, in `golden/`, `golden/typed-decisions/` and
`golden/multilingual/`.

Per-forward temporaries live in an ffi arena that closes with the call,
so a long-running process stays at the size of the loaded weights. That
is about 1.7 GB f32 per ModernBERT-large checkpoint and 1.3 GB for
mmBERT-base, of which the server keeps `--max-loaded`. The questions of
one call go through the encoder as one batch of up to 8 rows, padded to
the longest and masked, sharing every matmul. The intermediates are one
workspace per call, around 260 MB at 8 x 512.

Speed on `english`, a 10-core M-series laptop, Accelerate, single call.
One question takes around 95 ms at 55 tokens and around 350 ms at 512,
linear in between. It used to be quadratic, 4.8 s at 512, until
attention went through per-head sgemm with the sliding layers scoring
only their 129-key band. The softmax, LayerNorm and GELU kernels are
vectorized, and a call's questions share one batched forward. The
bundled `demo`, 4 questions at around 90 tokens each, takes around 290
ms. `email` on a 3,000-character body, 5 questions at the 512 cap, takes
around 1.5 s.

The matmuls are Accelerate's, multi-core. Attention over heads and query
blocks, the swiglu GELU and LayerNorm over rows run on the kernel
library's own thread pool, `LEV_THREADS` wide, defaulting to the online
processors. You can change it at runtime with
`lev.tensors/set-threads!`. The thread count changes only the schedule,
never a result. Every task runs the same arithmetic, and the suite
checks that the bytes are identical at 1, 2, 3 and 8 threads. On one
thread the same call takes around 520 ms at 512 tokens, and `email`
around 2.3 s.
