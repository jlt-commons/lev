(ns lev.llm
  "llama.cpp through native/lev_llm.c (the `lev_llm` native, built by
  `jolt llama`; optional, so a tree without it runs the encoders alone).

  A model is a handle on one GGUF and one context. `generate` completes a
  prompt; `decide` is what the thinker engine runs per question: decode a
  chat prompt, let the model think until its closing tag, force an answer
  prefix, then score each candidate answer by teacher-forcing its tokens
  on a copy of the KV state, answering one log probability per option.
  `jev` is the vendored fork's decision engine (native/lev_decision.cpp):
  many fields over one decoded context in one pass, the thinker's path
  when it does not think. `escape` makes caller text safe to put in a
  prompt that is tokenized with special tokens parsed.
  `chat-prompt` renders ChatML with the model's thinking switch, as
  MiniCPM5's template does (`<think>\\n` to think, `<think>\\n\\n</think>\\n\\n`
  not to)."
  (:require [clojure.string :as str]
            [jolt.ffi :as ffi]))

;; :blocking (the collector may run while a call is inside C, seconds for a
;; thought) rules out :string arguments: every string goes in as an
;; arena-owned pointer that outlives the call
(ffi/defcfn load* "lev_llm_load" [:pointer :int :int :int :int] :pointer :blocking)
(ffi/defcfn free* "lev_llm_free" [:pointer] :void)
(ffi/defcfn ok* "lev_llm_ok" [:pointer] :int)
(ffi/defcfn error* "lev_llm_error" [:pointer] :string)
(ffi/defcfn version* "lev_llm_version" [] :string)
(ffi/defcfn n-ctx* "lev_llm_n_ctx" [:pointer] :int)
(ffi/defcfn n-seq-max* "lev_llm_n_seq_max" [:pointer] :int)
(ffi/defcfn count-tokens* "lev_llm_count_tokens" [:pointer :string] :int)
(ffi/defcfn generate* "lev_llm_generate"
  [:pointer :pointer :int :pointer :float :float :float :uint32 :pointer :int] :int :blocking)
(ffi/defcfn escape* "lev_llm_escape" [:pointer :string :pointer :int] :int)
(ffi/defcfn jev* "lev_llm_jev"
  [:pointer :pointer :pointer :int :pointer :pointer :pointer :int :int :int :pointer :pointer :pointer] :int :blocking)
(ffi/defcfn decide* "lev_llm_decide"
  [:pointer :pointer :int :pointer :pointer :pointer :int :pointer :float :float :float :uint32 :pointer :pointer :int]
  :int :blocking)

(defn available?
  "Is the llm native loaded? (jolt llama builds it.)"
  []
  (some? (ffi/find-symbol "lev_llm_load")))

(defn version [] (version*))

(def defaults
  {:n-ctx 4096 :n-gpu-layers -1 :threads 0 :n-seq-max 32
   :temperature 1.0 :top-p 0.95 :min-p 0.0 :seed 42
   :think-max 2048 :think-end "</think>" :answer-prefix "\n\nANSWER: " :answer-end "<|im_end|>"})

(defn load
  "Load a GGUF: {:n-ctx (4096) :n-gpu-layers (-1 = all) :threads (0 =
  llama.cpp's) :n-seq-max (32, the most options one question can have)}.
  Throws {:type :model-unavailable} when the native is missing or the
  file cannot be loaded."
  [path opts]
  (when-not (available?)
    (throw (ex-info "the llm native is not built: run jolt llama (see README, Native dependencies)"
                    {:type :model-unavailable :model path})))
  (let [{:keys [n-ctx n-gpu-layers threads n-seq-max]} (merge defaults opts)
        h (ffi/with-arena [a]
            (load* (ffi/string->ptr a (str path)) (int n-ctx) (int n-gpu-layers) (int threads) (int n-seq-max)))]
    (when (or (zero? (ffi/address h)) (zero? (ok* h)))
      (let [msg (if (zero? (ffi/address h)) "out of memory" (error* h))]
        (when-not (zero? (ffi/address h)) (free* h))
        (throw (ex-info (str "cannot load " path ": " msg) {:type :model-unavailable :model path}))))
    {:p h :path (str path) :opts (merge defaults opts)}))

(defn ok? [m] (and m (pos? (ok* (:p m)))))
(defn n-ctx [m] (n-ctx* (:p m)))
(defn n-seq-max [m] (n-seq-max* (:p m)))
(defn free! [m] (free* (:p m)) nil)
(defn count-tokens [m text] (count-tokens* (:p m) text))

(defn escape
  "`text` with every control token's text in it broken by a zero-width
  space, so caller text (a state, a question) cannot forge the chat's
  tags: prompts are tokenized with special tokens parsed, and a state
  holding \"<|im_end|>\" would otherwise close its turn."
  [m text]
  (let [text (str text)
        cap (+ 16 (* 4 (count (.getBytes text "UTF-8"))))]
    (ffi/with-arena [a]
      (let [out (ffi/alloc a cap)]
        (when (neg? (escape* (:p m) text out (int cap)))
          (throw (ex-info (str "escape failed: " (error* (:p m))) {:type :llm-error})))
        (ffi/ptr->string out)))))

(def templates
  "The chat formats a thinker's prompt can take, by name: how a turn is
  written, how the assistant's turn opens, the thought tags (open, closed
  empty, the closing tag `decide` waits for), what separates a closed
  thought from the answer, and the turn's end (what closes a scored
  answer). chatml is Qwen's and MiniCPM's; gemma4 is Gemma 4's (and
  Winnow's), whose empty thought is `<|channel>thought\\n<channel|>`."
  {"chatml" {:open "<|im_start|>" :close "<|im_end|>\n" :roles {}
             :assistant "<|im_start|>assistant\n"
             :think-open "<think>\n" :think-closed "<think>\n\n</think>" :think-end "</think>"
             :after-thought "\n\n" :answer-end "<|im_end|>"}
   "gemma4" {:open "<|turn>" :close "<turn|>\n" :roles {"assistant" "model"}
             :assistant "<|turn>model\n"
             :think-open "<|channel>thought\n" :think-closed "<|channel>thought\n<channel|>" :think-end "<channel|>"
             :after-thought "" :answer-end "<turn|>"}})

(defn template
  "The chat format named `t` (default chatml); throws on an unknown name."
  [t]
  (let [k (if (keyword? t) (name t) (str (or t "chatml")))]
    (or (get templates k)
        (throw (ex-info (str "unknown chat template " (pr-str t) "; choose one of " (str/join ", " (sort (keys templates))))
                        {:type :invalid-config :template t})))))

(defn chat-prompt
  "`messages` ({:role :content}, string keys accepted) through the
  assistant turn, in the chat format opts :template names (chatml by
  default, see `templates`). opts :thinking (true: the open thought tag,
  for `decide` to close; false: the closed empty thought, so the model
  answers at once; nil: neither). Either way what follows the closing
  tag is the answer prefix (\"\\n\\nANSWER: \" by default for chatml), so
  a decided answer is always scored after `</think>\\n\\nANSWER: `, the
  shape the template produces."
  [_m messages {:keys [thinking] :as opts}]
  (let [{:keys [open close roles assistant think-open think-closed]} (template (:template opts))
        turn (fn [{:keys [role content] :as msg}]
               (let [role (or role (get msg "role"))]
                 (str open (get roles role role) "\n" (or content (get msg "content")) close)))]
    (str (apply str (map turn messages))
         assistant
         (case thinking
           true think-open
           false think-closed
           ""))))

(defn generate
  "Complete `prompt`: {:max-tokens (256) :stop (nil) :temperature (model
  default; 0 = greedy) :top-p :min-p :seed} -> {:text :tokens}."
  [m prompt {:keys [max-tokens stop temperature top-p min-p seed]
             :or {max-tokens 256}}]
  (let [{:keys [opts]} m
        cap (* 8 (+ max-tokens 16))]
    (ffi/with-arena [a]
      (let [out (ffi/alloc a cap)
            n (generate* (:p m) (ffi/string->ptr a prompt) (int max-tokens) (ffi/string->ptr a (or stop ""))
                         (float (or temperature (:temperature opts))) (float (or top-p (:top-p opts)))
                         (float (or min-p (:min-p opts))) (long (or seed (:seed opts))) out (int cap))]
        (when (neg? n)
          (throw (ex-info (str "generation failed: " (error* (:p m))) {:type :llm-error})))
        {:text (ffi/ptr->string out) :tokens n}))))

(defn decide
  "Score `options` as answers to `prompt` (a chat-prompt): {:think-max
  (tokens of thought; 0 = none, the prompt must then close the thought)
  :think-end :answer-prefix :answer-end :temperature :top-p :min-p :seed}
  -> {:logp [per option] :thought text :tokens thought-tokens}."
  [m prompt options {:keys [think-max think-end answer-prefix answer-end temperature top-p min-p seed]}]
  (let [opts (:opts m)
        n (count options)
        think-max (long (or think-max (:think-max opts)))
        think-cap (* 8 (+ think-max 64))]
    (ffi/with-arena [a]
      (let [optv (ffi/alloc a (* 8 n))
            strs (mapv (fn [o] (ffi/string->ptr a (str o))) options)
            _ (dotimes [i n] (ffi/write optv :pointer (nth strs i) (* 8 i)))
            logp (ffi/alloc a (* 8 n))
            thought (ffi/alloc a think-cap)
            made (decide* (:p m) (ffi/string->ptr a prompt) (int think-max)
                          (ffi/string->ptr a (or think-end (:think-end opts)))
                          (ffi/string->ptr a (or answer-prefix (:answer-prefix opts)))
                          optv (int n)
                          (ffi/string->ptr a (or answer-end (:answer-end opts)))
                          (float (or temperature (:temperature opts))) (float (or top-p (:top-p opts)))
                          (float (or min-p (:min-p opts))) (long (or seed (:seed opts)))
                          logp thought (int think-cap))]
        (when (neg? made)
          (throw (ex-info (str "decide failed: " (error* (:p m))) {:type :llm-error})))
        {:logp (mapv #(ffi/read logp :double (* 8 %)) (range n))
         :thought (ffi/ptr->string thought)
         :tokens made}))))

(defn jev
  "Jev mode (the decision engine of the vendored llama.cpp fork): every
  field answered for every context in one batched pass.
  {:shared text before every context, decoded once and kept while it
   stays the same (\"\" for none)
   :contexts [text ...] each continuing :shared
   :fields [{:suffix text after the context :values [allowed text ...]}]
   :cache? (true) reuse the kept :shared
   :split-boundary? (false) tokenize suffixes and values apart}
  -> {:probs [per context [per field [per value p]]] :context-tokens [n ...]
      :prefill-ms :scoring-ms :rounds :rows :cache-hit :shared-tokens}.
  Each field is its own sequence forked from its context, so fields
  cannot see each other; its values are scored at every node where their
  tokens diverge, normalised over the allowed tokens there, which makes
  the probabilities exact over the values."
  [m {:keys [shared contexts fields cache? split-boundary?] :or {shared "" cache? true}}]
  (let [nc (count contexts)
        nf (count fields)
        nv (mapv (comp count :values) fields)
        total (reduce + nv)
        strs (fn [a xs]
               (let [v (ffi/alloc a (* 8 (max 1 (count xs))))]
                 (dotimes [i (count xs)] (ffi/write v :pointer (ffi/string->ptr a (str (nth xs i))) (* 8 i)))
                 v))]
    (ffi/with-arena [a]
      (let [nvp (ffi/alloc a (* 4 (max 1 nf)))
            _ (dotimes [i nf] (ffi/write nvp :int (int (nth nv i)) (* 4 i)))
            probs (ffi/alloc a (* 8 (max 1 (* nc total))))
            stats (ffi/alloc a (* 8 6))
            ctoks (ffi/alloc a (* 4 (max 1 nc)))
            rc (jev* (:p m) (ffi/string->ptr a (str shared)) (strs a contexts) (int nc)
                     (strs a (mapv :suffix fields)) nvp (strs a (mapcat :values fields)) (int nf)
                     (int (if cache? 1 0)) (int (if split-boundary? 1 0)) probs stats ctoks)]
        (when (neg? rc)
          (throw (ex-info (str "jev failed: " (error* (:p m))) {:type :llm-error})))
        (let [flat (mapv #(ffi/read probs :double (* 8 %)) (range (* nc total)))
              stat #(ffi/read stats :double (* 8 %))
              per-context (fn [row]
                            (first (reduce (fn [[acc off] n] [(conj acc (subvec row off (+ off n))) (+ off n)])
                                           [[] 0] nv)))]
          {:probs (mapv #(per-context (subvec flat (* % total) (* (inc %) total))) (range nc))
           :context-tokens (mapv #(ffi/read ctoks :int (* 4 %)) (range nc))
           :prefill-ms (stat 0) :scoring-ms (stat 1)
           :rounds (long (stat 2)) :rows (long (stat 3))
           :cache-hit (= 1.0 (stat 4)) :shared-tokens (long (stat 5))})))))

(defn argmax [xs]
  (reduce (fn [bi i] (if (> (nth xs i) (nth xs bi)) i bi)) 0 (range (count xs))))

(defn softmax
  "Probabilities from log scores."
  [logp]
  (let [mx (reduce max logp)
        ex (mapv #(Math/exp (- (double %) mx)) logp)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))
