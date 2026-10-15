(ns lev.tinydecide
  "TinyDecide (TheREZOR/TinyDecide): an ELECTRA-small encoder (12 layers,
  width 256, 4-bit weights, 6 MB) with typed heads. The state and every
  question go into one sequence: the state as block 0, each question its
  own block that attends to the state and itself only, so one pass
  answers them all and no answer depends on what else was asked. A
  choice / score question's options are marker tokens scored against its
  answer token; a noul's answer token goes through one logistic head.
  English only, and it reads the first 127 tokens of the state.

  The release's meta.json + model.bin (its Q4 / int8 matrices are
  dequantized to f32 at load) run here on the C kernels and cblas, held
  to the release's JavaScript reference engine by golden/tinydecide
  (bench/tinydecide_golden.py, from its conformance/ suite). Its span
  questions and corrections (per-question prototypes) are not exposed:
  the request has no type for either.

  An agent is {:kind :tinydecide :name :cfg :m}; lev.think/thinker builds
  one when a thinker's config says :engine \"tinydecide\" (config.edn
  :thinkers {\"tinydecide\" {:engine \"tinydecide\" :model DIR}}, DIR the
  downloaded release, or its full/ for the 13.8M build), so the router
  serves it like any other thinker. The answers' probabilities are the
  release's own calibrated ones; a lev.calibrate file refits over them."
  (:require [clojure.data.json :as json]
            [clojure.java.io :as io]
            [jolt.ffi :as ffi]
            [lev.agent :as ag]
            [lev.sequence :as seq]
            [lev.tensors :as t]
            [lev.tokenizer :as tk]))

(ffi/defcfn read-file-range* "lev_read_file_range" [:pointer :int64 :int64 :pointer] :int64)
(ffi/defcfn dequant-q4* "lev_dequant_q4" [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn dequant-i8* "lev_dequant_i8" [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn block-mask* "lev_block_mask" [:pointer :int64 :pointer] :void)
(ffi/defcfn add-bias* "lev_add_bias" [:pointer :pointer :int64 :int64] :void)

(def defaults {:calibration {:temperature [1.0 1.0 1.0]}})

;; --- the release -------------------------------------------------------------

(defn- unavailable [dir msg]
  (throw (ex-info (str "tinydecide: " msg) {:type :model-unavailable :model dir})))

(defn release?
  "Does `dir` hold a release (meta.json and model.bin)?"
  [dir]
  (boolean (and dir (.exists (io/file dir "meta.json")) (.exists (io/file dir "model.bin")))))

(defn- at [p bytes] (ffi/segment (+ (ffi/address p) bytes)))

(defn- load-weights
  "{name tensor} of model.bin, f32, malloc'd (the agent's :close frees
  them). A vector is a [1 x n] tensor."
  [meta path]
  (let [n (long (get meta "bytes"))
        buf (ffi/alloc n)]
    (try
      (when (not= n (ffi/with-arena [a] (read-file-range* (ffi/string->ptr a path) 0 n buf)))
        (unavailable path (str "cannot read " n " bytes of " path)))
      (into {}
            (for [{:strs [name shape dtype offset scale_offset]} (get meta "tensors")]
              (let [[r c] (if (= 1 (count shape)) [1 (first shape)] shape)
                    tt (binding [t/*arena* nil] (t/make [r c]))]
                (case dtype
                  ;; a copy: 0 + 1 * the stored floats onto the zeroed tensor
                  "f32" (t/add-scaled! tt tt {:p (at buf offset) :shape [r c] :size (* r c)} 1.0)
                  "q4" (dequant-q4* (at buf offset) (at buf scale_offset) r c (t/ptr tt))
                  "int8" (dequant-i8* (at buf offset) (at buf scale_offset) r c (t/ptr tt)))
                [name tt])))
      (finally (ffi/free buf)))))

(defn load-release
  "The release at `dir`: {:meta :cfg :w :vocab :unk :prefix :max-chars}."
  [dir]
  (when-not (release? dir)
    (unavailable dir (str dir " has no meta.json and model.bin (hf download TheREZOR/TinyDecide)")))
  (when-not (t/available?)
    (unavailable dir "no kernels or BLAS loaded (jolt kernels; OpenBLAS on linux)"))
  (let [meta (json/read-str (slurp (str dir "/meta.json")))
        cfg (get meta "cfg")
        tok (get meta "tokenizer")]
    (when-not (and (= "electra" (get cfg "arch")) (= "all" (get cfg "fusion")) (zero? (or (get cfg "L_a") 0))
                   (= "wordpiece" (get tok "kind")))
      (unavailable dir (str "unsupported build " (pr-str (select-keys cfg ["arch" "fusion" "L_a"])))))
    (let [vocab (into {} (keep-indexed (fn [i p] (when p [p i]))) (get tok "vocab"))]
      {:meta meta :cfg cfg
       :w (load-weights meta (str dir "/model.bin"))
       :vocab vocab :unk (get vocab (get tok "unk")) :prefix (get tok "prefix") :max-chars (get tok "max_chars")})))

(defn free-release! [{:keys [w]}]
  (doseq [x (vals w)] (ffi/free (t/ptr x))))

;; --- the tokenizer: BERT's basic tokenizer, then greedy WordPiece -------------

(defn- cjk? [c]
  (or (<= 0x4E00 c 0x9FFF) (<= 0x3400 c 0x4DBF) (<= 0x20000 c 0x2A6DF) (<= 0x2A700 c 0x2B73F)
      (<= 0x2B740 c 0x2B81F) (<= 0x2B820 c 0x2CEAF) (<= 0xF900 c 0xFAFF) (<= 0x2F800 c 0x2FA1F)))

(def ^:private punct-types #{19 20 21 22 23 28 29})

(defn- punct? [c]
  (or (<= 33 c 47) (<= 58 c 64) (<= 91 c 96) (<= 123 c 126)
      (and (>= c 128) (contains? punct-types (tk/char-type c)))))

(def ^:private js-space?
  "JavaScript's \\s: WhiteSpace and LineTerminator."
  (into #{9 10 11 12 13 32 0xA0 0x1680 0x2028 0x2029 0x202F 0x205F 0x3000 0xFEFF} (range 0x2000 0x200B)))

(defn- normalised
  "The code points one input character stands for once lowercased and
  stripped of accents (nil: skipped; [32]: a word break)."
  [c]
  (cond
    (< c 128) (cond (or (= c 9) (= c 10) (= c 13) (= c 32)) [32]
                    (or (< c 32) (= c 127)) nil
                    (<= 65 c 90) [(+ c 32)]
                    :else [c])
    (or (= c 0xFFFD) (#{15 16} (tk/char-type c))) nil
    (= 12 (tk/char-type c)) [32]
    (cjk? c) [32 c 32]
    :else (vec (for [d (tk/nfd (str (char c)))
                     :when (not= 6 (tk/char-type (long d)))
                     l (tk/lower (str d))]
                 (long l)))))

(defn- word-ids
  "Greedy longest-match WordPiece over one word (code points); too long or
  unmatched -> one unknown token."
  [{:keys [vocab unk prefix max-chars]} w]
  (if (> (count w) max-chars)
    [unk]
    (let [s (apply str (map char w))
          n (count s)]
      (loop [start 0 out []]
        (if (>= start n)
          out
          (let [[end id] (loop [end n]
                           (when (> end start)
                             (let [piece (subs s start end)
                                   id (get vocab (if (pos? start) (str prefix piece) piece))]
                               (if id [end id] (recur (dec end))))))]
            (if id (recur end (conj out id)) [unk])))))))

(defn tokenize-until
  "The ids of `text`, reading no further once there are more than `limit`
  (so the result is a prefix of the whole encoding, longer than `limit`
  exactly when the whole one is)."
  [m text limit]
  (let [cps (map long (str text))
        step (fn [[ids cur] c]
               (cond
                 (or (= c 32) (js-space? c)) [(cond-> ids (seq cur) (into (word-ids m cur))) []]
                 (punct? c) [(-> ids (cond-> (seq cur) (into (word-ids m cur))) (into (word-ids m [c]))) []]
                 (> (count cur) (:max-chars m)) [ids (assoc cur (dec (count cur)) c)]
                 :else [ids (conj cur c)]))]
    (loop [cps cps acc [[] []]]
      (let [[ids cur] acc]
        (cond
          (> (count ids) limit) ids
          (empty? cps) (cond-> ids (seq cur) (into (word-ids m cur)))
          :else (recur (rest cps) (reduce step acc (normalised (first cps)))))))))

(defn tokenize [m text] (tokenize-until m text Long/MAX_VALUE))

;; --- the request -------------------------------------------------------------

(def ^:private qtypes #{"choice" "noul" "score"})

(defn- invalid [q i msg]
  (let [qid (get q :qid (inc i))]
    (throw (ex-info (str "question " (pr-str qid) ": " msg) {:type :invalid-question :qid qid :field "criteria"}))))

(defn encode-request
  "The release's layout of a state and questions ([{:type :text :options
  :qid}], :qid only for the errors): {:ids :pos :blk :qs [{:type :ans
  :opt}] :truncated :dropped}. The state gets the first ts_max - 1 tokens
  (:dropped says how many more it had); each question's block starts at
  position p_q."
  [{:keys [meta] :as m} state questions]
  (let [sp (get meta "specials")
        {:strs [ts_max p_q q_max k_max]} (get meta "format")
        st (tokenize-until m state (dec ts_max))
        n (min (count st) (dec ts_max))
        dropped (if (> (count st) n) (- (count (tokenize m state)) n) 0)
        ids (into [(get sp "<|state|>")] (take n st))
        base {:ids ids :pos (vec (range (count ids))) :blk (vec (repeat (count ids) 0)) :qs []}]
    (assoc (reduce
            (fn [acc [i {:keys [type text options] :as q}]]
              (let [k (count options)]
                (when-not (qtypes type) (invalid q i (str "unknown type " (pr-str type))))
                (when (and (#{"choice" "score"} type) (or (< k 2) (> k k_max)))
                  (invalid q i (str "needs 2 to " k_max " options, not " k)))
                (when (and (= "noul" type) (pos? k)) (invalid q i "a noul question takes no options"))
                (let [[b opt] (reduce (fn [[b opt] o]
                                        (let [b (into b (tokenize m o))]
                                          [(conj b (get sp (if (= "choice" type) "<|o|>" "<|lv|>"))) (conj opt (count b))]))
                                      [(cond-> (into [(get sp (str "<|" type "|>"))] (tokenize m text))
                                         (pos? k) (conj (get sp "<|sep|>")))
                                       []]
                                      options)
                      ans (count b)
                      b (conj b (get sp "<|ans|>"))
                      start (count (:ids acc))]
                  (when (> (count b) q_max)
                    (invalid q i (str "is too long (" (count b) " tokens, max " q_max ")")))
                  (-> acc
                      (update :ids into b)
                      (update :pos into (map #(+ p_q %) (range (count b))))
                      (update :blk into (repeat (count b) (inc i)))
                      (update :qs conj {:type type :ans (+ start ans) :opt (mapv #(+ start %) opt)})))))
            base (map-indexed vector questions))
           :truncated (pos? dropped) :dropped dropped)))

;; --- the encoder and heads ---------------------------------------------------------

(defn- linear
  ([x W] (t/mmul x W))
  ([x W b] (let [y (t/mmul x W)] (add-bias* (t/ptr y) (t/ptr b) (first (t/shape y)) (second (t/shape y))) y)))

(defn- gather [src rows]
  (let [d (second (t/shape src))]
    (t/gather src (t/from-ints rows) (count rows) d)))

(defn- hidden
  "The encoder's [T x d] final states for a laid-out request."
  [{:keys [cfg w]} {:keys [ids pos blk]}]
  (let [g #(get w %)
        T (count ids)
        {:strs [d heads layers ln_eps]} cfg
        hd (quot d heads)
        raw (t/add-scaled (gather (g "m.word.weight") ids) (gather (g "m.posemb.weight") pos) 1.0)
        _ (add-bias* (t/ptr raw) (t/ptr (g "m.type0")) T (second (t/shape raw)))
        x (linear (t/layernorm raw (g "m.eln.weight") (g "m.eln.bias") ln_eps) (g "m.proj.weight") (g "m.proj.bias"))
        mask (t/byte-matrix T T)
        _ (block-mask* (t/ptr (t/from-ints blk)) T (t/ptr mask))
        scale (/ 1.0 (Math/sqrt hd))]
    (reduce (fn [x l]
              (let [p #(g (str "m.blocks." l "." %))
                    heads-of #(t/split-heads (linear x (p (str % ".weight")) (p (str % ".bias"))) T heads hd)
                    ctx (t/attention (heads-of "q") (heads-of "k") (heads-of "v") mask heads T hd scale)
                    o (linear ctx (p "o.weight") (p "o.bias"))
                    x (t/layernorm (t/add-scaled! o o x 1.0) (p "ln1.weight") (p "ln1.bias") ln_eps)
                    f (linear (t/gelu! (linear x (p "fc.weight") (p "fc.bias"))) (p "fc2.weight") (p "fc2.bias"))]
                (t/layernorm (t/add-scaled! f f x 1.0) (p "ln2.weight") (p "ln2.bias") ln_eps)))
            x (range layers))))

(defn- dot [a b] (reduce + (map * a b)))

(defn logits
  "Per question of a laid-out request, its option logits divided by the
  release's temperature for the type, so their softmax is the release's
  probabilities: a choice's / score's in option order, a noul's [0 z]."
  [{:keys [w meta] :as m} enc]
  (ffi/with-arena [a]
    (binding [t/*arena* a]
      (let [g #(get w %)
            hq (t/layernorm (hidden m enc) (g "h.norm.weight") (g "h.norm.bias") 1e-5)
            temp (get meta "temp")
            scales (t/to-floats (g "h.scale"))
            dh (Math/sqrt (double (get-in meta ["cfg" "dh_head"])))]
        (mapv (fn [{:keys [type ans opt]}]
                (let [h (t/rows hq ans 1)]
                  (if (= "noul" type)
                    (let [z (+ (dot (t/to-floats (g "h.noul.weight")) (t/to-floats h)) (first (t/to-floats (g "h.noul.bias"))))]
                      [0.0 (/ z (nth temp 1))])
                    (let [sel (if (= "score" type) 1 0)
                          qa (linear h (g (str "h.A." sel ".weight")))
                          zs (t/to-floats (linear (linear (gather hq opt) (g (str "h.O." sel ".weight"))) qa))
                          s (Math/exp (nth scales sel))
                          tt (nth temp (if (= "score" type) 2 0))]
                      (mapv #(/ (* s %) dh tt) zs)))))
              (:qs enc))))))

(defn- softmax [xs]
  (let [mx (reduce max xs)
        ex (mapv #(Math/exp (- (double %) mx)) xs)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn answer
  "The release's own answers to questions ([{:type :text :options}]) about
  `state`: {:type \"choice\"|\"score\" :probs :pick :confidence (:score, 0
  to 1)} or {:type \"noul\" :p}, in order."
  [m state questions]
  (let [enc (encode-request m state questions)]
    (mapv (fn [{:keys [type]} z]
            (let [p (softmax z)
                  k (count p)]
              (if (= "noul" type)
                {:type type :p (second p)}
                (cond-> {:type type :probs p
                         :pick (reduce (fn [bi i] (if (> (nth p i) (nth p bi)) i bi)) 0 (range k))
                         :confidence (ag/confidence-from-probs p k)}
                  (= "score" type) (assoc :score (/ (reduce + (map-indexed * p)) (dec k)))))))
          (:qs enc) (logits m enc))))

;; --- the agent ------------------------------------------------------------------

(defn tinydecide
  "A TinyDecide agent from its config ({:name :model (the release
  directory) :calibration})."
  [cfg]
  (let [cfg (merge defaults cfg)
        m (load-release (:model cfg))]
    {:kind :tinydecide :name (or (:name cfg) "tinydecide") :cfg cfg :m m
     :close (fn [_] (free-release! m))}))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- as-question
  "A validated lev question as the release asks it: a choice's options are
  their descriptions (the label when there is none: bench/README.md has
  the measurement), a score's its levels, a noul's statement its
  instructions."
  [qid {:keys [t ins crit]}]
  {:qid qid :type t :text ins
   :options (case t
              "choice" (mapv (fn [[k v]] (if (and (string? v) (seq v)) v (key-str k))) crit)
              "score" (mapv #(if (string? %) % (seq/json-str %)) crit)
              nil)})

(defn- prepare [questions]
  (mapv (fn [[qid qdef]] (let [qdef (ag/validate-question qid qdef)] {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
        questions))

(defn- run [{:keys [m]} state prepared]
  (let [enc (encode-request m (seq/serialize-state state) (map (fn [{:keys [qid q]}] (as-question qid q)) prepared))]
    {:logits (logits m enc) :tokens (count (:ids enc)) :dropped (:dropped enc)}))

(defmethod ag/forward :tinydecide [agent state questions]
  (let [prepared (prepare questions)
        {:keys [logits tokens]} (run agent state prepared)]
    (mapv (fn [{:keys [q] :as pq} z]
            (assoc pq :qtype (seq/qtypes (:t q)) :k (count z) :logits z :tokens tokens))
          prepared logits)))

(defmethod ag/system-one* :tinydecide
  [{:keys [cfg] :as agent} state questions {:keys [constraints on-infeasible]}]
  (let [prepared (prepare questions)
        cs (ag/prepare-constraints (map (juxt :qid :qdef) prepared) constraints)
        {:keys [logits tokens dropped]} (run agent state prepared)
        cal (:calibration cfg)
        ps (mapv (fn [{:keys [q]} z]
                   (let [tmp (ag/temperature-for cal (seq/qtypes (:t q)) (count z))]
                     (softmax (mapv #(/ (double %) tmp) z))))
                 prepared logits)
        solution (when cs
                   (ag/decide-constraints cs (map (fn [{:keys [qid q]} p] [qid q p]) prepared ps) on-infeasible))
        answers (seq/ordered-map
                 (map (fn [{:keys [qid q]} p]
                        [qid (ag/typed-answer q p (count p) (when solution (ag/decided-label cs solution qid)) nil)])
                      prepared ps))]
    (seq/ordered-map
     (concat [["model" (:name agent)]
              ["answers" answers]
              ["usage" (array-map "input_tokens" tokens "output_tokens" 0)]]
             (when (pos? dropped) [["truncated" (seq/ordered-map (map (fn [{:keys [qid]}] [qid dropped]) prepared))]])
             (when solution [["constraints" (ag/constraints-report cs solution)]])))))
