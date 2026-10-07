(ns lev.clef
  "Clef (Cloudflare/clef): a Qwen3.5-architecture backbone with a joint
  schema head. No text is generated: the state and every question with its
  options go into one prompt, the backbone's final-normed hidden state at
  every token goes to the head, and the head answers one logit per option
  of every question at once (it routes evidence from the whole prompt to
  each option and lets the questions attend to each other). Same request,
  same answer shapes as the encoders and thinkers.

  The backbone is a GGUF (convert_hf_to_gguf.py --no-mtp on the release)
  loaded through lev.llm; its hidden states and the output embedding rows
  the head's lexical prior reads come from native/lev_clef.cpp. The head
  is the release's joint_head.safetensors (bf16, read into f32) run here
  over cblas and lev_clef.cpp's small ops, held to the release's torch
  head by golden/clef (bench/clef_golden.py).

  An agent is {:kind :clef :name :cfg :llm :head :rows}; lev.think/thinker
  builds one when a thinker's config says :engine \"clef\" (config.edn
  :thinkers {\"clef\" {:engine \"clef\" :model GGUF :head DIR}}), so the
  router loads it like any other thinker."
  (:require [clojure.data.json :as json]
            [clojure.string :as str]
            [jolt.ffi :as ffi]
            [lev.agent :as ag]
            [lev.llm :as llm]
            [lev.prepare :as prep]
            [lev.sequence :as seq]
            [lev.tensors :as t]))

(ffi/defcfn tokenize* "lev_llm_tokenize" [:pointer :pointer :int :int :pointer :int] :int)
(ffi/defcfn hidden* "lev_llm_hidden" [:pointer :pointer :int :pointer] :int :blocking)
(ffi/defcfn n-embd* "lev_llm_n_embd" [:pointer] :int)
(ffi/defcfn rows-open* "lev_rows_open" [:pointer :pointer] :pointer)
(ffi/defcfn rows-n-cols* "lev_rows_n_cols" [:pointer] :int64)
(ffi/defcfn rows-get* "lev_rows_get" [:pointer :pointer :int :pointer] :int)
(ffi/defcfn rows-close* "lev_rows_close" [:pointer] :void)
(ffi/defcfn bf16-read* "lev_clef_bf16_read" [:pointer :int64 :int64 :pointer] :int64)
(ffi/defcfn add-bias* "lev_clef_add_bias" [:pointer :pointer :int64 :int64] :void)
(ffi/defcfn span-mean* "lev_clef_span_mean" [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn attend-pool* "lev_clef_attend_pool" [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn pair-features* "lev_clef_pair_features" [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn clef-logits* "lev_clef_logits"
  [:pointer :pointer :int64 :int64 :pointer :pointer :int64 :pointer :float :float :float :pointer] :void)

(def system-prompt
  "Read the complete state and schema. Decide every field jointly. Each answer must be exactly one of that field's allowed options.")

(def defaults
  {:n-ctx 16384 :n-gpu-layers -1 :threads 0
   ;; the release's own bound on a prompt (encode_record's max_length)
   :max-length 16384
   :calibration {:temperature [1.0 1.0 1.0]}})

;; --- the head's weights -------------------------------------------------------

(defn load-head
  "The joint schema head of the release at `dir` (joint_head.safetensors
  and joint_head_config.json) as f32 tensors: {:config {...} :w {name
  tensor} :scalars {name double}}. A bias of the attention's packed
  projection is [3 x width], so its q / k / v parts are row views."
  [dir]
  (let [path (str dir "/joint_head.safetensors")
        config (into {} (map (fn [[k v]] [(keyword k) v])) (json/read-str (slurp (str dir "/joint_head_config.json"))))
        [hdr base] (prep/read-header path)
        width (:width config)
        read (fn [{:strs [dtype shape data_offsets]} name]
               (when-not (= "BF16" dtype)
                 (throw (ex-info (str "clef head: " name " is " dtype ", expected BF16") {:type :model-unavailable :name name})))
               (let [n (reduce * 1 shape)
                     shape (cond (empty? shape) [1 1]
                                 (str/ends-with? name "in_proj_bias") [3 width]
                                 (= 1 (count shape)) [1 (first shape)]
                                 :else shape)
                     tt (binding [t/*arena* nil] (t/make shape))]
                 (ffi/with-arena [a]
                   (when (not= n (bf16-read* (ffi/string->ptr a path) (+ base (first data_offsets)) n (t/ptr tt)))
                     (throw (ex-info (str "clef head: cannot read " name) {:type :model-unavailable :name name}))))
                 tt))
        w (into {} (for [[name meta] (dissoc hdr "__metadata__")] [name (read meta name)]))
        scalar #(double (t/get* (t/ptr (get w %)) 0))]
    {:config config
     :w w
     :scalars {:prior (scalar "prior_logit_scale") :joint (scalar "joint_logit_scale") :gate (scalar "residual_gate")}}))

(defn free-head! [{:keys [w]}]
  (doseq [x (vals w)] (ffi/free (t/ptr x))))

;; --- tensor helpers -------------------------------------------------------------

(defn- offset [p bytes] (ffi/segment (+ (ffi/address p) bytes)))

(defn- linear
  "x W^T (+ b): torch Linear over [n x in] -> [n x out]."
  ([x W] (t/mmul x W))
  ([x W b] (let [y (t/mmul x W)] (add-bias* (t/ptr y) (t/ptr b) (first (t/shape y)) (second (t/shape y))) y)))

(defn- ln [x w b] (t/layernorm x w b 1e-5))

(defn- add! [a b] (t/add-scaled! a a b 1.0))

(defn- span-rows
  "[count x d] of the means of src's rows over each [s e) span."
  [src spans]
  (let [d (second (t/shape src))
        out (t/make [(count spans) d])]
    (doseq [[i [s e]] (map-indexed vector spans)]
      (span-mean* (t/ptr src) d s e (offset (t/ptr out) (* 4 i d))))
    out))

(defn mean-rows
  "[1 x d]: the mean of the rows `ids` of the [v x d] table (the lexical
  vector of an option, over the output embedding)."
  [table ids]
  (let [acc (t/make [1 (second (t/shape table))])]
    (doseq [i ids] (add! acc (t/rows table i 1)))
    (t/add-scaled! acc acc acc (- (/ 1.0 (max 1 (count ids))) 1.0))))

(defn- mha
  "torch MultiheadAttention (batch_first, no mask): queries x [nq x w]
  over memory m [nk x w], packed in_proj, `heads` heads."
  [w prefix x m heads]
  (let [W (get w (str prefix ".in_proj_weight"))
        B (get w (str prefix ".in_proj_bias"))
        width (second (t/shape x))
        hd (quot width heads)
        nq (first (t/shape x))
        nk (first (t/shape m))
        q (linear x (t/rows W 0 width) (t/rows B 0 1))
        k (linear m (t/rows W width width) (t/rows B 1 1))
        v (linear m (t/rows W (* 2 width) width) (t/rows B 2 1))
        scores (t/make [nq nk])
        ctx (t/make [nq width])
        scale (/ 1.0 (Math/sqrt hd))]
    (dotimes [h heads]
      (let [off (* 4 h hd)]
        (t/cblas-sgemm* 101 111 112 nq nk hd (float scale)
                        (offset (t/ptr q) off) width (offset (t/ptr k) off) width
                        0.0 (t/ptr scores) nk)
        (let [p (t/softmax scores nk)]
          (t/cblas-sgemm* 101 111 111 nq hd nk 1.0
                          (t/ptr p) nk (offset (t/ptr v) off) width
                          0.0 (offset (t/ptr ctx) off) width))))
    (linear ctx (get w (str prefix ".out_proj.weight")) (get w (str prefix ".out_proj.bias")))))

(defn- ff [w l1 l2 x]
  (linear (t/gelu! (linear x (get w (str l1 ".weight")) (get w (str l1 ".bias"))))
          (get w (str l2 ".weight")) (get w (str l2 ".bias"))))

(defn- routing-layer
  "EvidenceRoutingLayer: options attend over the (normed) memory, then a
  feedforward, both residual."
  [w i x memory heads]
  (let [p (str "evidence_layers." i)
        g #(get w (str p "." %))
        att (mha w (str p ".attention") (ln x (g "query_norm.weight") (g "query_norm.bias"))
                 (ln memory (g "memory_norm.weight") (g "memory_norm.bias")) heads)
        x (add! x att)]
    (add! x (ff w (str p ".feedforward.0") (str p ".feedforward.3")
                (ln x (g "feedforward_norm.weight") (g "feedforward_norm.bias"))))))

(defn- decoder-layer
  "nn.TransformerDecoderLayer, norm_first, gelu: self-attention over the
  fields, cross-attention over the memory, feedforward."
  [w i x memory heads]
  (let [p (str "layers." i)
        g #(get w (str p "." %))
        x (add! x (let [y (ln x (g "norm1.weight") (g "norm1.bias"))] (mha w (str p ".self_attn") y y heads)))
        x (add! x (mha w (str p ".multihead_attn") (ln x (g "norm2.weight") (g "norm2.bias")) memory heads))]
    (add! x (ff w (str p ".linear1") (str p ".linear2") (ln x (g "norm3.weight") (g "norm3.bias"))))))

(def ^:private type-index {"noul" 0 "choice" 1 "score" 2})

(defn head-logits
  "The head on one record: hidden [n x D] (the backbone's final-normed
  states), input-ids (n token ids), questions [{:type :span [s e]
  :option-spans [[s e] ...]}] (token spans, as encode_record lays them),
  and `lexical` (token ids -> [1 x D], the mean output embedding of
  them). Answers per question its options' logits, in Clef's option order."
  [{:keys [config w scalars]} hidden input-ids questions lexical]
  (ffi/with-arena [a]
    (binding [t/*arena* a]
      (let [{:keys [heads routing_layers layers width]} config
            [n D] (t/shape hidden)
            g #(get w %)
            H (ln hidden (g "hidden_norm.weight") (g "hidden_norm.bias"))
            memory (linear H (g "memory_projection.weight"))
            gv (t/rows H (dec n) 1)
            qv (span-rows H (mapv :span questions))
            counts (mapv (comp count :option-spans) questions)
            ospans (vec (mapcat :option-spans questions))
            no (count ospans)
            octx (span-rows H ospans)
            lex (t/make [no D])
            _ (doseq [[i [s e]] (map-indexed vector ospans)]
                (span-mean* (t/ptr (lexical (subvec (vec input-ids) s e))) D 0 1 (offset (t/ptr lex) (* 4 i D))))
            ;; each option's question vector, repeated per option
            oq (span-rows qv (vec (mapcat (fn [qi k] (repeat k [qi (inc qi)])) (range) counts)))
            routed (reduce (fn [x i] (routing-layer w i x memory heads))
                           (-> (linear octx (g "option_context_projection.weight"))
                               (add! (linear lex (g "option_lexical_projection.weight")))
                               (add! (linear oq (g "option_question_projection.weight"))))
                           (range routing_layers))
            starts (vec (reductions + 0 counts))
            base (linear qv (g "question_projection.weight"))
            nq (count questions)
            summaries (t/make [nq width])
            _ (dotimes [qi nq]
                (attend-pool* (t/ptr (t/rows base qi 1)) (t/ptr (t/rows routed (nth starts qi) (nth counts qi)))
                              (nth counts qi) width (offset (t/ptr summaries) (* 4 qi width))))
            fields (-> (t/add-scaled (t/make [nq width]) base 1.0)
                       (add! (ln summaries (g "option_summary_norm.weight") (g "option_summary_norm.bias"))))
            _ (add-bias* (t/ptr fields) (t/ptr (linear gv (g "global_projection.weight"))) nq width)
            _ (doseq [[qi {:keys [type]}] (map-indexed vector questions)]
                (add-bias* (t/ptr (t/rows fields qi 1)) (t/ptr (t/rows (g "type_embedding.weight") (type-index type) 1)) 1 width))
            fields (reduce (fn [x i] (decoder-layer w i x memory heads)) fields (range layers))
            fields (ln fields (g "field_norm.weight") (g "field_norm.bias"))
            cap (Math/log 100.0)
            prior-scale (Math/exp (min (:prior scalars) cap))
            joint-scale (Math/exp (min (:joint scalars) cap))
            gate (/ 1.0 (+ 1.0 (Math/exp (- (:gate scalars)))))]
        (mapv (fn [qi]
                (let [k (nth counts qi)
                      s (nth starts qi)
                      field (t/rows fields qi 1)
                      opts (ln (t/rows routed s k) (g "option_norm.weight") (g "option_norm.bias"))
                      feats (t/make [k (* 4 width)])
                      _ (pair-features* (t/ptr field) (t/ptr opts) k width (t/ptr feats))
                      residual (linear (t/gelu! (linear feats (g "residual_scorer.0.weight") (g "residual_scorer.0.bias")))
                                       (g "residual_scorer.3.weight") (g "residual_scorer.3.bias"))
                      anchor (t/add-scaled (t/rows qv qi 1) gv 1.0)
                      out (t/make [1 k])]
                  (clef-logits* (t/ptr field) (t/ptr opts) k width (t/ptr (t/rows lex s k)) (t/ptr anchor) D
                                (t/ptr residual) (float prior-scale) (float joint-scale) (float gate) (t/ptr out))
                  (t/to-floats out)))
              (range nq))))))

;; --- the record -------------------------------------------------------------------

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- sorted-json
  "v with every map's keys sorted, as json.dumps(sort_keys=True) orders them."
  [v]
  (cond
    (map? v) (seq/ordered-map (sort-by first (map (fn [[k x]] [(key-str k) (sorted-json x)]) v)))
    (sequential? v) (mapv sorted-json v)
    :else v))

(defn render
  "The release's render: a string as itself, anything else compact JSON
  with sorted keys (ensure_ascii off)."
  [v]
  (if (string? v) v (seq/json-str (sorted-json v) {:compact true})))

(defn- qget [m k] (if (contains? m k) (get m k) (get m (keyword k))))

(defn question-options
  "[[option-id description] ...] in Clef's order: a noul's true, false
  (the release's default descriptions under the caller's); a choice's by
  sorted id (a list of options is ids without descriptions); a score's
  levels by index."
  [qdef]
  (let [crit (qget qdef "criteria")]
    (case (qget qdef "type")
      "noul" (let [c (merge {"true" "The proposition is true or the answer is yes."
                             "false" "The proposition is false or the answer is no."}
                            (into {} (map (fn [[k v]] [(key-str k) v])) crit))]
               [["true" (get c "true")] ["false" (get c "false")]])
      "choice" (vec (sort-by first (if (map? crit)
                                     (map (fn [[k v]] [(key-str k) v]) crit)
                                     (map (fn [c] [(key-str c) nil]) crit))))
      "score" (vec (map-indexed (fn [i v] [(str i) v]) crit)))))

(defn- tokens
  "The ids of `text`; control tokens' texts are read as those tokens only
  when `special?` (the template's own text, never the caller's)."
  [m text special?]
  (let [text (str text)
        cap (+ 16 (* 2 (count (.getBytes text "UTF-8"))))]
    (ffi/with-arena [a]
      (let [out (ffi/alloc a (* 4 cap))
            n (tokenize* (:p m) (ffi/string->ptr a text) 0 (if special? 1 0) out (int cap))]
        (when (neg? n)
          (throw (ex-info (str "clef: cannot tokenize: " (llm/error* (:p m))) {:type :llm-error})))
        (mapv #(ffi/read out :int (* 4 %)) (range n))))))

(defn encode
  "encode_record: the token ids of a state and its questions ([qid qdef]
  pairs or a map), and per question its {:id :type :span :option-spans
  :option-ids} over them. The state is cut from its end to fit
  :max-length (:state-dropped says how many tokens went)."
  [{:keys [llm cfg]} state questions]
  (let [tok #(tokens llm %1 %2)
        [schema qs] (reduce (fn [[ids qs] [qi [qid qdef]]]
                              (let [ids (into ids (tok (str "\nFIELD " (inc qi) "\nID: " (key-str qid) "\nTYPE: " (qget qdef "type") "\nINSTRUCTION: ") false))
                                    ins (qget qdef "instructions")
                                    qs0 (count ids)
                                    ids (into ids (tok (render (if (or (nil? ins) (= "" ins)) (key-str qid) ins)) false))
                                    span [qs0 (count ids)]
                                    ids (into ids (tok "\nALLOWED OPTIONS:\n" false))
                                    opts (question-options qdef)
                                    [ids ospans] (reduce (fn [[ids sp] [oi [oid desc]]]
                                                           (let [ids (into ids (tok (str "OPTION " (inc oi) ": ") false))
                                                                 s (count ids)
                                                                 ids (into ids (tok (render (cond-> (array-map "option_id" oid)
                                                                                              (some? desc) (assoc "description" desc)))
                                                                                    false))
                                                                 e (count ids)]
                                                             [(into ids (tok "\n" false)) (conj sp [s e])]))
                                                         [ids []] (map-indexed vector opts))
                                    ids (into ids (tok "END FIELD\n" false))]
                                [ids (conj qs {:id (key-str qid) :type (qget qdef "type") :span span
                                               :option-spans ospans :option-ids (mapv first opts)})]))
                            [(tok "\n\nSCHEMA FIELDS:\n" false) []]
                            (map-indexed vector questions))
        prefix (tok (str "<|im_start|>system\n" system-prompt "<|im_end|>\n<|im_start|>user\nSTATE:\n") true)
        suffix (tok "\n<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\nJOINT SCHEMA DECISIONS:" true)
        fixed (+ (count prefix) (count schema) (count suffix))
        max-length (min (:max-length cfg) (llm/n-ctx llm))
        _ (when (> fixed max-length)
            (throw (ex-info (str "clef: the schema takes " fixed " tokens before the state; the most is " max-length)
                            {:type :invalid-request :tokens fixed :max max-length})))
        state-ids (tok (render state) false)
        kept (vec (take (- max-length fixed) state-ids))
        shift (+ (count prefix) (count kept))
        mv (fn [[s e]] [(+ s shift) (+ e shift)])]
    {:input-ids (vec (concat prefix kept schema suffix))
     :state-dropped (- (count state-ids) (count kept))
     :questions (mapv #(-> % (update :span mv) (update :option-spans (partial mapv mv))) qs)}))

;; --- the agent ---------------------------------------------------------------------

(defn- hidden-states
  "[n x D] final-normed hidden states of the ids, in the current arena."
  [{:keys [llm]} ids]
  (let [n (count ids)
        d (n-embd* (:p llm))
        out (t/make [n d])]
    (ffi/with-arena [a]
      (let [p (ffi/alloc a (* 4 n))]
        (dotimes [i n] (ffi/write p :int (int (nth ids i)) (* 4 i)))
        (when (neg? (hidden* (:p llm) p (int n) (t/ptr out)))
          (throw (ex-info (str "clef: " (llm/error* (:p llm))) {:type :llm-error})))))
    out))

(defn- lexical-fn
  "token ids -> [1 x D]: their mean row of the GGUF's output embedding."
  [rows d]
  (fn [ids]
    (let [k (count ids)
          buf (t/make [(max 1 k) d])
          out (t/make [1 d])]
      (ffi/with-arena [a]
        (let [p (ffi/alloc a (* 4 (max 1 k)))]
          (dotimes [i k] (ffi/write p :int (int (nth ids i)) (* 4 i)))
          (when (neg? (rows-get* rows p (int k) (t/ptr buf)))
            (throw (ex-info "clef: cannot read output embedding rows" {:type :llm-error})))))
      (span-mean* (t/ptr buf) d 0 k (t/ptr out))
      out)))

(defn clef
  "A Clef agent from its config ({:name :model (the backbone GGUF) :head
  (the release directory with joint_head.*) :n-ctx :n-gpu-layers :threads
  :max-length :calibration})."
  [cfg]
  (let [cfg (merge defaults cfg)
        _ (when-not (:head cfg)
            (throw (ex-info "a clef model needs :head, the release directory holding joint_head.safetensors"
                            {:type :invalid-config :model (:name cfg)})))
        m (llm/load (:model cfg) (assoc (select-keys cfg [:n-ctx :n-gpu-layers :threads]) :n-seq-max 1))
        head (load-head (:head cfg))
        rows (ffi/with-arena [a] (rows-open* (ffi/string->ptr a (:model cfg)) (ffi/string->ptr a "output.weight")))]
    (when (zero? (ffi/address rows))
      (llm/free! m)
      (throw (ex-info (str (:model cfg) " has no readable output.weight") {:type :model-unavailable :model (:model cfg)})))
    {:kind :clef :name (or (:name cfg) "clef") :cfg cfg :llm m :head head :rows rows
     :close (fn [_] (rows-close* rows) (free-head! head) (llm/free! m))}))

(defn- prepare [questions]
  (mapv (fn [[qid qdef]] (let [qdef (ag/validate-question qid qdef)] {:qid qid :qdef qdef :q (ag/to-internal qdef)}))
        questions))

(defn- answer-order
  "The logits of a question in lev's answer order (the caller's options;
  a noul's false, true) from Clef's order."
  [{:keys [q]} option-ids logits]
  (let [by-id (zipmap option-ids logits)]
    (case (:t q)
      "noul" [(get by-id "false") (get by-id "true")]
      "score" (mapv #(get by-id (str %)) (range (count (:crit q))))
      (mapv #(get by-id (key-str %)) (keys (:crit q))))))

(defn- run
  "Per prepared question {:logits (answer order) :k}, the input tokens and
  the state tokens dropped."
  [agent state prepared]
  (let [{:keys [input-ids questions state-dropped]} (encode agent state (map (juxt :qid :qdef) prepared))
        logits (ffi/with-arena [a]
                 (binding [t/*arena* a]
                   (head-logits (:head agent) (hidden-states agent input-ids) input-ids questions
                                (lexical-fn (:rows agent) (n-embd* (:p (:llm agent)))))))]
    {:asked (mapv (fn [pq {:keys [option-ids]} l] {:logits (answer-order pq option-ids l) :k (count l)})
                  prepared questions logits)
     :tokens (count input-ids)
     :dropped state-dropped}))

(defmethod ag/forward :clef [agent state questions]
  (let [prepared (prepare questions)
        {:keys [asked tokens]} (run agent state prepared)]
    (mapv (fn [{:keys [q] :as pq} {:keys [logits k]}]
            (assoc pq :qtype (seq/qtypes (:t q)) :k k :logits logits :tokens tokens))
          prepared asked)))

(defmethod ag/system-one* :clef
  [{:keys [cfg] :as agent} state questions {:keys [constraints on-infeasible]}]
  (let [prepared (prepare questions)
        cs (ag/prepare-constraints (map (juxt :qid :qdef) prepared) constraints)
        {:keys [asked tokens dropped]} (run agent state prepared)
        cal (:calibration cfg)
        ps (mapv (fn [{:keys [q]} {:keys [logits k]}]
                   (let [tmp (ag/temperature-for cal (seq/qtypes (:t q)) k)
                         z (mapv #(/ (double %) tmp) logits)
                         mx (reduce max z)
                         ex (mapv #(Math/exp (- % mx)) z)
                         s (reduce + ex)]
                     (mapv #(/ % s) ex)))
                 prepared asked)
        solution (when cs
                   (ag/decide-constraints cs (map (fn [{:keys [qid q]} p] [qid q p]) prepared ps) on-infeasible))
        answers (seq/ordered-map
                 (map (fn [{:keys [qid q]} {:keys [k]} p]
                        [qid (ag/typed-answer q p k (when solution (ag/decided-label cs solution qid)) nil)])
                      prepared asked ps))]
    (seq/ordered-map
     (concat [["model" (:name agent)]
              ["answers" answers]
              ["usage" (array-map "input_tokens" tokens "output_tokens" 0)]]
             (when (pos? dropped) [["truncated" (seq/ordered-map (map (fn [{:keys [qid]}] [qid dropped]) prepared))]])
             (when solution [["constraints" (ag/constraints-report cs solution)]])))))
