(ns lev.agent
  "system-one over an encoder: state + typed questions -> calibrated typed
  answers, as the checkpoints' own Python package computes them
  (temperature calibration, entropy confidence, the `action` extension)."
  (:require [clojure.edn :as edn]
            [jolt.ffi :as ffi]
            [lev.constraints :as c]
            [lev.model :as m]
            [lev.sequence :as seq]
            [lev.tensors :as t]
            [lev.tokenizer :as tk]))

(defn with-limits
  "The agent with :max-len / :head-max-len overridden (only the keys given).
  The checkpoint's trained max_len is kept as :trained-max-len for the logs:
  RoPE has no position table, so a longer sequence works mechanically, but
  quality past the trained length is unmeasured. Throws on values that
  cannot form a sequence."
  [agent {:keys [max-len head-max-len] :as limits}]
  (let [cfg (:cfg agent)
        max-len (or max-len (:max-len cfg))
        head-max-len (or head-max-len (:head-max-len cfg))
        bad (fn [msg] (throw (ex-info (str "invalid sequence limits " (pr-str limits) ": " msg)
                                      {:type :invalid-limits :limits limits})))]
    (when-not (and (integer? max-len) (integer? head-max-len)) (bad "max-len and head-max-len must be integers"))
    (when (< head-max-len 16) (bad "head-max-len must be at least 16 (build_sequence's own floor)"))
    (when (< max-len (+ head-max-len 8)) (bad "max-len must exceed head-max-len with room for the state"))
    (-> agent
        (update :cfg assoc :max-len max-len :head-max-len head-max-len)
        (assoc :trained-max-len (or (:trained-max-len agent) (:max-len cfg))))))

(defn agent-shell
  "Everything of an encoder agent but its weights: config, tokenizer,
  manifest, the prefix cache, `limits` ({:max-len :head-max-len}, see
  with-limits) over the checkpoint's, and :name, what the answers report
  as \"model\" (the router passes the checkpoint's name; alone, an agent is
  \"encoder\"). A backend completes it with :w (its weights, whatever it
  holds them in) and :forward, its lev.model/forward-batch."
  [data-dir limits]
  (let [manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        tok (tk/load (str data-dir "/tokenizer.edn"))
        agent {:kind :encoder :name (or (:name limits) "encoder") :backend :cpu :dtype :f32
               :cfg cfg :tok tok :manifest manifest :data-dir data-dir :trained-max-len (:max-len cfg)
               ;; the questions' prefixes, tokenized once and kept across
               ;; calls (lev.sequence/prefix-cache); the map copies that
               ;; with-limits and with-calibration make share it
               :prefix-cache (seq/prefix-cache)}
        limits (dissoc limits :name :backend :dtype :calibration :selected-head)]
    (if (seq limits) (with-limits agent limits) agent)))

(defn load-agent
  "Load config + tokenizer + all weights from data-dir onto the C kernels;
  `limits` as agent-shell's. The weights are malloc'd: :close gives them
  back (the router calls it when the agent is evicted or unloaded)."
  ([data-dir] (load-agent data-dir nil))
  ([data-dir limits]
   (let [agent (agent-shell data-dir limits)
         freed (atom false)]
     (assoc agent
            :w (m/load-weights (:manifest agent) data-dir)
            :forward m/forward-batch
            ;; once: a second close (the same map in two routers) must not double-free
            :close (fn [a] (when (compare-and-set! freed false true)
                             (doseq [t (vals (:w a))] (ffi/free (t/ptr t)))))))))

(defn- qget
  "Question defs may carry keyword keys (Clojure literals) or string keys
  (parsed JSON, as the Python API receives them)."
  [qdef k]
  (if (contains? qdef k) (get qdef k) (get qdef (name k))))

(defn- invalid [qid field msg]
  (throw (ex-info (str "question " (pr-str qid) ": " msg)
                  {:type :invalid-question :qid qid :field field})))

(defn validate-question
  "The shape rl_agent_api needs (it fails with KeyError/AttributeError
  otherwise) and the Jev API documents: a known type, instructions, and
  criteria fitting the type. Throws ex-info {:type :invalid-question}."
  [qid qdef]
  (when-not (map? qdef) (invalid qid nil "must be an object"))
  (let [t (qget qdef :type)
        crit (qget qdef :criteria)]
    (when-not (contains? seq/qtypes t)
      (invalid qid "type" (str "unknown type " (pr-str t) "; expected choice, score or noul")))
    (when (nil? (qget qdef :instructions))
      (invalid qid "instructions" "instructions is required"))
    (case t
      "choice" (when-not (or (and (map? crit) (seq crit)) (and (sequential? crit) (seq crit)))
                 (invalid qid "criteria" "choice criteria must be a non-empty map of option -> description (or a list of options)"))
      "score" (when-not (and (sequential? crit) (>= (count crit) 2))
                (invalid qid "criteria" "score criteria must be a list of at least 2 levels"))
      "noul" (when-not (or (nil? crit) (map? crit))
               (invalid qid "criteria" "noul criteria must be a map with optional \"true\" / \"false\" descriptions")))
    qdef))

(defn to-internal
  "Jev question def -> {:t :ins :crit} (rl_agent_api.RLAgent._to_internal).
  A list of choice criteria becomes {c: None}; non-string instructions are
  json.dumps'd with its default ensure_ascii=True."
  [qdef]
  (let [t (qget qdef :type)
        crit (qget qdef :criteria)
        crit (if (and (= t "choice") (sequential? crit))
               (seq/ordered-map (map (fn [c] [c nil]) crit))
               crit)]
    {:t t
     :ins (let [ins (qget qdef :instructions)]
            (if (string? ins) ins (seq/json-str ins {:ensure-ascii true})))
     :crit crit}))

(def max-batch
  "Rows per forward-batch. Padded to the longest row, 8 rows of 512 tokens
  take a ~260 MB workspace; more rows buy little once the gemms are wide."
  8)

(defn round4 [x] (/ (double (Math/round (* 1e4 (double x)))) 1e4))

(defn- softmax [xs]
  (let [mx (reduce max xs)
        ex (mapv #(Math/exp (- (double %) mx)) xs)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn confidence-from-probs
  "1 - normalized entropy, clipped to [0, 1] (confidence_from_probs)."
  [p k]
  (if (< k 2)
    1.0
    (let [ent (- (reduce + (map (fn [pi] (* pi (Math/log (max pi 1e-12)))) (take k p))))]
      (-> (- 1.0 (/ ent (Math/log k))) (max 0.0) (min 1.0)))))

(defn- argmax [xs]
  (reduce (fn [bi i] (if (> (nth xs i) (nth xs bi)) i bi)) 0 (range (count xs))))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn typed-answer
  "One typed answer from the calibrated probabilities p (in option order;
  a noul's are [false true], so \"noul\" is p[1]). With a decision
  (`decided`, the label the constrained decoder settled on) it follows the
  model's own answer field; `extra` ([k v] pairs, e.g. the encoders' action
  head) closes the map. Without either, the map is the Python package's exactly."
  [q p k decided extra]
  (let [decided-kv (when (some? decided) [["decided" decided]])]
    (seq/ordered-map
     (case (:t q)
       "choice"
       (let [ks (mapv key-str (keys (:crit q)))]
         (concat [["type" "choice"] ["choice" (nth ks (argmax p))]]
                 decided-kv
                 [["probabilities" (seq/ordered-map (map-indexed (fn [i c] [c (round4 (nth p i))]) ks))]
                  ["confidence" (round4 (confidence-from-probs p k))]]
                 extra))
       "score"
       (concat [["type" "score"] ["score" (round4 (reduce + (map-indexed (fn [i pi] (* i pi)) p)))]]
               decided-kv
               [["legend" (seq/ordered-map (map-indexed (fn [i c] [(str i) c]) (:crit q)))]
                ["probabilities" (seq/ordered-map (map-indexed (fn [i pi] [(str i) (round4 pi)]) p))]
                ["confidence" (round4 (confidence-from-probs p k))]]
               extra)
       (let [p1 (double (nth p 1))]
         (concat [["type" "noul"] ["noul" (round4 p1)]]
                 decided-kv
                 [["confidence" (round4 (max p1 (- 1.0 p1)))]]
                 extra))))))

(defn- label-probs
  "p in the constraint schema's label order: a noul's options are rendered
  false first (\"noul\" is p[1]), its constraint labels are [true false]."
  [q p]
  (if (= "noul" (:t q)) [(nth p 1) (nth p 0)] p))

;; --- constraints, shared by every engine -------------------------------------

(defn prepare-constraints
  "The constraint schema and parsed nodes for `constraints` over the
  validated question defs ([qid qdef] pairs), checked before any
  inference; nil when there are no constraints (nil, not [])."
  [qid-qdefs constraints]
  (when (some? constraints)
    (let [schema (c/schema (map (fn [[qid qdef]] [(key-str qid) qdef]) qid-qdefs))]
      {:schema schema :nodes (c/parse schema constraints)})))

(defn decide-constraints
  "The joint decision over the per-question probabilities ([qid q p]
  triples, p in answer order) under prepared constraints: the solution
  (see lev.constraints/decode), throwing {:type :infeasible} when
  on-infeasible is \"raise\" and nothing fits."
  [{:keys [schema nodes]} qid-q-ps on-infeasible]
  (let [sol (c/decode schema
                      (into {} (map (fn [[qid q p]] [(key-str qid) (label-probs q p)]) qid-q-ps))
                      nodes {})]
    (when (and (not (:feasible sol)) (= "raise" (some-> on-infeasible name)))
      (throw (ex-info "no assignment satisfies the constraints"
                      {:type :infeasible
                       :violations (mapv #(c/canonical (nth nodes %)) (:violations sol))})))
    sol))

(defn decided-label
  "The label the solution chose for qid, as the answer spells it."
  [{:keys [schema]} solution qid]
  (let [q (key-str qid)]
    (nth (get-in schema [q :labels]) (get-in solution [:assignment q]))))

(defn answer-probs
  "The probabilities back out of a typed answer, in answer order ([false
  true] for a noul), for deciding constraints over answers that came from
  different models (lev.patterns/escalate)."
  [answer]
  (case (get answer "type")
    "choice" (vec (vals (get answer "probabilities")))
    "score" (mapv #(get (get answer "probabilities") (str %)) (range (count (get answer "legend"))))
    (let [p (double (get answer "noul"))] [(- 1.0 p) p])))

(defn with-decided
  "The answer with `decided` right after its own field (choice / score /
  noul), replacing one already there."
  [answer decided]
  (let [own (get answer "type")]
    (seq/ordered-map
     (mapcat (fn [[k v]]
               (cond (= k "decided") []
                     (= k own) [[k v] ["decided" decided]]
                     :else [[k v]]))
             answer))))

(defn constraints-report [{:keys [nodes]} sol]
  (seq/ordered-map [["feasible" (:feasible sol)]
                    ["decoder" (:decoder sol)]
                    ["exact" (:exact sol)]
                    ["violations" (mapv #(c/canonical (nth nodes %)) (:violations sol))]]))

(defmulti system-one*
  "The engine behind system-one, by the agent's :kind: :encoder (here),
  :thinker (lev.think) or :clef (lev.clef)."
  (fn [agent _state _questions _opts] (:kind agent :encoder)))

(defn system-one
  "state + {qid -> qdef} -> Jev answer map (ordered to match json.dumps).

  opts (all optional): :constraints, a list of lev.constraints over the
  question ids, decided jointly after the forward pass. When it is given
  (even empty) every answer carries `decided` (choice: the option; score:
  the level index; noul: the boolean) after its own answer field, and the
  result a `constraints` report {feasible decoder exact violations}; the
  model's own fields never change. :on-infeasible `min_violations`
  (default: the fewest violated constraints, then the best score) or
  `raise` (ex-info {:type :infeasible :violations [...]}). A bad
  constraint is ex-info {:type :invalid-constraint :index i}, thrown
  before any inference. When the state did not fit a question's sequence
  the result carries \"truncated\" {qid tokens-dropped}. A thinker agent
  (lev.think) takes :thinking and :thought as well."
  ([agent state questions] (system-one agent state questions nil))
  ([agent state questions opts] (system-one* agent state questions opts)))

(defmulti system-one-batch*
  "The engine behind system-one-batch, by the agent's :kind; the default
  answers the states one by one."
  (fn [agent _states _questions _opts] (:kind agent :encoder)))

(defmethod system-one-batch* :default [agent states questions opts]
  (mapv #(system-one* agent % questions opts) states))

(defn system-one-batch
  "Several states against the same questions: a vector of what
  system-one answers for each, in order. A thinker in Jev mode decides
  them in one pass (lev.think); anything else answers them one by one."
  ([agent states questions] (system-one-batch agent states questions nil))
  ([agent states questions opts] (system-one-batch* agent (vec states) questions opts)))

(defn- calibration-path
  "Where an agent keeps its temperatures: an encoder's are its
  checkpoint config's; a thinker's (and a Clef model's, lev.clef) sit
  under its config's :calibration, since its :temperature is the
  sampling one."
  [agent]
  (if (#{:thinker :clef} (:kind agent)) [:cfg :calibration] [:cfg]))

(defn calibration-of
  "The agent's {:temperature [per type] :temperature-by-options {bucket T}}."
  [agent]
  (get-in agent (calibration-path agent)))

(defn with-calibration
  "The agent with refitted temperatures (lev.calibrate's {:temperature
  [per type] :temperature-by-options {bucket T}}) over its own, bucket by
  bucket."
  [agent {:keys [temperature temperature-by-options]}]
  (let [path (calibration-path agent)]
    (cond-> agent
      temperature (assoc-in (conj path :temperature) (vec temperature))
      temperature-by-options (update-in (conj path :temperature-by-options) merge temperature-by-options))))

(defn temperature-for
  "The calibration temperature for a question type and option count: the
  (type, count-bucket) entry, else the type's. max(1e-3, T): a degenerate
  fitted temperature saturates the softmax instead of dividing by zero."
  [cfg qtype k]
  (max 1e-3 (double (get (:temperature-by-options cfg)
                         (seq/temp-bucket qtype k)
                         (nth (:temperature cfg) qtype)))))

(defn encoder-forward
  "The encoder on state + questions, before calibration: one map per
  question, in order — {:qid :qdef :q :qtype :k :logits (the k option
  logits) :act (the act head's two) :tokens :dropped (state tokens that
  did not fit)}. What system-one calibrates and lev.calibrate refits on."
  [agent state questions]
  (let [{:keys [cfg tok w prefix-cache]} agent
        ;; the backend's forward over the prepared rows: the C kernels
        ;; (lev.model/forward-batch) unless the agent brought its own
        forward (:forward agent m/forward-batch)
        ;; the state is tokenized once for every question in the call (46
        ;; ms at 1.4k tokens, the same for each question before); a
        ;; question's prefix comes from the agent's cache when it has been
        ;; asked before
        state-ids (seq/encode-state tok state)
        prepared (mapv (fn [[qid qdef]]
                         (let [qdef (validate-question qid qdef)
                               q (to-internal qdef)
                               [pids pmarkers] (seq/cached-prefix prefix-cache tok q (:head-max-len cfg))
                               [ids markers dropped] (seq/assemble pids pmarkers state-ids (:max-len cfg))]
                           (when (not= (count markers) (count (seq/render-options q)))
                             ;; a debias rotation's id is [qid r]: name the question
                             (let [shown (if (vector? qid) (first qid) qid)]
                               (throw (ex-info (format "question %s: options do not fit in head_max_len=%d tokens"
                                                       (pr-str shown) (:head-max-len cfg))
                                               {:type :invalid-question :qid shown :field "criteria"
                                                :head-max-len (:head-max-len cfg)}))))
                           {:qid qid :qdef qdef :q q :ids ids :markers markers :qtype (seq/qtypes (:t q)) :dropped dropped}))
                       questions)
        ;; the questions share every gemm of the forward, in batches of at
        ;; most max-batch rows so the workspace stays a few hundred MB
        outputs (mapcat (fn [chunk]
                          (forward w cfg
                                           (mapv (fn [{:keys [ids markers qtype]}]
                                                   {:ids ids :att (vec (repeat (count ids) 1))
                                                    :markers markers :marker-mask (vec (repeat (count markers) 1))
                                                    :qtype qtype})
                                                 chunk)))
                        (partition-all max-batch prepared))]
    (mapv (fn [{:keys [markers ids] :as pq} [logits act]]
            (let [k (count markers)]
              (-> (dissoc pq :ids :markers)
                  (assoc :k k :logits (vec (take k logits)) :act (vec act) :tokens (count ids)))))
          prepared outputs)))

(defmulti forward
  "Any agent on state + questions, before calibration: one map per
  question, in order, with at least {:qid :qdef :q :qtype :k :logits (the
  k option scores in answer order, a noul's [false true])}. What
  lev.calibrate refits on; an encoder's is encoder-forward, a thinker's
  is in lev.think."
  (fn [agent _state _questions] (:kind agent)))

(defmethod forward :encoder [agent state questions] (encoder-forward agent state questions))

(defn- rotations
  "The choice question `qdef` with its options rotated r places, for r in
  0..k-1 (a rotation keeps every option's description)."
  [qdef]
  (let [crit (qget qdef :criteria)
        pairs (if (map? crit) (vec crit) (mapv (fn [c] [c nil]) crit))
        k (count pairs)]
    (mapv (fn [r] (assoc qdef (if (contains? qdef :criteria) :criteria "criteria")
                         (seq/ordered-map (concat (drop r pairs) (take r pairs)))))
          (range k))))

(defn- rotate-back
  "p over a rotation's option order, put back in the caller's order."
  [p r k]
  (mapv (fn [i] (nth p (mod (- i r) k))) (range k)))

(defmethod system-one* :encoder
  [agent state questions {:keys [constraints on-infeasible debias]}]
  (let [{:keys [cfg]} agent
        ;; the questions and constraints are checked before the forward
        ;; pass, so a bad one costs nothing
        validated (mapv (fn [[qid qdef]] [qid (validate-question qid qdef)]) questions)
        cs (prepare-constraints validated constraints)
        ;; :debias: a choice with 3+ options runs once per rotation of its
        ;; options, all in the same batch, and its probabilities are the
        ;; average (bench/: 14% of answers move under rotation; averaging
        ;; is +2.8 points and a lower ECE on english)
        rotated? (fn [qdef] (and debias (= "choice" (qget qdef :type)) (>= (count (qget qdef :criteria)) 3)))
        expanded (vec (mapcat (fn [[qid qdef]]
                                (if (rotated? qdef)
                                  (map-indexed (fn [r q] [[qid r] q]) (rotations qdef))
                                  [[[qid 0] qdef]]))
                              validated))
        forwards (encoder-forward agent state expanded)
        n-tokens (reduce + (map :tokens forwards))
        by-qid (group-by (fn [f] (first (:qid f))) forwards)
        prepared (mapv (fn [[qid qdef]]
                         (let [fs (get by-qid qid)
                               base (assoc (first fs) :qid qid :qdef qdef)]
                           (assoc base :rotations (count fs)
                                  :ps (mapv (fn [f] (rotate-back (softmax (mapv #(/ (double %) (temperature-for cfg (:qtype f) (:k f))) (:logits f)))
                                                                 (second (:qid f)) (:k f)))
                                            fs))))
                       validated)
        calibrated (mapv (fn [{:keys [k ps act]}]
                           {:k k
                            :p (mapv (fn [i] (/ (reduce + (map #(nth % i) ps)) (count ps))) (range k))
                            :actp (first (softmax act))})
                         prepared)
        solution (when cs
                   (decide-constraints cs (map (fn [{:keys [qid q]} {:keys [p]}] [qid q p]) prepared calibrated)
                                       on-infeasible))
        answers (seq/ordered-map
                 (map (fn [{:keys [qid q]} {:keys [k p actp]}]
                        [qid (typed-answer q p k (when solution (decided-label cs solution qid))
                                           [["action" (array-map "act_probability" (round4 actp))]])])
                      prepared calibrated))
        ;; the state is cut from the end to fit max_len, silently otherwise:
        ;; say so per question, with how much did not fit
        truncated (seq/ordered-map (keep (fn [{:keys [qid dropped]}] (when (pos? dropped) [qid dropped])) prepared))
        debiased (seq/ordered-map (keep (fn [{:keys [qid rotations]}] (when (> rotations 1) [qid rotations])) prepared))]
    (seq/ordered-map
     (concat [["model" (:name agent "encoder")]
              ["answers" answers]
              ["usage" (array-map "input_tokens" n-tokens "output_tokens" 0)]]
             (when (seq truncated) [["truncated" truncated]])
             (when (seq debiased) [["debias" debiased]])
             (when solution [["constraints" (constraints-report cs solution)]])))))
