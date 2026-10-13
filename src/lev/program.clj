(ns lev.program
  "A classifier written as data, to be authored once (by a person or a
  large model, from a sample of the inputs) and run deterministically on
  every input:

    {\"labels\"    [label ...]                 optional; rules are checked against it
     \"extract\"   {fact regex}                a fact holds when its regex finds a match
     \"questions\" {qid question}              typed questions (choice / score / noul)
     \"rules\"     [{\"when\" condition \"then\" label} ...]
     \"default\"   label                       when no rule fires
     \"escalate\"  {\"model\" thinker \"threshold\" t}}   optional; needs a router

  A condition is one of
    {\"fact\" f}
    {\"noul\" qid op x}       op one of >= > <= <, on the probability of yes
    {\"score\" qid op x}      on the expected level index
    {\"choice\" qid \"is\" option} / {\"choice\" qid \"in\" [option ...]}
    {\"all\" [c ...]} / {\"any\" [c ...]} / {\"not\" c}

  `run` matches the facts first and evaluates the rules in order with the
  answers unknown: when the first rule that is not false is decided by the
  facts, no model is called. Otherwise the agent answers, in one call,
  the questions the rules from there on read, and the first true rule
  names the label. With \"escalate\" the answers below the threshold are
  re-asked on the thinker (lev.patterns/escalate)."
  (:require [lev.agent :as ag]
            [lev.api :as api]
            [lev.patterns :as pat]
            [lev.sequence :as seq]))

(def ^:private ops {">=" >= ">" > "<=" <= "<" <})

(defn- bad [field msg]
  (throw (ex-info (str field ": " msg) {:type :invalid-request :field field})))

(defn- qtype [program qid] (get-in program ["questions" qid "type"]))

(defn- options [program qid]
  (set (map (fn [k] (if (keyword? k) (name k) (str k)))
            (keys (:crit (ag/to-internal (get-in program ["questions" qid])))))))

(defn- check-condition [program field c]
  (let [fail #(bad field (str % ": " (pr-str c)))
        compare-ok (fn [t]
                     (let [qid (get c t)
                           op (first (filter #(contains? c %) (keys ops)))]
                       (when-not (= t (qtype program qid)) (fail (str "no " t " question " (pr-str qid))))
                       (when-not (and op (number? (get c op))) (fail "needs one of >= > <= < with a number"))))]
    (when-not (map? c) (fail "a condition is an object"))
    (cond
      (contains? c "fact") (when-not (contains? (get program "extract") (get c "fact"))
                             (fail (str "no extracted fact " (pr-str (get c "fact")))))
      (contains? c "noul") (compare-ok "noul")
      (contains? c "score") (compare-ok "score")
      (contains? c "choice") (let [qid (get c "choice")]
                               (when-not (= "choice" (qtype program qid)) (fail (str "no choice question " (pr-str qid))))
                               (let [os (cond (contains? c "is") [(get c "is")]
                                              (sequential? (get c "in")) (get c "in")
                                              :else (fail "a choice condition needs \"is\" or \"in\""))]
                                 (doseq [o os]
                                   (when-not (contains? (options program qid) o) (fail (str "unknown option " (pr-str o)))))))
      (contains? c "all") (if (sequential? (get c "all")) (doseq [x (get c "all")] (check-condition program field x)) (fail "\"all\" takes a list"))
      (contains? c "any") (if (sequential? (get c "any")) (doseq [x (get c "any")] (check-condition program field x)) (fail "\"any\" takes a list"))
      (contains? c "not") (check-condition program field (get c "not"))
      :else (fail "unknown condition"))))

(defn validate
  "The program, or {:type :invalid-request :field path} naming the first
  thing wrong with it: a regex that does not compile, a question lev
  would refuse, a rule reading a fact or question that is not there (or
  a question of another type, or an option it does not have), a label
  outside \"labels\", a missing \"default\"."
  [program]
  (when-not (map? program) (bad "program" "must be an object"))
  (doseq [[f re] (get program "extract")]
    (try (re-pattern re) (catch Exception e (bad (str "extract." f) (str "bad regex: " (ex-message e))))))
  (doseq [[qid qdef] (get program "questions")]
    (try (ag/validate-question qid qdef)
         (catch Exception e (bad (str "questions." qid) (ex-message e)))))
  (let [labels (get program "labels")
        label-ok (fn [field l]
                   (when-not (string? l) (bad field "a label is a string"))
                   (when (and labels (not (some #{l} labels))) (bad field (str (pr-str l) " is not in labels"))))]
    (when-not (sequential? (get program "rules" [])) (bad "rules" "must be a list"))
    (doseq [[i r] (map-indexed vector (get program "rules"))]
      (check-condition program (str "rules[" i "].when") (get r "when"))
      (label-ok (str "rules[" i "].then") (get r "then")))
    (when-not (contains? program "default") (bad "default" "is required"))
    (label-ok "default" (get program "default")))
  (when-let [e (get program "escalate")]
    (when-not (string? (get e "model")) (bad "escalate.model" "names the thinker")))
  program)

;; --- evaluation: true, false or ::unknown (an answer not asked yet) ----------

(defn- holds [c facts answers]
  (let [answer (fn [qid] (get answers qid))]
    (cond
      (contains? c "fact") (boolean (get facts (get c "fact")))
      (contains? c "all") (let [vs (map #(holds % facts answers) (get c "all"))]
                            (cond (some false? vs) false (some #{::unknown} vs) ::unknown :else true))
      (contains? c "any") (let [vs (map #(holds % facts answers) (get c "any"))]
                            (cond (some true? vs) true (some #{::unknown} vs) ::unknown :else false))
      (contains? c "not") (let [v (holds (get c "not") facts answers)] (if (= ::unknown v) v (not v)))
      :else
      (let [t (first (filter #(contains? c %) ["noul" "score" "choice"]))]
        (if-let [a (answer (get c t))]
          (if (= t "choice")
            (contains? (set (if (contains? c "is") [(get c "is")] (get c "in"))) (get a "choice"))
            (let [op (first (filter #(contains? c %) (keys ops)))]
              ((ops op) (double (get a t)) (double (get c op)))))
          ::unknown)))))

(defn- first-open
  "[index value] of the first rule that is not false, or nil."
  [rules facts answers]
  (first (keep-indexed (fn [i r] (let [v (holds (get r "when") facts answers)] (when-not (false? v) [i v]))) rules)))

(defn- reads [c]
  (cond
    (contains? c "all") (mapcat reads (get c "all"))
    (contains? c "any") (mapcat reads (get c "any"))
    (contains? c "not") (reads (get c "not"))
    :else (keep #(get c %) ["noul" "score" "choice"])))

(defn facts
  "{fact matched?} for the input: a string as it is, anything else as JSON."
  [program input]
  (let [text (if (string? input) input (seq/json-str input))]
    (into {} (map (fn [[f re]] [f (some? (re-find (re-pattern re) text))])) (get program "extract"))))

(defn label
  "[label rule] for the facts and answers (nil answers: none asked): the
  first rule that holds, or [default nil]. Pure; what `run` decides
  with, and what `fit` replays."
  [program facts answers]
  (let [rules (get program "rules" [])]
    (or (first (keep-indexed (fn [i r] (when (true? (holds (get r "when") facts answers)) [(get r "then") i])) rules))
        [(get program "default") nil])))

(defn run
  "The program's label for one input on an agent or router: an ordered
  map {label rule facts}, rule the index of the rule that fired (nil:
  the default); when the model was asked, its answers, model and usage
  follow (and the escalation report when there was one). Validates the
  program first. opts go to lev.api/system-one (:model :lang on a
  router; with escalation, :model is the fast one)."
  ([agent input program] (run agent input program nil))
  ([agent input program opts]
  (let [program (validate program)
        rules (get program "rules" [])
        fs (facts program input)
        [i v] (first-open rules fs nil)
        done (fn [answers tail]
               (let [[l r] (label program fs answers)]
                 (seq/ordered-map (concat [["label" l] ["rule" r] ["facts" fs]] tail))))]
    (if (not= ::unknown v)
      (done nil nil)
      (let [needed (set (mapcat #(reads (get % "when")) (subvec (vec rules) i)))
            questions (seq/ordered-map (filter (fn [[qid _]] (contains? needed qid)) (get program "questions")))
            esc (get program "escalate")
            resp (if esc
                   (pat/escalate agent input questions {:model (get esc "model") :threshold (get esc "threshold")
                                                        :fast-model (:model opts) :lang (:lang opts)})
                   (api/system-one agent input questions opts))
            answers (get resp "answers")]
        (done answers
              (concat [["answers" answers] ["model" (get resp "model")] ["usage" (get resp "usage")]]
                      (when esc [["escalation" (get resp "escalation")]]))))))))

(defn observe
  "{facts answers} for one input with every question of the program
  answered (no escalation): what `fit` replays."
  [agent input program]
  (let [program (validate program)]
    (seq/ordered-map [["facts" (facts program input)]
                      ["answers" (when (seq (get program "questions"))
                                   (get (api/system-one agent input (get program "questions")) "answers"))]])))

;; --- fitting the cutoffs on labelled cases ------------------------------------

(defn- cutoffs
  "[path-to-the-number candidates] for every noul / score comparison in
  the rules: a noul's cutoffs 0.01..0.99, a score's 0..(levels - 1) by 0.05."
  [program]
  (letfn [(walk [path c]
            (cond
              (contains? c "all") (mapcat #(walk (conj path "all" %1) %2) (range) (get c "all"))
              (contains? c "any") (mapcat #(walk (conj path "any" %1) %2) (range) (get c "any"))
              (contains? c "not") (walk (conj path "not") (get c "not"))
              (or (contains? c "noul") (contains? c "score"))
              (let [op (first (filter #(contains? c %) (keys ops)))
                    cands (if (contains? c "noul")
                            (map #(ag/round4 (/ % 100.0)) (range 1 100))
                            (let [top (dec (count (get-in program ["questions" (get c "score") "criteria"])))]
                              (map #(ag/round4 (* % 0.05)) (range 0 (inc (* 20 top))))))]
                [[(conj path op) cands]])
              :else nil))]
    (vec (mapcat (fn [i r] (walk ["rules" i "when"] (get r "when"))) (range) (get program "rules")))))

(defn- accuracy [program cases]
  (/ (count (filter (fn [c] (= (get c "label") (first (label program (get c "facts") (get c "answers"))))) cases))
     (double (max 1 (count cases)))))

(defn fit
  "The program's numeric cutoffs fitted to labelled cases ({facts answers
  label}, as `observe` answers plus the gold label): coordinate ascent,
  three passes over every noul / score comparison, each moved to the
  median of the cutoffs that label the most cases right (kept where it
  already is one of them). Deterministic; no model is asked. Answers
  {program accuracy before}."
  [program cases]
  (let [program (validate program)
        cases (vec cases)
        points (cutoffs program)
        step (fn [p [path cands]]
               (let [scored (mapv (fn [x] [x (accuracy (assoc-in p path x) cases)]) cands)
                     top (reduce max (map second scored))
                     best (mapv first (filter #(= top (second %)) scored))]
                 (if (some #{(get-in p path)} best)
                   p
                   (assoc-in p path (nth best (quot (dec (count best)) 2))))))
        fitted (reduce (fn [p _] (reduce step p points)) program (range 3))]
    (seq/ordered-map [["program" fitted]
                      ["accuracy" (ag/round4 (accuracy fitted cases))]
                      ["before" (ag/round4 (accuracy program cases))]])))
