(ns lev.calibrate
  "Refit a model's temperatures on labeled cases: an encoder's, or a
  thinker's (whose raw option scores are far more overconfident: ECE
  about 0.4 at T 1 for small instruct models on typed-decisions, ollaya's
  llm-logits measurements).

  The checkpoints ship one temperature per (question type, option-count
  bucket), fitted on their training distribution; on other traffic the
  probabilities can be far off (bench/: the english encoder's yes/no
  answers on BoolQ average 0.92 confidence at 72.5% accuracy, so the
  confidence gate never escalates them). Temperature scaling does not
  change any argmax; it makes `probabilities` and `confidence` mean what
  they say, which is what a gate needs.

  Labeled cases are maps {\"state\" .. \"questions\" {qid qdef} \"labels\"
  {qid label}} (JSON or EDN), a label being a choice's option, a score's
  level index or a noul's boolean; \"targets\" {qid [p ...]} (a gold
  distribution in answer order, typed-decisions' teacher mean) makes the
  fit minimise the cross-entropy against it instead of the label's NLL. `collect` runs the encoder and keeps
  each question's raw logits with the label's index; `calibration` fits
  one temperature per bucket by minimising the negative log likelihood
  (a 1-D golden-section search over log T) and answers the map
  lev.agent/with-calibration takes: {:temperature-by-options {bucket T}
  :temperature [per type]}; `evaluate` reports NLL, ECE and accuracy per
  bucket before and after.

    jolt -M:calibrate --labels cases.jsonl --out calibration.edn [--data DIR] [--model NAME] [--all]

  and then config.edn :calibration {\"english\" \"calibration.edn\"} (or
  --calibration PATH) loads it over the checkpoint's own; a thinker's
  (--model qwen3.5-4b) goes under its name the same way, or inline as its
  :calibration."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq])
  (:gen-class))

(defn softmax [xs]
  (let [mx (reduce max xs)
        ex (mapv #(Math/exp (- (double %) mx)) xs)
        s (reduce + ex)]
    (mapv #(/ % s) ex)))

(defn- key-str [k] (if (keyword? k) (name k) (str k)))

(defn- index-of [v x] (first (keep-indexed (fn [i y] (when (= x y) i)) v)))

(defn- qget [m k] (if (contains? m k) (get m k) (get m (name k))))

(defn- label-index
  "The label's index in answer order: a choice option among its criteria
  keys, a score level index, a noul boolean as [false true]."
  [qid q label]
  (let [bad (fn [msg] (throw (ex-info (str "label for " (pr-str qid) ": " msg) {:type :invalid-label :qid qid :label label})))]
    (case (:t q)
      "choice" (let [ks (mapv key-str (keys (:crit q)))]
                 (or (index-of ks (key-str label))
                     (bad (str (pr-str label) " is not one of " (pr-str ks)))))
      "score" (if (and (integer? label) (< -1 label (count (:crit q))))
                label
                (bad (str (pr-str label) " is not a level index below " (count (:crit q)))))
      (cond (true? label) 1
            (false? label) 0
            (= "true" label) 1
            (= "false" label) 0
            :else (bad (str (pr-str label) " is not true or false"))))))

(defn collect
  "Run the model (lev.agent/forward: an encoder or a thinker) over the
  cases; one item per labeled question: {:bucket :type :qtype :k :logits
  :label}, plus :target with the case's gold distribution for it.
  Questions without a label are skipped."
  [agent cases]
  (vec (mapcat (fn [c]
                 (let [labels (qget c :labels)
                       targets (qget c :targets)
                       questions (qget c :questions)
                       asked (into (seq/ordered-map [])
                                   (filter (fn [[qid _]] (or (contains? labels qid) (contains? labels (key-str qid)))) questions))]
                   (when (seq asked)
                     (map (fn [{:keys [qid q qtype k logits]}]
                            (let [label (if (contains? labels qid) (get labels qid) (get labels (key-str qid)))
                                  target (or (get targets qid) (get targets (key-str qid)))]
                              (when (and target (not= k (count target)))
                                (throw (ex-info (str "target for " (pr-str qid) ": " (count target) " probabilities for " k " options")
                                                {:type :invalid-label :qid qid})))
                              (cond-> {:bucket (seq/temp-bucket qtype k) :type (:t q) :qtype qtype :k k
                                       :logits logits :label (label-index qid q label)}
                                target (assoc :target (mapv double target)))))
                          (ag/forward agent (qget c :state) asked)))))
               cases)))

(defn nll
  "Mean negative log likelihood of the labels under softmax(logits / t);
  an item with a :target distribution counts its cross-entropy instead."
  [items t]
  (/ (reduce + (map (fn [{:keys [logits label target]}]
                      (let [p (softmax (mapv #(/ (double %) t) logits))]
                        (if target
                          (let [z (reduce + target)]
                            (- (reduce + (map (fn [q pi] (* (/ q z) (Math/log (max 1e-12 pi)))) target p))))
                          (- (Math/log (max 1e-12 (nth p label)))))))
                    items))
     (double (max 1 (count items)))))

(defn ece
  "Expected calibration error on the top probability, ten bins."
  [items t]
  (let [scored (map (fn [{:keys [logits label]}]
                      (let [p (softmax (mapv #(/ (double %) t) logits))
                            top (reduce max p)]
                        [top (= label (index-of p top))]))
                    items)
        bins (group-by (fn [[c _]] (min 9 (int (* 10 c)))) scored)]
    (/ (reduce + (for [[_ xs] bins]
                   (* (count xs) (Math/abs (- (/ (count (filter second xs)) (double (count xs)))
                                              (/ (reduce + (map first xs)) (double (count xs))))))))
       (double (max 1 (count scored))))))

(defn accuracy [items]
  (/ (count (filter (fn [{:keys [logits label]}] (= label (index-of logits (reduce max logits)))) items))
     (double (max 1 (count items)))))

(defn fit-temperature
  "The temperature minimising the NLL of `items`: golden-section search
  over log T in [ln 0.05, ln 100] (the NLL is unimodal in T; a thinker's
  noul scores can want 40, ollaya's llm-logits fits)."
  [items]
  (let [f (fn [u] (nll items (Math/exp u)))
        phi (/ (- (Math/sqrt 5.0) 1.0) 2.0)]
    (loop [a (Math/log 0.05) b (Math/log 100.0) i 0]
      (if (or (= i 60) (< (- b a) 1e-4))
        (Math/exp (/ (+ a b) 2.0))
        (let [c (- b (* phi (- b a)))
              d (+ a (* phi (- b a)))]
          (if (< (f c) (f d))
            (recur a d (inc i))
            (recur c b (inc i))))))))

(defn calibration
  "One temperature per bucket present in `items`, and per type (its
  buckets' items together; 1.0 for a type with none): what
  lev.agent/with-calibration applies."
  [items]
  (let [by-bucket (group-by :bucket items)
        by-type (group-by :qtype items)]
    {:temperature-by-options (into (sorted-map) (map (fn [[b xs]] [b (fit-temperature xs)])) by-bucket)
     :temperature (mapv (fn [qtype] (if-let [xs (get by-type qtype)] (fit-temperature xs) 1.0)) [0 1 2])
     :counts (into (sorted-map) (map (fn [[b xs]] [b (count xs)])) by-bucket)}))

(defn evaluate
  "Per bucket: n, accuracy, and NLL / ECE under the agent's own
  temperature and under the calibration's."
  [agent items cal]
  (into (sorted-map)
        (map (fn [[b xs]]
               (let [{:keys [qtype k]} (first xs)
                     before (ag/temperature-for (ag/calibration-of agent) qtype k)
                     after (get-in cal [:temperature-by-options b] before)]
                 [b {:n (count xs) :accuracy (accuracy xs)
                     :before {:t before :nll (nll xs before) :ece (ece xs before)}
                     :after {:t after :nll (nll xs after) :ece (ece xs after)}}]))
             (group-by :bucket items))))

(defn read-cases
  "Labeled cases from a JSON-lines or EDN file."
  [path]
  (let [text (slurp path)]
    (if (str/ends-with? path ".edn")
      (edn/read-string text)
      (mapv json/read-str (remove str/blank? (str/split-lines text))))))

(defn load-calibration
  "A calibration file (EDN with :temperature-by-options / :temperature)."
  [path]
  (let [m (edn/read-string (slurp path))]
    (when-not (and (map? m) (or (:temperature-by-options m) (:temperature m)))
      (throw (ex-info (str path " is not a calibration file") {:type :invalid-request :file path})))
    m))

(defn -main
  "jolt -M:calibrate --labels cases.jsonl --out calibration.edn [--data DIR] [--model NAME] [--all]
   Fits on the even-numbered cases and reports on the odd ones held out
   (--all fits on every case), prints the per-bucket report, writes the
   calibration."
  [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        labels (or (get opts "--labels") (do (println "--labels FILE is required") (System/exit 2)))
        out (get opts "--out" "calibration.edn")
        all? (true? (get opts "--all"))
        model (get opts "--model" "english")
        rt (router/make-router {:data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
                                :models (cfg/encoders ctx)
                                :thinkers (cfg/thinkers ctx)
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)})
        agent (router/load-model rt model)
        cases (vec (read-cases labels))
        [fit-cases held] (if all?
                           [cases []]
                           [(vec (keep-indexed (fn [i c] (when (even? i) c)) cases))
                            (vec (keep-indexed (fn [i c] (when (odd? i) c)) cases))])
        _ (println (format "%s: %d cases to fit on%s" model (count fit-cases)
                           (if (seq held) (format ", %d held out" (count held)) "")))
        ;; each case is run once: a thinker takes seconds a state
        fit-items (collect agent fit-cases)
        fitted (calibration fit-items)
        report (fn [title items]
                 (println title)
                 (println (format "  %-12s %5s %6s   %-22s %-22s" "bucket" "n" "acc" "before: T / NLL / ECE" "after: T / NLL / ECE"))
                 (doseq [[b {:keys [n accuracy before after]}] (evaluate agent items fitted)]
                   (println (format "  %-12s %5d %5.1f%%   %5.2f / %5.3f / %5.3f   %5.2f / %5.3f / %5.3f"
                                    b n (* 100 accuracy) (:t before) (:nll before) (:ece before) (:t after) (:nll after) (:ece after)))))]
    (report "fitted cases:" fit-items)
    (when (seq held) (report "held-out cases:" (collect agent held)))
    (spit out (pr-str (dissoc fitted :counts)))
    (println "wrote" out)))
