(ns bench.typed-decisions
  "A model on the typed-decisions test split (LocalLLaMA/typed-decisions,
  Apache-2.0): 400 states from four workflows, five typed questions each,
  2,000 decisions; gold is the mean of three teacher samples, so accuracy
  is agreement with the teacher (its own self-agreement is 0.735, the prior
  0.470). The set Jev (0.727 on the dataset's card, 0.738 in Winnow's
  report), ollaya's models and the laya checkpoints have published numbers
  on. Build the data with `uv run --with pyarrow python
  bench/typed_decisions_data.py`.

    jolt -M bench/typed_decisions.clj [--model NAME] [--thinking true|false] [--calibration FILE]
                                      [--limit N] [--file bench/data/typed-decisions-test.jsonl] [--out results.jsonl]

  --model is a checkpoint (english, the default; typed-decisions, which
  was trained on this dataset's train split; multilingual) or a thinker.
  --calibration applies a lev.calibrate file to the model, a thinker's
  included (jolt -M:calibrate --model NAME --labels
  bench/data/typed-decisions-calib.jsonl fits one on the train split).
  Prints accuracy (argmax against the gold's label) overall, per type and
  per workflow, KL(gold || model), Brier and ECE (top probability, 15
  bins, ollaya's), how often a noul is answered true against how often the
  gold says so (small instruct models lean yes), and ms per state; --out
  writes one line per question."
  (:require [clojure.edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq]))

(defn- rows [file]
  (mapv json/read-str (remove str/blank? (str/split-lines (slurp file)))))

(defn- distribution
  "The answer as {option p}, in the gold's option spelling."
  [a]
  (if (= "noul" (get a "type"))
    (let [p (double (get a "noul"))] (array-map "false" (- 1.0 p) "true" p))
    (get a "probabilities")))

(defn- argmax-key [m] (key (apply max-key val (seq m))))

(defn run
  "One result per question: {:id :workflow :qid :type :gold-label :predicted
  :p (the model's {option p}) :gold (the gold's) :ms (the state's call)}."
  [agent rs opts]
  (vec (mapcat (fn [r]
                 (let [t0 (System/nanoTime)
                       out (ag/system-one agent (get r "state") (get r "questions") opts)
                       ms (/ (- (System/nanoTime) t0) 1e6)]
                   (for [[qid q] (get r "questions")
                         :let [a (get-in out ["answers" qid])
                               g (get-in r ["gold" qid])
                               p (distribution a)]]
                     {:id (get r "id") :workflow (get r "workflow") :qid qid :type (get q "type")
                      :gold-label (str (get g "label")) :predicted (argmax-key p)
                      :p p :gold (get g "probabilities") :ms ms})))
               rs)))

(defn- mean [xs] (if (seq xs) (/ (reduce + xs) (double (count xs))) 0.0))

(defn- correct? [r] (= (:gold-label r) (:predicted r)))

(defn- accuracy [rs] (mean (map #(if (correct? %) 1.0 0.0) rs)))

(defn- kl
  "KL(gold || model) over the gold's options, the model's floored at 1e-6
  (answers are rounded to four decimals)."
  [{:keys [p gold]}]
  (reduce + (for [[k g] gold :let [g (double g)] :when (pos? g)]
              (* g (Math/log (/ g (max 1e-6 (double (get p k 0.0)))))))))

(defn- brier [{:keys [p gold]}]
  (reduce + (for [k (distinct (concat (keys gold) (keys p)))]
              (let [d (- (double (get p k 0.0)) (double (get gold k 0.0)))] (* d d)))))

(defn ece
  "Expected calibration error of the top probability, 15 bins."
  [rs]
  (let [scored (map (fn [r] [(reduce max (map double (vals (:p r)))) (correct? r)]) rs)
        bins (group-by (fn [[c _]] (min 14 (int (* 15 c)))) scored)]
    (/ (reduce + (for [[_ xs] bins]
                   (* (count xs) (Math/abs (- (/ (count (filter second xs)) (double (count xs)))
                                              (mean (map first xs)))))))
       (double (max 1 (count scored))))))

(defn report [results]
  (let [line (fn [label rs]
               (println (format "  %-28s %4d  acc %.3f  KL %.3f  Brier %.3f  ECE %.3f"
                                label (count rs) (accuracy rs) (mean (map kl rs)) (mean (map brier rs)) (ece rs))))
        by-state (vals (group-by :id results))]
    (line "all" results)
    (doseq [[t rs] (sort-by key (group-by :type results))] (line t rs))
    (doseq [[w rs] (sort-by key (group-by :workflow results))] (line w rs))
    (let [nouls (filter #(= "noul" (:type %)) results)]
      (when (seq nouls)
        (println (format "  noul answered true %.1f%%, gold true %.1f%%"
                         (* 100 (mean (map #(if (= "true" (:predicted %)) 1.0 0.0) nouls)))
                         (* 100 (mean (map #(if (= "true" (:gold-label %)) 1.0 0.0) nouls)))))))
    (let [ms (sort (map (comp :ms first) by-state))]
      (println (format "  %.0f ms per state (mean), %.0f (median), %d states"
                       (mean ms) (nth ms (quot (count ms) 2)) (count ms))))))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        model (get opts "--model" "english")
        data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
        file (get opts "--file" "bench/data/typed-decisions-test.jsonl")
        rt (router/make-router {:data data :models (cfg/encoders ctx) :thinkers (cfg/thinkers ctx)
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)})
        agent (router/load-model rt model)
        ;; --calibration covers every encoder through the router; a thinker's here
        agent (if-let [path (and (router/thinker? rt model) (get opts "--calibration"))]
                (ag/with-calibration agent (clojure.edn/read-string (slurp path)))
                agent)
        run-opts (when (contains? opts "--thinking") {:thinking (= "true" (get opts "--thinking"))})
        rs (cond->> (rows file)
             (get opts "--limit") (take (Long/parseLong (get opts "--limit"))))
        _ (run agent (take 2 rs) run-opts)            ; warm up
        results (run agent rs run-opts)]
    (println (str model (when run-opts (str " (thinking " (:thinking run-opts) ")"))
                  (when-let [c (get opts "--calibration")] (str ", calibration " c)) " on " file ":"))
    (report results)
    (when-let [out (get opts "--out")]
      (spit out (str/join "\n" (map seq/json-str results)))
      (println "wrote" out))))

(apply -main *command-line-args*)
