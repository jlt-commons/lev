(ns bench.authored144
  "Accuracy and latency of a prepared checkpoint on von's authored144
  benchmark (bench/data/authored144.jsonl, 144 three-way decisions:
  evidence interpretation, rule application, candidate selection; from
  github.com/wfzyx/von, Apache-2.0) and its 108 output-blind perturbations.
  Each row is one choice question: the row's question as the instructions,
  its options as the criteria, its state as the state.

    jolt -M bench/authored144.clj [--data DIR] [--model NAME] [--thinking true|false] [--debias]
                                  [--calibration FILE] [--file bench/data/authored144.jsonl] [--out results.jsonl]

  --model is a checkpoint (english, the default; typed-decisions;
  multilingual) or a thinker from config.edn :thinkers / LEV_THINKER
  (`thinker`), with --thinking overriding its default. Prints per-family
  accuracy, balanced accuracy and ms per case; --out writes one line per
  case with the prediction and the probabilities."
  (:require [clojure.edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq]))

(defn- rows [file]
  (mapv json/read-str (remove str/blank? (str/split-lines (slurp file)))))

(defn run
  "[{:id :family :expected :predicted :probabilities :ms :output-tokens} ...]"
  [agent rs opts]
  (mapv (fn [r]
          (let [criteria (seq/ordered-map (map (fn [o] [(get o "id") (get o "description")]) (get r "options")))
                q (array-map "type" "choice" "instructions" (get r "question") "criteria" criteria)
                t0 (System/nanoTime)
                out (ag/system-one agent (get r "state") {"decision" q} opts)
                ms (/ (- (System/nanoTime) t0) 1e6)
                a (get-in out ["answers" "decision"])]
            {:id (get r "id") :family (get r "family")
             :expected (get (nth (get r "options") (get r "label")) "id")
             :predicted (get a "choice") :probabilities (get a "probabilities") :ms ms
             :output-tokens (get-in out ["usage" "output_tokens"] 0)}))
        rs))

(defn ece
  "Expected calibration error of the top probability, ten bins."
  [results]
  (let [scored (keep (fn [{:keys [probabilities expected predicted]}]
                       (when (seq probabilities) [(reduce max (vals probabilities)) (= expected predicted)]))
                     results)
        bins (group-by (fn [[c _]] (min 9 (int (* 10 c)))) scored)]
    (/ (reduce + (for [[_ xs] bins]
                   (* (count xs) (Math/abs (- (/ (count (filter second xs)) (double (count xs)))
                                              (/ (reduce + (map first xs)) (double (count xs))))))))
       (double (max 1 (count scored))))))

(defn confidence
  "1 - normalised entropy, the answer's own confidence for a choice."
  [probabilities]
  (let [p (vals probabilities) k (count p)]
    (if (< k 2) 1.0
        (- 1.0 (/ (- (reduce + (map (fn [x] (* x (Math/log (max x 1e-12)))) p))) (Math/log k))))))

(defn gate-report
  "For each threshold: what a confidence gate keeps on this model, how
  accurate that part is, and how accurate the escalated part would have
  been without escalation."
  [results]
  (println "  gate: kept / accuracy of the kept part / accuracy the escalated part had")
  (doseq [thr [0.3 0.5 0.7 0.8 0.9]]
    (let [conf (fn [r] (if (seq (:probabilities r)) (confidence (:probabilities r)) 1.0))
          kept (filter #(>= (conf %) thr) results)
          esc (remove #(>= (conf %) thr) results)
          acc (fn [rs] (* 100.0 (/ (count (filter #(= (:expected %) (:predicted %)) rs)) (max 1 (count rs)))))]
      (println (format "    >= %.1f: %3d/%d kept @ %5.1f%%   escalated %3d @ %5.1f%%"
                       thr (count kept) (count results) (acc kept) (count esc) (acc esc))))))

(defn report [results]
  (let [acc (fn [rs] (/ (count (filter #(= (:expected %) (:predicted %)) rs)) (double (max 1 (count rs)))))
        by-class (group-by :expected results)
        balanced (/ (reduce + (map (fn [[_ rs]] (acc rs)) by-class)) (double (count by-class)))]
    (doseq [[fam rs] (sort-by key (group-by :family results))]
      (println (format "  %-24s %3d/%3d  %5.1f%%" fam (count (filter #(= (:expected %) (:predicted %)) rs)) (count rs) (* 100 (acc rs)))))
    (println (format "  %-24s %3d/%3d  %5.1f%%   balanced %5.1f%%   %.0f ms/case (mean), %.0f ms (median), %.0f output tokens/case"
                     "all" (count (filter #(= (:expected %) (:predicted %)) results)) (count results)
                     (* 100 (acc results)) (* 100 balanced)
                     (/ (reduce + (map :ms results)) (count results))
                     (nth (sort (map :ms results)) (quot (count results) 2))
                     (/ (double (reduce + (map :output-tokens results))) (count results))))
    (println (format "  ECE %.3f (top probability, 10 bins)" (ece results)))
    (gate-report results)))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        model (get opts "--model" "english")
        data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
        file (get opts "--file" "bench/data/authored144.jsonl")
        rt (router/make-router {:data data :thinkers (cfg/thinkers ctx)
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)})
        agent (router/load-model rt model)
        ;; --calibration covers every encoder through the router; a thinker's here
        agent (if-let [path (and (router/thinker? rt model) (get opts "--calibration"))]
                (ag/with-calibration agent (clojure.edn/read-string (slurp path)))
                agent)
        run-opts (cond-> {}
                   (contains? opts "--thinking") (assoc :thinking (= "true" (get opts "--thinking")))
                   (true? (get opts "--debias")) (assoc :debias true))
        run-opts (when (seq run-opts) run-opts)
        rs (rows file)
        _ (run agent (take 3 rs) run-opts)            ; warm up
        results (run agent rs run-opts)]
    (println (str model (when run-opts (str " (thinking " (:thinking run-opts) ")")) " on " file ":"))
    (report results)
    (when-let [out (get opts "--out")]
      (spit out (str/join "\n" (map #(seq/json-str (into (array-map) %)) results)))
      (println "wrote" out))))

(apply -main *command-line-args*)
