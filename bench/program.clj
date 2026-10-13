(ns bench.program
  "Scores a lev.program on a labelled split (bench/program_data.py builds
  them): accuracy, macro F1 over the labels, how many inputs needed the
  model, how many answers escalated, and the time per input. The
  baselines are programs too: the target asked directly as one question
  and mapped to the label by one rule per answer.

    jolt -M bench/program.clj --program bench/programs/toxic-direct.json --split toxic-test
         [--model NAME] [--fit SPLIT [--save FILE]] [--limit N] [--out FILE] [--quiet]

  --fit answers every question of the program on another split (the
  train half) and fits its cutoffs there (lev.program/fit) before
  scoring; --save writes the fitted program. --out writes one line per
  input, {id label predicted rule answers}."
  (:require [lev.json :as json]
            [clojure.string :as str]
            [lev.config :as cfg]
            [lev.program :as prog]
            [lev.router :as router]
            [lev.sequence :as seq]))

(defn- read-jsonl [path]
  (map json/read-str (remove str/blank? (str/split-lines (slurp path)))))

(defn- f1s [pairs labels]
  (into (sorted-map)
        (map (fn [l]
               (let [tp (count (filter (fn [[g p]] (and (= g l) (= p l))) pairs))
                     fp (count (filter (fn [[g p]] (and (not= g l) (= p l))) pairs))
                     fn* (count (filter (fn [[g p]] (and (= g l) (not= p l))) pairs))
                     prec (if (pos? (+ tp fp)) (/ tp (double (+ tp fp))) 0.0)
                     rec (if (pos? (+ tp fn*)) (/ tp (double (+ tp fn*))) 0.0)]
                 [l {:f1 (if (pos? (+ prec rec)) (/ (* 2 prec rec) (+ prec rec)) 0.0) :n (+ tp fn*)}])))
        labels))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        program (prog/validate (json/read-str (slurp (get opts "--program"))))
        cases (cond->> (read-jsonl (str "bench/data/" (get opts "--split") ".jsonl"))
                (get opts "--limit") (take (Long/parseLong (get opts "--limit"))))
        model (get opts "--model")
        rt (router/make-router {:data (cfg/setting ctx "--data" "LEV_DATA" :data "data")
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)
                                :thinkers (cfg/thinkers ctx)})
        agent (if model (router/load-model rt model) rt)
        agent (if (get program "escalate") rt agent)
        program (if-let [fs (get opts "--fit")]
                  (let [fit-agent (if model agent (router/load-model rt "english"))
                        observed (mapv (fn [c] (assoc (prog/observe fit-agent (get c "text") program) "label" (get c "label")))
                                       (read-jsonl (str "bench/data/" fs ".jsonl")))
                        r (prog/fit program observed)]
                    (println (format "fitted on %s (%d): accuracy %.3f -> %.3f" fs (count observed) (get r "before") (get r "accuracy")))
                    (when-let [o (get opts "--save")] (spit o (str (seq/json-str (get r "program")) "\n")))
                    (get r "program"))
                  program)
        _ (prog/run agent (get (first cases) "text") program) ; warm
        rows (vec (for [c cases]
                    (let [t0 (System/nanoTime)
                          r (prog/run agent (get c "text") program)
                          ms (/ (- (System/nanoTime) t0) 1e6)]
                      (when (and (not (contains? opts "--quiet")) (not= (get r "label") (get c "label")))
                        (println (format "  miss %s: %s -> %s  %s" (get c "id") (get c "label") (get r "label")
                                         (subs (get c "text") 0 (min 90 (count (get c "text")))))))
                      {:id (get c "id") :answers (get r "answers") :rule (get r "rule")
                       :gold (get c "label") :pred (get r "label") :ms ms
                       :asked (some? (get r "answers"))
                       :escalated (count (get-in r ["escalation" "escalated"]))})))
        _ (when-let [o (get opts "--out")]
            (spit o (apply str (map (fn [{:keys [id gold pred rule answers]}]
                                      (str (seq/json-str (seq/ordered-map [["id" id] ["label" gold] ["predicted" pred]
                                                                           ["rule" rule] ["answers" answers]]))
                                           "\n"))
                                    rows))))
        pairs (map (juxt :gold :pred) rows)
        labels (distinct (concat (map :gold rows) (map :pred rows)))
        per (f1s pairs labels)
        n (count rows)]
    (println (format "%s on %s (%d inputs), model %s"
                     (get opts "--program") (get opts "--split") n
                     (or model (get-in program ["escalate" "model"]) "routed")))
    (println (format "  accuracy %.3f   macro F1 %.3f   model asked %d/%d   answers escalated %d   %.1f ms/input (p50 %.1f)"
                     (/ (count (filter (fn [[g p]] (= g p)) pairs)) (double n))
                     (/ (reduce + (map :f1 (vals per))) (double (count per)))
                     (count (filter :asked rows)) n
                     (reduce + (map :escalated rows))
                     (/ (reduce + (map :ms rows)) n)
                     (nth (vec (sort (map :ms rows))) (quot n 2))))
    (doseq [[l {:keys [f1 n]}] per]
      (println (format "    %-10s F1 %.3f  (n %d)" l f1 n)))))

(apply -main *command-line-args*)
