(ns lev.calibrate-test
  "lev.calibrate: refitting the encoder's temperatures on labeled cases.
  The fit is checked on synthetic logits with a known temperature; the
  plumbing (collecting logits, applying a calibration file) on the real
  english encoder with a handful of cases."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.calibrate :as cal]
            [lev.think :as think]
            [lev.test-util :as tu]))

(defn- lcg [seed]
  (rest (iterate (fn [x] (mod (+ (* x 1103515245) 12345) 2147483648)) seed)))

(defn- synthetic
  "n items whose labels are drawn from softmax(logits / true-t): the NLL
  minimiser over the temperature should land near true-t."
  [seed n k true-t]
  (let [rs (atom (lcg seed))
        next! (fn [] (let [x (first @rs)] (swap! rs rest) (/ (double x) 2147483648.0)))]
    (vec (for [_ (range n)]
           (let [logits (vec (repeatedly k #(* 6.0 (- (next!) 0.5))))
                 p (cal/softmax (mapv #(/ % true-t) logits))
                 u (next!)
                 label (loop [i 0 acc 0.0]
                         (let [acc (+ acc (nth p i))]
                           (if (or (< u acc) (= i (dec k))) i (recur (inc i) acc))))]
             {:bucket (str "choice:" k) :qtype 0 :logits logits :label label})))))

(deftest the-fit-recovers-a-known-temperature
  (doseq [[t k] [[2.0 3] [0.5 4] [1.5 2]]]
    (let [items (synthetic 7 4000 k t)
          fitted (cal/fit-temperature items)]
      (is (< (Math/abs (- (Math/log fitted) (Math/log t))) 0.1)
          (str "true " t ", fitted " fitted " on " k " options"))))
  (testing "the fit lowers NLL and ECE against the wrong temperature"
    (let [items (synthetic 11 2000 3 2.0)]
      (is (< (cal/nll items (cal/fit-temperature items)) (cal/nll items 1.0)))
      (is (< (cal/ece items (cal/fit-temperature items)) (cal/ece items 1.0))))))

(deftest a-calibration-groups-by-bucket-and-falls-back-per-type
  (let [items (concat (synthetic 1 1500 3 2.0) (synthetic 2 1500 2 0.7)
                      (map #(assoc % :bucket "noul:2" :qtype 2) (synthetic 3 1500 2 1.3)))
        c (cal/calibration items)]
    (is (= #{"choice:3" "choice:2" "noul:2"} (set (keys (:temperature-by-options c)))))
    (is (< 1.7 (get-in c [:temperature-by-options "choice:3"]) 2.3))
    (is (< 0.55 (get-in c [:temperature-by-options "choice:2"]) 0.85))
    (is (< 1.1 (get-in c [:temperature-by-options "noul:2"]) 1.5))
    (is (= 3 (count (:temperature c))) "one per type: choice, score, noul")
    (is (= 1.0 (nth (:temperature c) 1)) "a type with no cases keeps 1.0")
    (is (< 1.1 (nth (:temperature c) 2) 1.5) "noul's is its bucket's")))

(deftest logits-come-off-the-encoder-and-a-calibration-applies-to-it
  (let [agent @tu/agent
        cases [{"state" "Refund me before Friday or we cancel."
                "questions" {"churn" {"type" "noul" "instructions" "Does the customer threaten to leave?"}
                             "team" {"type" "choice" "instructions" "Which team?" "criteria" {"billing" "money" "other" nil}}
                             "anger" {"type" "score" "instructions" "How angry?" "criteria" ["calm" "annoyed" "furious"]}}
                "labels" {"churn" true "team" "billing" "anger" 2}}]
        items (cal/collect agent cases)]
    (testing "one item per question, with the bucket the agent uses, the raw logits and the label's index in answer order"
      (is (= ["noul:2" "choice:2" "score:3-5"] (map :bucket items)))
      (is (= [1 0 2] (map :label items)) "a noul's label true is index 1 ([false true])")
      (is (= [2 2 3] (map #(count (:logits %)) items))))
    (testing "an applied calibration changes the probabilities, not the argmax"
      (let [before (ag/system-one agent (get (first cases) "state") (get (first cases) "questions"))
            hot (ag/with-calibration agent {:temperature-by-options {"noul:2" 0.2}})
            after (ag/system-one hot (get (first cases) "state") (get (first cases) "questions"))]
        (is (not= (get-in before ["answers" "churn" "noul"]) (get-in after ["answers" "churn" "noul"])))
        (is (= (> 0.5 (get-in before ["answers" "churn" "noul"])) (> 0.5 (get-in after ["answers" "churn" "noul"]))))
        (is (= (get-in before ["answers" "team"]) (get-in after ["answers" "team"])) "other buckets untouched")))
    (testing "bad labels are refused with the question named"
      (is (thrown-with-msg? Exception #"team" (cal/collect agent [(assoc-in (first cases) ["labels" "team"] "legal")])))
      (is (thrown-with-msg? Exception #"anger" (cal/collect agent [(assoc-in (first cases) ["labels" "anger"] 7)]))))))

(deftest a-thinker-is-refit-like-an-encoder
  (let [t (think/thinker {:name "fake" :thinking false :jev false}
                         {:decide (fn [_ options _] {:logp (vec (map-indexed (fn [i _] (- (* 2.0 i))) options)) :thought "" :tokens 0})
                          :count-tokens (constantly 1)})
        cases [{"state" "Refund me before Friday or we cancel."
                "questions" {"churn" {"type" "noul" "instructions" "Does the customer threaten to leave?"}
                             "team" {"type" "choice" "instructions" "Which team?" "criteria" {"billing" "money" "legal" nil "other" nil}}}
                "labels" {"churn" false "team" "legal"}}]
        items (cal/collect t cases)]
    (is (= ["noul:2" "choice:3-5"] (map :bucket items)))
    (is (= [[-2.0 0.0] [0.0 -2.0 -4.0]] (map :logits items)) "the raw scores, a noul's as [false true]")
    (is (= [0 1] (map :label items)))
    (let [fitted (cal/calibration items)
          refit (ag/with-calibration t fitted)]
      (is (every? #(> % 1.0) (vals (:temperature-by-options fitted))) "confidently wrong: the fit softens")
      (is (= (:temperature-by-options fitted) (:temperature-by-options (ag/calibration-of refit))))
      (is (< (get-in (think/system-one refit "Refund me." {"churn" (get-in (first cases) ["questions" "churn"])} nil)
                     ["answers" "churn" "confidence"])
             (get-in (think/system-one t "Refund me." {"churn" (get-in (first cases) ["questions" "churn"])} nil)
                     ["answers" "churn" "confidence"]))))))

(deftest a-gold-distribution-is-fit-by-cross-entropy
  (testing "the NLL of an item with a target is its cross-entropy"
    (let [item {:logits [2.0 0.0] :label 0 :target [0.6 0.4]}
          p (cal/softmax [2.0 0.0])]
      (is (< (Math/abs (- (cal/nll [item] 1.0)
                          (- (+ (* 0.6 (Math/log (first p))) (* 0.4 (Math/log (second p)))))))
             1e-12))))
  (testing "soft gold pulls the temperature to where the model's spread matches it"
    (let [items (vec (for [x [1.0 2.0 3.0 4.0]] {:logits [x 0.0] :label 0 :target (cal/softmax [(/ x 4.0) 0.0])}))]
      (is (< (Math/abs (- 4.0 (cal/fit-temperature items))) 0.01)))))
