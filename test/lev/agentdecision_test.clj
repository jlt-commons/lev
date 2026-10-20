(ns lev.agentdecision-test
  "AgentDecision-4B (JevAgentHarness/AgentDecision-4B) as a thinker with
  :prompt \"agentdecision\", against its research server's own prompts and
  label logits (bench/agentdecision_golden.py -> golden/agentdecision/).
  The prompts are checked whenever the golden file is there; the scores
  need the merged GGUF (bench/agentdecision_gguf.py) under
  LEV_TEST_AGENTDECISION, else ~/src/models/AgentDecision-4B-Q8_0.gguf,
  and are skipped without it."
  (:require [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.json :as json]
            [lev.llm :as llm]
            [lev.test-util :as tu]
            [lev.think :as think]))

(def golden-file (io/file tu/golden-dir "agentdecision" "requests.json"))

(def requests (delay (when (.exists golden-file) (json/read-str (slurp golden-file)))))

(def gguf
  (let [home (System/getProperty "user.home")]
    (some #(when (and % (.exists (io/file %))) %)
          [(System/getenv "LEV_TEST_AGENTDECISION") (str home "/src/models/AgentDecision-4B-Q8_0.gguf")])))

(def cfg {:name "agentdecision-4b" :prompt "agentdecision" :thinking false})

(deftest the-prompts-are-the-servers
  (when @requests
    (doseq [r @requests]
      (let [calls (atom [])
            t (think/thinker (assoc cfg :jev false)
                             {:decide (fn [prompt options opts]
                                        (swap! calls conj (str prompt (:answer-prefix opts)))
                                        {:logp (mapv (constantly -1.0) options) :thought "" :tokens 0})
                              :count-tokens (constantly 1)})]
        (think/system-one t (get r "state") (get r "questions") nil)
        (is (= (vec (vals (get r "prompts"))) @calls) (get r "name"))))))

(defn- softmax-1 [xs] (llm/softmax (mapv double xs)))

(defn- in-answer-order
  "The golden scores (prompt order, a noul's yes first) in lev's answer
  order (a noul's [false true])."
  [q xs]
  (if (= "noul" (get q "type")) [(nth xs 1) (nth xs 0)] (vec xs)))

(deftest the-scores-are-the-servers
  (when (and @requests gguf (llm/available?))
    (doseq [jev? [false true]]
      (let [t (think/thinker (assoc cfg :model gguf :jev jev? :n-ctx 4096))]
        (try
          (doseq [r @requests]
            (let [rows (ag/forward t (get r "state") (get r "questions"))]
              (doseq [{:keys [qid logits]} rows]
                (let [q (get-in r ["questions" qid])
                      want (softmax-1 (in-answer-order q (get-in r ["logits" qid])))
                      got (softmax-1 logits)]
                  (testing (str (get r "name") " " qid (if jev? " (jev)" ""))
                    ;; Q8_0 against the f32 merge: the same pick, the
                    ;; probabilities within a few points
                    (is (= (apply max-key want (range (count want))) (apply max-key got (range (count got)))))
                    (is (tu/approx= 0.03 want got) (pr-str {:want want :got got})))))))
          (finally ((:close t) t)))))))
