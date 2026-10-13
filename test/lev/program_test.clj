(ns lev.program-test
  "lev.program: a classifier written as data (regex facts, typed
  questions, ordered rules) and run deterministically. Canned agents, no
  model loaded."
  (:require [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.api :as api]
            [lev.program :as prog]
            [lev.router :as router]
            [lev.sequence :as seq]))

(def calls (atom []))

(defmethod ag/system-one* :canned-program
  [{:keys [name table]} _state questions _opts]
  (swap! calls conj [name (vec (map (fn [[k _]] k) questions))])
  (seq/ordered-map
   [["model" name]
    ["answers" (seq/ordered-map
                (map (fn [[qid qdef]]
                       (let [q (ag/to-internal (ag/validate-question qid qdef))
                             p (or (get table qid) (throw (ex-info (str "no answer for " qid) {})))]
                         [qid (ag/typed-answer q p (count p) nil nil)]))
                     questions))]
    ["usage" (array-map "input_tokens" (* 10 (count questions)) "output_tokens" 0)]]))

(defn- canned [name table] {:kind :canned-program :name name :table table})

(def program
  {"labels" ["oom" "abuse" "refund" "ok"]
   "extract" {"oom" "(?i)out of memory|oom-killed"}
   "questions" (array-map
                "insult" (api/noul "Does the text insult someone?")
                "intent" (api/choice "What does the writer want?" (array-map "refund" nil "help" nil "other" nil))
                "anger" (api/score "How angry is the writer?" ["calm" "annoyed" "furious"]))
   "rules" [{"when" {"fact" "oom"} "then" "oom"}
            {"when" {"any" [{"noul" "insult" ">=" 0.7}
                            {"score" "anger" ">=" 1.5}]} "then" "abuse"}
            {"when" {"all" [{"choice" "intent" "is" "refund"}
                            {"not" {"noul" "insult" ">=" 0.5}}]} "then" "refund"}]
   "default" "ok"})

(def calm-refund {"insult" [0.9 0.1] "intent" [0.8 0.1 0.1] "anger" [0.7 0.2 0.1]})

(deftest a-fact-rule-decides-without-the-model
  (reset! calls [])
  (let [r (prog/run (canned "fast" calm-refund) "pod oom-killed at 3am" program)]
    (is (= "oom" (get r "label")))
    (is (= 0 (get r "rule")))
    (is (= {"oom" true} (get r "facts")))
    (is (nil? (get r "answers")))
    (is (= [] @calls))))

(deftest the-model-answers-what-the-rules-still-need
  (reset! calls [])
  (let [r (prog/run (canned "fast" calm-refund) "I want my money back" program)]
    (is (= "refund" (get r "label")))
    (is (= 2 (get r "rule")))
    (is (= ["label" "rule" "facts" "answers" "model" "usage"] (vec (keys r))))
    (is (= 0.8 (get-in r ["answers" "intent" "probabilities" "refund"])))
    (testing "one call, only the questions rules from the first undecided one on read"
      (is (= [["fast" ["insult" "intent" "anger"]]] @calls))))
  (testing "a later rule's questions only"
    (reset! calls [])
    (let [p (assoc program "rules" [{"when" {"fact" "oom"} "then" "oom"}
                                    {"when" {"choice" "intent" "in" ["refund" "help"]} "then" "refund"}])]
      (prog/run (canned "fast" calm-refund) "hello" p)
      (is (= [["fast" ["intent"]]] @calls)))))

(deftest rules-fire-in-order-and-the-default-closes
  (let [angry {"insult" [0.2 0.8] "intent" [0.8 0.1 0.1] "anger" [0.1 0.2 0.7]}
        calm-other {"insult" [0.9 0.1] "intent" [0.1 0.1 0.8] "anger" [0.9 0.05 0.05]}]
    (is (= ["abuse" 1] ((juxt #(get % "label") #(get % "rule")) (prog/run (canned "f" angry) "x" program))))
    (let [r (prog/run (canned "f" calm-other) "x" program)]
      (is (= "ok" (get r "label")))
      (is (nil? (get r "rule"))))))

(deftest a-program-without-questions-never-calls
  (reset! calls [])
  (let [p {"extract" {"tlb" "data TLB error"} "rules" [{"when" {"fact" "tlb"} "then" "KERNDTLB"}] "default" "normal"}]
    (is (= "KERNDTLB" (get (prog/run (canned "f" {}) "data TLB error interrupt" p) "label")))
    (is (= "normal" (get (prog/run (canned "f" {}) "all quiet" p) "label")))
    (is (= [] @calls))))

(deftest a-bad-program-is-refused-before-anything-runs
  (let [field (fn [p] (try (prog/validate p) nil
                           (catch Exception e
                             (is (= :invalid-request (:type (ex-data e))))
                             (:field (ex-data e)))))]
    (is (= program (prog/validate program)))
    (is (= "extract.oom" (field (assoc-in program ["extract" "oom"] "(unclosed"))))
    (is (= "rules[0].when" (field (assoc-in program ["rules" 0 "when"] {"fact" "nope"}))))
    (is (= "rules[1].when" (field (assoc-in program ["rules" 1 "when"] {"noul" "intent" ">=" 0.5}))))
    (is (= "rules[2].when" (field (assoc-in program ["rules" 2 "when"] {"choice" "intent" "is" "nope"}))))
    (is (= "rules[1].when" (field (assoc-in program ["rules" 1 "when"] {"noul" "insult" ">=" "high"}))))
    (is (= "rules[1].when" (field (assoc-in program ["rules" 1 "when"] {"what" 1}))))
    (is (= "rules[0].then" (field (assoc-in program ["rules" 0 "then"] "unlisted"))))
    (is (= "default" (field (dissoc program "default"))))
    (is (= "questions.insult" (field (assoc-in program ["questions" "insult"] {"type" "noul"}))))))

(deftest unsure-answers-escalate-to-a-thinker
  (reset! calls [])
  (let [fast (canned "fast" {"insult" [0.4 0.6] "intent" [0.98 0.01 0.01] "anger" [0.98 0.01 0.01]})
        slow (canned "slow" {"insult" [0.05 0.95]})
        rt (router/make-router {:loader (fn [name & _] (if (= name "english") fast (throw (ex-info "no" {}))))
                                :thinkers {"slow" {:model "x.gguf"}}
                                :thinker-loader (fn [_ _] slow)})
        r (prog/run rt "x" (assoc program "escalate" {"model" "slow" "threshold" 0.8}))]
    (is (= "abuse" (get r "label")))
    (is (= ["insult"] (get-in r ["escalation" "escalated"])))
    (is (= 0.95 (get-in r ["answers" "insult" "noul"])))
    (is (= [["fast" ["insult" "intent" "anger"]] ["slow" ["insult"]]] @calls))))

(deftest label-is-the-rules-over-facts-and-answers
  (let [a (fn [table] (get (prog/run (canned "f" table) "x" program) "answers"))
        angry (a {"insult" [0.2 0.8] "intent" [0.8 0.1 0.1] "anger" [0.1 0.2 0.7]})]
    (is (= ["abuse" 1] (prog/label program {"oom" false} angry)))
    (is (= ["oom" 0] (prog/label program {"oom" true} angry)))
    (is (= ["ok" nil] (prog/label {"rules" [] "default" "ok"} {} nil)))))

(deftest observe-answers-every-question
  (reset! calls [])
  (let [o (prog/observe (canned "fast" calm-refund) "pod oom-killed" program)]
    (is (= {"oom" true} (get o "facts")))
    (is (= ["insult" "intent" "anger"] (vec (keys (get o "answers")))))
    (is (= [["fast" ["insult" "intent" "anger"]]] @calls))))

(deftest fit-moves-each-cutoff-to-the-middle-of-its-best-range
  (let [p {"questions" {"toxic" (api/noul "Toxic?")}
           "rules" [{"when" {"any" [{"noul" "toxic" ">=" 0.5}]} "then" "toxic"}]
           "default" "ok"}
        case* (fn [prob label] {"facts" {} "answers" {"toxic" {"type" "noul" "noul" prob}} "label" label})
        cases [(case* 0.6 "toxic") (case* 0.35 "toxic") (case* 0.25 "ok") (case* 0.1 "ok")]
        r (prog/fit p cases)]
    (is (= 0.75 (get r "before")))
    (is (= 1.0 (get r "accuracy")))
    (testing "the best cutoffs are 0.26..0.35; the median of that run is kept"
      (is (= 0.3 (get-in r ["program" "rules" 0 "when" "any" 0 ">="]))))
    (is (= (prog/validate (get r "program")) (get r "program")))
    (testing "nothing to gain: the program is unchanged"
      (is (= p (get (prog/fit p [(case* 0.9 "toxic") (case* 0.1 "ok")]) "program"))))))
