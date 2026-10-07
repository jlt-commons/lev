(ns lev.clef-test
  "lev.clef: Clef's joint schema head and record encoding against the
  release's own Python (bench/clef_golden.py -> golden/clef/). The head
  test needs the release's joint_head.safetensors (LEV_TEST_CLEF, else
  ~/src/models/clef); the encoding and end-to-end tests also need its
  backbone as a GGUF (LEV_TEST_CLEF_GGUF, else
  ~/src/models/clef-backbone-Q8_0.gguf) and the llm native. Each is
  skipped without what it needs."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.clef :as clef]
            [lev.json :as json]
            [lev.llm :as llm]
            [lev.tensors :as t]
            [lev.test-util :as tu]
            [lev.think :as think]))

(def clef-dir
  (let [home (System/getProperty "user.home")]
    (some #(when (and % (.exists (io/file % "joint_head.safetensors"))) %)
          [(System/getenv "LEV_TEST_CLEF") (str home "/src/models/clef")])))

(def clef-gguf
  (let [home (System/getProperty "user.home")]
    (some #(when (and % (.exists (io/file %))) %)
          [(System/getenv "LEV_TEST_CLEF_GGUF") (str home "/src/models/clef-backbone-Q8_0.gguf")])))

(def head (delay (clef/load-head clef-dir)))

(def model
  "The 27B backbone, loaded once: ~29 GB mapped."
  (delay (clef/clef {:name "clef" :model clef-gguf :head clef-dir :n-ctx 4096})))

(defn- golden [name] (edn/read-string (slurp (str tu/golden-dir "/clef/" name))))

(deftest the-head-matches-the-release
  (when (and clef-dir (llm/available?))
    (let [{:keys [n v input-ids questions logits]} (golden "head.edn")
          d 5120
          hidden (t/load-file (str tu/golden-dir "/clef/hidden.f32") [n d])
          lm-head (t/load-file (str tu/golden-dir "/clef/lm_head.f32") [v d])
          ;; the lexical vector of an option: the mean output embedding of its tokens
          lexical (fn [ids] (clef/mean-rows lm-head ids))
          got (clef/head-logits @head hidden input-ids questions lexical)]
      (is (tu/approx= 1e-4 logits got) (pr-str {:want logits :got got})))))

(deftest a-request-encodes-as-the-release-encodes-it
  (when (and clef-gguf (llm/available?))
    (let [{:keys [request input-ids questions]} (golden "encode.edn")
          req (json/read-str request)
          enc (clef/encode @model (get req "state") (get req "questions"))]
      (is (= input-ids (:input-ids enc)) "token ids")
      (is (= (mapv #(select-keys % [:id :type :span :option-spans :option-ids]) questions)
             (mapv #(select-keys % [:id :type :span :option-spans :option-ids]) (:questions enc)))))))

(deftest answers-a-request-in-the-jev-shape
  (when (and clef-gguf clef-dir (llm/available?))
    (let [out (ag/system-one @model
                             "Our checkout started returning errors and orders are blocked."
                             (array-map "department" {"type" "choice" "instructions" "Which team should handle the message?"
                                                      "criteria" (array-map "billing" "Payments or invoices" "technical" "Bugs or outages")}
                                        "urgency" {"type" "score" "instructions" "How urgent is it?"
                                                   "criteria" ["Can wait" "This week" "Today"]}
                                        "outage" {"type" "noul" "instructions" "Is a service down?"})
                             nil)
          a (get out "answers")]
      (is (= "clef" (get out "model")))
      (is (= ["department" "urgency" "outage"] (vec (keys a))))
      (testing "the obvious answers"
        (is (= "technical" (get-in a ["department" "choice"])) (pr-str a))
        (is (< 1.0 (get-in a ["urgency" "score"])) (pr-str a))
        (is (< 0.5 (get-in a ["outage" "noul"])) (pr-str a)))
      (testing "choice probabilities in the caller's order"
        (is (= ["billing" "technical"] (vec (keys (get-in a ["department" "probabilities"])))))))))

(deftest a-thinker-config-names-the-clef-engine
  (testing "without :head it is a config error, before anything loads"
    (let [e (try (think/thinker {:engine "clef" :model "target/nope.gguf"}) nil (catch Exception e e))]
      (is (= :invalid-config (:type (ex-data e))) (pr-str e)))))
