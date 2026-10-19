(ns lev.server-test
  "The HTTP API (lev.server), mirroring TypeSafe's POST /v1/systemone.
  Handler-level tests call the ring handler with request maps; one test
  starts a real server and talks to it with curl."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [lev.agent :as ag]
            [lev.json :as json]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.server :as srv]
            [lev.test-util :as tu]
            [lev.workflows :as wf]))

(def agent tu/agent)
(def workflows (delay (wf/load-workflows ["workflows"])))

(defn- req [method uri & {:keys [body headers]}]
  {:request-method method :uri uri :headers (or headers {}) :body body})

(defn- call
  "[status parsed-body raw-response]"
  [h r]
  (let [resp (h r)]
    [(:status resp) (json/read-str (:body resp)) resp]))

(defn- readme-request []
  (let [cases (tu/read-golden "cases")]
    (seq/json-str (seq/ordered-map [["state" (:readme-state cases)]
                                    ["model" "lev"]
                                    ["questions" (:readme-questions cases)]]))))

(deftest systemone-reproduces-the-python-answer
  (let [h (srv/handler @agent {})
        resp (h (req :post "/v1/systemone" :body (readme-request)))]
    (is (= 200 (:status resp)))
    (is (= "application/json" (get-in resp [:headers "Content-Type"])))
    (testing "the body is json.dumps(Agent.system_one(...)) plus the routing decision"
      (let [body (json/read-str (str/trim (:body resp)))]
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc body "routing")))
        (is (= ["model" "answers" "usage" "routing"] (keys body)))
        (is (= "english" (get-in body ["routing" "model"])))))))

(deftest model-field-selects-the-checkpoint
  (let [h (srv/handler @agent {})
        base (json/read-str (readme-request))
        one-q (seq/ordered-map [["state" "Refund me."]
                                ["questions" (seq/ordered-map [["q" (get-in base ["questions" "churn_risk"])]])]])]
    (testing "absent, or the engine's own name: routed by content, with the decision attached"
      (doseq [body [one-q (assoc one-q "model" "lev") (assoc one-q "model" "jev-latest")]]
        (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str body)))]
          (is (= 200 st))
          (is (= "english" (get b "model")))
          (is (= ["model" "answers" "usage" "routing"] (keys b)))
          (is (= "english" (get-in b ["routing" "model"])))
          (is (= "English Latin text" (get-in b ["routing" "reason"]))))))
    (testing "a checkpoint name or alias is an explicit choice"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "en"))))]
        (is (= 200 st))
        (is (= "english" (get-in b ["routing" "model"])))
        (is (= "explicit model='en'" (get-in b ["routing" "reason"])))))
    (testing "lang and task route too"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "lang" "en-GB"))))]
        (is (= 200 st))
        (is (= "explicit lang='en-GB'" (get-in b ["routing" "reason"])))))
    (testing "an unknown model is a 422 naming the choices"
      (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "gpt-4"))))]
        (is (= 422 st))
        (is (= ["body" "model"] (get-in b ["detail" 0 "loc"])))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "multilingual"))))
    (testing "a checkpoint that is not prepared is a 503, not a crash"
      (let [h (srv/handler (router/preloaded @agent "english" {:models {"multilingual" "target/not-prepared"}}) {})
            [st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q "model" "multilingual"))))]
        (is (= 503 st))
        (is (str/includes? (get b "detail") "multilingual"))))
    (testing "must be strings"
      (doseq [k ["model" "lang" "task"]]
        (let [[st b] (call h (req :post "/v1/systemone" :body (seq/json-str (assoc one-q k 3))))]
          (is (= 422 st) k)
          (is (= ["body" k] (get-in b ["detail" 0 "loc"])) k))))))

(deftest route-without-inference
  (let [boom (fn [name & _] (throw (ex-info (str "must not load " name) {})))
        h (srv/handler (router/make-router {:loader boom}) {})]
    (testing "POST /v1/route answers the decision and loads nothing"
      (let [[st b] (call h (req :post "/v1/route" :body (seq/json-str {"state" {"body" "\u092e\u0941\u091d\u0938\u0947 \u0926\u094b"}})))]
        (is (= 200 st))
        (is (= "multilingual" (get b "model")))
        (is (= "convaiinnovations/laya/multilingual" (get b "repo")))
        (is (= "devanagari" (get-in b ["detection" "script"]))))
      (let [[st b] (call h (req :post "/v1/route" :body (seq/json-str {"state" "hi" "task" "typed_decisions"})))]
        (is (= 200 st))
        (is (= "typed-decisions" (get b "model")))))
    (testing "questions are optional here, state is not"
      (is (= 422 (first (call h (req :post "/v1/route" :body "{}"))))))))

(deftest models-and-workflows-listing
  (let [h (srv/handler @agent {:workflows @workflows})]
    (testing "GET /v1/models: the three checkpoints, which are prepared and loaded"
      (let [[st b] (call h (req :get "/v1/models"))]
        (is (= 200 st))
        (is (= ["english" "multilingual" "typed-decisions"] (keys (get b "models"))))
        (is (= "convaiinnovations/laya" (get-in b ["models" "english" "repo"])))
        (is (true? (get-in b ["models" "english" "loaded"])))
        (is (true? (get-in b ["models" "english" "available"])))
        (is (= {"max_len" 512 "head_max_len" 192} (get-in b ["models" "english" "limits"])))
        (is (= (get-in b ["models" "multilingual" "available"]) (some? (get-in b ["models" "multilingual" "limits"])))
            "limits are known exactly when the checkpoint is prepared")
        (is (= "english" (get b "default")))))
    (testing "GET /v1/workflows: name, description, question ids, whether options are taken"
      (let [[st b] (call h (req :get "/v1/workflows"))]
        (is (= 200 st))
        (is (= ["demo" "email" "guard" "llm-router" "moderation" "security" "triage"] (keys (get b "workflows"))))
        (is (= ["category" "is_spam" "is_phishing" "urgency" "needs_reply"] (get-in b ["workflows" "email" "questions"])))
        (is (true? (get-in b ["workflows" "email" "options"])))
        (is (false? (get-in b ["workflows" "demo" "options"])))
        (is (str/starts-with? (get-in b ["workflows" "email" "description"]) "Email triage"))))))

(deftest workflow-endpoint
  (let [h (srv/handler @agent {:workflows @workflows})
        g (tu/read-golden "email_answers")
        {:keys [body-index state result]} (first (:cases g))
        raw (first (nth (:clean (tu/read-golden "email")) body-index))]
    (testing "POST /v1/workflows/email builds the state, asks its questions, reports both"
      (let [[st b] (call h (req :post "/v1/workflows/email"
                                :body (seq/json-str (seq/ordered-map [["input" (seq/ordered-map [["subject" "Support request"] ["body" raw] ["from" "someone@example.com"]])]]))))
            want (json/read-str result)]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "constraints" "routing" "workflow" "state"] (keys b)))
        (is (= "email" (get b "workflow")))
        (is (= (into {} state) (into {} (get b "state"))) "the cleaned email the model read")
        (is (= (dissoc (get want "answers") "wide")
               (into {} (map (fn [[k a]] [k (dissoc a "decided")]) (get b "answers"))))
            "the golden answers, plus the workflow's constrained decisions")))
    (testing "options reach the workflow; a bare string input is allowed"
      (let [[st b] (call h (req :post "/v1/workflows/email"
                                :body (seq/json-str {"input" "Refund me" "options" {"categories" {"refund" "money back" "other" "else"}}})))]
        (is (= 200 st))
        (is (= ["refund" "other"] (keys (get-in b ["answers" "category" "probabilities"]))))))
    (testing "demo needs no input at all"
      (let [[st b] (call h (req :post "/v1/workflows/demo" :body "{}"))]
        (is (= 200 st))
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc b "routing" "workflow" "state")))))
    (testing "model / lang / task route the workflow request too"
      (let [[st b] (call h (req :post "/v1/workflows/demo" :body (seq/json-str {"model" "english"})))]
        (is (= 200 st))
        (is (= "explicit model='english'" (get-in b ["routing" "reason"])))))
    (testing "unknown workflow is a 404 that lists the known ones"
      (let [[st b] (call h (req :post "/v1/workflows/nope" :body "{}"))]
        (is (= 404 st))
        (is (str/includes? (get b "detail") "email"))))
    (testing "options to a workflow that takes none, or a non-object body, are 422"
      (is (= 422 (first (call h (req :post "/v1/workflows/demo" :body (seq/json-str {"options" {"x" 1}}))))))
      (is (= 422 (first (call h (req :post "/v1/workflows/demo" :body "[1]"))))))))

(deftest bearer-auth
  (let [h (srv/handler @agent {:api-key "s3cret"})
        body (readme-request)]
    (testing "missing or wrong key"
      (doseq [headers [{} {"authorization" "Bearer nope"} {"authorization" "s3cret"} {"authorization" "Basic s3cret"}]]
        (let [[st b] (call h (req :post "/v1/systemone" :body body :headers headers))]
          (is (= 401 st) (pr-str headers))
          (is (= "Missing or invalid API key. Check the Authorization header." (get b "detail"))))))
    (testing "right key"
      (is (= 200 (first (call h (req :post "/v1/systemone" :body body
                                     :headers {"authorization" "Bearer s3cret"}))))))
    (testing "health needs no key"
      (is (= 200 (first (call h (req :get "/health"))))))))

(deftest validation-errors-are-422-with-a-location
  (let [h (srv/handler @agent {})
        post (fn [body] (call h (req :post "/v1/systemone" :body body)))
        q {"type" "noul" "instructions" "ok?"}
        loc (fn [b] (get-in b ["detail" 0 "loc"]))]
    (testing "malformed json"
      (let [[st b] (post "{\"state\": ")]
        (is (= 422 st))
        (is (= ["body"] (loc b)))
        (is (str/starts-with? (get-in b ["detail" 0 "msg"]) "json:"))))
    (testing "empty body"
      (is (= 422 (first (post "")))))
    (testing "not an object"
      (is (= ["body"] (loc (second (post "[1, 2]"))))))
    (testing "state"
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"questions" {"q" q}}))))))
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"state" nil "questions" {"q" q}}))))))
      (is (= ["body" "state"] (loc (second (post (seq/json-str {"state" 3 "questions" {"q" q}})))))))
    (testing "questions"
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s"}))))))
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s" "questions" {}}))))))
      (is (= ["body" "questions"] (loc (second (post (seq/json-str {"state" "s" "questions" [q]})))))))
    (testing "each bad question is reported with its id and field"
      (let [[st b] (post (seq/json-str (seq/ordered-map
                                        [["state" "s"]
                                         ["questions" (seq/ordered-map
                                                       [["ok" q]
                                                        ["bad_type" {"type" "yesno" "instructions" "x"}]
                                                        ["bad_crit" {"type" "score" "instructions" "x" "criteria" ["one"]}]
                                                        ["no_ins" {"type" "choice" "criteria" ["a" "b"]}]])]])))]
        (is (= 422 st))
        (is (= [["body" "questions" "bad_type" "type"]
                ["body" "questions" "bad_crit" "criteria"]
                ["body" "questions" "no_ins" "instructions"]]
               (map #(get % "loc") (get b "detail"))))
        (is (every? #(= "value_error" (get % "type")) (get b "detail")))))
    (testing "options that cannot fit"
      (let [many (seq/ordered-map (map (fn [i] [(str "option-" i) nil]) (range 200)))
            [st b] (post (seq/json-str {"state" "s" "questions" {"wide" {"type" "choice" "instructions" "x" "criteria" many}}}))]
        (is (= 422 st))
        (is (= ["body" "questions" "wide" "criteria"] (loc b)))))))

(deftest routing
  (let [h (srv/handler @agent {:workflows @workflows})]
    (is (= 404 (first (call h (req :get "/nope")))))
    (is (= 405 (first (call h (req :get "/v1/systemone")))))
    (is (= 405 (first (call h (req :post "/health")))))
    (is (= 405 (first (call h (req :get "/v1/workflows/email")))))
    (is (= 405 (first (call h (req :post "/v1/models")))))
    (let [[st b] (call h (req :get "/health"))]
      (is (= 200 st))
      (is (= {"status" "ok" "model" "lev" "loaded" ["english"] "thinkers" []
              "workflows" ["demo" "email" "guard" "llm-router" "moderation" "security" "triage"]} b)))
    (testing "ruuter 2: a segment is a segment"
      ;; :name spans one segment, not the rest of the path (1.3.5's `.*` did)
      (is (= 404 (first (call h (req :post "/v1/workflows/email/extra" :body "{}")))))
      (is (= 404 (first (call h (req :post "/v1/patterns/gate/extra" :body "{}")))))
      ;; empty segments are dropped, so a trailing slash reaches the route
      (is (= 200 (first (call h (req :get "/health/")))))
      (is (= 200 (first (call h (req :get "/v1/models/")))))
      (is (= 405 (first (call h (req :get "/v1/systemone/"))))))))

(deftest wire-order-is-preserved
  (testing "14 options and 9 questions keep their JSON order end to end"
    (let [h (srv/handler @agent {})
          opts ["billing" "refund" "bug" "outage" "login" "pricing" "demo"
                "hiring" "payroll" "legal" "shipping" "returns" "feedback" "other"]
          q (str "{\"type\": \"choice\", \"instructions\": \"Pick the closest topic.\", \"criteria\": {"
                 (str/join ", " (map #(str "\"" % "\": null") opts)) "}}")
          body (str "{\"state\": \"Refund the duplicate March invoice.\", \"questions\": {"
                    (str/join ", " (map #(str "\"q" % "\": " q) (range 9))) "}}")
          [st b] (call h (req :post "/v1/systemone" :body body))]
      (is (= 200 st))
      (is (= (map #(str "q" %) (range 9)) (keys (get b "answers"))))
      (is (= opts (keys (get-in b ["answers" "q0" "probabilities"]))))
      (is (= (get-in b ["answers" "q0"]) (get-in b ["answers" "q8"]))))))

(deftest real-http-roundtrip
  (let [server (srv/start @agent {:port 18321 :host "127.0.0.1" :api-key "k"})
        base "http://127.0.0.1:18321"]
    (try
      (let [curl (fn [args] (jolt.host/sh-out (str "curl -s " args)))
            health (curl (str base "/health"))
            body (str/replace (readme-request) "'" "'\\''")
            answer (curl (str "-X POST -H 'Content-Type: application/json' -H 'Authorization: Bearer k' -d '"
                             body "' " base "/v1/systemone"))
            denied (curl (str "-o /dev/null -w '%{http_code}' -X POST -d '{}' " base "/v1/systemone"))]
        (is (= {"status" "ok" "model" "lev" "loaded" ["english"] "thinkers" [] "workflows" []}
               (json/read-str (str/trim health))))
        (tu/answers-match (:system-one (tu/read-golden "readme")) (seq/json-str (dissoc (json/read-str (str/trim answer)) "routing")))
        (is (= "401" (str/trim denied))))
      (finally
        (srv/stop server)))))

(deftest constraints-on-the-wire
  (let [h (srv/handler @agent {:workflows @workflows})
        base (json/read-str (readme-request))
        post (fn [path body] (call h (req :post path :body (seq/json-str body))))]
    (testing "constraints decide jointly; the report rides along after usage"
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" [["implies" ["department" "billing"] ["churn_risk" true]]]))]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "constraints" "routing"] (keys b)))
        (is (= "billing" (get-in b ["answers" "department" "decided"])))
        (is (true? (get-in b ["answers" "churn_risk" "decided"])))
        (is (= 0.4312 (get-in b ["answers" "churn_risk" "noul"])))
        (is (= {"feasible" true "decoder" "exact" "exact" true "violations" []} (get b "constraints")))))
    (testing "without them the body is exactly what it was"
      (let [[_ b] (post "/v1/systemone" base)]
        (is (= ["model" "answers" "usage" "routing"] (keys b)))
        (is (not (contains? (get-in b ["answers" "department"]) "decided")))))
    (testing "a bad constraint is a 422 at its index, before any inference"
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" [["not" ["churn_risk" true]] ["implies" ["department" "legal"] ["churn_risk" true]]]))]
        (is (= 422 st))
        (is (= ["body" "constraints" 1] (get-in b ["detail" 0 "loc"])))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "\"legal\" is not an option of \"department\""))
        (is (= "value_error" (get-in b ["detail" 0 "type"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "constraints" {"implies" 1}))]
        (is (= 422 st))
        (is (= ["body" "constraints"] (get-in b ["detail" 0 "loc"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "on_infeasible" "explode"))]
        (is (= 422 st))
        (is (= ["body" "on_infeasible"] (get-in b ["detail" 0 "loc"])))))
    (testing "on_infeasible: min_violations reports, raise is a 422 with the violations"
      (let [contradiction [["all-of" ["churn_risk" true] ["churn_risk" false]]]
            [st b] (post "/v1/systemone" (assoc base "constraints" contradiction))]
        (is (= 200 st))
        (is (false? (get-in b ["constraints" "feasible"])))
        (is (= contradiction (get-in b ["constraints" "violations"])))
        (let [[st b] (post "/v1/systemone" (assoc base "constraints" contradiction "on_infeasible" "raise"))]
          (is (= 422 st))
          (is (= ["body" "constraints"] (get-in b ["detail" 0 "loc"])))
          (is (= "infeasible" (get-in b ["detail" 0 "type"])))
          (is (= contradiction (get-in b ["detail" 0 "violations"]))))))
    (testing "a workflow's own constraints apply, and the request's add to them"
      (let [[st b] (post "/v1/workflows/email" {"input" {"subject" "WIN A PRIZE" "body" "You have been selected! Click here to claim your reward now!!!"}})]
        (is (= 200 st))
        (is (contains? (get-in b ["answers" "is_spam"]) "decided"))
        (is (true? (get-in b ["constraints" "feasible"])))
        (when (true? (get-in b ["answers" "is_spam" "decided"]))
          (is (false? (get-in b ["answers" "needs_reply" "decided"])) "spam implies no reply")))
      (let [[st b] (post "/v1/workflows/email" {"input" "Refund me" "constraints" [["min-level" "urgency" 2]]})]
        (is (= 200 st))
        (is (= 2 (get-in b ["answers" "urgency" "decided"]))))
      (let [[st b] (post "/v1/workflows/email" {"input" "Refund me" "constraints" [["min-level" "urgency" 5]]})]
        (is (= 422 st))
        (is (= ["body" "constraints" 0] (get-in b ["detail" 0 "loc"])) "indexed among the request's own")))
    (testing "the listing shows a workflow's constraints"
      (let [[_ b] (call h (req :get "/v1/workflows"))]
        (is (= [["implies" ["is_spam" true] ["needs_reply" false]]
                ["implies" ["is_phishing" true] ["needs_reply" false]]]
               (get-in b ["workflows" "email" "constraints"])))
        (is (= [] (get-in b ["workflows" "demo" "constraints"])))))))

(deftest a-thinker-behind-the-api
  (let [seen (atom [])
        fake (fn [name cfg]
               {:kind :thinker :name name :cfg (merge {:thinking true :max-think-tokens 64 :temperature 1.0 :top-p 0.95 :min-p 0.0 :seed 1
                                                       :system "sys"} cfg)
                :decide (fn [prompt options opts]
                          (swap! seen conj {:options options :opts opts})
                          {:logp (vec (map-indexed (fn [i _] (- (* 1.5 i) 0.5)) options)) :thought "hmm\n</think>" :tokens 3})
                :count-tokens (fn [_] 5)})
        rt (router/preloaded @agent "english" {:thinkers {"minicpm5" {:model "target/no-such.gguf" :thinking true}}
                                               :thinker-loader fake})
        h (srv/handler rt {:workflows @workflows})
        base (json/read-str (readme-request))
        post (fn [path body] (call h (req :post path :body (seq/json-str body))))]
    (testing "model: <thinker> answers with the thinker's shapes and a thinking report"
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5"))]
        (is (= 200 st) (pr-str b))
        (is (= ["model" "answers" "usage" "thinking" "routing"] (keys b)))
        (is (= "minicpm5" (get b "model")))
        (is (= "minicpm5" (get-in b ["routing" "model"])))
        (is (= ["type" "choice" "probabilities" "confidence"] (keys (get-in b ["answers" "department"]))))
        (is (= 12 (get-in b ["usage" "output_tokens"])))
        (is (true? (get-in b ["thinking" "enabled"])))
        (is (every? #(= 64 (:think-max (:opts %))) @seen))))
    (testing "think_budget and think_deadline_ms per request"
      (reset! seen [])
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5" "think_budget" 16 "think_deadline_ms" 60000))]
        (is (= 200 st) (pr-str b))
        (is (every? #(= 16 (:think-max (:opts %))) @seen))
        (is (every? #(number? (:deadline-ms (:opts %))) @seen))
        (is (= 16 (get-in b ["thinking" "max_tokens"])))
        (is (= 60000 (get-in b ["thinking" "deadline_ms"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5" "think_budget" 0))]
        (is (= 200 st))
        (is (false? (get-in b ["thinking" "enabled"]))))
      (let [loc (fn [body] (let [[st b] (post "/v1/systemone" body)] [st (get-in b ["detail" 0 "loc"])]))]
        (is (= [422 ["body" "think_budget"]] (loc (assoc base "model" "minicpm5" "think_budget" 65)))
            "past the thinker's max-think-tokens")
        (is (= [422 ["body" "think_budget"]] (loc (assoc base "model" "minicpm5" "think_budget" -1))))
        (is (= [422 ["body" "think_budget"]] (loc (assoc base "model" "minicpm5" "think_budget" 1.5))))
        (is (= [422 ["body" "think_budget"]] (loc (assoc base "model" "minicpm5" "think_budget" 8 "thinking" false))))
        (is (= [422 ["body" "think_deadline_ms"]] (loc (assoc base "model" "minicpm5" "think_deadline_ms" "soon"))))
        (is (= [422 ["body" "think_budget"]] (loc (assoc base "model" "english" "think_budget" 8)))
            "a checkpoint does not think")))
    (testing "thinking: false and thought: true per request"
      (reset! seen [])
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5" "thinking" false "thought" true))]
        (is (= 200 st))
        (is (false? (get-in b ["thinking" "enabled"])))
        (is (every? #(= 0 (:think-max (:opts %))) @seen))
        (is (= "hmm\n</think>" (get-in b ["answers" "department" "thought"])))))
    (testing "thinking / thought must be booleans, and mean nothing to a checkpoint"
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5" "thinking" "yes"))]
        (is (= 422 st))
        (is (= ["body" "thinking"] (get-in b ["detail" 0 "loc"]))))
      (let [[st b] (post "/v1/systemone" (assoc base "model" "english" "thinking" true))]
        (is (= 422 st))
        (is (= ["body" "thinking"] (get-in b ["detail" 0 "loc"])))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "thinker"))))
    (testing "constraints apply to a thinker's answers too"
      (let [[st b] (post "/v1/systemone" (assoc base "model" "minicpm5" "constraints" [["not" ["department" "other"]]]))]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "thinking" "constraints" "routing"] (keys b)))
        (is (contains? (get-in b ["answers" "department"]) "decided"))))
    (testing "workflows run on a thinker as well"
      (let [[st b] (post "/v1/workflows/demo" {"model" "minicpm5" "thinking" false})]
        (is (= 200 st))
        (is (= "minicpm5" (get b "model")))))
    (testing "a batch of states on a thinker: one result per state, each a systemone answer"
      (let [[st b] (post "/v1/systemone/batch" (-> base (dissoc "state")
                                                   (assoc "model" "minicpm5" "thinking" false
                                                          "states" [(get base "state") "Cancel my plan now."])))]
        (is (= 200 st) (pr-str b))
        (is (= ["results"] (keys b)))
        (is (= 2 (count (get b "results"))))
        (is (every? #(= ["model" "answers" "usage" "thinking" "routing"] (keys %)) (get b "results")))))
    (testing "a batch on the encoders routes and answers each state"
      (let [[st b] (post "/v1/systemone/batch" (-> base (dissoc "state" "model") (assoc "states" [(get base "state")])))
            [_ one] (post "/v1/systemone" base)]
        (is (= 200 st) (pr-str b))
        (is (= (get one "answers") (get-in b ["results" 0 "answers"])))))
    (testing "a batch body is validated like systemone's, per state"
      (doseq [[body loc] [[(dissoc base "state") ["body" "states"]]
                          [(-> base (dissoc "state") (assoc "states" [])) ["body" "states"]]
                          [(-> base (dissoc "state") (assoc "states" "x")) ["body" "states"]]
                          [(-> base (dissoc "state") (assoc "states" [(get base "state") 7])) ["body" "states" 1]]
                          [(-> base (dissoc "state") (assoc "states" (vec (repeat 257 "x")))) ["body" "states"]]
                          [(-> base (assoc "states" ["x"])) ["body" "state"]]
                          [(-> base (dissoc "state") (assoc "states" ["x"] "escalate" {"model" "minicpm5"})) ["body" "escalate"]]]]
        (let [[st b] (post "/v1/systemone/batch" body)]
          (is (= 422 st) (pr-str loc))
          (is (= loc (get-in b ["detail" 0 "loc"])) (pr-str b)))))
    (testing "the listing and health show thinkers"
      (let [[_ b] (call h (req :get "/v1/models"))]
        (is (= {"model" "target/no-such.gguf" "available" false "loaded" true "thinking" true}
               (get-in b ["thinkers" "minicpm5"]))))
      (let [[_ b] (call h (req :get "/health"))]
        (is (= ["english"] (get b "loaded")))
        (is (= ["minicpm5"] (get b "thinkers")))))
    (testing "an unknown model still says so, and lists thinkers among the choices"
      (let [[st b] (post "/v1/systemone" (assoc base "model" "gpt"))]
        (is (= 422 st))
        (is (str/includes? (get-in b ["detail" 0 "msg"]) "minicpm5"))))))

(deftest escalation-and-patterns-on-the-wire
  (let [fake (fn [name cfg]
               {:kind :thinker :name name
                :cfg (merge {:thinking true :max-think-tokens 64 :temperature 1.0 :top-p 0.95 :min-p 0.0 :seed 1 :system "sys"} cfg)
                ;; the thinker is sure of the last option, always
                :decide (fn [_ options _] {:logp (vec (map-indexed (fn [i _] (if (= i (dec (count options))) 0.0 -6.0)) options))
                                           :thought "" :tokens 9})
                :count-tokens (fn [_] 5)})
        rt (router/preloaded @agent "english" {:thinkers {"slow" {:model "target/no-such.gguf"}} :thinker-loader fake})
        h (srv/handler rt {:workflows @workflows})
        base (json/read-str (readme-request))
        post (fn [path body] (call h (req :post path :body (seq/json-str body))))]
    (testing "escalate: the unsure answers are re-asked on the thinker"
      (let [[st b] (post "/v1/systemone" (assoc base "escalate" {"threshold" 0.6 "model" "slow"}))]
        (is (= 200 st) (pr-str b))
        (is (= ["model" "answers" "usage" "escalation" "routing"] (keys b)))
        (is (= "english" (get b "model")))
        (is (= "billing" (get-in b ["answers" "department" "choice"])) "0.9+: the encoder's")
        (is (= 0.0312 (get-in b ["answers" "is_phishing" "noul"])) "0.97 sure: the encoder's")
        (is (contains? (set (get-in b ["escalation" "escalated"])) "churn_risk") "0.57 sure: escalated")
        (is (> 0.01 (get-in b ["answers" "churn_risk" "noul"])) "the thinker's: its last option, false")
        (is (= "slow" (get-in b ["escalation" "model"])))
        (is (= 0.6 (get-in b ["escalation" "threshold"])))
        (is (pos? (get-in b ["escalation" "usage" "output_tokens"])))))
    (testing "with constraints, decided over the merged answers; and on a workflow"
      (let [[st b] (post "/v1/systemone" (assoc base "escalate" {"threshold" 0.6 "model" "slow"}
                                                "constraints" [["implies" ["department" "billing"] ["churn_risk" false]]]))]
        (is (= 200 st))
        (is (= ["model" "answers" "usage" "escalation" "constraints" "routing"] (keys b)))
        (is (false? (get-in b ["answers" "churn_risk" "decided"]))))
      (let [[st b] (post "/v1/workflows/demo" {"escalate" {"threshold" 0.6 "model" "slow" "thinking" false}})]
        (is (= 200 st))
        (is (= "demo" (get b "workflow")))
        (is (contains? b "escalation"))))
    (testing "escalate carries the thinker's think_budget, and checks it"
      (let [[st b] (post "/v1/systemone" (assoc base "escalate" {"threshold" 0.6 "model" "slow" "think_budget" 0}))]
        (is (= 200 st) (pr-str b)))
      (is (= ["body" "escalate" "think_budget"]
             (get-in (second (post "/v1/systemone" (assoc base "escalate" {"model" "slow" "think_budget" "lots"}))) ["detail" 0 "loc"]))))
    (testing "escalate must name a thinker and a threshold in [0, 1]"
      (let [loc (fn [b] (get-in b ["detail" 0 "loc"]))]
        (is (= ["body" "escalate" "model"] (loc (second (post "/v1/systemone" (assoc base "escalate" {"threshold" 0.5}))))))
        (is (= ["body" "escalate" "model"] (loc (second (post "/v1/systemone" (assoc base "escalate" {"model" "english"}))))))
        (is (= ["body" "escalate" "threshold"] (loc (second (post "/v1/systemone" (assoc base "escalate" {"model" "slow" "threshold" 2}))))))
        (is (= ["body" "escalate" "threshold"] (loc (second (post "/v1/systemone" (assoc base "escalate" {"model" "slow" "threshold" {"bool" 0.5}}))))))
        (let [[st b] (post "/v1/systemone" (assoc base "escalate" {"model" "slow" "threshold" {"noul" 0.99 "choice" 0.0}}))]
          (is (= 200 st))
          (is (= {"choice" 0.0 "score" 0.8 "noul" 0.99} (get-in b ["escalation" "threshold"])))
          (is (contains? (set (get-in b ["escalation" "escalated"])) "churn_risk"))
          (is (not (contains? (set (get-in b ["escalation" "escalated"])) "department"))))
        (is (= ["body" "escalate"] (loc (second (post "/v1/systemone" (assoc base "escalate" "slow"))))))))
    (testing "the patterns as endpoints"
      (let [[st b] (post "/v1/patterns/confidence-gate" (assoc base "threshold" 0.6))]
        (is (= 200 st))
        (is (= ["automatic" "escalate" "response"] (keys b)))
        (is (contains? (get b "automatic") "department"))
        (is (contains? (get b "escalate") "churn_risk")))
      (let [[st b] (post "/v1/patterns/composite-score" (assoc base "weights" {"urgency" 2.0}))]
        (is (= 200 st))
        (is (= ["score" "breakdown" "response"] (keys b)))
        (is (= ["urgency" "churn_risk" "is_phishing"] (keys (get b "breakdown"))))
        (is (<= 0.0 (get b "score") 1.0)))
      (let [[st b] (post "/v1/patterns/two-stage-choice"
                         {"state" "Postgres replica lag exceeded the limit on the primary."
                          "taxonomy" (seq/ordered-map [["cloud" (seq/ordered-map [["aws" "Amazon Web Services"] ["gcp" "Google Cloud"]])]
                                                       ["database" (seq/ordered-map [["postgres" "PostgreSQL"] ["redis" "Redis"]])]])})]
        (is (= 200 st) (pr-str b))
        (is (= ["category" "category_confidence" "choice" "choice_confidence" "combined_confidence"] (keys b)))
        (is (contains? #{"cloud" "database"} (get b "category"))))
      (let [utility (fn [extra] (post "/v1/patterns/utility-choice"
                                      (merge {"state" (get base "state")
                                              "question" (get-in base ["questions" "department"])}
                                             extra)))
            [st b] (utility {"value" 10 "costs" {"billing" 100}})]
        (is (= 200 st) (pr-str b))
        (is (= ["choice" "utility" "utilities" "answer"] (keys b)))
        (is (= "billing" (get-in b ["answer" "choice"])) "the encoder's own pick")
        (is (not= "billing" (get b "choice")) "priced out")
        (is (> -90.0 (get-in b ["utilities" "billing"])))
        (is (= ["body" "costs"] (get-in (second (utility {"costs" {"nope" 1}})) ["detail" 0 "loc"])))
        (is (= ["body" "value"] (get-in (second (utility {"value" "x"})) ["detail" 0 "loc"])))
        (is (= ["body" "question"] (get-in (second (utility {"question" {"type" "noul" "instructions" "x?"}})) ["detail" 0 "loc"])))
        (is (= ["body" "question"] (get-in (second (post "/v1/patterns/utility-choice" {"state" "s"})) ["detail" 0 "loc"]))))
      (testing "their validation"
        (is (= 422 (first (post "/v1/patterns/two-stage-choice" {"state" "s"}))))
        (is (= 422 (first (post "/v1/patterns/composite-score" (assoc base "weights" [1 2])))))
        (is (= 422 (first (post "/v1/patterns/confidence-gate" (assoc base "threshold" "high")))))
        (is (= 404 (first (post "/v1/patterns/nope" base))))))))

(deftest debias-on-the-wire
  (let [h (srv/handler @agent {})
        base (json/read-str (readme-request))
        post (fn [body] (call h (req :post "/v1/systemone" :body (seq/json-str body))))]
    (let [[st b] (post (assoc base "debias" true))]
      (is (= 200 st))
      (is (= {"department" 4} (get b "debias")))
      (is (= ["model" "answers" "usage" "debias" "routing"] (keys b))))
    (let [[st b] (post (assoc base "debias" "yes"))]
      (is (= 422 st))
      (is (= ["body" "debias"] (get-in b ["detail" 0 "loc"]))))))

(deftest embeddings-on-the-wire
  (let [rt (router/preloaded @agent "english" {:thinkers {"slow" {:model "target/no-such.gguf"}}
                                               :thinker-loader (fn [n _] {:kind :thinker :name n})})
        h (srv/handler rt {:workflows @workflows})
        post (fn [body] (call h (req :post "/v1/embed" :body (seq/json-str body))))
        loc (fn [b] (get-in b ["detail" 0 "loc"]))]
    (testing "one state, or several"
      (let [[st b] (post {"state" {"service" "payments" "signal" "OOM kills"}})]
        (is (= 200 st) (pr-str b))
        (is (= ["model" "embeddings" "dimensions" "usage" "routing"] (keys b)))
        (is (= "english" (get b "model")))
        (is (= 1 (count (get b "embeddings"))))
        (is (= (get b "dimensions") (count (first (get b "embeddings")))))
        (is (< (Math/abs (- 1.0 (Math/sqrt (reduce + (map #(* % %) (first (get b "embeddings"))))))) 1e-4)))
      (let [[st b] (post {"states" ["heap climbing" "refund my invoice" "CPU saturated"]})]
        (is (= 200 st))
        (is (= 3 (count (get b "embeddings"))))))
    (testing "validation"
      (is (= ["body" "state"] (loc (second (post {})))))
      (is (= ["body" "state"] (loc (second (post {"state" "a" "states" ["b"]})))) "one or the other")
      (is (= ["body" "states"] (loc (second (post {"states" []})))))
      (is (= ["body" "model"] (loc (second (post {"state" "a" "model" "slow"})))) "a thinker cannot embed")
      (is (= ["body" "model"] (loc (second (post {"state" "a" "model" "nope"})))))
      (is (= 405 (first (call h (req :get "/v1/embed"))))))))

(deftest programs-on-the-wire
  (let [rt (router/preloaded @agent "english" {})
        h (srv/handler rt {:workflows @workflows})
        post (fn [body] (call h (req :post "/v1/program" :body (seq/json-str body))))
        loc (fn [b] (get-in b ["detail" 0 "loc"]))
        p {"extract" {"oom" "(?i)oom-killed|out of memory"}
           "questions" {"refund" {"type" "noul" "instructions" "Does the customer ask for money back?"}}
           "rules" [{"when" {"fact" "oom"} "then" "incident"}
                    {"when" {"noul" "refund" ">=" 0.5} "then" "refund"}]
           "default" "other"}]
    (testing "a fact decides without the model"
      (let [[st b] (post {"program" p "state" "pods OOM-killed on payments"})]
        (is (= 200 st) (pr-str b))
        (is (= ["label" "rule" "facts"] (keys b)))
        (is (= "incident" (get b "label")))))
    (testing "otherwise the encoder answers the questions the rules read"
      (let [[st b] (post {"program" p "state" "I was charged twice, please refund the duplicate charge"})]
        (is (= 200 st))
        (is (= "refund" (get b "label")))
        (is (= "english" (get b "model")))))
    (testing "several states"
      (let [[st b] (post {"program" p "states" ["out of memory again" "hello there"]})]
        (is (= 200 st))
        (is (= ["incident" 0] ((juxt #(get % "label") #(get % "rule")) (first (get b "results")))))
        (is (= 2 (count (get b "results"))))))
    (testing "validation"
      (is (= ["body" "program"] (loc (second (post {"state" "a"})))))
      (is (= ["body" "state"] (loc (second (post {"program" p})))))
      (is (= ["body" "program.rules[0].when"]
             (loc (second (post {"program" (assoc-in p ["rules" 0 "when"] {"fact" "nope"}) "state" "a"})))))
      (is (= ["body" "program.extract.oom"]
             (loc (second (post {"program" (assoc-in p ["extract" "oom"] "(") "state" "a"})))))
      (is (= 405 (first (call h (req :get "/v1/program"))))))))

(deftest an-engine-thinker-is-listed-as-not-thinking
  ;; Clef and TinyDecide generate nothing: there is no thought to switch on
  (let [rt (router/make-router {:thinkers {"clef" {:engine "clef" :model "target/no-such.gguf" :head "target/nope"}
                                           "tiny" {:engine "tinydecide" :model "target/nope"}}})
        [_ b] (call (srv/handler rt {}) (req :get "/v1/models"))]
    (is (false? (get-in b ["thinkers" "clef" "thinking"])))
    (is (false? (get-in b ["thinkers" "tiny" "thinking"])))))
