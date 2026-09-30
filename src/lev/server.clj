(ns lev.server
  "HTTP front for the checkpoints and workflows, mirroring the TypeSafe Jev
  API (https://docs.typesafe.ai/api) and adding what the Python package's
  Router and presets offer:

    POST /v1/systemone        {\"state\" ..., \"questions\" {...},
                               \"constraints\"? [...], \"on_infeasible\"?,
                               \"model\"? \"lang\"? \"task\"?
                               \"thinking\"? bool, \"thought\"? bool, \"debias\"? bool}
                              -> {\"model\" \"english\", \"answers\" {...},
                                  \"usage\" {...}, \"thinking\"? {...},
                                  \"constraints\"? {...}, \"routing\" {...}}
                              with \"escalate\" {\"model\" thinker, \"threshold\"?
                              0.8 or {type: t}, \"thinking\"?}: the answers below the
                              threshold are re-asked on the thinker and the
                              body carries an \"escalation\" report (lev.patterns)
    POST /v1/systemone/batch  the same body with \"states\" [...] (1-256) in place
                              of \"state\" -> {\"results\" [one systemone answer per
                              state, in order]}; a thinker with thinking off
                              decides them all in one pass (Jev mode), the
                              encoders route and answer each; no escalate
    POST /v1/route            same body, questions optional -> the routing
                              decision alone, nothing loaded or run
    POST /v1/patterns/confidence-gate   systemone body + \"threshold\"? -> {\"automatic\" \"escalate\" \"response\"}
    POST /v1/patterns/composite-score   systemone body + \"weights\"? \"normalize\"? -> {\"score\" \"breakdown\" \"response\"}
    POST /v1/patterns/two-stage-choice  {\"state\", \"taxonomy\" {category {option description}},
                                         \"instructions_category\"? \"instructions_option\"?, \"model\"? ...}
                                        -> {\"category\" .. \"choice\" .. \"combined_confidence\"}
    POST /v1/workflows/:name  {\"input\" ..., \"options\"? {...},
                               \"constraints\"? [...], \"on_infeasible\"?,
                               \"model\"? \"lang\"? \"task\"?}
                              -> systemone answer + \"workflow\" + the built \"state\"
    GET  /v1/models           the checkpoints: repo, data dir, prepared, loaded
    GET  /v1/workflows        the loaded workflows: description, question ids
    GET  /health              {\"status\" \"ok\", \"model\" \"lev\",
                               \"loaded\" [...], \"workflows\" [...]}

  `model` is absent (or the engine's own name, lev, or a Jev SDK's default
  jev-latest / jev-preview) to route by
  content, a checkpoint name / alias (english, multilingual,
  typed-decisions, en, ml, ...) to pick one, or a thinker's name (the
  generative models config.edn :thinkers / --thinker declare; lev.think)
  for the slow accurate answer, with `thinking` (bool) overriding the
  thinker's default and `thought` (bool) adding its reasoning to each
  answer; `lang` and `task` are the Router's other hints. `constraints` is a list of lev.constraints over
  the question ids (a workflow's own come first, the request's are added),
  decided jointly after the forward pass: every answer then carries
  `decided` and the body a `constraints` report; `on_infeasible` is
  min_violations (default) or raise (a 422 of type infeasible listing the
  violated constraints). Anything else is a 422. Routes are dispatched by
  ruuter.

  Auth is `Authorization: Bearer <key>` on /v1/* when the server is started
  with an :api-key (LEV_API_KEY); without one every request is accepted.
  Errors: 401 for a missing/invalid key, 422 with a FastAPI-style
  {\"detail\": [{\"loc\": [...], \"msg\": ..., \"type\": ...}]} for anything
  wrong with the body, 404 for unknown routes and workflows, 405 for the
  wrong method, 503 when the chosen checkpoint has no prepared data, 413
  from the adapter when the body exceeds :max-request-bytes.

  As a library: (handler router-or-agent opts) is a plain ring handler to
  mount in another app; (start router-or-agent opts) / (stop server) run it
  on ring-chez-adapter. As a binary: `jolt build -m lev.server -o
  lev-server` (the `binary` task), then `./lev-server --data data`.

  Requests are parsed with lev.json, which keeps object key order: the
  order of options, questions and state fields is model input. Inference
  (and checkpoint loading) runs one request at a time behind a lock; the
  adapter's workers only overlap on I/O."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.llm]
            [lev.mcp :as mcp]
            [lev.patterns :as pat]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.tokenizer :as tk]
            [lev.workflows :as wf]
            [ring-chez.adapter :as adapter]
            [ruuter.core :as ruuter])
  (:gen-class))

(def engine-name "lev")

;; the engine's own name, and the defaults a TypeSafe SDK sends: "route for
;; me", not a model choice
(def ^:private engine-names #{"lev" "jev-latest" "jev-preview"})

;; --- responses ------------------------------------------------------------------

(defn- json-response [status body]
  {:status status
   :headers {"Content-Type" "application/json"}
   :body (str (seq/json-str body) "\n")})

(defn- detail [loc msg type]
  (seq/ordered-map [["loc" (vec loc)] ["msg" msg] ["type" type]]))

(defn- unprocessable [details]
  (json-response 422 {"detail" (vec details)}))

(def ^:private unauthorized
  (json-response 401 {"detail" "Missing or invalid API key. Check the Authorization header."}))

(def ^:private not-found (json-response 404 {"detail" "Not Found"}))
(def ^:private method-not-allowed (json-response 405 {"detail" "Method Not Allowed"}))

(defn- question-loc [{:keys [qid field]}]
  (cond-> ["body" "questions" (str qid)] field (conj field)))

(defn- error-response
  "The HTTP shape of the exceptions the library throws; rethrows the rest."
  [e]
  (let [data (ex-data e)]
    (case (:type data)
      :invalid-json (unprocessable [(detail ["body"] (ex-message e) "json_invalid")])
      :invalid-question (unprocessable [(detail (question-loc data) (ex-message e) "value_error")])
      :unknown-model (unprocessable [(detail ["body" "model"] (ex-message e) "value_error")])
      :invalid-request (unprocessable [(detail ["body" (or (:field data) "options")] (ex-message e) "value_error")])
      :invalid-constraint (unprocessable [(detail (cond-> ["body" "constraints"]
                                                    (:index data) (conj (- (:index data) (:offset data 0))))
                                                  (ex-message e) "value_error")])
      :infeasible (unprocessable [(assoc (detail ["body" "constraints"] (ex-message e) "infeasible")
                                         "violations" (:violations data))])
      :model-unavailable (json-response 503 {"detail" (ex-message e)})
      (throw e))))

;; --- request ---------------------------------------------------------------------

(defn- body-string [body]
  (cond (nil? body) "" (string? body) body :else (slurp body)))

(defn- parse-body
  "The JSON document in the request body; ex-info :invalid-json otherwise.
  With allow-empty?, an empty body reads as {}."
  ([req] (parse-body req false))
  ([req allow-empty?]
   (let [text (body-string (:body req))]
     (if (str/blank? text)
       (if allow-empty? {} (throw (ex-info "json: request body is empty" {:type :invalid-json})))
       (json/read-str text)))))

(defn- check-routing-fields
  "model / lang / task must be strings, and model a known checkpoint or
  thinker (or the engine's own name); thinking / thought booleans, and
  only with a thinker."
  [rt body]
  (let [m (get body "model")
        known (when (and (string? m) (not (engine-names m)))
                (try (router/normalise-name rt m) nil
                     (catch Exception e
                       [(detail ["body" "model"] (ex-message e) "value_error")])))
        thinker? (and (string? m) (not (engine-names m)) (nil? known) (router/thinker? rt m))]
    (concat
     (for [k ["model" "lang" "task"]
           :when (and (contains? body k) (not (string? (get body k))))]
       (detail ["body" k] (str k " must be a string") "type_error"))
     known
     (for [k ["thinking" "thought"]
           :when (contains? body k)
           :let [v (get body k)]
           :when (or (not (boolean? v)) (not thinker?))]
       (if (boolean? v)
         (detail ["body" k] (str k " applies to a thinker model (model: one of "
                                (str/join ", " (router/thinker-names rt)) ")") "value_error")
         (detail ["body" k] (str k " must be true or false") "type_error"))))))

(def ^:private infeasible-modes #{"min_violations" "raise"})

(defn- check-constraints
  "constraints, when present, is a list (each entry is checked against the
  questions by the library, before any inference); on_infeasible one of
  the two modes."
  [body]
  (concat
   (when (and (contains? body "constraints") (not (sequential? (get body "constraints"))))
     [(detail ["body" "constraints"] "constraints must be a list of constraints" "type_error")])
   (when (and (contains? body "on_infeasible") (not (contains? infeasible-modes (get body "on_infeasible"))))
     [(detail ["body" "on_infeasible"] "on_infeasible must be min_violations or raise" "value_error")])
   (when (and (contains? body "debias") (not (boolean? (get body "debias"))))
     [(detail ["body" "debias"] "debias must be true or false" "type_error")])))

(defn- check-escalate
  "escalate, when present, is an object naming a thinker under model,
  with an optional threshold in [0, 1] and thinking boolean."
  [rt body]
  (when (contains? body "escalate")
    (let [e (get body "escalate")]
      (if-not (map? e)
        [(detail ["body" "escalate"] "escalate must be an object: {\"model\": thinker, \"threshold\": 0.8}" "type_error")]
        (concat
         (let [m (get e "model")]
           (cond
             (not (string? m)) [(detail ["body" "escalate" "model"] "escalate.model (a thinker's name) is required" "value_error")]
             (not (try (router/thinker? rt m) (catch Exception _ false)))
             [(detail ["body" "escalate" "model"] (str "escalate.model must name a thinker: one of "
                                                       (str/join ", " (router/thinker-names rt))) "value_error")]))
         (when (contains? e "threshold")
           (try (pat/thresholds (get e "threshold")) nil
                (catch Exception ex
                  [(detail ["body" "escalate" "threshold"] (str "escalate." (ex-message ex)) "value_error")])))
         (when (and (contains? e "thinking") (not (boolean? (get e "thinking"))))
           [(detail ["body" "escalate" "thinking"] "escalate.thinking must be true or false" "type_error")]))))))

(defn- check-state [body]
  (let [state (get body "state")]
    (when-not (or (string? state) (map? state) (sequential? state))
      [(detail ["body" "state"] "state is required: a string, object or array" "value_error")])))

(defn- check-questions [qs]
  (cond
    (not (map? qs)) [(detail ["body" "questions"] "questions is required: an object of question id -> question" "value_error")]
    (empty? qs) [(detail ["body" "questions"] "questions must not be empty" "value_error")]
    :else (keep (fn [[qid qdef]]
                  (try (ag/validate-question qid qdef) nil
                       (catch Exception e
                         (detail (question-loc (ex-data e)) (ex-message e) "value_error"))))
                qs)))

(defn- check-request
  "Everything wrong with a parsed /v1/systemone body, as detail entries.
  Each question is checked with agent/validate-question, so the reasons
  match the library."
  [rt body]
  (if-not (map? body)
    [(detail ["body"] "request body must be a JSON object" "type_error")]
    (vec (concat (check-state body)
                 (check-routing-fields rt body)
                 (check-constraints body)
                 (check-escalate rt body)
                 (check-questions (get body "questions"))))))

(defn- routing-opts
  "The Router hints in a body: :model (nil for the engine's own name),
  :lang, :task, plus :constraints / :on-infeasible for the decoder."
  [body]
  (let [m (get body "model")]
    {:model (when-not (engine-names m) m)
     :lang (get body "lang")
     :task (get body "task")
     :constraints (get body "constraints")
     :on-infeasible (get body "on_infeasible")
     :thinking (get body "thinking")
     :thought (get body "thought")
     :debias (get body "debias")
     :escalate (get body "escalate")}))

(defn- predict
  "router/predict under the inference lock (or lev.patterns/escalate when
  the body has an escalate object), with the errors the decoder throws
  pointed at the request: a bad constraint at its index among the
  request's own (`offset` of them belong to the workflow)."
  [rt lock state questions {:keys [model lang task constraints on-infeasible thinking thought debias escalate]} offset]
  (try (locking lock
         (if escalate
           (pat/escalate rt state questions
                         (cond-> {:model (get escalate "model") :fast-model model :lang lang :task task
                                  :constraints constraints :on-infeasible on-infeasible
                                  :thought thought :debias debias}
                           (contains? escalate "threshold") (assoc :threshold (get escalate "threshold"))
                           (contains? escalate "thinking") (assoc :thinking (get escalate "thinking"))))
           (router/predict rt state questions :model model :lang lang :task task
                           :constraints constraints :on-infeasible on-infeasible
                           :thinking thinking :thought thought :debias debias)))
       (catch Exception e
         (if (= :invalid-constraint (:type (ex-data e)))
           (throw (ex-info (ex-message e) (assoc (ex-data e) :offset offset)))
           (throw e)))))

;; --- endpoints ---------------------------------------------------------------------

(defn systemone
  "Answer one parsed /v1/systemone body: route, load if needed, infer.
  Inference is serialized on lock."
  [rt lock body]
  (let [details (check-request rt body)]
    (if (seq details)
      (unprocessable details)
      (json-response 200 (predict rt lock (get body "state") (get body "questions") (routing-opts body) 0)))))

(def max-batch-states
  "States one /v1/systemone/batch call may carry."
  256)

(defn- check-states [body]
  (let [states (get body "states")]
    (cond
      (contains? body "state")
      [(detail ["body" "state"] "a batch takes states (a list), not state" "value_error")]
      (not (sequential? states))
      [(detail ["body" "states"] "states is required: a list of states" "value_error")]
      (empty? states)
      [(detail ["body" "states"] "states must not be empty" "value_error")]
      (> (count states) max-batch-states)
      [(detail ["body" "states"] (str "at most " max-batch-states " states a call") "value_error")]
      :else
      (keep-indexed (fn [i s]
                      (when-not (or (string? s) (map? s) (sequential? s))
                        (detail ["body" "states" i] "each state is a string, object or array" "value_error")))
                    states))))

(defn systemone-batch
  "Answer one parsed /v1/systemone/batch body: {\"results\" [...]}, each
  what /v1/systemone answers for that state. Inference is serialized on
  lock."
  [rt lock body]
  (let [details (if-not (map? body)
                  [(detail ["body"] "request body must be a JSON object" "type_error")]
                  (vec (concat (check-states body)
                               (when (contains? body "escalate")
                                 [(detail ["body" "escalate"] "escalate is not available on a batch" "value_error")])
                               (check-routing-fields rt body)
                               (check-constraints body)
                               (check-questions (get body "questions")))))]
    (if (seq details)
      (unprocessable details)
      (let [{:keys [model lang task constraints on-infeasible thinking thought debias]} (routing-opts body)]
        (json-response 200 {"results" (try (locking lock
                                             (router/predict-batch rt (get body "states") (get body "questions")
                                                                   :model model :lang lang :task task
                                                                   :constraints constraints :on-infeasible on-infeasible
                                                                   :thinking thinking :thought thought :debias debias))
                                           (catch Exception e
                                             (if (= :invalid-constraint (:type (ex-data e)))
                                               (throw (ex-info (ex-message e) (assoc (ex-data e) :offset 0)))
                                               (throw e))))})))))

(defn route-only
  "POST /v1/route: the decision for a body, without loading or running."
  [rt body]
  (let [details (if-not (map? body)
                  [(detail ["body"] "request body must be a JSON object" "type_error")]
                  (concat (check-state body)
                          (check-routing-fields rt body)
                          (when (and (contains? body "questions") (not (map? (get body "questions"))))
                            [(detail ["body" "questions"] "questions must be an object" "type_error")])))]
    (if (seq details)
      (unprocessable details)
      (let [{:keys [model lang task]} (routing-opts body)]
        (json-response 200 (router/route rt (get body "state") (get body "questions" {})
                                         :model model :lang lang :task task))))))

(defn run-workflow
  "POST /v1/workflows/:name: the workflow's state fn on \"input\", its
  questions with \"options\", routed and answered; the answer carries the
  workflow name and the state the model actually read."
  [rt lock workflows name body]
  (let [w (get workflows name)]
    (cond
      (nil? w)
      (json-response 404 {"detail" (str "no workflow named " (pr-str name) "; known: "
                                        (str/join ", " (sort (keys workflows))))})

      (not (map? body))
      (unprocessable [(detail ["body"] "request body must be a JSON object" "type_error")])

      (and (contains? body "options") (not (map? (get body "options"))))
      (unprocessable [(detail ["body" "options"] "options must be an object" "type_error")])

      :else
      (let [details (concat (check-routing-fields rt body) (check-constraints body) (check-escalate rt body))]
        (if (seq details)
          (unprocessable details)
          (let [state (wf/state w (get body "input"))
                questions (wf/questions w (get body "options" {}))
                details (concat (check-state {"state" state}) (check-questions questions))]
            (if (seq details)
              (unprocessable details)
              ;; the workflow's constraints first, then the request's; a
              ;; workflow that declares none leaves the answers plain
              ;; unless the request brings some
              (let [own (wf/constraints w (get body "options" {}))
                    theirs (get body "constraints")
                    opts (assoc (routing-opts body)
                                :constraints (when (or own theirs) (vec (concat own theirs))))
                    result (predict rt lock state questions opts (count own))]
                (json-response 200 (assoc result "workflow" name "state" state))))))))))

(defn pattern
  "POST /v1/patterns/:name: confidence-gate, composite-score and
  two-stage-choice over a systemone body (state + questions, or state +
  taxonomy), routed like one."
  [rt lock name body]
  (let [opts (fn [& ks] (into {} (keep (fn [[k key]] (when (contains? body k) [key (get body k)]))
                                        (partition 2 ks))))
        run (fn [details f]
              (if (seq details)
                (unprocessable details)
                (json-response 200 (locking lock (f)))))
        routing (fn [] (let [{:keys [model lang task]} (routing-opts body)] {:model model :lang lang :task task}))]
    (case name
      "confidence-gate"
      (run (concat (check-request rt body)
                   (when (contains? body "threshold")
                     (try (pat/thresholds (get body "threshold")) nil
                          (catch Exception ex [(detail ["body" "threshold"] (ex-message ex) "value_error")]))))
           #(pat/confidence-gate rt (get body "state") (get body "questions")
                                 (merge (routing) (opts "threshold" :threshold))))
      "composite-score"
      (run (concat (check-request rt body)
                   (when (and (contains? body "weights") (not (map? (get body "weights"))))
                     [(detail ["body" "weights"] "weights must be an object of question id -> number" "type_error")])
                   (when (and (contains? body "normalize") (not (boolean? (get body "normalize"))))
                     [(detail ["body" "normalize"] "normalize must be true or false" "type_error")]))
           #(pat/composite-score rt (get body "state") (get body "questions")
                                 (merge (routing) (opts "weights" :weights "normalize" :normalize))))
      "two-stage-choice"
      (run (if-not (map? body)
             [(detail ["body"] "request body must be a JSON object" "type_error")]
             (concat (check-state body)
                     (check-routing-fields rt body)
                     (let [t (get body "taxonomy")]
                       (when-not (and (map? t) (seq t) (every? (fn [[_ v]] (and (map? v) (seq v))) t))
                         [(detail ["body" "taxonomy"] "taxonomy is required: {category: {option: description}}, no empty levels" "value_error")]))))
           #(pat/two-stage-choice rt (get body "state") (get body "taxonomy")
                                  (merge (routing) (opts "instructions_category" :instructions-category
                                                         "instructions_option" :instructions-option))))
      not-found)))

(defn- health [rt workflows]
  (json-response 200 (seq/ordered-map [["status" "ok"] ["model" engine-name]
                                       ["loaded" (router/loaded rt)]
                                       ["thinkers" (router/loaded-thinkers rt)]
                                       ["workflows" (vec (sort (keys workflows)))]])))

(defn- models [rt]
  (let [loaded (set (router/loaded rt))
        loaded-thinkers (set (router/loaded-thinkers rt))]
    (json-response 200 (seq/ordered-map
                        [["default" (:default rt)]
                         ["max_loaded" (:max-loaded rt)]
                         ["thinkers" (seq/ordered-map
                                      (for [name (router/thinker-names rt)
                                            :let [cfg (get (:thinkers rt) name)]]
                                        [name (seq/ordered-map [["model" (:model cfg)]
                                                                ["available" (router/available? rt name)]
                                                                ["loaded" (contains? loaded-thinkers name)]
                                                                ["thinking" (boolean (get cfg :thinking true))]])]))]
                         ["models" (seq/ordered-map
                                    (for [[name dir] (:models rt)]
                                      [name (seq/ordered-map [["repo" (get router/repos name)]
                                                              ["data" dir]
                                                              ["available" (router/available? rt name)]
                                                              ["loaded" (contains? loaded name)]
                                                              ["limits" (when-let [l (router/effective-limits rt name)]
                                                                          (seq/ordered-map [["max_len" (:max-len l)]
                                                                                            ["head_max_len" (:head-max-len l)]]))]])]))]]))))

(defn- list-workflows [workflows]
  (json-response 200 {"workflows"
                      (seq/ordered-map
                       (for [[name w] (sort-by key workflows)]
                         [name (seq/ordered-map [["description" (:doc w)]
                                                 ["file" (:file w)]
                                                 ["questions" (vec (map #(if (keyword? %) (clojure.core/name %) (str %))
                                                                        (keys (wf/questions w))))]
                                                 ["constraints" (vec (wf/constraints w))]
                                                 ["options" (:options? w)]])]))}))

;; --- the handler --------------------------------------------------------------------

(defn- authorized? [req api-key]
  (or (nil? api-key)
      (= (get-in req [:headers "authorization"]) (str "Bearer " api-key))))

(def ^:private http-methods [:get :post :put :delete :patch :head :options])

(defn- routes
  "The ruuter route table: the real routes, a 405 entry for every other
  method on each of their paths, and the 404. ruuter 2 matches by
  specificity and method, so the order of the vector carries nothing."
  [rt lock workflows api-key]
  (let [guard (fn [f] (fn [req] (if (authorized? req api-key) (f req) unauthorized)))
        json-in (fn [f allow-empty?]
                  (fn [req] (try (f (parse-body req allow-empty?))
                                 (catch Exception e (error-response e)))))
        real [{:path "/health" :method :get :response (fn [_] (health rt workflows))}
              {:path "/v1/systemone" :method :post
               :response (guard (json-in #(systemone rt lock %) false))}
              {:path "/v1/systemone/batch" :method :post
               :response (guard (json-in #(systemone-batch rt lock %) false))}
              {:path "/v1/route" :method :post
               :response (guard (json-in #(route-only rt %) false))}
              {:path "/v1/models" :method :get :response (guard (fn [_] (models rt)))}
              {:path "/v1/workflows" :method :get :response (guard (fn [_] (list-workflows workflows)))}
              {:path "/v1/workflows/:name" :method :post
               :response (guard (fn [req]
                                  (try (run-workflow rt lock workflows (get-in req [:params :name])
                                                     (parse-body req true))
                                       (catch Exception e (error-response e)))))}
              {:path "/v1/patterns/:name" :method :post
               :response (guard (fn [req]
                                  (try (pattern rt lock (get-in req [:params :name]) (parse-body req false))
                                       (catch Exception e (error-response e)))))}]
        disallowed (for [{:keys [path method]} real
                         m http-methods :when (not= m method)]
                     {:path path :method m :response method-not-allowed})]
    (vec (concat real disallowed [{:path :not-found :response not-found}]))))

(defn handler
  "A ring handler for the API. `rt` is a lev.router router, or one loaded
  agent (served as the english checkpoint). opts: :api-key (nil = no auth),
  :workflows {name workflow} from lev.workflows/load-workflows."
  [rt {:keys [api-key workflows]}]
  (let [rt (if (router/router? rt) rt (router/preloaded rt))
        ;; compiled once: handed the raw vector, ruuter/route would look its
        ;; trie up in a memoize cache on every request (hashing the table)
        ;; and keep one trie per handler ever made
        table (ruuter/compile-routes (routes rt (Object.) (or workflows {}) api-key))]
    (fn [req] (ruuter/route table req))))

;; --- server ----------------------------------------------------------------------

(defn start
  "Run the API on ring-chez-adapter; answers the server handle.
  opts: :port (8080), :host (\"127.0.0.1\"; \"0.0.0.0\" for all interfaces),
  :api-key, :workflows, :max-request-bytes (4 MiB), plus anything the
  adapter takes."
  [rt {:keys [port host max-request-bytes]
       :or {port 8080 host "127.0.0.1" max-request-bytes (* 4 1024 1024)}
       :as opts}]
  (adapter/run-server
   (handler rt opts)
   (merge {:on-failure (fn [_ ex]
                         (binding [*out* *err*] (println "lev.server:" (ex-message ex)))
                         (json-response 500 {"detail" "Internal Server Error"}))}
          (dissoc opts :api-key :workflows)
          {:port port :host host :max-request-bytes max-request-bytes})))

(defn stop [server]
  (adapter/stop-server server))

(defn self-test
  "What a built binary proves before it ships. With the english encoder
  prepared: the README quickstart through the handler against the
  json.dumps(Agent.system_one(...)) pinned in <golden-dir>/readme.edn,
  plus the tokenizer paths a release build has miscompiled before. With
  a thinker available: one question through it without thinking, the
  proof that llama.cpp is linked in and runs. Answers [[name ok? detail]
  ...]; a setup without one kind of model skips that kind's rows. The
  test suite runs interpreted; this is what checks the AOT binary."
  [agent golden-dir]
  (let [rt (if (router/router? agent) agent (router/preloaded agent))
        golden (fn [name] (edn/read-string {:readers {'laya/omap seq/ordered-map}}
                                           (slurp (str golden-dir "/" name ".edn"))))
        cases (golden "cases")
        encoder-rows
        (when (router/available? rt "english")
          (let [agent (router/load-model rt "english")
                h (handler rt {})
                want (:system-one (golden "readme"))
                body (seq/json-str (seq/ordered-map [["state" (:readme-state cases)]
                                                     ["model" "english"]
                                                     ["questions" (:readme-questions cases)]]))
                resp (h {:request-method :post :uri "/v1/systemone" :headers {} :body body})
                ;; the python JSON plus a routing key; probabilities are compared to
                ;; one unit in the fourth decimal, since a value on a rounding
                ;; boundary can land either side under a different BLAS
                got (json/read-str (str/trim (:body resp)))
                approx (fn approx [a b]
                         (cond (and (number? a) (number? b)) (<= (Math/abs (- (double a) (double b))) 1.0001e-4)
                               (and (map? a) (map? b)) (and (= (set (keys a)) (set (keys b)))
                                                            (every? (fn [[k v]] (approx v (get b k))) a))
                               (and (sequential? a) (sequential? b)) (and (= (count a) (count b))
                                                                          (every? true? (map approx a b)))
                               :else (= a b)))
                ;; the golden carries the Python package's own model name; lev
                ;; reports the checkpoint's
                answer-ok (and (approx (dissoc (json/read-str want) "model") (dissoc got "routing" "model" "truncated"))
                               (= "english" (get got "model"))
                               (= "english" (get-in got ["routing" "model"])))
                tok (:tok agent)
                tok-golden (:cases (golden "tok"))
                tok-ok (every? (fn [[i c]] (= (get tok-golden (str i)) (tk/encode tok c)))
                               (map-indexed vector (:tok-cases cases)))
                nfc-ok (= (apply str (repeat 300 "क़"))
                          (tk/nfc (apply str (repeat 300 "क़"))))]
            [["README quickstart answer matches the Python engine" answer-ok
              (when-not answer-ok (str "got " (seq/json-str got)))]
             ["tokenizer reproduces golden/tok.edn" tok-ok nil]
             ["NFC expansion path" nfc-ok nil]]))
        thinker-row
        (when-let [name (first (filter #(router/available? rt %) (router/thinker-names rt)))]
          (let [[ok detail] (try (let [out (router/predict rt (:readme-state cases)
                                                           (select-keys (:readme-questions cases) ["is_phishing"])
                                                           :model name :thinking false)]
                                   [(contains? #{true false} (< 0.5 (get-in out ["answers" "is_phishing" "noul"])))
                                    (str name ": " (seq/json-str (get-in out ["answers" "is_phishing"])))])
                                 (catch Exception e [false (ex-message e)]))]
            [(str "the thinker answers (" (lev.llm/version) ")") ok detail]))]
    (vec (concat encoder-rows (when thinker-row [thinker-row])))))

(defn -main
  "jolt -M:serve [--data DIR] [--port N] [--host ADDR] [--api-key KEY]
                 [--max-loaded N] [--default-model NAME] [--workflows DIR[:DIR]]
                 [--max-len N] [--head-max-len N] [--thinker PATH.gguf] [--max-thinkers N] [--calibration FILE]
   Each falls back to an environment variable (LEV_DATA, PORT, LEV_HOST,
   LEV_API_KEY, LEV_MAX_LOADED, LEV_DEFAULT_MODEL, LEV_WORKFLOWS,
   LEV_MAX_LEN, LEV_HEAD_MAX_LEN, LEV_THINKER, LEV_MAX_THINKERS), then to
   ~/.config/lev/config.edn (:data :port :host :api-key :max-loaded
   :default-model :auto-task-detection :workflow-dirs :max-len
   :head-max-len :checkpoints :thinkers :max-thinkers), then to a default.
   DIR is the data root jolt prepare writes: DIR/ (english),
   DIR/multilingual, DIR/typed-decisions; the default checkpoint is loaded
   at startup, the others and the thinkers on first use.
   --self-test [--golden DIR]: load, verify against golden/, exit 0 or 1.
   --mcp: serve the same models as MCP tools on stdin/stdout (lev.mcp)
   instead of HTTP."
  [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        arg (fn [flag env key default] (cfg/setting ctx flag env key default))
        data-dir (arg "--data" "LEV_DATA" :data "data")
        rt (router/make-router {:data data-dir
                                :models (cfg/encoders ctx)
                                :max-loaded (Long/parseLong (str (arg "--max-loaded" "LEV_MAX_LOADED" :max-loaded "1")))
                                :max-thinkers (Long/parseLong (str (arg "--max-thinkers" "LEV_MAX_THINKERS" :max-thinkers "1")))
                                :thinkers (cfg/thinkers ctx)
                                :default (arg "--default-model" "LEV_DEFAULT_MODEL" :default-model "english")
                                :auto-task-detection (or (true? (get opts "--auto-task-detection"))
                                                         (true? (:auto-task-detection (:config ctx))))
                                ;; cfg/limits resolves CLI > env > config per name; the
                                ;; router keeps the per-checkpoint results
                                :checkpoints (into {} (map (fn [n] [n (cfg/limits ctx n)])) router/names)
                                :calibrations (cfg/calibrations ctx)})
        log (fn [& xs] (binding [*out* *err*] (apply println "lev:" xs)))
        t0 (System/nanoTime)
        every-model (concat router/names (router/thinker-names rt))
        _ (when-not (some #(router/available? rt %) every-model)
            (log "no model is available: no prepared encoder under" data-dir "(jolt prepare) and no thinker"
                 "(config.edn :thinkers or --thinker PATH.gguf)")
            (System/exit 1))
        ;; the default is loaded at startup when it is there; a setup with only
        ;; the other kind of model still serves (a request for the default is a 503)
        _ (if (router/available? rt (:default rt))
            (do (log "loading" (:default rt)
                     (if (router/thinker? rt (:default rt))
                       (str "(" (:model (get (:thinkers rt) (:default rt))) ")")
                       (str "from " (get (:models rt) (:default rt)))) "...")
                (let [agent (router/load-model rt (:default rt))]
                  (if (router/thinker? rt (:default rt))
                    (log (format "thinker %s loaded in %.1fs" (:default rt) (/ (- (System/nanoTime) t0) 1e9)))
                    (log (format "%d tensors loaded on %s in %.1fs; max_len %d%s, head_max_len %d"
                                 (count (:tensors (:manifest agent)))
                                 (str (name (:backend agent :cpu)) " " (name (:dtype agent :f32)))
                                 (/ (- (System/nanoTime) t0) 1e9)
                                 (:max-len (:cfg agent))
                                 (if (not= (:max-len (:cfg agent)) (:trained-max-len agent))
                                   (str " (trained " (:trained-max-len agent) ")") "")
                                 (:head-max-len (:cfg agent)))))))
            (log "default model" (:default rt) "is not available; the others are served"))]
    (cond
      (get opts "--self-test")
      (let [results (self-test rt (arg "--golden" "LEV_GOLDEN" :golden "golden"))]
        (doseq [[name ok? detail] results]
          (println (if ok? "ok  " "FAIL") name (or detail "")))
        (System/exit (if (every? second results) 0 1)))

      ;; the same models and workflows as MCP tools over stdio (lev.mcp);
      ;; everything but the protocol goes to stderr
      (get opts "--mcp")
      (let [workflows (wf/load-workflows (cfg/workflow-dirs ctx))]
        (log "serving MCP on stdio; workflows:" (if (seq workflows) (str/join ", " (sort (keys workflows))) "none"))
        (mcp/serve (handler rt {:workflows workflows}) {:workflows workflows}))

      :else
      (let [workflows (wf/load-workflows (cfg/workflow-dirs ctx))
            port (Long/parseLong (str (arg "--port" "PORT" :port "8080")))
            host (arg "--host" "LEV_HOST" :host "127.0.0.1")
            api-key (arg "--api-key" "LEV_API_KEY" :api-key nil)
            server (start rt {:port port :host host :api-key api-key :workflows workflows})]
        (log "workflows:" (if (seq workflows) (str/join ", " (sort (keys workflows))) "none"))
        (log "checkpoints:" (str/join ", " (for [n router/names]
                                             (str n (if (router/available? rt n) "" " (not prepared)")))))
        (log "thinkers:" (if (seq (router/thinker-names rt))
                           (str/join ", " (for [n (router/thinker-names rt)]
                                            (str n " (" (:model (get (:thinkers rt) n))
                                                 (if (router/available? rt n) ")" "; not available)"))))
                           "none (config.edn :thinkers or --thinker PATH.gguf)"))
        (log (format "listening on http://%s:%d (auth %s)" host (:port server) (if api-key "on" "off")))
        @(promise)))))
