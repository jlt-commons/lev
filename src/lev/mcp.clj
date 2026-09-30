(ns lev.mcp
  "The Model Context Protocol over stdio: lev's models as tools an agent
  (Claude Code, Claude Desktop, Cursor, ...) calls for a typed decision
  in milliseconds, with no API key and nothing leaving the machine.

    ./lev-server --mcp [the server's flags]        (or jolt -M:mcp)
    claude mcp add lev -- /path/to/lev-server --mcp

  A thin layer over the HTTP API's ring handler, called in process: the
  `decide` tool answers exactly what POST /v1/systemone answers, errors
  included (as a tool error carrying the 422's body). Tools: decide,
  run_workflow, list_models, list_workflows. Resources: lev://models,
  lev://workflows and lev://workflows/<name> (a workflow's questions and
  constraints, to reuse or adapt).

  JSON-RPC 2.0, one message a line on stdin/stdout; logs go to stderr.
  Messages are read with lev.json, so a request's key order (option and
  question order are model input) survives to the handler."
  (:require [clojure.string :as str]
            [lev.json :as json]
            [lev.sequence :as seq]
            [lev.workflows :as wf]))

(def protocol-versions
  "The MCP revisions this server speaks, newest first: an initialize asking
  for one of them gets it, anything else the newest."
  ["2025-11-25" "2025-06-18" "2025-03-26" "2024-11-05"])

(def instructions
  "Lev answers typed questions about a state (a text or any JSON) with calibrated probabilities. Call `decide` whenever you need a judgement you would otherwise ask an LLM for in prose: classify (choice), rate on a scale (score) or check a yes/no statement (noul). The encoders answer in about 100 ms on a CPU, so it is cheap enough to run before every routing, triage, moderation or escalation step. Act on high confidence; escalate (a thinker model, `escalate`) or ask when it is low. `list_models` says what is configured; `list_workflows` and the lev://workflows resources show ready-made question sets.")

(def ^:private question-schema
  {"type" "object"
   "description" "Typed questions by id; the order is kept and is model input."
   "additionalProperties"
   {"type" "object"
    "properties" {"type" {"type" "string" "enum" ["choice" "score" "noul"]}
                  "instructions" {"description" "The question, a string (or any JSON)."}
                  "criteria" {"description" "choice: {option: description or null} or [option, ...]; score: [level description, ...] from 0 up; noul: optional {\"true\": description, \"false\": description}."}}
    "required" ["type" "instructions"]}})

(def tools
  [{"name" "decide"
    "title" "Decide typed questions"
    "description" "Answer typed questions about a state in one call: POST /v1/systemone. Each answer has probabilities and a confidence in [0, 1]; a choice names its option, a score its expected level, a noul p(true)."
    "inputSchema" {"type" "object"
                   "properties" {"state" {"description" "What the questions are about: a string or any JSON."}
                                 "questions" question-schema
                                 "model" {"type" "string" "description" "An encoder (english, multilingual, typed-decisions) or a configured thinker; absent routes by content."}
                                 "constraints" {"type" "array" "description" "lev.constraints over the question ids, decided jointly."}
                                 "thinking" {"type" "boolean" "description" "A thinker only: think before answering (slower)."}
                                 "escalate" {"type" "object" "description" "{\"model\": thinker, \"threshold\"?: 0.8}: re-ask the answers below the threshold on the thinker."}}
                   "required" ["state" "questions"]}}
   {"name" "run_workflow"
    "title" "Run a workflow"
    "description" "Run a named workflow (a ready-made question set, see list_workflows) on its input: POST /v1/workflows/<name>."
    "inputSchema" {"type" "object"
                   "properties" {"name" {"type" "string"}
                                 "input" {"description" "The workflow's input, as its description says."}
                                 "options" {"type" "object"}
                                 "model" {"type" "string"}}
                   "required" ["name" "input"]}}
   {"name" "list_models"
    "title" "List models"
    "description" "The encoders and thinkers this server has: available, loaded, the default."
    "inputSchema" {"type" "object" "properties" {}}}
   {"name" "list_workflows"
    "title" "List workflows"
    "description" "The loaded workflows: description, question ids, constraints."
    "inputSchema" {"type" "object" "properties" {}}}])

(defn- call-handler
  "The handler's answer to one request: [status parsed-body]."
  [handler method uri body]
  (let [resp (handler {:request-method method :uri uri :headers {}
                       :body (when (some? body) (seq/json-str body))})
        text (str/trim (str (:body resp)))]
    [(:status resp) (if (str/blank? text) nil (json/read-str text))]))

(defn- tool-result [[status body]]
  (let [ok? (= 200 status)]
    (seq/ordered-map
     (concat [["content" [(array-map "type" "text" "text" (seq/json-str body))]]]
             (when (and ok? (map? body)) [["structuredContent" body]])
             [["isError" (not ok?)]]))))

(defn- dissoc-keys [m ks] (seq/ordered-map (remove (fn [[k _]] (contains? ks k)) m)))

(defn call-tool
  "A tools/call's result for tool `tool` with `args`, through the handler."
  [handler tool args]
  (let [args (or args {})]
    (case tool
      "decide" (tool-result (call-handler handler :post "/v1/systemone" args))
      "run_workflow" (let [n (get args "name")]
                       (if (and (string? n) (re-matches #"[A-Za-z0-9_.-]+" n))
                         (tool-result (call-handler handler :post (str "/v1/workflows/" n) (dissoc-keys args #{"name"})))
                         (tool-result [422 {"detail" "name must be a workflow name (list_workflows)"}])))
      "list_models" (tool-result (call-handler handler :get "/v1/models" nil))
      "list_workflows" (tool-result (call-handler handler :get "/v1/workflows" nil))
      (throw (ex-info (str "unknown tool " (pr-str tool)) {:code -32602})))))

(defn- workflow-resource [name w]
  (seq/ordered-map [["name" name] ["description" (:doc w)]
                    ["questions" (wf/questions w)]
                    ["constraints" (vec (wf/constraints w))]]))

(defn resources [workflows]
  (vec (concat [(array-map "uri" "lev://models" "name" "models" "mimeType" "application/json"
                           "description" "The configured encoders and thinkers")
                (array-map "uri" "lev://workflows" "name" "workflows" "mimeType" "application/json"
                           "description" "The loaded workflows")]
               (for [[name w] (sort-by key workflows)]
                 (array-map "uri" (str "lev://workflows/" name) "name" (str "workflow " name)
                            "mimeType" "application/json" "description" (or (:doc w) name))))))

(defn read-resource [handler workflows uri]
  (let [body (cond
               (= uri "lev://models") (second (call-handler handler :get "/v1/models" nil))
               (= uri "lev://workflows") (second (call-handler handler :get "/v1/workflows" nil))
               (str/starts-with? uri "lev://workflows/")
               (let [n (subs uri (count "lev://workflows/"))]
                 (if-let [w (get workflows n)]
                   (workflow-resource n w)
                   (throw (ex-info (str "no workflow " (pr-str n)) {:code -32002}))))
               :else (throw (ex-info (str "unknown resource " (pr-str uri)) {:code -32002})))]
    {"contents" [(array-map "uri" uri "mimeType" "application/json" "text" (seq/json-str body))]}))

(defn handle
  "One JSON-RPC message -> the response to write, or nil for a
  notification. opts: :workflows, :version (the server's)."
  [handler {:keys [workflows version]} msg]
  (let [id (get msg "id")
        method (get msg "method")
        params (get msg "params")
        reply (fn [result] (array-map "jsonrpc" "2.0" "id" id "result" result))
        error (fn [code message] (array-map "jsonrpc" "2.0" "id" id "error" (array-map "code" code "message" message)))]
    (cond
      (not (map? msg)) (array-map "jsonrpc" "2.0" "id" nil "error" (array-map "code" -32600 "message" "invalid request"))
      ;; notifications (initialized, cancelled) and the client's answers to
      ;; requests this server never makes
      (or (not (contains? msg "id")) (not (contains? msg "method"))) nil
      :else
      (try
        (case method
          "initialize" (let [asked (get params "protocolVersion")]
                         (reply (seq/ordered-map
                                 [["protocolVersion" (if (some #{asked} protocol-versions) asked (first protocol-versions))]
                                  ["capabilities" (array-map "tools" {} "resources" {})]
                                  ["serverInfo" (array-map "name" "lev" "version" (or version "dev"))]
                                  ["instructions" instructions]])))
          "ping" (reply {})
          "tools/list" (reply {"tools" tools})
          "tools/call" (reply (call-tool handler (get params "name") (get params "arguments")))
          "resources/list" (reply {"resources" (resources workflows)})
          "resources/read" (reply (read-resource handler workflows (get params "uri")))
          "resources/templates/list" (reply {"resourceTemplates" []})
          "prompts/list" (reply {"prompts" []})
          (error -32601 (str "method not found: " method)))
        (catch Exception e
          (error (or (:code (ex-data e)) -32603) (or (ex-message e) (str e))))))))

(defn serve
  "Read JSON-RPC messages from *in* a line at a time until it closes,
  answering each on *out*."
  [handler opts]
  (loop []
    (when-let [line (read-line)]
      (when-not (str/blank? line)
        (let [resp (try (handle handler opts (json/read-str line))
                        (catch Exception e
                          (array-map "jsonrpc" "2.0" "id" nil
                                     "error" (array-map "code" -32700 "message" (str "parse error: " (ex-message e))))))]
          (when resp
            (println (seq/json-str resp))
            (flush))))
      (recur))))
