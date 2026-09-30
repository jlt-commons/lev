(ns lev.mcp-test
  "lev.mcp: the JSON-RPC handshake and methods, and that `decide` is
  POST /v1/systemone through the same handler (a fake thinker behind a
  real lev.server handler, so no model is loaded)."
  (:require [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [lev.json :as json]
            [lev.mcp :as mcp]
            [lev.router :as router]
            [lev.sequence :as seq]
            [lev.server :as server]
            [lev.think :as think]))

(defn- fake-router []
  (router/make-router
   {:thinkers {"fake" {:model "x.gguf" :thinking false}}
    :default "fake"
    :loader (fn [& _] (throw (ex-info "no encoders here" {:type :model-unavailable})))
    :thinker-loader (fn [name cfg]
                      (think/thinker (assoc cfg :name name :jev false)
                                     {:decide (fn [_ options _] {:logp (vec (map-indexed (fn [i _] (- i)) options)) :thought "" :tokens 0})
                                      :count-tokens (constantly 3)}))}))

(def workflows
  {"tiny" {:doc "A one-question workflow." :file "tiny.clj"
           :questions (fn ([] {"angry" {"type" "noul" "instructions" "Is the customer angry?"}}) ([_] {"angry" {"type" "noul" "instructions" "Is the customer angry?"}}))}})

(defn- rpc [h msg] (mcp/handle h {:workflows {}} msg))

(deftest the-handshake-and-the-listings
  (let [h (fn [_] {:status 200 :body "{}"})]
    (testing "initialize answers the revision asked for when it knows it, else its newest"
      (let [r (rpc h {"jsonrpc" "2.0" "id" 1 "method" "initialize" "params" {"protocolVersion" "2025-06-18"}})]
        (is (= 1 (get r "id")))
        (is (= "2025-06-18" (get-in r ["result" "protocolVersion"])))
        (is (= "lev" (get-in r ["result" "serverInfo" "name"])))
        (is (contains? (get-in r ["result" "capabilities"]) "tools")))
      (is (= (first mcp/protocol-versions)
             (get-in (rpc h {"jsonrpc" "2.0" "id" 2 "method" "initialize" "params" {"protocolVersion" "1999-01-01"}})
                     ["result" "protocolVersion"]))))
    (testing "notifications and client responses get no answer"
      (is (nil? (rpc h {"jsonrpc" "2.0" "method" "notifications/initialized"})))
      (is (nil? (rpc h {"jsonrpc" "2.0" "id" 9 "result" {}}))))
    (testing "the tools, each with an input schema"
      (let [tools (get-in (rpc h {"jsonrpc" "2.0" "id" 3 "method" "tools/list"}) ["result" "tools"])]
        (is (= ["decide" "run_workflow" "list_models" "list_workflows"] (map #(get % "name") tools)))
        (is (every? #(= "object" (get-in % ["inputSchema" "type"])) tools))))
    (testing "an unknown method is -32601, an unknown tool -32602"
      (is (= -32601 (get-in (rpc h {"jsonrpc" "2.0" "id" 4 "method" "nope"}) ["error" "code"])))
      (is (= -32602 (get-in (rpc h {"jsonrpc" "2.0" "id" 5 "method" "tools/call" "params" {"name" "nope"}}) ["error" "code"]))))))

(deftest decide-is-systemone
  (let [h (server/handler (fake-router) {:workflows workflows})
        body (seq/ordered-map [["state" "Refund me or I leave."]
                               ["questions" (seq/ordered-map [["team" {"type" "choice" "instructions" "Which team?"
                                                                       "criteria" ["billing" "support" "other"]}]
                                                              ["leaving" {"type" "noul" "instructions" "Will they leave?"}]])]
                               ["model" "fake"]])
        http (json/read-str (str/trim (:body (h {:request-method :post :uri "/v1/systemone" :headers {} :body (seq/json-str body)}))))
        r (rpc h {"jsonrpc" "2.0" "id" 7 "method" "tools/call" "params" {"name" "decide" "arguments" body}})
        result (get r "result")]
    (is (false? (get result "isError")))
    (is (= http (get result "structuredContent")))
    (is (= http (json/read-str (get-in result ["content" 0 "text"]))))
    (is (= ["team" "leaving"] (keys (get http "answers"))) "question order survives")
    (testing "a bad request is a tool error carrying the 422's body"
      (let [bad (get (rpc h {"jsonrpc" "2.0" "id" 8 "method" "tools/call"
                             "params" {"name" "decide" "arguments" {"state" "x" "questions" {"q" {"type" "maybe" "instructions" "?"}}}}})
                     "result")]
        (is (true? (get bad "isError")))
        (is (contains? (json/read-str (get-in bad ["content" 0 "text"])) "detail"))))))

(deftest workflows-are-tools-and-resources
  (let [h (server/handler (fake-router) {:workflows workflows})
        call (fn [id method params] (mcp/handle h {:workflows workflows} {"jsonrpc" "2.0" "id" id "method" method "params" params}))]
    (is (= ["lev://models" "lev://workflows" "lev://workflows/tiny"]
           (map #(get % "uri") (get-in (call 1 "resources/list" nil) ["result" "resources"]))))
    (let [text (get-in (call 2 "resources/read" {"uri" "lev://workflows/tiny"}) ["result" "contents" 0 "text"])]
      (is (= ["angry"] (keys (get (json/read-str text) "questions")))))
    (is (= -32002 (get-in (call 3 "resources/read" {"uri" "lev://workflows/none"}) ["error" "code"])))
    (is (= ["tiny"] (keys (get-in (call 4 "tools/call" {"name" "list_workflows"}) ["result" "structuredContent" "workflows"]))))
    (is (true? (get-in (call 5 "tools/call" {"name" "run_workflow" "arguments" {"name" "../etc" "input" "x"}}) ["result" "isError"])))))
