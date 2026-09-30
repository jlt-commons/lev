(ns lev.router
  "Route a request to the encoder checkpoint best suited to it (the port of
  the checkpoints' Python router; decisions pinned by golden/router.edn).

  Three checkpoints, all in the Hub repo convaiinnovations/laya:

    english          root                ModernBERT-large 421M, 512 tokens
    multilingual     multilingual/       mmBERT-base 322M, 1024 tokens, 100+ languages
    typed-decisions  typed-decisions/    ModernBERT-large 421M, 1024 tokens, fine-tuned
                                         on the four typed-decisions workflows

  The English checkpoint does not degrade gracefully off English, it
  collapses while staying confident, so script detection (lev.lang) is the
  primary routing signal. typed-decisions is never picked automatically
  unless :auto-task-detection is on or task=\"typed_decisions\" is passed.

  Precedence: explicit model > explicit task > detected workflow (opt-in) >
  explicit lang > detected script/language > default.

  A router holds prepared data directories per checkpoint ({name dir}, by
  default data/, data/multilingual, data/typed-decisions), loads an agent on
  first use and keeps :max-loaded of them resident, evicting the least
  recently used: all three together are ~4.6 GB of f32.

  It also holds the thinkers (:thinkers {name lev.think config}, from
  config.edn): generative models that answer the same questions slowly
  and much more accurately (lev.think). A thinker is a model name like a
  checkpoint's, chosen explicitly only (never by content), loaded on first
  use into its own slot (:max-thinkers, default 1: they are GBs each)."
  (:require [clojure.edn]
            [clojure.string :as str]
            [lev.agent :as ag]
            [lev.lang :as lang]
            [lev.llm :as llm]
            [lev.mlx :as mlx]
            [lev.sequence :as seq]
            [lev.think :as think]))

(def bundle-repo "convaiinnovations/laya")

(def repos
  "Checkpoint name -> where it lives on the Hub (repo or repo/subfolder)."
  (seq/ordered-map [["english" bundle-repo]
                    ["multilingual" (str bundle-repo "/multilingual")]
                    ["typed-decisions" (str bundle-repo "/typed-decisions")]]))

(def names ["english" "multilingual" "typed-decisions"])

(def aliases
  "Spellings people are likely to type."
  {"en" "english" "default" "english"
   "multi" "multilingual" "ml" "multilingual"
   "typed" "typed-decisions" "typed_decisions" "typed-decisions" "decisions" "typed-decisions"})

(def typed-decisions-workflows
  "Question-id signatures of the four typed-decisions workflows, used only
  when :auto-task-detection is on."
  {"agent_trace_observability" #{"action" "needs_review" "outcome" "risk" "urgency"}
   "customer_service" #{"action" "category" "churn_risk" "needs_human" "urgency"}
   "invoice_processing" #{"discrepancy_severity" "disposition" "duplicate" "matches_order" "urgency"}
   "security_incidents" #{"credential_compromise" "disposition" "severity" "true_positive" "urgency"}})

(defn default-models
  "The bundle layout jolt prepare writes under one data root."
  [data-root]
  (seq/ordered-map [["english" data-root]
                    ["multilingual" (str data-root "/multilingual")]
                    ["typed-decisions" (str data-root "/typed-decisions")]]))

(defn normalise-name
  "Canonical checkpoint name for a name or alias, case- and space-insensitive;
  with a router, one of its thinkers' names too. Throws {:type
  :unknown-model} otherwise."
  ([name] (normalise-name nil name))
  ([router name]
   (let [key (str/lower-case (str/trim (str name)))
         key (get aliases key key)
         thinkers (when router (sort (keys (:thinkers router))))]
     (when-not (or (contains? repos key) (some #(= key (str/lower-case %)) thinkers))
       (throw (ex-info (str "unknown model " (pr-str name) "; choose one of " (str/join ", " (concat names thinkers))
                           " (or an alias: " (str/join ", " (sort (keys aliases))) ")")
                       {:type :unknown-model :model name :known (vec (concat names thinkers))
                        :aliases (sort (keys aliases))})))
     (if (contains? repos key)
       key
       (some #(when (= key (str/lower-case %)) %) thinkers)))))

(defn match-typed-decisions-workflow
  "Name of the typed-decisions workflow whose question ids these are exactly,
  else nil. A schema that merely contains \"urgency\" is never captured."
  [questions]
  (let [ids (set (map #(if (keyword? %) (name %) (str %)) (keys (or questions {}))))]
    (some (fn [[wf sig]] (when (= ids sig) wf)) (sort typed-decisions-workflows))))

;; --- the router -----------------------------------------------------------------------

(defn load-prepared
  "The default loader: lev.agent/load-agent on a prepared data directory
  (or lev.mlx/load-agent under :backend :mlx, at :dtype :f32 or :f16),
  with the sequence limits configured for that checkpoint and, under
  :calibration, a lev.calibrate file applied over the checkpoint's
  temperatures. Throws {:type :model-unavailable} when there is nothing
  there, or when the mlx native is not built."
  ([name dir] (load-prepared name dir nil))
  ([name dir limits]
   (when-not (and dir (.exists (clojure.java.io/file dir "manifest.edn")))
     (throw (ex-info (str "checkpoint " name " is not prepared (no " dir "/manifest.edn); "
                          "run jolt prepare for it or point :models at its data directory")
                     {:type :model-unavailable :model name :dir dir})))
   (let [opts (assoc (dissoc limits :calibration) :name name)
         agent (case (:backend limits :cpu)
                 :mlx (mlx/load-agent dir opts)
                 (ag/load-agent dir opts))]
     (if-let [path (:calibration limits)]
       (do (when-not (.exists (clojure.java.io/file path))
             (throw (ex-info (str "calibration file " path " for " name " does not exist") {:type :model-unavailable :model name :file path})))
           (ag/with-calibration agent (clojure.edn/read-string (slurp path))))
       agent))))

(defn make-router
  "Options: :models {name data-dir} (default (default-models \"data\")),
  :data root for default-models, :max-loaded (default 1), :default
  checkpoint (\"english\"), :auto-task-detection (false), :limits
  {:max-len :head-max-len} for every checkpoint and :checkpoints {name
  {...}} per checkpoint (lev.config/limits builds these from config.edn),
  :calibrations {name path} (lev.calibrate files, lev.config/calibrations),
  :loader (fn [name dir limits] agent) for tests (default load-prepared);
  :thinkers {name lev.think config} (lev.config/thinkers), :max-thinkers
  (default 1), :thinker-loader (fn [name cfg] agent) for tests (default
  lev.think/thinker)."
  [{:keys [models data max-loaded default auto-task-detection loader limits checkpoints calibrations
           thinkers max-thinkers thinker-loader]
    :or {max-loaded 1 default "english" auto-task-detection false max-thinkers 1}}]
  (let [thinkers (into {} (map (fn [[k v]] [(if (keyword? k) (name k) (str k)) v])) thinkers)
        named-thinkers {:thinkers thinkers}]
    {:models (into (default-models (or data "data"))
                   (map (fn [[k v]] [(normalise-name k) v])) models)
     :max-loaded (max 1 (long max-loaded))
     ;; the default may be a thinker: resolved once the thinkers are known
     :default (normalise-name named-thinkers default)
     :auto-task-detection (boolean auto-task-detection)
     :limits (or limits {})
     :checkpoints (into {} (map (fn [[k v]] [(normalise-name k) v])) checkpoints)
     ;; an encoder's or a thinker's: a thinker's applies when it loads
     :calibrations (into {} (map (fn [[k v]] [(normalise-name named-thinkers k) v])) calibrations)
     :loader (or loader load-prepared)
     :thinkers thinkers
     :max-thinkers (max 1 (long max-thinkers))
     :thinker-loader (or thinker-loader (fn [name cfg] (think/thinker (assoc cfg :name name))))
     :custom-thinker-loader (some? thinker-loader)
     :agents (atom {})
     :order (atom [])             ; least recently used first
     :thinker-agents (atom {})
     :thinker-order (atom [])}))

(defn limits-for
  "The configured sequence limits for one checkpoint: the router-wide ones
  under its per-checkpoint entry, plus its :calibration file when one is
  configured."
  [router name]
  (let [key (normalise-name router name)]
    (cond-> (merge (:limits router) (get (:checkpoints router) key))
      (get (:calibrations router) key) (assoc :calibration (get (:calibrations router) key)))))

(defn effective-limits
  "{:max-len :head-max-len} the checkpoint runs with: its prepared config.edn
  under the configured overrides; nil when it is not prepared."
  [router name]
  (let [key (normalise-name router name)
        dir (get (:models router) key)
        f (clojure.java.io/file dir "config.edn")]
    (when (and dir (.exists f))
      (let [cfg (clojure.edn/read-string (slurp f))]
        (merge (select-keys cfg [:max-len :head-max-len]) (dissoc (limits-for router key) :calibration))))))

(defn preloaded
  "A router that already holds `agent` as checkpoint `name` (default
  english): what (lev.server/handler agent opts) builds, and what a test
  with one loaded agent wants."
  ([agent] (preloaded agent "english"))
  ([agent name] (preloaded agent name {}))
  ([agent name opts]
   (let [r (make-router opts)
         key (normalise-name r name)]
     (swap! (:agents r) assoc key (assoc agent :name key))
     (swap! (:order r) conj key)
     r)))

(defn router?
  [x]
  (and (map? x) (contains? x :agents) (contains? x :loader)))

(defn thinker-names
  "The configured thinkers, sorted."
  [router]
  (vec (sort (keys (:thinkers router)))))

(defn thinker?
  "Does `name` name one of the router's thinkers?"
  [router name]
  (contains? (:thinkers router) (normalise-name router name)))

(defn available?
  "Is this checkpoint's data directory prepared (or, for a thinker, its
  GGUF on disk and the llm native built — a router given its own
  :thinker-loader vouches for the native itself)?"
  [router name]
  (let [key (normalise-name router name)]
    (if (thinker? router key)
      (boolean (and (or (:custom-thinker-loader router) (llm/available?))
                    (.exists (clojure.java.io/file (:model (get (:thinkers router) key))))))
      (let [dir (get (:models router) key)]
        (boolean (and dir (.exists (clojure.java.io/file dir "manifest.edn"))))))))

(defn loaded
  "Resident checkpoints, least recently used first."
  [router]
  @(:order router))

(defn loaded-thinkers
  "Resident thinkers, least recently used first."
  [router]
  @(:thinker-order router))

(defn- touch! [order key]
  (swap! order #(conj (vec (remove #{key} %)) key)))

(defn- close!
  "Give an agent's native state back (an MLX model, a thinker's llama
  context, the kernels' weights): its :close, when it has one."
  [agent]
  (when-let [c (:close agent)] (c agent)))

(defn- evict! [agents order max]
  (while (> (count @order) max)
    (let [victim (first @order)]
      (swap! order subvec 1)
      (close! (get @agents victim))
      (swap! agents dissoc victim))))

(defn load-model
  "The agent for `name`, loading it on first use (and evicting the least
  recently used past :max-loaded; thinkers have their own slot,
  :max-thinkers)."
  [router name]
  (let [key (normalise-name router name)]
    (if (thinker? router key)
      (let [{:keys [thinker-agents thinker-order max-thinkers]} router]
        (if-let [agent (get @thinker-agents key)]
          (do (touch! thinker-order key) agent)
          (let [agent ((:thinker-loader router) key (get (:thinkers router) key))
                agent (if-let [path (get (:calibrations router) key)]
                        (do (when-not (.exists (clojure.java.io/file path))
                              (throw (ex-info (str "calibration file " path " for " key " does not exist")
                                              {:type :model-unavailable :model key :file path})))
                            (ag/with-calibration agent (clojure.edn/read-string (slurp path))))
                        agent)]
            (swap! thinker-agents assoc key agent)
            (touch! thinker-order key)
            (evict! thinker-agents thinker-order max-thinkers)
            agent)))
      (if-let [agent (get @(:agents router) key)]
        (do (touch! (:order router) key) agent)
        (let [dir (get (:models router) key)
              agent ((:loader router) key dir (limits-for router key))]
          (swap! (:agents router) assoc key agent)
          (touch! (:order router) key)
          (evict! (:agents router) (:order router) (:max-loaded router))
          agent)))))

(defn unload
  "Free one model, or all of them."
  ([router]
   (doseq [a (concat (vals @(:agents router)) (vals @(:thinker-agents router)))] (close! a))
   (reset! (:agents router) {})
   (reset! (:order router) [])
   (reset! (:thinker-agents router) {})
   (reset! (:thinker-order router) []))
  ([router name]
   (let [key (normalise-name router name)
         [agents order] (if (thinker? router key)
                          [(:thinker-agents router) (:thinker-order router)]
                          [(:agents router) (:order router)])]
     (close! (get @agents key))
     (swap! agents dissoc key)
     (swap! order #(vec (remove #{key} %))))))

;; --- routing --------------------------------------------------------------------------

(defn- py-repr
  "Python %r of a string argument, as the reasons quote it."
  [s]
  (str "'" s "'"))

(defn- decision [model reason detection workflow]
  ;; a thinker has no Hub repo: its "repo" is nil
  (seq/ordered-map [["model" model] ["repo" (get repos model)] ["reason" reason]
                    ["detection" detection] ["workflow" workflow]]))

(defn route
  "Decide which checkpoint to use, without loading or running anything:
  {model repo reason detection workflow}, keyed like the Python RouteDecision.
  Keyword options :model :task :lang, all optional."
  [router state questions & {:keys [model task lang]}]
  (cond
    (some? model)
    (decision (normalise-name router model) (str "explicit model=" (py-repr model)) nil nil)

    (some? task)
    (let [key (normalise-name (if (= "typed_decisions" (str/replace (str/lower-case (str task)) "-" "_"))
                                "typed-decisions" task))]
      (decision key (str "explicit task=" (py-repr task)) nil nil))

    :else
    (let [workflow (match-typed-decisions-workflow questions)]
      (cond
        (and workflow (:auto-task-detection router))
        (decision "typed-decisions"
                  (str "question ids match the " (py-repr workflow) " typed-decisions workflow")
                  nil workflow)

        (some? lang)
        (let [key (if (contains? #{"en" "eng" "english"} (first (str/split (str/lower-case (str lang)) #"-")))
                    "english" "multilingual")]
          (decision key (str "explicit lang=" (py-repr lang)) nil workflow))

        :else
        (let [det (lang/analyse state)
              script (get det "script")]
          (cond
            (= script "unknown")
            (decision (:default router)
                      (str "no letters detected in state; using default (" (:default router) ")")
                      det workflow)

            (not= script "latin")
            (decision "multilingual"
                      (format "non-Latin script (%s, %.0f%% of letters); the English checkpoint cannot read it"
                              script (* 100.0 (double (get det "non_latin_fraction"))))
                      det workflow)

            (not (get det "is_english"))
            (decision "multilingual"
                      (str "Latin script but language looks like " (py-repr (get det "language")) ", not English")
                      det workflow)

            :else
            (decision "english" "English Latin text" det workflow)))))))

(defn resolve-decision
  "The decision `route` made, unless it names an encoder that is not
  prepared, the request did not ask for it by name, and the default
  model is an available thinker: then the thinker takes it (it reads any
  language) and the reason says so. An encoder default is never a
  substitute for another encoder (english cannot read what multilingual
  was chosen for), and an explicit unavailable model stays as decided, so
  loading it fails with the usual :model-unavailable."
  [router d explicit?]
  (let [chosen (get d "model")
        default (:default router)]
    (if (and (not explicit?)
             (not (thinker? router chosen))
             (not (available? router chosen))
             (thinker? router default)
             (available? router default))
      (assoc d "model" default
             "reason" (str (get d "reason") "; " chosen " is not prepared, using the default (" default ")"))
      d)))

(defn predict
  "Route, then answer every question on the chosen model: the system-one
  map plus a \"routing\" key with the decision. :constraints /
  :on-infeasible go to agent/system-one; :thinking / :thought to a thinker.
  A content-routed model that is not available falls back to the default
  (resolve-decision)."
  [router state questions & {:keys [model task lang constraints on-infeasible thinking thought debias]}]
  (let [d (resolve-decision router (route router state questions :model model :task task :lang lang) (some? model))
        agent (load-model router (get d "model"))]
    (assoc (ag/system-one agent state questions
                          (cond-> {:constraints constraints :on-infeasible on-infeasible}
                            (some? thinking) (assoc :thinking thinking)
                            (some? thought) (assoc :thought thought)
                            (some? debias) (assoc :debias debias)))
           "routing" d)))

(defn predict-batch
  "`predict` for several states against the same questions, a vector in
  order. A thinker (named: thinkers are never content-routed) gets them
  all in one lev.agent/system-one-batch, which is one pass in Jev mode;
  otherwise every state is routed and answered on its own, since content
  routing looks at the state."
  [router states questions & {:keys [model constraints on-infeasible thinking thought debias] :as opts}]
  (if (and (some? model) (thinker? router model))
    (let [d (resolve-decision router (route router (first states) questions :model model) true)
          agent (load-model router (get d "model"))]
      (mapv #(assoc % "routing" d)
            (ag/system-one-batch agent states questions
                                 (cond-> {:constraints constraints :on-infeasible on-infeasible}
                                   (some? thinking) (assoc :thinking thinking)
                                   (some? thought) (assoc :thought thought)
                                   (some? debias) (assoc :debias debias)))))
    (mapv #(apply predict router % questions (mapcat identity opts)) states)))
