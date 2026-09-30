(ns lev.config
  "~/.config/lev: `config.edn` (where the prepared model lives, server
  defaults, extra workflow directories) and the one precedence rule every
  entry point follows: CLI flag > environment variable > config.edn > default.

  config.edn keys:
    :data            prepared data root (jolt prepare --out)
    :checkpoints-home  the Hub checkpoint directory jolt prepare reads
    :encoders        {\"name\" dir} prepared data directories, overriding the :data layout
    :workflow-dirs   extra directories of workflow .clj files
    :port :host :api-key :max-loaded :default-model :auto-task-detection
                     server defaults
    :max-len :head-max-len
                     sequence limits for every checkpoint, instead of the
                     ones prepare took from rl_agent_config.json
    :backend :dtype  which engine runs the encoders: \"cpu\" (the C kernels, the
                     default) or \"mlx\" (Apple's GPU, jolt mlx), at \"f32\" (the
                     default; the goldens' precision) or \"f16\" (mlx only: half the
                     memory, faster, values off by up to 1e-2)
    :checkpoints     {\"name\" {:max-len .. :head-max-len .. :backend .. :dtype ..}} per checkpoint
    :thinkers        {\"name\" {:model \"x.gguf\" :thinking true ...}} generative
                     models (lev.think) the server offers under that name

  Environment:
    LEV_CONFIG_DIR  instead of $XDG_CONFIG_HOME/lev or ~/.config/lev
    LEV_WORKFLOWS   dir[:dir...] to scan for workflows INSTEAD of the
                     defaults (bundled ./workflows, :workflow-dirs, the
                     config dir's workflows/)
    LEV_THINKER     a GGUF path: the thinker named `thinker` (--thinker)
    LEV_BACKEND / LEV_DTYPE   :backend / :dtype for every checkpoint (--backend, --dtype)

  The pure functions take an `env` map so tests never read the real
  environment; nil means the process environment."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]))

(defn parse-args
  "--flag value pairs, bare --flags (true) and positionals under :args:
  (parse-args [\"demo\" \"--port\" \"80\" \"--self-test\"])
  => {:args [\"demo\"] \"--port\" \"80\" \"--self-test\" true}"
  [args]
  (loop [args (seq args) out {:args []}]
    (if (empty? args)
      out
      (let [[k v & more] args]
        (cond
          (not (str/starts-with? k "--")) (recur (rest args) (update out :args conj k))
          (and v (not (str/starts-with? v "--"))) (recur more (assoc out k v))
          :else (recur (rest args) (assoc out k true)))))))

(defn- getenv [env k]
  (if (nil? env) (System/getenv k) (get env k)))

(defn config-dir
  "LEV_CONFIG_DIR, else $XDG_CONFIG_HOME/lev, else ~/.config/lev."
  ([] (config-dir nil))
  ([env]
   (or (getenv env "LEV_CONFIG_DIR")
       (some-> (getenv env "XDG_CONFIG_HOME") (str "/lev"))
       (str (or (getenv env "HOME") (System/getProperty "user.home")) "/.config/lev"))))

(defn load-config
  "The map in <dir>/config.edn, or {} when there is no such file."
  ([] (load-config (config-dir)))
  ([dir]
   (let [f (io/file dir "config.edn")]
     (if-not (.exists f)
       {}
       (let [m (try (edn/read-string (slurp f))
                    (catch Exception e
                      (throw (ex-info (str "cannot read " f ": " (ex-message e)) {:file (str f)} e))))]
         (when-not (map? m)
           (throw (ex-info (str f " must hold a map, not " (pr-str m)) {:file (str f)})))
         m)))))

(defn context
  "What the resolvers below need: parsed CLI opts ({\"--data\" \"dir\"}), the
  process environment, and config.edn."
  [opts]
  {:opts opts :env nil :config (load-config (config-dir))})

(defn setting
  "CLI flag > env var > config.edn key > default. A bare --flag (true) is
  not a value."
  [{:keys [opts env config]} flag env-var key default]
  (let [v (get opts flag)
        e (getenv env env-var)]
    (cond
      (string? v) v
      (some? e) e
      (contains? config key) (get config key)
      :else default)))

(defn workflow-dirs
  "Directories to load workflows from, in load order (later wins on a name
  clash): the bundled ./workflows, config.edn :workflow-dirs, then the
  config dir's own workflows/. --workflows or LEV_WORKFLOWS (dir[:dir...])
  replaces that list, which is what a test wants."
  [{:keys [opts env config]}]
  (let [cli (get opts "--workflows")
        from-env (getenv env "LEV_WORKFLOWS")
        split #(vec (remove str/blank? (str/split % #":")))]
    (cond
      (string? cli) (split cli)
      from-env (split from-env)
      :else (vec (concat ["workflows"]
                         (:workflow-dirs config)
                         [(str (config-dir env) "/workflows")])))))

(defn- as-int [v]
  (cond (integer? v) (long v)
        (string? v) (Long/parseLong (str/trim v))
        :else (throw (ex-info (str "not an integer: " (pr-str v)) {:value v}))))

(defn- as-choice
  "v as one of the keywords in `allowed` (a keyword or a string, any case)."
  [what allowed v]
  (let [k (keyword (str/lower-case (str/trim (name v))))]
    (when-not (contains? allowed k)
      (throw (ex-info (str "unknown " what " " (pr-str v) "; one of " (str/join ", " (map name (sort allowed))))
                      {:type :invalid-config :key what :value v :allowed allowed})))
    k))

(defn limits
  "What the loader needs for checkpoint `name`: the sequence limits
  {:max-len :head-max-len} and the engine {:backend :dtype}, only the keys
  that are set. config.edn's top-level keys apply to every checkpoint, its
  :checkpoints {name {...}} entry to one; the environment (LEV_MAX_LEN,
  LEV_HEAD_MAX_LEN, LEV_BACKEND, LEV_DTYPE) and the CLI (--max-len,
  --head-max-len, --backend, --dtype) beat both, for every checkpoint. {}
  means the checkpoint's own values stand, on the C kernels."
  [{:keys [opts env config]} name]
  (let [pick (fn [m k] (when (contains? m k) {k (as-int (get m k))}))
        choice (fn [m k allowed] (when (some? (get m k)) {k (as-choice (clojure.core/name k) allowed (get m k))}))
        engine (fn [m] (merge (choice m :backend #{:cpu :mlx}) (choice m :dtype #{:f32 :f16})))
        layer (fn [m] (merge (pick m :max-len) (pick m :head-max-len) (engine m)))
        from-env (merge (when-let [v (getenv env "LEV_MAX_LEN")] {:max-len (as-int v)})
                        (when-let [v (getenv env "LEV_HEAD_MAX_LEN")] {:head-max-len (as-int v)})
                        (engine {:backend (getenv env "LEV_BACKEND") :dtype (getenv env "LEV_DTYPE")}))
        from-cli (merge (when (string? (get opts "--max-len")) {:max-len (as-int (get opts "--max-len"))})
                        (when (string? (get opts "--head-max-len")) {:head-max-len (as-int (get opts "--head-max-len"))})
                        (engine {:backend (let [v (get opts "--backend")] (when (string? v) v))
                                 :dtype (let [v (get opts "--dtype")] (when (string? v) v))}))]
    (or (merge (layer config)
               (layer (get-in config [:checkpoints name] {}))
               from-env
               from-cli)
        {})))

(defn thinkers
  "The generative models a server can offer, {name {:model gguf-path ...}}:
  config.edn :thinkers (names as keywords or strings; the rest of each
  entry is lev.think config), plus --thinker PATH or LEV_THINKER as the
  one named `thinker`. Throws when an entry names no :model."
  [{:keys [opts env config]}]
  (let [from-config (into {}
                          (map (fn [[k v]]
                                 (let [name (if (keyword? k) (clojure.core/name k) (str k))]
                                   (when-not (and (map? v) (string? (:model v)))
                                     (throw (ex-info (str "config.edn :thinkers " (pr-str name) " needs a :model (a GGUF path)")
                                                     {:thinker name :entry v})))
                                   [name v])))
                          (:thinkers config))
        cli (get opts "--thinker")
        from-env (getenv env "LEV_THINKER")
        one (cond (string? cli) cli from-env from-env)]
    (cond-> from-config
      one (assoc "thinker" {:model one}))))

(defn encoders
  "Prepared encoder data directories named in config.edn :encoders
  ({name dir}, names as keywords or strings), overriding the :data root's
  layout entry by entry; {} when none are named."
  [{:keys [config]}]
  (into {} (map (fn [[k v]] [(if (keyword? k) (clojure.core/name k) (str k)) (str v)])) (:encoders config)))

(defn calibrations
  "Calibration files (lev.calibrate) per model, an encoder's or a
  thinker's: config.edn :calibration {name path}; --calibration PATH or
  LEV_CALIBRATION applies one file to every encoder."
  [{:keys [opts env config]}]
  (let [per (into {} (map (fn [[k v]] [(if (keyword? k) (clojure.core/name k) (str k)) (str v)])) (:calibration config))
        cli (get opts "--calibration")
        one (cond (string? cli) cli (getenv env "LEV_CALIBRATION") (getenv env "LEV_CALIBRATION"))]
    (if one
      (into {} (map (fn [n] [n one])) ["english" "multilingual" "typed-decisions"])
      per)))
