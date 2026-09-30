(ns lev.sequence
  "build_sequence + option rendering, mirroring rl_common.py.

  Sequence layout:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
  The [MASK] positions (one per option, in label order) are returned as markers."
  (:require [clojure.string :as str]
            [lev.tokenizer :as tk]))

(def qtypes {"choice" 0 "score" 1 "noul" 2})
(def qtype-names {0 "choice" 1 "score" 2 "noul"})

(defn ordered-map
  "An insertion-ordered map from [k v] pairs, however many. Python dicts
  keep insertion order and both option order and answer order are part of
  the contract; `into`/`assoc` on an array-map silently become a hash-map
  past 8 entries, `(apply array-map ...)` does not."
  [kvs]
  (apply array-map (mapcat identity kvs)))

(defn py-float-str
  "Python float.__repr__ / json.dumps of a double: the shortest round-trip
  digits, fixed notation when 1e-4 <= |x| < 1e16, else d.ddde+XX with a
  signed two-digit-minimum exponent. json.dumps spells the non-finite values
  NaN / Infinity / -Infinity."
  [x]
  (let [x (double x)]
    (cond
      (Double/isNaN x) "NaN"
      (= x ##Inf) "Infinity"
      (= x ##-Inf) "-Infinity"
      (zero? x) (if (neg? (/ 1.0 x)) "-0.0" "0.0")
      :else
      ;; Chez prints the same shortest digits, Java-style: d.ddd or d.dddE[-]n.
      (let [s (str x)
            minus? (str/starts-with? s "-")
            s (if minus? (subs s 1) s)
            [mant e] (str/split s #"E")
            [ip fp] (str/split mant #"\.")
            fp (or fp "")
            e (if e (Long/parseLong e) 0)
            raw (str ip fp)
            lead (count (take-while #(= % \0) raw))
            digits (str/replace (subs raw lead) #"0+$" "")
            digits (if (= "" digits) "0" digits)
            ;; digits before the decimal point when value = 0.D x 10^decpt
            decpt (- (+ (count ip) e) lead)
            nd (count digits)
            body (if (or (<= decpt -4) (> decpt 16))
                   (let [ex (dec decpt)]
                     (str (subs digits 0 1)
                          (when (> nd 1) (str "." (subs digits 1)))
                          "e" (if (neg? ex) "-" "+")
                          (format "%02d" (Math/abs ex))))
                   (cond
                     (<= decpt 0) (str "0." (apply str (repeat (- decpt) "0")) digits)
                     (< decpt nd) (str (subs digits 0 decpt) "." (subs digits decpt))
                     :else (str digits (apply str (repeat (- decpt nd) "0")) ".0")))]
        (str (when minus? "-") body)))))

(defn- hex4 [n] (format "\\u%04x" n))

(defn- json-escape
  "json.dumps string escaping. ensure_ascii=False only escapes controls,
  backslash and quote; ensure_ascii=True also \\u-escapes every codepoint
  outside 0x20..0x7E (astral ones as a surrogate pair)."
  [s ensure-ascii?]
  (let [sb (StringBuilder.)]
    (doseq [c s]
      (let [cp (int c)]
        (cond
          (= c \") (.append sb "\\\"")
          (= c \\) (.append sb "\\\\")
          (= c \newline) (.append sb "\\n")
          (= c \return) (.append sb "\\r")
          (= c \tab) (.append sb "\\t")
          (= cp 8) (.append sb "\\b")
          (= cp 12) (.append sb "\\f")
          (< cp 0x20) (.append sb (hex4 cp))
          (and ensure-ascii? (> cp 0x7E))
          (if (< cp 0x10000)
            (.append sb (hex4 cp))
            (let [v (- cp 0x10000)]
              (.append sb (hex4 (bit-or 0xD800 (bit-shift-right v 10))))
              (.append sb (hex4 (bit-or 0xDC00 (bit-and v 0x3FF))))))
          :else (.append sb c))))
    (.toString sb)))

(defn- jkey [k] (if (keyword? k) (name k) (str k)))

(defn json-str
  "Serialize v like Python json.dumps(v) with the default separators
  (\", \" and \": \"). ensure_ascii defaults to FALSE here because the
  state is serialized that way (serialize_state); pass {:ensure-ascii true}
  for json.dumps' own default (_to_internal's instructions), {:compact
  true} for separators=(\",\", \":\"), where NaN and infinities are null
  (nlohmann's dump, what Winnow's prompt uses)."
  ([v] (json-str v {}))
  ([v {:keys [ensure-ascii compact] :as opts}]
   (let [[comma colon] (if compact ["," ":"] [", " ": "])]
     (cond
       (string? v) (str "\"" (json-escape v ensure-ascii) "\"")
       (keyword? v) (json-str (name v) opts)
       (map? v) (str "{" (str/join comma (map (fn [[k val]]
                                                (str (json-str (jkey k) opts) colon (json-str val opts)))
                                              v)) "}")
       (or (sequential? v) (set? v)) (str "[" (str/join comma (map #(json-str % opts) v)) "]")
       (boolean? v) (if v "true" "false")
       (nil? v) "null"
       (integer? v) (str v)
       (and compact (number? v) (or (Double/isNaN (double v)) (Double/isInfinite (double v)))) "null"
       (number? v) (py-float-str v)
       :else (json-str (str v) opts)))))

(defn serialize-state [state]
  (if (string? state) state (json-str state)))

(defn- no-description?
  "As the Python package: only None and \"\" mean \"no description\"; 0 and False are
  legitimate criterion values (the old `not v` test dropped them)."
  [v]
  (or (nil? v) (= v "")))

(defn render-criterion
  "render_criterion: strings pass through, anything structured becomes
  compact JSON (json.dumps with separators (\", \", \": \"), ensure_ascii
  off), so a rubric reads as JSON rather than a Python repr."
  [v]
  (if (string? v) v (json-str v)))

(defn render-options
  "Option texts in label-index order. noul is always [false, true] so p[1] == noul."
  [q]
  (let [t (:t q) crit (:crit q)]
    (case t
      "choice" (mapv (fn [[k v]] (if (no-description? v) (str k) (str k ": " (render-criterion v)))) crit)
      "score" (mapv (fn [i c] (str "level " i ": " (render-criterion c))) (range) crit)
      (let [c (or crit {})
            f (get c "false")
            tr (get c "true")]
        [(str "false: " (if (no-description? f) "no, the statement does not hold" (render-criterion f)))
         (str "true: " (if (no-description? tr) "yes, the statement holds" (render-criterion tr)))]))))

(defn temp-bucket
  "Per-cardinality temperature key, e.g. \"choice:3-5\"."
  [qtype k]
  (let [size (if (<= k 2) "2" (if (<= k 5) "3-5" (if (<= k 10) "6-10" "11+")))]
    (str (qtype-names qtype) ":" size)))

(defn build-prefix
  "The question-only prefix of a sequence, [ids markers]:
    [CLS] <type> question: <instructions> [SEP] [MASK] opt0 [MASK] opt1 ... [SEP]
  (build_prefix: everything before the state, before the max-len cut). It
  depends on the question and head-max-len alone, so it can be cached
  across calls (`cached-prefix`); the state is tokenized once per call
  (`encode-state`) and `assemble` joins the two."
  [tok q head-max-len]
  (let [mask (tk/mask-token tok)
        sp (:specials tok)
        opts (render-options q)
        ins (str/replace (str (:ins q)) mask " ")
        raw-head (tk/encode tok (str (:t q) " question: " ins))
        opt-ids (mapv (fn [o]
                        (into [(:mask sp)]
                              (take 48 (tk/encode tok (str " " (str/replace o mask " "))))))
                      opts)
        opt-budget (- head-max-len (reduce + (map count opt-ids)))
        opt-ids (if (< opt-budget 16)
                  (let [per (max 4 (quot (- head-max-len 16) (max 1 (count opt-ids))))]
                    (mapv #(vec (take per %)) opt-ids))
                  opt-ids)
        opt-budget (- head-max-len (reduce + (map count opt-ids)))
        head-ids (vec (take (max 8 opt-budget) raw-head))
        base (into [(:cls sp)] (concat head-ids [(:sep sp)]))
        [ids markers] (reduce (fn [[ids ms] o]
                                [(into ids o) (conj ms (count ids))])
                              [base []]
                              opt-ids)]
    [(conj ids (:sep sp)) markers]))

(defn encode-state
  "The state's token ids: json.dumps for a non-string, [MASK] blanked so
  the text cannot inject a marker. The whole state, however long: assemble
  cuts it to the room a prefix leaves and reports how much did not fit."
  [tok state]
  (tk/encode tok (str/replace (serialize-state state) (tk/mask-token tok) " ")))

(defn assemble
  "A prefix ([ids markers] from build-prefix) + the state's ids -> [ids
  markers dropped] as build-sequence answers them: the state cut to the
  room the prefix leaves for it and a closing [SEP], the whole cut at
  max-len, markers past the cut gone, and how many state tokens were
  dropped (0 when the state is read whole)."
  [prefix-ids prefix-markers state-ids max-len]
  (let [room (max 0 (- max-len (count prefix-ids) 1))
        st (vec (take room state-ids))
        sep (peek prefix-ids)]
    [(vec (take max-len (into prefix-ids (conj st sep))))
     (filterv #(< % max-len) prefix-markers)
     (- (count state-ids) (count st))]))

(defn build-sequence
  "Returns [ids markers dropped]: token ids, the [MASK] marker positions,
  and how many of the state's tokens did not fit in max-len (0 when the
  state is read whole). build-prefix + encode-state + assemble in one, for
  a single question; a call with many questions shares the state's ids."
  [tok state q max-len head-max-len]
  (let [[pids pmarkers] (build-prefix tok q head-max-len)]
    (assemble pids pmarkers (encode-state tok state) max-len)))

;; --- prefix cache ---------------------------------------------------------------

(defn prefix-cache
  "A bounded LRU of built prefixes (laya-mlx's PrefixCache): one per agent,
  since the key does not name the tokenizer. Workloads are exactly its hit
  pattern, the same questions over a fresh state: a workflow's fixed
  questions, a game's per-tick questions, a benchmark's templates. A hit
  saves the prefix's tokenization (~1 ms a question); with `encode-state`
  once per call, a repeat call tokenizes nothing but the state."
  ([] (prefix-cache 128))
  ([capacity] (atom {:capacity capacity :tick 0 :entries {}})))

(defn prefix-cache-size [cache] (count (:entries @cache)))

(defn- touch
  "The cache with key marked as just used (and, for a miss, holding v),
  the least recently used entry dropped past capacity."
  [{:keys [capacity tick entries] :as c} k v]
  (let [tick (inc tick)
        entries (assoc entries k {:v v :t tick})
        entries (if (> (count entries) capacity)
                  (dissoc entries (key (apply min-key (comp :t val) entries)))
                  entries)]
    (assoc c :tick tick :entries entries)))

(defn cached-prefix
  "build-prefix through the cache (nil: no cache, just build). Keyed by the
  question's type, instructions and rendered options and by head-max-len:
  everything build-prefix reads besides the tokenizer."
  [cache tok q head-max-len]
  (if (nil? cache)
    (build-prefix tok q head-max-len)
    (let [k [head-max-len (:t q) (:ins q) (render-options q)]]
      (if-let [hit (get-in @cache [:entries k :v])]
        (do (swap! cache touch k hit) hit)
        (let [v (build-prefix tok q head-max-len)]
          (swap! cache touch k v)
          v)))))
