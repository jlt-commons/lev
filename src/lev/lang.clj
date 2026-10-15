(ns lev.lang
  "Dependency-free language/script detection used to route between the
  encoder checkpoints (the port of their Python package's lang.py, pinned
  by golden/lang.edn).

  Routing needs one decision: is this English Latin text, or something the
  English checkpoint cannot read? The English checkpoint collapses to near
  random on non-Latin scripts while staying confident, so *script* is the
  primary signal; whether Latin text is English is a best-effort
  stopword/diacritic guess. Pass an explicit model or lang when you know.

  Python's str.isalpha / \\w classes come from the Unicode categories,
  as in the tokenizer (lev.tokenizer/char-type)."
  (:require [clojure.string :as str]
            [lev.sequence :as seq]
            [lev.tokenizer :as tk]))

;; Unicode blocks the English (50k English BPE) checkpoint cannot read, in
;; the order Python checks them (first match wins).
(def script-ranges
  [["greek" [[0x0370 0x03FF] [0x1F00 0x1FFF]]]
   ["cyrillic" [[0x0400 0x052F] [0x2DE0 0x2DFF] [0xA640 0xA69F]]]
   ["hebrew" [[0x0590 0x05FF]]]
   ["arabic" [[0x0600 0x06FF] [0x0750 0x077F] [0x08A0 0x08FF] [0xFB50 0xFDFF] [0xFE70 0xFEFF]]]
   ["devanagari" [[0x0900 0x097F] [0xA8E0 0xA8FF]]]
   ["bengali" [[0x0980 0x09FF]]]
   ["gurmukhi" [[0x0A00 0x0A7F]]]
   ["gujarati" [[0x0A80 0x0AFF]]]
   ["oriya" [[0x0B00 0x0B7F]]]
   ["tamil" [[0x0B80 0x0BFF]]]
   ["telugu" [[0x0C00 0x0C7F]]]
   ["kannada" [[0x0C80 0x0CFF]]]
   ["malayalam" [[0x0D00 0x0D7F]]]
   ["sinhala" [[0x0D80 0x0DFF]]]
   ["thai" [[0x0E00 0x0E7F]]]
   ["lao" [[0x0E80 0x0EFF]]]
   ["tibetan" [[0x0F00 0x0FFF]]]
   ["myanmar" [[0x1000 0x109F]]]
   ["georgian" [[0x10A0 0x10FF]]]
   ["ethiopic" [[0x1200 0x137F]]]
   ["khmer" [[0x1780 0x17FF]]]
   ["hangul" [[0x1100 0x11FF] [0x3130 0x318F] [0xAC00 0xD7AF]]]
   ["kana" [[0x3040 0x309F] [0x30A0 0x30FF] [0x31F0 0x31FF]]]
   ["han" [[0x3400 0x4DBF] [0x4E00 0x9FFF] [0xF900 0xFAFF]]]])

;; Function words. Latin-script languages overlap heavily (de/la/le/un/e/que),
;; so each hit is weighted and a margin is required before calling something
;; non-English. Order matters: ties go to the first language.
(def stopwords
  [["en" #{"the" "and" "is" "are" "was" "were" "to" "of" "in" "for" "with" "that"
           "this" "it" "you" "have" "has" "not" "but" "on" "at" "be" "as" "from"
           "will" "can" "would" "there" "their" "what" "which" "please" "we" "i"}]
   ["fr" #{"le" "la" "les" "des" "une" "est" "pour" "dans" "que" "qui" "avec" "sur"
           "pas" "plus" "nous" "vous" "être" "cette" "mais" "sont" "ont" "aux" "ce"}]
   ["de" #{"der" "die" "das" "und" "ist" "ein" "eine" "den" "dem" "nicht" "mit" "für"
           "auf" "von" "zu" "sich" "auch" "werden" "wurde" "haben" "sind" "oder" "aber"}]
   ["es" #{"el" "los" "las" "que" "por" "con" "para" "una" "es" "se" "del" "como"
           "pero" "son" "está" "este" "esta" "todo" "más" "muy" "hay" "sus"}]
   ["pt" #{"os" "as" "que" "em" "um" "uma" "para" "com" "não" "é" "se" "do" "da"
           "dos" "das" "mas" "são" "está" "este" "esta" "muito" "pelo" "pela"}]
   ["it" #{"il" "lo" "gli" "che" "di" "per" "con" "non" "è" "si" "del" "della" "sono"
           "questo" "questa" "anche" "come" "più" "nella" "alla"}]   ; python lists "sono" twice
   ["nl" #{"het" "een" "van" "is" "op" "te" "dat" "niet" "met" "voor" "zijn" "aan"
           "door" "maar" "ook" "worden" "deze" "naar" "wordt"}]])

(def non-en-diacritics (set "àâäãáåçéèêëíìîïñóòôöõøúùûüýÿßæœđłşţğıåäö"))

;; --- python character classes -------------------------------------------------

(defn- alpha?
  "str.isalpha(): categories Lu Ll Lt Lm Lo."
  [c]
  (tk/letter? c))

(defn- word-char?
  "The class [^\\W\\d_] of Python's re: \\w (isalnum or _) minus decimal
  digits (Nd) minus underscore, i.e. letters plus Nl/No numerics."
  [c]
  (let [t (tk/char-type c)]
    (or (and (>= t 1) (<= t 5)) (= t 10) (= t 11))))

(defn- words
  "_WORD.findall(text): maximal runs of word chars."
  [text]
  (loop [cs (seq text) cur (StringBuilder.) out (transient [])]
    (if (empty? cs)
      (persistent! (if (pos? (.length cur)) (conj! out (.toString cur)) out))
      (let [c (first cs)]
        (if (word-char? c)
          (recur (rest cs) (.append cur c) out)
          (recur (rest cs) (StringBuilder.)
                 (if (pos? (.length cur)) (conj! out (.toString cur)) out)))))))

(defn- round-to [x places]
  (let [f (Math/pow 10 places)]
    (/ (Math/round (* (double x) f)) f)))

;; --- state flattening -------------------------------------------------------------

(defn- iter-text [state depth]
  (cond
    (or (> depth 6) (nil? state)) []
    (string? state) [state]
    (map? state) (into [] (mapcat #(iter-text % (inc depth))) (vals state))
    (sequential? state) (into [] (mapcat #(iter-text % (inc depth))) state)
    :else []))

(defn state-text
  "Flatten a state (string / map / vector) into the text detection looks at:
  the string leaves joined by spaces, keys ignored (they are usually English),
  capped at 4000 chars."
  ([state] (state-text state 4000))
  ([state max-chars]
   (let [s (str/join " " (iter-text state 0))]
     (if (> (count s) max-chars) (subs s 0 max-chars) s))))

;; --- scripts ----------------------------------------------------------------------

(defn- latin? [cp]
  (or (< cp 0x0250) (<= 0x1E00 cp 0x1EFF)))

(defn- script-of [cp]
  (some (fn [[name ranges]]
          (when (some (fn [[lo hi]] (<= lo cp hi)) ranges) name))
        script-ranges))

(defn- script-counts
  "[[script count] ...] in first-seen order, latin appended last (the order
  Python's dict has after the loop, which decides ties), plus the latin count."
  [text]
  (loop [cs (seq text) counts (seq/ordered-map []) latin 0]
    (if (empty? cs)
      [counts latin]
      (let [c (first cs)]
        (if-not (alpha? c)
          (recur (rest cs) counts latin)
          (let [cp (long c)]
            (cond
              (latin? cp) (recur (rest cs) counts (inc latin))
              :else (if-let [name (script-of cp)]
                      (recur (rest cs) (assoc counts name (inc (get counts name 0))) latin)
                      (recur (rest cs) counts latin)))))))))

(defn detect-script
  "Dominant script: \"latin\", \"han\", \"devanagari\", ... or \"unknown\" when
  there are no letters. Ties go to the script seen first, latin last."
  [text]
  (let [[counts latin] (script-counts text)
        counts (assoc counts "latin" latin)
        total (reduce + (vals counts))]
    (if (zero? total)
      "unknown"
      (first (reduce (fn [[_ bv :as best] [k v]] (if (> v bv) [k v] best))
                     (first counts) (rest counts))))))

(defn script-profile
  "Fraction of alphabetic characters in each detected script (latin first,
  as Python seeds its dict with it); {} when there are none."
  [text]
  (let [[counts latin] (script-counts text)
        total (+ latin (reduce + (vals counts)))]
    (if (zero? total)
      (seq/ordered-map [])
      (seq/ordered-map
       (for [[k v] (cons ["latin" latin] counts) :when (pos? v)]
         [k (/ (double v) total)])))))

;; --- Latin-script language guess ----------------------------------------------------

(defn guess-latin-language
  "Best-effort language code for Latin-script text, or nil when undecided.
  Scores function-word hits per language and requires the winner to beat
  English by a margin, so ordinary English is never misrouted. Short inputs
  usually return nil on purpose."
  [text]
  (let [ws (mapv str/lower-case (words text))]
    (when (>= (count ws) 4)
      (let [scores (into {} (map (fn [[lg sw]] [lg (count (filter sw ws))])) stopwords)
            lowered (str/lower-case text)
            diac (count (filter non-en-diacritics lowered))
            diac-rate (/ (double diac) (max 1 (count lowered)))
            en (get scores "en" 0)
            ;; Python's max() over the non-English languages: the first one
            ;; wins ties, and it is *always* some language, even at score 0,
            ;; which is what makes diacritic-heavy text with no stopword hits
            ;; come out as "fr"
            [best-lg best] (reduce (fn [[_ bs :as best] [lg _]]
                                     (let [s (get scores lg)] (if (> s bs) [lg s] best)))
                                   (let [[lg _] (second stopwords)] [lg (get scores lg)])
                                   (drop 2 stopwords))]
        (cond
          (and (zero? best) (< diac-rate 0.02)) (when (pos? en) "en")
          (and best-lg (>= best (max 2 (+ en 2)))) best-lg
          (and (>= diac-rate 0.04) best-lg (>= best en)) best-lg
          :else (when (pos? en) "en"))))))

;; --- the full decision ----------------------------------------------------------------

(defn analyse
  "Detection result for a state, keyed like the Python dict: script,
  script_profile, language (best effort, may be nil), is_english,
  non_latin_fraction."
  [state]
  (let [text (state-text state)
        prof (script-profile text)
        script (detect-script text)
        non-latin (if (seq prof) (round-to (- 1.0 (get prof "latin" 0.0)) 4) 0.0)
        result (fn [script language english? non-latin]
                 (seq/ordered-map [["script" script] ["script_profile" prof] ["language" language]
                                   ["is_english" english?] ["non_latin_fraction" non-latin]]))]
    (cond
      (= script "unknown") (result "unknown" nil true 0.0)
      (not= script "latin") (result script nil false non-latin)
      :else (let [lang (guess-latin-language text)]
              (result "latin" lang (contains? #{nil "en"} lang) non-latin)))))

(defn is-english?
  "True when the English checkpoint can be expected to read this state."
  [state]
  (true? (get (analyse state) "is_english")))
