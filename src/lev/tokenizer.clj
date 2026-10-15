(ns lev.tokenizer
  "The two tokenizers in the checkpoint bundle, faithful to their tokenizer.json:

  :byte-level (english, typed-decisions) - GPT-2 byte-level BPE: NFC
  normalize, GPT-2 pre-tokenization (hand-written scanner with exact
  \\p{L}/\\p{N} classes from the Unicode categories, since the
  pattern needs lookahead that irregex lacks), byte-to-unicode mapping, and
  lowest-rank-first BPE merges.

  :sentencepiece (multilingual, the mmBERT/Gemma tokenizer) - added tokens
  extracted first (leftmost-longest, `lstrip` swallowing whitespace), then
  per segment: spaces to \u2581, a \u2581 prepended, split keeping \u2581
  with the piece it starts, and BPE over each piece with UTF-8 byte
  fallback (<0xNN>) and fused <unk>.

  Encoding parity is pinned by golden/tok.edn and golden/multilingual/tok.edn
  (input_ids from the real transformers fast tokenizers) and the scanner by
  the ByteLevel pre-tokenizer boundaries in golden/cases.edn."
  (:require [clojure.edn :as edn]
            [clojure.string :as str]))

;; --- Unicode: normalization + character classes ------------------------------
;; From the runtime's own tables (jolt: Chez's, the same Unicode data the
;; JVM's Character and Normalizer read), so no ICU is loaded and every host
;; answers alike, whatever ICU it has. They track a newer Unicode than an
;; older ICU: the code points they differ on are the ones assigned since.

(defn- ascii? [s] (every? #(< (long %) 128) s))

(defn nfd
  "NFD-normalize. ASCII-only input is identity (fast path)."
  [s]
  (if (ascii? s) s (java.text.Normalizer/normalize s java.text.Normalizer$Form/NFD)))

(defn lower
  "Full Unicode lowercase in the root locale (JavaScript's toLowerCase:
  İ is i + U+0307, never a Turkish dotless i)."
  [s]
  (str/lower-case s))

(defn char-type
  "ICU's UCharCategory of a code point (1 Lu .. 6 Mn, 12 Zs, 15 Cc, 16 Cf,
  19..23 and 28..29 the punctuation). The JVM's Character.getType numbers
  them alike but skips 17, so its private use (18) to final punctuation
  (30) are ICU's 17 to 29."
  [c]
  (let [t (Character/getType (int c))]
    (if (> t 17) (dec t) t)))

(defn nfc
  "NFC-normalize. ASCII-only input is identity (fast path)."
  [s]
  (if (ascii? s) s (java.text.Normalizer/normalize s java.text.Normalizer$Form/NFC)))

(def ^:private white-space
  "The White_Space property (PropList.txt), regex \\s: unchanged since
  Unicode 6.3. Not Character/isWhitespace, which leaves out the no-break
  spaces and takes the U+001C..U+001F separators the GPT-2 pattern treats
  as ordinary chars."
  (into #{0x20 0x85 0xA0 0x1680 0x2028 0x2029 0x202F 0x205F 0x3000}
        (concat (range 0x09 0x0E) (range 0x2000 0x200B))))

;; \\p{L} = UCharType 1..5; \\p{N} = 9..11 (Nd, Nl, No)
(defn letter? [c] (let [t (char-type c)] (and (>= t 1) (<= t 5))))
(defn number? [c] (let [t (char-type c)] (and (>= t 9) (<= t 11))))
(defn space? [c] (contains? white-space (long c)))

;; --- byte-level BPE -----------------------------------------------------------

(def ^:private byte-chars
  "GPT-2 bytes_to_unicode: bytes in the printable ranges map to themselves,
  every other byte b maps to 256+n (n = its ordinal among excluded bytes)."
  (let [in-ranges (fn [b]
                    (boolean (some (fn [[lo hi]] (and (>= b lo) (<= b hi)))
                                   [[33 126] [161 172] [174 255]])))]
    (loop [b 0 n 0 out []]
      (if (= b 256)
        out
        (if (in-ranges b)
          (recur (inc b) n (conj out b))
          (recur (inc b) (inc n) (conj out (+ 256 n))))))))

(defn string->utf8-bytes
  "Codepoint string -> UTF-8 byte sequence."
  [s]
  (mapcat (fn [ch]
            (let [cp (long ch)]
              (cond
                (< cp 0x80) [cp]
                (< cp 0x800) [(bit-or 0xC0 (bit-shift-right cp 6))
                              (bit-or 0x80 (bit-and cp 0x3F))]
                (< cp 0x10000) [(bit-or 0xE0 (bit-shift-right cp 12))
                                (bit-or 0x80 (bit-and (bit-shift-right cp 6) 0x3F))
                                (bit-or 0x80 (bit-and cp 0x3F))]
                :else [(bit-or 0xF0 (bit-shift-right cp 18))
                       (bit-or 0x80 (bit-and (bit-shift-right cp 12) 0x3F))
                       (bit-or 0x80 (bit-and (bit-shift-right cp 6) 0x3F))
                       (bit-or 0x80 (bit-and cp 0x3F))])))
          s))

(defn piece->chars
  "One pre-token -> the vocab-key char sequence (byte-level encoding)."
  [s]
  (map #(char (nth byte-chars %)) (string->utf8-bytes s)))

(defn pre-tokenize
  "GPT-2 pattern, ordered alternation:
    's|'t|'re|'ve|'m|'ll|'d| ?\\p{L}+| ?\\p{N}+| ?[^\\s\\p{L}\\p{N}]+|\\s+(?!\\S)|\\s+
  Hand scanner over codepoints; returns pre-token strings.

  The optional prefix is a literal U+0020 only: a tab or NBSP before a word
  is its own \\s+ token. A whitespace run followed by a non-space gives up
  its last char (\\s+(?!\\S) backtracks one), unless the run is a single
  char, in which case plain \\s+ takes it."
  [s]
  (let [cps (vec s)
        n (count cps)
        contraction? (fn [i]
                       (when (and (= (nth cps i) \') (< (inc i) n))
                         (let [c1 (nth cps (inc i))
                               c2 (when (< (+ i 2) n) (nth cps (+ i 2)))]
                           (cond (= c1 \s) (+ i 2)
                                 (= c1 \t) (+ i 2)
                                 (and (= c1 \r) (= c2 \e)) (+ i 3)
                                 (and (= c1 \v) (= c2 \e)) (+ i 3)
                                 (= c1 \m) (+ i 2)
                                 (and (= c1 \l) (= c2 \l)) (+ i 3)
                                 (= c1 \d) (+ i 2)
                                 :else nil))))
        run (fn [i pred]
              (loop [k i]
                (if (or (>= k n) (not (pred (nth cps k))))
                  k
                  (recur (inc k)))))
        other-run (fn [i]
                    (loop [k i]
                      (if (or (>= k n)
                              (space? (nth cps k))
                              (letter? (nth cps k))
                              (number? (nth cps k)))
                        k
                        (recur (inc k)))))
        next-c (fn [i] (when (< (inc i) n) (nth cps (inc i))))]
    (loop [i 0 rev ()]
      (if (>= i n)
        (vec (reverse rev))
        (let [c (nth cps i)
              nc (next-c i)
              end (or (contraction? i)
                      (cond
                        ;; " ?\\p{L}+" / " ?\\p{N}+" / " ?[^\\s\\p{L}\\p{N}]+"
                        (and (= c \space) nc (letter? nc)) (run (inc i) letter?)
                        (letter? c) (run i letter?)
                        (and (= c \space) nc (number? nc)) (run (inc i) number?)
                        (number? c) (run i number?)
                        (and (= c \space) nc (not (space? nc))) (other-run (inc i))

                        ;; "\\s+(?!\\S)|\\s+"
                        (space? c)
                        (let [k (run i space?)]
                          (if (and (< k n) (> (- k i) 1)) (dec k) k))

                        :else (other-run i)))
              tok (subs s i end)]
          (recur end (conj rev tok)))))))

(defrecord Tokenizer [vocab ranks specials added max-added kind mask-token lstrip byte-fallback fuse-unk])

(defn load
  "Load data/tokenizer.edn -> Tokenizer. :kind defaults to :byte-level (the
  format-1 files jolt prepare writes for GPT-2 style tokenizers)."
  [path]
  (let [m (edn/read-string (slurp path))
        added (into {} (:added m))]
    (map->Tokenizer
     {:vocab (:vocab m)
      :ranks (into {} (map-indexed (fn [i mg]
                                     (let [[a b] (str/split mg #" ")]
                                       [[a b] i])))
                   (:merges m))
      :specials (:specials m)
      :added added
      :max-added (reduce (fn [mx [k _]] (max mx (count k))) 0 added)
      :kind (or (:kind m) :byte-level)
      :mask-token (or (:mask-token m) "[MASK]")
      :lstrip (set (:lstrip m))
      :byte-fallback (boolean (:byte-fallback m))
      :fuse-unk (boolean (:fuse-unk m))})))

(defn mask-token
  "The mask token's text ([MASK] or <mask>): build-sequence blanks it out of
  user text so it cannot inject a marker."
  [tok]
  (:mask-token tok))

(defn- lowest-ranked-pair
  "[a b rank] of the adjacent pair with the lowest merge rank, or nil.
  A reduce whose accumulator starts as nil and is tested with nil?: jolt
  0.8.9's release build typed the accumulator from its init alone and
  folded the nil? to true (the last pair won and every BPE merge in the
  standalone binary went wrong); 0.8.10 fixed it. `jolt binary` runs the
  built server's self-test against golden/ so a regression cannot ship."
  [parts ranks]
  (reduce (fn [best [a b]]
            (let [r (get ranks [a b])]
              (if (and r (or (nil? best) (< (long r) (long (nth best 2)))))
                [a b r]
                best)))
          nil
          (map vector parts (rest parts))))

(defn apply-bpe
  "GPT-2 BPE: repeatedly merge the adjacent pair with the lowest rank."
  [parts ranks]
  (loop [parts parts]
    (if (< (count parts) 2)
      parts
      (let [best (lowest-ranked-pair parts ranks)]
        (if (nil? best)
          parts
          (let [[a b] best
                merged (str a b)]
            (recur (loop [src parts out []]
                     (if (empty? src)
                       out
                       (if (and (= (first src) a)
                                (seq (rest src))
                                (= (second src) b))
                         (recur (rest (rest src)) (conj out merged))
                         (recur (rest src) (conj out (first src)))))))))))))

(defn added-segments
  "Split s leftmost-longest on the added-token set. Returns a vector of
  strings (gaps to pre-tokenize) and longs (added-token ids to emit verbatim).
  HuggingFace extracts added tokens after normalization and before the
  ByteLevel pre-tokenizer, so a run like \"  \" becomes its own id rather
  than a pair of byte-level spaces."
  [s added max-added]
  (let [n (count s)]
    (loop [i 0 start 0 out []]
      (if (>= i n)
        (if (< start n) (conj out (subs s start n)) out)
        (let [lim (min max-added (- n i))
              hit (loop [l lim]
                    (when (pos? l)
                      (let [sub (subs s i (+ i l))]
                        (if (contains? added sub) l (recur (dec l))))))]
          (if hit
            (recur (+ i hit) (+ i hit)
                   (conj (if (< start i) (conj out (subs s start i)) out)
                         (get added (subs s i (+ i hit)))))
            (recur (inc i) start out)))))))

(defn- byte-level-encode [tok text]
  (let [s (nfc text)]
    (into []
          (mapcat (fn [seg]
                    (if (string? seg)
                      (mapcat (fn [pre]
                                (let [parts (apply-bpe (map str (piece->chars pre))
                                                       (:ranks tok))]
                                  (keep #(get (:vocab tok) %) parts)))
                              (pre-tokenize seg))
                      [seg])))
          (added-segments s (:added tok) (:max-added tok)))))

;; --- sentencepiece BPE (mmBERT / Gemma) -----------------------------------------

(def ^:private lower-one-eighth-block "\u2581")

(defn sp-segments
  "Added-token extraction for the sentencepiece kind: like added-segments
  (leftmost-longest over the raw text; HF matches these before any
  normalization) but a token in `lstrip` also swallows the Unicode
  whitespace before it, up to the previous match. Strings are gaps, longs
  are ids."
  [s added max-added lstrip]
  (let [n (count s)
        emit-gap (fn [out from to] (if (< from to) (conj out (subs s from to)) out))]
    (loop [i 0 start 0 out []]
      (if (>= i n)
        (emit-gap out start n)
        (let [lim (min max-added (- n i))
              hit (loop [l lim]
                    (when (pos? l)
                      (let [sub (subs s i (+ i l))]
                        (if (contains? added sub) l (recur (dec l))))))]
          (if hit
            (let [content (subs s i (+ i hit))
                  from (if (contains? lstrip content)
                         (loop [k i] (if (and (> k start) (space? (nth s (dec k)))) (recur (dec k)) k))
                         i)]
              (recur (+ i hit) (+ i hit)
                     (conj (emit-gap out start from) (get added content))))
            (recur (inc i) start out)))))))

(defn metaspace-pieces
  "One gap -> BPE words, as tokenizers' Metaspace(prepend_scheme=always,
  split=true) after the Replace(\" \" -> \u2581) normalizer: every space is
  a \u2581, one more is prepended unless the text already starts with one,
  and the text is split so each piece starts with its \u2581 (an empty gap
  yields nothing: prepend on an empty string is a no-op)."
  [seg]
  (let [s (str/replace seg " " lower-one-eighth-block)]
    (if (empty? s)
      []
      (let [s (if (str/starts-with? s lower-one-eighth-block) s (str lower-one-eighth-block s))
            n (count s)]
        (loop [i 1 from 0 out []]
          (cond
            (>= i n) (conj out (subs s from n))
            (= (nth s i) (first lower-one-eighth-block)) (recur (inc i) i (conj out (subs s from i)))
            :else (recur (inc i) from out)))))))

(defn sp-symbols
  "BPE::merge_word's starting symbols for one piece: each char that is a
  vocab token as itself; otherwise its UTF-8 bytes as <0xNN> tokens when
  byte fallback is on and every one exists; otherwise the unk token, with
  consecutive unks fused into one when fuse-unk is on."
  [piece vocab byte-fallback? fuse-unk? unk]
  (loop [cs (seq piece) out [] unk-run? false]
    (if (empty? cs)
      out
      (let [c (str (first cs))]
        (if (contains? vocab c)
          (recur (rest cs) (conj out c) false)
          (let [bytes (when byte-fallback?
                        (let [ts (map #(format "<0x%02X>" %) (string->utf8-bytes c))]
                          (when (every? #(contains? vocab %) ts) ts)))]
            (cond
              bytes (recur (rest cs) (into out bytes) false)
              (and unk-run? fuse-unk?) (recur (rest cs) out true)
              :else (recur (rest cs) (conj out unk) true))))))))

(defn- sentencepiece-encode [tok text]
  (let [vocab (:vocab tok)
        unk (some (fn [[t i]] (when (= i (:unk (:specials tok))) t)) (:added tok))]
    (into []
          (mapcat (fn [seg]
                    (if (string? seg)
                      (mapcat (fn [piece]
                                (->> (sp-symbols piece vocab (:byte-fallback tok) (:fuse-unk tok) unk)
                                     (#(apply-bpe % (:ranks tok)))
                                     (keep #(get vocab %))))
                              (metaspace-pieces seg))
                      [seg])))
          (sp-segments text (:added tok) (:max-added tok) (:lstrip tok)))))

(defn encode
  "Text -> input_ids (no special tokens), matching
  tok(text, add_special_tokens=False)['input_ids'] for the tokenizer's kind."
  [tok text]
  (case (:kind tok)
    :sentencepiece (sentencepiece-encode tok text)
    (byte-level-encode tok text)))

(defn special
  "id of :cls/:sep/:pad/:mask/:unk."
  [tok k]
  (get (:specials tok) k))
