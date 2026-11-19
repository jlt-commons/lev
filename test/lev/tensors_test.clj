(ns lev.tensors-test
  "Parity of the C kernels and the sgemm matmul against the torch oracle.

  golden/masks.edn  - exact allowed-attention matrices from transformers'
                      own mask builders (padded batch, lens [64 45])
  golden/layers/*   - f32 sidecars from the torch oracle dump (ids, embeddings)
  data/model/*      - the checkpoint's own weights, upcast f32"
  (:require [clojure.edn :as edn]
            [clojure.test :refer [deftest is testing]]
            [jolt.ffi]
            [lev.model :as lev.model]
            [lev.tensors :as t]
            [lev.test-util :as tu]))

(def golden-dir
  (or (System/getenv "LEV_GOLDEN") "golden"))

(def data-dir
  (or (System/getenv "LEV_DATA") "data"))

(defn- slurp-edn [f]
  (edn/read-string (slurp f)))

(defn- read-golden-f32 [k edn]
  (let [{:keys [shape file]} (get edn k)]
    (t/load-file (str golden-dir "/" file) shape)))

(defn max-abs-diff [a b]
  (let [n (t/size a)
        na (t/ptr a)
        nb (t/ptr b)]
    (loop [i 0, m 0.0]
      (if (= i n)
        m
        (let [d (Math/abs (- (t/get na i) (t/get nb i)))]
          (recur (inc i) (max m d)))))))

(defn rel-l2 [a b]
  (let [n (t/size a)
        na (t/ptr a)
        nb (t/ptr b)]
    (loop [i 0, num 0.0, den 0.0]
      (if (= i n)
        (Math/sqrt (/ num (max den 1e-30)))
        (let [x (t/get na i)
              y (t/get nb i)
              d (- x y)]
          (recur (inc i) (+ num (* d d)) (+ den (* y y))))))))

;; --- the attention masks transformers itself builds -------------------------

(deftest mask-builder-matches-transformers
  (let [m (slurp-edn (str golden-dir "/masks.edn"))
        lens (vec (m :lens))
        L 64
        att (t/from-bytes (concat (repeat L 1)
                                  (repeat (lens 1) 1)
                                  (repeat (- L (lens 1)) 0))
                          [2 L])]
    (is (= [2 L] (t/shape att)) "att rows for the padded batch")
    (doseq [k [:full :sliding]]
      (let [gold (t/from-bytes (flatten (first (get m k))) [L L])
            got (t/allowed-mask att 1 L (if (= k :sliding) 64 -1))]
        (testing (str k " allowed matrix, padded batch row 1")
          (is (t/eq-bytes? gold got)
              (str "first mismatch " (t/first-mismatch gold got))))))))

(deftest first-mismatch-finds-the-first-differing-byte
  (let [a (t/from-bytes [1 0 1 1] [4])]
    (is (= :none (t/first-mismatch a (t/from-bytes [1 0 1 1] [4]))))
    (is (= [0 1 0] (t/first-mismatch a (t/from-bytes [0 0 1 1] [4]))))
    (is (= [2 1 0] (t/first-mismatch a (t/from-bytes [1 0 0 1] [4]))))
    (is (= [3 1 0] (t/first-mismatch a (t/from-bytes [1 0 1 0] [4]))))))

;; --- embeddings: row gather + weight-only LayerNorm -------------------------

(deftest embeddings-match-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        emb-w (t/load-tensor manifest data-dir "encoder.embeddings.tok_embeddings.weight")
        norm-w (t/load-tensor manifest data-dir "encoder.embeddings.norm.weight")
        ids (map int (flatten (layers :ids)))
        n-tokens (count (flatten (layers :att)))
        got (t/embeddings emb-w (t/from-ints ids) n-tokens (cfg :hidden-size) norm-w)
        gold (read-golden-f32 :embeddings layers)]
    (is (< (max-abs-diff got gold) 1e-4)
        (str "embeddings max-abs " (max-abs-diff got gold)))))

;; --- matmul: sgemm vs a golden torch product ---------------------------------

(deftest sgemm-matches-torch-linear
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        gold (read-golden-f32 :matmul layers)
        X (read-golden-f32 :matmul-x layers)
        W (t/load-tensor manifest data-dir "scorer.1.weight")
        got (t/mmul X W)]
    (is (= (t/shape gold) (t/shape got)))
    (is (< (rel-l2 got gold) 1e-5)
        (str "sgemm rel-l2 " (rel-l2 got gold)))))

;; --- encoder: layer-by-layer vs golden ---------------------------------------

;; Bounds on the residual stream. Early layers sit at f32 noise. By layer 27
;; the outlier dimensions reach ~3e4, where an f32 ulp is 2e-3, and sgemm
;; summation order shows: Accelerate lands at 0.008 (english) to 0.25
;; (typed-decisions), OpenBLAS on the linux runners higher still, so the
;; bound is relative to the stream's scale (english 3e-7, typed-decisions
;; 9e-6 on Accelerate). The head layers, after their LayerNorm, are back to
;; O(1) values: ~5e-4 on Accelerate, 1.7e-3 on OpenBLAS.
(def ^:private last-layer-rel-tol 5e-5)
(def ^:private head-layer-tol 5e-3)

(deftest encoder-matches-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        w (lev.model/load-weights manifest data-dir)
        ids0 (t/from-ints (map int (first (layers :ids))))
        att0 (t/from-bytes (first (layers :att)) [74])
        gold-emb (read-golden-f32 :embeddings layers)
        emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                          ids0 74 1024 (w "encoder.embeddings.norm.weight"))
        full (t/allowed-mask att0 0 74 -1)
        slid (t/allowed-mask att0 0 74 64)
        mx (fn [a gold off n]
             (loop [i 0 r 0.0]
               (if (= i n) r
                   (recur (inc i)
                          (max r (Math/abs (- (t/get (t/ptr a) i)
                                              (t/get (t/ptr gold) (+ off i)))))))))]
    (is (< (mx emb gold-emb 0 (* 74 1024)) 1e-4)
        "embeddings row 0 within 1e-4 of torch")
    (doseq [i [0 1 2 27]]
      (let [out (reduce (fn [h k] (if (= k i)
                                    (reduced (lev.model/encoder-layer! w cfg k h [[full slid]]))
                                    (lev.model/encoder-layer! w cfg k h [[full slid]])))
                        emb (range (inc i)))
            gold (read-golden-f32 (keyword (str "layer-" i)) layers)]
        (if (= i 27)
          (let [[rel diff scale] (tu/relative-max-abs out gold (* 74 1024))]
            (is (< rel last-layer-rel-tol)
                (format "layer 27: max-abs %.4g of a %.4g-scale stream (rel %.2e)" diff scale rel)))
          (is (< (mx out gold 0 (* 74 1024)) 1e-4)
              (str "layer " i " max-abs " (mx out gold 0 (* 74 1024)))))))
    (testing "padded batch row 1 (64 real tokens): real rows match, pad rows ignored"
      (let [ids1 (t/from-ints (map int (second (layers :ids))))
            att1 (t/from-bytes (second (layers :att)) [74])
            emb1 (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                               ids1 74 1024 (w "encoder.embeddings.norm.weight"))
            full1 (t/allowed-mask att1 0 74 -1)
            slid1 (t/allowed-mask att1 0 74 64)
            n-real (* 64 1024)
            off (* 74 1024)]
        (is (< (mx emb1 gold-emb off n-real) 1e-4) "embeddings row 1")
        (reduce (fn [h k]
                  (let [h2 (lev.model/encoder-layer! w cfg k h [[full1 slid1]])]
                    (when (#{0 1 2 27} k)
                      (let [gold (read-golden-f32 (keyword (str "layer-" k)) layers)]
                        (if (= k 27)
                          ;; row 1 sits at offset `off` in the golden; compare through a view
                          (let [gold-row {:p (jolt.ffi/segment (+ (jolt.ffi/address (t/ptr gold)) (* off 4)))}
                                [rel diff scale] (tu/relative-max-abs h2 gold-row n-real)]
                            (is (< rel last-layer-rel-tol)
                                (format "row 1 layer 27: max-abs %.4g of a %.4g-scale stream (rel %.2e)" diff scale rel)))
                          (is (< (mx h2 gold off n-real) 1e-4)
                              (str "row 1 layer " k " max-abs " (mx h2 gold off n-real))))))
                    h2))
                emb1 (range 28))))))

;; --- head: TransformerEncoderLayers + scorer + ReLU discrimination ----------

(deftest head-matches-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        w (lev.model/load-weights manifest data-dir)
        att0 (t/from-bytes (first (layers :att)) [74])
        full (t/allowed-mask att0 0 74 -1)
        gold (fn [k shape] (t/load-file (str golden-dir "/layers/" k ".f32") shape))
        hin (let [g (gold "head-in" [2 74 1024])]
              {:p (t/ptr g) :shape [74 1024] :size (* 74 1024)})
        h0 (lev.model/head-layer! w cfg 0 hin [full])
        h1 (lev.model/head-layer! w cfg 1 h0 [full])
        mx (fn [a k shape]
             (let [g (gold k shape)]
               (loop [i 0 r 0.0]
                 (if (= i (* 74 1024)) r
                     (recur (inc i)
                            (max r (Math/abs (- (t/get (t/ptr a) i)
                                                (t/get (t/ptr g) i)))))))))]
    (is (< (mx h0 "head-layer-0" [2 74 1024]) head-layer-tol)
        (str "head layer 0 max-abs " (mx h0 "head-layer-0" [2 74 1024])))
    (is (< (mx h1 "head-layer-1" [2 74 1024]) head-layer-tol)
        (str "head layer 1 max-abs " (mx h1 "head-layer-1" [2 74 1024])))
    (let [markers (t/reshape (gold "markers" [2 4 1024]) [8 1024])
          logits (lev.model/scorer w markers)
          gold-logits (gold "logits" [2 4])
          ldiff (loop [i 0 r 0.0]
                  (if (= i 4) r
                      (recur (inc i)
                             (max r (Math/abs (- (t/get (t/ptr logits) i)
                                                 (t/get (t/ptr gold-logits) i)))))))
          gelu-logits (gold "gelu-logits" [2 4])
          ggap (loop [i 0 r 0.0]
                 (if (= i 4) r
                     (recur (inc i)
                            (max r (Math/abs (- (t/get (t/ptr logits) i)
                                                (t/get (t/ptr gelu-logits) i)))))))]
      (is (< ldiff 1e-3) (str "logits max-abs " ldiff))
      (is (> ggap (* 0.9 (layers :relu-vs-gelu-gap)))
          "jolt head is ReLU: it must sit a full gap away from the GELU trace"))
    (testing "padded row 1 through the head with its key-padding mask"
      (let [att1 (t/from-bytes (second (layers :att)) [74])
            full1 (t/allowed-mask att1 0 74 -1)
            g (gold "head-in" [2 74 1024])
            hin1 {:p (jolt.ffi/segment (+ (jolt.ffi/address (t/ptr g)) (* 74 1024 4)))
                  :shape [74 1024] :size (* 74 1024)}
            h1 (lev.model/head-layer! w cfg 1 (lev.model/head-layer! w cfg 0 hin1 [full1]) [full1])
            gold1 (gold "head-layer-1" [2 74 1024])
            mx1 (loop [i 0 r 0.0]
                  (if (= i (* 64 1024)) r
                      (recur (inc i)
                             (max r (Math/abs (- (t/get (t/ptr h1) i)
                                                 (t/get (t/ptr gold1) (+ (* 74 1024) i))))))))
            m1 (t/gather h1 (t/from-ints (take 2 (second (layers :marker-pos)))) 2 1024)
            lg1 (t/to-floats (lev.model/scorer w m1))
            gold-logits (gold "logits" [2 4])]
        (is (< mx1 head-layer-tol) (str "row 1 head layer 1 max-abs " mx1))
        (is (< (Math/abs (- (first lg1) (t/get (t/ptr gold-logits) 4))) 1e-3))
        (is (< (Math/abs (- (second lg1) (t/get (t/ptr gold-logits) 5))) 1e-3))))))

;; --- act-head features: top1, top1-top2, normalized entropy, k/255 ---------

(deftest act-features-match-torch
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        gold-logits (t/load-file (str golden-dir "/layers/logits.f32") [2 4])
        feats (layers :act-feats)]
    (doseq [r [0 1]]
      (let [mask (nth (layers :marker-mask) r)
            k (count (filter pos? mask))
            ;; per-row forward has no padded markers: only the k real logits
            lg (mapv #(t/get (t/ptr gold-logits) (+ (* r 4) %)) (range k))
            got (lev.model/answer-features (t/from-floats lg) (repeat k 1))
            want (nth feats r)]
        (doseq [[i name] [[0 "top1"] [1 "top1-top2"] [2 "entropy"] [3 "k/255"]]]
          (is (< (Math/abs (- (double (nth got i)) (double (nth want i)))) 1e-5)
              (str "row " r " " name " got " (nth got i) " want " (nth want i))))))))

(deftest top2-of-any-order
  (is (= [0.5 0.3] (lev.model/top2 [0.5 0.3 0.1 0.1])) "descending input")
  (is (= [0.5 0.3] (lev.model/top2 [0.1 0.3 0.5])) "ascending input")
  (is (= [0.5 0.3] (lev.model/top2 [0.3 0.5 0.1])) "mixed")
  (is (= [0.4 0.4] (lev.model/top2 [0.4 0.2 0.4])) "ties keep both"))

;; --- attention: per-head sgemm + masked softmax vs a double reference ------

(defn- lcg-floats
  "Deterministic pseudo-random floats in [-0.5, 0.5): no java.util.Random
  needed, and the same numbers on every platform."
  [seed n]
  (loop [x (long seed), i 0, acc (transient [])]
    (if (= i n)
      (persistent! acc)
      ;; rand_r's 31-bit recurrence: the product stays well inside a long
      (let [x (mod (+ (* x 1103515245) 12345) 2147483648)]
        (recur x (inc i) (conj! acc (- (/ (double x) 2147483648.0) 0.5)))))))

(defn- ref-attention
  "softmax(scale * q k^T, over allowed keys) v, per head, in double precision.
  qh/kh/vh flat head-major [H x L x hd]; allowed flat [L x L] 0/1.
  Answers the token-major [L x (H*hd)] context as a flat vector."
  [qh kh vh allowed H L hd scale]
  (let [at (fn [xs h t x] (double (nth xs (+ (* (+ (* h L) t) hd) x))))]
    (vec
     (for [i (range L) h (range H) x (range hd)]
       (let [js (filter #(pos? (nth allowed (+ (* i L) %))) (range L))
             s (map (fn [j] (* scale (reduce + (map #(* (at qh h i %) (at kh h j %)) (range hd))))) js)]
         (if (empty? js)
           0.0
           (let [m (reduce max s)
                 e (map #(Math/exp (- % m)) s)
                 z (reduce + e)]
             (reduce + (map (fn [j ej] (* (/ ej z) (at vh h j x))) js e)))))))))

(deftest attention-matches-double-reference
  (let [H 2 L 7 hd 4 d (* H hd)
        scale (/ 1.0 (Math/sqrt hd))
        qh (lcg-floats 1 (* H L hd))
        kh (lcg-floats 2 (* H L hd))
        vh (lcg-floats 3 (* H L hd))
        ;; token 6 is padding; window 2 makes it a banded matrix too
        att (t/from-bytes [1 1 1 1 1 1 0] [1 L])
        run (fn [allowed]
              (let [ctx (t/attention (t/reshape (t/from-floats qh) [(* H L) hd])
                                     (t/reshape (t/from-floats kh) [(* H L) hd])
                                     (t/reshape (t/from-floats vh) [(* H L) hd])
                                     allowed H L hd scale)
                    ;; the allowed matrix is uint8, not f32: read it as bytes
                    want (ref-attention qh kh vh (mapv #(jolt.ffi/read (t/ptr allowed) :uint8 %) (range (* L L))) H L hd scale)]
                (is (= [L d] (t/shape ctx)))
                (reduce max (map #(Math/abs (- (double %1) %2)) (t/to-floats ctx) want))))]
    (is (< (run (t/allowed-mask att 0 L -1)) 1e-6) "full attention with a padded key")
    (is (< (run (t/allowed-mask att 0 L 2)) 1e-6) "sliding window |i-j|<=2")
    (testing "a query row with no allowed key (the pad row under a window) is left at zero"
      (let [allowed (t/allowed-mask att 0 L 0)   ; only the diagonal, and the pad column is masked
            ctx (t/attention (t/reshape (t/from-floats qh) [(* H L) hd])
                             (t/reshape (t/from-floats kh) [(* H L) hd])
                             (t/reshape (t/from-floats vh) [(* H L) hd])
                             allowed H L hd scale)]
        (is (every? zero? (subvec (t/to-floats ctx) (* 6 d) (* 7 d))))))))

(deftest banded-attention-matches-full-mask
  ;; sliding layers only need the |i-j|<=window band; the blocked path must
  ;; give the same context as scoring every key and masking (several blocks)
  (let [H 1 L 300 hd 4 d (* H hd) window 64
        scale (/ 1.0 (Math/sqrt hd))
        qh (lcg-floats 11 (* H L hd))
        kh (lcg-floats 12 (* H L hd))
        vh (lcg-floats 13 (* H L hd))
        att (t/from-bytes (concat (repeat 290 1) (repeat 10 0)) [1 L])   ; a padded tail too
        allowed (t/allowed-mask att 0 L window)
        mk (fn [xs] (t/reshape (t/from-floats xs) [(* H L) hd]))
        banded (t/attention (mk qh) (mk kh) (mk vh) allowed H L hd scale window)
        full (t/attention (mk qh) (mk kh) (mk vh) allowed H L hd scale)
        want (ref-attention qh kh vh (mapv #(jolt.ffi/read (t/ptr allowed) :uint8 %) (range (* L L))) H L hd scale)]
    (is (= [L d] (t/shape banded)))
    (is (< (reduce max (map #(Math/abs (- (double %1) %2)) (t/to-floats banded) want)) 1e-6)
        "banded vs double reference")
    (is (< (reduce max (map #(Math/abs (- (double %1) (double %2))) (t/to-floats banded) (t/to-floats full))) 1e-6)
        "banded vs unbanded")))

(deftest masked-softmax-matches-double-reference
  ;; the kernel's exp is a vectorizable polynomial: it has to hold to ~1 ulp
  ;; against libm across the whole range a max-subtracted score can take
  (let [rows 6 cols 200
        raw (lcg-floats 21 (* rows cols))
        ;; spread scores over [-95, 5]: row 0 spans the full range, later rows narrower
        scores (vec (map-indexed (fn [i x] (let [r (quot i cols)] (* (+ x 0.5) (- (/ 100.0 (inc r)))))) raw))
        allowed (vec (for [i (range rows) j (range cols)]
                       (cond (= i 5) 0                        ; a fully masked row
                             (zero? (mod (+ i j) 7)) 0         ; scattered masked keys
                             :else 1)))
        got (t/to-floats (t/masked-softmax (t/reshape (t/from-floats scores) [rows cols])
                                           (t/from-bytes allowed [rows cols]) rows cols))
        want (vec (for [i (range rows)]
                    (let [js (filter #(pos? (nth allowed (+ (* i cols) %))) (range cols))
                          s (map #(double (nth scores (+ (* i cols) %))) js)]
                      (if (empty? js)
                        (vec (repeat cols 0.0))
                        (let [m (reduce max s)
                              e (zipmap js (map #(Math/exp (- % m)) s))
                              z (reduce + (vals e))]
                          (mapv #(/ (get e % 0.0) z) (range cols)))))))
        want (vec (apply concat want))]
    (is (= (* rows cols) (count got)))
    (is (every? zero? (subvec got (* 5 cols))) "a row with no allowed key stays zero")
    (let [worst (reduce max (map (fn [g w] (/ (Math/abs (- (double g) w)) (+ 1e-9 w))) got want))]
      (is (< worst 2e-6) (str "worst relative error on a softmax weight " worst)))))

;; --- workspace: one set of intermediates per forward, reused by every layer --

(deftest workspace-layer-equals-allocating-layer
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        w (lev.model/load-weights manifest data-dir)
        L 74
        ids0 (t/from-ints (map int (first (layers :ids))))
        att0 (t/from-bytes (first (layers :att)) [L])
        emb (t/embeddings (w "encoder.embeddings.tok_embeddings.weight")
                          ids0 L 1024 (w "encoder.embeddings.norm.weight"))
        before (t/to-floats emb)
        full (t/allowed-mask att0 0 L -1)
        slid (t/allowed-mask att0 0 L 64)
        ws (lev.model/workspace cfg 1 L)
        plain (lev.model/encoder-layer! w cfg 0 emb [[full slid]])
        via-ws (lev.model/encoder-layer! w cfg 0 emb [[full slid]] ws)
        same? (fn [a b] (< (reduce max (map #(Math/abs (- (double %1) (double %2))) (t/to-floats a) (t/to-floats b))) 1e-6))]
    (is (same? plain via-ws) "a layer through the workspace matches the allocating layer")
    (is (= before (t/to-floats emb)) "the input h is not written")
    (testing "the residual stream alternates between the two workspace buffers"
      (let [next-ws (lev.model/encoder-layer! w cfg 1 via-ws [[full slid]] ws)
            next-plain (lev.model/encoder-layer! w cfg 1 plain [[full slid]])]
        (is (not= (jolt.ffi/address (t/ptr via-ws)) (jolt.ffi/address (t/ptr next-ws))) "layer i+1 does not overwrite its input")
        (is (same? next-plain next-ws))
        ;; layer 0 wrote into one residual buffer, layer 1 into the other; layer 2 reuses the first
        (is (= (jolt.ffi/address (t/ptr via-ws))
               (jolt.ffi/address (t/ptr (lev.model/encoder-layer! w cfg 2 next-ws [[full slid]] ws)))))))
    (testing "the head layer takes the same workspace"
      (let [h0 (lev.model/head-layer! w cfg 0 plain [full])
            h0-ws (lev.model/head-layer! w cfg 0 plain [full] ws)]
        (is (same? h0 h0-ws))))))

;; --- elementwise kernels vs libm / double references ------------------------

(deftest swiglu-matches-libm-gelu
  ;; swiglu's erf is a vectorizable polynomial (NR erfcc over exp_fast);
  ;; lev_gelu keeps libm erff and is the oracle: swiglu(x) == gelu(a) * g
  ;; to a few f32 ulps. The floor of 1e-6 is for the deep negative tail
  ;; (a < -4, outputs ~1e-6), where the oracle's own 1 + erff(u) in f32
  ;; is the inaccurate side.
  (let [n 40 mid 300
        xs (mapv #(* 16.0 %) (lcg-floats 31 (* n 2 mid)))      ; a, g in [-8, 8)
        x (t/reshape (t/from-floats xs) [n (* 2 mid)])
        got (t/to-floats (t/swiglu x mid))
        a (t/from-floats (for [i (range n) j (range mid)] (nth xs (+ (* i 2 mid) j))))
        g (for [i (range n) j (range mid)] (double (nth xs (+ (* i 2 mid) mid j))))
        want (map * (t/to-floats (t/gelu! a)) g)
        worst (reduce max (map (fn [got want] (/ (Math/abs (- (double got) want))
                                                  (+ 1e-6 (* 5e-7 (Math/abs want)))))
                               got want))]
    (is (= (* n mid) (count got)))
    (is (< worst 1.0) (str "worst error in units of (1e-6 + 5e-7 |ref|): " worst))))

(deftest layernorm-matches-double-reference
  (let [n 20 d 1024
        xs (mapv #(* 40.0 %) (lcg-floats 41 (* n d)))          ; stream-like magnitudes
        w (mapv #(+ 1.0 %) (lcg-floats 42 d))
        b (lcg-floats 43 d)
        eps 1e-5
        got-b (t/to-floats (t/layernorm (t/reshape (t/from-floats xs) [n d]) (t/from-floats w) (t/from-floats b) eps))
        got (t/to-floats (t/layernorm (t/reshape (t/from-floats xs) [n d]) (t/from-floats w) eps))
        want (fn [bias?]
               (vec (apply concat
                           (for [i (range n)]
                             (let [row (subvec xs (* i d) (* (inc i) d))
                                   ;; the f32 inputs the kernel sees, in double
                                   row (mapv #(double (float %)) row)
                                   mean (/ (reduce + row) d)
                                   var (/ (reduce + (map #(* (- % mean) (- % mean)) row)) d)
                                   inv (/ 1.0 (Math/sqrt (+ var eps)))]
                               (map-indexed (fn [j v] (+ (* (- v mean) inv (double (float (nth w j))))
                                                         (if bias? (double (float (nth b j))) 0.0)))
                                            row))))))
        err (fn [got want] (reduce max (map #(Math/abs (- (double %1) %2)) got want)))]
    (is (< (err got (want false)) 1e-5) "weight-only layernorm")
    (is (< (err got-b (want true)) 1e-5) "layernorm with bias")))

;; --- the kernel thread pool -------------------------------------------------

(deftest kernels-are-bit-identical-on-any-thread-count
  ;; the pool only changes the schedule: every (head, block) or row task
  ;; runs the same arithmetic in the same order, so the bytes must not
  ;; move with the thread count, and one thread must equal the serial
  ;; result the double-precision references above pin
  (let [H 16 L 300 hd 64 d (* H hd) window 64 scale 0.125
        mk (fn [seed n shape] (t/reshape (t/from-floats (lcg-floats seed n)) shape))
        qh (mk 51 (* H L hd) [(* H L) hd])
        kh (mk 52 (* H L hd) [(* H L) hd])
        vh (mk 53 (* H L hd) [(* H L) hd])
        att (t/from-bytes (concat (repeat 290 1) (repeat 10 0)) [1 L])
        full (t/allowed-mask att 0 L -1)
        slid (t/allowed-mask att 0 L window)
        x (mk 54 (* L 600) [L 600])
        lx (mk 55 (* L d) [L d])
        w (t/from-floats (map inc (lcg-floats 56 d)))
        run (fn [] {:full (t/to-floats (t/attention qh kh vh full H L hd scale))
                    :slid (t/to-floats (t/attention qh kh vh slid H L hd scale window))
                    :sw (t/to-floats (t/swiglu x 300))
                    :ln (t/to-floats (t/layernorm lx w 1e-5))})
        before (t/threads)]
    (is (pos? before) "the pool has a size before anyone asks")
    (try
      (t/set-threads! 1)
      (is (= 1 (t/threads)))
      (let [one (run)]
        (is (= (* L d) (count (:full one))))
        (doseq [n [2 3 8]]
          (t/set-threads! n)
          (is (= n (t/threads)))
          (let [many (run)]
            (doseq [k [:full :slid :sw :ln]]
              (is (= (get one k) (get many k)) (str n " threads: " (name k)))))))
      (finally (t/set-threads! before)))
    (is (= before (t/threads)))))

;; --- batched forward: B padded rows through one gemm stream --------------

(deftest batched-forward-equals-row-forwards
  ;; the golden batch: row 0 has 74 real tokens, row 1 has 64 (+10 pad);
  ;; a choice question and a noul one. Batched, the rows share every gemm
  ;; and only attention is per row; the real tokens must come out as if
  ;; each row ran alone, and as torch's padded batch did (golden logits).
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        w (lev.model/load-weights manifest data-dir)
        lens [74 64]
        rows (vec (for [r [0 1]]
                    (let [L (lens r)
                          k (count (filter pos? (nth (layers :marker-mask) r)))]
                      {:ids (vec (take L (map int (nth (layers :ids) r))))
                       :att (vec (repeat L 1))
                       :markers (vec (take k (nth (layers :marker-pos) r)))
                       :marker-mask (vec (repeat k 1))
                       :qtype (nth (layers :qtype) r)})))
        alone (mapv (fn [{:keys [ids att markers marker-mask qtype]}]
                      (lev.model/forward-row w cfg ids att markers marker-mask qtype))
                    rows)
        batched (lev.model/forward-batch w cfg rows)
        gold-logits (t/load-file (str golden-dir "/layers/logits.f32") [2 4])
        ;; relative past magnitude 1: the act logits sit around 4000 (the
        ;; head is saturated), where an f32 ulp is 5e-4, and a BLAS picks a
        ;; different summation order for a different M (OpenBLAS: batched
        ;; vs alone differ by ~3e-7 of the value; Accelerate: not at all)
        close (fn [a b tol] (every? true? (map #(<= (Math/abs (- (double %1) (double %2)))
                                                    (* tol (max 1.0 (Math/abs (double %2)))))
                                               a b)))]
    (is (= 2 (count batched)))
    (doseq [r [0 1]]
      (let [[lg act] (nth batched r)
            [lg1 act1] (nth alone r)
            k (count (:markers (nth rows r)))]
        (is (= k (count lg)) (str "row " r " answers one logit per marker"))
        (is (close lg lg1 1e-4) (str "row " r " logits: batched " lg " alone " lg1))
        (is (close act act1 1e-4) (str "row " r " act logits: batched " act " alone " act1))
        (is (close lg (map #(t/get (t/ptr gold-logits) (+ (* r 4) %)) (range k)) 1e-3)
            (str "row " r " logits vs torch's padded batch"))))
    (testing "row order and batch size do not matter"
      (let [[[lg-b act-b]] (lev.model/forward-batch w cfg [(rows 1)])
            swapped (lev.model/forward-batch w cfg [(rows 1) (rows 0)])]
        (is (close lg-b (first (nth batched 1)) 1e-5))
        (is (close (first (second swapped)) (first (nth batched 0)) 1e-5))
        (is (close (second (first swapped)) (second (nth batched 1)) 1e-5))))))

(deftest selected-head-equals-full-head
  ;; the last head layer's out-projection and FFN only at the CLS and
  ;; marker rows (the default; :selected-head false is the full layer):
  ;; no other row of its output is read, so the logits and act logits
  ;; must be the full layer's (laya-mlx's selected_head, exact
  ;; dependency pruning)
  (let [layers (slurp-edn (str golden-dir "/layers.edn"))
        manifest (edn/read-string (slurp (str data-dir "/manifest.edn")))
        cfg (edn/read-string (slurp (str data-dir "/config.edn")))
        w (lev.model/load-weights manifest data-dir)
        lens [74 64]
        rows (vec (for [r [0 1]]
                    (let [L (lens r)
                          k (count (filter pos? (nth (layers :marker-mask) r)))]
                      {:ids (vec (take L (map int (nth (layers :ids) r))))
                       :att (vec (repeat L 1))
                       :markers (vec (take k (nth (layers :marker-pos) r)))
                       :marker-mask (vec (repeat k 1))
                       :qtype (nth (layers :qtype) r)})))
        close (fn [a b tol] (every? true? (map #(<= (Math/abs (- (double %1) (double %2)))
                                                    (* tol (max 1.0 (Math/abs (double %2)))))
                                               a b)))
        full (lev.model/forward-batch w (assoc cfg :selected-head false) rows)
        selected (lev.model/forward-batch w cfg rows)]
    (is (= 2 (count selected)))
    (doseq [r [0 1]]
      (let [[lg act] (nth selected r)
            [lg-f act-f] (nth full r)]
        (is (= (count lg-f) (count lg)) (str "row " r " answers one logit per marker"))
        (is (close lg lg-f 1e-4) (str "row " r " logits: selected " lg " full " lg-f))
        (is (close act act-f 1e-4) (str "row " r " act logits: selected " act " full " act-f))))
    (testing "a batch of one and a swapped batch"
      (let [[[lg-b act-b]] (lev.model/forward-batch w cfg [(rows 1)])
            swapped (lev.model/forward-batch w cfg [(rows 1) (rows 0)])]
        (is (close lg-b (first (nth full 1)) 1e-4))
        (is (close act-b (second (nth full 1)) 1e-4))
        (is (close (first (second swapped)) (first (nth full 0)) 1e-4))
        (is (close (first (first swapped)) (first (nth full 1)) 1e-4))))))

(deftest mean-rows-averages-the-rows
  (let [x (assoc (t/from-floats [1 2 3, 3 4 5]) :shape [2 3])]
    (is (tu/approx= 1e-6 [2.0 3.0 4.0] (t/to-floats (t/mean-rows x))))
    (is (tu/approx= 1e-6 [3.0 4.0 5.0] (t/to-floats (t/mean-rows (t/rows x 1 1)))) "on a view")))
