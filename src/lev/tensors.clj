(ns lev.tensors
  "f32 tensors as (ffi) buffer views + the C kernel/BLAS ops the model needs.

  A tensor is {:p ptr :shape [rows cols] :size n} — raw pointers into ffi
  buffers, row-major, little-endian f32. Matmuls call cblas_sgemm through
  the blas native (Accelerate on mac, OpenBLAS on linux); the elementwise
  and reduction ops are native/lev_kernels.c, whose row-wise kernels and
  attention run on the library's own thread pool (`threads`,
  `set-threads!`, LEV_THREADS). Nothing per-element crosses back into
  Clojure."
  (:require [clojure.java.io :as io]
            [jolt.ffi :as ffi]))

;; --- cblas (Accelerate / OpenBLAS) -------------------------------------------
;; CBLAS_ROW_MAJOR=101. sgemm computes C = alpha*A*B + beta*C, all row-major
;; f32. Our matmuls are always out = X @ W^T with W stored [out-dim x in-dim]
;; (torch Linear convention), so W is transposed via the TransB=1 flag.

(ffi/defcfn cblas-sgemm* "cblas_sgemm"
  [:int :int :int :int64 :int64 :int64
   :float :pointer :int64 :pointer :int64
   :float :pointer :int64] :void)

(ffi/defcfn cblas-sgemv* "cblas_sgemv"
  [:int :int :int64 :int64
   :float :pointer :int64 :pointer :int64
   :float :pointer :int64] :void)

(def ^:private RowMajor 101)
(def ^:private NoTrans 111)
(def ^:private Trans 112)

;; --- kernels (native/lev_kernels.c) ------------------------------------------

(ffi/defcfn gather-rows* "lev_gather_rows"
  [:pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn layernorm* "lev_layernorm"
  [:pointer :pointer :pointer :int64 :int64 :float :pointer] :void)
(ffi/defcfn gelu* "lev_gelu" [:pointer :int64 :pointer] :void)
(ffi/defcfn relu* "lev_relu" [:pointer :int64 :pointer] :void)
(ffi/defcfn silu* "lev_silu" [:pointer :int64 :pointer] :void)
(ffi/defcfn swiglu* "lev_swiglu" [:pointer :int64 :int64 :pointer] :void)
(ffi/defcfn rope-tables* "lev_rope_tables"
  [:double :int64 :int64 :pointer :pointer] :void)
(ffi/defcfn rope-apply* "lev_rope_apply"
  [:pointer :pointer :pointer :int64 :int64 :int64] :void)
(ffi/defcfn split-heads* "lev_split_heads"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn merge-heads* "lev_merge_heads"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn allowed-mask* "lev_allowed_mask"
  [:pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn masked-softmax* "lev_masked_softmax"
  [:pointer :pointer :int64 :int64 :int64 :pointer] :void)
(ffi/defcfn add-scaled* "lev_add_scaled"
  [:pointer :pointer :int64 :int64 :float :pointer] :void)
(ffi/defcfn add-qtype-bias* "lev_add_qtype_bias"
  [:pointer :pointer :pointer :int64 :int64 :pointer] :void)
(ffi/defcfn softmax* "lev_softmax" [:pointer :int64 :int64 :pointer] :void)
(ffi/defcfn attention* "lev_attention"
  [:pointer :pointer :pointer :pointer :pointer :int64 :int64 :int64 :float :int64 :int64 :pointer] :void)
(ffi/defcfn threads* "lev_threads" [] :int)
(ffi/defcfn set-threads* "lev_set_threads" [:int] :void)

(defn threads
  "Threads the kernel pool runs on (the caller included): LEV_THREADS,
  else the online processors, until set-threads! says otherwise."
  []
  (threads*))

(defn set-threads!
  "Resize the kernel pool; 1 runs everything on the calling thread. The
  results do not depend on it, only the time."
  [n]
  (set-threads* (int (max 1 n)))
  n)

(defn available?
  "Are the kernels and a BLAS loaded? Both are declared natives, BLAS an
  optional one: without them nothing here runs."
  []
  (boolean (and (ffi/find-symbol "lev_set_threads") (ffi/find-symbol "cblas_sgemm"))))

(def ^:private sgemm-address
  "cblas_sgemm as the kernel library reaches it: the same BLAS entry the
  binding above calls, found once through the declared natives."
  (delay (or (ffi/find-symbol "cblas_sgemm")
             (throw (ex-info "cblas_sgemm not found in the loaded natives" {})))))

;; --- tensor plumbing ----------------------------------------------------------

(defn ptr [t] (:p t))
(defn shape [t] (:shape t))
(defn size [t] (:size t))

(def ^:dynamic *arena*
  "When bound to an ffi arena, every tensor made here is owned by it and is
  released when the arena closes. A forward pass allocates hundreds of MB of
  intermediates; model/forward-row binds one arena per row so none of it
  outlives the call. Unbound (nil) means caller-owned malloc, which is what
  weights and cached rope tables want."
  nil)

(defn alloc-bytes
  "Zeroed native memory, owned by *arena* when one is bound."
  [n]
  (if *arena* (ffi/alloc *arena* n) (ffi/alloc n)))

(defn make
  [shape]
  {:p (alloc-bytes (* 4 (apply * shape)))
   :shape (vec shape)
   :size (apply * shape)})

(defn get*
  "Read element i (pointer, not tensor)."
  [p i]
  (ffi/read p :float (* 4 i)))

;; every :float argument here is passed as a double: (float x) is a boxed
;; Float, which jolt 0.8.20's FFI refuses
(defn set*
  [p i v]
  (ffi/write p :float (double v) (* 4 i)))

(defn from-floats
  [xs]
  (let [n (count xs)
        t (make [n])]
    (dotimes [i n]
      (set* (:p t) i (nth xs i)))
    t))

(defn from-ints
  "int64 buffer for ids (embedding lookup indices)."
  [xs]
  (let [n (count xs)
        t {:p (alloc-bytes (* 8 n)) :shape [n] :size n}]
    (dotimes [i n]
      (ffi/write (:p t) :int64 (long (nth xs i)) (* 8 i)))
    t))

(defn from-bytes
  "uint8 buffer from 0/1 values."
  ([xs] (from-bytes xs [(count xs)]))
  ([xs shape]
   (let [n (count xs)
         t {:p (alloc-bytes n) :shape (vec shape) :size (apply * shape)}]
     (dotimes [i n]
       (ffi/write (:p t) :uint8 (byte (nth xs i)) i))
     t)))

(defn to-floats
  [t]
  (mapv #(get* (:p t) %) (range (:size t))))

(defn load-file
  "Read a raw f32 file into a fresh tensor of the given shape."
  [path shape]
  (let [n (apply * shape)
        f (io/file path)
        expected (* 4 n)
        len (.length f)]
    (when-not (= len expected)
      (throw (ex-info "file size mismatch" {:path path :expected expected :got len})))
    ;; io/file + Files/readAllBytes shim: copy the raw bytes then write-array.
    ;; Weights live for the whole process: never arena-owned.
    (let [t (binding [*arena* nil] (make shape))
          bytes (java.nio.file.Files/readAllBytes (.toPath (io/file path)))]
      (ffi/write-array (:p t) bytes)
      t)))

(defn load-tensor
  "Load one named tensor from data/manifest.edn."
  [manifest data-dir name]
  (let [{:keys [shape file]} (get-in manifest [:tensors name])]
    (when-not shape
      (throw (ex-info "tensor not in manifest" {:name name})))
    (load-file (str data-dir "/" file) shape)))

;; --- ops ----------------------------------------------------------------------

(defn mmul!
  "out = X @ W^T into the given out [m x n]. X [m x k], W [n x k]. torch Linear."
  [out X W]
  (let [[m k] (:shape X)
        [n k2] (:shape W)]
    (when (or (nil? n) (not= k k2) (not= [m n] (:shape out)))
      (throw (ex-info "mmul shape mismatch" {:x (:shape X) :w (:shape W) :out (:shape out)})))
    (cblas-sgemm* RowMajor NoTrans Trans
                  (long m) (long n) (long k)
                  1.0 (:p X) (long k)
                  (:p W) (long k)
                  0.0 (:p out) (long n))
    out))

(defn mean-rows
  "The mean of the rows of t [n x d] as a [d] tensor: ones/n @ t, one
  sgemv."
  [t]
  (let [[n d] (:shape t)
        ones (from-floats (repeat n 1.0))
        out (make [d])]
    (cblas-sgemv* RowMajor Trans (long n) (long d)
                  (/ 1.0 (double n)) (:p t) (long d)
                  (:p ones) 1
                  0.0 (:p out) 1)
    out))

(defn mmul
  "out = X @ W^T. X [m x k], W [n x k] -> a new out [m x n]. torch Linear."
  [X W]
  (mmul! (make [(first (:shape X)) (first (:shape W))]) X W))

(defn embeddings
  "Gather rows then LayerNorm (weight-only, norm_bias=false) exactly as
  ModernBertEmbeddings: norm(tok_embeddings(ids))."
  [emb-w ids n d norm-w]
  (let [g (make [n d])
        out (make [n d])]
    (gather-rows* (:p emb-w) (:p ids) (long n) (long d) (:p g))
    (layernorm* (:p g) (:p norm-w) 0 (long n) (long d) 1e-5 (:p out))
    out))

(defn gather
  "Row gather: out[i,:] = src[ids[i],:]."
  [src ids n d]
  (let [out (make [n d])]
    (gather-rows* (:p src) (:p ids) (long n) (long d) (:p out))
    out))

(defn layernorm!
  "LayerNorm with optional bias (torch nn.LayerNorm), into out [n x d]."
  ([out x w eps] (layernorm! out x w 0 eps))
  ([out x w b eps]
   (let [[n d] (:shape x)
         bp (if (map? b) (long (:p b)) 0)]
     (layernorm* (:p x) (:p w) bp (long n) (long d) (double eps) (:p out))
     out)))

(defn layernorm
  "LayerNorm with optional bias (torch nn.LayerNorm), into a new tensor."
  ([x w eps] (layernorm x w 0 eps))
  ([x w b eps] (layernorm! (make (:shape x)) x w b eps)))

(defn gelu [x]
  (let [n (:size x) out (make (:shape x))]
    (gelu* (:p x) (long n) (:p out))
    out))

(defn relu [x]
  (let [n (:size x) out (make (:shape x))]
    (relu* (:p x) (long n) (:p out))
    out))

(defn swiglu!
  "in [n x 2*mid] -> out [n x mid], act(input)*gate with erf-gelu."
  [out x mid]
  (let [[n _] (:shape x)]
    (swiglu* (:p x) (long n) (long mid) (:p out))
    out))

(defn swiglu
  [x mid]
  (swiglu! (make [(first (:shape x)) mid]) x mid))

(defn rope-tables
  [theta d len]
  (let [cos-t (make [len d])
        sin-t (make [len d])]
    (rope-tables* (double theta) (long d) (long len) (:p cos-t) (:p sin-t))
    [cos-t sin-t]))

(defn rope-apply!
  "Apply rope to head-major q (and k): modifies q in place."
  [q cos-t sin-t n-heads L d]
  (rope-apply* (:p q) (:p cos-t) (:p sin-t) (long n-heads) (long L) (long d))
  q)

(defn split-heads
  [x L H hd]
  (let [out (make [(* H L) hd])]
    (split-heads* (:p x) (long L) (long H) (long hd) (:p out))
    out))

(defn merge-heads
  [src H L hd]
  (let [out (make [L (* H hd)])]
    (merge-heads* (:p src) (long H) (long L) (long hd) (:p out))
    out))

(defn byte-matrix
  "uint8 matrix [r x c] in a raw byte buffer."
  [r c]
  {:p (alloc-bytes (* r c)) :shape [r c] :size (* r c)})

(defn allowed-mask
  "Allowed matrix [L x L] for batch row b: full when window<0, else |i-j|<=window."
  [att b L window]
  (let [m (byte-matrix L L)]
    (allowed-mask* (:p att) (long b) (long L) (long window) (:p m))
    m))

(defn- at-offset
  "Pointer `bytes` past the start of tensor t (a view, not a copy)."
  [t bytes]
  (ffi/segment (+ (ffi/address (:p t)) bytes)))

(defn rows
  "A view of rows [r0, r0+n) of the [m x c] tensor t: no copy, same memory."
  [t r0 n]
  (let [c (second (:shape t))]
    {:p (at-offset t (* 4 r0 c)) :shape [n c] :size (* n c)}))

(defn masked-softmax
  "Row softmax over the allowed keys of a [rows x cols] score block; the
  mask is the top-left [rows x cols] of `allowed` (leading dimension cols
  here; `attention` passes a view into an [L x L] mask)."
  [scores allowed rows cols]
  (let [out (make [rows cols])]
    (masked-softmax* (:p scores) (:p allowed) (long cols) (long rows) (long cols) (:p out))
    out))

(def full-attention-block
  "Query rows per task on a full-attention layer: 128 gives 16 heads x 4
  blocks at L=512 for the pool, with [128 x L] score blocks per thread."
  128)

(defn attention!
  "softmax(scale * Q K^T over the allowed keys) V for every head. q/k/v
  head-major [H*L x hd] (lev_split_qkv's layout), allowed [L x L] bytes ->
  token-major ctx [L x (H*hd)], written into `ctx`.

  lev_attention runs (head, query block) tasks on the kernel pool, each
  two sgemm calls (through the cblas_sgemm the library is handed, the
  same two gemms torch's math-path SDPA runs; P V lands straight in the
  head's columns of ctx through ldc = H*hd) around a masked softmax on
  the thread's scratch block. A query row with no allowed key (padding
  under a window) stays zero.

  With a `window` (a sliding layer) the queries go in blocks of 2*window
  rows and each block only scores the keys within window of it, so an
  L=1024 layer touches 256 keys per query instead of 1024. The mask is
  still applied inside the block, so the band is only a saving, never the
  semantics; the full path scores every key in blocks of
  full-attention-block rows."
  [ctx qh kh vh allowed H L hd scale window]
  (attention* @sgemm-address (:p qh) (:p kh) (:p vh) (:p allowed)
              (long H) (long L) (long hd) (double scale) (long window) (long full-attention-block)
              (:p ctx))
  ctx)

(defn attention
  "attention! into a fresh context; see attention!."
  ([qh kh vh allowed H L hd scale] (attention qh kh vh allowed H L hd scale -1))
  ([qh kh vh allowed H L hd scale window]
   (attention! (make [L (* H hd)]) qh kh vh allowed H L hd scale window)))

(defn add-qtype!
  "h += type_emb[qtype] per row (in place). h [n x d], bias [3 x d]."
  [h bias qtype-ids n d]
  (add-qtype-bias* (:p h) (:p bias) (:p qtype-ids) (long n) (long d) (:p h))
  h)

(defn softmax
  "Row softmax over k columns."
  [x k]
  (let [n (quot (:size x) k)
        out (make (:shape x))]
    (softmax* (:p x) (long n) (long k) (:p out))
    out))

(defn add-scaled!
  "out = a + alpha*b over an [n x d] block; out may be a itself."
  [out a b alpha]
  (let [n (long (first (:shape a)))
        d (long (if (second (:shape a)) (second (:shape a)) 1))]
    (add-scaled* (:p a) (:p b) n d (double alpha) (:p out))
    out))

(defn add-scaled
  "out = a + alpha*b over an [n x d] block, into a new tensor."
  [a b alpha]
  (add-scaled! (make (:shape a)) a b alpha))

(defn relu!
  "In-place ReLU."
  [x]
  (relu* (:p x) (long (:size x)) (:p x))
  x)

(defn gelu!
  "In-place erf-GELU."
  [x]
  (gelu* (:p x) (long (:size x)) (:p x))
  x)

(defn reshape
  [t shape]
  (assoc t :shape (vec shape)))

(defn t-size [t] (:size t))

;; --- test helpers -------------------------------------------------------------

(defn eq-bytes?
  "Byte-wise equality of two uint8 'tensors' (allowed masks)."
  [a b]
  (and (= (:size a) (:size b))
       (loop [i 0]
         (or (= i (:size a))
             (and (= (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i))
                  (recur (inc i)))))))

(defn first-mismatch
  [a b]
  (loop [i 0]
    (if (or (= i (:size a)) (= (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)))
      (if (= i (:size a)) :none [i (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)])
      [i (ffi/read (:p a) :uint8 i) (ffi/read (:p b) :uint8 i)])))

(defn get
  [p i]
  (ffi/read p :float (* 4 i)))
