(ns lev.prepare-test
  "jolt prepare must reproduce the reference Python conversion byte for
  byte: golden/prepare.edn pins the size and zlib CRC-32 of every
  file the Python converter wrote for this checkpoint. Needs the checkpoint
  (LEV_CHECKPOINTS, config.edn :checkpoints-home, else ../laya); writes to target/prepare-test and removes it."
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [clojure.string :as str]
            [clojure.test :refer [deftest is testing]]
            [lev.config :as cfg]
            [lev.prepare :as prep]))

(def golden-dir
  (or (System/getenv "LEV_GOLDEN") "golden"))

(def home
  "Where jolt prepare looks: LEV_CHECKPOINTS, config.edn :checkpoints-home, ../laya."
  (cfg/setting (cfg/context {}) "--checkpoints" "LEV_CHECKPOINTS" :checkpoints-home "../laya"))

(deftest jolt-prepare-reproduces-python-conversion
  (is (.exists (io/file home "model.safetensors"))
      (str "checkpoint not found under " home " (set LEV_CHECKPOINTS)"))
  (let [golden (:files (edn/read-string (slurp (str golden-dir "/prepare.edn"))))
        out "target/prepare-test"]
    (jolt.host/delete-tree! out)
    (prep/convert home out)
    (testing "every file matches the oracle's size and crc32"
      (is (= 209 (count golden)))
      (doseq [[fn {:keys [size crc32]}] golden]
        (is (= [size crc32] (prep/file-crc32 (str out "/" fn))) fn)))
    (testing "nothing extra was written"
      (is (= (set (keys golden))
             (set (for [f (file-seq (io/file out)) :when (.isFile f)]
                    (subs (.getPath f) (inc (count out))))))))
    (jolt.host/delete-tree! out)))

(deftest edn-writer-escapes-like-the-oracle
  (is (= "\"a\\\\b\\\"c\\n\\r\\t\\u0001é\"" (prep/edn-str "a\\b\"c\n\r\t\u0001é"))))

(deftest config-values-print-like-python
  (testing "str(v) for the JSON scalars: ints bare, floats as repr, strings quoted"
    (is (= "1024" (prep/py-str 1024)))
    (is (= "160000.0" (prep/py-str 160000.0)))
    (is (= "1e-05" (prep/py-str 1.0E-5)))
    (is (= "[\"a\" 1.5 2]" (prep/config-value ["a" 1.5 2])))
    (is (= "{\"choice:2\" 1.9063563346862793}" (prep/config-value {"choice:2" 1.9063563346862793})))))

(deftest missing-checkpoint-says-where-to-get-it
  ;; the usual mistake: ../laya is a checkout of the package's GitHub repo (the
  ;; Python package), not the weights from the Hub
  (let [dir "target/prepare-test-empty"]
    (jolt.host/delete-tree! dir)
    (io/make-parents (io/file dir "x"))
    (spit (str dir "/README.md") "not a checkpoint")
    (let [e (try (prep/convert dir "target/prepare-test-empty-out") nil
                 (catch Exception e e))]
      (is (some? e) "convert must refuse a directory without the checkpoint files")
      (is (= :checkpoint-missing (:type (ex-data e))))
      (is (= ["model.safetensors" "tokenizer/tokenizer.json" "tokenizer/tokenizer_config.json"
              "encoder/config.json" "rl_agent_config.json"]
             (:missing (ex-data e))))
      (is (str/includes? (ex-message e) "https://huggingface.co/convaiinnovations/laya"))
      (is (str/includes? (ex-message e) dir)))
    (jolt.host/delete-tree! dir)
    (jolt.host/delete-tree! "target/prepare-test-empty-out")))

(defn- fake-checkpoint! [dir]
  (doseq [f prep/checkpoint-files]
    (io/make-parents (io/file dir f))
    (spit (str dir "/" f) "")))

(deftest bundle-plan
  ;; the Hub repo bundles three checkpoints: the root is english, the two
  ;; subfolders are optional downloads
  (let [home "target/prepare-plan"]
    (jolt.host/delete-tree! home)
    (fake-checkpoint! home)
    (fake-checkpoint! (str home "/typed-decisions"))
    (testing "english is the root, the others live in subfolders, both in and out"
      (is (= [["english" home "out"]
              ["typed-decisions" (str home "/typed-decisions") "out/typed-decisions"]]
             (mapv (juxt :name :src :out) (prep/plan home "out" nil))))
      (is (= "out/multilingual" (prep/out-dir "out" "multilingual"))))
    (testing "a missing subfolder is skipped, not an error, unless asked for by name"
      (is (= ["english" "typed-decisions"] (mapv :name (prep/plan home "out" nil))))
      (is (= ["typed-decisions"] (mapv :name (prep/plan home "out" ["typed-decisions"]))))
      (is (thrown-with-msg? Exception #"multilingual" (prep/plan home "out" ["multilingual"]))))
    (testing "the root is required"
      (jolt.host/delete-tree! (str home "/model.safetensors"))
      (is (= :checkpoint-missing (:type (ex-data (try (prep/plan home "out" nil) nil (catch Exception e e)))))))
    (testing "names are checked"
      (is (= :unknown-model (:type (ex-data (try (prep/plan home "out" ["nope"]) nil (catch Exception e e)))))))
    (jolt.host/delete-tree! home)))

(deftest jolt-prepare-reproduces-typed-decisions
  ;; same architecture and tokenizer as english, its own weights and config
  (let [src (prep/checkpoint-dir home "typed-decisions")]
    (is (.exists (io/file src "model.safetensors"))
        (str "typed-decisions checkpoint not found under " src " (download the subfolder of the Hub repo)"))
    (let [golden (:files (edn/read-string (slurp (str golden-dir "/typed-decisions/prepare.edn"))))
          out "target/prepare-test-td"]
      (jolt.host/delete-tree! out)
      (prep/convert src out)
      (is (= 209 (count golden)))
      (doseq [[fn {:keys [size crc32]}] golden]
        (is (= [size crc32] (prep/file-crc32 (str out "/" fn))) fn))
      (testing "the config carries the checkpoint's own limits and temperatures"
        (let [cfg (edn/read-string (slurp (str out "/config.edn")))]
          (is (= 1024 (:max-len cfg)))
          (is (= 256 (:head-max-len cfg)))
          (is (= [1.0148024559020996 1.0374259948730469 1.0575125217437744] (:temperature cfg)))))
      (jolt.host/delete-tree! out))))

(deftest tokenizer-specials-come-from-tokenizer-config
  (testing "plain strings or AddedToken objects"
    (is (= {"cls" 2 "sep" 1 "mask" 4 "pad" 0 "unk" 3}
           (prep/tokenizer-specials {"cls_token" "<bos>" "sep_token" "<eos>" "mask_token" {"content" "<mask>"}
                                     "pad_token" "<pad>" "unk_token" "<unk>"}
                                    {"<bos>" 2 "<eos>" 1 "<mask>" 4 "<pad>" 0 "<unk>" 3}))))
  (testing "a name that is not an added token is an error, not a nil id"
    (is (thrown-with-msg? Exception #"mask_token" (prep/tokenizer-specials {"cls_token" "[CLS]" "sep_token" "[SEP]" "mask_token" "[MASK]" "pad_token" "[PAD]" "unk_token" "[UNK]"}
                                                                          {"[CLS]" 1 "[SEP]" 2 "[PAD]" 3 "[UNK]" 4})))))

(deftest tokenizer-kind-from-the-pre-tokenizer
  (is (= :byte-level (prep/tokenizer-kind {"pre_tokenizer" {"type" "ByteLevel"} "normalizer" {"type" "NFC"}})))
  (is (= :sentencepiece (prep/tokenizer-kind {"pre_tokenizer" {"type" "Metaspace" "replacement" "▁" "prepend_scheme" "always" "split" true}
                                              "normalizer" {"type" "Replace" "pattern" {"String" " "} "content" "▁"}})))
  (is (thrown-with-msg? Exception #"Whitespace" (prep/tokenizer-kind {"pre_tokenizer" {"type" "Whitespace"}}))))

(deftest jolt-prepare-reproduces-multilingual
  ;; mmBERT-base weights and config through the same converter; the
  ;; tokenizer.edn is this port's own sentencepiece format, checked by the
  ;; tokenizer cases in checkpoints_test rather than by the Python converter's CRC
  (let [src (prep/checkpoint-dir home "multilingual")]
    (is (.exists (io/file src "model.safetensors"))
        (str "multilingual checkpoint not found under " src " (download the subfolder of the Hub repo)"))
    (let [golden (:files (edn/read-string (slurp (str golden-dir "/multilingual/prepare.edn"))))
          out "target/prepare-test-ml"]
      (jolt.host/delete-tree! out)
      (prep/convert src out)
      (doseq [[fn {:keys [size crc32]}] golden :when (not= fn "tokenizer.edn")]
        (is (= [size crc32] (prep/file-crc32 (str out "/" fn))) fn))
      (let [cfg (edn/read-string (slurp (str out "/config.edn")))
            tok (edn/read-string (slurp (str out "/tokenizer.edn")))]
        (is (= 768 (:hidden-size cfg)))
        (is (= 22 (:num-layers cfg)))
        (is (= 160000 (:rope-local cfg)) "mmBERT uses 160k for sliding layers too")
        (is (= :sentencepiece (:kind tok)))
        (is (= "<mask>" (:mask-token tok)))
        (is (= {:cls 2 :sep 1 :mask 4 :pad 0 :unk 3} (:specials tok)))
        (is (= ["<mask>"] (:lstrip tok)))
        (is (true? (:byte-fallback tok)))
        (is (= 256000 (count (:vocab tok))))
        (is (= 580604 (count (:merges tok)))))
      (jolt.host/delete-tree! out))))
