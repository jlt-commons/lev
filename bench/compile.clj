(ns bench.compile
  "A local model writes the lev.program: it sees the task, the program
  format, and a sample of the train split (up to --per-label inputs a
  label, distinct after digits are masked), answers greedily with JSON,
  and gets lev.program/validate's error back up to twice. The program
  is written to --save for bench/program.clj to fit and score.

    jolt -M bench/compile.clj --split bgl-train --task \"...\" --save FILE
         [--gguf PATH | --model NAME] [--per-label 8] [--max-tokens 3000]"
  (:require [clojure.string :as str]
            [lev.config :as cfg]
            [lev.json :as json]
            [lev.llm :as llm]
            [lev.program :as prog]
            [lev.sequence :as seq]))

(def format-spec
  "A program is one JSON object:
{\"labels\": [label, ...],
 \"extract\": {fact: regex},                        a fact holds when its (Java) regex finds a match in the input
 \"questions\": {qid: question},                    asked to a small text encoder when a rule needs them
 \"rules\": [{\"when\": condition, \"then\": label}, ...],   the first rule that holds names the label
 \"default\": label}                                when no rule holds
A question is {\"type\": \"noul\", \"instructions\": \"a yes/no question about the input\"},
{\"type\": \"choice\", \"instructions\": \"...\", \"criteria\": {option: description}} or
{\"type\": \"score\", \"instructions\": \"...\", \"criteria\": [level, ...]}.
A condition is {\"fact\": f}, {\"noul\": qid, \">=\": p} (p the probability of yes; also > <= <),
{\"score\": qid, \">=\": x}, {\"choice\": qid, \"is\": option}, {\"choice\": qid, \"in\": [option, ...]},
{\"all\": [condition, ...]}, {\"any\": [condition, ...]} or {\"not\": condition}.
Facts cost nothing; questions cost about 50 ms each and the encoder is weak on long or technical text,
so use a regex wherever the input's wording decides the label, and a question where meaning does.
Numeric cutoffs are refitted on labelled data afterwards.")

(defn- sample [cases per-label]
  (mapcat (fn [[label cs]]
            (->> cs
                 (map #(get % "text"))
                 (reduce (fn [[seen out] t]
                           (let [k (str/replace t #"[0-9a-fA-F]{2,}|\d+" "N")]
                             (if (contains? seen k) [seen out] [(conj seen k) (conj out t)])))
                         [#{} []])
                 second
                 (take per-label)
                 (map (fn [t] [label t]))))
          (group-by #(get % "label") cases)))

(defn- prompt-for [task cases per-label]
  (str "Write a classifier program for this task: " task "\n\n" format-spec
       "\n\nLabelled examples (label<TAB>input), from the training data:\n"
       (str/join "\n" (map (fn [[l t]] (str l "\t" (subs t 0 (min 300 (count t))))) (sample cases per-label)))
       "\n\nAnswer with the JSON program only."))

(defn- json-in [text]
  (let [a (str/index-of text "{") b (str/last-index-of text "}")]
    (when (and a b (< a b)) (json/read-str (subs text a (inc b))))))

(defn -main [& args]
  (let [opts (cfg/parse-args args)
        ctx (cfg/context opts)
        gguf (or (get opts "--gguf") (:model (get (cfg/thinkers ctx) (get opts "--model" "qwen3.8-27b"))))
        cases (map json/read-str (remove str/blank? (str/split-lines (slurp (str "bench/data/" (get opts "--split") ".jsonl")))))
        m (llm/load gguf {:n-ctx 16384 :n-seq-max 1})
        max-tokens (Long/parseLong (get opts "--max-tokens" "3000"))
        ask (fn [messages]
              (let [t0 (System/nanoTime)
                    r (llm/generate m (llm/chat-prompt m messages {:thinking false}) {:max-tokens max-tokens :temperature 0.0 :stop "<|im_end|>"})]
                (println (format "  %d tokens in %.1f s" (:tokens r) (/ (- (System/nanoTime) t0) 1e9)))
                (:text r)))
        first-msg {:role "user" :content (prompt-for (get opts "--task") cases (Long/parseLong (get opts "--per-label" "8")))}]
    (println "compiling with" gguf)
    (loop [messages [first-msg] tries 0]
      (let [text (ask messages)
            [program err] (try [(prog/validate (json-in text)) nil]
                               (catch Exception e [nil (ex-message e)]))]
        (cond
          program (do (spit (get opts "--save") (str (seq/json-str program) "\n"))
                      (println "valid program:" (count (get program "extract")) "facts,"
                               (count (get program "questions")) "questions," (count (get program "rules")) "rules")
                      (println (seq/json-str program)))
          (< tries 2) (do (println "invalid:" err)
                          (recur (conj messages {:role "assistant" :content text}
                                       {:role "user" :content (str "That program is invalid: " err ". Answer with the corrected JSON program only.")})
                                 (inc tries)))
          :else (do (println "gave up:" err) (println text) (System/exit 1)))))))

(apply -main *command-line-args*)
