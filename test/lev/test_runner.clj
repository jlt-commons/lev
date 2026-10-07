(ns lev.test-runner
  (:require [clojure.test :as t])
  (:gen-class))

(def test-namespaces
  '[lev.tensors-test
    lev.tokenizer-test
    lev.sequence-test
    lev.json-test
    lev.constraints-test
    lev.llm-test
    lev.think-test
    lev.patterns-test
    lev.calibrate-test
    workflows.email-test
    workflows.presets-test
    lev.lang-test
    lev.router-test
    lev.config-test
    lev.workflows-test
    lev.run-test
    lev.prepare-test
    lev.agent-test
    lev.server-test
    lev.checkpoints-test])

(defn -main
  "jolt -M:test [namespace ...]: the whole suite, or only the namespaces
  named (CI runs lev.checkpoints-test one checkpoint at a time, each in
  its own process: three loaded checkpoints do not fit the mac runner)."
  [& names]
  (let [nss (if (seq names) (map symbol names) test-namespaces)]
    (doseq [ns nss] (require ns))
    (let [{:keys [fail error]} (apply t/run-tests nss)]
      (System/exit (if (zero? (+ fail error)) 0 1)))))
