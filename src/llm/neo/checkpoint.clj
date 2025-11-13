(ns llm.neo.checkpoint
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [llm.neo.model :as model]))

(defn save-checkpoint [model-state filepath]
  (let [data {:config (:config model-state)
              :params (:params model-state)
              :optimizer-state (:optimizer-state model-state)
              :step (:step model-state)
              :loss (get model-state :loss 0.0)}]
    (spit filepath (binding [*print-length* nil *print-level* nil] (pr-str data)))
    (println (format "Saved checkpoint at step %s" (:step model-state)))))

(defn load-checkpoint [filepath]
  (let [data (edn/read-string (slurp filepath))
        config (model/map->GPT2Config (:config data))
        params (model/map->ParameterTensors (:params data))
        state (model/->ModelState config params (:optimizer-state data) (:step data))]
    (println (format "Loaded checkpoint from %s" filepath))
    state))

(defn checkpoint-exists? [filepath]
  (.exists (io/file filepath)))