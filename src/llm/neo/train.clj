(ns llm.neo.train
  (:require [llm.neo.forward :as fwd]
            [llm.neo.loss :as loss]
            [llm.neo.core :as neo]))

(defn compute-loss [params batch config]
  (let [logits-nested (fwd/gpt2-forward (:inputs batch) {:config config :params params})
        B (count logits-nested) T (count (first logits-nested))
        logits-flat (vec (for [b (range B) t (range T)] (nth (nth logits-nested b) t)))
        logits-mat (neo/vec->matrix logits-flat)]
    (loss/cross-entropy-loss logits-mat (:targets batch))))

(defn training-step [model-state batch]
  (let [{:keys [config params step]} model-state
        current-loss (compute-loss params batch config)]
    (assoc model-state :step (inc step) :loss current-loss)))

(defn train-epoch [model-state dataset]
  (reduce (fn [state batch]
            (let [updated (training-step state batch)]
              (when (zero? (mod (:step updated) 10))
                (println (format "Step %d, Loss: %.4f" (:step updated) (:loss updated))))
              updated))
          model-state
          dataset))