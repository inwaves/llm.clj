(ns llm.softmax
  (:require [clojure.math :as math])
  (:require [llm.utils :refer [t_idx t_size t_zeros_like t_allclose]])
)

(defn softmax_forward [probs logits]
  ;; input: logits are (B, T, V) of the unnormalised log probabilities
  ;; output: probs are (B, T, V) of the probabilities (sums to 1.0 in 
  ;; each b, t position)
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [logits_bt @(t_idx logits b t)
              maxval (apply max logits_bt)
              probs_bt (map #(math/exp (- % maxval)) logits_bt)
              sum_probs (reduce + probs_bt)]
          (swap! probs update-in [b t] (fn [_] (mapv #(/ % sum_probs) probs_bt)))
        )
      )
    )
  )
)

