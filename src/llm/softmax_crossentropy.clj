(ns llm.softmax_crossentropy
  (:require [clojure.math :as math])
  (:require [llm.utils :refer [t_idx t_size t_zeros_like t_allclose]])
)

(defn softmax_forward [probs logits]
  "input: logits are (B, T, V) of the unnormalised log probabilities
   output: probs are (B, T, V) of the probabilities (sums to 1.0 in 
   each b, t position)"
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [logits_bt (t_idx logits b t)
              maxval (apply max @logits_bt)
              probs_bt (atom (map #(math/exp (- % maxval)) @logits_bt))
              sum_probs (reduce + @probs_bt)]
          (swap! probs update-in [b t] (fn [_] (mapv #(/ % sum_probs) @probs_bt)))
        )
      )
    )
  )
)

(defn crossentropy_forward [losses probs targets]
  "output: losses is (B T) of the individual losses at each position.
      Each loss is -log(probs[target]).
   input: probs are (B T V) of the probabilities
   input: targets is (B T) of integers giving the correct index in logits
  "
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [probs_bt (t_idx probs b t) ;; last dimension of probs
              idx (t_idx targets b t)]  ;; index of current target.
          (swap! losses update-in [b t] (fn [_] (mapv #(- (math.log @(t_idx probs_bt idx)))))) 
          )))
  )
)

(defn crossentropy_softmax_backward [dlogits dlosses probs targets]
  "backwards through both softmax and crossentropy"
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [dlogits_bt (t_idx dlogits b t)
              probs_bt (t_idx probs b t)
              dloss (t_idx dlosses b t)
              idx (t_idx targets b t)]
          (dotimes [v V]
            (let [indicator (if (= v idx) 1.0 0.0)]
            (swap! 
              dlogits_bt 
              update-in [v] (
                  fn [_] (mapv #(
                    + @(t_idx dlogits_bt v) 
                    (* (- @(t_idx probs_bt v) indicator) dloss)))
            )))))  
      )
    )
  )
)
