#_{:clj-kondo/ignore [:underscore-in-namespace]}
(ns llm.softmax_crossentropy
  (:require [clojure.math :as math])
  (:require [llm.utils :refer [t_idx t_size t_zeros_like t_allclose t_item]]))

(defn softmax_forward
  "input: logits are (B, T, V) of the unnormalised log probabilities
   output: probs are (B, T, V) of the probabilities (sums to 1.0 in
   each b, t position)"
  [probs logits]
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [logits_bt (t_idx logits b t)
              maxval (apply max @logits_bt)
              probs_bt (atom (map #(math/exp (- % maxval)) @logits_bt))
              sum_probs (reduce + @probs_bt)]
          (swap! probs update-in [b t] (fn [_] (mapv #(/ % sum_probs) @probs_bt))))))))

(defn crossentropy_forward
  "output: losses is (B T) of the individual losses at each position.
      Each loss is -log(probs[target]).
   input: probs is (B T V) of the probabilities
   input: targets is (B T) of integers giving the correct index in logits
  "
  [losses probs targets]
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [probs_bt (t_idx probs b t) ;; last dimension of probs
              idx @(t_idx targets b t)]  ;; index of current target.
          (swap! losses update-in [b t] (fn [_] (- (math/log (t_item (t_idx probs_bt idx)))))))))))

(defn crossentropy_softmax_backward
  "output: dlogits is (B T V) the gradient of the loss wrt xe-softmax
        input: dlosses is (B T) of per-token losses
        input: probs is (B T V) of the probabilities
        input: targets is (B T) of integers giving the correct index in logits

  backwards through both softmax and crossentropy"
  [dlogits dlosses probs targets]
  (let [[B T V] @(t_size probs)]
    (dotimes [b B]
      (dotimes [t T]
        (let [probs_bt (t_idx probs b t)    ;; (v, ) shaped tensor
              dloss (t_item (t_idx dlosses b t)) ;; float
              idx (t_idx targets b t)]  ;; int
          (dotimes [v V]
            (let [indicator (if (= v idx) 1.0 0.0)] ;; float, obviously
              (swap!
               dlogits
               update-in [b t v] (fn [dlogits_btv] (+ dlogits_btv
                                                      (* (- (t_item (t_idx probs_bt v)) indicator) dloss)))))))))))
