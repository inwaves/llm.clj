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

(let [logits (atom [[[0.1 0.2 0.3 0.4]
                     [0.5 0.6 0.7 0.8]
                     [0.9 1.0 1.1 1.2]]
                    [[1.3 1.4 1.5 1.6]
                     [1.7 1.8 1.9 2.0]
                     [2.1 2.2 2.3 2.4]]])
      probs (t_zeros_like logits)
      targets (atom [[2 1 3] [0 0 3]])
      dlosses (atom [[0.1 0.2 0.3] [0.4 0.5 0.6]])
      dlogits (t_zeros_like logits)
      expected (atom [[[0.02497919 0.02663422 -0.07838659 0.02677318]
                       [0.05323514 -0.14647294 0.04647489 0.04676291]
                       [0.07175925 0.07383042 0.07589582 -0.22148549]]
                      [[0.35004634 -0.14142654 -0.11676554 -0.09185426]
                       [0.43755726 -0.17678290 -0.14595641 -0.11481795]
                       [0.07512523 0.07663095 0.07814872 -0.23990489]]])]
  (softmax_forward probs logits)
  (crossentropy_softmax_backward dlogits dlosses probs targets)
  (println @dlogits))
