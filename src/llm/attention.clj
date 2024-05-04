(ns llm.attention
  (:require
   [clojure.math :as math]
   [llm.utils :refer [t_idx t_size]]))

(defn attention_forward
  "input is (B, T, 3C) holding the query, key, value (Q, K, V) vectors
    preatt, att are (B, NH, T, T). NH = number of heads, T = sequence length
    that holds the pre-attention and post-attention scores (used in backward)
    output is (B, T, C)
    attention is the only layer that mixes information across time
    every other operation is applied at every (b,t) position independently
    (and of course, no layer mixes information across batch)"
  [outp preatt att inp]
  (let [[B T C3] @(t_size inp)
        C (/ C3 3)
        [_ NH _ _] @(t_size att)
        hs (/ C NH)
        scale (/ 1.0 (math/sqrt hs))]
    (dotimes [b B]
      (dotimes [t T]
        (dotimes [h NH]
          (dotimes [t2 t]
            ;; exp((query_t) x (key_t2) - maxvalue)
            ;; TODO: should initialise with really large neg values to track maxval?
            ;; TODO: indexing correct?
            (swap! preatt update-in [b t h t2] (fn [_] (* scale (reduce + (map * (t_idx inp b t h) (t_idx inp b t2 (+ C h)))))))
            (swap! att update-in [b t h t2]
                   (fn [_]
                     (math/exp  (- (t_idx preatt b t h t2) (max (t_idx preatt b t h))))))

                    ;; normalise by the sum to get softmax
            (dotimes [t2 t]
              (let [raw_expsum (reduce + (t_idx att b t h))
                    expsum_inv (if (= raw_expsum 0) 0.0 (/ 1 raw_expsum))] ;; avoid division by 0
                (swap! att update-in [b t h t2]
                       (fn [att_btht2] (* att_btht2 expsum_inv)))))

            ;; accumulate weighted values into output of attention
            (dotimes [t2 t]
              (dotimes [i hs]
                (swap! outp update-in [b t h i] (fn [outp_bthi] (+ outp_bthi (* (t_idx att b t h t2) (t_idx inp b t2 (+ (* 2 C) h) i)))))))))))))
