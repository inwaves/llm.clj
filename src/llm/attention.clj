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
            (dotimes [t2 (inc t)] ;; t2 <= t
              (let [raw_expsum (reduce + (t_idx att b t h))
                    expsum_inv (if (= raw_expsum 0) 0.0 (/ 1 raw_expsum))] ;; avoid division by 0
                (swap! att update-in [b t h t2]
                       (fn [att_btht2] (* att_btht2 expsum_inv)))))

            ;; accumulate weighted values into output of attention
            (dotimes [t2 (inc t)]
              (dotimes [i hs]
                (swap! outp update-in [b t h i] (fn [outp_bthi] (+ outp_bthi (* (t_idx att b t h t2) (t_idx inp b t2 (+ (* 2 C) h) i)))))))))))))

;; Used Claude for this, prompted with my port above
;; and the two originals in C by karpathy.
(defn attention_backward
  "inp/dinp are (B, T, 3C) Q,K,V
   att/datt/dpreatt are (B, NH, T, T)
   dout is (B, T, C)"
  [dinp dpreatt datt dout inp att]
  (let [[B T C3] @(t_size inp)
        C (/ C3 3)
        [_ NH _ _] @(t_size att)
        hs (/ C NH)
        scale (/ 1.0 (math/sqrt hs))]
    (dotimes [b B]
      (dotimes [t T]
        (dotimes [h NH]
          (let [att_bth (t_idx att b t h)
                datt_bth (t_idx datt b t h)
                dpreatt_bth (t_idx dpreatt b t h)
                query_t (t_idx inp b t h)
                dout_bth (t_idx dout b t h)]
            ;; backward pass 4, through the value accumulation
            (dotimes [t2 (inc t)]
              (let [value_t2 (t_idx inp b t2 (+ (* 2 C) h))]
                (dotimes [i hs]
                  ;; in the forward pass this was:
                  ;; out_bth[i] += att_bth[t2] * value_t2[i];
                  ;; so now we have:
                  (swap! datt update [b t h t2] (fn [datt_btht2] (+ datt_btht2 (* (t_idx value_t2 i) (t_idx dout_bth i)))))
                  (swap! dinp update [b t2 (+ (* 2 C) h i)] (fn [dinp_bt2c2hi] (+ dinp_bt2c2hi (* (t_idx att_bth t2) (t_idx dout_bth i))))))))

            ;; backward pass 2 & 3, the softmax
            ;; note that softmax (like e.g. tanh) doesn't need the input (preatt) to backward
            (dotimes [t2 t]
              (dotimes [t3 t]
                (let [indicator (if (= t2 t3) 1.0 0.0)
                      local_derivative (* (t_idx att_bth t2) (- indicator (t_idx att_bth t3)))]
                  (swap! dpreatt update [b t h t3] (fn [dpreatt_btht3] (+ dpreatt_btht3 (* local_derivative (t_idx datt_bth t2))))))))

            ;; backward pass 1, the query @ key matmul
            (dotimes [t2 t]
              (dotimes [i hs]
                ;; in the forward pass this was:
                ;; preatt_bth[t2] += (query_t[i] * key_t2[i]) * scale;
                ;; so now we have:
                (swap! dinp update [b t (+ h i)] (fn [dinp_bthi] (+ dinp_bthi (* (t_idx inp b t2 (+ C h i)) (t_idx dpreatt_bth t2) scale))))
                (swap! dinp update [b t2 (+ C h i)] (fn [dinp_bt2chi] (+ dinp_bt2chi (* (t_idx query_t i) (t_idx dpreatt_bth t2) scale))))))))))))
