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
  [out preatt att inp]
  (let [[B T C3] @(t_size inp)
        [_ NH T _] @(t_size att)
        hs (/ (/ C3 3) NH)
        scale (/ 1.0 (math/sqrt hs))]
    (dotimes [b B]
      (dotimes [t T]
        (dotimes [h NH]
          (let [query_t (t_idx query t)
                preatt_bth (t_idx preatt b t h)
                att_bth (t_idx att b t h)]
                ;; TODO: Q@K.T * scale
            ))))))
