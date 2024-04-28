#_{:clj-kondo/ignore [:misplaced-docstring]}
(ns llm.encoder
  (:require
   [llm.utils :refer [t_idx t_item t_size]]))

(defn encoder_forward
  [out inp wte wpe]
  "out is (B,T,C). At each position (b,t), a C-dimensional vector summarizing token & position
  inp is (B,T) of integers, holding the token ids at each (b,t) position
  wte is (V,C) of token embeddings, short for 'weight token embeddings'
  wpe is (maxT,C) of position embeddings, short for 'weight positional embedding'"
  (let [[B T C] @(t_size out)]
    (dotimes [b B]
      (dotimes [t T]
        (let [idx (t_idx inp b t)
              wte_idx (t_idx wte idx)
              wpe_t (t_idx wpe t)]
          (swap! out update-in [b t] (fn [_] (+ wte_idx wpe_t))))))))

(defn encoder_backward
  [dwte dwpe dout inp]
  (let [[B T C] @(t_size dout)]
    (dotimes [b B]
      (dotimes [t T]
        (let [dout_bt (t_idx dout b t)
              idx (t_idx inp b t)]
          (dotimes [c C]
            (swap! dwte update-in [idx c]
                   (fn [dwte_idx_c] (+ dwte_idx_c (t_item (t_idx dout_bt c)))))
            (swap! dwpe update-in [t c]
                   (fn [dwpe_tc] (+ dwpe_tc (t_item (t_idx dout_bt)))))))))))
