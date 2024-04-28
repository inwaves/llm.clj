(ns llm.layernorm
  (:require
   [clojure.math :refer [sqrt]]
   [llm.utils :refer [t_idx t_item t_mean t_size t_var]]))

(defn layernorm_forward
  "reference: https://pytorch.org/docs/stable/generated/torch.nn.LayerNorm.html
both inp and out are (B,T,C) of the activations
mean and rstd are (B,T) buffers, to be used later in backward pass
at each position (b,t) of the input, the C-dimensional vector
of activations gets normalized, then scaled and shifted
input: weight (C,)
input: bias (C,)
"
  [out mean rstd inp weight bias]
  (let [[B T C] @(t_size out)
        eps 1e-5]
    (dotimes [b B]
      (dotimes [t T]
        (let [mean_inp_bt (t_mean (t_idx inp b t))
              var_inp_bt (t_var (t_idx inp b t))
              rstd_bt (/ 1.0 (sqrt (+ var_inp_bt eps)))]
          (dotimes [c C]
            (let [weight_c (t_item (t_idx weight c))
                  bias_c (t_item (t_idx bias c))
                  inp_btc (t_item (t_idx inp b t c))]
            ;; For each element out[b, t, c], scale and shift inp[b, t, c].
              (swap! out update-in [b t c]
                     (fn [_]
                       (+ bias_c (* weight_c (* rstd_bt (- inp_btc mean_inp_bt)))))))
            (swap! mean update-in [b t] mean_inp_bt)
            (swap! rstd update-in [b t] rstd_bt)))))))

(defn layernorm_backward
  "Backpropagating through the layer normalisation operation."
  [dinp dweight dbias dout inp weight mean rstd]
  (let [[B T C] @(t_size dout)]
    (dotimes [b B]
      (dotimes [t T]
        (let [dout_bt (t_idx dout b t)
              inp_bt (t_idx inp b t)
              mean_bt (t_item (t_idx mean b t))
              rstd_bt (t_item (t_idx rstd b t))

              ;; If something is wrong with this function, it's likely to be here.
              dnorm_mean (/ (reduce + (map * dout_bt weight)) C)
              dnorm_norm_mean (/ (reduce + (map *
                                                (map * dout_bt weight) ;; dnorm_i
                                                (map (fn [inp_bt_i] (* rstd_bt (- inp_bt_i mean_bt))) inp_bt))) C)]
          (dotimes [c C]
            (let [norm_btc (* (- (t_item (t_idx inp_bt c)) mean_bt) rstd_bt)
                  dout_btc (t_item (t_idx dout_bt c))
                  dnorm_c (* (t_item (t_idx weight c)) dout_btc)
                  dval (* (- (- dnorm_c dnorm_mean) (* norm_btc dnorm_norm_mean)) rstd_bt)]
              (swap! dbias update-in [c] (fn [dbias_c] (+ dbias_c dout_btc)))
              (swap! dweight update-in [c] (fn [dweight_c] (+ dweight_c (* norm_btc dout_btc))))
              (swap! dinp update-in [b t c] (fn [dinp_btc] (+ dinp_btc dval))))))))))
