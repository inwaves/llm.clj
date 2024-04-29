(ns llm.matmul
  (:require
   [llm.utils :refer [t_idx t_item t_size]]))

(defn matmul_forward
  "most of the running time is spent here and in matmul_backward
    OC is short for 'output channels'
    inp is (B,T,C), weight is (OC, C), bias is (OC)
    out will be (B,T,OC)"
  [outp inp weight bias]
  (let [[B T _] @(t_size inp)
        OC (last @(t_size weight))] ;; don't get the size from bias – it can be nil.
    (dotimes [b B]
      (dotimes [t T]
        (let [inp_bt (t_idx inp b t)]
          (dotimes [o OC]
            (let [val (if (nil? bias) 0.0 (t_item (t_idx bias o)))
                  wrow (t_idx weight o)]
            ;; FIXME: This is broken, the last item doesn't get updated at all.
              (swap! outp update-in [b t o] (fn [_] (+ val (reduce + (map * @inp_bt @wrow))))))))))))

(defn matmul_backward
  "most of the running time is spent here and in matmul_forward
    this backward could be done in a single 'round' of loops
    but that doesn't afford an efficient parallelization strategy"
  [dinp dweight dbias dout inp weight]
  (let [[B T C] @(t_size dinp)
        OC (last @(t_size dweight))]

  ;; grad wrt/ inputs
    (dotimes [b B]
      (dotimes [t T]
        (dotimes [o OC]
          (let [wrow (t_idx weight o)
                d (t_item (t_idx dout b t o))]
            (dotimes [c C]
              (swap! dinp update-in [b t c] (fn [dinp_btc] (+ dinp_btc (* (t_item (t_idx wrow c)) d)))))))))

    ;; wrt/ weight and bias matrices
    (dotimes [o OC]
      (dotimes [b B]
        (dotimes [t T]
          (let [dout_bto (t_item (t_idx dout b t o))]
            (if (not (nil? dbias)) (swap! dbias [o] (fn [dbias_o] (+ dbias_o dout_bto))) nil)
            (dotimes [c C]
              (swap! dweight [o c] (fn [dweight_oc] (+ dweight_oc (* (t_item (t_idx inp b t c)) dout_bto)))))))))))
