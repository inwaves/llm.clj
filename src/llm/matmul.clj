(ns llm.matmul
  (:require
   [llm.utils :refer [t_idx t_item t_size]]))

(defn matmul_forward
  "most of the running time is spent here and in matmul_backward
    OC is short for 'output channels'
    inp is (B,T,C), weight is (OC, C), bias is (OC)
    out will be (B,T,OC)"
  [outp inp weight bias]
  (let [[B T _] (t_size inp)
        OC (first @(t_size bias))]
    (dotimes [b B]
      (dotimes [t T]
        (let [inp_bt (t_idx inp b t)]
          (dotimes [o OC]
            (let [val (if (nil? bias) 0.0 (t_item (t_idx bias o)))
                  wrow (t_idx weight o)]
              (swap! outp update-in [o] (fn [_] (+ val (reduce + (map * inp_bt wrow))))))))))))

(defn matmul_backward
  "most of the running time is spent here and in matmul_forward
    this backward could be done in a single 'round' of loops
    but that doesn't afford an efficient parallelization strategy"
  [dinp dweight dbias dout inp weight])
