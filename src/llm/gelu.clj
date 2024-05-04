(ns llm.gelu
  (:require [clojure.math :as math]))

(def GELU_SCALING_FACTOR (math/sqrt (/ 2.0 math/PI)))

(defn gelu_transformation [single_inp]
  (let [cube (* 0.044715 single_inp single_inp single_inp)]
    (* 0.5 single_inp (+ 1.0 (math/tanh (* GELU_SCALING_FACTOR (+ single_inp cube)))))))

(defn gelu_back_transformation [single_inp]
  (let [cube (* 0.044715 single_inp single_inp single_inp)
        tanh_arg (* GELU_SCALING_FACTOR (+ single_inp cube))
        tanh_out (math/tanh tanh_arg)
        coshf_out (math/cosh tanh_arg)
        sech_out (/ 1.0 (* coshf_out coshf_out))]
    (+ (* 0.5 (+ 1.0 tanh_out)) (* single_inp 0.5 sech_out GELU_SCALING_FACTOR (+ 1.0 (* 3.0 0.044715 single_inp single_inp))))))

(defn backward [single_dinput single_input single_doutput]
  (+ single_dinput (* (gelu_back_transformation single_input) single_doutput)))

(defn gelu_forward_pure_fn [input]
  (atom (vec (map gelu_transformation @input))))

(defn gelu_forward [output input]
  (dotimes [idx (count @input)]
    (let [result (gelu_transformation (nth @input idx))]
      (swap! output assoc idx result))))

(defn gelu_backward_pure_fn [dinput input doutput]
  (atom (vec (map backward @dinput @input @doutput))))

(defn gelu_backward [dinput input doutput]
  (dotimes [idx (count @input)]
    (let [result (backward (nth @dinput idx) (nth @input idx) (nth @doutput idx))]
      (swap! dinput assoc idx result))))
