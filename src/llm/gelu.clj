(ns llm.gelu
  (:require [clojure.math :as math])
  (:require [llm.utils :refer [tensor_allclose]])
)

(def GELU_SCALING_FACTOR (math/sqrt (/ 2.0 math/PI)))

(defn gelu_transformation [single_inp]
  (let [cube (* 0.044715 single_inp single_inp single_inp)]
    (* 0.5 single_inp (+ 1.0 (math/tanh (* GELU_SCALING_FACTOR (+ single_inp cube)))))
    )
)

(defn gelu_back_transformation [single_inp]
  (let [cube (* 0.044715 single_inp single_inp single_inp)
        tanh_arg (* GELU_SCALING_FACTOR (+ single_inp cube))
        tanh_out (math/tanh tanh_arg)
        coshf_out (math/cosh tanh_arg)
        sech_out (/ 1.0 (* coshf_out coshf_out))
        ]
  (+ (* 0.5 (+ 1.0 tanh_out)) (* single_inp 0.5 sech_out GELU_SCALING_FACTOR (+ 1.0 (* 3.0 0.044715 single_inp single_inp))))
  ))

(defn backward [single_dinput single_input single_doutput]
  (+ single_dinput (* (gelu_back_transformation single_input) single_doutput)))

(defn gelu_forward_pure_fn [output input]
    (map gelu_transformation input)
)

(defn gelu_forward [output input]
  (dotimes [idx (count input)]
    (let [result (gelu_transformation (nth input idx))]
         (swap! output assoc idx result))
    )
)

(defn gelu_backward_pure_fn [dinput input doutput]
  (map backward dinput input doutput)
  )

(defn gelu_backward [dinput input doutput]
  (dotimes [idx (count input)]
    (let [result (backward (nth @dinput idx) (nth input idx) (nth @doutput idx))]
      (swap! dinput assoc idx result)
      )
    )
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;           TESTS        ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(let [expected [0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075 4.999999770820381]
]
  (assert (= (gelu_forward_pure_fn [0 0 0 0 0] [1 2 3 4 5]) expected))
  (println "[OK] gelu_forward_pure_fn...")
  )

(let [input [1 2 3 4 5] 
      output (atom [0 0 0 0 0])
      expected [0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075 4.999999770820381]]
  (gelu_forward output input)
  (assert (tensor_allclose @output expected 0.1 0))
  (assert (tensor_allclose @output (gelu_forward_pure_fn [0 0 0 0 0] input) 0.1 0))
  (println "[OK] gelu_forward...")
  )

(let [input [1 2 3 4 5]
      dinput [1 2 3 4 5]
      expected [1.9110 4.1229 6.0311 8.0013 10.0]
      result (gelu_backward_pure_fn input (gelu_forward_pure_fn [0 0 0 0 0] [1 2 3 4 5]) dinput)]
  (assert (tensor_allclose result expected 0.1 0))
  (println "[OK] gelu_backward_pure_fn...")
)

(let [input [1 2 3 4 5] 
      dinput (atom [1 2 3 4 5])
      output (atom [0 0 0 0 0])
      expected [1.9110 4.1229 6.0311 8.0013 10.0]]
  (gelu_forward output input)
  (gelu_backward dinput input output)
  (assert (tensor_allclose @dinput expected))
  (assert (tensor_allclose @dinput (gelu_backward_pure_fn input (gelu_forward_pure_fn [0 0 0 0 0] [1 2 3 4 5]) [1 2 3 4 5])))
  (println "[OK] gelu_backward...")
)
