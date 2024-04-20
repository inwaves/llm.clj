(require '[clojure.math :as math])

(def GELU_SCALING_FACTOR (math/sqrt (/ 2.0 math/PI)))

;; In the original these work on the output vector
;; provided. This kinda conflicts with Clojure's no side-effects
;; rule, AFAICT. But if you keep returning and reassigning massive
;; matrices, your memory usage will suck.
;; TODO: Operate on direct output instead of returning.
(defn gelu_forward [output input]
  (defn gelu_transformation [single_inp]
    (let [cube (* 0.044715 single_inp single_inp single_inp)]
      (* 0.5 single_inp (+ 1.0 (math/tanh (* GELU_SCALING_FACTOR (+ single_inp cube)))))
      )
    )
    (map gelu_transformation input)
)

(defn gelu_backward [dinput input doutput]
  (defn gelu_back_transf [single_inp]
    (let [cube (* 0.044715 single_inp single_inp single_inp)
          tanh_arg (* GELU_SCALING_FACTOR (+ single_inp cube))
          tanh_out (math/tanh tanh_arg)
          coshf_out (math/cosh tanh_arg)
          sech_out (/ 1.0 (* coshf_out coshf_out))
          ]
    (+ (* 0.5 (+ 1.0 tanh_out)) (* single_inp 0.5 sech_out GELU_SCALING_FACTOR (+ 1.0 (* 3.0 0.044715 single_inp single_inp))))
    ))
  (defn backward [single_dinput single_input single_doutput]
    (+ single_dinput (* (gelu_back_transf single_input) single_doutput)))

  (map backward dinput input doutput)
  )

(defn tensor_allclose [input other rtol atol]
  ;; |input - other | <= atol + rtol * |other|
  (defn single_allclose [single_inp single_other rtol atol]
    (<= (abs (- single_inp single_other)) (+ atol (* rtol (abs single_other))))
  )
  (map single_allclose input other rtol atol)
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;           TESTS        ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(assert = (tensor_allclose [1.1 2.1 3.1 4.1] [1.0 2.0 3.0 4.0] 0.1 1))
(println "[OK] tensor_allclose...")

(let [expected [0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075 4.999999770820381]
]
  (assert (= (gelu_forward [0 0 0 0 0] [1 2 3 4 5]) expected))
  (println "[OK] gelu_forward...")
  )
(let [expected [1.9110 4.1229 6.0311 8.0013 10.0]
      result (gelu_backward [1 2 3 4 5] (gelu_forward [0 0 0 0 0] [1 2 3 4 5]) [1 2 3 4 5])]
  (assert (tensor_allclose result expected 0.1 1)))
  (println "[OK] gelu_backward...")
