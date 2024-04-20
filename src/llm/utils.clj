(ns llm.utils)

(defn tensor_allclose 
  ([input other rtol atol]
  ;; |input - other | <= atol + rtol * |other|
  (let [atol (double atol)
        rtol (double rtol)]
  (defn single_allclose [single_inp single_other]
    (<= (abs (- single_inp single_other)) (+ atol (* rtol (abs single_other))))
  )
  (every? true? (map single_allclose input other))
  ))
  ([input other] (tensor_allclose input other 0.1 0))
)

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;           TESTS        ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(assert = (tensor_allclose [1.1 2.1 3.1 4.1] [1.0 2.0 3.0 4.0] 0.1 1))
(println "[OK] tensor_allclose...")
