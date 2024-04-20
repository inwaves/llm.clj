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

(defn idx [tensor & indices]
  ;; Python-style slicing. Takes in an atom, returns an atom.
  ;; Probably wasn't necessary, could use modified get-in.
  (letfn [(resolve-index [coll indices]
            (if (seq indices)
              (let [[index & rest-indices] indices]
                (if (vector? index)
                  (mapv #(resolve-index % rest-indices) (mapv coll index))
                  (recur (nth coll index) rest-indices)))
              (if (vector? coll)
                coll
                [coll])))]
    (atom (resolve-index @tensor indices))))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;;;           TESTS        ;;;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(assert = (tensor_allclose [1.1 2.1 3.1 4.1] [1.0 2.0 3.0 4.0] 0.1 1))
(println "[OK] tensor_allclose...")

(let [inp (atom [[[0 1] [1 2] [2 3]] [[3 4] [4 5] [5 6]]])]
  (assert (= @(idx inp 0 0 1) @(atom [1])))
  (assert (= @(idx inp 0) @(atom [[0 1] [1 2] [2 3]])))
  (assert (= @(idx inp [0 1] 0) @(atom [[0 1] [3 4]])))
  (println @(idx inp [0 1] 0))
  (println (get-in inp [[0 1] 0]))
  (println "[OK] idx...")
  )
