(ns llm.utils)

(defn atom? [x] (= (type x) clojure.lang.Atom))
(defn in? [xs el] (some #(= % el) xs))

(defn assert_all [_fn & test_cases]
)

(defn t_allclose 
  ([input other rtol atol]
  ;; |input - other | <= atol + rtol * |other|
  (let [atol (double atol)
        rtol (double rtol)]
  (defn single_allclose [single_inp single_other]
    (<= (abs (- single_inp single_other)) (+ atol (* rtol (abs single_other))))
  )
  (every? true? (map single_allclose input other))
  ))
  ([input other] (t_allclose input other 0.1 0))
)

(defn t_idx [tensor & indices]
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

(defn t_size [tensor]
  (let [dims (atom [(count @tensor)])]
     (loop [t_copy @tensor]
       (when (vector? (first t_copy))
         (swap! dims conj (count (first t_copy)))
         (recur (first t_copy))
         )
    )
  dims)
)


(defn t_fill [fill_value sizes]
  (if (in? sizes 0)
    (throw (IllegalArgumentException. "Cannot have size 0.")))
  (atom
  (if (= (count sizes) 1)
    (vec (repeat (first sizes) fill_value)) ;; Base case: repeat value.
    (vec (repeat 
           (first sizes) 
           @(t_fill fill_value (rest sizes))
           ) ;; Repeat prev vector.
    )
  )
)
)

(defn t_zeros [sizes]
  (t_fill 0 sizes))

(defn t_ones [sizes]
  (t_fill 1 sizes))

(defn t_fill_like [t_other]
  ;; TODO:
  (t_size t_other)
  )

;; zeros_like, ones_like
