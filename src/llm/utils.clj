(ns llm.utils)

(defn atom? [x] (= (type x) clojure.lang.Atom))
(defn in? [xs el] (some #(= % el) xs))

(defn assert_all [_fn & test_cases]
  ;; TODO:
)

(defn flatten_tensor [tensor]
  (if (vector? tensor)
    (vec (mapcat flatten_tensor tensor))
    [tensor]))


(defn t_idx [tensor & indices]
  "Python-style slicing.
   Params:
    tensor (Atom) - the tensor to slice into;
    indices (vector) - indices to select;
   Returns:
    sliced_tensor (Atom) - requested values from the original tensor."

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
  "Get the size of a given tensor, return it as a tensor."
  (let [dims (atom [(count @tensor)])]
     (loop [t_copy @tensor]
       (when (vector? (first t_copy))
         (swap! dims conj (count (first t_copy)))
         (recur (first t_copy))
         )
    )
  dims)
)

(defn t_flatten [tensor]
  (let [size (t_size tensor)
        total_elements (reduce * @size)
        flattened (flatten_tensor @tensor)]
    (assert (= total_elements (count flattened)))
    (atom flattened))
)

(defn t_allclose 
  ([input other rtol atol]
  "|input - other | <= atol + rtol * |other|"
  (let [atol (double atol)
        rtol (double rtol)
        flat_input (t_flatten input)
        flat_other (t_flatten other)]
  (defn single_allclose [single_inp single_other]
    (<= (abs (- single_inp single_other)) (+ atol (* rtol (abs single_other))))
  )
  (every? true? (map single_allclose @flat_input @flat_other))
  ))
  ([input other] (t_allclose input other 0.1 0))
)

(defn t_fill [fill_value sizes]
  "Fill a tensor of a given shape with fill_value."
  (if (in? sizes 0)
    (throw (IllegalArgumentException. "Cannot have size 0.")))
  (atom
    (if (= (count sizes) 1)
      (vec (repeat (first sizes) fill_value)) ;; Base case: repeat value.
      (vec (repeat 
            (first sizes) 
            @(t_fill fill_value (rest sizes))
            ) ;; Repeat prev tensor.
      )
    )
  )
)

(defn t_zeros [sizes]
  (t_fill 0 sizes))

(defn t_ones [sizes]
  (t_fill 1 sizes))

(defn t_fill_like [fill_value t_other]
  (t_fill fill_value @(t_size t_other))
)

(defn t_zeros_like [t_other]
  (t_fill_like 0 t_other))

(defn t_ones_like [t_other]
  (t_fill_like 1 t_other))
