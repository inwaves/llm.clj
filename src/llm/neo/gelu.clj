(ns llm.neo.gelu
  "GELU activation function using Neanderthal.
  
  GELU (Gaussian Error Linear Unit) formula:
  GELU(x) = x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"
  (:use [uncomplicate.neanderthal core native])
  (:require [llm.neo.core :as neo]))

(def gelu-scaling-factor (Math/sqrt (/ 2.0 Math/PI)))

(defn gelu-element
  "Apply GELU to a single element."
  [x]
  (let [cube (* 0.044715 x x x)
        tanh-arg (* gelu-scaling-factor (+ x cube))
        tanh-out (Math/tanh tanh-arg)]
    (* 0.5 x (+ 1.0 tanh-out))))

(defn gelu-backward-element
  "Compute GELU gradient for a single element."
  [x]
  (let [cube (* 0.044715 x x x)
        tanh-arg (* gelu-scaling-factor (+ x cube))
        tanh-out (Math/tanh tanh-arg)
        cosh-out (Math/cosh tanh-arg)
        sech-out (/ 1.0 (* cosh-out cosh-out))
        derivative-factor (+ 1.0 (* 3.0 0.044715 x x))]
    (+ (* 0.5 (+ 1.0 tanh-out))
       (* x 0.5 sech-out gelu-scaling-factor derivative-factor))))

(defn gelu-forward
  "Apply GELU activation element-wise to a matrix.
  
  Parameters:
    x - Input matrix (any shape)
  
  Returns:
    Output matrix with GELU applied element-wise"
  [x]
  (let [rows (mrows x)
        cols (ncols x)
        out (dge rows cols)]
    ;; Apply GELU element by element
    (dotimes [i rows]
      (dotimes [j cols]
        (entry! out i j (gelu-element (entry x i j)))))
    out))

(defn gelu-backward
  "Compute GELU gradient.
  
  Parameters:
    x    - Original input from forward pass  
    dout - Gradient from upstream
  
  Returns:
    dx - Gradient w.r.t. input"
  [x dout]
  (let [rows (mrows x)
        cols (ncols x)
        dx (dge rows cols)]
    ;; Compute gradient element by element
    (dotimes [i rows]
      (dotimes [j cols]
        (let [grad (gelu-backward-element (entry x i j))
              upstream-grad (entry dout i j)]
          (entry! dx i j (* grad upstream-grad)))))
    dx))

(defn gelu-forward-from-vec
  "Convenience wrapper for nested vector input."
  [x-vec]
  (let [x (neo/vec->matrix x-vec)
        result (gelu-forward x)]
    (neo/matrix->vec result)))

(defn gelu-backward-from-vec
  "Convenience wrapper for nested vector input."
  [x-vec dout-vec]
  (let [x (neo/vec->matrix x-vec)
        dout (neo/vec->matrix dout-vec)
        dx (gelu-backward x dout)]
    (neo/matrix->vec dx)))

(comment
  ;; REPL usage examples
  (require '[llm.neo.gelu :as gelu])
  (require '[llm.neo.core :as neo])
  
  ;; Test on small matrix
  (def x (dge 2 3 [-1.0 0.0 1.0 -0.5 0.5 2.0]))
  (def result (gelu/gelu-forward x))
  (neo/print-matrix result)
  
  ;; Backward pass
  (def dout (dge 2 3 (repeat 6 1.0)))
  (def dx (gelu/gelu-backward x dout))
  (neo/print-matrix dx)
  )