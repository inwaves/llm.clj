(ns llm.neo.residual
  "Residual connections using Neanderthal.
  
  Residual connections add the input to the output: y = x + f(x)"
  (:require [uncomplicate.neanderthal
             [core :refer [axpy! copy! mrows ncols]]
             [native :refer [dge]]]))

(defn residual-forward
  "Add two matrices element-wise (residual connection).
  
  Computes: out = x1 + x2
  
  Args:
    x1: first input matrix
    x2: second input matrix (must be same shape as x1)
    
  Returns:
    sum matrix (x1 + x2)"
  [x1 x2]
  (let [out (copy! x1 (dge (mrows x1) (ncols x1)))]
    ;; axpy! computes: out = alpha*x2 + out
    (axpy! 1.0 x2 out)
    out))

(defn residual-forward-inplace
  "Add x2 to x1 in-place (mutates x1).
  
  More memory efficient when you don't need to preserve x1.
  
  Args:
    x1: matrix to mutate (will contain result)
    x2: matrix to add
    
  Returns:
    x1 (modified)"
  [x1 x2]
  (axpy! 1.0 x2 x1))