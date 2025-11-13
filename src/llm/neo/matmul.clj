(ns llm.neo.matmul
  "Matrix multiplication using Neanderthal (BLAS-backed).
   
   This namespace provides Neanderthal-based implementations of matrix multiplication,
   offering 10-50x speedup over the pure Clojure implementation in llm.matmul.
   
   Key operations:
   - matmul-forward: Forward pass of fully-connected layer
   - matmul-backward: Backward pass (gradients w.r.t inputs, weights, bias)
   
   Interface matches llm.matmul for easy comparison."
  (:use [uncomplicate.neanderthal core native])
  (:require [uncomplicate.neanderthal.vect-math :as vmath]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Forward Pass
;; ============================================================================

(defn matmul-forward
  "Matrix multiplication forward pass using Neanderthal.
   
   Computes: out = inp @ weight^T + bias
   
   Parameters:
     inp    - Input matrix (B×T, C) where B=batch, T=sequence, C=channels
     weight - Weight matrix (OC, C) where OC=output channels
     bias   - Bias vector (OC,) - can be nil
   
   Returns:
     Output matrix (B×T, OC)
   
   Implementation notes:
   - Uses BLAS gemm (General Matrix Multiply) for optimal performance
   - Weight is transposed during multiplication (matches PyTorch convention)
   - Bias is added via axpy! (y = ax + y) for efficiency"
  [inp weight bias]
  (let [;; Extract dimensions
        bt (mrows inp)      ; B*T (flattened batch and sequence)
        c (ncols inp)       ; Input channels
        oc (mrows weight)   ; Output channels
        
        ;; Validate shapes
        _ (when (not= c (ncols weight))
            (throw (ex-info "Shape mismatch in matmul"
                           {:inp-shape [(mrows inp) (ncols inp)]
                            :weight-shape [(mrows weight) (ncols weight)]
                            :expected-weight-cols c})))
        
        ;; Create output matrix
        out (dge bt oc)
        
        ;; Perform matrix multiplication: out = inp @ weight^T
        _ (mm! 1.0 inp (trans weight) 0.0 out)]
    
    ;; Add bias if provided
    (when bias
      ;; Broadcast bias across all rows
      (dotimes [i bt]
        (axpy! 1.0 bias (row out i))))
    
    out))

(defn matmul-forward-matrices
  "Forward pass operating directly on Neanderthal matrices.
   
   This is the core implementation used by matmul-forward.
   Use this version when working with matrices directly for better performance.
   
   Computes: out = inp @ weight^T + bias
   
   Arguments:
   - inp-mat: Input matrix (B*T, C) as Neanderthal matrix
   - weight-mat: Weight matrix (OC, C) as Neanderthal matrix  
   - bias: Optional bias vector (OC,) as Clojure vector or nil
   
   Returns:
   - Output matrix (B*T, OC) as Neanderthal matrix"
  [inp-mat weight-mat bias]
  (let [BT (mrows inp-mat)
        C (ncols inp-mat)
        OC (mrows weight-mat)]
    (assert (= C (ncols weight-mat))
            (str "Dimension mismatch: inp has " C " columns but weight has " 
                 (ncols weight-mat) " columns"))
    
    (if bias
      (let [out (dge BT OC
                        (vec (apply concat (repeat BT bias)))
                        {:layout :row})]
        (mm! 1.0 inp-mat (trans weight-mat) 1.0 out)
        out)
      (let [out (dge BT OC)]
        (mm! 1.0 inp-mat (trans weight-mat) 0.0 out)
        out))))

;; ============================================================================
;; Backward Pass
;; ============================================================================

(defn matmul-backward
  "Matrix multiplication backward pass.
   
   Given gradient w.r.t output (dout), computes gradients w.r.t:
   - dinp: Input
   - dweight: Weight
   - dbias: Bias (optional)
   
   Parameters:
     dout   - Gradient w.r.t output (B×T, OC)
     inp    - Input from forward pass (B×T, C)
     weight - Weight from forward pass (OC, C)
   
   Returns:
     Map with keys:
       :dinp    - Gradient w.r.t input (B×T, C)
       :dweight - Gradient w.r.t weight (OC, C)
       :dbias   - Gradient w.r.t bias (OC,) - optional
   
   Mathematical derivation:
     Forward:  out = inp @ weight^T + bias
     Backward: dinp = dout @ weight
               dweight = dout^T @ inp
               dbias = sum(dout, axis=0)"
  [dout inp weight]
  (let [bt (mrows inp)
        c (ncols inp)
        oc (ncols dout)
        
        ;; Gradient w.r.t input: dinp = dout @ weight
        dinp (dge bt c)
        _ (mm! 1.0 dout weight 0.0 dinp)
        
        ;; Gradient w.r.t weight: dweight = dout^T @ inp
        dweight (dge oc c)
        _ (mm! 1.0 (trans dout) inp 0.0 dweight)
        
        ;; Gradient w.r.t bias: sum dout across batch dimension
        dbias (dv oc)
        _ (dotimes [i bt]
            (axpy! 1.0 (row dout i) dbias))]
    
    {:dinp dinp
     :dweight dweight
     :dbias dbias}))

;; ============================================================================
;; Convenience Functions
;; ============================================================================

(defn matmul-forward-from-vecs
  "Convenience wrapper that accepts nested Clojure vectors.
   
   Converts to Neanderthal matrices, performs computation, converts back."
  [inp-vec weight-vec bias-vec]
  (let [inp (neo/vec->matrix inp-vec)
        weight (neo/vec->matrix weight-vec)
        bias (when bias-vec (dv bias-vec))
        result (matmul-forward inp weight bias)]
    (neo/matrix->vec result)))

(defn matmul-backward-from-vecs
  "Convenience wrapper for backward pass with nested vectors."
  [dout-vec inp-vec weight-vec]
  (let [dout (neo/vec->matrix dout-vec)
        inp (neo/vec->matrix inp-vec)
        weight (neo/vec->matrix weight-vec)
        {:keys [dinp dweight dbias]} (matmul-backward dout inp weight)]
    {:dinp (neo/matrix->vec dinp)
     :dweight (neo/matrix->vec dweight)
     ;; Correctly extract all entries from dbias vector
     :dbias (mapv #(entry dbias %) (range (dim dbias)))}))

;; ============================================================================
;; Testing and Benchmarking
;; ============================================================================

(comment
  ;; REPL exploration examples
  
  ;; Small test case
  (require '[llm.neo.matmul :as neo-mm])
  (require '[llm.neo.core :as neo])
  
  ;; Create small test matrices (dge and dv are in scope from :use)
  (def inp (dge 2 3 [1 2 3 4 5 6]))  ; 2x3 matrix
  (def weight (dge 4 3 (range 12)))   ; 4x3 matrix
  (def bias (dv [1 2 3 4]))           ; bias vector
  
  ;; Forward pass
  (def result (neo-mm/matmul-forward inp weight bias))
  (neo/print-matrix result)
  
  ;; Backward pass
  (def dout (dge 2 4 (repeat 8 1.0)))  ; gradient all ones
  (def grads (neo-mm/matmul-backward dout inp weight))
  
  (neo/print-matrix (:dinp grads))
  (neo/print-matrix (:dweight grads))
  (println (:dbias grads))
  
  )
