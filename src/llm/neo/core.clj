(ns llm.neo.core
  "Core utilities for Neanderthal-based implementations.
   
   This namespace provides:
   - Matrix creation and conversion utilities
   - Validation against pure Clojure implementations
   - Performance benchmarking helpers
   - Common configuration"
  (:require [uncomplicate.neanderthal.core :as nc]
            [uncomplicate.neanderthal.native :as nn]
            [uncomplicate.neanderthal.vect-math :as vm]))

;; ============================================================================
;; Matrix Creation and Conversion
;; ============================================================================

(defn vec->matrix
  "Convert a Clojure nested vector to a Neanderthal matrix.
   
   Example:
     (vec->matrix [[1 2 3] [4 5 6]]) 
     => 2x3 matrix with row-major layout"
  [nested-vec]
  (let [rows (count nested-vec)
        cols (count (first nested-vec))
        flat (vec (apply concat nested-vec))]
    (nn/dge rows cols flat {:layout :row})))

(defn matrix->vec
  "Convert a Neanderthal matrix to a Clojure nested vector.
   
   Example:
     (matrix->vec (nn/dge 2 3 [1 2 3 4 5 6]))
     => [[1.0 2.0 3.0] [4.0 5.0 6.0]]"
  [matrix]
  (let [rows (nc/mrows matrix)
        cols (nc/ncols matrix)]
    (vec (for [i (range rows)]
           (vec (for [j (range cols)]
                  (nc/entry matrix i j)))))))

(defn tensor->matrices
  "Convert a 3D tensor (B, T, C) to a vector of Neanderthal matrices.
   Each matrix is (T, C) for a single batch element."
  [tensor]
  (mapv vec->matrix tensor))

(defn matrices->tensor
  "Convert a vector of Neanderthal matrices back to a 3D tensor."
  [matrices]
  (mapv matrix->vec matrices))

;; ============================================================================
;; Validation Helpers
;; ============================================================================

(defn close-enough?
  "Check if two numbers are close within relative and absolute tolerance.
   
   |a - b| <= atol + rtol * |b|"
  ([a b] (close-enough? a b 1e-5 1e-8))
  ([a b rtol atol]
   (<= (Math/abs (- a b))
       (+ atol (* rtol (Math/abs b))))))

(defn matrices-close?
  "Check if two matrices are numerically close.
   
   Compares element-wise within tolerance."
  ([m1 m2] (matrices-close? m1 m2 1e-5 1e-8))
  ([m1 m2 rtol atol]
   (and (= (nc/mrows m1) (nc/mrows m2))
        (= (nc/ncols m1) (nc/ncols m2))
        (every? identity
                (for [i (range (nc/mrows m1))
                      j (range (nc/ncols m1))]
                  (close-enough? (nc/entry m1 i j)
                                (nc/entry m2 i j)
                                rtol atol))))))

(defn tensors-close?
  "Check if two tensors (as nested vectors) are numerically close."
  ([t1 t2] (tensors-close? t1 t2 1e-5 1e-8))
  ([t1 t2 rtol atol]
   (let [flat1 (flatten t1)
         flat2 (flatten t2)]
     (and (= (count flat1) (count flat2))
          (every? (fn [[a b]] (close-enough? a b rtol atol))
                  (map vector flat1 flat2))))))

;; ============================================================================
;; Numerical Validation
;; ============================================================================

(defn max-error
  "Find maximum absolute error between two matrices/tensors.
  
  Args:
    a, b - Either both Neanderthal matrices or both nested vectors
    
  Returns:
    Maximum absolute difference across all elements"
  [a b]
  (cond
    ;; Both are Neanderthal matrices
    (and (instance? uncomplicate.neanderthal.internal.api.Matrix a)
         (instance? uncomplicate.neanderthal.internal.api.Matrix b))
    (let [rows (nc/mrows a)
          cols (nc/ncols a)]
      (apply max
        (for [i (range rows)
              j (range cols)]
          (Math/abs (- (nc/entry a i j) (nc/entry b i j))))))
    
    ;; Both are nested vectors
    :else
    (let [flat-a (flatten a)
          flat-b (flatten b)]
      (apply max (map #(Math/abs (- %1 %2)) flat-a flat-b)))))

(defn allclose
  "Check if two matrices/tensors are numerically close within tolerance.
  
  Uses same formula as NumPy: |a - b| <= atol + rtol * |b|
  
  Args:
    a, b - Either both Neanderthal matrices or both nested vectors
    rtol - Relative tolerance (default 1e-5)
    atol - Absolute tolerance (default 1e-8)
    
  Returns:
    true if all elements are within tolerance"
  ([a b] (allclose a b 1e-5 1e-8))
  ([a b rtol atol]
   (cond
     ;; Both are Neanderthal matrices
     (and (instance? uncomplicate.neanderthal.internal.api.Matrix a)
          (instance? uncomplicate.neanderthal.internal.api.Matrix b))
     (matrices-close? a b rtol atol)
     
     ;; Both are nested vectors
     :else
     (tensors-close? a b rtol atol))))

;; ============================================================================
;; Performance Benchmarking
;; ============================================================================

(defmacro time-ms
  "Time an expression and return [result time-in-ms]"
  [expr]
  `(let [start# (System/nanoTime)
         result# ~expr
         end# (System/nanoTime)]
     [result# (/ (- end# start#) 1000000.0)]))

(defn benchmark
  "Run a function n times and return statistics.
   
   Returns map with :mean, :min, :max, :median (all in milliseconds)"
  [f n]
  (let [times (vec (repeatedly n #(second (time-ms (f)))))
        sorted-times (sort times)]
    {:mean (/ (reduce + times) n)
     :min (first sorted-times)
     :max (last sorted-times)
     :median (nth sorted-times (quot n 2))
     :samples n}))

(defn compare-performance
  "Compare performance of two functions with the same inputs.
   
   Returns map with :baseline, :optimized, and :speedup"
  [baseline-fn optimized-fn n]
  (let [baseline-stats (benchmark baseline-fn n)
        optimized-stats (benchmark optimized-fn n)
        speedup (/ (:mean baseline-stats) (:mean optimized-stats))]
    {:baseline baseline-stats
     :optimized optimized-stats
     :speedup speedup}))

;; ============================================================================
;; Display Helpers
;; ============================================================================

(defn print-comparison
  "Pretty-print a performance comparison result"
  [{:keys [baseline optimized speedup]}]
  (println "Performance Comparison:")
  (println "----------------------")
  (printf "Baseline:  %.2f ms (min: %.2f, max: %.2f)\n"
          (:mean baseline) (:min baseline) (:max baseline))
  (printf "Optimized: %.2f ms (min: %.2f, max: %.2f)\n"
          (:mean optimized) (:min optimized) (:max optimized))
  (printf "Speedup:   %.1fx\n" speedup))