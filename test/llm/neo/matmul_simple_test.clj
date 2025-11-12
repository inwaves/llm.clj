(ns llm.neo.matmul-simple-test
  "Simple focused tests for Neanderthal matmul implementation.
  
  Note: Neanderthal uses column-major order (like BLAS/Fortran).
  For a 2x3 matrix with rows [[1 2 3],[4 5 6]], the data is [1 4 2 5 3 6]."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.matmul :as matmul]))

(deftest basic-forward-test
  (testing "Basic forward pass with known values (column-major order)"
    (let [;; Create 2x3 input matrix
          ;; Rows: [[1 2 3],[4 5 6]]
          ;; Column-major: [1 4 | 2 5 | 3 6]
          inp (dge 2 3 [1.0 4.0 2.0 5.0 3.0 6.0])
          
          ;; Create 4x3 weight matrix  
          ;; Rows: [[0 1 2],[1 2 3],[2 3 4],[3 4 5]]
          ;; Column-major: [0 1 2 3 | 1 2 3 4 | 2 3 4 5]
          weight (dge 4 3 [0.0 1.0 2.0 3.0 1.0 2.0 3.0 4.0 2.0 3.0 4.0 5.0])
          
          ;; Bias vector: [0 1 2 3]
          bias (dv [0.0 1.0 2.0 3.0])
          
          ;; Forward: out = inp @ weight^T + bias
          result (matmul/matmul-forward inp weight bias)]
      
      ;; Check dimensions: should be 2x4
      (is (= 2 (mrows result)) "Output should have 2 rows")
      (is (= 4 (ncols result)) "Output should have 4 columns")
      
      ;; Expected output (hand-calculated):
      ;; Row 0: [[8 15 22 29]]
      ;; Row 1: [[17 33 49 65]]
      (is (< (Math/abs (- 8.0 (entry result 0 0))) 1e-5) "result[0,0] should be 8")
      (is (< (Math/abs (- 15.0 (entry result 0 1))) 1e-5) "result[0,1] should be 15")
      (is (< (Math/abs (- 22.0 (entry result 0 2))) 1e-5) "result[0,2] should be 22")
      (is (< (Math/abs (- 29.0 (entry result 0 3))) 1e-5) "result[0,3] should be 29")
      (is (< (Math/abs (- 17.0 (entry result 1 0))) 1e-5) "result[1,0] should be 17")
      (is (< (Math/abs (- 33.0 (entry result 1 1))) 1e-5) "result[1,1] should be 33")
      (is (< (Math/abs (- 49.0 (entry result 1 2))) 1e-5) "result[1,2] should be 49")
      (is (< (Math/abs (- 65.0 (entry result 1 3))) 1e-5) "result[1,3] should be 65"))))

(deftest basic-backward-test
  (testing "Basic backward pass produces correct gradient shapes"
    (let [;; Create test matrices
          inp (dge 4 8 (repeat 32 1.0))
          weight (dge 16 8 (repeat 128 0.5))
          dout (dge 4 16 (repeat 64 1.0))
          
          ;; Compute backward pass
          {:keys [dinp dweight dbias]} (matmul/matmul-backward dout inp weight)]
      
      ;; Verify shapes
      (is (= [4 8] [(mrows dinp) (ncols dinp)]) "dinp should be 4x8")
      (is (= [16 8] [(mrows dweight) (ncols dweight)]) "dweight should be 16x8")
      (is (= 16 (dim dbias)) "dbias should have 16 elements")
      
      ;; Verify gradients are non-zero
      (is (> (asum dinp) 0) "dinp should have non-zero gradients")
      (is (> (asum dweight) 0) "dweight should have non-zero gradients")
      (is (> (asum dbias) 0) "dbias should have non-zero gradients"))))

(deftest forward-no-bias-test
  (testing "Forward pass without bias"
    (let [;; 2x3 matrix (column-major)
          inp (dge 2 3 [1.0 4.0 2.0 5.0 3.0 6.0])
          ;; 4x3 weight (column-major)
          weight (dge 4 3 (vec (range 12)))
          
          result (matmul/matmul-forward inp weight nil)]
      
      ;; Should still work and have correct shape
      (is (= [2 4] [(mrows result) (ncols result)]) 
          "Output shape should be 2x4 even without bias"))))

(deftest performance-measurement
  (testing "Measure Neanderthal matmul performance"
    (let [;; Medium-sized matrices  
          inp (dge 32 64 (repeat 2048 1.0))
          weight (dge 128 64 (repeat 8192 0.5))
          bias (dv (repeat 128 0.1))
          
          ;; Time a single forward pass
          start (System/nanoTime)
          _ (matmul/matmul-forward inp weight bias)
          duration-ms (/ (- (System/nanoTime) start) 1e6)]
      
      ;; Just print for information, no flaky assertions
      (println (format "\nNe anderthal forward pass (32×64→128): %.2f ms" duration-ms)))))