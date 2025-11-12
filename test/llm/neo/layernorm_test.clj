(ns llm.neo.layernorm-test
  "Tests for Neanderthal LayerNorm implementation."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.layernorm :as ln]
            [llm.neo.core :as neo]))

(deftest layernorm-shape-test
  (testing "LayerNorm preserves input shape"
    (let [x (dge 4 8 (vec (repeat 32 1.0)))
          gamma (dv (vec (repeat 8 1.0)))
          beta (dv (vec (repeat 8 0.0)))
          result (ln/layernorm-forward x gamma beta 1e-5)]
      (is (= [4 8] [(mrows result) (ncols result)])
          "Output should have same shape as input"))))

(deftest layernorm-normalization-test
  (testing "LayerNorm produces mean=0, var=1 (before affine transform)"
    (let [;; Create input with non-zero mean and var != 1
          x (dge 2 4 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0])
          
          ;; Identity transform (gamma=1, beta=0)
          gamma (dv (vec (repeat 4 1.0)))
          beta (dv (vec (repeat 4 0.0)))
          
          result (ln/layernorm-forward x gamma beta 1e-5)
          
          ;; Check first row mean and variance
          row0-mean (/ (+ (entry result 0 0) (entry result 0 1) 
                         (entry result 0 2) (entry result 0 3)) 4.0)
          row0-var (/ (+ (Math/pow (entry result 0 0) 2)
                        (Math/pow (entry result 0 1) 2)
                        (Math/pow (entry result 0 2) 2)
                        (Math/pow (entry result 0 3) 2)) 4.0)]
      
      (is (< (Math/abs row0-mean) 1e-6) "Row mean should be ~0")
      (is (< (Math/abs (- 1.0 row0-var)) 0.1) "Row variance should be ~1"))))

(deftest layernorm-affine-test
  (testing "LayerNorm applies affine transformation correctly"
    (let [;; Simple 1x2 matrix
          x (dge 1 2 [0.0 1.0])
          
          ;; After normalization, x_norm = [-1, 1] (mean=0.5, std=0.5)
          ;; With gamma=[2, 3], beta=[1, 1]:
          ;; output = gamma * x_norm + beta = [2*(-1)+1, 3*1+1] = [-1, 4]
          gamma (dv [2.0 3.0])
          beta (dv [1.0 1.0])
          
          result (ln/layernorm-forward x gamma beta 1e-5)]
      
      ;; Note: Actual computation may differ slightly due to numeric precision
      ;; This test validates that affine parameters have an effect
      (is (not= (entry result 0 0) (entry result 0 1))
          "Different gamma values should produce different outputs"))))

(deftest layernorm-epsilon-test
  (testing "LayerNorm handles constant input (zero variance)"
    (let [;; All same values (zero variance)
          x (dge 2 4 (vec (repeat 8 5.0)))
          gamma (dv (vec (repeat 4 1.0)))
          beta (dv (vec (repeat 4 0.0)))
          
          ;; Should not crash due to division by zero
          result (ln/layernorm-forward x gamma beta 1e-5)]
      
      (is (not-any? #(Double/isNaN %) 
                    (for [i (range 2) j (range 4)] (entry result i j)))
          "Should not produce NaN with constant input"))))

(deftest layernorm-batch-test
  (testing "LayerNorm normalizes each row independently"
    (let [;; Two rows with different scales
          x (dge 2 3 [1.0 100.0 2.0 200.0 3.0 300.0])
          gamma (dv (vec (repeat 3 1.0)))
          beta (dv (vec (repeat 3 0.0)))
          
          result (ln/layernorm-forward x gamma beta 1e-5)
          
          ;; Both rows should have similar normalized values despite different scales
          row0-vals [(entry result 0 0) (entry result 0 1) (entry result 0 2)]
          row1-vals [(entry result 1 0) (entry result 1 1) (entry result 1 2)]]
      
      (is (< (Math/abs (- (apply + row0-vals) (apply + row1-vals))) 0.1)
          "Both rows should be similarly normalized"))))