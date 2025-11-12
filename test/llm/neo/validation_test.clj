(ns llm.neo.validation-test
  "Tests for the PyTorch validation framework.
  
  Demonstrates how to validate operations against PyTorch ground truth."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.validation :as val]))

(deftest test-vector-listing
  (testing "Can list test vector files"
    (let [available-tests (val/list-test-vectors)]
      (is (vector? available-tests) "Should return a vector")
      ;; If test vectors exist, verify format
      (when (seq available-tests)
        (is (every? #(re-find #"\.edn$" %) available-tests)
            "All test files should be .edn format")))))

(deftest edn-conversion
  (testing "EDN to Neanderthal conversions work"
    (let [;; Simple nested list (row-major representation)
          edn-matrix [[1.0 2.0 3.0] [4.0 5.0 6.0]]
          edn-vector [1.0 2.0 3.0 4.0]
          
          ;; Convert to Neanderthal
          neo-matrix (val/edn->matrix edn-matrix)
          neo-vector (val/edn->vector edn-vector)]
      
      ;; Verify shapes
      (is (= [2 3] [(mrows neo-matrix) (ncols neo-matrix)]))
      (is (= 4 (dim neo-vector)))
      
      ;; Verify first value (accounting for column-major conversion)
      (is (< (Math/abs (- 1.0 (entry neo-matrix 0 0))) 1e-6))
      (is (< (Math/abs (- 1.0 (entry neo-vector 0))) 1e-6)))))

(comment
  ;; Integration test example (add when test vectors are generated):
  
  ;; This requires llm.neo.matmul to exist, so include in comment for now
  (require '[llm.neo.matmul :as matmul])
  
  ;; 1. Load test vectors
  (def matmul-test (val/load-test-vectors "dev/test_vectors/matmul_small.edn"))
  
  ;; 2. Create forward wrapper
  (defn test-matmul-forward [inputs]
    (let [inp (val/edn->matrix (:inp inputs))
          weight (val/edn->matrix (:weight inputs))
          bias (val/edn->vector (:bias inputs))]
      (matmul/matmul-forward inp weight bias)))
  
  ;; 3. Validate
  (def result (val/validate-operation matmul-test test-matmul-forward))
  (val/print-validation-result result)
  )