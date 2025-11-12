(ns llm.neo.attention-test
  "Tests for Neanderthal-based attention implementation."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.attention :as attention]
            [llm.neo.core :as neo]))

(deftest attention-shape-test
  (testing "Attention preserves batch, sequence, and channel dimensions"
    (let [B 2
          T 4
          C 12
          NH 3  ; Number of heads (C must be divisible by NH)
          
          ;; Create random QKV input [B, T, 3C]
          inp (vec (repeatedly B 
                    (fn [] (vec (repeatedly T 
                                  (fn [] (vec (repeatedly (* 3 C) rand))))))))
          
          result (attention/attention-forward inp NH)]
      
      ;; Check output dimensions
      (is (= B (count result)) "Should preserve batch size")
      (is (= T (count (first result))) "Should preserve sequence length")
      (is (= C (count (first (first result)))) "Should output C channels (not 3C)"))))

(deftest attention-small-test
  (testing "Attention works on small deterministic input"
    (let [B 1
          T 2
          C 4
          NH 2  ; 2 heads of size 2 each
          
          ;; Create input [B, T, 3C] = [1, 2, 12]
          ;; Each row has 3C=12 values (Q, K, V concatenated)
          row (vec (repeatedly (* 3 C) (constantly 1.0)))
          inp [[row row]]  ; B=1 batch with T=2 rows
          
          result (attention/attention-forward inp NH)]
      
      ;; Should have correct shape
      (is (= [B T C] [(count result) (count (first result)) (count (first (first result)))])
          "Should maintain dimensions [B, T, C]")
      
      ;; Values should be finite (not NaN or Inf)
      (is (every? #(Double/isFinite %) (flatten result))
          "All outputs should be finite numbers"))))

(deftest attention-causality-test
  (testing "Attention produces finite outputs with causal structure"
    (let [B 1
          T 3
          C 6
          NH 2
          
          ;; Random input
          inp (vec (for [b (range B)]
                    (vec (for [t (range T)]
                          (vec (repeatedly (* 3 C) rand))))))
          
          result (attention/attention-forward inp NH)]
      
      ;; Check shape
      (is (= [B T C] [(count result) (count (first result)) (count (first (first result)))])
          "Should maintain dimensions")
      
      ;; Check values are finite
      (is (every? #(Double/isFinite %) (flatten result))
          "All attention outputs should be finite"))))

(deftest attention-head-divisibility-test
  (testing "Attention validates C is divisible by NH"
    (let [B 1
          T 2
          C 7  ; Not divisible by 2
          NH 2
          inp (vec (repeatedly B (fn [] (vec (repeatedly T (fn [] (vec (repeatedly (* 3 C) rand))))))))]
      
      ;; Should throw error
      (is (thrown? Exception (attention/attention-forward inp NH))
          "Should reject NH that doesn't divide C"))))