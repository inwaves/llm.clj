(ns llm.neo.residual-test
  "Tests for Neanderthal residual connection implementation."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.residual :as res]))

(deftest residual-forward-test
  (testing "Residual connection adds matrices element-wise"
    (let [x1 (dge 2 3 [1.0 2.0 3.0 4.0 5.0 6.0])
          x2 (dge 2 3 [10.0 20.0 30.0 40.0 50.0 60.0])
          result (res/residual-forward x1 x2)]
      
      ;; Check element-wise addition
      (is (= 11.0 (entry result 0 0)) "1 + 10 = 11")
      (is (= 22.0 (entry result 1 0)) "2 + 20 = 22")
      (is (= 33.0 (entry result 0 1)) "3 + 30 = 33")
      (is (= 66.0 (entry result 1 2)) "6 + 60 = 66"))))

(deftest residual-preserves-input-test
  (testing "Residual-forward preserves input matrices"
    (let [x1 (dge 2 2 [1.0 2.0 3.0 4.0])
          x2 (dge 2 2 [5.0 6.0 7.0 8.0])
          x1-copy (copy x1)
          x2-copy (copy x2)
          _ (res/residual-forward x1 x2)]
      
      ;; Original matrices should be unchanged
      (is (= (entry x1 0 0) (entry x1-copy 0 0)))
      (is (= (entry x2 1 1) (entry x2-copy 1 1))))))

(deftest residual-inplace-test
  (testing "Residual-forward-inplace mutates first argument"
    (let [x1 (dge 2 2 [1.0 2.0 3.0 4.0])
          x2 (dge 2 2 [10.0 20.0 30.0 40.0])
          original-x1-00 (entry x1 0 0)
          result (res/residual-forward-inplace x1 x2)]
      
      ;; Result should be same object as x1
      (is (identical? result x1) "Should return same object")
      
      ;; x1 should be modified
      (is (= 11.0 (entry x1 0 0)) "x1 should be modified")
      (is (not= original-x1-00 (entry x1 0 0))))))

(deftest residual-shape-test
  (testing "Residual works with various matrix shapes"
    (doseq [[rows cols] [[1 1] [4 8] [16 32] [128 256]]]
      (let [x1 (dge rows cols (repeat (* rows cols) 1.0))
            x2 (dge rows cols (repeat (* rows cols) 2.0))
            result (res/residual-forward x1 x2)]
        
        (is (= [rows cols] [(mrows result) (ncols result)])
            (format "Shape preserved for [%d %d]" rows cols))
        
        (is (< (Math/abs (- 3.0 (entry result 0 0))) 1e-6)
            "1 + 2 = 3 for all positions")))))

(deftest residual-zero-test
  (testing "Residual with zero matrix acts as identity"
    (let [x1 (dge 2 3 [1.0 2.0 3.0 4.0 5.0 6.0])
          zeros (dge 2 3 (repeat 6 0.0))
          result (res/residual-forward x1 zeros)]
      
      ;; Adding zeros should give back x1
      (dotimes [i 2]
        (dotimes [j 3]
          (is (= (entry x1 i j) (entry result i j))
              "x + 0 = x"))))))