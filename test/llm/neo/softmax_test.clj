(ns llm.neo.softmax-test
  "Tests for Neanderthal Softmax implementation."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.softmax :as sm]))

(deftest softmax-probability-test
  (testing "Softmax outputs valid probabilities"
    (let [x (dge 3 4 (range 12))
          result (sm/softmax-forward x)]
      
      ;; All values should be positive
      (dotimes [i 3]
        (dotimes [j 4]
          (is (>= (entry result i j) 0.0) "Probabilities should be non-negative")
          (is (<= (entry result i j) 1.0) "Probabilities should be <= 1")))
      
      ;; Each row should sum to 1
      (dotimes [i 3]
        (let [row-sum (reduce + (for [j (range 4)] (entry result i j)))]
          (is (< (Math/abs (- 1.0 row-sum)) 1e-6)
              (format "Row %d should sum to 1, got %.6f" i row-sum)))))))

(deftest softmax-max-stability-test
  (testing "Softmax with large values doesn't overflow"
    (let [;; Large values that would overflow without max subtraction
          x (dge 2 3 [1000.0 1001.0 1001.0 1002.0 1002.0 1003.0])
          result (sm/softmax-forward x)]
      
      ;; Should not produce NaN or Inf
      (dotimes [i 2]
        (dotimes [j 3]
          (is (not (Double/isNaN (entry result i j))) "Should not be NaN")
          (is (not (Double/isInfinite (entry result i j))) "Should not be Inf")))
      
      ;; Should still sum to 1
      (dotimes [i 2]
        (let [row-sum (reduce + (for [j (range 3)] (entry result i j)))]
          (is (< (Math/abs (- 1.0 row-sum)) 1e-6) "Row should sum to 1"))))))

(deftest softmax-uniform-input-test
  (testing "Softmax of uniform values gives uniform distribution"
    (let [x (dge 2 4 (repeat 8 5.0))
          result (sm/softmax-forward x)]
      
      ;; All probabilities in a row should be equal (1/4)
      (dotimes [i 2]
        (dotimes [j 4]
          (is (< (Math/abs (- 0.25 (entry result i j))) 1e-6)
              "Uniform input should give uniform output"))))))

(deftest softmax-argmax-test
  (testing "Softmax concentrates probability on maximum value"
    (let [;; Row 0: max at position 2
          x (dge 1 4 [1.0 2.0 10.0 3.0])
          result (sm/softmax-forward x)]
      
      ;; Probability at position 2 should be much larger
      (is (> (entry result 0 2) 0.9)
          "Max position should have high probability")
      
      ;; Other positions should have low probability
      (is (< (entry result 0 0) 0.05))
      (is (< (entry result 0 1) 0.05))
      (is (< (entry result 0 3) 0.05)))))

(deftest softmax-autoregressive-test
  (testing "Autoregressive softmax masks future positions"
    (let [;; 3x3 attention scores
          x (dge 3 3 [1.0 2.0 3.0 4.0 5.0 6.0 7.0 8.0 9.0])
          result (sm/softmax-autoregressive x)]
      
      ;; Row 0: only position 0 should be non-zero
      (is (< (Math/abs (- 1.0 (entry result 0 0))) 1e-6)
          "Position 0 should attend only to itself")
      (is (= 0.0 (entry result 0 1)) "Future position should be 0")
      (is (= 0.0 (entry result 0 2)) "Future position should be 0")
      
      ;; Row 1: positions 0 and 1 should be non-zero, sum to 1
      (is (> (entry result 1 0) 0.0) "Can attend to past")
      (is (> (entry result 1 1) 0.0) "Can attend to self")
      (is (= 0.0 (entry result 1 2)) "Cannot attend to future")
      (let [row1-sum (+ (entry result 1 0) (entry result 1 1))]
        (is (< (Math/abs (- 1.0 row1-sum)) 1e-6) "Unmasked positions should sum to 1"))
      
      ;; Row 2: all positions should be non-zero, sum to 1
      (is (> (entry result 2 0) 0.0))
      (is (> (entry result 2 1) 0.0))
      (is (> (entry result 2 2) 0.0))
      (let [row2-sum (reduce + (for [j (range 3)] (entry result 2 j)))]
        (is (< (Math/abs (- 1.0 row2-sum)) 1e-6))))))

(deftest softmax-shape-test
  (testing "Softmax preserves input shape"
    (doseq [[rows cols] [[1 4] [8 16] [32 64]]]
      (let [x (dge rows cols (repeatedly (* rows cols) rand))
            result (sm/softmax-forward x)]
        (is (= [rows cols] [(mrows result) (ncols result)])
            (format "Shape preserved for [%d %d]" rows cols))))))