(ns llm.neo.matmul-simple-test
  "Tests for Neanderthal-based matrix multiplication.
   
   This test suite validates:
   1. Numerical correctness against pure Clojure implementation
   2. Performance improvements
   3. Gradient computation correctness"
  (:require [clojure.test :refer [deftest is testing]]
            [llm.matmul :as matmul-pure]
            [llm.neo.matmul :as matmul-neo]
            [llm.neo.core :as neo]
            [llm.utils :as utils]))

;; ============================================================================
;; Test Data
;; ============================================================================

(def small-test-data
  "Small test case matching the original test"
  {:inp [[1 2 3] [4 5 6]]         ;; (2, 3) - two samples, 3 features
   :weight [[0 1 2]                ;; (4, 3) - 4 output channels
            [1 2 3]
            [2 3 4]
            [3 4 5]]
   :bias [0 1 2 3]                 ;; (4,) - one bias per output channel
   :expected [[8 15 22 29]         ;; (2, 4) - expected output
              [17 33 49 65]]})

(def medium-test-data
  "Medium test case for performance comparison"
  (let [BT 64
        C 256
        OC 512
        inp (vec (repeatedly BT #(vec (repeatedly C rand))))
        weight (vec (repeatedly OC #(vec (repeatedly C rand))))
        bias (vec (repeatedly OC rand))]
    {:inp inp
     :weight weight
     :bias bias
     :BT BT
     :C C
     :OC OC}))

;; ============================================================================
;; Correctness Tests
;; ============================================================================

(deftest matmul-forward-correctness-test
  (testing "Neanderthal matmul produces same results as pure Clojure"
    (let [{:keys [inp weight bias expected]} small-test-data
          pure-result @(matmul-pure/matmul_forward 
                        (utils/t_zeros [2 4])
                        (atom inp)
                        (atom weight)
                        (atom bias))
          neo-result (matmul-neo/matmul-forward inp weight bias)]
      
      ;; Check dimensions
      (is (= (count neo-result) (count expected)))
      (is (= (count (first neo-result)) (count (first expected))))
      
      ;; Check numerical equivalence with pure implementation
      (is (neo/tensors-close? neo-result pure-result 1e-5 1e-8)
          "Neanderthal result should match pure Clojure")
      
      ;; Check against expected values
      (is (neo/tensors-close? neo-result expected 1e-5 1e-8)
          "Should match expected output"))))

(deftest matmul-no-bias-test
  (testing "Matmul works correctly without bias"
    (let [{:keys [inp weight]} small-test-data
          ;; Pure implementation with nil bias
          pure-result @(matmul-pure/matmul_forward
                        (utils/t_zeros [2 4])
                        (atom inp)
                        (atom weight)
                        nil)
          ;; Neo implementation with nil bias  
          neo-result (matmul-neo/matmul-forward inp weight nil)]
      
      (is (neo/tensors-close? neo-result pure-result 1e-5 1e-8)
          "No-bias case should match pure implementation"))))

(deftest matmul-backward-correctness-test
  (testing "Backward pass produces correct gradients"
    (let [{:keys [inp weight]} small-test-data
          ;; Gradient of loss w.r.t. output (random for testing)
          dout [[1.0 0.5 0.3 0.2]
                [0.8 0.6 0.4 0.3]]
          
          ;; Pure implementation
          dinp-pure (utils/t_zeros_like (atom inp))
          dweight-pure (utils/t_zeros_like (atom weight))
          dbias-pure (atom (vec (repeat 4 0.0)))
          _ (matmul-pure/matmul_backward
             dinp-pure dweight-pure dbias-pure
             (atom dout) (atom inp) (atom weight))
          
          ;; Neo implementation
          {:keys [dinp dweight dbias]} 
          (matmul-neo/matmul-backward dout inp weight)]
      
      ;; Check gradients match
      (is (neo/tensors-close? dinp @dinp-pure 1e-5 1e-8)
          "dinp should match")
      (is (neo/tensors-close? dweight @dweight-pure 1e-5 1e-8)
          "dweight should match")
      (is (neo/tensors-close? [dbias] [@dbias-pure] 1e-5 1e-8)
          "dbias should match"))))

;; ============================================================================
;; Performance Tests
;; ============================================================================

(deftest matmul-performance-test
  (testing "Neanderthal provides significant speedup"
    (let [{:keys [inp weight bias]} medium-test-data
          inp-atom (atom inp)
          weight-atom (atom weight)
          bias-atom (atom bias)
          out-atom (atom (vec (repeatedly (:BT medium-test-data) 
                                         #(vec (repeat (:OC medium-test-data) 0.0)))))
          
          ;; Benchmark pure Clojure (fewer iterations - it's slow!)
          pure-fn #(matmul-pure/matmul_forward out-atom inp-atom weight-atom bias-atom)
          pure-stats (neo/benchmark pure-fn 3)
          
          ;; Benchmark Neanderthal (more iterations)
          neo-fn #(matmul-neo/matmul-forward inp weight bias)
          neo-stats (neo/benchmark neo-fn 10)
          
          speedup (/ (:mean pure-stats) (:mean neo-stats))]
      
      ;; Print results
      (println "\nPerformance Test Results:")
      (println "------------------------")
      (printf "Input dimensions: (%d, %d) x (%d, %d)\n"
              (:BT medium-test-data) (:C medium-test-data)
              (:OC medium-test-data) (:C medium-test-data))
      (printf "Pure Clojure:  %.2f ms (n=%d)\n" 
              (:mean pure-stats) (:samples pure-stats))
      (printf "Neanderthal:   %.2f ms (n=%d)\n" 
              (:mean neo-stats) (:samples neo-stats))
      (printf "Speedup:       %.1fx\n" speedup)
      
      ;; Assert we get meaningful speedup (at least 5x on CPU)
      (is (> speedup 5.0)
          (format "Expected >5x speedup, got %.1fx" speedup)))))

;; ============================================================================
;; Integration Test
;; ============================================================================

(deftest full-forward-backward-cycle-test
  (testing "Full forward and backward pass with gradient check"
    (let [BT 4
          C 8
          OC 16
          ;; Create test data
          inp (vec (repeatedly BT (fn [] (vec (repeatedly C rand)))))
          weight (vec (repeatedly OC (fn [] (vec (repeatedly C (fn [] (* 0.01 (- (rand) 0.5))))))))
          bias (vec (repeatedly OC (fn [] (* 0.01 (rand)))))
          
          ;; Forward pass
          output (matmul-neo/matmul-forward inp weight bias)
          
          ;; Fake gradient from "loss"
          dout (vec (repeatedly BT (fn [] (vec (repeatedly OC (fn [] (* 0.01 (- (rand) 0.5))))))))
          
          ;; Backward pass
          {:keys [dinp dweight dbias]} 
          (matmul-neo/matmul-backward dout inp weight)]
      
      ;; Basic sanity checks
      (is (= (count dinp) BT))
      (is (= (count (first dinp)) C))
      (is (= (count dweight) OC))
      (is (= (count (first dweight)) C))
      (is (= (count dbias) OC))
      
      ;; Check gradients are not all zeros (they should have been updated)
      (is (not (every? zero? (flatten dinp))))
      (is (not (every? zero? (flatten dweight))))
      (is (not (every? zero? dbias))))))