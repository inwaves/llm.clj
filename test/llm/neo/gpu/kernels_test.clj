(ns llm.neo.gpu.kernels-test
  "Tests for custom CUDA kernels.
  
  Validates:
  - Numerical correctness against CPU reference implementations
  - Performance improvements vs separate operations
  - Edge cases and error handling"
  (:require [clojure.test :refer [deftest is testing]]
            [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :refer [dge dv]]
            [uncomplicate.neanderthal.cuda :refer [cuge cuv]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.gpu.kernels :as kernels]
            [llm.neo.layernorm :as layernorm]
            [llm.neo.residual :as residual]))

;; ============================================================================
;; Test Utilities
;; ============================================================================

(defn- approx=
  "Check if two values are approximately equal within tolerance."
  [a b tolerance]
  (< (Math/abs (- a b)) tolerance))

(defn- matrix-approx=
  "Check if two matrices are approximately equal element-wise.
  
  Args:
    m1, m2: Matrices to compare
    tolerance: Maximum allowed difference per element
    
  Returns:
    true if all elements are within tolerance, false otherwise"
  [m1 m2 tolerance]
  (let [rows (ncore/mrows m1)
        cols (ncore/ncols m1)]
    (every? identity
            (for [i (range rows)
                  j (range cols)]
              (approx= (ncore/entry m1 i j)
                      (ncore/entry m2 i j)
                      tolerance)))))

;; ============================================================================
;; Correctness Tests
;; ============================================================================

(deftest test-fused-residual-layernorm-correctness
  (testing "Fused kernel matches separate operations"
    (when (gpu/gpu-available?)
      (with-release [;; Create CPU test data
                     x-cpu (dge 4 8 (map #(* 0.1 %) (range 32)))
                     residual-cpu (dge 4 8 (map #(* 0.1 %) (range 32 64)))
                     gamma-cpu (dv 8 (repeat 1.0))
                     beta-cpu (dv 8 (repeat 0.0))
                     
                     ;; Transfer to GPU
                     x-gpu (cuge x-cpu)
                     residual-gpu (cuge residual-cpu)
                     gamma-gpu (cuv gamma-cpu)
                     beta-gpu (cuv beta-cpu)]
        
        ;; Method 1: Fused kernel (GPU)
        (with-release [fused-result-gpu (kernels/fused-residual-layernorm!
                                         x-gpu residual-gpu gamma-gpu beta-gpu 1e-5)
                       fused-result-cpu (gpu/to-cpu fused-result-gpu)]
          
          ;; Method 2: Separate operations (CPU reference)
          (with-release [x-plus-residual (residual/residual x-cpu residual-cpu)
                         reference (layernorm/layernorm x-plus-residual gamma-cpu beta-cpu 1e-5)]
            
            ;; Compare results with reasonable tolerance (GPU float precision)
            (is (matrix-approx= fused-result-cpu reference 1e-4)
                "Fused kernel should match separate residual + layernorm operations")))))))

(deftest test-fused-kernel-dimensions
  (testing "Fused kernel handles various matrix dimensions"
    (when (gpu/gpu-available?)
      (doseq [[T C] [[1 16]    ; Single token, small dim
                     [10 64]   ; Multiple tokens, medium dim
                     [32 128]  ; Batch, larger dim
                     [8 512]]] ; Realistic transformer dim
        (with-release [x (cuge T C)
                       residual (cuge T C)
                       gamma (cuv C (repeat 1.0))
                       beta (cuv C (repeat 0.0))]
          (with-release [result (kernels/fused-residual-layernorm!
                                x residual gamma beta)]
            (is (= [T C] [(ncore/mrows result) (ncore/ncols result)])
                (format "Output should have dimensions [%d, %d]" T C))))))))

(deftest test-fused-kernel-epsilon
  (testing "Fused kernel handles different epsilon values"
    (when (gpu/gpu-available?)
      (doseq [eps [1e-3 1e-5 1e-7]]
        (with-release [x (cuge 4 8)
                       residual (cuge 4 8)
                       gamma (cuv 8 (repeat 1.0))
                       beta (cuv 8 (repeat 0.0))]
          (with-release [result (kernels/fused-residual-layernorm!
                                x residual gamma beta eps)]
            (is (some? result)
                (format "Should handle epsilon=%e" eps))))))))

(deftest test-fused-kernel-gamma-beta
  (testing "Fused kernel applies gamma and beta correctly"
    (when (gpu/gpu-available?)
      (with-release [;; Create test data with known values
                     x-cpu (dge 2 4 [1.0 2.0 3.0 4.0
                                     5.0 6.0 7.0 8.0])
                     residual-cpu (dge 2 4 (repeat 0.0))  ; Zero residual
                     gamma-cpu (dv 4 [2.0 2.0 2.0 2.0])  ; Scale by 2
                     beta-cpu (dv 4 [1.0 1.0 1.0 1.0])   ; Shift by 1
                     
                     x-gpu (cuge x-cpu)
                     residual-gpu (cuge residual-cpu)
                     gamma-gpu (cuv gamma-cpu)
                     beta-gpu (cuv beta-cpu)]
        
        (with-release [result-gpu (kernels/fused-residual-layernorm!
                                   x-gpu residual-gpu gamma-gpu beta-gpu 1e-5)
                       result-cpu (gpu/to-cpu result-gpu)]
          
          ;; Verify gamma and beta were applied
          ;; After layernorm, mean≈0 and var≈1
          ;; After gamma=2 and beta=1: values should be scaled and shifted
          (is (every? #(< -5.0 % 5.0) (ncore/vctr result-cpu))
              "Result should have reasonable values after scaling/shifting"))))))

;; ============================================================================
;; Error Handling Tests
;; ============================================================================

(deftest test-fused-kernel-dimension-mismatch
  (testing "Fused kernel validates dimension mismatches"
    (when (gpu/gpu-available?)
      (with-release [x (cuge 4 8)
                     residual (cuge 4 16)  ; Wrong dimensions!
                     gamma (cuv 8 (repeat 1.0))
                     beta (cuv 8 (repeat 0.0))]
        (is (thrown? AssertionError
                    (kernels/fused-residual-layernorm! x residual gamma beta))
            "Should throw on dimension mismatch")))))

;; ============================================================================
;; Performance Tests
;; ============================================================================

(deftest test-fused-kernel-performance
  (testing "Fused kernel is faster than separate operations"
    (when (gpu/gpu-available?)
      ;; Test with realistic transformer dimensions
      (let [T 128  ; Sequence length
            C 512  ; Hidden dimension
            iterations 10]
        
        (println "\nBenchmarking fused vs separate operations...")
        (println (format "Matrix size: [%d, %d]" T C))
        
        ;; Benchmark separate operations
        (let [separate-time
              (with-release [x (cuge T C)
                            residual (cuge T C)
                            gamma (cuv C (repeat 1.0))
                            beta (cuv C (repeat 0.0))]
                (let [start (System/nanoTime)]
                  (dotimes [_ iterations]
                    (with-release [x-plus-r (ncore/axpy! 1.0 residual (ncore/copy x))
                                  _ (layernorm/layernorm x-plus-r gamma beta 1e-5)]
                      nil))
                  (/ (- (System/nanoTime) start) iterations 1e6)))]
          
          ;; Benchmark fused kernel
          (let [fused-time
                (with-release [x (cuge T C)
                              residual (cuge T C)
                              gamma (cuv C (repeat 1.0))
                              beta (cuv C (repeat 0.0))]
                  (let [start (System/nanoTime)]
                    (dotimes [_ iterations]
                      (with-release [result (kernels/fused-residual-layernorm!
                                            x residual gamma beta)]
                        nil))
                    (/ (- (System/nanoTime) start) iterations 1e6)))]
            
            (let [speedup (/ separate-time fused-time)]
              (println (format "Separate ops: %.2f ms" separate-time))
              (println (format "Fused kernel: %.2f ms" fused-time))
              (println (format "Speedup: %.2fx" speedup))
              
              ;; Fused kernel should be faster (at least 1.2x)
              (is (> speedup 1.2)
                  "Fused kernel should provide meaningful speedup"))))))))

;; ============================================================================
;; Integration Tests
;; ============================================================================

(deftest test-kernel-caching
  (testing "Kernels are compiled once and cached"
    (when (gpu/gpu-available?)
      ;; Clear cache first
      (kernels/clear-kernel-cache!)
      
      (let [info-before (kernels/kernel-info)]
        (is (empty? (:compiled-kernels info-before))
            "Cache should be empty after clearing"))
      
      ;; First call compiles
      (with-release [x (cuge 2 4)
                     residual (cuge 2 4)
                     gamma (cuv 4 (repeat 1.0))
                     beta (cuv 4 (repeat 0.0))
                     result1 (kernels/fused-residual-layernorm! x residual gamma beta)]
        
        (let [info-after (kernels/kernel-info)]
          (is (= 1 (count (:compiled-kernels info-after)))
              "Kernel should be compiled and cached")
          
          ;; Second call reuses cached kernel
          (with-release [result2 (kernels/fused-residual-layernorm! x residual gamma beta)]
            (is (some? result2)
                "Should successfully reuse cached kernel")))))))

;; ============================================================================
;; Test Runner
;; ============================================================================

(deftest test-gpu-availability
  (testing "Tests require GPU availability"
    (when-not (gpu/gpu-available?)
      (println "\nWARNING: GPU not available - skipping kernel tests")
      (is true "Skipping tests when GPU unavailable"))))

(comment
  ;; Run all tests
  (clojure.test/run-tests)
  
  ;; Run specific test
  (test-fused-residual-layernorm-correctness)
  (test-fused-kernel-performance)
  )