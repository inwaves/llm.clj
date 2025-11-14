(ns llm.neo.gpu.attention-test
  "Tests for GPU-accelerated multi-head attention."
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.gpu.attention :as gpu-attn]
            [llm.neo.attention :as cpu-attn]
            [llm.neo.core :as neo]
            [uncomplicate.commons.core :refer [release]]))

(deftest gpu-availability-test
  (testing "GPU detection works without errors"
    (let [available (gpu/gpu-available?)
          info (gpu/initialize-gpu)]
      (is (boolean? available))
      (is (map? info))
      (is (contains? info :gpu-available))
      (is (contains? info :recommendation)))))

(deftest attention-forward-qkv-correctness-test
  (testing "GPU attention forward matches CPU within tolerance"
    ;; Deterministic input (fixed values, not random)
    (let [t 4
          c 8
          nh 2
          ;; Fixed QKV values for deterministic test
          qkv-single [(vec (range 0.0 24.0))      ; pos 0
                      (vec (range 24.0 48.0))     ; pos 1  
                      (vec (range 48.0 72.0))     ; pos 2
                      (vec (range 72.0 96.0))]    ; pos 3
          qkv-batch [qkv-single]
          
          ;; CPU result
          cpu-result (cpu-attn/attention-forward qkv-batch nh)]
      
      (if (gpu/gpu-available?)
        (let [gpu-res (gpu-attn/attention-forward-qkv-hybrid qkv-single nh)
              gpu-result (:out gpu-res)
              cache (:cache gpu-res)]
          (try
            (is (neo/allclose (first cpu-result) gpu-result 1e-3)
                "GPU result should match CPU within tolerance")
            (finally
              ;; Clean up cache resources even if assertion fails
              (when-let [q (:q cache)] (release q))
              (when-let [k (:k cache)] (release k))
              (when-let [v (:v cache)] (release v))
              (doseq [p (:att-probs cache)] (release p)))))
        
        (is true "Skipped - no GPU available")))))

(deftest attention-backward-qkv-gradient-test
  (testing "GPU attention backward produces valid gradients with correct shape"
    (if (gpu/gpu-available?)
      (let [t 4
            c 8  
            nh 2
            ;; Fixed inputs (deterministic)
            qkv-input (vec (repeatedly t #(vec (concat (range 0.0 8.0)
                                                       (range 8.0 16.0)
                                                       (range 16.0 24.0)))))
            dout (vec (repeatedly t #(vec (repeat c 1.0))))]  ; uniform gradient
        
        ;; Forward to get cache
        (let [fwd-res (gpu-attn/attention-forward-qkv-hybrid qkv-input nh)
              cache (:cache fwd-res)]
          (try
            ;; Backward (note: backward-hybrid releases cache internally, but we use try/finally for robustness)
            (let [dqkv (gpu-attn/attention-backward-qkv-hybrid dout cache nh)]
              
              ;; Verify shape
              (is (= t (count dqkv)) "Gradient should have T rows")
              (is (= (* 3 c) (count (first dqkv))) "Gradient should have 3C columns")
              
              ;; Verify non-zero gradients (attention should propagate signal)
              (let [sum-abs-grad (reduce + (map #(Math/abs %) (flatten dqkv)))]
                (is (> sum-abs-grad 0.0) "Gradients should be non-zero")))
            
            (finally
              ;; Ensure cache cleanup even if backward or assertions fail
              ;; Note: backward-hybrid also releases, so this is defensive
              (try
                (when-let [q (:q cache)] (release q))
                (when-let [k (:k cache)] (release k))
                (when-let [v (:v cache)] (release v))
                (doseq [p (:att-probs cache)] (release p))
                (catch Exception _))))))  ; Ignore double-release errors
      
      (is true "Skipped - no GPU available"))))

(deftest attention-performance-benchmark-test
  (testing "GPU attention performance measurement"
    (if (gpu/gpu-available?)
      (let [;; Medium-sized problem
            t 64
            c 128
            nh 8
            iterations 3
            
            ;; Generate deterministic test data
            qkv-input (vec (repeatedly t #(vec (take (* 3 c) (cycle (range 100))))))
            
            ;; Time CPU (with warmup)
            _ (cpu-attn/attention-forward [qkv-input] nh)  ; warmup
            cpu-times (vec (repeatedly iterations
                                      #(let [start (System/nanoTime)]
                                         (cpu-attn/attention-forward [qkv-input] nh)
                                         (/ (- (System/nanoTime) start) 1e6))))
            cpu-mean (/ (reduce + cpu-times) iterations)
            
            ;; Time GPU (with warmup)
            _ (let [res (gpu-attn/attention-forward-qkv-hybrid qkv-input nh)
                    cache (:cache res)]
                (when-let [q (:q cache)] (release q))
                (when-let [k (:k cache)] (release k))
                (when-let [v (:v cache)] (release v))
                (doseq [p (:att-probs cache)] (release p)))
            
            gpu-times (vec (repeatedly iterations
                                      #(let [start (System/nanoTime)
                                             res (gpu-attn/attention-forward-qkv-hybrid qkv-input nh)
                                             elapsed (/ (- (System/nanoTime) start) 1e6)
                                             cache (:cache res)]
                                         (try
                                           elapsed
                                           (finally
                                             ;; Clean up even if timing measurement has issues
                                             (when-let [q (:q cache)] (release q))
                                             (when-let [k (:k cache)] (release k))
                                             (when-let [v (:v cache)] (release v))
                                             (doseq [p (:att-probs cache)] (release p)))))))
            gpu-mean (/ (reduce + gpu-times) iterations)
            speedup (/ cpu-mean gpu-mean)]
        
        ;; Verify both run and produce positive times
        (is (pos? cpu-mean) "CPU time should be positive")
        (is (pos? gpu-mean) "GPU time should be positive")
        (is (number? speedup) "Speedup should be calculable")
        
        (println (format "\nAttention Performance (T=%d, C=%d, NH=%d):" t c nh))
        (println (format "  CPU: %.2f ms (n=%d)" cpu-mean iterations))
        (println (format "  GPU: %.2f ms (n=%d)" gpu-mean iterations))
        (println (format "  Speedup: %.1fx" speedup))
        (println (format "  Note: Attention involves multiple matmuls (major GPU benefit)")))
      
      (is true "Skipped - no GPU available"))))

(comment
  (require '[clojure.test :as t])
  (t/run-tests 'llm.neo.gpu.attention-test)
  
  ;; Run individual tests
  (gpu-availability-test)
  (attention-forward-qkv-correctness-test)
  (attention-backward-qkv-gradient-test)
  (attention-performance-benchmark-test)
  )