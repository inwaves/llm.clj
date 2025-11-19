(ns llm.neo.gpu.matmul-test
  "Tests for GPU matrix multiplication."
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.gpu.matmul :as gpu-mm]
            [llm.neo.matmul :as cpu-mm]
            [llm.neo.core :as neo]))

(deftest gpu-availability-test
  (testing "GPU detection works without errors"
    (let [available (gpu/gpu-available?)
          info (gpu/initialize-gpu)]
      (is (boolean? available))
      (is (map? info))
      (is (contains? info :gpu-available))
      (is (contains? info :recommendation)))))

(deftest hybrid-matmul-correctness-test
  (testing "Hybrid matmul correctness against CPU"
    (let [inp [[1.0 2.0 3.0] [4.0 5.0 6.0]]
          weight [[0.0 1.0 2.0] [1.0 2.0 3.0] [2.0 3.0 4.0] [3.0 4.0 5.0]]
          bias [0.0 1.0 2.0 3.0]
          cpu-result (cpu-mm/matmul-forward-from-vecs inp weight bias)]
      
      (if (gpu/gpu-available?)
        (let [hybrid-result (gpu-mm/matmul-forward-hybrid inp weight bias)]
          (is (neo/allclose cpu-result hybrid-result 1e-4)
              "GPU result should match CPU within tolerance"))
        (is true "Skipped - no GPU available")))))

(deftest matmul-benchmark-test
  (testing "GPU matmul benchmark runs and returns valid results"
    (let [result (gpu-mm/benchmark-matmul 64 256 128 3)]
      ;; Validate return structure
      (is (number? (:cpu-time-ms result)))
      (is (map? (:dimensions result)))
      
      (when (gpu/gpu-available?)
        (is (number? (:gpu-time-ms result)))
        (is (number? (:speedup result)))
        ;; Just verify both timings are positive, don't assert speed superiority
        (is (pos? (:cpu-time-ms result)))
        (is (pos? (:gpu-time-ms result)))))))

(comment
  ;; Run tests
  (gpu-availability-test)
  (hybrid-matmul-correctness-test)
  (matmul-benchmark-test)
  )