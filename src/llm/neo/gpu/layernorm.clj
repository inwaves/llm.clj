(ns llm.neo.gpu.layernorm
  "GPU-accelerated Layer Normalization using Neanderthal CUDA.
  
  LayerNorm normalizes each row independently:
  output = gamma * (x - mean) / sqrt(var + eps) + beta"
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mrows ncols entry entry! dim]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Helper Functions
;; ============================================================================

(defn- mean-along-cols-gpu
  "Compute mean of each row on GPU matrix.
  Returns device vector of row means."
  [x-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)
        means ((resolve 'uncomplicate.neanderthal.cuda/cuv) rows)]
    (dotimes [i rows]
      (let [row-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (recur (inc j) (+ sum (entry x-gpu i j)))
                        sum))]
        (entry! means i (/ row-sum cols))))
    means))

(defn- variance-along-cols-gpu
  "Compute variance of each row on GPU matrix given means.
  Returns device vector of row variances."
  [x-gpu means-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)
        variances ((resolve 'uncomplicate.neanderthal.cuda/cuv) rows)]
    (dotimes [i rows]
      (let [mu (entry means-gpu i)
            var-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (let [diff (- (entry x-gpu i j) mu)]
                          (recur (inc j) (+ sum (* diff diff))))
                        sum))]
        (entry! variances i (/ var-sum cols))))
    variances))

;; ============================================================================
;; Pure GPU Operations
;; ============================================================================

(defn layernorm-forward-gpu
  "LayerNorm forward pass on GPU.
  
  Normalizes each row to mean=0, var=1, then applies affine transform.
  Caller responsible for resource management.
  
  Args:
    x-gpu: Input matrix on GPU [rows, cols]
    gamma-gpu: Scale parameters on GPU [cols]
    beta-gpu: Shift parameters on GPU [cols]
    epsilon: Small value for numerical stability
    
  Returns:
    Output matrix on GPU [rows, cols]"
  [x-gpu gamma-gpu beta-gpu epsilon]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)]
    
    ;; Compute statistics per row with proper resource management
    (with-release [means (mean-along-cols-gpu x-gpu)]
      (with-release [vars (variance-along-cols-gpu x-gpu means)]
        ;; Create output matrix
        (let [out ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols)]
          
          ;; Normalize and apply affine transform
          (dotimes [i rows]
            (let [mu (entry means i)
                  std (Math/sqrt (+ (entry vars i) epsilon))]
              (dotimes [j cols]
                (let [x-val (entry x-gpu i j)
                      x-norm (/ (- x-val mu) std)
                      y (+ (* (entry gamma-gpu j) x-norm) (entry beta-gpu j))]
                  (entry! out i j y)))))
          
          out)))))

(defn layernorm-backward-gpu
  "LayerNorm backward pass on GPU.
  
  Caller responsible for resource management.
  
  Args:
    x-gpu: Original input [rows, cols]
    dout-gpu: Gradient from upstream [rows, cols]
    gamma-gpu: Scale parameters [cols]
    beta-gpu: Shift parameters [cols] (unused but kept for API)
    epsilon: Numerical stability constant
    
  Returns:
    {:dx-gpu [rows, cols] :dgamma-gpu [cols] :dbeta-gpu [cols]}"
  [x-gpu dout-gpu gamma-gpu beta-gpu epsilon]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)]
    
    ;; Recompute forward statistics with proper resource management
    (with-release [means (mean-along-cols-gpu x-gpu)]
      (with-release [vars (variance-along-cols-gpu x-gpu means)]
        ;; Initialize gradient accumulators
        (let [dx ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols)
              dgamma ((resolve 'uncomplicate.neanderthal.cuda/cuv) cols)
              dbeta ((resolve 'uncomplicate.neanderthal.cuda/cuv) cols)]
          
          ;; Zero-initialize accumulators
          (dotimes [j cols]
            (entry! dgamma j 0.0)
            (entry! dbeta j 0.0))
          
          ;; Process each row
          (dotimes [i rows]
            (let [mu (entry means i)
                  var (entry vars i)
                  std (Math/sqrt (+ var epsilon))
                  inv-std (/ 1.0 std)]
              
              ;; Accumulate dgamma and dbeta
              (dotimes [j cols]
                (let [x-norm (* (- (entry x-gpu i j) mu) inv-std)
                      dout-val (entry dout-gpu i j)]
                  (entry! dgamma j (+ (entry dgamma j) (* dout-val x-norm)))
                  (entry! dbeta j (+ (entry dbeta j) dout-val))))
              
              ;; Compute dx for this row
              (let [sum-dout-gamma (loop [j 0 sum 0.0]
                                     (if (< j cols)
                                       (recur (inc j) (+ sum (* (entry dout-gpu i j) (entry gamma-gpu j))))
                                       sum))
                    sum-dout-gamma-xnorm (loop [j 0 sum 0.0]
                                           (if (< j cols)
                                             (let [x-norm (* (- (entry x-gpu i j) mu) inv-std)]
                                               (recur (inc j) (+ sum (* (entry dout-gpu i j) (entry gamma-gpu j) x-norm))))
                                             sum))]
                
                (dotimes [j cols]
                  (let [x-norm (* (- (entry x-gpu i j) mu) inv-std)
                        dout-gamma (* (entry dout-gpu i j) (entry gamma-gpu j))
                        dx-val (* inv-std
                                 (- dout-gamma
                                    (/ sum-dout-gamma cols)
                                    (* x-norm (/ sum-dout-gamma-xnorm cols))))]
                    (entry! dx i j dx-val))))))
          
          {:dx-gpu dx :dgamma-gpu dgamma :dbeta-gpu dbeta})))))

;; ============================================================================
;; CPU Interface Wrappers
;; ============================================================================

(defn layernorm-forward-hybrid
  "LayerNorm with CPU inputs/outputs, GPU computation.
  
  Args:
    x: CPU nested vector [rows, cols]
    gamma: CPU vector [cols]
    beta: CPU vector [cols]
    epsilon: Small value for stability
    
  Returns:
    CPU nested vector [rows, cols]"
  [x gamma beta epsilon]
  (ncore/with-default-engine (gpu/cuda-engine)
    (require '[uncomplicate.neanderthal.cuda :as cuda])
    (require '[uncomplicate.neanderthal.native :as native])
    
    (with-release [x-mat-cpu (neo/vec->matrix x)
                   gamma-cpu (native/dv (double-array gamma))
                   beta-cpu (native/dv (double-array beta))]
      (with-release [x-gpu (gpu/to-gpu x-mat-cpu)
                     gamma-gpu (gpu/to-gpu gamma-cpu)
                     beta-gpu (gpu/to-gpu beta-cpu)]
        (with-release [result-gpu (layernorm-forward-gpu x-gpu gamma-gpu beta-gpu epsilon)]
          (with-release [result-cpu (gpu/to-cpu result-gpu)]
            (neo/matrix->vec result-cpu)))))))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-layernorm
  "Benchmark GPU vs CPU layer normalization.
  
  Args:
    rows: Number of samples
    cols: Feature dimension
    iterations: Number of runs
    
  Returns:
    Performance comparison map"
  [rows cols iterations]
  (require '[uncomplicate.neanderthal.native :as native])
  (require '[llm.neo.layernorm :as cpu-ln])
  (require '[uncomplicate.clojurecuda.core :as cu])
  
  (with-release [x-cpu (native/dge rows cols (vec (repeatedly (* rows cols) #(- (rand) 0.5))))
                 gamma-cpu (native/dv (vec (repeatedly cols #(+ 0.8 (* 0.4 (rand))))))
                 beta-cpu (native/dv (vec (repeatedly cols #(* 0.2 (- (rand) 0.5)))))]
    
    ;; CPU benchmark
    (let [cpu-times (vec (repeatedly iterations
                                    #(let [start (System/nanoTime)]
                                       (ncore/with-default-engine (gpu/cpu-engine)
                                         (with-release [out (cpu-ln/layernorm-forward x-cpu gamma-cpu beta-cpu 1e-5)]
                                           out))
                                       (/ (- (System/nanoTime) start) 1e6))))
          cpu-mean (/ (reduce + cpu-times) iterations)]
      
      (if (gpu/gpu-available?)
        ;; GPU benchmark
        (let [gpu-times (vec (repeatedly iterations
                                        #(let [start (System/nanoTime)]
                                           (ncore/with-default-engine (gpu/cuda-engine)
                                             (with-release [x-gpu (gpu/to-gpu x-cpu)
                                                            gamma-gpu (gpu/to-gpu gamma-cpu)
                                                            beta-gpu (gpu/to-gpu beta-cpu)]
                                               (with-release [out-gpu (layernorm-forward-gpu x-gpu gamma-gpu beta-gpu 1e-5)]
                                                 ((resolve 'uncomplicate.clojurecuda.core/synchronize!)))))
                                           (/ (- (System/nanoTime) start) 1e6))))
              gpu-mean (/ (reduce + gpu-times) iterations)]
          {:cpu-time-ms cpu-mean
           :gpu-time-ms gpu-mean
           :speedup (/ cpu-mean gpu-mean)
           :dimensions {:rows rows :cols cols}
           :note "LayerNorm involves reductions which may show limited GPU benefit"})
        
        {:cpu-time-ms cpu-mean
         :gpu-time-ms nil
         :speedup nil
         :status "GPU unavailable"
         :dimensions {:rows rows :cols cols}}))))

(comment
  ;; REPL usage
  (require '[llm.neo.gpu.layernorm :as gpu-ln])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  
  ;; Test small layernorm
  (def x [[1.0 2.0 3.0] [4.0 5.0 6.0]])
  (def gamma [1.0 1.0 1.0])
  (def beta [0.0 0.0 0.0])
  
  (when (gpu/gpu-available?)
    (gpu-ln/layernorm-forward-hybrid x gamma beta 1e-5))
  
  ;; Benchmark
  (gpu-ln/benchmark-layernorm 128 256 10)
  )