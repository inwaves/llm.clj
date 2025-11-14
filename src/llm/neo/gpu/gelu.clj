(ns llm.neo.gpu.gelu
  "GPU-accelerated GELU activation using Neanderthal CUDA.
  
  GELU formula: x * 0.5 * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))"
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mrows ncols entry entry!]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.core :as neo]))

(def gelu-scaling-factor (Math/sqrt (/ 2.0 Math/PI)))

;; ============================================================================
;; Element-wise Operations
;; ============================================================================

(defn- gelu-element
  "Apply GELU to a single element."
  [x]
  (let [cube (* 0.044715 x x x)
        tanh-arg (* gelu-scaling-factor (+ x cube))
        tanh-out (Math/tanh tanh-arg)]
    (* 0.5 x (+ 1.0 tanh-out))))

(defn- gelu-gradient-element
  "Compute GELU gradient for a single element."
  [x]
  (let [cube (* 0.044715 x x x)
        tanh-arg (* gelu-scaling-factor (+ x cube))
        tanh-out (Math/tanh tanh-arg)
        cosh-out (Math/cosh tanh-arg)
        sech-out (/ 1.0 (* cosh-out cosh-out))
        derivative-factor (+ 1.0 (* 3.0 0.044715 x x))]
    (+ (* 0.5 (+ 1.0 tanh-out))
       (* x 0.5 sech-out gelu-scaling-factor derivative-factor))))

;; ============================================================================
;; Pure GPU Operations
;; ============================================================================

(defn gelu-forward-gpu
  "GELU forward pass on GPU.
  
  Applies element-wise GELU to GPU matrix.
  Caller responsible for resource management.
  
  Args:
    x-gpu: Input matrix on GPU [rows, cols]
    
  Returns:
    Output matrix on GPU [rows, cols] with GELU applied"
  [x-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)
        out ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols)]
    
    ;; Element-wise GELU application
    ;; Note: This is not optimized - a custom CUDA kernel would be much faster
    ;; For production, consider implementing a fused GELU kernel
    (dotimes [i rows]
      (dotimes [j cols]
        (entry! out i j (gelu-element (entry x-gpu i j)))))
    
    out))

(defn gelu-backward-gpu
  "GELU backward pass on GPU.
  
  Caller responsible for resource management.
  
  Args:
    x-gpu: Original input from forward pass [rows, cols]
    dout-gpu: Gradient from upstream [rows, cols]
    
  Returns:
    dx-gpu: Gradient w.r.t input [rows, cols]"
  [x-gpu dout-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [rows (mrows x-gpu)
        cols (ncols x-gpu)
        dx ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols)]
    
    ;; Element-wise gradient computation
    (dotimes [i rows]
      (dotimes [j cols]
        (let [grad (gelu-gradient-element (entry x-gpu i j))
              upstream (entry dout-gpu i j)]
          (entry! dx i j (* grad upstream)))))
    
    dx))

;; ============================================================================
;; CPU Interface Wrappers
;; ============================================================================

(defn gelu-forward-hybrid
  "GELU with CPU inputs/outputs, GPU computation.
  
  Args:
    x: CPU nested vector [rows, cols]
    
  Returns:
    CPU nested vector [rows, cols]"
  [x]
  (ncore/with-default-engine (gpu/cuda-engine)
    (require '[uncomplicate.neanderthal.cuda :as cuda])
    
    (with-release [x-mat-cpu (neo/vec->matrix x)]
      (with-release [x-gpu (gpu/to-gpu x-mat-cpu)]
        (with-release [result-gpu (gelu-forward-gpu x-gpu)]
          (with-release [result-cpu (gpu/to-cpu result-gpu)]
            (neo/matrix->vec result-cpu)))))))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-gelu
  "Benchmark GPU vs CPU GELU activation.
  
  Args:
    rows: Matrix rows
    cols: Matrix columns
    iterations: Number of runs
    
  Returns:
    Performance comparison map"
  [rows cols iterations]
  (require '[uncomplicate.neanderthal.native :as native])
  (require '[llm.neo.gelu :as cpu-gelu])
  (require '[uncomplicate.clojurecuda.core :as cu])
  
  (with-release [x-cpu (native/dge rows cols (vec (repeatedly (* rows cols) #(- (rand) 0.5))))]
    
    ;; CPU benchmark
    (let [cpu-times (vec (repeatedly iterations
                                    #(let [start (System/nanoTime)]
                                       (ncore/with-default-engine (gpu/cpu-engine)
                                         (with-release [out (cpu-gelu/gelu-forward x-cpu)]
                                           out))
                                       (/ (- (System/nanoTime) start) 1e6))))
          cpu-mean (/ (reduce + cpu-times) iterations)]
      
      (if (gpu/gpu-available?)
        ;; GPU benchmark
        (let [gpu-times (vec (repeatedly iterations
                                        #(let [start (System/nanoTime)]
                                           (ncore/with-default-engine (gpu/cuda-engine)
                                             (with-release [x-gpu (gpu/to-gpu x-cpu)]
                                               (with-release [out-gpu (gelu-forward-gpu x-gpu)]
                                                 ((resolve 'uncomplicate.clojurecuda.core/synchronize!)))))
                                           (/ (- (System/nanoTime) start) 1e6))))
              gpu-mean (/ (reduce + gpu-times) iterations)]
          {:cpu-time-ms cpu-mean
           :gpu-time-ms gpu-mean
           :speedup (/ cpu-mean gpu-mean)
           :dimensions {:rows rows :cols cols}
           :note "Element-wise operations may show limited speedup without custom kernels"})
        
        {:cpu-time-ms cpu-mean
         :gpu-time-ms nil
         :speedup nil
         :status "GPU unavailable"
         :dimensions {:rows rows :cols cols}}))))

(comment
  ;; REPL usage
  (require '[llm.neo.gpu.gelu :as gpu-gelu])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  
  ;; Test small GELU
  (def x [[1.0 -0.5 0.0] [2.0 -1.0 0.5]])
  
  (when (gpu/gpu-available?)
    (gpu-gelu/gelu-forward-hybrid x))
  
  ;; Benchmark
  (gpu-gelu/benchmark-gelu 128 256 10)
  )