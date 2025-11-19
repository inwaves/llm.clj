(ns llm.neo.gpu.residual
  "GPU-accelerated residual connections using Neanderthal CUDA."
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore :refer [axpy! copy!]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Pure GPU Operations
;; ============================================================================

(defn residual-forward-gpu
  "Residual connection forward pass on GPU.
  
  Computes: out = x1 + x2
  
  Caller responsible for resource management.
  
  Args:
    x1-gpu: First input matrix on GPU
    x2-gpu: Second input matrix on GPU (same shape as x1)
    
  Returns:
    Output matrix on GPU with element-wise sum"
  [x1-gpu x2-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [out (ncore/copy x1-gpu)]
    ;; axpy! computes: out = alpha*x2 + out
    ;; Since out = x1, this gives: out = x2 + x1
    (axpy! 1.0 x2-gpu out)
    out))

(defn residual-forward-gpu-inplace
  "Residual connection in-place on GPU (mutates x1).
  
  More memory efficient when x1 can be modified.
  
  Args:
    x1-gpu: Matrix to mutate (will contain result)
    x2-gpu: Matrix to add
    
  Returns:
    x1-gpu (modified)"
  [x1-gpu x2-gpu]
  (axpy! 1.0 x2-gpu x1-gpu))

(defn residual-backward-gpu
  "Residual connection backward pass on GPU.
  
  Forward: y = x1 + x2
  Backward: Both inputs receive the same gradient
  
  Args:
    dout-gpu: Gradient w.r.t output
    
  Returns:
    {:dx1-gpu dout-gpu :dx2-gpu dout-gpu}
  
  Note: Returns the same matrix reference for both gradients.
  Caller must copy if independent modifications needed."
  [dout-gpu]
  {:dx1-gpu dout-gpu
   :dx2-gpu dout-gpu})

;; ============================================================================
;; CPU Interface Wrappers
;; ============================================================================

(defn residual-forward-hybrid
  "Residual connection with CPU inputs/outputs, GPU computation.
  
  Args:
    x1: CPU nested vector
    x2: CPU nested vector (same shape as x1)
    
  Returns:
    CPU nested vector (x1 + x2)"
  [x1 x2]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  
  (with-release [x1-mat-cpu (neo/vec->matrix x1)
                 x2-mat-cpu (neo/vec->matrix x2)]
    (with-release [x1-gpu (gpu/to-gpu x1-mat-cpu)
                   x2-gpu (gpu/to-gpu x2-mat-cpu)]
      (with-release [result-gpu (residual-forward-gpu x1-gpu x2-gpu)]
        (with-release [result-cpu (gpu/to-cpu result-gpu)]
          (neo/matrix->vec result-cpu))))))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-residual
  "Benchmark GPU vs CPU residual connections.
  
  Args:
    rows: Matrix rows
    cols: Matrix columns
    iterations: Number of runs
    
  Returns:
    Performance comparison map"
  [rows cols iterations]
  (require '[uncomplicate.neanderthal.native :as native])
  (require '[llm.neo.residual :as cpu-residual])
  (require '[uncomplicate.clojurecuda.core :as cu])
  
  (with-release [x1-cpu (native/dge rows cols (vec (repeatedly (* rows cols) rand)))
                 x2-cpu (native/dge rows cols (vec (repeatedly (* rows cols) rand)))]
    
    ;; CPU benchmark
    (let [cpu-times (vec (repeatedly iterations
                                    #(let [start (System/nanoTime)]
                                       (with-release [out (cpu-residual/residual-forward x1-cpu x2-cpu)]
                                         out)
                                       (/ (- (System/nanoTime) start) 1e6))))
          cpu-mean (/ (reduce + cpu-times) iterations)]
      
      (if (gpu/gpu-available?)
        ;; GPU benchmark
        (let [gpu-times (vec (repeatedly iterations
                                        #(let [start (System/nanoTime)]
                                           (with-release [x1-gpu (gpu/to-gpu x1-cpu)
                                                          x2-gpu (gpu/to-gpu x2-cpu)]
                                             (with-release [out-gpu (residual-forward-gpu x1-gpu x2-gpu)]
                                               ((resolve 'uncomplicate.clojurecuda.core/synchronize!))))
                                           (/ (- (System/nanoTime) start) 1e6))))
              gpu-mean (/ (reduce + gpu-times) iterations)]
          {:cpu-time-ms cpu-mean
           :gpu-time-ms gpu-mean
           :speedup (/ cpu-mean gpu-mean)
           :dimensions {:rows rows :cols cols}
           :note "Simple element-wise operations have transfer overhead"})
        
        {:cpu-time-ms cpu-mean
         :gpu-time-ms nil
         :speedup nil
         :status "GPU unavailable"
         :dimensions {:rows rows :cols cols}}))))

(comment
  ;; REPL usage
  (require '[llm.neo.gpu.residual :as gpu-residual])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  
  ;; Test small residual
  (def x1 [[1.0 2.0] [3.0 4.0]])
  (def x2 [[0.1 0.2] [0.3 0.4]])
  
  (when (gpu/gpu-available?)
    (gpu-residual/residual-forward-hybrid x1 x2))
  
  ;; Benchmark
  (gpu-residual/benchmark-residual 128 256 10)
  )