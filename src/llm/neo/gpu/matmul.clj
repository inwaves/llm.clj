(ns llm.neo.gpu.matmul
  "GPU-accelerated matrix multiplication using Neanderthal CUDA.
  
  Provides GPU matmul operations with proper resource management."
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mm! axpy! trans mrows ncols entry!]]
            [uncomplicate.neanderthal.native :as native]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Pure GPU Operations
;; ============================================================================

(defn matmul-forward-gpu
  "Matrix multiplication forward pass on GPU.
  
  All inputs and outputs are GPU matrices (cuge).
  Caller responsible for resource management.
  
  Args:
    inp-gpu: Input matrix on GPU [BT, C]
    weight-gpu: Weight matrix on GPU [OC, C]
    bias-gpu: Bias vector on GPU [OC] or nil
    
  Returns:
    Output matrix on GPU [BT, OC]"
  [inp-gpu weight-gpu bias-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [bt (mrows inp-gpu)
        oc (mrows weight-gpu)
        c (ncols inp-gpu)
        
        _ (when (not= c (ncols weight-gpu))
            (throw (ex-info "Shape mismatch in GPU matmul"
                           {:inp-shape [bt c]
                            :weight-shape [(mrows weight-gpu) (ncols weight-gpu)]})))
        
        out ((resolve 'uncomplicate.neanderthal.cuda/cuge) bt oc)
        _ (mm! 1.0 inp-gpu (trans weight-gpu) 0.0 out)]
    
    (when bias-gpu
      (dotimes [i bt]
        (axpy! 1.0 bias-gpu (ncore/row out i))))
    
    out))

(defn matmul-backward-gpu
  "Matrix multiplication backward pass on GPU.
  
  Caller responsible for resource management.
  
  Args:
    dout-gpu: Gradient w.r.t output [BT, OC]
    inp-gpu: Input from forward pass [BT, C]
    weight-gpu: Weight from forward pass [OC, C]
    
  Returns:
    {:dinp GPU matrix [BT, C]
     :dweight GPU matrix [OC, C]
     :dbias GPU vector [OC]}"
  [dout-gpu inp-gpu weight-gpu]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [bt (mrows inp-gpu)
        c (ncols inp-gpu)
        oc (ncols dout-gpu)
        
        dinp ((resolve 'uncomplicate.neanderthal.cuda/cuge) bt c)
        _ (mm! 1.0 dout-gpu weight-gpu 0.0 dinp)
        
        dweight ((resolve 'uncomplicate.neanderthal.cuda/cuge) oc c)
        _ (mm! 1.0 (trans dout-gpu) inp-gpu 0.0 dweight)
        
        ;; Create zero-initialized bias gradient
        dbias ((resolve 'uncomplicate.neanderthal.cuda/cuv) oc (double-array (repeat oc 0.0)))
        _ (dotimes [i bt]
            (axpy! 1.0 (ncore/row dout-gpu i) dbias))]
    
    {:dinp dinp
     :dweight dweight
     :dbias dbias}))

;; ============================================================================
;; CPU Interface Wrappers
;; ============================================================================

(defn matmul-forward-hybrid
  "Matrix multiplication with CPU inputs/outputs, GPU computation.
  
  Properly manages GPU resources with automatic cleanup.
  
  Args:
    inp: CPU nested vector [BT, C]
    weight: CPU nested vector [OC, C]
    bias: CPU vector [OC] or nil
    
  Returns:
    CPU nested vector [BT, OC]"
  [inp weight bias]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  
  (let [bias-cpu (when bias (native/dv (double-array bias)))]
    (with-release [inp-mat-cpu (neo/vec->matrix inp)
                   weight-mat-cpu (neo/vec->matrix weight)]
      (with-release [inp-gpu (gpu/to-gpu inp-mat-cpu)
                     weight-gpu (gpu/to-gpu weight-mat-cpu)
                     bias-gpu (when bias-cpu (gpu/to-gpu bias-cpu))]
        
        (with-release [result-gpu (matmul-forward-gpu inp-gpu weight-gpu bias-gpu)]
          (with-release [result-cpu (gpu/to-cpu result-gpu)]
            (when bias-cpu (release bias-cpu))
            (neo/matrix->vec result-cpu)))))))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-matmul
  "Benchmark GPU vs CPU matrix multiplication.
  
  Args:
    bt: Batch × Time dimension
    c: Input channels
    oc: Output channels
    iterations: Number of runs
    
  Returns:
    Performance comparison map"
  [bt c oc iterations]
  (require '[uncomplicate.neanderthal.native :as native])
  (require '[uncomplicate.clojurecuda.core :as cu])
  
  (with-release [inp-cpu (native/dge bt c (vec (repeatedly (* bt c) rand)))
                 weight-cpu (native/dge oc c (vec (repeatedly (* oc c) rand)))]
    
    ;; CPU benchmark
    (let [cpu-times (vec (repeatedly iterations
                                    #(let [start (System/nanoTime)]
                                       (with-release [out (native/dge bt oc)]
                                         (mm! 1.0 inp-cpu (trans weight-cpu) 0.0 out))
                                       (/ (- (System/nanoTime) start) 1e6))))
          cpu-mean (/ (reduce + cpu-times) iterations)]
      
      (if (gpu/gpu-available?)
        ;; GPU benchmark
        (let [gpu-times (vec (repeatedly iterations
                                        #(let [start (System/nanoTime)]
                                           (with-release [inp-gpu (gpu/to-gpu inp-cpu)
                                                          weight-gpu (gpu/to-gpu weight-cpu)]
                                             (require '[uncomplicate.neanderthal.cuda :as cuda])
                                             (with-release [out-gpu ((resolve 'uncomplicate.neanderthal.cuda/cuge) bt oc)]
                                               (mm! 1.0 inp-gpu (trans weight-gpu) 0.0 out-gpu)
                                               ((resolve 'uncomplicate.clojurecuda.core/synchronize!))))
                                           (/ (- (System/nanoTime) start) 1e6))))
              gpu-mean (/ (reduce + gpu-times) iterations)]
          {:cpu-time-ms cpu-mean
           :gpu-time-ms gpu-mean
           :speedup (/ cpu-mean gpu-mean)
           :dimensions {:bt bt :c c :oc oc}})
        
        {:cpu-time-ms cpu-mean
         :gpu-time-ms nil
         :speedup nil
         :status "GPU unavailable"
         :dimensions {:bt bt :c c :oc oc}}))))

(comment
  ;; REPL usage
  (require '[llm.neo.gpu.matmul :as gpu-mm])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  
  ;; Test small matmul
  (def inp [[1.0 2.0 3.0] [4.0 5.0 6.0]])
  (def weight [[1.0 2.0 3.0] [4.0 5.0 6.0] [7.0 8.0 9.0] [10.0 11.0 12.0]])
  (def bias [0.1 0.2 0.3 0.4])
  
  (when (gpu/gpu-available?)
    (gpu-mm/matmul-forward-hybrid inp weight bias))
  
  ;; Benchmark
  (gpu-mm/benchmark-matmul 64 256 128 10)
  )