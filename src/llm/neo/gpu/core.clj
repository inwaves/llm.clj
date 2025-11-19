(ns llm.neo.gpu.core
  "GPU acceleration utilities for Neanderthal CUDA backend.
  
  Provides GPU detection, engine management, CPU↔GPU transfers, and benchmarking.
  Falls back to CPU when GPU is unavailable.
  
  Based on Neanderthal 0.57.0 and ClojureCUDA 0.16.0 APIs."
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore]
            [uncomplicate.neanderthal.native :refer [dge dv]]))

;; ============================================================================
;; GPU Detection
;; ============================================================================

(defn gpu-available?
  "Check if CUDA GPU is available by attempting to query device count."
  []
  (try
    (require '[uncomplicate.clojurecuda.core :as cu])
    ;; Attempt initialization and device count
    ((resolve 'uncomplicate.clojurecuda.core/init))
    (> ((resolve 'uncomplicate.clojurecuda.core/device-count)) 0)
    (catch Exception _ false)))

(defn gpu-info
  "Get information about available GPUs."
  []
  (when (gpu-available?)
    (try
      (require '[uncomplicate.clojurecuda.core :as cu])
      (let [device-count ((resolve 'uncomplicate.clojurecuda.core/device-count))]
        {:device-count device-count
         :status "CUDA initialized successfully"})
      (catch Exception e
        {:error (.getMessage e)}))))

;; ============================================================================
;; Engine Management
;; ============================================================================

(defn cuda-factory
  "Return CUDA factory instance for creating GPU matrices/vectors.
  
  Factory instances are used to create matrices and vectors on GPU.
  Based on Neanderthal 0.57.0 factory pattern."
  []
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  ((resolve 'uncomplicate.neanderthal.cuda/cuda-float)))

(defn native-factory
  "Return native factory instance for creating CPU matrices/vectors.
  
  Factory instances are used to create matrices and vectors on CPU.
  Based on Neanderthal 0.57.0 factory pattern."
  []
  (require '[uncomplicate.neanderthal.native :as native])
  ((resolve 'uncomplicate.neanderthal.native/native-float)))

;; ============================================================================
;; Memory Transfer
;; ============================================================================

(defn to-gpu
  "Transfer CPU matrix/vector to GPU.
  
  Uses CUDA constructors which accept CPU data and create GPU copies.
  
  Args:
    cpu-data: Neanderthal CPU matrix or vector
    
  Returns:
    GPU matrix or vector with same data
    
  Example:
    (def cpu-mat (dge 3 3 (range 9)))
    (def gpu-mat (to-gpu cpu-mat))  ; Data now on GPU"
  [cpu-data]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  (let [cuge (resolve 'uncomplicate.neanderthal.cuda/cuge)
        cuv (resolve 'uncomplicate.neanderthal.cuda/cuv)]
    ;; CUDA constructors accept source matrices/vectors and copy to GPU
    (if (try (ncore/mrows cpu-data) true (catch Exception _ false))
      ;; Matrix - use cuge
      (cuge cpu-data)
      ;; Vector - use cuv
      (cuv cpu-data))))

(defn to-cpu
  "Transfer GPU matrix/vector to CPU.
  
  Uses native constructors which accept GPU data and create CPU copies.
  
  Args:
    gpu-data: Neanderthal GPU matrix or vector
    
  Returns:
    CPU matrix or vector with same data
    
  Example:
    (def cpu-mat (to-cpu gpu-mat))  ; Data now on CPU"
  [gpu-data]
  ;; Native constructors accept source matrices/vectors and copy to CPU
  (if (try (ncore/mrows gpu-data) true (catch Exception _ false))
    ;; Matrix - use dge (already imported from native)
    (dge gpu-data)
    ;; Vector - use dv (already imported from native)
    (dv gpu-data)))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-gpu-vs-cpu
  "Compare operation performance on GPU vs CPU.
  
  Args:
    cpu-operation-fn: Function that creates and operates on CPU matrices
                      Should be self-contained (create data, compute, return)
    gpu-operation-fn: Function that creates and operates on GPU matrices
                      Should be self-contained (create data, compute, return)
    iterations: Number of runs for averaging
    
  Returns:
    Map with :cpu-time-ms, :gpu-time-ms, and :speedup
    
  Example:
    (benchmark-gpu-vs-cpu
      ;; CPU operation
      (fn [] 
        (let [a (dge 1000 1000)
              b (dge 1000 1000)]
          (mm! a b a)))
      ;; GPU operation  
      (fn []
        (require '[uncomplicate.neanderthal.cuda :refer [cuge]])
        (with-release [a (cuge 1000 1000)
                      b (cuge 1000 1000)]
          (mm! a b a)))
      10)"
  [cpu-operation-fn gpu-operation-fn iterations]
  (let [;; CPU timing
        cpu-times (repeatedly iterations
                             #(let [start (System/nanoTime)]
                                (cpu-operation-fn)
                                (/ (- (System/nanoTime) start) 1e6)))
        cpu-mean (/ (reduce + cpu-times) iterations)]
    
    (if (gpu-available?)
      (let [;; GPU timing with synchronization
            gpu-times (repeatedly iterations
                                 #(let [start (System/nanoTime)
                                        result (gpu-operation-fn)]
                                    ;; Synchronize GPU before stopping timer
                                    (require '[uncomplicate.clojurecuda.core :as cu])
                                    ((resolve 'uncomplicate.clojurecuda.core/synchronize!))
                                    (/ (- (System/nanoTime) start) 1e6)))
            gpu-mean (/ (reduce + gpu-times) iterations)]
        {:cpu-time-ms cpu-mean
         :gpu-time-ms gpu-mean
         :speedup (/ cpu-mean gpu-mean)})
      {:cpu-time-ms cpu-mean
       :gpu-time-ms nil
       :speedup nil
       :status "GPU unavailable"})))

;; ============================================================================
;; Initialization
;; ============================================================================

(defn initialize-gpu
  "Initialize GPU context and verify functionality.
  
  Call once at application startup to check GPU availability.
  
  Returns:
    Map with :gpu-available, :gpu-info, and :recommendation"
  []
  (let [available (gpu-available?)
        info (when available (gpu-info))]
    {:gpu-available available
     :gpu-info info
     :recommendation (if available
                       "Use GPU factories (cuda-float) for acceleration"
                       "GPU unavailable - use CPU factories (native-float)")}))

(comment
  ;; REPL exploration
  (require '[uncomplicate.neanderthal.core :refer [mm!]])
  (require '[uncomplicate.neanderthal.native :refer [dge]])
  (require '[uncomplicate.commons.core :refer [with-release]])
  
  ;; Check GPU availability
  (gpu-available?)
  (gpu-info)
  (initialize-gpu)
  
  ;; Create CPU matrix
  (def cpu-mat (dge 3 3 (range 9)))
  
  ;; Transfer to GPU (if available)
  (when (gpu-available?)
    (def gpu-mat (to-gpu cpu-mat))
    ;; Transfer back
    (def cpu-mat-2 (to-cpu gpu-mat)))
  
  ;; Benchmark example
  (when (gpu-available?)
    (benchmark-gpu-vs-cpu
      ;; CPU operation
      (fn [] 
        (let [a (dge 1000 1000)
              b (dge 1000 1000)]
          (mm! a b a)))
      ;; GPU operation  
      (fn []
        (require '[uncomplicate.neanderthal.cuda :refer [cuge]])
        (with-release [a (cuge 1000 1000)
                      b (cuge 1000 1000)]
          (mm! a b a)))
      5))
  )