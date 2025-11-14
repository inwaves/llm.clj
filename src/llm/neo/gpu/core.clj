(ns llm.neo.gpu.core
  "GPU acceleration utilities for Neanderthal CUDA backend.
  
  Provides GPU detection, engine management, CPU↔GPU transfers, and benchmarking.
  Falls back to CPU when GPU is unavailable.
  
  Based on Neanderthal 0.57.0 and ClojureCUDA 0.16.0 APIs."
  (:require [uncomplicate.commons.core :refer [with-release]]
            [uncomplicate.neanderthal.core :as ncore]))

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

(defn cuda-engine
  "Create and return CUDA engine Factory instance."
  []
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  ((resolve 'uncomplicate.neanderthal.cuda/cuda-float)))

(defn cpu-engine
  "Create and return CPU engine Factory instance."
  []
  (require '[uncomplicate.neanderthal.native :as native])
  ((resolve 'uncomplicate.neanderthal.native/native-float)))

(defn select-engine
  "Select Neanderthal engine based on GPU availability and preference.
  
  Returns Factory instance (not function)."
  [prefer-gpu]
  (if (and prefer-gpu (gpu-available?))
    (cuda-engine)
    (cpu-engine)))

(defmacro with-engine
  "Execute body with specified Neanderthal engine.
  
  Engine type is :gpu or :cpu.
  
  Example:
    (with-engine :gpu
      (require '[uncomplicate.neanderthal.cuda :refer [cuge]])
      (with-release [m (cuge 1000 1000)]
        (mm! m m m)))"
  [engine-type & body]
  `(ncore/with-default-engine (select-engine (= ~engine-type :gpu))
     ~@body))

;; ============================================================================
;; Memory Transfer
;; ============================================================================

(defn to-gpu
  "Transfer data to GPU using ncore/copy under CUDA engine."
  [cpu-data]
  (ncore/with-default-engine (cuda-engine)
    (ncore/copy cpu-data)))

(defn to-cpu
  "Transfer data to CPU using ncore/copy under CPU engine."
  [gpu-data]
  (ncore/with-default-engine (cpu-engine)
    (ncore/copy gpu-data)))

;; ============================================================================
;; Benchmarking
;; ============================================================================

(defn benchmark-gpu-vs-cpu
  "Compare operation performance on GPU vs CPU.
  
  Args:
    operation-fn: Function using active engine
    iterations: Number of runs for averaging
    
  Returns:
    {:cpu-time-ms :gpu-time-ms :speedup}"
  [operation-fn iterations]
  (let [;; CPU timing
        cpu-times (repeatedly iterations
                             #(let [start (System/nanoTime)]
                                (ncore/with-default-engine (cpu-engine)
                                  (operation-fn))
                                (/ (- (System/nanoTime) start) 1e6)))
        cpu-mean (/ (reduce + cpu-times) iterations)]
    
    (if (gpu-available?)
      (let [;; GPU timing with synchronization
            gpu-times (repeatedly iterations
                                 #(let [start (System/nanoTime)]
                                    (ncore/with-default-engine (cuda-engine)
                                      (operation-fn)
                                      ;; Synchronize GPU before stopping timer
                                      (require '[uncomplicate.clojurecuda.core :as cu])
                                      ((resolve 'uncomplicate.clojurecuda.core/synchronize!)))
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
  
  Call once at application startup."
  []
  (let [available (gpu-available?)
        info (when available (gpu-info))]
    {:gpu-available available
     :gpu-info info
     :recommendation (if available
                       "Use :gpu engine for GPU acceleration"
                       "GPU unavailable - use :cpu engine")}))

(comment
  ;; REPL exploration
  (gpu-available?)
  (gpu-info)
  (initialize-gpu)
  
  ;; Create matrices on different engines
  (with-engine :cpu
    (require '[uncomplicate.neanderthal.native :refer [dge]])
    (dge 3 3))
  
  (when (gpu-available?)
    (with-engine :gpu
      (require '[uncomplicate.neanderthal.cuda :refer [cuge]])
      (with-release [m (cuge 3 3)]
        m)))
  )