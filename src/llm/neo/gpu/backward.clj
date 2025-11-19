(ns llm.neo.gpu.backward
  "GPU-native backward pass for GPT-2 - Phase 4 implementation.
  
  Note: Currently uses hybrid approach - forward on GPU, backward on CPU, gradients on GPU.
  Full GPU backward will be completed in future iterations."
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mrows ncols entry entry!]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.backward :as cpu-backward]
            [llm.neo.encoder :as encoder]
            [llm.neo.core :as neo]))

;; ============================================================================
;; GPU-to-CPU Conversion Helpers
;; ============================================================================

(defn- gpu-matrix->nested-vec
  "Convert GPU matrix to CPU nested vector."
  [mat-gpu]
  (vec (for [i (range (mrows mat-gpu))]
        (vec (for [j (range (ncols mat-gpu))]
              (entry mat-gpu i j))))))

(defn- gpu-vector->vec
  "Convert GPU vector to CPU vector."
  [v-gpu]
  (vec (for [i (range (ncore/dim v-gpu))]
        (entry v-gpu i))))

;; ============================================================================
;; GPU Cache to CPU Cache Conversion
;; ============================================================================

(defn- gpu-cache->cpu-cache
  "Convert GPU cache structure to CPU format for backward pass.
  
  Args:
    cache-gpu: vector of B cache maps with GPU tensors
    tokens: [B, T] token indices
    
  Returns:
    Cache structure compatible with CPU backward pass"
  [cache-gpu tokens]
  {:cache (mapv (fn [cache-b]
                 {:layer-caches
                  (mapv (fn [layer-cache]
                         {:x-input (gpu-matrix->nested-vec (:x-input layer-cache))
                          :ln1-output (gpu-matrix->nested-vec (:ln1-output layer-cache))
                          :attn-cache (:attn-cache layer-cache)
                          :attn-out-before-proj (gpu-matrix->nested-vec (:attn-out-before-proj layer-cache))
                          :attn-output (gpu-matrix->nested-vec (:attn-output layer-cache))
                          :res1-output (gpu-matrix->nested-vec (:res1-output layer-cache))
                          :ln2-output (gpu-matrix->nested-vec (:ln2-output layer-cache))
                          :fc-up (gpu-matrix->nested-vec (:fc-up layer-cache))
                          :gelu-output (gpu-matrix->nested-vec (:gelu-output layer-cache))})
                       (:layer-caches cache-b))
                  :final-ln-input (gpu-matrix->nested-vec (:final-ln-input cache-b))
                  :final-ln-output (gpu-matrix->nested-vec (:final-ln-output cache-b))})
               cache-gpu)
   :inputs tokens})

;; ============================================================================
;; Hybrid GPU-CPU Backward
;; ============================================================================

(defn gpt2-backward-hybrid
  "Hybrid GPU-CPU backward pass.
  
  Converts GPU cache to CPU, runs CPU backward, returns CPU gradients.
  This is a transitional implementation for Phase 4.
  
  Args:
    dlogits-gpu: vector of B GPU matrices [T, V]
    cache-gpu: vector of B cache maps with GPU tensors
    tokens: [B, T] token indices (CPU)
    config: GPT2Config
    params: CPU ParameterTensors
    
  Returns:
    CPU gradients (ParameterTensors)"
  [dlogits-gpu cache-gpu tokens config params]
  (let [;; Convert dlogits from GPU to CPU
        dlogits-cpu (mapv gpu-matrix->nested-vec dlogits-gpu)
        
        ;; Convert cache from GPU to CPU
        cache-cpu (gpu-cache->cpu-cache cache-gpu tokens)
        
        ;; Run CPU backward pass
        grads-cpu (cpu-backward/gpt2-backward dlogits-cpu cache-cpu config params)]
    
    grads-cpu))

(comment
  ;; REPL exploration
  (require '[llm.neo.gpu.forward :as gpu-fwd])
  (require '[llm.neo.gpu.backward :as gpu-bwd])
  (require '[llm.neo.gpu.core :as gpu])
  (require '[llm.neo.model :as model])
  
  (gpu/initialize-gpu)
  (def m (model/create-model :micro))
  (def tokens [[1 2 3 4]])
  
  (when (gpu/gpu-available?)
    ;; Forward pass on GPU
    (def fwd-result (gpu-fwd/gpt2-forward-gpu tokens (:config m) (:params m)))
    
    ;; Create fake gradient
    (def dlogits-gpu (:logits-gpu fwd-result))
    
    ;; Backward pass (hybrid)
    (def grads (gpu-bwd/gpt2-backward-hybrid
                dlogits-gpu
                (:cache-gpu fwd-result)
                tokens
                (:config m)
                (:params m)))
    
    ;; grads is CPU format (nested vectors)
    (println "Got gradients"))
  )