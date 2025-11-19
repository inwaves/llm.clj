(ns llm.neo.gpu.train
  "GPU training loop for GPT-2 - Phase 4 implementation.
  
  Current approach: GPU forward, CPU backward, CPU parameter updates.
  This is a transitional implementation that provides some GPU acceleration
  while maintaining compatibility with existing CPU infrastructure.
  
  Future: Full GPU backward and GPU parameter updates for maximum performance."
  (:require [llm.neo.gpu.forward :as gpu-fwd]
            [llm.neo.gpu.backward :as gpu-bwd]
            [llm.neo.loss :as loss]
            [llm.neo.train :as cpu-train]
            [llm.neo.core :as neo]))

;; ============================================================================
;; GPU Loss Computation (on CPU for now)
;; ============================================================================

(defn- gpu-logits->cpu
  "Transfer GPU logits to CPU nested vectors."
  [logits-gpu]
  (require '[llm.neo.gpu.core :as gpu])
  (mapv (fn [logits-mat-gpu]
          (let [logits-cpu (gpu/to-cpu logits-mat-gpu)]
            (neo/matrix->vec logits-cpu)))
        logits-gpu))

;; ============================================================================
;; Training Step
;; ============================================================================

(defn training-step-gpu
  "Single training step with GPU forward pass.
  
  Current implementation:
  - Forward: GPU (fast, minimizes transfers)
  - Loss: CPU (simple cross-entropy)
  - Backward: CPU (uses existing tested implementation)
  - Update: CPU (uses existing optimizer)
  
  Args:
    model-state: ModelState (CPU parameters)
    batch: {:inputs [B, T] :targets [B, T]}
    learning-rate: float
    
  Returns:
    Updated ModelState with new parameters and loss/grad-norm metrics"
  [model-state batch learning-rate]
  (let [{:keys [config params step]} model-state
        
        ;; Forward pass on GPU
        forward-result (gpu-fwd/gpt2-forward-gpu (:inputs batch) config params)
        logits-gpu (:logits-gpu forward-result)
        cache-gpu (:cache-gpu forward-result)
        
        ;; Transfer logits to CPU for loss computation
        logits-cpu (gpu-logits->cpu logits-gpu)
        
        ;; Compute loss and gradient (CPU)
        {:keys [loss dlogits]} (cpu-train/compute-loss-gradient
                                 logits-cpu
                                 (:targets batch))
        
        ;; Convert dlogits to GPU for backward
        dlogits-gpu (mapv neo/vec->matrix dlogits)
        dlogits-gpu-matrices (mapv (fn [dlogits-mat]
                                    (require '[llm.neo.gpu.core :as gpu])
                                    (gpu/to-gpu dlogits-mat))
                                  dlogits-gpu)
        
        ;; Backward pass (hybrid GPU-CPU)
        grads (gpu-bwd/gpt2-backward-hybrid
                dlogits-gpu-matrices
                cache-gpu
                (:inputs batch)
                config
                params)
        
        ;; Compute gradient norm (CPU)
        grad-norm (cpu-train/compute-gradient-norm grads)
        
        ;; Apply optimizer (CPU)
        updated-params (cpu-train/apply-sgd-update params grads learning-rate)
        
        new-step (inc step)]
    
    ;; Cleanup GPU resources
    (require '[uncomplicate.commons.core :refer [release]])
    (doseq [logits-mat logits-gpu]
      (release logits-mat))
    (doseq [dlogits-mat dlogits-gpu-matrices]
      (release dlogits-mat))
    (doseq [cache-b cache-gpu]
      (release (:final-ln-input cache-b))
      (release (:final-ln-output cache-b))
      (doseq [layer-cache (:layer-caches cache-b)]
        (release (:x-input layer-cache))
        (release (:ln1-output layer-cache))
        (release (:attn-out-before-proj layer-cache))
        (release (:attn-output layer-cache))
        (release (:res1-output layer-cache))
        (release (:ln2-output layer-cache))
        (release (:fc-up layer-cache))
        (release (:gelu-output layer-cache))
        ;; Release attention cache
        (when-let [attn-cache (:attn-cache layer-cache)]
          (when-let [q (:q attn-cache)] (release q))
          (when-let [k (:k attn-cache)] (release k))
          (when-let [v (:v attn-cache)] (release v))
          (doseq [p (:att-probs attn-cache)]
            (release p)))))
    
    (assoc model-state
           :step new-step
           :loss loss
           :grad-norm grad-norm
           :params updated-params)))

;; ============================================================================
;; Training Loop
;; ============================================================================

(defn train-epoch-gpu
  "Train for one epoch using GPU forward pass.
  
  Args:
    model-state: ModelState
    dataset: sequence of batches
    learning-rate: float
    
  Returns:
    Updated ModelState"
  [model-state dataset learning-rate]
  (reduce
    (fn [state batch]
      (let [updated (training-step-gpu state batch learning-rate)]
        (when (zero? (mod (:step updated) 10))
          (println (format "Step %d, Loss: %.4f, Grad Norm: %.4f"
                          (:step updated)
                          (:loss updated 0.0)
                          (:grad-norm updated 0.0))))
        updated))
    model-state
    dataset))

(comment
  ;; REPL usage
  (require '[llm.neo.model :as model])
  (require '[llm.neo.gpu.train :as gpu-train])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  
  ;; Create model
  (def m (model/create-model :micro))
  
  ;; Create dummy batch
  (def batch {:inputs [[1 2 3 4] [5 6 7 8]]
              :targets [[2 3 4 5] [6 7 8 9]]})
  
  (when (gpu/gpu-available?)
    ;; Train one step
    (def updated (gpu-train/training-step-gpu m batch 0.0001))
    
    ;; Check loss
    (println "Loss:" (:loss updated)))
  )