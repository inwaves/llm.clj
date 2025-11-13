(ns llm.neo.demo-train
  "Simple training demo to validate Phase 2 implementation.
  
  This creates a micro model and trains it on synthetic data to prove
  that the backward pass and training loop work correctly."
  (:require [llm.neo.model :as model]
            [llm.neo.train :as train]))

(defn create-synthetic-batch
  "Create a simple synthetic batch for training demo.
  Just uses sequential tokens to verify the mechanics work."
  [batch-size seq-len vocab-size]
  (let [;; Create simple sequential token patterns
        inputs (vec (for [b (range batch-size)]
                     (vec (map #(mod (+ b %) vocab-size) (range seq-len)))))
        ;; Targets are shifted by 1
        targets (vec (for [b (range batch-size)]
                      (vec (map #(mod (+ b % 1) vocab-size) (range seq-len)))))]
    {:inputs inputs
     :targets targets}))

(defn run-training-demo
  "Run a simple training demo with a micro model.
  
  This proves that:
  - Forward pass computes logits
  - Loss is computed correctly
  - Backward pass computes gradients
  - Parameters are updated
  - Loss decreases over steps (proving gradients flow correctly)"
  []
  (println "=== Phase 2 Training Demo (Micro Model) ===\n")
  
  ;; Create micro model using the public API
  (println "Creating micro GPT-2 model...")
  (def model-state (model/create-model :micro))
  (println (format "  Vocab size: %d" (get-in model-state [:config :vocab-size])))
  (println (format "  Layers: %d" (get-in model-state [:config :num-layers])))
  (println (format "  Channels: %d" (get-in model-state [:config :channels])))
  (println (format "  Heads: %d\n" (get-in model-state [:config :num-heads])))
  
  ;; Minimal training parameters
  (def batch-size 1)
  (def seq-len 4)
  (def vocab-size (get-in model-state [:config :vocab-size]))
  (def learning-rate 0.01)
  (def num-steps 3)
  
  (println (format "Training configuration:"))
  (println (format "  Batch size: %d" batch-size))
  (println (format "  Sequence length: %d" seq-len))
  (println (format "  Learning rate: %.5f" learning-rate))
  (println (format "  Training steps: %d\n" num-steps))
  
  ;; Create synthetic training data
  (def training-batches
    (vec (repeatedly num-steps
                    #(create-synthetic-batch batch-size seq-len vocab-size))))
  
  (println "Starting training...\n")
  
  ;; Train for a few steps
  (def final-state
    (loop [step 0
           state model-state
           losses []]
      (if (< step num-steps)
        (do
          (println (format "Computing step %d..." step))
          (let [batch (nth training-batches step)
                updated-state (train/training-step state batch learning-rate)
                loss (:loss updated-state)
                grad-norm (get updated-state :grad-norm 0.0)]
            
            (println (format "Step %d: Loss = %.6f, Grad Norm = %.6f"
                            step loss grad-norm))
            
            (recur (inc step) updated-state (conj losses loss))))
        
        (do
          (println "\n=== Training Complete ===")
          (println (format "Initial loss: %.6f" (first losses)))
          (println (format "Final loss: %.6f" (last losses)))
          (println (format "Loss improvement: %.6f" (- (first losses) (last losses))))
          (if (< (last losses) (first losses))
            (println "✓ Loss decreased - gradients are flowing correctly!")
            (println "✗ Loss did not decrease - there may be an issue"))
          state))))
  
  (println "\nDemo complete!"))

(defn -main
  "Entry point for demo"
  [& args]
  (run-training-demo))

(comment
  ;; Run demo from REPL
  (run-training-demo)
  )