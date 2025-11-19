(ns llm.neo.train
  "Training loop for GPT-2 model."
  (:require [llm.neo.forward :as fwd]
            [llm.neo.backward :as bwd]
            [llm.neo.loss :as loss]
            [llm.neo.core :as neo]
            [llm.neo.model :as model]))

(defn compute-loss-gradient
  "Compute cross-entropy loss and its gradient.
  
  Args:
    logits: [B, T, V] nested vectors
    targets: [B, T] token indices
    
  Returns:
    {:loss scalar
     :dlogits [B, T, V] gradient}"
  [logits targets]
  (let [B (count logits)
        T (count (first logits))
        V (count (first (first logits)))
        
        ;; Flatten for loss computation
        logits-flat (vec (for [b (range B) t (range T)] 
                          (nth (nth logits b) t)))
        targets-flat (vec (for [b (range B) t (range T)]
                           (nth (nth targets b) t)))
        
        ;; Compute loss and gradient
        logits-mat (neo/vec->matrix logits-flat)
        loss-val (loss/cross-entropy-loss logits-mat targets-flat)
        dlogits-mat (loss/cross-entropy-loss-gradient logits-mat targets-flat)
        
        ;; Reshape gradient back to [B, T, V]
        dlogits-flat-vec (neo/matrix->vec dlogits-mat)
        dlogits-nested (mapv 
                         (fn [b]
                           (mapv 
                             (fn [t] 
                               (nth dlogits-flat-vec (+ (* b T) t)))
                             (range T)))
                         (range B))]
    
    {:loss loss-val
     :dlogits dlogits-nested}))

(defn- apply-sgd-update
  "Apply simple SGD update: param = param - lr * grad
  
  Operates on nested vector structures."
  [params grads learning-rate]
  (letfn [(update-vec [p g]
            (if (vector? (first p))
              ;; Nested vector - recurse
              (mapv update-vec p g)
              ;; Leaf vector - apply update
              (mapv (fn [pi gi] (- pi (* learning-rate gi))) p g)))]
    (model/->ParameterTensors
      (update-vec (:wte params) (:wte grads))
      (update-vec (:wpe params) (:wpe grads))
      (update-vec (:ln1w params) (:ln1w grads))
      (update-vec (:ln1b params) (:ln1b grads))
      (update-vec (:qkvw params) (:qkvw grads))
      (update-vec (:qkvb params) (:qkvb grads))
      (update-vec (:attprojw params) (:attprojw grads))
      (update-vec (:attprojb params) (:attprojb grads))
      (update-vec (:ln2w params) (:ln2w grads))
      (update-vec (:ln2b params) (:ln2b grads))
      (update-vec (:fcw params) (:fcw grads))
      (update-vec (:fcb params) (:fcb grads))
      (update-vec (:fcprojw params) (:fcprojw grads))
      (update-vec (:fcprojb params) (:fcprojb grads))
      (update-vec (:lnfw params) (:lnfw grads))
      (update-vec (:lnfb params) (:lnfb grads)))))

(defn- compute-gradient-norm
  "Compute L2 norm of all gradients (for monitoring)."
  [grads]
  (letfn [(sum-squares [v]
            (if (vector? (first v))
              (reduce + (map sum-squares v))
              (reduce + (map #(* % %) v))))]
    (Math/sqrt
      (+ (sum-squares (:wte grads))
         (sum-squares (:wpe grads))
         (sum-squares (:ln1w grads))
         (sum-squares (:ln1b grads))
         (sum-squares (:qkvw grads))
         (sum-squares (:qkvb grads))
         (sum-squares (:attprojw grads))
         (sum-squares (:attprojb grads))
         (sum-squares (:ln2w grads))
         (sum-squares (:ln2b grads))
         (sum-squares (:fcw grads))
         (sum-squares (:fcb grads))
         (sum-squares (:fcprojw grads))
         (sum-squares (:fcprojb grads))
         (sum-squares (:lnfw grads))
         (sum-squares (:lnfb grads))))))

(defn training-step
  "Single training step: forward → loss → backward → update.
  
  Args:
    model-state: ModelState record
    batch: {:inputs [B, T] :targets [B, T]}
    learning-rate: learning rate for optimizer
    
  Returns:
    Updated ModelState with new parameters and incremented step"
  [model-state batch learning-rate]
  (let [{:keys [config params step]} model-state
        
        ;; Forward pass with caching
        forward-result (fwd/gpt2-forward-with-cache 
                         (:inputs batch) config params)
        
        ;; Compute loss and gradient
        {:keys [loss dlogits]} (compute-loss-gradient
                                 (:logits forward-result)
                                 (:targets batch))
        
        ;; Backward pass
        grads (bwd/gpt2-backward 
                dlogits 
                forward-result  ; Pass entire result (has :cache and :inputs)
                config
                params)
        
        ;; Compute gradient norm for monitoring
        grad-norm (compute-gradient-norm grads)
        
        ;; Apply optimizer (simple SGD for now)
        updated-params (apply-sgd-update params grads learning-rate)
        
        new-step (inc step)]
    
    (assoc model-state
           :step new-step
           :loss loss
           :grad-norm grad-norm
           :params updated-params)))

(defn train-epoch
  "Train for one epoch over dataset.
  
  Args:
    model-state: ModelState
    dataset: sequence of batches
    learning-rate: learning rate
    
  Returns:
    Updated ModelState"
  [model-state dataset learning-rate]
  (reduce 
    (fn [state batch]
      (let [updated (training-step state batch learning-rate)]
        (when (zero? (mod (:step updated) 10))
          (println (format "Step %d, Loss: %.4f, Grad Norm: %.4f" 
                          (:step updated) 
                          (:loss updated)
                          (:grad-norm updated 0.0))))
        updated))
    model-state
    dataset))

(comment
  ;; Example usage
  (require '[llm.neo.model :as model])
  (require '[llm.neo.train :as train])
  
  ;; Create tiny model
  (def model (model/create-model :tiny))
  
  ;; Create dummy batch
  (def batch {:inputs [[1 2 3 4] [5 6 7 8]]
              :targets [[2 3 4 5] [6 7 8 9]]})
  
  ;; Train one step
  (def updated (train/training-step model batch 0.0001))
  
  ;; Check that parameters changed
  (not= (:params model) (:params updated))
  
  ;; Check loss
  (:loss updated)
  )