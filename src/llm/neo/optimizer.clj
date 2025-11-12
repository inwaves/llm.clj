(ns llm.neo.optimizer
  "AdamW optimizer implementation."
  (:use [uncomplicate.neanderthal core native]))

(defrecord AdamWState [m v step])

(defn create-optimizer-state
  "Initialize optimizer state (m, v) for all parameters.
  For simplicity, using flat representations."
  [param-count]
  (->AdamWState
    (vec (repeat param-count 0.0))  ; First moment
    (vec (repeat param-count 0.0))  ; Second moment
    0))                              ; Step counter

(defn adamw-update
  "AdamW optimizer step.
  Args:
    params: parameter vector
    grads: gradient vector  
    state: AdamWState
    lr: learning rate
    beta1: first moment decay (default 0.9)
    beta2: second moment decay (default 0.999)
    eps: numerical stability (default 1e-8)
    weight-decay: L2 regularization (default 0.01)
  Returns: [updated-params updated-state]"
  [params grads state lr & {:keys [beta1 beta2 eps weight-decay]
                            :or {beta1 0.9 beta2 0.999 eps 1e-8 weight-decay 0.01}}]
  (let [step (inc (:step state))
        
        ;; Update biased first moment estimate
        m-new (mapv (fn [m-i g-i]
                     (+ (* beta1 m-i) (* (- 1.0 beta1) g-i)))
                   (:m state) grads)
        
        ;; Update biased second moment estimate
        v-new (mapv (fn [v-i g-i]
                     (+ (* beta2 v-i) (* (- 1.0 beta2) (* g-i g-i))))
                   (:v state) grads)
        
        ;; Bias correction
        m-hat (mapv #(/ % (- 1.0 (Math/pow beta1 step))) m-new)
        v-hat (mapv #(/ % (- 1.0 (Math/pow beta2 step))) v-new)
        
        ;; Parameter update with weight decay
        params-new (mapv (fn [p m v]
                          (let [update (/ m (+ (Math/sqrt v) eps))]
                            (- p (* lr (+ update (* weight-decay p))))))
                        params m-hat v-hat)
        
        state-new (->AdamWState m-new v-new step)]
    
    [params-new state-new]))