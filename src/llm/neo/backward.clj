(ns llm.neo.backward
  "Composed backward pass for GPT-2 model.
  
  This implements the full backpropagation through the model by chaining
  individual operation backward passes in reverse order of the forward pass."
  (:use [uncomplicate.neanderthal core native])
  (:require
    [llm.neo.core :as neo]
    [llm.neo.encoder :as enc]
    [llm.neo.layernorm :as ln]
    [llm.neo.gelu :as gelu]
    [llm.neo.matmul :as mm]
    [llm.neo.residual :as res]
    [llm.neo.attention :as attn]
    [llm.neo.softmax :as sm]
    [llm.neo.model :as model]))

;; ============================================================================
;; Helper Functions
;; ============================================================================

(defn- accumulate-gradients
  "Sum a list of gradient matrices element-wise.
  All matrices must have the same shape."
  [grad-list]
  (when (empty? grad-list)
    (throw (ex-info "Cannot accumulate empty gradient list" {})))
  (let [first-grad (first grad-list)
        rows (mrows first-grad)
        cols (ncols first-grad)
        result (copy! first-grad (dge rows cols))]
    (doseq [grad (rest grad-list)]
      (axpy! 1.0 grad result))
    result))

(defn- add-vector-gradients
  "Add two gradient vectors element-wise."
  [v1 v2]
  (mapv + v1 v2))

(defn- zero-gradients
  "Initialize zero gradients matching parameter structure."
  [config]
  (let [{:keys [vocab-size max-seq-len num-layers channels]} config
        C channels
        L num-layers
        V vocab-size]
    (model/->ParameterTensors
      (vec (repeat V (vec (repeat C 0.0))))      ; wte
      (vec (repeat max-seq-len (vec (repeat C 0.0))))  ; wpe
      (vec (repeat L (vec (repeat C 0.0))))      ; ln1w
      (vec (repeat L (vec (repeat C 0.0))))      ; ln1b
      (vec (repeat (* L 3 C) (vec (repeat C 0.0))))  ; qkvw
      (vec (repeat (* L 3 C) 0.0))               ; qkvb
      (vec (repeat (* L C) (vec (repeat C 0.0))))    ; attprojw
      (vec (repeat (* L C) 0.0))                 ; attprojb
      (vec (repeat L (vec (repeat C 0.0))))      ; ln2w
      (vec (repeat L (vec (repeat C 0.0))))      ; ln2b
      (vec (repeat (* L 4 C) (vec (repeat C 0.0))))  ; fcw
      (vec (repeat (* L 4 C) 0.0))               ; fcb
      (vec (repeat (* L C) (vec (repeat (* 4 C) 0.0))))  ; fcprojw
      (vec (repeat (* L C) 0.0))                 ; fcprojb
      (vec (repeat C 0.0))                       ; lnfw
      (vec (repeat C 0.0)))))                    ; lnfb

(defn- slice-rows->dge
  "Copy rows [start, start+count) from matrix into new matrix."
  [mat start count]
  (let [cols (ncols mat)
        out (dge count cols)]
    (dotimes [i count]
      (dotimes [j cols]
        (entry! out i j (entry mat (+ start i) j))))
    out))

(defn- matrix->nested-vec
  "Convert matrix rows into nested vector structure."
  [mat start count]
  (vec (for [i (range start (+ start count))]
         (vec (for [j (range (ncols mat))]
                (entry mat i j))))))

(defn- add-matrix-to-nested-vec
  "Add matrix gradient to nested vector gradient structure.
  Returns updated nested vector."
  [nested-vec grad-mat start-row]
  (let [grad-rows (mrows grad-mat)
        grad-cols (ncols grad-mat)]
    (loop [i 0 result nested-vec]
      (if (< i grad-rows)
        (let [target-row (+ start-row i)
              new-row (vec (for [j (range grad-cols)]
                            (+ (get-in result [target-row j])
                               (entry grad-mat i j))))]
          (recur (inc i) (assoc result target-row new-row)))
        result))))

;; ============================================================================
;; Block Backward Pass
;; ============================================================================

(def ^:private ln-eps 1e-5)

(defn- block-backward
  "Backward pass through one transformer block.
  
  Uses standalone attention-backward from attention.clj.
  
  Args:
    dx-in: gradient flowing into this block [T, C]
    cache: cached activations from forward pass
    l: layer index
    params: ParameterTensors
    config: GPT2Config
    grads: current gradient accumulator
    
  Returns:
    {:dx-out gradient flowing out [T, C]
     :grads updated gradients}"
  [dx-in cache l params config grads]
  (let [{:keys [num-heads channels]} config
        C channels
        
        ;; Extract cached activations
        {:keys [x-input ln1-output qkv-matrix attn-cache attn-out-before-proj
                res1-output ln2-output fc-up gelu-output]} cache
        
        ;; Extract layer parameters
        gamma2 (dv (nth (:ln2w params) l))
        beta2  (dv (nth (:ln2b params) l))
        gamma1 (dv (nth (:ln1w params) l))
        beta1  (dv (nth (:ln1b params) l))
        
        ;; Convert stacked weights to matrices
        attprojw-m (neo/vec->matrix (:attprojw params))
        fcprojw-m (neo/vec->matrix (:fcprojw params))
        fcw-m (neo/vec->matrix (:fcw params))
        qkvw-m (neo/vec->matrix (:qkvw params))
        
        ;; Backward through second residual
        {:keys [dx1 dx2]} (res/residual-backward dx-in)
        dx-fc dx2
        dx-res1 dx1
        
        ;; Backward through MLP projection (down)
        fc2-w (slice-rows->dge fcprojw-m (* l C) C)
        fc2-grad (mm/matmul-backward dx-fc gelu-output fc2-w)
        
        ;; Update fcprojw gradients
        fcprojw-updated (add-matrix-to-nested-vec 
                          (:fcprojw grads)
                          (:dweight fc2-grad)
                          (* l C))
        
        ;; Update fcprojb gradients
        fc2-dbias-vec (mapv #(entry (:dbias fc2-grad) %) 
                           (range (dim (:dbias fc2-grad))))
        fcprojb-updated (loop [i 0 result (:fcprojb grads)]
                         (if (< i C)
                           (let [idx (+ (* l C) i)]
                             (recur (inc i) 
                                   (assoc result idx 
                                         (+ (nth result idx) (nth fc2-dbias-vec i)))))
                           result))
        
        ;; GELU backward
        dx-gelu (gelu/gelu-backward fc-up (:dinp fc2-grad))
        
        ;; Backward through MLP projection (up)
        fc1-w (slice-rows->dge fcw-m (* l 4 C) (* 4 C))
        fc1-grad (mm/matmul-backward dx-gelu ln2-output fc1-w)
        
        ;; Update fcw gradients
        fcw-updated (add-matrix-to-nested-vec
                      (:fcw grads)
                      (:dweight fc1-grad)
                      (* l 4 C))
        
        ;; Update fcb gradients
        fc1-dbias-vec (mapv #(entry (:dbias fc1-grad) %)
                           (range (dim (:dbias fc1-grad))))
        fcb-updated (loop [i 0 result (:fcb grads)]
                     (if (< i (* 4 C))
                       (let [idx (+ (* l 4 C) i)]
                         (recur (inc i)
                               (assoc result idx
                                     (+ (nth result idx) (nth fc1-dbias-vec i)))))
                       result))
        
        ;; Backward through second layer norm
        ln2-grad (ln/layernorm-backward res1-output (:dinp fc1-grad) gamma2 beta2 ln-eps)
        
        ;; Update LN2 gradients
        ln2w-vec (mapv #(entry (:dgamma ln2-grad) %) (range C))
        ln2b-vec (mapv #(entry (:dbeta ln2-grad) %) (range C))
        ln2w-updated (assoc (:ln2w grads) l 
                           (add-vector-gradients (nth (:ln2w grads) l) ln2w-vec))
        ln2b-updated (assoc (:ln2b grads) l
                           (add-vector-gradients (nth (:ln2b grads) l) ln2b-vec))
        
        ;; Combine gradients at first residual
        dx-to-res1 (accumulate-gradients [dx-res1 (:dx ln2-grad)])
        
        ;; Backward through first residual
        {:keys [dx1 dx2]} (res/residual-backward dx-to-res1)
        dx-attn-proj dx2  ; Gradient for attention output projection
        dx-to-ln1 dx1     ; Gradient continuing from input
        
        ;; Backward through attention output projection
        att-w (slice-rows->dge attprojw-m (* l C) C)
        att-proj-grad (mm/matmul-backward dx-attn-proj attn-out-before-proj att-w)
        
        ;; Update attprojw gradients
        attprojw-updated (add-matrix-to-nested-vec
                           (:attprojw grads)
                           (:dweight att-proj-grad)
                           (* l C))
        
        ;; Update attprojb gradients
        att-proj-dbias-vec (mapv #(entry (:dbias att-proj-grad) %)
                                (range (dim (:dbias att-proj-grad))))
        attprojb-updated (loop [i 0 result (:attprojb grads)]
                          (if (< i C)
                            (let [idx (+ (* l C) i)]
                              (recur (inc i)
                                    (assoc result idx
                                          (+ (nth result idx) (nth att-proj-dbias-vec i)))))
                            result))
        ;; Backward through attention mechanism using standalone module
        ;; Wrap as [B=1, T, C] for batch interface
        dx-attn-vec (neo/matrix->vec (:dinp att-proj-grad))  ; [T, C] as nested vector
        dx-attn-nested [dx-attn-vec]  ; Wrap for batch dimension [B=1, T, C]
        dx-qkv-result (llm.neo.attention/attention-backward dx-attn-nested [attn-cache] num-heads)
        
        ;; Extract [T, 3C] from [B=1, T, 3C]
        dx-qkv-vec (first dx-qkv-result)  ; Get batch 0: [T, 3C]
        dx-qkv (neo/vec->matrix dx-qkv-vec)
        
        ;; Backward through QKV projection
        qkv-w (slice-rows->dge qkvw-m (* l 3 C) (* 3 C))
        qkv-grad (mm/matmul-backward dx-qkv ln1-output qkv-w)
        
        ;; Update qkvw gradients
        qkvw-updated (add-matrix-to-nested-vec
                       (:qkvw grads)
                       (:dweight qkv-grad)
                       (* l 3 C))
        
        ;; Update qkvb gradients
        qkv-dbias-vec (mapv #(entry (:dbias qkv-grad) %)
                           (range (dim (:dbias qkv-grad))))
        qkvb-updated (loop [i 0 result (:qkvb grads)]
                      (if (< i (* 3 C))
                        (let [idx (+ (* l 3 C) i)]
                          (recur (inc i)
                                (assoc result idx
                                      (+ (nth result idx) (nth qkv-dbias-vec i)))))
                        result))
        
        ;; Backward through first layer norm
        ln1-grad (ln/layernorm-backward x-input (:dinp qkv-grad) gamma1 beta1 ln-eps)
        
        ;; Update LN1 gradients
        ln1w-vec (mapv #(entry (:dgamma ln1-grad) %) (range C))
        ln1b-vec (mapv #(entry (:dbeta ln1-grad) %) (range C))
        ln1w-updated (assoc (:ln1w grads) l
                           (add-vector-gradients (nth (:ln1w grads) l) ln1w-vec))
        ln1b-updated (assoc (:ln1b grads) l
                           (add-vector-gradients (nth (:ln1b grads) l) ln1b-vec))
        
        ;; Combine final gradients
        dx-out (accumulate-gradients [dx-to-ln1 (:dx ln1-grad)])
        
        ;; Build updated grads with all gradients
        grads-updated (assoc grads
                            :fcprojw fcprojw-updated
                            :fcprojb fcprojb-updated
                            :fcw fcw-updated
                            :fcb fcb-updated
                            :ln2w ln2w-updated
                            :ln2b ln2b-updated
                            :attprojw attprojw-updated
                            :attprojb attprojb-updated
                            :qkvw qkvw-updated
                            :qkvb qkvb-updated
                            :ln1w ln1w-updated
                            :ln1b ln1b-updated)]
    
    {:dx-out dx-out
     :grads grads-updated}))

;; ============================================================================
;; Full Model Backward Pass
;; ============================================================================

(defn gpt2-backward
  "Composed backward pass for GPT-2.
  
  Args:
    loss-grad: gradient of loss w.r.t. logits [B, T, V] nested vectors
    cache: map with :cache (per-batch caches) and :inputs (token ids)
    config: GPT2Config
    params: ParameterTensors
    
  Returns:
    ParameterTensors with gradients for all parameters"
  [loss-grad cache config params]
  (let [{:keys [vocab-size num-layers channels]} config
        C channels
        L num-layers
        V vocab-size
        
        ;; Initialize gradients
        grads-init (zero-gradients config)
        
        B (count loss-grad)
        wte-m (neo/vec->matrix (:wte params))]
    
    ;; Process each batch element
    (loop [b 0
           grads grads-init]
      (if (< b B)
        (let [dlogits-b (neo/vec->matrix (nth loss-grad b))
              cache-b (nth (:cache cache) b)
              T (mrows dlogits-b)
              
              ;; Backward through output projection (tied to wte)
              dx-final-ln (dge T C)
              _ (mm! 1.0 dlogits-b wte-m 0.0 dx-final-ln)
              
              ;; Gradient for wte from projection: dwte = dlogits^T @ x
              dwte-proj (dge V C)
              _ (mm! 1.0 (trans dlogits-b) (:final-ln-output cache-b) 0.0 dwte-proj)
              dwte-proj-vec (matrix->nested-vec dwte-proj 0 V)
              
              ;; Backward through final layer norm
              lnfw (dv (:lnfw params))
              lnfb (dv (:lnfb params))
              lnf-grad (ln/layernorm-backward
                         (:final-ln-input cache-b)
                         dx-final-ln
                         lnfw lnfb ln-eps)
              
              ;; Update final LN gradients
              lnfw-vec (mapv #(entry (:dgamma lnf-grad) %) (range C))
              lnfb-vec (mapv #(entry (:dbeta lnf-grad) %) (range C))
              grads-with-lnf (assoc grads
                                   :lnfw (add-vector-gradients (:lnfw grads) lnfw-vec)
                                   :lnfb (add-vector-gradients (:lnfb grads) lnfb-vec))
              
              ;; Backward through blocks (in reverse)
              [dx-encoder grads-after-blocks]
              (loop [l (dec L)
                     dx (:dx lnf-grad)
                     g grads-with-lnf]
                (if (>= l 0)
                  (let [layer-cache (nth (:layer-caches cache-b) l)
                        {:keys [dx-out grads]} (block-backward dx layer-cache l params config g)]
                    (recur (dec l) dx-out grads))
                  [dx g]))
              
              ;; Backward through encoder
              encoder-input (nth (:inputs cache) b)
              ;; Convert dx-encoder [T, C] to nested vector [B=1, T, C]
              dx-encoder-vec (neo/matrix->vec dx-encoder)  ; [[c0,c1,...], [c0,c1,...], ...] shape [T, C]
              encoder-grad (enc/encoder-backward
                             [dx-encoder-vec]  ; Wrap as batch: [B=1, T, C]
                             [encoder-input]
                             vocab-size
                             (:max-seq-len config)
                             channels)
              
              ;; Accumulate encoder + projection gradients for wte
              wte-updated (loop [i 0 result (:wte grads-after-blocks)]
                           (if (< i V)
                             (let [combined (add-vector-gradients
                                             (add-vector-gradients (nth result i) (nth dwte-proj-vec i))
                                             (nth (:dwte encoder-grad) i))]
                               (recur (inc i) (assoc result i combined)))
                             result))
              
              ;; Accumulate wpe gradients
              wpe-updated (loop [i 0 result (:wpe grads-after-blocks)]
                           (if (< i (:max-seq-len config))
                             (recur (inc i)
                                   (assoc result i
                                         (add-vector-gradients (nth result i) (nth (:dwpe encoder-grad) i))))
                             result))
              
              grads-final (assoc grads-after-blocks
                                :wte wte-updated
                                :wpe wpe-updated)]
          
          (recur (inc b) grads-final))
        
        ;; Return final gradients
        grads))))