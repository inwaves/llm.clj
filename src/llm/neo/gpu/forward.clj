(ns llm.neo.gpu.forward
  "GPU-native forward pass for GPT-2 - Phase 4 implementation with kernel fusion.
  
  Uses custom CUDA kernels for fused operations to eliminate synchronization overhead."
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mrows ncols entry entry! copy!]]
            [uncomplicate.neanderthal.native :as native]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.gpu.matmul :as gpu-mm]
            [llm.neo.gpu.attention :as gpu-attn]
            [llm.neo.gpu.kernels :as kernels]
            [llm.neo.gpu.residual :as gpu-res]
            [llm.neo.core :as neo]))

;; ============================================================================
;; GPU Constructor Helpers
;; ============================================================================

(defn- cuge
  "Allocate CUDA matrix [rows cols]."
  [rows cols]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols))

(defn- cuv
  "Allocate CUDA vector [n]."
  [n]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  ((resolve 'uncomplicate.neanderthal.cuda/cuv) n))

;; ============================================================================
;; GPU Encoder
;; ============================================================================

(defn encoder-forward-gpu
  "Encoder: token + position embeddings on GPU.
  
  Args:
    tokens: [B, T] nested vector (CPU)
    wte-gpu: [V, C] token embeddings on GPU
    wpe-gpu: [maxT, C] position embeddings on GPU
    
  Returns:
    Vector of B GPU matrices [T, C]. Caller must release each."
  [tokens wte-gpu wpe-gpu]
  (mapv
    (fn [batch-tokens]
      (let [T (count batch-tokens)
            C (ncols wte-gpu)
            out (cuge T C)]
        (dotimes [t T]
          (let [tok (nth batch-tokens t)]
            (dotimes [c C]
              (entry! out t c (+ (entry wte-gpu tok c) (entry wpe-gpu t c))))))
        out))
    tokens))

;; ============================================================================
;; Helper: Parameter Slicing on GPU
;; ============================================================================

(defn- slice-gpu-rows
  "Extract rows [start, start+count) from GPU matrix. Caller must release."
  [mat-gpu start count]
  (let [cols (ncols mat-gpu)
        out (cuge count cols)]
    (dotimes [i count]
      (dotimes [j cols]
        (entry! out i j (entry mat-gpu (+ start i) j))))
    out))

(defn- slice-gpu-vector
  "Extract elements [start, start+len) from GPU vector. Caller must release."
  [vec-gpu start len]
  (let [out (cuv len)]
    (dotimes [i len]
      (entry! out i (entry vec-gpu (+ start i))))
    out))

;; ============================================================================
;; GPU Block Forward
;; ============================================================================

(defn block-forward-gpu
  "Transformer block forward on GPU with fused kernels.
  
  Uses custom CUDA kernels for layernorm, fused residual+layernorm, and GELU.
  
  Args:
    x-gpu: [T, C] input
    l: layer index
    cfg: GPT2Config
    [GPU parameter tensors]: qkvw-gpu, qkvb-gpu, etc.
    
  Returns:
    {:output [T, C] GPU matrix
     :cache map of GPU tensors}
     
  Ownership:
    Caller must release :output and all cache tensors."
  [x-gpu l cfg
   qkvw-gpu qkvb-gpu attprojw-gpu attprojb-gpu
   ln1w-gpu ln1b-gpu ln2w-gpu ln2b-gpu
   fcw-gpu fcb-gpu fcprojw-gpu fcprojb-gpu]
  (let [{:keys [num-heads channels]} cfg
        C channels
        ln-eps 1e-5
        
        ;; Cache input
        x-input (copy! x-gpu (cuge (mrows x-gpu) (ncols x-gpu)))
        
        ;; Extract layer params
        gamma1-gpu (slice-gpu-vector ln1w-gpu (* l C) C)
        beta1-gpu (slice-gpu-vector ln1b-gpu (* l C) C)
        gamma2-gpu (slice-gpu-vector ln2w-gpu (* l C) C)
        beta2-gpu (slice-gpu-vector ln2b-gpu (* l C) C)
        qkv-w-gpu (slice-gpu-rows qkvw-gpu (* l 3 C) (* 3 C))
        qkv-b-gpu (slice-gpu-vector qkvb-gpu (* l 3 C) (* 3 C))
        att-w-gpu (slice-gpu-rows attprojw-gpu (* l C) C)
        att-b-gpu (slice-gpu-vector attprojb-gpu (* l C) C)
        fc1-w-gpu (slice-gpu-rows fcw-gpu (* l 4 C) (* 4 C))
        fc1-b-gpu (slice-gpu-vector fcb-gpu (* l 4 C) (* 4 C))
        fc2-w-gpu (slice-gpu-rows fcprojw-gpu (* l C) C)
        fc2-b-gpu (slice-gpu-vector fcprojb-gpu (* l C) C)
        
        ;; LN1 - using kernel-optimized version
        x-norm1 (kernels/layernorm! x-gpu gamma1-gpu beta1-gpu ln-eps)
        ln1-output (copy! x-norm1 (cuge (mrows x-norm1) (ncols x-norm1)))
        
        ;; QKV projection
        qkv (gpu-mm/matmul-forward-gpu x-norm1 qkv-w-gpu qkv-b-gpu)
        
        ;; Attention
        attn-result (gpu-attn/attention-forward-qkv-gpu qkv num-heads)
        attn-out-before-proj (:out-gpu attn-result)
        attn-cache (:cache attn-result)
        
        ;; Attention output projection
        attn-out (gpu-mm/matmul-forward-gpu attn-out-before-proj att-w-gpu att-b-gpu)
        attn-output (copy! attn-out (cuge (mrows attn-out) (ncols attn-out)))
        
        ;; Fused Residual 1 + LN2 - using kernel fusion
        x-norm2 (kernels/fused-residual-layernorm! x-gpu attn-out gamma2-gpu beta2-gpu ln-eps)
        ln2-output (copy! x-norm2 (cuge (mrows x-norm2) (ncols x-norm2)))
        res1-output (gpu-res/residual-forward-gpu x-gpu attn-out)
        
        ;; MLP up
        fc-up (gpu-mm/matmul-forward-gpu x-norm2 fc1-w-gpu fc1-b-gpu)
        fc-up-cached (copy! fc-up (cuge (mrows fc-up) (ncols fc-up)))
        
        ;; GELU - using kernel-optimized version
        act (kernels/gelu! fc-up)
        gelu-output (copy! act (cuge (mrows act) (ncols act)))
        
        ;; MLP down
        fc-down (gpu-mm/matmul-forward-gpu act fc2-w-gpu fc2-b-gpu)
        
        ;; Residual 2
        x-out (gpu-res/residual-forward-gpu res1-output fc-down)]
    
    ;; Release temporaries (not cached)
    (release x-norm1)
    (release qkv)
    (release attn-out)
    (release x-norm2)
    (release fc-up)
    (release act)
    (release fc-down)
    (release gamma1-gpu) (release beta1-gpu)
    (release gamma2-gpu) (release beta2-gpu)
    (release qkv-w-gpu) (release qkv-b-gpu)
    (release att-w-gpu) (release att-b-gpu)
    (release fc1-w-gpu) (release fc1-b-gpu)
    (release fc2-w-gpu) (release fc2-b-gpu)
    
    {:output x-out
     :cache {:x-input x-input
             :ln1-output ln1-output
             :attn-cache attn-cache
             :attn-out-before-proj attn-out-before-proj
             :attn-output attn-output
             :res1-output res1-output
             :ln2-output ln2-output
             :fc-up fc-up-cached
             :gelu-output gelu-output}}))

;; ============================================================================
;; Full Forward
;; ============================================================================

(defn gpt2-forward-gpu
  "Full GPT-2 forward on GPU.
  
  Args:
    tokens: [B, T] token indices (CPU)
    config: GPT2Config
    params: ParameterTensors (CPU)
    
  Returns:
    {:logits-gpu vector of B GPU matrices [T, V]
     :cache-gpu vector of B cache maps}
     
  Ownership:
    Caller must release all GPU tensors in return value."
  [tokens config params]
  (let [{:keys [vocab-size max-seq-len num-layers channels]} config
        C channels L num-layers V vocab-size ln-eps 1e-5]
    
    (with-release [wte-m (neo/vec->matrix (:wte params))
                   wpe-m (neo/vec->matrix (:wpe params))
                   qkvw-m (neo/vec->matrix (:qkvw params))
                   attprojw-m (neo/vec->matrix (:attprojw params))
                   fcw-m (neo/vec->matrix (:fcw params))
                   fcprojw-m (neo/vec->matrix (:fcprojw params))
                   qkvb-v (native/dv (double-array (:qkvb params)))
                   attprojb-v (native/dv (double-array (:attprojb params)))
                   fcb-v (native/dv (double-array (:fcb params)))
                   fcprojb-v (native/dv (double-array (:fcprojb params)))
                   ln1w-flat (native/dv (double-array (flatten (:ln1w params))))
                   ln1b-flat (native/dv (double-array (flatten (:ln1b params))))
                   ln2w-flat (native/dv (double-array (flatten (:ln2w params))))
                   ln2b-flat (native/dv (double-array (flatten (:ln2b params))))
                   lnfw-v (native/dv (double-array (:lnfw params)))
                   lnfb-v (native/dv (double-array (:lnfb params)))]
      
      (let [wte-gpu (gpu/to-gpu wte-m)
            wpe-gpu (gpu/to-gpu wpe-m)
            qkvw-gpu (gpu/to-gpu qkvw-m)
            qkvb-gpu (gpu/to-gpu qkvb-v)
            attprojw-gpu (gpu/to-gpu attprojw-m)
            attprojb-gpu (gpu/to-gpu attprojb-v)
            fcw-gpu (gpu/to-gpu fcw-m)
            fcb-gpu (gpu/to-gpu fcb-v)
            fcprojw-gpu (gpu/to-gpu fcprojw-m)
            fcprojb-gpu (gpu/to-gpu fcprojb-v)
            ln1w-gpu (gpu/to-gpu ln1w-flat)
            ln1b-gpu (gpu/to-gpu ln1b-flat)
            ln2w-gpu (gpu/to-gpu ln2w-flat)
            ln2b-gpu (gpu/to-gpu ln2b-flat)
            lnfw-gpu (gpu/to-gpu lnfw-v)
            lnfb-gpu (gpu/to-gpu lnfb-v)
            
            xs-gpu (encoder-forward-gpu tokens wte-gpu wpe-gpu)
            
            results (mapv
                      (fn [x0-gpu]
                        (let [[xL-gpu layer-caches]
                              (loop [l 0, x x0-gpu, caches []]
                                (if (< l L)
                                  (let [{:keys [output cache]}
                                        (block-forward-gpu x l config
                                                          qkvw-gpu qkvb-gpu
                                                          attprojw-gpu attprojb-gpu
                                                          ln1w-gpu ln1b-gpu
                                                          ln2w-gpu ln2b-gpu
                                                          fcw-gpu fcb-gpu
                                                          fcprojw-gpu fcprojb-gpu)]
                                    (when (> l 0) (release x))
                                    (recur (inc l) output (conj caches cache)))
                                  [x caches]))
                              
                              xL-copy (copy! xL-gpu (cuge (mrows xL-gpu) (ncols xL-gpu)))
                              xF-gpu (kernels/layernorm! xL-gpu lnfw-gpu lnfb-gpu ln-eps)
                              xF-copy (copy! xF-gpu (cuge (mrows xF-gpu) (ncols xF-gpu)))
                              logits-gpu (gpu-mm/matmul-forward-gpu xF-gpu wte-gpu nil)]
                          
                          (release xL-gpu)
                          (release xF-gpu)
                          
                          {:logits logits-gpu
                           :cache {:layer-caches layer-caches
                                   :final-ln-input xL-copy
                                   :final-ln-output xF-copy}}))
                      xs-gpu)]
        
        (release wte-gpu) (release wpe-gpu)
        (release qkvw-gpu) (release qkvb-gpu)
        (release attprojw-gpu) (release attprojb-gpu)
        (release fcw-gpu) (release fcb-gpu)
        (release fcprojw-gpu) (release fcprojb-gpu)
        (release ln1w-gpu) (release ln1b-gpu)
        (release ln2w-gpu) (release ln2b-gpu)
        (release lnfw-gpu) (release lnfb-gpu)
        
        {:logits-gpu (mapv :logits results)
         :cache-gpu (mapv :cache results)}))))

(comment
  (require '[llm.neo.model :as model])
  (require '[llm.neo.gpu.core :as gpu])
  
  (gpu/initialize-gpu)
  (def m (model/create-model :micro))
  (def tokens [[1 2 3 4]])
  
  (when (gpu/gpu-available?)
    (def result (gpt2-forward-gpu tokens (:config m) (:params m)))
    ;; Release when done
    (doseq [logits (:logits-gpu result)]
      (release logits))
    ;; Release caches...
    )
  )
