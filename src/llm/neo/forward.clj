(ns llm.neo.forward
  "GPT-2 forward pass using Neanderthal-backed operations.

  This composes:
  - Encoder (token + position embeddings)
  - L blocks of: LN -> masked multi-head self-attn -> residual -> LN -> MLP -> residual
  - Final LN
  - Output projection (tied to token embeddings wte)

  Shapes:
  - inp: [B, T] ints (token indices)
  - wte: [V, C]
  - wpe: [maxT, C]
  - internal tensors per batch are Neanderthal matrices of shape [T, C]
  - logits per batch are [T, V]
  - final return (gpt2-forward): [B, T, V] nested vectors"
  (:use [uncomplicate.neanderthal core native])
  (:require
    [llm.neo.core :as neo]
    [llm.neo.encoder :as enc]
    [llm.neo.layernorm :as ln]
    [llm.neo.gelu :as gelu]
    [llm.neo.matmul :as mm]
    [llm.neo.residual :as res]
    [llm.neo.model :as model]
    [llm.neo.attention :as attn]))

;; ============================================================================
;; Helpers: slicing
;; ============================================================================

(defn- slice-rows->dge
  "Copy a set of consecutive rows from mat into a new dge with row-count rows.
  start-row inclusive, row-count number of rows."
  [mat start-row row-count]
  (let [cols (ncols mat)
        out (dge row-count cols)]
    (dotimes [i row-count]
      (dotimes [j cols]
        (entry! out i j (entry mat (+ start-row i) j))))
    out))

(defn- subvec->dv
  "Convert a subvector of v [start, start+len) into a Neanderthal vector (dv)."
  [v start len]
  (dv (subvec v start (+ start len))))



;; ============================================================================
;; Per-layer block forward
;; ============================================================================

(def ^:private ln-eps 1e-5)

;; ============================================================================
;; Forward pass with activation caching for backpropagation
;; ============================================================================

(defn- block-forward-cached
  "One transformer block with activation caching.
  
  Uses the standalone attention module from attention.clj.
  
  Returns:
    {:output [T, C]
     :cache map of cached activations}"
  [x l params cfg qkvw-m attprojw-m fcw-m fcprojw-m]
  (let [{:keys [num-heads channels]} cfg
        C channels
        gamma1 (dv (nth (:ln1w params) l))
        beta1  (dv (nth (:ln1b params) l))
        gamma2 (dv (nth (:ln2w params) l))
        beta2  (dv (nth (:ln2b params) l))
        qkv-w  (slice-rows->dge qkvw-m (* l 3 C) (* 3 C))
        qkv-b  (subvec->dv (:qkvb params) (* l 3 C) (* 3 C))
        att-w  (slice-rows->dge attprojw-m (* l C) C)
        att-b  (subvec->dv (:attprojb params) (* l C) C)
        fc1-w  (slice-rows->dge fcw-m (* l 4 C) (* 4 C))
        fc1-b  (subvec->dv (:fcb params) (* l 4 C) (* 4 C))
        fc2-w  (slice-rows->dge fcprojw-m (* l C) C)
        fc2-b  (subvec->dv (:fcprojb params) (* l C) C)
        
        x-input (copy! x (dge (mrows x) (ncols x)))
        
        x-norm1 (ln/layernorm-forward x gamma1 beta1 ln-eps)
        ln1-output (copy! x-norm1 (dge (mrows x-norm1) (ncols x-norm1)))
        ;; Project to QKV [T, 3C]
        qkv (mm/matmul-forward x-norm1 qkv-w qkv-b)
        qkv-vec (neo/matrix->vec qkv)  ; Convert [T, 3C] matrix to nested vector
        
        ;; Use standalone attention module
        ;; Wrap as [B=1, T, 3C] for batch-oriented interface
        qkv-nested [qkv-vec]  ; Single element vector for batch dimension
        attn-result (llm.neo.attention/attention-forward-cached qkv-nested num-heads)
        
        ;; Extract results from [B=1, T, C] format
        attn-out-nested (first (:output attn-result))  ; Get batch 0: [T, C]
        attn-out-before-proj (neo/vec->matrix attn-out-nested)  ; Back to matrix [T, C]
        attn-cache (first (:cache attn-result))
        
        attn-out (mm/matmul-forward attn-out-before-proj att-w att-b)
        attn-output (copy! attn-out (dge (mrows attn-out) (ncols attn-out)))
        
        x-res1 (res/residual-forward x attn-out)
        res1-output (copy! x-res1 (dge (mrows x-res1) (ncols x-res1)))
        
        x-norm2 (ln/layernorm-forward x-res1 gamma2 beta2 ln-eps)
        ln2-output (copy! x-norm2 (dge (mrows x-norm2) (ncols x-norm2)))
        
        fc-up (mm/matmul-forward x-norm2 fc1-w fc1-b)
        fc-up-cached (copy! fc-up (dge (mrows fc-up) (ncols fc-up)))
        act (gelu/gelu-forward fc-up)
        gelu-output (copy! act (dge (mrows act) (ncols act)))
        fc-down (mm/matmul-forward act fc2-w fc2-b)
        
        x-out (res/residual-forward x-res1 fc-down)]
    
    {:output x-out
     :cache {:x-input x-input
             :ln1-input x-input
             :ln1-output ln1-output
             :qkv-matrix qkv
             :attn-cache attn-cache
             :attn-out-before-proj attn-out-before-proj
             :attn-output attn-output
             :res1-output res1-output
             :ln2-input res1-output
             :ln2-output ln2-output
             :fc-up fc-up-cached
             :gelu-output gelu-output}}))

(defn- block-forward
  "One transformer block: LN -> MHA -> residual -> LN -> MLP -> residual

  Uses the standalone attention module from attention.clj.

  Args:
    x            - [T, C]
    l            - layer index
    params       - ParameterTensors
    cfg          - GPT2Config

  Returns:
    x' - [T, C]"
  [x l params cfg
   qkvw-m attprojw-m fcw-m fcprojw-m]
  (let [{:keys [num-heads channels]} cfg
        C channels
        gamma1 (dv (nth (:ln1w params) l))
        beta1  (dv (nth (:ln1b params) l))
        gamma2 (dv (nth (:ln2w params) l))
        beta2  (dv (nth (:ln2b params) l))
        qkv-w  (slice-rows->dge qkvw-m (* l 3 C) (* 3 C))
        qkv-b  (subvec->dv (:qkvb params) (* l 3 C) (* 3 C))
        att-w  (slice-rows->dge attprojw-m (* l C) C)
        att-b  (subvec->dv (:attprojb params) (* l C) C)
        fc1-w  (slice-rows->dge fcw-m (* l 4 C) (* 4 C))
        fc1-b  (subvec->dv (:fcb params) (* l 4 C) (* 4 C))
        fc2-w  (slice-rows->dge fcprojw-m (* l C) C)
        fc2-b  (subvec->dv (:fcprojb params) (* l C) C)]

    (let [x-norm1 (ln/layernorm-forward x gamma1 beta1 ln-eps)
          qkv (mm/matmul-forward x-norm1 qkv-w qkv-b)
          qkv-vec (neo/matrix->vec qkv)
          qkv-nested [qkv-vec]
          attn-out-nested (first (llm.neo.attention/attention-forward qkv-nested num-heads))
          attn-out-before-proj (neo/vec->matrix attn-out-nested)
          attn-out (mm/matmul-forward attn-out-before-proj att-w att-b)
          x-res1 (res/residual-forward x attn-out)
          x-norm2 (ln/layernorm-forward x-res1 gamma2 beta2 ln-eps)
          fc-up   (mm/matmul-forward x-norm2 fc1-w fc1-b)
          act     (gelu/gelu-forward fc-up)
          fc-down (mm/matmul-forward act fc2-w fc2-b)
          x-out   (res/residual-forward x-res1 fc-down)]
      x-out)))

;; ============================================================================
;; Full model forward
;; ============================================================================

(defn gpt2-forward-matrices
  "Forward pass returning Neanderthal matrices.

  Args:
    inp    - [B, T] nested vector of token ids
    config - GPT2Config
    params - ParameterTensors

  Returns:
    Vector of B matrices, each [T, V] logits."
  [inp config params]
  (let [{:keys [vocab-size max-seq-len num-layers channels]} config
        C channels
        L num-layers
        V vocab-size
        ;; Convert main embeddings to matrices
        wte-m (neo/vec->matrix (:wte params))        ;; [V, C]
        wpe-m (neo/vec->matrix (:wpe params))        ;; [maxT, C]
        ;; Aggregated per-stack weights to matrices
        qkvw-m     (neo/vec->matrix (:qkvw params))      ;; [L*3C, C]
        attprojw-m (neo/vec->matrix (:attprojw params))  ;; [L*C, C]
        fcw-m      (neo/vec->matrix (:fcw params))       ;; [L*4C, C]
        fcprojw-m  (neo/vec->matrix (:fcprojw params))   ;; [L*C, 4C]
        ;; Encoder: B of [T, C]
        xs (enc/encoder-forward-matrices inp wte-m wpe-m)
        ;; Final LN params
        lnfw (dv (:lnfw params))
        lnfb (dv (:lnfb params))]
    ;; Process each batch independently (attention is per sample)
    (mapv
      (fn [x0]
        (let [T (mrows x0)
              _ (assert (<= T max-seq-len)
                        (str "Sequence length " T " exceeds max " max-seq-len))]
          ;; Stack of blocks
          (let [xL (loop [l 0, x x0]
                     (if (< l L)
                       (recur (inc l) (block-forward x l params config
                                                     qkvw-m attprojw-m fcw-m fcprojw-m))
                       x))
                ;; Final LN
                xF (ln/layernorm-forward xL lnfw lnfb ln-eps)
                ;; Output projection tied to wte: logits = xF @ wte^T
                logits (mm/matmul-forward xF wte-m nil)]
            logits)))
      xs)))

(defn gpt2-forward
  "Forward pass returning nested vectors [B, T, V].

  Args:
    inp      - [B, T] nested vector of token ids
    model    - ModelState (from llm.neo.model/create-model) or map with :config and :params
               Alternatively, pass config and params explicitly with keys :config, :params.

  Returns:
    Nested vector logits of shape [B, T, V]."
  [inp model]
  (let [{:keys [config params]} (if (and (map? model) (:config model) (:params model))
                                  model
                                  (throw (ex-info "Model must provide :config and :params"
                                                  {:provided (keys model)})))
        mats (gpt2-forward-matrices inp config params)]
    (mapv neo/matrix->vec mats)))

;; ============================================================================
;; Convenience wrapper when you have ModelState record
;; ============================================================================

(defn forward
  "Alias for gpt2-forward when given a ModelState record."
  [inp state]
  (gpt2-forward inp {:config (:config state) :params (:params state)}))

(defn gpt2-forward-with-cache
  "Forward pass that returns both logits and cached activations.
  
  Args:
    inp    - [B, T] nested vector of token ids
    config - GPT2Config
    params - ParameterTensors
  
  Returns:
    {:logits [B, T, V] nested vectors
     :cache vector of B cache maps}"
  [inp config params]
  (let [{:keys [vocab-size max-seq-len num-layers channels]} config
        C channels
        L num-layers
        wte-m (neo/vec->matrix (:wte params))
        wpe-m (neo/vec->matrix (:wpe params))
        qkvw-m (neo/vec->matrix (:qkvw params))
        attprojw-m (neo/vec->matrix (:attprojw params))
        fcw-m (neo/vec->matrix (:fcw params))
        fcprojw-m (neo/vec->matrix (:fcprojw params))
        xs (enc/encoder-forward-matrices inp wte-m wpe-m)
        lnfw (dv (:lnfw params))
        lnfb (dv (:lnfb params))
        
        results (mapv
                  (fn [x0]
                    (let [T (mrows x0)
                          _ (assert (<= T max-seq-len))
                          
                          [xL layer-caches]
                          (loop [l 0, x x0, caches []]
                            (if (< l L)
                              (let [{:keys [output cache]} 
                                    (block-forward-cached x l params config
                                                         qkvw-m attprojw-m fcw-m fcprojw-m)]
                                (recur (inc l) output (conj caches cache)))
                              [x caches]))
                          
                          xL-copy (copy! xL (dge (mrows xL) (ncols xL)))
                          
                          xF (ln/layernorm-forward xL lnfw lnfb ln-eps)
                          xF-copy (copy! xF (dge (mrows xF) (ncols xF)))
                          
                          logits (mm/matmul-forward xF wte-m nil)]
                      
                      {:logits (neo/matrix->vec logits)
                       :cache {:layer-caches layer-caches
                               :final-ln-input xL-copy
                               :final-ln-output xF-copy}}))
                  xs)]
    
    {:logits (mapv :logits results)
     :cache (mapv :cache results)
     :inputs inp}))