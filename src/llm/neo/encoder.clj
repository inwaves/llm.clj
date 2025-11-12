(ns llm.neo.encoder
  "Token and position embedding encoder using Neanderthal.
  
  The encoder:
  1. Looks up token embeddings from wte based on input indices
  2. Looks up position embeddings from wpe based on positions  
  3. Adds them element-wise to produce encoded representations"
  (:require [uncomplicate.neanderthal
             [core :refer [mrows ncols entry entry!]]
             [native :refer [dge]]]
            [llm.neo.core :as neo]))

(defn encoder-forward
  "Forward pass for token + position embedding.
  
  Args:
    inp - Token indices [B, T] as nested vector of integers
    wte - Token embedding weights [V, C] as nested vector
    wpe - Position embedding weights [maxT, C] as nested vector
    
  Returns:
    Encoded output [B, T, C] as nested vector where each position
    contains the sum of token and position embeddings"
  [inp wte wpe]
  (let [B (count inp)
        T (count (first inp))
        C (count (first wte))
        
        ;; Convert embeddings to matrices for efficient row access
        wte-mat (neo/vec->matrix wte)
        wpe-mat (neo/vec->matrix wpe)
        
        ;; Create output as 3D tensor (one matrix per batch element)
        output (vec (for [b (range B)]
                     (let [batch-out (dge T C)]
                       ;; For each position in this batch element
                       (doseq [t (range T)]
                         (let [;; Get token index and position
                               token-idx (get-in inp [b t])]
                               
                           ;; Add token embedding and position embedding
                           ;; We copy token embedding to output, then add position embedding
                           (doseq [c (range C)]
                             (entry! batch-out t c 
                                    (+ (entry wte-mat token-idx c)
                                       (entry wpe-mat t c))))))
                       ;; Convert matrix back to nested vector for this batch element
                       (neo/matrix->vec batch-out))))]
    output))

(defn encoder-forward-matrices
  "Forward pass returning Neanderthal matrices (more efficient for pipelines).
  
  Args:
    inp - Token indices [B, T] as nested vector of integers
    wte-mat - Token embedding weights [V, C] as Neanderthal matrix
    wpe-mat - Position embedding weights [maxT, C] as Neanderthal matrix
    
  Returns:
    Vector of B matrices, each [T, C]"
  [inp wte-mat wpe-mat]
  (let [B (count inp)
        T (count (first inp))
        C (ncols wte-mat)]
    
    ;; Create one matrix per batch element
    (vec (for [b (range B)]
           (let [batch-out (dge T C)]
             ;; For each position
             (doseq [t (range T)]
               (let [token-idx (get-in inp [b t])]
                 ;; Add token and position embeddings
                 (doseq [c (range C)]
                   (entry! batch-out t c
                          (+ (entry wte-mat token-idx c)
                             (entry wpe-mat t c))))))
             batch-out)))))

(defn encoder-backward
  "Backward pass for encoder.
  
  Args:
    dout - Gradient of loss w.r.t. output [B, T, C] as nested vector
    inp - Token indices from forward pass [B, T] as nested vector
    vocab-size - Size of vocabulary (V)
    max-T - Maximum sequence length
    
  Returns:
    Map with:
      :dwte - Gradient w.r.t. token embeddings [V, C]
      :dwpe - Gradient w.r.t. position embeddings [maxT, C]"
  [dout inp vocab-size max-T]
  (let [B (count dout)
        T (count (first dout))
        C (count (first (first dout)))
        
        ;; Initialize gradient matrices (zero-initialized in JVM)
        dwte (dge vocab-size C)
        dwpe (dge max-T C)]
    
    ;; Accumulate gradients for each batch and position
    (doseq [b (range B)
            t (range T)]
      (let [token-idx (get-in inp [b t])
            grad (get-in dout [b t])]
        ;; Add gradient to token embedding
        (doseq [c (range C)]
          (entry! dwte token-idx c
                  (+ (entry dwte token-idx c) (get grad c))))
        ;; Add gradient to position embedding
        (doseq [c (range C)]
          (entry! dwpe t c
                  (+ (entry dwpe t c) (get grad c))))))
    
    {:dwte (neo/matrix->vec dwte)
     :dwpe (neo/matrix->vec dwpe)}))