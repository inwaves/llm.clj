(ns llm.neo.encoder
  "Token and position embedding encoder using Neanderthal."
  (:use [uncomplicate.neanderthal core native])
  (:require [llm.neo.core :as neo]))

(defn encoder-forward
  "Forward pass for token + position embedding.
  
  Args:
    inp - Token indices [B, T] as nested vector of integers
    wte - Token embedding weights [V, C] as nested vector
    wpe - Position embedding weights [maxT, C] as nested vector
    
  Returns:
    Encoded output [B, T, C] as nested vector"
  [inp wte wpe]
  (let [B (count inp)
        T (count (first inp))
        C (count (first wte))
        wte-mat (neo/vec->matrix wte)
        wpe-mat (neo/vec->matrix wpe)
        output (vec (for [b (range B)]
                     (let [batch-out (dge T C)]
                       (doseq [t (range T)]
                         (let [token-idx (get-in inp [b t])]
                           (doseq [c (range C)]
                             (entry! batch-out t c 
                                    (+ (entry wte-mat token-idx c)
                                       (entry wpe-mat t c))))))
                       (neo/matrix->vec batch-out))))]
    output))

(defn encoder-forward-matrices
  "Forward pass returning Neanderthal matrices.
  
  Args:
    inp - Token indices [B, T]
    wte-mat - Token embeddings [V, C] as matrix
    wpe-mat - Position embeddings [maxT, C] as matrix
    
  Returns:
    Vector of B matrices [T, C]"
  [inp wte-mat wpe-mat]
  (let [B (count inp)
        T (count (first inp))
        C (ncols wte-mat)]
    (vec (for [b (range B)]
           (let [batch-out (dge T C)]
             (doseq [t (range T)]
               (let [token-idx (get-in inp [b t])]
                 (doseq [c (range C)]
                   (entry! batch-out t c
                          (+ (entry wte-mat token-idx c)
                             (entry wpe-mat t c))))))
             batch-out)))))

(defn encoder-backward
  "Backward pass for encoder (token + position embeddings).
  
  Scatters gradients from dx to:
  - wte: based on token indices (scatter-add)
  - wpe: based on positions (accumulate)
  
  Args:
    dx: [B, T, C] gradient w.r.t. encoder output (nested vectors)
    tokens: [B, T] token indices (nested vectors)
    vocab-size: V
    max-seq-len: maximum sequence length
    channels: C
    
  Returns:
    {:dwte [V, C] nested vector of gradients for token embeddings
     :dwpe [maxT, C] nested vector of gradients for position embeddings}"
  [dx tokens vocab-size max-seq-len channels]
  (let [B (count dx)
        ;; Initialize zero gradients as nested vectors
        dwte (vec (repeat vocab-size (vec (repeat channels 0.0))))
        dwpe (vec (repeat max-seq-len (vec (repeat channels 0.0))))]
    
    ;; Accumulate gradients for each batch
    (loop [b 0
           wte-grad dwte
           wpe-grad dwpe]
      (if (< b B)
        (let [dx-b (nth dx b)
              tokens-b (nth tokens b)
              T (count tokens-b)
              ;; Accumulate gradients for this sequence
              [wte-g wpe-g]
              (loop [t 0
                     wte-g wte-grad
                     wpe-g wpe-grad]
                (if (< t T)
                  (let [token (nth tokens-b t)
                        dx-bt (nth dx-b t)
                        ;; Scatter-add to wte[token]: wte_grad[token] += dx[b, t, :]
                        wte-row (nth wte-g token)
                        wte-row-new (mapv + wte-row dx-bt)
                        wte-g-new (assoc wte-g token wte-row-new)
                        ;; Accumulate to wpe[t]: wpe_grad[t] += dx[b, t, :]
                        wpe-row (nth wpe-g t)
                        wpe-row-new (mapv + wpe-row dx-bt)
                        wpe-g-new (assoc wpe-g t wpe-row-new)]
                    (recur (inc t) wte-g-new wpe-g-new))
                  [wte-g wpe-g]))]
          (recur (inc b) wte-g wpe-g))
        {:dwte wte-grad :dwpe wpe-grad}))))