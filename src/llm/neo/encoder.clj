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
  "Backward pass for encoder.
  
  Args:
    dx-encoded - Gradient [B, T, C] as nested vector
    inputs - [B, T] token indices  
    vocab-size - V
    max-seq-len - maxT
    channels - C
    
  Returns:
    {:dwte [V, C] :dwpe [maxT, C]} as nested vectors"
  [dx-encoded inputs vocab-size max-seq-len channels]
  (let [B (count inputs)
        T (count (first inputs))
        dwte (vec (repeat vocab-size (vec (repeat channels 0.0))))
        dwpe (vec (repeat max-seq-len (vec (repeat channels 0.0))))]
    
    (loop [b 0 dwte-acc dwte dwpe-acc dwpe]
      (if (< b B)
        (let [tokens (nth inputs b)
              dx-batch (nth dx-encoded b)]
          (recur (inc b)
                 (loop [t 0 dwte2 dwte-acc]
                   (if (< t T)
                     (let [token-id (nth tokens t)
                           grad-vec (nth dx-batch t)]
                       (recur (inc t)
                              (update dwte2 token-id (fn [old] (mapv + old grad-vec)))))
                     dwte2))
                 (loop [t 0 dwpe2 dwpe-acc]
                   (if (< t T)
                     (let [grad-vec (nth dx-batch t)]
                       (recur (inc t)
                              (update dwpe2 t (fn [old] (mapv + old grad-vec)))))
                     dwpe2))))
        {:dwte dwte-acc
         :dwpe dwpe-acc}))))