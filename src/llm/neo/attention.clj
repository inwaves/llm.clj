(ns llm.neo.attention
  "Multi-head attention using Neanderthal with forward and backward passes."
  (:require [uncomplicate.neanderthal.core :refer [mrows ncols entry entry! mm! trans]]
            [uncomplicate.neanderthal.native :as nn]
            [llm.neo.core :as neo]
            [llm.neo.softmax :as softmax]))

;; ============================================================================
;; Helper Functions
;; ============================================================================

(defn- split-qkv
  "Split concatenated QKV matrix [T, 3C] into three matrices [T, C] each."
  [qkv]
  (let [rows (mrows qkv)
        c3 (ncols qkv)]
    (when (not (zero? (mod c3 3)))
      (throw (ex-info "Input last dimension must be divisible by 3 (Q,K,V concatenation)"
                      {:cols c3})))
    (let [c (quot c3 3)
          q (nn/dge rows c)
          k (nn/dge rows c)
          v (nn/dge rows c)]
      (doseq [i (range rows)
              j (range c)]
        (entry! q i j (entry qkv i j))
        (entry! k i j (entry qkv i (+ c j)))
        (entry! v i j (entry qkv i (+ (* 2 c) j))))
      {:q q :k k :v v})))

(defn- slice-cols->dge
  "Extract columns [start, start+count) from matrix m."
  [m start count]
  (let [rows (mrows m)
        out (nn/dge rows count)]
    (dotimes [i rows]
      (dotimes [j count]
        (entry! out i j (entry m i (+ start j)))))
    out))

(defn- scale-matrix!
  "Scale all elements of matrix m by factor a in-place."
  [m a]
  (let [rows (mrows m)
        cols (ncols m)]
    (dotimes [i rows]
      (dotimes [j cols]
        (entry! m i j (* a (entry m i j)))))
    m))

(defn- compute-attention-head
  "Compute single-head attention. Returns [T, HS]."
  [Q K V T HS]
  (let [scale (/ 1.0 (Math/sqrt (double HS)))
        scores (nn/dge T T)
        _ (mm! scale Q (trans K) 0.0 scores)
        att-probs (softmax/softmax-autoregressive scores)
        out (nn/dge T HS)]
    (mm! 1.0 att-probs V 0.0 out)
    out))

;; ============================================================================
;; Forward Pass
;; ============================================================================

(defn attention-forward
  "Multi-head attention forward pass.
  
  Args:
    inp - Nested vector [B, T, 3C] (Q, K, V concatenated)
    NH - Number of attention heads
    
  Returns:
    Nested vector [B, T, C]"
  [inp NH]
  (when (or (nil? NH) (<= NH 0))
    (throw (ex-info "NH (number of heads) must be a positive integer" {:NH NH})))
  (let [B (count inp)
        T (count (nth inp 0))
        C3 (count (nth (nth inp 0) 0))]
    (when (not (zero? (mod C3 3)))
      (throw (ex-info "Last dimension must be 3*C (Q,K,V concatenated)" {:C3 C3})))
    (let [C (quot C3 3)]
      (when (not (zero? (mod C NH)))
        (throw (ex-info "C must be divisible by NH" {:C C :NH NH})))
      (let [HS (quot C NH)]
        (vec
         (for [b (range B)]
           (let [batch-inp (neo/vec->matrix (nth inp b))
                 {:keys [q k v]} (split-qkv batch-inp)
                 batch-out (nn/dge T C)]
             (doseq [h (range NH)]
               (let [col0 (* h HS)
                     Q-h (slice-cols->dge q col0 HS)
                     K-h (slice-cols->dge k col0 HS)
                     V-h (slice-cols->dge v col0 HS)
                     O-h (compute-attention-head Q-h K-h V-h T HS)]
                 (doseq [t (range T)
                         i (range HS)]
                   (entry! batch-out t (+ col0 i) (entry O-h t i)))))
             (neo/matrix->vec batch-out))))))))

(defn attention-forward-cached
  "Multi-head attention forward that caches for backward.
  
  Returns:
    {:output [B, T, C]
     :cache vector of B maps with :q :k :v :att-probs}"
  [inp NH]
  (when (or (nil? NH) (<= NH 0))
    (throw (ex-info "NH must be positive" {:NH NH})))
  (let [B (count inp)
        T (count (nth inp 0))
        C3 (count (nth (nth inp 0) 0))]
    (when (not (zero? (mod C3 3)))
      (throw (ex-info "Last dimension must be 3*C" {:C3 C3})))
    (let [C (quot C3 3)]
      (when (not (zero? (mod C NH)))
        (throw (ex-info "C must be divisible by NH" {:C C :NH NH})))
      (let [HS (quot C NH)
            outputs (transient [])
            caches (transient [])]
        (doseq [b (range B)]
          (let [batch-inp (neo/vec->matrix (nth inp b))
                {:keys [q k v]} (split-qkv batch-inp)
                batch-out (nn/dge T C)
                head-probs (transient [])]
            (doseq [h (range NH)]
              (let [col0 (* h HS)
                    Q-h (slice-cols->dge q col0 HS)
                    K-h (slice-cols->dge k col0 HS)
                    V-h (slice-cols->dge v col0 HS)
                    scale (/ 1.0 (Math/sqrt (double HS)))
                    scores (nn/dge T T)
                    _ (mm! scale Q-h (trans K-h) 0.0 scores)
                    P-h (softmax/softmax-autoregressive scores)
                    O-h (nn/dge T HS)
                    _ (mm! 1.0 P-h V-h 0.0 O-h)]
                (conj! head-probs P-h)
                (doseq [t (range T)
                        i (range HS)]
                  (entry! batch-out t (+ col0 i) (entry O-h t i)))))
            (conj! outputs (neo/matrix->vec batch-out))
            (conj! caches {:q q :k k :v v :att-probs (persistent! head-probs)})))
        {:output (persistent! outputs)
         :cache (persistent! caches)}))))

;; ============================================================================
;; Backward Pass
;; ============================================================================

(defn attention-backward
  "Multi-head attention backward pass.
  
  Args:
    dout - Gradient [B, T, C] nested vector
    cache - Forward cache from attention-forward-cached
    NH - Number of heads
    
  Returns:
    Gradient [B, T, 3C] nested vector"
  [dout cache NH]
  (when (or (nil? NH) (<= NH 0))
    (throw (ex-info "NH must be positive" {:NH NH})))
  (let [B (count dout)
        T (count (first dout))
        C (count (first (first dout)))]
    (when (not (zero? (mod C NH)))
      (throw (ex-info "C not divisible by NH" {:C C :NH NH})))
    (let [HS (quot C NH)
          scale (/ 1.0 (Math/sqrt (double HS)))]
      
      (vec
       (for [b (range B)]
         (let [dout-b (neo/vec->matrix (nth dout b))
               {:keys [q k v att-probs]} (nth cache b)
               ;; Zero-initialized via JVM default (nn/dge without data)
               dQ (nn/dge T C)
               dK (nn/dge T C)
               dV (nn/dge T C)]
           
           ;; Process each head
           (doseq [h (range NH)]
             (let [col0 (* h HS)
                   Q-h (slice-cols->dge q col0 HS)
                   K-h (slice-cols->dge k col0 HS)
                   V-h (slice-cols->dge v col0 HS)
                   P-h (nth att-probs h)
                   dO-h (slice-cols->dge dout-b col0 HS)
                   
                   ;; Backward: O = P @ V
                   dP (nn/dge T T)
                   _ (mm! 1.0 dO-h (trans V-h) 0.0 dP)
                   dV-h (nn/dge T HS)
                   _ (mm! 1.0 (trans P-h) dO-h 0.0 dV-h)
                   
                   ;; Softmax backward
                   dS (softmax/softmax-autoregressive-backward P-h dP)
                   _ (scale-matrix! dS scale)
                   
                   ;; Backward: S = Q @ K^T
                   dQ-h (nn/dge T HS)
                   _ (mm! 1.0 dS K-h 0.0 dQ-h)
                   dK-h (nn/dge T HS)
                   _ (mm! 1.0 (trans dS) Q-h 0.0 dK-h)]
               
               ;; Direct assignment (ranges don't overlap)
               (doseq [t (range T)
                       i (range HS)]
                 (entry! dQ t (+ col0 i) (entry dQ-h t i))
                 (entry! dK t (+ col0 i) (entry dK-h t i))
                 (entry! dV t (+ col0 i) (entry dV-h t i)))))
           
           ;; Concatenate dQ, dK, dV -> [T, 3C]
           (let [dQKV (nn/dge T (* 3 C))]
             (doseq [t (range T)
                     c (range C)]
               (entry! dQKV t c (entry dQ t c))
               (entry! dQKV t (+ C c) (entry dK t c))
               (entry! dQKV t (+ (* 2 C) c) (entry dV t c)))
             (neo/matrix->vec dQKV))))))))