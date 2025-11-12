(ns llm.neo.attention
  "Multi-head attention using Neanderthal.

  Implements scaled dot-product attention with multiple heads:
  Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d_k)) @ V

  Applies autoregressive (causal) masking for language modeling.

  Shapes:
  - inp: [B, T, 3C] as nested vectors (Q, K, V concatenated on last dim)
  - out: [B, T, C] as nested vectors
  - NH: number of attention heads, HS = C / NH"
  (:require [uncomplicate.neanderthal
             [core :refer [mrows ncols entry entry! mm! trans]]
             [native :refer [dge]]]
            [llm.neo.core :as neo]
            [llm.neo.softmax :as softmax]))

(defn- split-qkv
  "Split concatenated QKV matrix [T, 3C] into three matrices [T, C] each.
  Returns {:q Q :k K :v V}."
  [qkv]
  (let [rows (mrows qkv)
        c3 (ncols qkv)]
    (when (not (zero? (mod c3 3)))
      (throw (ex-info "Input last dimension must be divisible by 3 (Q,K,V concatenation)"
                      {:cols c3})))
    (let [c (quot c3 3)
          q (dge rows c)
          k (dge rows c)
          v (dge rows c)]
      (doseq [i (range rows)
              j (range c)]
        (entry! q i j (entry qkv i j))
        (entry! k i j (entry qkv i (+ c j)))
        (entry! v i j (entry qkv i (+ (* 2 c) j))))
      {:q q :k k :v v})))

(defn- compute-attention-head
  "Compute single-head attention.
   Q, K, V: [T, HS] matrices
   Returns [T, HS]."
  [Q K V T HS]
  (let [scale (/ 1.0 (Math/sqrt (double HS)))
        scores (dge T T)
        _ (mm! scale Q (trans K) 0.0 scores)
        att-probs (softmax/softmax-autoregressive scores)
        out (dge T HS)]
    (mm! 1.0 att-probs V 0.0 out)
    out))

(defn attention-forward
  "Multi-head attention forward pass.

  Args:
    inp - Nested vector of shape [B, T, 3C] (Q, K, V concatenated)
    NH  - Number of attention heads (must divide C)

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
           (let [batch-inp (neo/vec->matrix (nth inp b)) ; [T, 3C]
                 {:keys [q k v]} (split-qkv batch-inp)
                 batch-out (dge T C)]
             (doseq [h (range NH)]
               (let [Q-h (dge T HS)
                     K-h (dge T HS)
                     V-h (dge T HS)
                     col0 (* h HS)]
                 ;; Extract head-specific slices
                 (doseq [t (range T)
                         i (range HS)]
                   (entry! Q-h t i (entry q t (+ col0 i)))
                   (entry! K-h t i (entry k t (+ col0 i)))
                   (entry! V-h t i (entry v t (+ col0 i))))
                 ;; Compute attention for this head
                 (let [head-out (compute-attention-head Q-h K-h V-h T HS)]
                   ;; Write back into [T, C] at head's column range
                   (doseq [t (range T)
                           i (range HS)]
                     (entry! batch-out t (+ col0 i) (entry head-out t i))))))
             (neo/matrix->vec batch-out))))))))