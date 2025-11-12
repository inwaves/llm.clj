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
    [llm.neo.softmax :as sm]
    [llm.neo.model :as model]))

;; ============================================================================
;; Helpers: slicing, splitting, concatenation
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

(defn- slice-cols->dge
  "Copy a set of consecutive columns from mat into a new dge with col-count columns.
  start-col inclusive, col-count number of columns."
  [mat start-col col-count]
  (let [rows (mrows mat)
        out (dge rows col-count)]
    (dotimes [i rows]
      (dotimes [j col-count]
        (entry! out i j (entry mat i (+ start-col j)))))
    out))

(defn- subvec->dv
  "Convert a subvector of v [start, start+len) into a Neanderthal vector (dv)."
  [v start len]
  (dv (subvec v start (+ start len))))

(defn- concat-cols
  "Concatenate a vector of matrices with same number of rows along the column dimension.
  Returns a new matrix with total columns equal to sum of columns."
  [mats]
  (let [rows (mrows (first mats))
        total-cols (reduce + (map ncols mats))
        out (dge rows total-cols)]
    (loop [col-off 0, ms mats]
      (when (seq ms)
        (let [m (first ms)
              cols (ncols m)]
          (dotimes [i rows]
            (dotimes [j cols]
              (entry! out i (+ col-off j) (entry m i j))))
          (recur (+ col-off cols) (rest ms)))))
    out))

(defn- split-cols-even
  "Split matrix m into k chunks along columns, evenly sized (assumes divisible).
  Returns vector of k matrices."
  [m k]
  (let [cols (ncols m)
        _ (assert (zero? (mod cols k))
                  (str "Columns " cols " not divisible by " k))
        each (quot cols k)]
    (mapv (fn [i] (slice-cols->dge m (* i each) each)) (range k))))

(defn- qkv-split
  "Split a [T, 3C] packed QKV matrix into [Q, K, V], each [T, C]."
  [qkv C]
  (let [Q (slice-cols->dge qkv 0 C)
        K (slice-cols->dge qkv C C)
        V (slice-cols->dge qkv (* 2 C) C)]
    [Q K V]))

(defn- scale-matrix!
  "In-place scale of every element of matrix m by factor a. Returns m."
  [m a]
  (let [rows (mrows m)
        cols (ncols m)]
    (dotimes [i rows]
      (dotimes [j cols]
        (entry! m i j (* a (entry m i j)))))
    m))

;; ============================================================================
;; Multi-head self-attention (masked/causal)
;; ============================================================================

(defn- mha-forward
  "Multi-head masked self-attention for a single batch element.

  Args:
    x      - input matrix [T, C]
    qkv-w  - weight matrix [3C, C]
    qkv-b  - bias vector dv [3C]
    proj-w - weight matrix [C, C]
    proj-b - bias vector dv [C]
    num-heads - number of attention heads

  Returns:
    attn-out: [T, C]"
  [x qkv-w qkv-b proj-w proj-b num-heads]
  (let [T (mrows x)
        C (ncols x)
        _ (assert (zero? (mod C num-heads))
                  (str "Channels " C " not divisible by heads " num-heads))
        head-d (quot C num-heads)
        ;; Project to packed QKV: [T, 3C]
        qkv (mm/matmul-forward x qkv-w qkv-b)
        [Q K V] (qkv-split qkv C)
        ;; Split into heads: vectors of k matrices, each [T, head-d]
        Qh (split-cols-even Q num-heads)
        Kh (split-cols-even K num-heads)
        Vh (split-cols-even V num-heads)
        ;; Per-head attention
        heads
        (mapv
          (fn [Qh_i Kh_i Vh_i]
            ;; Scores S = Q K^T / sqrt(d), shape [T, T]
            (let [S (dge T T)]
              (mm! 1.0 Qh_i (trans Kh_i) 0.0 S)
              (scale-matrix! S (/ 1.0 (Math/sqrt (double head-d))))
              ;; Masked softmax
              (let [P (sm/softmax-autoregressive S)
                    Oh (dge T head-d)]
                ;; O = P V
                (mm! 1.0 P Vh_i 0.0 Oh)
                Oh)))
          Qh Kh Vh)
        ;; Concat heads -> [T, C]
        O (concat-cols heads)
        ;; Output projection -> [T, C]
        attn-out (mm/matmul-forward O proj-w proj-b)]
    attn-out))

;; ============================================================================
;; Per-layer block forward
;; ============================================================================

(def ^:private ln-eps 1e-5)

(defn- block-forward
  "One transformer block: LN -> MHA -> residual -> LN -> MLP -> residual

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
        ;; LayerNorm 1 params (vectors of length C)
        gamma1 (dv (nth (:ln1w params) l))
        beta1  (dv (nth (:ln1b params) l))
        ;; LayerNorm 2 params
        gamma2 (dv (nth (:ln2w params) l))
        beta2  (dv (nth (:ln2b params) l))
        ;; Sliced weights/biases for this layer
        ;; qkv: rows [l*3C .. l*3C + 3C)
        qkv-w  (slice-rows->dge qkvw-m (* l 3 C) (* 3 C))
        qkv-b  (subvec->dv (:qkvb params) (* l 3 C) (* 3 C))
        ;; attention proj: rows [l*C .. + C)
        att-w  (slice-rows->dge attprojw-m (* l C) C)
        att-b  (subvec->dv (:attprojb params) (* l C) C)
        ;; fc (up): rows [l*4C .. + 4C)
        fc1-w  (slice-rows->dge fcw-m (* l 4 C) (* 4 C))
        fc1-b  (subvec->dv (:fcb params) (* l 4 C) (* 4 C))
        ;; fc proj (down): rows [l*C .. + C), cols 4C
        fc2-w  (slice-rows->dge fcprojw-m (* l C) C)
        fc2-b  (subvec->dv (:fcprojb params) (* l C) C)]

    ;; LN1
    (let [x-norm1 (ln/layernorm-forward x gamma1 beta1 ln-eps)
          ;; MHA
          attn-out (mha-forward x-norm1 qkv-w qkv-b att-w att-b num-heads)
          ;; Residual 1
          x-res1 (res/residual-forward x attn-out)
          ;; LN2
          x-norm2 (ln/layernorm-forward x-res1 gamma2 beta2 ln-eps)
          ;; MLP: FC -> GELU -> FC
          fc-up   (mm/matmul-forward x-norm2 fc1-w fc1-b)
          act     (gelu/gelu-forward fc-up)
          fc-down (mm/matmul-forward act fc2-w fc2-b)
          ;; Residual 2
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
  [inp ^model/ModelState state]
  (gpt2-forward inp {:config (:config state) :params (:params state)}))