(ns llm.neo.gpu.attention
  "GPU multi-head self-attention using Neanderthal CUDA.

  Resource ownership and lifecycle:
  - Pure GPU functions (…-gpu) return GPU matrices. The caller OWNS and must release
    both outputs and all cached GPU resources.
  - Hybrid functions (…-hybrid) accept CPU data, compute on GPU, return CPU results,
    and:
      • Forward returns a cache containing LIVE GPU resources (q, k, v, att-probs).
        Forward does NOT release cached resources. The caller owns the cache.
      • Backward consumes the cache, computes gradients, transfers to CPU, and then
        releases ALL cached GPU resources and any temporary GPU allocations.
    This ensures no use-after-free and correct cache lifetime across forward/backward."
  (:require [uncomplicate.commons.core :refer [with-release release]]
            [uncomplicate.neanderthal.core :as ncore :refer [mrows ncols entry entry! mm! trans]]
            [llm.neo.gpu.core :as gpu]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Internal GPU Helpers
;; ============================================================================

(defn- cuge
  "Helper to allocate a CUDA matrix [rows cols]."
  [rows cols]
  (require '[uncomplicate.neanderthal.cuda :as cuda])
  ((resolve 'uncomplicate.neanderthal.cuda/cuge) rows cols))

(defn- softmax-autoregressive-gpu
  "Compute softmax with causal mask on GPU matrix scores [T, T].
   Returns a GPU matrix [T, T] with probabilities. Caller must release."
  [scores]
  (let [T (mrows scores)
        out (cuge T T)]
    (dotimes [i T]
      ;; Row max over all columns (matches CPU variant's behavior)
      (let [row-max (loop [j 0 mx Double/NEGATIVE_INFINITY]
                      (if (< j T)
                        (recur (inc j) (max mx (entry scores i j)))
                        mx))
            ;; First pass: exp(x - max) for causal positions and accumulate sum
            row-sum (loop [j 0 sum 0.0]
                      (if (< j T)
                        (if (<= j i)
                          (let [e (Math/exp (- (entry scores i j) row-max))]
                            (entry! out i j e)
                            (recur (inc j) (+ sum e)))
                          (do
                            (entry! out i j 0.0)
                            (recur (inc j) sum)))
                        sum))]
        ;; Normalize only valid positions
        (dotimes [j (inc i)]
          (entry! out i j (/ (entry out i j) row-sum)))))
    out))

(defn- softmax-autoregressive-backward-gpu
  "Backward for autoregressive softmax on GPU.
   p, dp: [T, T] cuge. Returns dx [T, T] cuge. Caller must release."
  [p dp]
  (let [T (mrows p)
        dx (cuge T T)]
    (dotimes [i T]
      ;; dot = sum_{j<=i} p[i,j]*dp[i,j]
      (let [dot (loop [j 0 sum 0.0]
                  (if (<= j i)
                    (recur (inc j) (+ sum (* (entry p i j) (entry dp i j))))
                    sum))]
        ;; dx[i,j] = p[i,j] * (dp[i,j] - dot) for j<=i; else 0
        (dotimes [j (inc i)]
          (entry! dx i j (* (entry p i j) (- (entry dp i j) dot))))
        (dotimes [j (- T i 1)]
          (entry! dx i (+ i 1 j) 0.0))))
    dx))

(defn- scale-matrix-gpu!
  "In-place scale of GPU matrix m by factor a. Returns m."
  [m a]
  (let [rows (mrows m)
        cols (ncols m)]
    (dotimes [i rows]
      (dotimes [j cols]
        (entry! m i j (* a (entry m i j)))))
    m))

(defn- split-qkv-gpu
  "Split concatenated QKV GPU matrix [T, 3C] into Q,K,V each [T, C].
   Returns {:q :k :v} with GPU matrices. Caller owns returned matrices."
  [qkv-gpu]
  (let [t (mrows qkv-gpu)
        c3 (ncols qkv-gpu)]
    (when (not (zero? (mod c3 3)))
      (throw (ex-info "Input last dimension must be divisible by 3 (Q,K,V concatenation)"
                      {:cols c3})))
    (let [c (quot c3 3)
          q (cuge t c)
          k (cuge t c)
          v (cuge t c)]
      (dotimes [i t]
        (dotimes [j c]
          (entry! q i j (entry qkv-gpu i j))
          (entry! k i j (entry qkv-gpu i (+ c j)))
          (entry! v i j (entry qkv-gpu i (+ (* 2 c) j)))))
      {:q q :k k :v v})))

(defn- slice-cols->cuge
  "Copy columns [start, start+count) from GPU matrix m into a new GPU matrix."
  [m start count]
  (let [rows (mrows m)
        out (cuge rows count)]
    (dotimes [i rows]
      (dotimes [j count]
        (entry! out i j (entry m i (+ start j)))))
    out))

;; ============================================================================
;; Pure GPU Forward/Backward (single sequence: [T, 3C] -> [T, C])
;; ============================================================================

(defn attention-forward-qkv-gpu
  "Forward pass for single-sequence self-attention on GPU.

  Args:
    qkv-gpu: cuge [T, 3C] containing concatenated Q, K, V.
    nh: number of heads.

  Returns:
    {:out-gpu cuge [T, C]
     :cache {:q cuge [T, C]
             :k cuge [T, C]
             :v cuge [T, C]
             :att-probs (vector of nh cuge [T, T])}}

  Ownership:
    - Caller owns :out-gpu and every GPU object in :cache and must release them.
    - This function does NOT wrap cached resources in with-release."
  [qkv-gpu nh]
  (let [t (mrows qkv-gpu)
        c3 (ncols qkv-gpu)]
    (when (or (nil? nh) (<= nh 0))
      (throw (ex-info "nh (number of heads) must be a positive integer" {:nh nh})))
    (when (not (zero? (mod c3 3)))
      (throw (ex-info "Last dimension must be 3*C (Q,K,V concatenated)" {:C3 c3})))
    (let [c (quot c3 3)]
      (when (not (zero? (mod c nh)))
        (throw (ex-info "C must be divisible by nh" {:C c :nh nh})))
      (let [hs (quot c nh)
            {:keys [q k v]} (split-qkv-gpu qkv-gpu)
            ;; concat-out must be returned INSIDE this let (Bug 1 fix)
            concat-out (cuge t c)
            head-probs (transient [])]
        (dotimes [h nh]
          (let [col0 (* h hs)
                ;; Extract per-head slices
                Q-h (slice-cols->cuge q col0 hs)
                K-h (slice-cols->cuge k col0 hs)
                V-h (slice-cols->cuge v col0 hs)
                ;; Compute scaled dot-product attention
                scale (/ 1.0 (Math/sqrt (double hs)))
                scores (cuge t t)]
            (mm! scale Q-h (trans K-h) 0.0 scores)
            (let [P-h (softmax-autoregressive-gpu scores)
                  O-h (cuge t hs)]
              (mm! 1.0 P-h V-h 0.0 O-h)
              ;; Save probabilities for backward
              (conj! head-probs P-h)
              ;; Write head output into concatenated output
              (dotimes [i t]
                (dotimes [j hs]
                  (entry! concat-out i (+ col0 j) (entry O-h i j))))
              ;; Release temporaries owned by this function scope
              (release O-h))
            (release scores)
            (release Q-h) (release K-h) (release V-h)))
        {:out-gpu concat-out
         :cache {:q q :k k :v v :att-probs (persistent! head-probs)}}))))

(defn attention-backward-qkv-gpu
  "Backward pass for single-sequence self-attention on GPU.

  Args:
    dout-gpu: cuge [T, C] gradient wrt attention output.
    cache: {:q :k :v :att-probs} returned by attention-forward-qkv-gpu.
    nh: number of heads.

  Returns:
    cuge [T, 3C] gradient wrt concatenated QKV.

  Ownership:
    - Caller owns returned GPU matrix and remains responsible for releasing
      cached resources separately (pure GPU function does not release cache)."
  [dout-gpu {:keys [q k v att-probs]} nh]
  (let [t (mrows q)
        c (ncols q)]
    (when (or (nil? nh) (<= nh 0))
      (throw (ex-info "nh (number of heads) must be a positive integer" {:nh nh})))
    (when (not (zero? (mod c nh)))
      (throw (ex-info "C not divisible by nh" {:C c :nh nh})))
    (let [hs (quot c nh)
          scale (/ 1.0 (Math/sqrt (double hs)))
          dQ (cuge t c)
          dK (cuge t c)
          dV (cuge t c)]
      (dotimes [h nh]
        (let [col0 (* h hs)
              Q-h (slice-cols->cuge q col0 hs)
              K-h (slice-cols->cuge k col0 hs)
              V-h (slice-cols->cuge v col0 hs)
              P-h (nth att-probs h)
              dO-h (slice-cols->cuge dout-gpu col0 hs)
              ;; dP = dO @ V^T
              dP (cuge t t)]
          (mm! 1.0 dO-h (trans V-h) 0.0 dP)
          ;; dV-h = P^T @ dO
          (let [dV-h (cuge t hs)]
            (mm! 1.0 (trans P-h) dO-h 0.0 dV-h)
            ;; dS = softmax_backward(P, dP), then scale for S = (Q K^T)/sqrt(HS)
            (let [dS (softmax-autoregressive-backward-gpu P-h dP)]
              (scale-matrix-gpu! dS scale)
              ;; dQ-h = dS @ K
              (let [dQ-h (cuge t hs)]
                (mm! 1.0 dS K-h 0.0 dQ-h)
                ;; dK-h = dS^T @ Q
                (let [dK-h (cuge t hs)]
                  (mm! 1.0 (trans dS) Q-h 0.0 dK-h)
                  ;; Scatter-add to dQ/dK/dV (no overlap -> direct assign)
                  (dotimes [i t]
                    (dotimes [j hs]
                      (entry! dQ i (+ col0 j) (entry dQ-h i j))
                      (entry! dK i (+ col0 j) (entry dK-h i j))
                      (entry! dV i (+ col0 j) (entry dV-h i j))))
                  (release dK-h))
                (release dQ-h))
              (release dS))
            (release dV-h))
          ;; Release temporaries bound in this head
          (release dP)
          (release dO-h)
          (release Q-h) (release K-h) (release V-h)))
      ;; Concatenate dQ,dK,dV into [T, 3C]
      (let [dQKV (cuge t (* 3 c))]
        (dotimes [i t]
          (dotimes [j c]
            (entry! dQKV i j (entry dQ i j))
            (entry! dQKV i (+ c j) (entry dK i j))
            (entry! dQKV i (+ (* 2 c) j) (entry dV i j))))
        ;; Release component grads after concatenation
        (release dQ) (release dK) (release dV)
        dQKV))))

;; ============================================================================
;; CPU Interface Wrappers (single sequence)
;; ============================================================================

(defn attention-forward-qkv-hybrid
  "Hybrid CPU↔GPU forward for single-sequence attention.

  Args:
    qkv: CPU nested vector [T, 3C]
    nh: number of heads

  Returns:
    {:out CPU nested vector [T, C]
     :cache GPU cache map {:q :k :v :att-probs}}

  Resource ownership:
    - Cache contains LIVE GPU resources (q,k,v,att-probs). Do NOT release them here.
    - Backward-hybrid will release everything in the cache.
    - This function releases only temporaries not stored in the cache."
  [qkv nh]
  (ncore/with-default-engine (gpu/cuda-engine)
    (require '[uncomplicate.neanderthal.native :as native])
    (with-release [qkv-cpu (neo/vec->matrix qkv)]
      ;; Transfer input to GPU; will be released in this function (not cached)
      (let [qkv-gpu (gpu/to-gpu qkv-cpu)
            res (attention-forward-qkv-gpu qkv-gpu nh)
            out-gpu (:out-gpu res)
            cache (:cache res)]
        (with-release [out-cpu (gpu/to-cpu out-gpu)]
          ;; Release only temporaries; keep cache alive (Bug 2 fix)
          (release out-gpu)
          (release qkv-gpu)
          {:out (neo/matrix->vec out-cpu)
           :cache cache})))))

(defn attention-backward-qkv-hybrid
  "Hybrid CPU↔GPU backward for single-sequence attention.

  Args:
    dout: CPU nested vector [T, C]
    cache: GPU cache returned by attention-forward-qkv-hybrid
    nh: number of heads

  Returns:
    CPU nested vector gradient [T, 3C]

  Resource ownership:
    - This function releases:
        • the returned GPU gradient (after copying to CPU),
        • ALL GPU resources stored in the cache (q,k,v,att-probs),
        • any temporary GPU allocations."
  [dout cache nh]
  (ncore/with-default-engine (gpu/cuda-engine)
    (require '[uncomplicate.neanderthal.native :as native])
    (with-release [dout-cpu (neo/vec->matrix dout)]
      (let [dout-gpu (gpu/to-gpu dout-cpu)
            dQKV-gpu (attention-backward-qkv-gpu dout-gpu cache nh)]
        (with-release [dQKV-cpu (gpu/to-cpu dQKV-gpu)]
          ;; Release GPU temporaries and cache (Bug 2 fix: release cached here)
          (release dout-gpu)
          (release dQKV-gpu)
          (when-let [q (:q cache)] (release q))
          (when-let [k (:k cache)] (release k))
          (when-let [v (:v cache)] (release v))
          (doseq [p (:att-probs cache)] (release p))
          (neo/matrix->vec dQKV-cpu))))))

;; ============================================================================
;; Notes on Bug Fixes
;; ============================================================================

;; Bug 1 (Scope):
;; - concat-out is now returned INSIDE the let that defines it in attention-forward-qkv-gpu.

;; Bug 2 (Cache lifetime):
;; - Forward hybrid returns cache containing LIVE GPU matrices (q,k,v,att-probs) and does not
;;   wrap them in with-release. Backward hybrid releases all cached resources after use.

;; Bug 3 (with-release syntax):
;; - All with-release uses proper [sym expr] bindings. For releasing pre-existing symbols,
;;   we use explicit (release ...) calls; we do not use bare symbols in with-release.