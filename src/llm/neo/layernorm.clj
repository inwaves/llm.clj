(ns llm.neo.layernorm
  "Layer Normalization using Neanderthal."
  (:use [uncomplicate.neanderthal core native math]))

(defn mean-along-cols
  "Compute mean of each row (across columns).
  Returns a column vector of row means."
  [x]
  (let [rows (mrows x)
        cols (ncols x)
        means (dv rows)]
    (dotimes [i rows]
      (let [row-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (recur (inc j) (+ sum (entry x i j)))
                        sum))]
        (entry! means i (/ row-sum cols))))
    means))

(defn variance-along-cols
  "Compute variance of each row (across columns) given means.
  Returns a column vector of row variances."
  [x means]
  (let [rows (mrows x)
        cols (ncols x)
        variances (dv rows)]
    (dotimes [i rows]
      (let [mu (entry means i)
            var-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (let [diff (- (entry x i j) mu)]
                          (recur (inc j) (+ sum (* diff diff))))
                        sum))]
        (entry! variances i (/ var-sum cols))))
    variances))

(defn layernorm-forward
  "Layer Normalization forward pass.
  
  Given input x of shape [B×T, C]:
  - Normalize each row (across C features) to have mean=0, var=1
  - Scale by gamma and shift by beta (both shape [C])
  
  Args:
    x: input matrix [B×T, C]
    gamma: scale parameters [C]
    beta: shift parameters [C]
    epsilon: small value for numerical stability
    
  Returns:
    normalized output matrix [B×T, C]"
  [x gamma beta epsilon]
  (let [rows (mrows x)
        cols (ncols x)
        
        ;; Compute statistics per row
        means (mean-along-cols x)
        vars (variance-along-cols x means)
        
        ;; Create output matrix
        out (dge rows cols)]
    
    ;; Normalize: x_norm = (x - mean) / sqrt(var + eps)
    ;; Then apply affine: y = gamma * x_norm + beta
    (dotimes [i rows]
      (let [mu (entry means i)
            std (sqrt (+ (entry vars i) epsilon))]
        (dotimes [j cols]
          (let [x-val (entry x i j)
                x-norm (/ (- x-val mu) std)
                ;; Apply affine transform: y = gamma * x_norm + beta
                y (+ (* (entry gamma j) x-norm) (entry beta j))]
            (entry! out i j y)))))
    
    out))

(defn layernorm-backward
  "Layer normalization backward pass.
  
  Computes gradients with respect to:
  - Input x
  - Scale parameter gamma
  - Shift parameter beta
  
  This uses cached statistics from the forward pass for efficiency.
  
  Args:
    x: original input [B×T, C]
    dout: gradient from upstream [B×T, C]
    gamma: scale parameters [C]
    beta: shift parameters [C] (unused but kept for API consistency)
    epsilon: numerical stability constant
    
  Returns:
    {:dx [B×T, C] :dgamma [C] :dbeta [C]}"
  [x dout gamma beta epsilon]
  (let [rows (mrows x)
        cols (ncols x)
        
        ;; Recompute forward statistics (could cache from forward for efficiency)
        means (mean-along-cols x)
        vars (variance-along-cols x means)
        
        ;; Initialize gradient accumulators
        dx (dge rows cols)
        dgamma (dv cols)
        dbeta (dv cols)]
    
    ;; Process each row independently
    (dotimes [i rows]
      (let [mu (entry means i)
            var (entry vars i)
            std (sqrt (+ var epsilon))
            inv-std (/ 1.0 std)]
        
        ;; Accumulate dgamma and dbeta for this row
        (dotimes [j cols]
          (let [x-norm (* (- (entry x i j) mu) inv-std)
                dout-val (entry dout i j)]
            ;; dgamma += dout * x_norm
            (entry! dgamma j (+ (entry dgamma j) (* dout-val x-norm)))
            ;; dbeta += dout
            (entry! dbeta j (+ (entry dbeta j) dout-val))))
        
        ;; Compute dx for this row
        ;; This involves all elements in the row due to mean/var dependencies
        (let [;; Compute intermediate values needed for dx
              sum-dout-gamma (loop [j 0 sum 0.0]
                               (if (< j cols)
                                 (recur (inc j) (+ sum (* (entry dout i j) (entry gamma j))))
                                 sum))
              sum-dout-gamma-xnorm (loop [j 0 sum 0.0]
                                     (if (< j cols)
                                       (let [x-norm (* (- (entry x i j) mu) inv-std)]
                                         (recur (inc j) (+ sum (* (entry dout i j) (entry gamma j) x-norm))))
                                       sum))]
          
          (dotimes [j cols]
            (let [x-norm (* (- (entry x i j) mu) inv-std)
                  dout-gamma (* (entry dout i j) (entry gamma j))
                  ;; Gradient formula for layer norm
                  dx-val (* inv-std
                           (- dout-gamma
                              (/ sum-dout-gamma cols)
                              (* x-norm (/ sum-dout-gamma-xnorm cols))))]
              (entry! dx i j dx-val))))))
    
    {:dx dx :dgamma dgamma :dbeta dbeta}))