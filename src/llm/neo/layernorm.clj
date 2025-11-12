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