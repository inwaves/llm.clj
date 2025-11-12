(ns llm.neo.softmax
  "Softmax operation using Neanderthal.
  
  Softmax converts logits to probabilities by exponentiating and normalizing.
  Uses the numerical stability trick: softmax(x) = softmax(x - max(x))"
  (:require [uncomplicate.neanderthal
             [core :refer [ncols mrows entry! entry]]
             [native :refer [dge]]
             [math :refer [exp]]]))

(defn max-along-cols
  "Find maximum value in each row (across columns).
  Returns a vector of row maxima."
  [x]
  (let [rows (mrows x)
        cols (ncols x)
        maxes (make-array Double/TYPE rows)]
    (dotimes [i rows]
      (let [row-max (loop [j 0 mx Double/NEGATIVE_INFINITY]
                      (if (< j cols)
                        (recur (inc j) (max mx (entry x i j)))
                        mx))]
        (aset maxes i row-max)))
    maxes))

(defn softmax-forward
  "Softmax forward pass.
  
  Computes: softmax(x)[i,j] = exp(x[i,j]) / sum_k(exp(x[i,k]))
  
  Uses numerical stability trick: subtract row max before exp.
  
  Args:
    x: input logits matrix [B×T, C]
    
  Returns:
    probability matrix [B×T, C] where each row sums to 1"
  [x]
  (let [rows (mrows x)
        cols (ncols x)
        
        ;; Find max per row for numerical stability
        row-maxes (max-along-cols x)
        
        ;; Create output matrix
        out (dge rows cols)]
    
    ;; Compute exp(x - max) and sum per row
    (dotimes [i rows]
      (let [row-max (aget row-maxes i)
            
            ;; First pass: exp(x - max) and accumulate sum
            row-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (let [exp-val (exp (- (entry x i j) row-max))]
                          (entry! out i j exp-val)
                          (recur (inc j) (+ sum exp-val)))
                        sum))]
        
        ;; Second pass: normalize by sum
        (dotimes [j cols]
          (entry! out i j (/ (entry out i j) row-sum)))))
    
    out))

(defn softmax-autoregressive
  "Softmax with autoregressive masking (causal attention).
  
  Applies mask so that position i can only attend to positions <= i.
  Sets future positions to 0 probability.
  
  Args:
    x: input logits matrix [T, T] (attention scores)
    
  Returns:
    masked probability matrix [T, T]"
  [x]
  (let [rows (mrows x)
        cols (ncols x)
        row-maxes (max-along-cols x)
        out (dge rows cols)]
    
    (dotimes [i rows]
      (let [row-max (aget row-maxes i)
            
            ;; Compute exp for all positions, accumulate sum only for j <= i
            row-sum (loop [j 0 sum 0.0]
                      (if (< j cols)
                        (if (<= j i)
                          ;; Past/present: compute exp and accumulate
                          (let [exp-val (exp (- (entry x i j) row-max))]
                            (entry! out i j exp-val)
                            (recur (inc j) (+ sum exp-val)))
                          ;; Future: set to 0 and don't accumulate
                          (do
                            (entry! out i j 0.0)
                            (recur (inc j) sum)))
                        sum))]
        
        ;; Normalize only the valid (past/present) positions
        (dotimes [j (inc i)]
          (entry! out i j (/ (entry out i j) row-sum)))))
    
    out))