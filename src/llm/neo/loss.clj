(ns llm.neo.loss
  (:use [uncomplicate.neanderthal core native]))

(defn cross-entropy-loss [logits targets]
  (let [BT (mrows logits) V (ncols logits)]
    (/ (reduce + (for [i (range BT)]
                   (let [tgt (nth targets i)
                         row (vec (for [j (range V)] (entry logits i j)))
                         mx (apply max row)
                         exps (mapv #(Math/exp (- % mx)) row)
                         nll (- (+ mx (Math/log (reduce + exps))) (nth row tgt))]
                     nll)))
       BT)))

(defn cross-entropy-loss-gradient
  "Compute gradient of cross-entropy loss w.r.t. logits.
  
  For softmax cross-entropy, the gradient is:
  dL/dlogits = softmax(logits) - one_hot(targets)
  
  Args:
    logits-mat: [N, V] matrix of logits
    targets: [N] vector of target class indices
    
  Returns:
    [N, V] matrix of gradients"
  [logits-mat targets]
  (let [N (mrows logits-mat)
        V (ncols logits-mat)
        dlogits (dge N V)]
    
    ;; Compute softmax probabilities
    (dotimes [i N]
      (let [row-logits (row logits-mat i)
            ;; Find max for numerical stability
            max-logit (loop [j 0 mx Double/NEGATIVE_INFINITY]
                       (if (< j V)
                         (recur (inc j) (max mx (entry row-logits j)))
                         mx))
            ;; Compute exp and sum
            exp-sum (loop [j 0 sum 0.0]
                     (if (< j V)
                       (let [exp-val (Math/exp (- (entry row-logits j) max-logit))]
                         (recur (inc j) (+ sum exp-val)))
                       sum))
            target-idx (nth targets i)]
        
        ;; Set gradient: prob - 1 for target class, prob for others
        (dotimes [j V]
          (let [prob (/ (Math/exp (- (entry row-logits j) max-logit)) exp-sum)
                grad (if (= j target-idx) (- prob 1.0) prob)]
            (entry! dlogits i j grad)))))
    
    dlogits))