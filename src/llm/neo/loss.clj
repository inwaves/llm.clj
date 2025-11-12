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