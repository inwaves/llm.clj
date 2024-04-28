(ns llm.residual
  (:require
   [llm.utils :refer [t_idx t_item t_size]]))

(defn residual_forward
  [out inp1 inp2]
  (reset! out (mapv + @inp1 @inp2)))
