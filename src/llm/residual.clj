(ns llm.residual)

(defn residual_forward
  [out inp1 inp2]
  (reset! out (mapv + @inp1 @inp2)))

(defn residual_backward
  [dinp1 dinp2 dout]
  (swap! dinp1 (fn [current_dinp1] (mapv + current_dinp1 @dout)))
  (swap! dinp2 (fn [current_dinp2] (mapv + current_dinp2 @dout))))
