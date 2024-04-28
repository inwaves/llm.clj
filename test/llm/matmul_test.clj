(ns llm.matmul-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [llm.matmul :refer [matmul_forward]]
   [llm.utils :refer [t_eq t_zeros]]))

(deftest matmul_forward_test
  (testing "Normal usage..."
    (let [inp (atom [[[1 2 3] [4 5 6]] [[7 8 9] [0 1 2]]])  ;; (B T C) = (2 2 3)
          outp (t_zeros [2 2 4]) ;; (B T OC)
          weight (atom [[0 1 2] [1 2 3] [2 3 4] [3 4 5]]) ;; (OC C) = (4 3)
          bias (atom [0 1 2 3])  ;; (OC,)
          expected (atom [[[8 15 22 29] [17 33 49 65]] [[26 51 76 101] [5 9 13 17]]])]
      (matmul_forward outp inp weight bias)
      (is (t_eq outp expected))
      (println "[OK] matmul_forward..."))))
