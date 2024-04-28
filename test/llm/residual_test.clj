(ns llm.residual-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [llm.residual :refer [residual_backward residual_forward]]
   [llm.utils :refer [t_eq]]))

(deftest residual_forward_test
  (testing "Normal usage..."
    (let [inp1 (atom [1 2 3])
          inp2 (atom [0 1 2])
          outp (atom [0 0 0])
          expected (atom [1 3 5])]
      (residual_forward outp inp1 inp2)
      (is (t_eq outp expected))
      (println "[OK] residual_forward..."))))

(deftest residual_backward_test
  (testing "Normal usage..."
    (let [dinp1 (atom [1 2 3])
          dinp2 (atom [0 1 2])
          dout (atom [3 4 5])
          expected_dinp1 (atom [4 6 8])
          expected_dinp2 (atom [3 5 7])]
      (residual_backward dinp1 dinp2 dout)
      (is (t_eq dinp1 expected_dinp1))
      (is (t_eq dinp2 expected_dinp2))
      (println "[OK] residual_backward..."))))
