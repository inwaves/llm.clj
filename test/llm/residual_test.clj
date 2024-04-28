(ns llm.residual-test
  (:require
   [clojure.test :refer [deftest is testing]]
   [llm.residual :refer [residual_forward]]))

(deftest residual_forward_test
  (testing "Normal usage..."
    (let [inp1 (atom [1 2 3])
          inp2 (atom [0 1 2])
          outp (atom [0 0 0])
          expected (atom [1 3 5])]
      (residual_forward outp inp1 inp2)
      (is (= @outp @expected))
      (println "[OK] residual_forward..."))))
