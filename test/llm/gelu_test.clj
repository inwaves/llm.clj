#_{:clj-kondo/ignore [:underscore-in-namespace]}
(ns llm.gelu_test
  (:require [llm.gelu :refer [gelu_forward gelu_backward gelu_forward_pure_fn gelu_backward_pure_fn]])
  (:require [llm.utils :refer [t_allclose]])
  (:require [clojure.test :refer [deftest testing is]]))

(deftest gelu_forward_pure_fn_test
  (testing "Normal usage..."
    (let [expected (atom [0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075 4.999999770820381])]
      (is (= @(gelu_forward_pure_fn (atom [1 2 3 4 5])) @expected))
      (println "[OK] gelu_forward_pure_fn..."))))

(deftest gelu_forward_test
  (testing "Normal usage..."
    (let [input (atom [1 2 3 4 5])
          output (atom [0 0 0 0 0])
          expected (atom [0.8411919906082768 1.954597694087775 2.996362607918227 3.9999297540518075 4.999999770820381])]
      (gelu_forward output input)
      (is (t_allclose output expected 0.1 0))
      (is (t_allclose output (gelu_forward_pure_fn input) 0.1 0))
      (println "[OK] gelu_forward..."))))

(deftest gelu_backward_pure_fn_test
  (testing "Normal usage..."
    (let [input (atom [1 2 3 4 5])
          dinput (atom [1 2 3 4 5])
          expected (atom [1.9110 4.1229 6.0311 8.0013 10.0])
          result (gelu_backward_pure_fn input (gelu_forward_pure_fn (atom [1 2 3 4 5])) dinput)]
      (is (t_allclose result expected 0.1 0))
      (println "[OK] gelu_backward_pure_fn..."))))

(deftest gelu_backward_test
  (testing "Normal usage"
    (let [input (atom [1 2 3 4 5])
          dinput (atom [1 2 3 4 5])
          output (atom [0 0 0 0 0])
          expected (atom [1.9110 4.1229 6.0311 8.0013 10.0])]
      (gelu_forward output input)
      (gelu_backward dinput input output)
      (is (t_allclose dinput expected))
      (is (t_allclose dinput (gelu_backward_pure_fn input (gelu_forward_pure_fn (atom [1 2 3 4 5])) (atom [1 2 3 4 5]))))
      (println "[OK] gelu_backward..."))))
