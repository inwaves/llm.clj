(ns llm.encoder-test
  (:require [clojure.test :refer [deftest testing is]]
            [llm.encoder :refer [encoder_forward encoder_backward]]
            [llm.utils :refer [t_eq t_zeros]]))

(deftest encoder_forward_test
  (testing "Normal usage..."
    (let [inp (atom [[0 1 2] [1 2 3]])
          wte (atom [[1 1] [2 2] [3 3] [4 4]])
          wpe (atom [[0 0] [10 10] [20 20]])
          out (t_zeros [2 3 2])
          expected (atom [[[1 1] [12 12] [23 23]]
                           [[2 2] [13 13] [24 24]]])]
      (encoder_forward out inp wte wpe)
      (is (t_eq out expected))
      (println "[OK] encoder_forward..."))))

(deftest encoder_backward_test
  (testing "Normal usage..."
    (let [inp (atom [[0 1]])
          dout (atom [[[1 2] [3 4]]])
          dwte (t_zeros [2 2])
          dwpe (t_zeros [2 2])
          expected_dwte (atom [[1 2] [3 4]])
          expected_dwpe (atom [[1 2] [3 4]])]
      (encoder_backward dwte dwpe dout inp)
      (is (t_eq dwte expected_dwte))
      (is (t_eq dwpe expected_dwpe))
      (println "[OK] encoder_backward..."))))
