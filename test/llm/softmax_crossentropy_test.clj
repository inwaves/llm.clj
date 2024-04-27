#_{:clj-kondo/ignore [:underscore-in-namespace]}
(ns llm.softmax_crossentropy_test
  (:require [llm.softmax_crossentropy :refer [softmax_forward crossentropy_forward crossentropy_softmax_backward]])
  (:require [llm.utils :refer [t_zeros_like t_allclose]])
  (:require [clojure.test :refer [deftest testing is]]))

(deftest softmax_forward_test
  (testing "Normal usage..."
    (let [logits (atom [[[9.6721 4.2873]] [[7.0474 0.0684]] [[5.2047 6.0802]]])
          probs (t_zeros_like logits)
          expected (atom [[0.9954 0.0045] [0.9907 0.0009] [0.2941 0.7058]])]
      (softmax_forward probs logits)
      (is (t_allclose probs expected))
      (println "[OK] softmax_forward..."))))

(deftest crossentropy_forward_test
  (testing "Normal usage..."
    (let [probs (atom [[[0.000008 0.131538 0.755605 0.458650]
                        [0.218959 0.047045 0.678865 0.679296]
                        [0.383502 0.519416 0.830965 0.034572]]
                       [[0.529700 0.671149 0.007698 0.383416]
                        [0.417486 0.686773 0.588977 0.930436]
                        [0.526929 0.091965 0.653919 0.415999]]])
          targets (atom [[2 1 3] [0 0 3]])
          losses (t_zeros_like targets)
          expected (atom [[0.2802 3.0567 3.3647] [0.6354 0.8735 0.8771]])]
      (crossentropy_forward losses probs targets)
      (is (t_allclose losses expected))
      (println "[OK] crossentropy_forward..."))))

(deftest crossentropy_softmax_backward_test
  "FIXME: this test doesn't work right now,
  find a ground truth to check against."
  (testing "Normal usage..."
    (let [logits (atom [[[0.1 0.2 0.3 0.4]
                         [0.5 0.6 0.7 0.8]
                         [0.9 1.0 1.1 1.2]]
                        [[1.3 1.4 1.5 1.6]
                         [1.7 1.8 1.9 2.0]
                         [2.1 2.2 2.3 2.4]]])
          probs (t_zeros_like logits)
          targets (atom [[2 1 3] [0 0 3]])
          dlosses (atom [[0.1 0.2 0.3] [0.4 0.5 0.6]])
          dlogits (t_zeros_like logits)
          expected (atom [[[0.02497919 0.02663422 -0.07838659 0.02677318]
                           [0.05323514 -0.14647294 0.04647489 0.04676291]
                           [0.07175925 0.07383042 0.07589582 -0.22148549]]
                          [[0.35004634 -0.14142654 -0.11676554 -0.09185426]
                           [0.43755726 -0.17678290 -0.14595641 -0.11481795]
                           [0.07512523 0.07663095 0.07814872 -0.23990489]]])]
      (softmax_forward probs logits)
      (println probs)
      (println targets)
      (crossentropy_softmax_backward dlogits dlosses probs targets)
      (is (t_allclose dlogits expected))
      (println "[OK] crossentropy_softmax_backward..."))))
