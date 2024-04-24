(ns llm.softmax_crossentropy_test
  (:require [llm.softmax_crossentropy :refer :all])
  (:require [llm.utils :refer [t_zeros_like t_allclose]])
  (:require [clojure.test :refer [deftest testing is]]))

(deftest softmax_forward_test
  (testing "Normal usage..."
      (let [logits (atom [[[9.6721 4.2873]] [[7.0474 0.0684]] [[5.2047 6.0802]]])
            probs (t_zeros_like logits) 
            expected (atom [[0.9954 0.0045] [0.9907 0.0009] [0.2941 0.7058]])]
        (softmax_forward probs logits)
        (is (t_allclose probs expected))
        (println "[OK] softmax_forward...")
      )
    )
)

(deftest crossentropy_forward_test
  (testing "Normal usage..."
    (let [
          probs (atom [[[0.000008 0.131538 0.755605 0.458650 ]
                          [0.218959 0.047045 0.678865 0.679296 ]
                          [0.383502 0.519416 0.830965 0.034572 ]]
                         [[0.529700 0.671149 0.007698 0.383416 ]
                          [0.417486 0.686773 0.588977 0.930436 ]
                          [0.526929 0.091965 0.653919 0.415999 ]]])
          targets (atom [[2 1 3] [0 0 3]])
          losses (t_zeros_like targets)
          expected (atom [[0.2802 3.0567 3.3647] [0.6354 0.8735 0.8771]])
          ]
      (crossentropy_forward losses probs targets)
      (is (t_allclose losses expected)) 
      (println "[OK] crossentropy_forward...")
    )
  )  
)
