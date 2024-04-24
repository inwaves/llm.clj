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
        (assert (t_allclose probs expected))
        (println "[OK] softmax_forward_test...")
      )
    )
)

(deftest crossentropy_forward_test
  (testing "Normal usage..."

  )  
)
