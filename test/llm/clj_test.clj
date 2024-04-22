(ns llm.clj-test
  (:require [clojure.test :refer :all]
            [llm.clj :refer :all]
            [llm.utils :refer :all]))

;; FIXME: Make all these tests work!

(deftest a-test
  (testing "FIXME, I fail."
    (is (= 0 1))))

(deftest t_allclose_test
  (testing "Normal usage..."
    (is true (t_allclose [1.1 2.1 3.1 4.1] [1.0 2.0 3.0 4.0] 0.1 1))
    (println "[OK] t_allclose..."))
)

(let [inp (atom [[[0 1] [1 2] [2 3]] [[3 4] [4 5] [5 6]]])]
  (assert (= @(t_idx inp 0 0 1) @(atom [1])))
  (assert (= @(t_idx inp 0) @(atom [[0 1] [1 2] [2 3]])))
  (assert (= @(t_idx inp [0 1] 0) @(atom [[0 1] [3 4]])))
  (println "[OK] t_idx...")
  )

(let [inp (atom [[1 2] [3 4]])]
  (assert (= @(t_size inp) '[2 2]))
  (assert (= @(t_size inp) '[2 2]))
  (println "[OK] t_size...")
  )

(deftest t_fill_test
  (testing "Normal usage..."
    (is = @(t_fill 0 '(3 2 2)) [[[0 0] [0 0]] [[0 0] [0 0]] [[0 0] [0 0]]]))
  (testing "Size 0..."
    (is (thrown? IllegalArgumentException (t_fill 0 '(0))))
    (is = @(t_fill 0 '(1)) [0])))
  (println "[OK] t_fill...")
)
