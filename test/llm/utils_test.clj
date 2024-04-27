(ns llm.utils_test
  (:require [clojure.test :refer :all]
            [llm.clj :refer :all]
            [llm.utils :refer :all]))

;; All these tests can be improved, leaving for later.

(deftest t_allclose_test
  (testing "Normal usage..."
    (is true (t_allclose (atom [1.1 2.1 3.1 4.1]) (atom [1.0 2.0 3.0 4.0]) 0.1 1))
    (println "[OK] t_allclose...")))

(deftest t_idx_test
  (testing "Normal usage..."
    (let [inp (atom [[[0 1] [1 2] [2 3]] [[3 4] [4 5] [5 6]]])]
      (is (= @(t_idx inp 0 0 1) @(atom [1])))
      (is (= @(t_idx inp 0) @(atom [[0 1] [1 2] [2 3]])))
      (is (= @(t_idx inp [0 1] 0) @(atom [[0 1] [3 4]])))))
  (testing "Corner cases...") ;; TODO:
  (println "[OK] t_idx..."))

(deftest t_size_test
  (testing "Normal usage..."
    (let [inp (atom [[1 2] [3 4]])]
      (is (= @(t_size inp) '[2 2]))
      (is (= @(t_size inp) '[2 2]))
      (println "[OK] t_size..."))))

(deftest t_fill_test
  (testing "Normal usage..."
    (is (= @(t_fill 0 '(3 2 2)) [[[0 0] [0 0]] [[0 0] [0 0]] [[0 0] [0 0]]])))
  (testing "Size 0..."
    (is (thrown? IllegalArgumentException (t_fill 0 '(0))))
    (is (= @(t_fill 0 '(1)) [0])))
  (println "[OK] t_fill..."))
