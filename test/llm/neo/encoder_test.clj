(ns llm.neo.encoder-test
  "Tests for Neanderthal-based encoder implementation."
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.test :refer [deftest is testing]]
            [llm.neo.encoder :as encoder]
            [llm.neo.core :as neo]))

(deftest encoder-forward-small-test
  (testing "Encoder forward pass on small known values"
    (let [;; Simple case: 1 batch, 2 positions, 3 channels
          ;; Token embeddings for vocab of size 4
          wte [[1.0 2.0 3.0]    ; token 0
               [4.0 5.0 6.0]    ; token 1  
               [7.0 8.0 9.0]    ; token 2
               [10.0 11.0 12.0]] ; token 3
          ;; Position embeddings for max 2 positions
          wpe [[0.1 0.2 0.3]    ; position 0
               [0.4 0.5 0.6]]   ; position 1
          ;; Input: batch=1, tokens=[1, 2] (select embeddings for tokens 1 and 2)
          inp [[1 2]]
          
          result (encoder/encoder-forward inp wte wpe)
          
          ;; Expected: token embedding + position embedding at each position
          ;; Position 0: [4 5 6] + [0.1 0.2 0.3] = [4.1 5.2 6.3]
          ;; Position 1: [7 8 9] + [0.4 0.5 0.6] = [7.4 8.5 9.6]
          expected [[[4.1 5.2 6.3] [7.4 8.5 9.6]]]]
      
      (is (neo/allclose result expected 1e-5 1e-8)
          "Should correctly sum token and position embeddings"))))

(deftest encoder-forward-batch-test
  (testing "Encoder handles multiple batches correctly"
    (let [wte [[1.0 2.0] [3.0 4.0] [5.0 6.0]]
          wpe [[0.1 0.1] [0.2 0.2]]
          ;; Two batches with different tokens
          inp [[0 1]  ; batch 0: tokens 0, 1
               [1 2]] ; batch 1: tokens 1, 2
          
          result (encoder/encoder-forward inp wte wpe)]
      
      ;; Check dimensions
      (is (= 2 (count result)) "Should have 2 batch elements")
      (is (= 2 (count (first result))) "Each batch should have T=2 positions")
      (is (= 2 (count (first (first result)))) "Each position should have C=2 channels")
      
      ;; Check first batch, first position: token 0 + pos 0 = [1 2] + [0.1 0.1] = [1.1 2.1]
      (is (neo/close-enough? (get-in result [0 0 0]) 1.1 1e-5 1e-8))
      (is (neo/close-enough? (get-in result [0 0 1]) 2.1 1e-5 1e-8)))))

(deftest encoder-backward-test
  (testing "Encoder backward accumulates gradients correctly"
    (let [;; Small test case
          inp [[0 1]]  ; 1 batch, 2 positions
          dout [[[1.0 1.0] [1.0 1.0]]]  ; gradient all ones
          vocab-size 3
          max-T 2
          C 2  ; channels
          
          {:keys [dwte dwpe]} (encoder/encoder-backward dout inp vocab-size max-T C)]
      
      ;; Check dimensions
      (is (= [3 2] [(count dwte) (count (first dwte))]))
      (is (= [2 2] [(count dwpe) (count (first dwpe))]))
      
      ;; dwte should have gradients only for tokens 0 and 1 (from inp)
      (is (not= [0.0 0.0] (get dwte 0)) "Token 0 should have gradient")
      (is (not= [0.0 0.0] (get dwte 1)) "Token 1 should have gradient")
      (is (= [0.0 0.0] (get dwte 2)) "Token 2 should have zero gradient (not used)")
      
      ;; dwpe should have gradients at both positions
      (is (not= [0.0 0.0] (get dwpe 0)) "Position 0 should have gradient")
      (is (not= [0.0 0.0] (get dwpe 1)) "Position 1 should have gradient"))))

(deftest encoder-shape-preservation-test
  (testing "Encoder produces correct output shape"
    (let [B 4
          T 8
          V 100
          C 64
          max-T 1024
          
          ;; Random embeddings
          wte (vec (repeatedly V (fn [] (vec (repeatedly C rand)))))
          wpe (vec (repeatedly max-T (fn [] (vec (repeatedly C rand)))))
          
          ;; Random token indices
          inp (vec (repeatedly B (fn [] (vec (repeatedly T #(rand-int V))))))
          
          result (encoder/encoder-forward inp wte wpe)]
      
      ;; Verify output shape
      (is (= B (count result)) "Should have B batch elements")
      (is (= T (count (first result))) "Each batch should have T positions")
      (is (= C (count (first (first result)))) "Each position should have C channels"))))