(ns llm.neo.dataloader
  "Data loading utilities for training."
  (:require [clojure.java.io :as io])
  (:import [java.nio.file Files Paths]
           [java.nio ByteBuffer ByteOrder]))

(defn load-tokens-from-file
  "Load tokenized data from a binary file.
  Args: filepath - Path to .bin file
  Returns: Vector of token integers"
  [filepath]
  (let [bytes (Files/readAllBytes (Paths/get filepath (into-array String [])))
        ;; Skip 1024-byte header, read uint16 tokens (little-endian)
        num-tokens (quot (- (count bytes) 1024) 2)]
    (vec (for [i (range num-tokens)]
           (let [offset (+ 1024 (* i 2))
                 b1 (bit-and (aget bytes offset) 0xFF)
                 b2 (bit-and (aget bytes (inc offset)) 0xFF)]
             (bit-or b1 (bit-shift-left b2 8)))))))

(defn create-batches
  "Create batches from tokens.
  Args: tokens vector, batch-size (B), seq-len (T)
  Returns: Seq of {:inputs [B,T] :targets [B*T]}"
  [tokens batch-size seq-len]
  (let [tokens-per-batch (* batch-size (inc seq-len))]
    (->> tokens
         (partition tokens-per-batch)  ; Only full chunks, no padding
         (map (fn [chunk]
                (let [sequences (partition (inc seq-len) chunk)
                      inputs (mapv #(vec (take seq-len %)) sequences)
                      targets (vec (mapcat #(drop 1 %) sequences))]
                  {:inputs inputs :targets targets}))))))

(defn simple-dataloader [filepath batch-size seq-len]
  (create-batches (load-tokens-from-file filepath) batch-size seq-len))