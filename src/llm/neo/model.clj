(ns llm.neo.model
  "GPT-2 model definition and state management."
  (:use [uncomplicate.neanderthal core native]))

(defrecord GPT2Config [max-seq-len vocab-size num-layers num-heads channels])

(defn create-gpt2-config
  "Create a GPT-2 configuration for a specific model size.
  Sizes: :micro (test), :tiny (30M), :small (124M), :medium (350M)"
  [size]
  (case size
    :micro  (->GPT2Config 16 100 1 2 32)      ; Minimal for testing
    :tiny   (->GPT2Config 1024 50257 6  6  384)
    :small  (->GPT2Config 1024 50257 12 12 768)
    :medium (->GPT2Config 1024 50257 24 16 1024)
    (throw (ex-info "Unknown model size" {:size size}))))

(defrecord ParameterTensors [wte wpe ln1w ln1b qkvw qkvb attprojw attprojb
                             ln2w ln2b fcw fcb fcprojw fcprojb lnfw lnfb])

(defn initialize-parameters
  "Initialize model parameters with small random values.
  Returns nested vectors (not Neanderthal matrices yet)."
  [config]
  (let [{:keys [vocab-size max-seq-len num-layers num-heads channels]} config
        C channels L num-layers
        rand-matrix (fn [rows cols scale]
                     (vec (repeatedly rows #(vec (repeatedly cols (fn [] (* scale (- (rand) 0.5))))))))]
    (->ParameterTensors
      (rand-matrix vocab-size C 0.02)           ; wte
      (rand-matrix max-seq-len C 0.02)          ; wpe
      (vec (repeat L (vec (repeat C 1.0))))     ; ln1w (init to 1)
      (vec (repeat L (vec (repeat C 0.0))))     ; ln1b (init to 0)
      (rand-matrix (* L 3 C) C 0.02)            ; qkvw
      (vec (repeat (* L 3 C) 0.0))              ; qkvb
      (rand-matrix (* L C) C 0.02)              ; attprojw
      (vec (repeat (* L C) 0.0))                ; attprojb
      (vec (repeat L (vec (repeat C 1.0))))     ; ln2w
      (vec (repeat L (vec (repeat C 0.0))))     ; ln2b
      (rand-matrix (* L 4 C) C 0.02)            ; fcw
      (vec (repeat (* L 4 C) 0.0))              ; fcb
      (rand-matrix (* L C) (* 4 C) 0.02)        ; fcprojw
      (vec (repeat (* L C) 0.0))                ; fcprojb
      (vec (repeat C 1.0))                      ; lnfw
      (vec (repeat C 0.0)))))                   ; lnfb

(defrecord ModelState [config params optimizer-state step])

(defn create-model
  "Create a new GPT-2 model with initialized parameters."
  [size]
  (let [config (create-gpt2-config size)
        params (initialize-parameters config)]
    (->ModelState config params nil 0)))