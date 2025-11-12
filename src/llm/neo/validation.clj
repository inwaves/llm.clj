(ns llm.neo.validation
  "Utilities for validating implementations against PyTorch ground truth.
  
  This namespace provides:
  - EDN test vector loading
  - Systematic comparison against PyTorch outputs
  - Test result reporting"
  (:use [uncomplicate.neanderthal core native])
  (:require [clojure.edn :as edn]
            [clojure.java.io :as io]
            [llm.neo.core :as neo]))

;; ============================================================================
;; Test Vector Loading
;; ============================================================================

(defn load-test-vectors
  "Load test vectors from an EDN file.
  
  Returns a map with keyword keys:
    :operation - Operation name
    :test-case - Test case identifier  
    :inputs    - Input data map
    :expected  - Expected outputs map"
  [filepath]
  (with-open [r (io/reader filepath)]
    (edn/read (java.io.PushbackReader. r))))

(defn list-test-vectors
  "List all available test vector files."
  []
  (let [test-dir (io/file "dev/test_vectors")]
    (if (.exists test-dir)
      (->> (.listFiles test-dir)
           (filter #(.endsWith (.getName %) ".edn"))
           (map #(.getPath %))
           sort
           vec)
      [])))  ; Return empty vector if directory doesn't exist

;; ============================================================================
;; Data Conversion
;; ============================================================================

(defn edn->matrix
  "Convert EDN nested list to Neanderthal matrix.
  
  Handles the conversion from PyTorch's row-major format to
  Neanderthal's column-major format."
  [edn-data]
  (neo/vec->matrix edn-data))

(defn edn->vector
  "Convert EDN list to Neanderthal vector."
  [edn-data]
  (dv edn-data))

;; ============================================================================
;; Validation
;; ============================================================================

(defn validate-forward
  "Validate forward pass against expected output.
  
  Parameters:
    result   - Neanderthal matrix/vector from implementation
    expected - EDN data from PyTorch
  
  Returns:
    {:match? boolean :max-error float}"
  [result expected-edn]
  (let [expected (if (vector? expected-edn)
                   (if (vector? (first expected-edn))
                     (edn->matrix expected-edn)
                     (edn->vector expected-edn))
                   expected-edn)
        match? (neo/allclose result expected)
        max-err (neo/max-error result expected)]
    {:match? match?
     :max-error max-err}))

(defn validate-backward
  "Validate backward pass gradients against expected values.
  
  Parameters:
    grads-map     - Map of gradient tensors from implementation
    expected-map  - Map of expected gradients from PyTorch
  
  Returns:
    Map with match status and errors for each gradient"
  [grads-map expected-map]
  (into {}
    (for [[grad-key grad-tensor] grads-map]
      (let [expected-edn (get expected-map grad-key)
            expected (cond
                      (nil? expected-edn) nil
                      (vector? (first expected-edn)) (edn->matrix expected-edn)
                      :else (edn->vector expected-edn))]
        (if expected
          [grad-key {:match? (neo/allclose grad-tensor expected)
                     :max-error (neo/max-error grad-tensor expected)}]
          [grad-key {:match? false
                     :max-error nil
                     :note "No expected value provided"}])))))

(defn validate-operation
  "Validate an operation implementation against PyTorch ground truth.
  
  Parameters:
    test-vectors - Loaded test vector data map
    forward-fn   - Forward pass function taking inputs map, returning result
    backward-fn  - Backward pass function taking inputs+grads, returning grads map (optional)
  
  Returns:
    Map with validation results for forward and optionally backward passes"
  [test-vectors forward-fn & [backward-fn]]
  (let [inputs (:inputs test-vectors)
        expected (:expected test-vectors)
        
        ;; Run forward pass
        forward-result (forward-fn inputs)
        forward-validation (validate-forward forward-result (:forward expected))
        
        ;; Run backward pass if function provided
        backward-validation (when backward-fn
                             (let [backward-result (backward-fn inputs expected)]
                               (validate-backward backward-result expected)))]
    
    {:operation (:operation test-vectors)
     :test-case (:test-case test-vectors)
     :forward forward-validation
     :backward backward-validation}))

(defn print-validation-result
  "Pretty print validation results."
  [result]
  (println "\n=== Validation Results ===")
  (printf "Operation: %s\n" (:operation result))
  (printf "Test Case: %s\n" (:test-case result))
  (println "\nForward Pass:")
  (printf "  Match: %s\n" (if (get-in result [:forward :match?]) "✓ YES" "✗ NO"))
  (printf "  Max Error: %.2e\n" (get-in result [:forward :max-error]))
  (when (:backward result)
    (println "\nBackward Pass:")
    (doseq [[grad-name grad-result] (:backward result)]
      (printf "  %s: %s (error: %.2e)\n" 
              (name grad-name)
              (if (:match? grad-result) "✓" "✗")
              (:max-error grad-result)))))

;; ============================================================================
;; Example Usage
;; ============================================================================

(comment
  ;; After generating test vectors with Python script:
  
  ;; 1. Load test vectors
  (def matmul-test (load-test-vectors "dev/test_vectors/matmul_small.edn"))
  
  ;; 2. Create forward wrapper
  (defn test-matmul-forward [inputs]
    (let [inp (edn->matrix (:inp inputs))
          weight (edn->matrix (:weight inputs))
          bias (edn->vector (:bias inputs))]
      (llm.neo.matmul/matmul-forward inp weight bias)))
  
  ;; 3. Validate
  (def result (validate-operation matmul-test test-matmul-forward))
  (print-validation-result result)
  ;; Should show: Forward Pass Match: ✓ YES
  )