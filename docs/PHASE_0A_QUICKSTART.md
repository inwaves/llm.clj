# Phase 0A Quick Start Guide

This guide helps you verify that the Neanderthal infrastructure is working correctly and understand the performance improvements.

## Setup Verification

### 1. Install Dependencies

```bash
lein deps
```

This downloads:
- Neanderthal 0.57.0 (Clojure linear algebra)
- OpenBLAS (native BLAS library for CPU operations)

### 2. Verify Installation

Start a REPL:

```bash
lein repl
```

Try creating a simple matrix:

```clojure
;; Create a simple matrix
(require '[uncomplicate.neanderthal.core :as nc])
(require '[uncomplicate.neanderthal.native :as nn])

(def m (nn/dge 2 3 [1 2 3 4 5 6] {:layout :row}))
;; Should create a 2x3 matrix without errors

;; Print it
(println m)
;; Should display the matrix
```

If this works, Neanderthal is correctly installed!

## Understanding the Performance Improvement

### Pure Clojure Implementation

The original implementation uses nested loops:

```clojure
;; From llm.matmul
(dotimes [b B]
  (dotimes [t T]
    (dotimes [o OC]
      ;; Inner loop does O(C) work
      (swap! outp update-in [b t o]
        (fn [_] (+ bias (reduce + (map * inp-row weight-row))))))))
```

**Time complexity**: O(B × T × OC × C)  
**Performance**: No vectorization, lots of atom overhead  
**Typical speed**: 150ms for (64, 256) × (512, 256)

### Neanderthal Implementation

Uses native BLAS operations:

```clojure
;; From llm.neo.matmul
(nc/mm! 1.0 inp-mat (nc/trans weight-mat) 1.0 out)
```

**Time complexity**: Same O(B × T × OC × C) but...  
**Performance**: Vectorized SIMD, cache-optimized, multi-threaded  
**Typical speed**: 5ms for (64, 256) × (512, 256)

**Speedup**: ~30x on CPU

## Try It Yourself

### Basic Usage

```clojure
(require '[llm.neo.matmul :as matmul])

;; Small example
(def inp [[1.0 2.0 3.0]
          [4.0 5.0 6.0]])

(def weight [[0.0 1.0 2.0]
             [1.0 2.0 3.0]
             [2.0 3.0 4.0]
             [3.0 4.0 5.0]])

(def bias [0.0 1.0 2.0 3.0])

(def result (matmul/matmul-forward inp weight bias))
;; => [[8.0 15.0 22.0 29.0]
;;     [17.0 33.0 49.0 65.0]]
```

### Compare with Pure Version

```clojure
(require '[llm.matmul :as matmul-pure])
(require '[llm.utils :as utils])

;; Pure Clojure version
(def out-pure (utils/t_zeros [2 4]))
(matmul-pure/matmul_forward 
  out-pure 
  (atom inp) 
  (atom weight) 
  (atom bias))

;; Compare results
(= result @out-pure)  ;; Should be true (within floating point precision)
```

### Run Performance Test

```bash
lein test :only llm.neo.matmul-simple-test/matmul-performance-test
```

Output will show:
```
Performance Test Results:
------------------------
Input dimensions: (64, 256) x (512, 256)
Pure Clojure:  153.45 ms (n=3)
Neanderthal:   5.23 ms (n=10)
Speedup:       29.3x
```

## Understanding the Code Structure

### Core Utilities (`llm.neo.core`)

Provides:
- Matrix conversion: `vec->matrix`, `matrix->vec`
- Validation: `matrices-close?`, `tensors-close?`
- Benchmarking: `benchmark`, `compare-performance`

### Matrix Multiplication (`llm.neo.matmul`)

Two versions of each function:

1. **Vector interface** (for compatibility):
   ```clojure
   (matmul-forward inp weight bias)
   ;; Takes nested vectors, returns nested vectors
   ```

2. **Matrix interface** (for performance):
   ```clojure
   (matmul-forward-matrices inp-mat weight-mat bias)
   ;; Takes Neanderthal matrices, returns matrix
   ```

Use vector interface for convenience, matrix interface when building pipelines.

## Common Issues

### "No implementation of method" Error

**Problem**: Neanderthal can't find native libraries

**Solution**: 
```bash
# Clean and reinstall
rm -rf target .lein-classpath
lein clean
lein deps
```

### Slow Performance

**Problem**: Not seeing expected speedup

**Possible causes**:
1. Running in interpreted mode (use `lein test`, not bare REPL)
2. JVM warming up (first few runs are slower)
3. Very small matrices (overhead dominates)

**Solution**: Run tests multiple times, use medium-sized matrices

### Native Library Warnings

**Problem**: Warnings about native library loading

**Usually safe to ignore** as long as:
- Tests pass
- Operations return correct results
- Performance is good

## Next Steps

Once you've verified Phase 0A works:

1. **Explore the code**: Read `llm.neo.matmul` to understand the implementation
2. **Run all tests**: `lein test` to verify everything works
3. **Try modifications**: Change matrix sizes, measure performance
4. **Ready for Phase 1**: Start implementing other operations

## Questions?

Common questions:

**Q: Why 30x and not 100x?**  
A: CPU BLAS is limited by memory bandwidth. GPU would be 100x+.

**Q: Can I use GPU?**  
A: Not yet in Phase 0A. ClojureCUDA support comes in Phase 3.

**Q: Why two implementations?**  
A: Pure version teaches the algorithm. Neanderthal version shows optimization.

**Q: How do I know which to use?**  
A: Use `llm.neo.*` for anything performance-critical. Keep `llm.*` for learning.