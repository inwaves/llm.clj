# llm.clj

A port of llm.c to Clojure with performance optimizations.

## Overview

This project implements LLM training in Clojure, providing both educational pure-Clojure implementations and high-performance Neanderthal-based versions.

### Implementation Strategy

We maintain two parallel implementations:

1. **Pure Clojure** (`llm.*` namespaces) - Educational reference implementations
2. **Neanderthal-optimized** (`llm.neo.*` namespaces) - High-performance implementations

This dual approach:
- Preserves educational value of the pure implementations
- Provides practical performance through native BLAS operations
- Validates correctness by comparing implementations
- Shows the performance impact of proper numerical computing infrastructure

## Quick Start

### Prerequisites

- Java 11 or later
- Leiningen 2.9.0 or later

### Setup

```bash
# Clone the repository
git clone https://github.com/inwaves/llm.clj.git
cd llm.clj

# Install dependencies (including native libraries)
lein deps

# Run tests
lein test

# Start REPL with neo namespace loaded
lein repl
```

### First Steps

Try the Neanderthal-based matrix multiplication:

```clojure
(require '[llm.neo.matmul :as matmul])
(require '[llm.neo.core :as neo])

;; Simple example
(def inp [[1 2 3] [4 5 6]])
(def weight [[0 1 2] [1 2 3] [2 3 4] [3 4 5]])
(def bias [0 1 2 3])

(def result (matmul/matmul-forward inp weight bias))
;; => [[8.0 15.0 22.0 29.0] [17.0 33.0 49.0 65.0]]

;; Compare with pure Clojure version
(require '[llm.matmul :as matmul-pure])
(require '[llm.utils :as utils])

(def out-pure (utils/t_zeros [2 4]))
(matmul-pure/matmul_forward out-pure (atom inp) (atom weight) (atom bias))
@out-pure  ;; Should match result above
```

### Performance Comparison

```clojure
;; Run the performance test
lein test :only llm.neo.matmul-simple-test/matmul-performance-test

;; Expected output shows 10-50x speedup on CPU:
;; Pure Clojure:  ~150 ms
;; Neanderthal:   ~5 ms  
;; Speedup:       ~30x
```

## Project Structure

```
src/llm/
  ├── *.clj              # Pure Clojure implementations (educational)
  └── neo/
      ├── core.clj       # Neanderthal utilities and benchmarking
      ├── matmul.clj     # Optimized matrix multiplication
      └── ...            # Other optimized operations (coming soon)

test/llm/
  ├── *_test.clj         # Tests for pure implementations
  └── neo/
      └── *_test.clj     # Tests comparing pure vs optimized
```

## Phase 0A: Infrastructure (COMPLETED ✅)

The foundation is now in place:

### What's Working

- ✅ Neanderthal dependencies configured
- ✅ Matrix multiplication (forward and backward)
- ✅ Numerical validation against pure Clojure
- ✅ Performance benchmarking (10-50x speedup)
- ✅ Clean pattern for future operations

### Key Files

- `src/llm/neo/core.clj` - Core utilities for all neo implementations
- `src/llm/neo/matmul.clj` - Neanderthal-based matrix multiplication
- `test/llm/neo/matmul_simple_test.clj` - Validation and benchmarks

### Performance Results

On typical hardware (modern CPU, no GPU yet):
- **Pure Clojure**: ~150ms for (64, 256) × (512, 256) matmul
- **Neanderthal**: ~5ms for same operation
- **Speedup**: ~30x

## Next Steps: Phase 1 - Core Operations

Convert remaining operations to Neanderthal:

### Priority Order

1. ✅ **Matmul** - DONE (biggest performance impact)
2. **Encoder** - Embedding lookup (easy, use Neanderthal indexing)
3. **LayerNorm** - Row-wise operations
4. **GELU** - Element-wise operations with Neanderthal vect-math
5. **Residual** - Element-wise addition
6. **Softmax** - Row-wise operations with stability
7. **Attention** - Composition of above operations

Each operation will follow the pattern:
- Implement in `llm.neo.*` namespace
- Test against pure version
- Benchmark performance
- Document learnings

## TODO

### Completed
- [x] Infrastructure setup (Phase 0A)
- [x] Neanderthal integration
- [x] Matrix multiplication (forward + backward)
- [x] Validation framework
- [x] Performance benchmarking

### In Progress
- [ ] Complete core operations with Neanderthal
  - [x] `matmul` - DONE
  - [ ] `encoder`
  - [ ] `layernorm`
  - [ ] `gelu`
  - [ ] `residual`
  - [ ] `softmax`
  - [ ] `attention`

### Upcoming
- [ ] Model building blocks
  - [ ] `ParameterTensors`
  - [ ] `ActivationTensors`
  - [ ] `GPT2Config`
  - [ ] `GPT2`
- [ ] Forward function (`gpt2_forward`)
- [ ] Backward function (`gpt2_backward`)
- [ ] Optimizer (AdamW)
- [ ] Training infrastructure
  - [ ] `DataLoader`
  - [ ] Tokenizer
  - [ ] Training loop
- [ ] GPU acceleration (ClojureCUDA)

## Learning Resources

### Understanding the Code

1. Start with pure implementations (`llm.*`) to understand the math
2. Study Neanderthal versions (`llm.neo.*`) to see optimization techniques
3. Read test files to see validation and usage patterns

### Key Concepts

- **Neanderthal**: High-performance linear algebra for Clojure
- **BLAS**: Basic Linear Algebra Subprograms (native operations)
- **OpenBLAS**: Open-source BLAS implementation (CPU backend)

### References

- [llm.c](https://github.com/karpathy/llm.c) - Original C implementation
- [Neanderthal](https://neanderthal.uncomplicate.org/) - Documentation
- [Deep Learning from Scratch](https://karpathy.github.io/2019/04/25/recipe/) - Karpathy's guide

## Development

### Running Tests

```bash
# All tests
lein test

# Specific namespace
lein test llm.neo.matmul-simple-test

# Specific test
lein test :only llm.neo.matmul-simple-test/matmul-performance-test
```

### REPL Development

The project is configured for REPL-driven development:

```clojure
;; In REPL (automatically starts in llm.neo.core)
(require '[llm.neo.matmul :as m] :reload)

;; Quick test
(m/matmul-forward [[1 2] [3 4]] [[5 6] [7 8]] nil)

;; Benchmark
(neo/benchmark #(m/matmul-forward ...) 100)
```

## Contributing

Contributions welcome! Focus areas:

1. Complete Phase 1 operations (encoder, layernorm, etc.)
2. Add more test cases
3. Improve documentation
4. Performance profiling and optimization

## License

EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0