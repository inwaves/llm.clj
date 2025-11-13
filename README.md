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

## Phase 1: Core Operations (COMPLETED ✅)

All core operations have been implemented with Neanderthal:

### Implemented Operations

- ✅ **Matmul** - Matrix multiplication (forward + backward)
- ✅ **Encoder** - Token and position embeddings (forward + backward)
- ✅ **LayerNorm** - Layer normalization (forward + backward)
- ✅ **GELU** - Activation function (forward + backward)
- ✅ **Residual** - Residual connections (forward + backward)
- ✅ **Softmax** - Softmax with autoregressive masking (forward + backward)
- ✅ **Attention** - Multi-head self-attention (forward + backward)

All operations achieve 10-50x speedup over pure Clojure implementations.

## Phase 2: Training Infrastructure (85% COMPLETE ⚠️)

Most training infrastructure is in place:

### Completed Components ✅

- ✅ **Model State Management** - GPT2Config, ParameterTensors, ModelState
- ✅ **Forward Pass** - Complete end-to-end GPT-2 forward pass
- ✅ **Loss Computation** - Cross-entropy loss
- ✅ **AdamW Optimizer** - Full implementation with bias correction
- ✅ **Training Loop** - Epoch iteration and step tracking
- ✅ **Data Pipeline** - Token loading and batch creation
- ✅ **Checkpointing** - Save/load model state

### Missing Component ❌

- ❌ **Composed Backward Pass** - While individual operations have backward methods, there's no `gpt2-backward` function that chains them together to compute gradients through the full model. This is the critical missing piece preventing actual end-to-end training.

**What this means:**
- ✅ You can run forward passes and compute loss
- ❌ You cannot compute gradients or train the model end-to-end
- ❌ The optimizer exists but has no gradients to apply

### Performance Results

On typical hardware (modern CPU, no GPU yet):
- **Pure Clojure**: ~150ms for (64, 256) × (512, 256) matmul
- **Neanderthal**: ~5ms for same operation
- **Speedup**: ~30x



## TODO

### Completed ✅
- [x] Infrastructure setup (Phase 0A)
- [x] Neanderthal integration
- [x] Matrix multiplication (forward + backward)
- [x] Validation framework
- [x] Performance benchmarking
- [x] Complete core operations with Neanderthal (Phase 1)
  - [x] `matmul`
  - [x] `encoder`
  - [x] `layernorm`
  - [x] `gelu`
  - [x] `residual`
  - [x] `softmax`
  - [x] `attention`
- [x] Model building blocks (Phase 2 - partial)
  - [x] `ParameterTensors`
  - [x] `ActivationTensors`
  - [x] `GPT2Config`
  - [x] `GPT2`
- [x] Forward function (`gpt2_forward`)
- [x] Optimizer (AdamW)
- [x] Training infrastructure (partial)
  - [x] `DataLoader`
  - [x] Training loop structure
  - [x] Checkpointing

### Critical Missing Piece ⚠️
- [ ] **Composed Backward Pass** - `gpt2_backward` function to chain individual operation gradients through the full network

### Upcoming (Phase 3+)
- [ ] GPU acceleration (ClojureCUDA)
- [ ] Multi-GPU support
- [ ] Further optimizations

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

## Next Steps: Complete Phase 2

To make the model fully trainable, implement the composed backward pass:

**Required Implementation:**
A `gpt2-backward` function (likely in `src/llm/neo/forward.clj` or new `src/llm/neo/backward.clj`) that:
1. Takes the loss gradient
2. Backpropagates through final layer norm
3. Backpropagates through each transformer block in reverse order
4. Accumulates gradients for all parameters
5. Returns gradient structure matching ParameterTensors

**Technical Approach:**
- Chain existing backward methods: `attention-backward`, `layernorm-backward`, `matmul-backward`, etc.
- Maintain activation caches from forward pass
- Accumulate gradients properly through residual connections
- Handle gradient flow through layer norm rescaling

Once this ~15% of Phase 2 is complete, the model will be fully trainable on CPU, ready for Phase 3 (GPU acceleration).

## Contributing

Contributions welcome! Focus areas:

1. Complete Phase 2 backward pass composition
2. Add more test cases
3. Improve documentation
4. Performance profiling and optimization

## License

EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0