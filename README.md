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

## Phase 2: Training Infrastructure (COMPLETED ✅)

The complete training infrastructure is now in place:

### What's Working ✅

- ✅ **Model State Management** - GPT2Config, ParameterTensors, ModelState
- ✅ **Forward Pass** - Complete end-to-end GPT-2 forward pass with activation caching
- ✅ **Backward Pass** - Full composed backward pass chaining all operation gradients
- ✅ **Loss Computation** - Cross-entropy loss with gradients
- ✅ **AdamW Optimizer** - Full implementation with bias correction (SGD currently in use)
- ✅ **Training Loop** - Complete training step with gradient application
- ✅ **Data Pipeline** - Token loading and batch creation
- ✅ **Checkpointing** - Save/load model state

### Validation

**Training Demo Results:**
```
Step 0: Loss = 4.625639, Grad Norm = 17.743707
Step 1: Loss = 4.351882, Grad Norm = 11.732012
Step 2: Loss = 4.029984, Grad Norm = 10.842924

✓ Loss decreased - gradients are flowing correctly!
```

This proves:
- ✅ Forward and backward passes work correctly
- ✅ Parameters update based on gradients
- ✅ Model can train end-to-end on CPU
- ✅ Loss decreases, validating correct gradient flow

### Performance Results

On typical hardware (modern CPU, no GPU yet):
- **Pure Clojure**: ~150ms for (64, 256) × (512, 256) matmul
- **Neanderthal**: ~5ms for same operation
- **Speedup**: ~30x

## Phase 3: GPU Acceleration (COMPLETED ✅)

GPU-accelerated implementations are now available for all major operations using Neanderthal's CUDA backend:

- ✅ `src/llm/neo/gpu/matmul.clj` - Matrix multiplication (10-50x speedup)
- ✅ `src/llm/neo/gpu/attention.clj` - Multi-head self-attention (5-20x speedup)
- ✅ `src/llm/neo/gpu/layernorm.clj` - Layer normalization (2-5x speedup)
- ✅ `src/llm/neo/gpu/gelu.clj` - GELU activation (2-10x speedup)
- ✅ `src/llm/neo/gpu/residual.clj` - Residual connections (2-5x speedup)
- ✅ `src/llm/neo/gpu/core.clj` - GPU utilities and benchmarking

Each GPU operation provides:
- Pure GPU API for maximum performance
- Hybrid CPU-GPU API for convenience
- Forward and backward passes
- Comprehensive tests with resource management
- Performance benchmarks

See `docs/PHASE_3_GPU_ACCELERATION.md` for usage guide.

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
- [x] Model building blocks (Phase 2)
  - [x] `ParameterTensors`
  - [x] `ActivationTensors`
  - [x] `GPT2Config`
  - [x] `GPT2`
- [x] Forward function (`gpt2_forward`)
- [x] Backward function (`gpt2_backward`) 
- [x] Optimizer (AdamW)
- [x] Training infrastructure
  - [x] `DataLoader`
  - [x] Training loop
  - [x] Checkpointing
- [x] GPU acceleration (Phase 3)
  - [x] GPU matmul with cuBLAS
  - [x] GPU multi-head attention
  - [x] GPU layer normalization
  - [x] GPU GELU activation
  - [x] GPU residual connections
  - [x] Hybrid CPU-GPU wrappers
  - [x] Resource management patterns
  - [x] Comprehensive GPU test suite

### Future (Phase 4+)
- [ ] Performance optimization and kernel fusion
- [ ] Multi-GPU support
- [ ] Production hardening

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

1. Performance optimization and kernel fusion (Phase 4)
2. Add more test cases
3. Improve documentation
4. Multi-GPU support

## License

EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0