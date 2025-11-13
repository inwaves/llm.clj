# llm.clj Implementation Plan

> **Transforming llm.clj into a Complete LLM Training Framework**

## Overview

This document outlines the plan to evolve llm.clj from its current state (individual operations in pure Clojure) into a fully functional LLM training framework that achieves ~50% of llm.c's performance while maintaining Clojure's expressiveness.

**Current State**: ~40-50% complete
- ✅ Core operations implemented in pure Clojure
- ✅ Forward passes for most layers
- ⚠️ Partial backward passes
- ❌ No training infrastructure
- ❌ No GPU acceleration

**Target State**: Complete training framework
- ✅ All operations with Neanderthal (CPU performance)
- ✅ GPU acceleration via ClojureCUDA
- ✅ End-to-end training loop
- ✅ AdamW optimizer
- ✅ Mixed precision support
- ✅ Multi-GPU data parallelism

## Philosophy: Parallel Track Development

We maintain two parallel implementations:

1. **Pure Clojure** (`llm.*` namespaces) - Educational reference
2. **Neanderthal + GPU** (`llm.neo.*` namespaces) - Production-capable

This approach:
- Preserves learning value of existing code
- Enables side-by-side comparison
- Validates correctness through numerical equivalence
- Shows performance improvements clearly

## Phase 0A: Infrastructure Setup (Week 1, First Session)

**Goal**: Prove Neanderthal integration works and establish foundation

### Tasks

1. **Update Dependencies**
   ```clojure
   ;; Add to project.clj
   [uncomplicate/neanderthal "0.47.0"]
   [uncomplicate/clojurecuda "0.16.0"]  ; For future GPU work
   [criterium "0.4.6"]                   ; For benchmarking
   ```

2. **Create Namespace Structure**
   ```
   src/llm/neo/
     ├── core.clj       ; Shared utilities
     ├── matmul.clj     ; Matrix multiplication
     └── validation.clj ; Testing utilities
   ```

3. **Implement First Operation: Matrix Multiply**
   - Use Neanderthal's `gemm!` (General Matrix Multiply)
   - Match interface of existing `llm.matmul/matmul_forward`
   - Create comparison tests

4. **Performance Validation**
   - Benchmark pure Clojure vs Neanderthal
   - Target: 10-50x speedup on CPU
   - Document results

### Success Criteria
- ✅ Dependencies install without errors
- ✅ Neanderthal operations execute correctly
- ✅ Numerical outputs match pure Clojure implementation
- ✅ Measurable performance improvement (10x minimum)

## Phase 0B: Validation Framework (Week 1, Remaining Sessions)

**Goal**: Systematic testing infrastructure

### Tasks

1. **PyTorch Test Vector Generation**
   ```python
   # Generate reference outputs for validation
   python dev/generate_test_vectors.py
   ```

2. **EDN Test Data Format**
   ```clojure
   {:operation "matmul"
    :inputs {:inp [...] :weight [...] :bias [...]}
    :expected {:output [...]}}
   ```

3. **Numerical Comparison Utilities**
   ```clojure
   (defn allclose [a b]
     "Compare tensors within tolerance")
   ```

4. **Performance Benchmarking Suite**
   ```clojure
   (defn benchmark-operation [op-fn inputs]
     "Time and compare implementations")
   ```

### Success Criteria
- ✅ Can load PyTorch test vectors
- ✅ Automated numerical validation
- ✅ Performance benchmarking utilities
- ✅ Clear pass/fail criteria

## Phase 1: CPU Operations with Neanderthal (Weeks 2-3)

**Goal**: Replace all operations with Neanderthal-based versions

### Implementation Order

1. **Matmul** (Session 1) ✅ Done in Phase 0A
2. **Encoder** (Session 2)
   - Embedding table lookup
   - Position encoding addition
3. **LayerNorm** (Session 3)
   - Mean/variance computation
   - Normalization and affine transform
4. **GELU** (Session 4)
   - Element-wise activation
5. **Residual** (Session 5)
   - Element-wise addition for skip connections
6. **Softmax** (Session 5)
   - Row-wise softmax with numerical stability
7. **Attention** (Sessions 6-7)
   - Q, K, V projections
   - Attention scores and masking
   - Weighted value aggregation

### Pattern for Each Operation

```clojure
;; 1. New namespace: llm.neo.OPERATION
(ns llm.neo.matmul
  (:require [uncomplicate.neanderthal.core :as m]
            [uncomplicate.neanderthal.native :as native]))

;; 2. Implementation using Neanderthal
(defn matmul-forward [...]
  (m/gemm! ...))

;; 3. Comparison test
(ns llm.neo.matmul-test
  (:require [llm.matmul :as pure]
            [llm.neo.matmul :as neo]))

(deftest compare-implementations
  (let [result-pure (pure/matmul-forward ...)
        result-neo (neo/matmul-forward ...)]
    (is (allclose result-pure result-neo))))
```

### Success Criteria
- ✅ All operations have Neanderthal versions
- ✅ All tests pass (numerical equivalence)
- ✅ 10-50x performance improvement per operation
- ✅ Clean, idiomatic Clojure code

## Phase 2: Training Infrastructure (Weeks 4-5)

**Goal**: Complete end-to-end training loop on CPU

### Components to Build

1. **Model State Management** (Session 1)
   ```clojure
   (def model-state
     (atom {:params {...}
            :optimizer-state {:m {...} :v {...}}
            :step 0}))
   ```

2. **Forward Pass Composition** (Session 2)
   ```clojure
   (defn forward-pass [params input]
     (-> input
         (embed params)
         (apply-transformer-blocks params)
         (final-projection params)))
   ```

3. **Manual Backward Pass** (Sessions 3-4)
   - Derive gradients for each operation
   - Implement backprop functions
   - Chain rule composition

4. **AdamW Optimizer** (Session 5)
   ```clojure
   (defn adamw-step [params grads optimizer-state]
     {:params updated-params
      :optimizer-state new-state})
   ```

5. **Training Loop** (Session 6)
   ```clojure
   (defn train-epoch [model-state dataloader]
     (doseq [batch dataloader]
       (train-step! model-state batch)))
   ```

6. **Data Pipeline** (Session 7)
   - Token batch loading
   - Efficient matrix construction
   - Mini-batch iteration

7. **Checkpointing** (Session 8)
   - Save/load model state
   - EDN format for readability
   - Resume training capability

### Success Criteria
- ✅ Can train GPT-2 small on TinyShakespeare
- ✅ Loss decreases over time
- ✅ Generates coherent text after training
- ✅ State persistence works correctly

## Phase 3: GPU Acceleration (Weeks 6-8)

**Goal**: CUDA acceleration for compute-intensive operations

### Approach

**Option 1: Neanderthal CUDA Backend** (Recommended first)
```clojure
(require '[uncomplicate.neanderthal.cuda :as cuda])

(with-default-engine (cuda/cuda-float)
  (let [gpu-matrix (cuda/cuv n)]
    (gemm! ...)))  ; Automatically runs on GPU
```

**Option 2: ClojureCUDA Custom Kernels** (If needed)
```clojure
(require '[uncomplicate.clojurecuda.core :as cuda])

(def custom-kernel
  (cuda/kernel
    [(cuda/global-id)]
    [x out]
    (aset out (cuda/global-id)
      (compute (aget x (cuda/global-id))))))
```

### Operations to Accelerate

1. **Matmul** (Session 1)
   - Biggest performance impact
   - Use cuBLAS through Neanderthal

2. **Attention** (Sessions 2-3)
   - Q, K, V matmuls
   - Softmax (custom kernel might be needed)
   - Output projection

3. **LayerNorm** (Session 4)
   - Custom kernel for mean/variance
   - Or use Neanderthal primitives

4. **Element-wise Operations** (Session 5)
   - GELU activation
   - Residual connections
   - Simple kernels via ClojureCUDA

### Memory Management

```clojure
;; CPU → GPU transfer
(def gpu-weights (cuda/transfer cpu-weights))

;; Computation on GPU
(def gpu-result (matmul! gpu-weights gpu-input))

;; GPU → CPU transfer (only when needed)
(def cpu-result (native/transfer gpu-result))
```

### Success Criteria
- ✅ GPU forward pass significantly faster than CPU
- ✅ Numerical outputs match CPU implementation
- ✅ Memory transfers optimized
- ✅ Can train on GPU end-to-end

## Phase 4: Optimization (Weeks 9-10)

**Goal**: Squeeze out more performance

### Optimization Targets

1. **Profiling**
   - Identify bottlenecks
   - Measure kernel occupancy
   - Memory bandwidth analysis

2. **Kernel Fusion**
   - Combine small operations
   - Reduce memory transfers
   - Example: fuse residual + layernorm

3. **Memory Layout**
   - Optimize for cache locality
   - Minimize CPU ↔ GPU transfers
   - Reuse GPU memory

4. **Activation Recomputation**
   - Trade compute for memory
   - Enable larger batch sizes
   - Selective recomputation

### Success Criteria
- ✅ 2-3x improvement over baseline GPU version
- ✅ Approaching 50% of llm.c performance
- ✅ Memory usage optimized

## Phase 5: Multi-GPU Support (Weeks 11-12)

**Goal**: Data parallelism across multiple GPUs

### Implementation

1. **Data Parallel Strategy**
   - Replicate model on each GPU
   - Split batches across GPUs
   - All-reduce gradients

2. **NCCL Integration**
   ```clojure
   ;; Conceptual - details depend on Clojure NCCL bindings
   (defn sync-gradients [gpu-grads]
     (nccl/all-reduce gpu-grads))
   ```

3. **Gradient Synchronization**
   - After backward pass on each GPU
   - Before optimizer update
   - Efficient collective operations

### Success Criteria
- ✅ 2-GPU training ~2x faster than 1-GPU
- ✅ Loss curves identical to single GPU
- ✅ Linear scaling (approximately)

## Phase 6: Polish and Documentation (Week 13)

**Goal**: Make the project usable and educational

### Tasks

1. **Documentation**
   - README with quickstart
   - API documentation
   - Tutorial notebooks
   - Architecture diagrams

2. **Examples**
   - Tiny Shakespeare training
   - GPT-2 fine-tuning
   - Text generation samples

3. **CLI Interface**
   - Command-line argument parsing
   - Configuration files
   - Training progress display

4. **Error Handling**
   - Meaningful error messages
   - Validation of inputs
   - Graceful failure modes

### Success Criteria
- ✅ New user can train a model in <30 minutes
- ✅ Clear documentation for all features
- ✅ Example notebooks run successfully

## Expected Outcomes

### Performance Targets

| Implementation | Relative Speed | Notes |
|----------------|----------------|-------|
| Pure Clojure (baseline) | 0.5% | Educational baseline |
| Neanderthal CPU | 5-10% | Phase 1 completion |
| Neanderthal CUDA | 40-60% | Phase 3 completion |
| Optimized CUDA | 50-80% | Phase 4 completion |

*Percentages relative to llm.c performance

### Completeness by Phase

| Phase | Completeness | Can You... |
|-------|--------------|------------|
| 0 | 50% | Run basic operations efficiently |
| 1 | 60% | Forward pass entire model |
| 2 | 75% | Train a small model on CPU |
| 3 | 85% | Train efficiently on GPU |
| 4 | 90% | Train at competitive speeds |
| 5 | 95% | Scale to multiple GPUs |
| 6 | 100% | Share with others easily |

## Development Workflow

### Daily Rhythm

1. **Morning**: Implement new feature
2. **Midday**: Write tests and validate
3. **Afternoon**: Debug and compare with pure Clojure
4. **Evening**: Document and plan next steps

### Git Workflow

```bash
# Feature branch per phase
git checkout -b phase-0a-neanderthal-setup

# Frequent commits
git commit -m "Add Neanderthal matmul implementation"

# PR when phase complete
# Tag major milestones
git tag v0.1.0-phase0-complete
```

### Testing Strategy

Every new operation requires:
1. Unit tests (correctness)
2. Comparison tests (vs pure Clojure)
3. Performance benchmarks
4. Integration tests (in full forward pass)

## Key Learnings from llm.c

### What We're Adopting

1. **Modular architecture** - Clean separation of operations
2. **Test-driven development** - Validate everything
3. **Progressive optimization** - Simple first, fast later
4. **Educational focus** - Code should teach

### What We're Adapting

1. **No manual memory management** - Let JVM handle it
2. **Functional composition** - Where it makes sense
3. **REPL-driven development** - Interactive experimentation
4. **Library leverage** - Use Neanderthal instead of raw BLAS

## Estimated Timeline

- **Weeks 1-3**: Foundation (Phases 0-1)
- **Weeks 4-5**: Training Infrastructure (Phase 2)
- **Weeks 6-8**: GPU Acceleration (Phase 3)
- **Weeks 9-10**: Optimization (Phase 4)
- **Weeks 11-12**: Multi-GPU (Phase 5)
- **Week 13**: Polish (Phase 6)

**Total**: ~13 weeks to complete framework

## Getting Started

### Prerequisites

```bash
# Install Leiningen
brew install leiningen

# Install CUDA toolkit (for GPU phases)
# Follow NVIDIA instructions for your platform

# Clone repository
git clone https://github.com/inwaves/llm.clj
cd llm.clj
```

### Phase 0A Quickstart

```bash
# Update dependencies (we'll do this next)
lein deps

# Run existing tests
lein test

# Start REPL for interactive development
lein repl
```

### First Code to Write

See next sections in this document for:
- Updated `project.clj`
- New `llm.neo.matmul` namespace
- Comparison tests
- Benchmarking utilities

## Next Steps

➡️ **We are here**: About to implement Phase 0A

The next file will be an updated `project.clj` with Neanderthal dependencies, followed by the first Neanderthal-based implementation.

---

*This plan is a living document. Update as we learn and adapt.*