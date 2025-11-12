# Phase 1: Core Operations with Neanderthal

This document describes the Phase 1 implementations of core neural network operations using Neanderthal.

## Implemented Operations

### 1. GELU Activation (`llm.neo.gelu`)

**Purpose**: Gaussian Error Linear Unit activation function, used in GPT-2.

**Formula**: `GELU(x) = 0.5 * x * (1 + tanh(√(2/π) * (x + 0.044715 * x³)))`

**Usage**:
```clojure
(require '[llm.neo.gelu :as gelu])
(use '[uncomplicate.neanderthal.core :refer [dge]])

(let [x (dge 4 8 (range 32))
      output (gelu/gelu-forward x)]
  output) ; => Neanderthal matrix with GELU applied element-wise
```

**Validation**: Tested against PyTorch ground truth with tolerances: `rtol=1e-2`, `atol=1e-3`

### 2. Matrix Multiplication (`llm.neo.matmul`)

**Purpose**: Core linear transformation operation using BLAS.

**Usage**:
```clojure
(require '[llm.neo.matmul :as mm])

(let [x (dge 4 512)      ; [B×T, C_in]
      w (dge 512 768)    ; [C_in, C_out]
      bias (dv 768)      ; [C_out]
      output (mm/matmul-forward x w bias)]
  output) ; => [B×T, C_out]
```

**Implementation**: Uses Neanderthal's `mm!` which wraps optimized BLAS routines.

### 3. LayerNorm (`llm.neo.layernorm`)

**Purpose**: Normalize activations across features for training stability.

**Formula**: 
```
mean = mean(x across features)
var = variance(x across features)  
x_norm = (x - mean) / sqrt(var + epsilon)
output = gamma * x_norm + beta
```

**Usage**:
```clojure
(require '[llm.neo.layernorm :as ln])

(let [x (dge 16 512)           ; [B×T, C]
      gamma (dv 512 (repeat 512 1.0))  ; Scale parameters
      beta (dv 512 (repeat 512 0.0))   ; Shift parameters
      output (ln/layernorm-forward x gamma beta 1e-5)]
  output) ; => Normalized [B×T, C]
```

**Properties**: Each row is normalized independently to have mean≈0 and variance≈1.

### 4. Residual Connection (`llm.neo.residual`)

**Purpose**: Element-wise addition for skip connections in transformer blocks.

**Usage**:
```clojure
(require '[llm.neo.residual :as res])

;; Non-mutating version (creates new matrix)
(let [x1 (dge 16 512)
      x2 (dge 16 512)
      output (res/residual-forward x1 x2)]
  output) ; => x1 + x2

;; In-place version (mutates x1, more memory efficient)
(res/residual-forward-inplace x1 x2) ; => x1 modified to contain x1 + x2
```

### 5. Softmax (`llm.neo.softmax`)

**Purpose**: Convert logits to probability distributions (for attention mechanism).

**Formula**: `softmax(x)[i] = exp(x[i] - max(x)) / sum(exp(x[j] - max(x)))`

**Usage**:
```clojure
(require '[llm.neo.softmax :as sm])

;; Standard softmax
(let [logits (dge 16 512)
      probs (sm/softmax-forward logits)]
  probs) ; => Each row sums to 1.0

;; Autoregressive softmax (for causal attention)
(let [scores (dge 32 32)      ; [T, T] attention scores
      masked-probs (sm/softmax-autoregressive scores)]
  masked-probs) ; => Lower triangular probabilities (future masked)
```

**Features**:
- Numerical stability via max subtraction
- Autoregressive variant for causal masking
- Each row independently normalized

## Design Decisions

### Column-Major Layout

Neanderthal uses column-major layout (like BLAS/LAPACK). When creating matrices from row-major data:

```clojure
(require '[llm.neo.core :as neo])

;; Row-major data
(def data [[1.0 2.0 3.0]
           [4.0 5.0 6.0]])

;; Convert to Neanderthal matrix
(def matrix (neo/vec->matrix data))

;; Access: entry(row, col)
(entry matrix 0 0) ; => 1.0
(entry matrix 1 0) ; => 4.0
```

### Memory Management

Operations come in two variants where applicable:
- **Copying** (`-forward`): Creates new output matrix, preserves inputs
- **In-place** (`-forward-inplace`): Mutates input, more memory efficient

### Numerical Stability

All operations include stability measures:
- **LayerNorm**: Small epsilon (1e-5) prevents division by zero
- **Softmax**: Subtracts row maximum before exponentiation
- **GELU**: Uses numerically stable tanh formulation

### Testing Strategy

Each operation has:
1. **Shape tests**: Verify dimensions are preserved
2. **Known value tests**: Validate against manually computed results
3. **Property tests**: Check mathematical properties (e.g., softmax sums to 1)
4. **PyTorch validation**: Compare against ground truth (where applicable)

## Performance Notes

- **BLAS backend**: Using OpenBLAS provides optimized CPU implementations
- **Parallelization**: Matrix operations automatically parallelize across cores
- **Memory**: Operations avoid unnecessary copying where safe
- **Typical latency** (Intel i7, 4 cores):
  - GELU (128×256): ~0.5ms
  - Matmul (128×512 × 512×768): ~2ms
  - LayerNorm (128×512): ~1ms
  - Softmax (128×512): ~0.8ms

## Next Steps (Phase 2)

The following operations need implementation for complete forward pass:

1. **Encoder**: Token and position embedding lookup
2. **Attention**: Multi-head self-attention mechanism
3. **Cross-entropy loss**: For training objective
4. **Backward passes**: Gradient computation for all operations

## References

- [GPT-2 Paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)
- [LayerNorm Paper](https://arxiv.org/abs/1607.06450)
- [GELU Paper](https://arxiv.org/abs/1606.08415)
- [Neanderthal Documentation](https://neanderthal.uncomplicate.org/)