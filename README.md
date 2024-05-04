# llm.clj

A port of llm.c to Clojure.

TODO:

- [x] check code I write against reference implementation
  - [x] print properly and check outputs
- [x] helpers
  - [x] torch-style functions that operate on tensors
- [ ] tests
  - [x] `GELU`
  - [x] `softmax`
  - [x] `crossentropy` (ish – check the backprop test)
  - [ ] `encoder`
  - [ ] `layernorm`
  - [~] `matmul`
  - [ ] `attention`
  - [x] `residual`
- [ ] layers' forwards and backwards passes
  - [x] `GELU`
  - [x] `softmax`
  - [x] `crossentropy`
  - [x] `encoder`
  - [x] `layernorm`
  - [x] `matmul` (linear/fully-connected/dense)
  - [~] `attention` (forward pass done)
  - [x] `residual`
- [ ] model building blocks
  - [ ] `ParameterTensors`
  - [ ] `ActivationTensors`
  - [ ] `GPT2Config`
  - [ ] `GPT2`
- [ ] forward function (`gpt2_forward`)
- [ ] backward function (`gpt2_backward`)
- [ ] optimiser step (`gpt2_update`)
- [ ] training logic
  - [ ] dataset, `DataLoader`
  - [ ] sampler
  - [ ] tokenizer
  - [ ] training loop
- [ ] think about parallelisation solutions, e.g. `matmul` in the original imp uses `#pragma omp parallel for collapse(2)`
- [ ] more idiomatic solutions for the building blocks that still works on tensors in-place?
  - specifically for attention – it looks horrible right now: loop inside loops inside loops...
