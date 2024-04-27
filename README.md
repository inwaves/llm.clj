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
  - [ ] `matmul`
  - [ ] `attention`
  - [ ] `residual`
- [ ] layers' forwards and backwards passes
  - [x] `GELU`
  - [x] `softmax`
  - [x] `crossentropy`
  - [x] `encoder`
  - [ ] `layernorm`
  - [ ] `matmul` (linear/fully-connected/dense)
  - [ ] `attention`
  - [ ] `residual`
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
