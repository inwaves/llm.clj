# llm.clj
A port of llm.c to Clojure.

TODO:
- [] layers' forwards and backwards passes
  - [] `encoder`
  - [] `layernorm`
  - [] `matmul` (linear/fully-connected/dense)
  - [] `attention`
  - [] `GELU`
  - [] `residual`
  - [] `softmax`
  - [] `crossentropy`
- [] model building blocks
  - [] `ParameterTensors`
  - [] `ActivationTensors`
  - [] `GPT2Config`
  - [] `GPT2`
- [] forward function (`gpt2_forward`)
- [] backward function (`gpt2_backward`)
- [] optimiser step (`gpt2_update`)
- [] training logic
  - [] dataset, `DataLoader`
  - [] sampler
  - [] tokenizer
  - [] training loop
