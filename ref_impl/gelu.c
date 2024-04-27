#include <math.h>
#include <stdio.h>

#define GELU_SCALING_FACTOR sqrtf(2.0f / M_PI)
void gelu_forward(float *out, float *inp, int N) {
  // (approximate) GeLU elementwise non-linearity in the MLP block of
  // Transformer
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    out[i] = 0.5f * x * (1.0f + tanhf(GELU_SCALING_FACTOR * (x + cube)));
  }
}

void gelu_backward(float *dinp, float *inp, float *dout, int N) {
  for (int i = 0; i < N; i++) {
    float x = inp[i];
    float cube = 0.044715f * x * x * x;
    float tanh_arg = GELU_SCALING_FACTOR * (x + cube);
    float tanh_out = tanhf(tanh_arg);
    float coshf_out = coshf(tanh_arg);
    float sech_out = 1.0f / (coshf_out * coshf_out);
    float local_grad =
        0.5f * (1.0f + tanh_out) + x * 0.5f * sech_out * GELU_SCALING_FACTOR *
                                       (1.0f + 3.0f * 0.044715f * x * x);
    dinp[i] += local_grad * dout[i];
  }
}

void print_inputs_outputs(float *output, float *input, int N) {
  printf("Input    Output\n");
  printf("---------------\n");
  for (int i = 0; i < N; i++) {
    printf("%.4f   %.4f\n", input[i], output[i]);
  }
}

void print_dinputs_inputs_outputs(float *dinput, float *output, float *input,
                                  int N) {
  printf("Input    Output   Dinput\n");
  printf("------------------------\n");
  for (int i = 0; i < N; i++) {
    printf("%.4f   %.4f   %.4f\n", input[i], output[i], dinput[i]);
  }
}
int main(int argc, char *argv[]) {
  float inp[] = {1, 2, 3, 4, 5};
  float outp[] = {0, 0, 0, 0, 0};
  printf("Before function gelu_forward...\n");
  print_inputs_outputs(outp, inp, 5);
  printf("After function gelu_forward...\n");
  gelu_forward(outp, inp, 5);
  print_inputs_outputs(outp, inp, 5);

  // Now let's do backward...
  float dinp[] = {1, 2, 3, 4, 5};
  printf("Before function gelu_backward...\n");
  print_dinputs_inputs_outputs(dinp, outp, inp, 5);

  printf("After function gelu_backward...\n");
  gelu_backward(dinp, inp, outp, 5);
  print_dinputs_inputs_outputs(dinp, outp, inp, 5);

  return 1;
}
