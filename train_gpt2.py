import math

GELU_SCALING_FACTOR = math.sqrt(2 / math.pi)


def gelu_forward(out, inp):
    for i in range(len(inp)):
        x = inp[i]
        cube = 0.044715 * x * x * x
        out[i] = 0.5 * x * (1.0 + math.tanh(GELU_SCALING_FACTOR * (x + cube)))


def gelu_backward(dinp, inp, dout):
    for i in range(len(inp)):
        x = inp[i]
        cube = 0.044715 * x * x * x
        tanh_arg = GELU_SCALING_FACTOR * (x + cube)
        tanh_out = math.tanh(tanh_arg)
        coshf_out = math.cosh(tanh_arg)
        sech_out = 1.0 / (coshf_out * coshf_out)
        local_grad = 0.5 * (
            1.0 + tanh_out
        ) + x * 0.5 * sech_out * GELU_SCALING_FACTOR * (1.0 + 3.0 * 0.044715 * x * x)
        dinp[i] += local_grad * dout[i]


def test_gelu_forward():
    inp = [1, 2, 3, 4, 5]
    out = [0] * 5
    gelu_forward(out, inp)
    print(out)


def test_gelu_backward():
    dinp = [1, 2, 3, 4, 5]
    inp = [1, 2, 3, 4, 5]
    dout = [1, 2, 3, 4, 5]
    gelu_backward(dinp, inp, dout)
    print(dinp)


if __name__ == "__main__":
    test_gelu_forward()
    test_gelu_backward()
