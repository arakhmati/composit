import sys

import persistent_numpy as pnp

THIS_MODULE = sys.modules[__name__]


def matmul_autograd(forward_instruction, output_gradient, forward_input_vars):
    input_0_var = forward_input_vars[0]
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient @ pnp.transpose(input_1_var, (1, 0))
    input_1_gradient = pnp.transpose(input_0_var, (0, 2, 1)) @ output_gradient
    return input_0_gradient, input_1_gradient


def add_autograd(forward_instruction, output_gradient, forward_input_vars):
    input_0_var = forward_input_vars[0]
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient
    input_1_gradient = output_gradient
    axes = []
    for axis, _ in enumerate(input_1_var.shape):
        if input_1_gradient.shape[axis] > input_1_var.shape[axis]:
            axes.append(axis)
    if axes:
        axes = tuple(axes)
        input_1_gradient = pnp.sum(input_1_gradient, axis=axes, keepdims=True)
    return input_0_gradient, input_1_gradient


__all__ = [attr for attr in THIS_MODULE.__dict__.keys() if "autograd" in attr]
