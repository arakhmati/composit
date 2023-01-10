import sys

import persistent_numpy as pnp

THIS_MODULE = sys.modules[__name__]


def matmul_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_0_var = forward_input_vars[0]
    input_1_var = forward_input_vars[1]
    input_1_var_axes = list(range(input_1_var.rank))
    input_1_var_axes[-2], input_1_var_axes[-1] = input_1_var_axes[-1], input_1_var_axes[-2]
    input_0_var_axes = list(range(input_0_var.rank))
    input_0_var_axes[-2], input_0_var_axes[-1] = input_0_var_axes[-1], input_0_var_axes[-2]
    input_0_gradient = output_gradient @ pnp.transpose(input_1_var, axes=tuple(input_1_var_axes))
    input_1_gradient = pnp.transpose(input_0_var, axes=tuple(input_0_var_axes)) @ output_gradient
    axes = tuple(range(input_1_gradient.rank))[: -(input_1_var.rank - input_1_gradient.rank)]
    if axes:
        input_1_gradient = pnp.sum(input_1_gradient, axis=axes, keepdims=False)
    return input_0_gradient, input_1_gradient


def _maybe_reduce_jacobian(input_gradient, input_var):
    axes = []
    for axis, _ in enumerate(input_var.shape):
        if input_gradient.shape[axis] > input_var.shape[axis]:
            axes.append(axis)
    if axes:
        axes = tuple(axes)
        input_gradient = pnp.sum(input_gradient, axis=axes, keepdims=True)
    return input_gradient


def add_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient
    input_1_gradient = output_gradient
    input_1_gradient = _maybe_reduce_jacobian(input_1_gradient, input_1_var)
    return input_0_gradient, input_1_gradient


def subtract_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient
    input_1_gradient = output_gradient * -1
    input_1_gradient = _maybe_reduce_jacobian(input_1_gradient, input_1_var)
    return input_0_gradient, input_1_gradient


def multiply_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_0_var = forward_input_vars[0]
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient * input_1_var
    input_1_gradient = output_gradient * input_0_var
    input_1_gradient = _maybe_reduce_jacobian(input_1_gradient, input_1_var)
    return input_0_gradient, input_1_gradient


def divide_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_0_var = forward_input_vars[0]
    input_1_var = forward_input_vars[1]
    input_0_gradient = output_gradient / input_1_var
    input_1_gradient = ((output_gradient * -1) * input_0_var) / (input_1_var * input_1_var)
    input_1_gradient = _maybe_reduce_jacobian(input_1_gradient, input_1_var)
    return input_0_gradient, input_1_gradient


def transpose_jacobian(forward_instruction, output_gradient, forward_input_vars):
    rank = len(forward_instruction.axes)
    axes = [None] * rank
    for index, axis in enumerate(forward_instruction.axes):
        axes[axis] = index
    axes = tuple(axes)
    input_gradient = pnp.transpose(output_gradient, axes)
    return (input_gradient,)


def reshape_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_var = forward_input_vars[0]
    input_gradient = pnp.reshape(output_gradient, input_var.shape)
    return (input_gradient,)


def sum_jacobian(forward_instruction, output_gradient, forward_input_vars):
    input_var = forward_input_vars[0]
    input_gradient = pnp.broadcast_to(output_gradient, input_var.shape)
    return (input_gradient,)


def max_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    input_gradient = pnp.broadcast_to(output_gradient, input_var.shape)
    return (input_gradient,)


def mean_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    input_gradient = pnp.broadcast_to(output_gradient, input_var.shape)
    return (input_gradient,)


def var_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    input_gradient = pnp.broadcast_to(output_gradient, input_var.shape)
    return (input_gradient,)


def sqrt_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    input_gradient = pnp.broadcast_to(output_gradient, input_var.shape)
    return (input_gradient,)


def gelu_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    return (input_var,)


def exp_jacobian(forward_instruction, output_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    return (input_var,)


__all__ = [attr for attr in THIS_MODULE.__dict__.keys() if "gradient" in attr]
