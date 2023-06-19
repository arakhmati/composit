import sys

import numpy as np

import composit as cnp
from composit.nn.vectorized_functions import cdf, pdf

THIS_MODULE = sys.modules[__name__]


def matmul_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]

    input_var_1_axes = list(range(input_var_1.rank))
    input_var_1_axes[-2], input_var_1_axes[-1] = input_var_1_axes[-1], input_var_1_axes[-2]

    input_var_0_axes = list(range(input_var_0.rank))
    input_var_0_axes[-2], input_var_0_axes[-1] = input_var_0_axes[-1], input_var_0_axes[-2]

    outgoing_gradient_0 = incoming_gradient @ cnp.transpose(input_var_1, axes=tuple(input_var_1_axes))
    outgoing_gradient_1 = cnp.transpose(input_var_0, axes=tuple(input_var_0_axes)) @ incoming_gradient

    reduction_axes = tuple(range(outgoing_gradient_1.rank))[: -(input_var_1.rank - outgoing_gradient_1.rank)]
    if reduction_axes:
        outgoing_gradient_1 = cnp.sum(outgoing_gradient_1, axis=reduction_axes, keepdims=False)
    return outgoing_gradient_0, outgoing_gradient_1


def _maybe_reduce_jacobian(outgoing_gradient, input_var):
    axes = []
    if len(input_var.shape) == 1 and len(outgoing_gradient.shape) == 3:
        # TODO: handle this case generically
        return cnp.sum(outgoing_gradient, axis=(0, 1))
    for axis, _ in enumerate(input_var.shape):
        if outgoing_gradient.shape[axis] > input_var.shape[axis]:
            axes.append(axis)
    if axes:
        axes = tuple(axes)
        outgoing_gradient = cnp.sum(outgoing_gradient, axis=axes, keepdims=True)
    return outgoing_gradient


def add_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient
    outgoing_gradient_1 = incoming_gradient
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def subtract_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient
    outgoing_gradient_1 = incoming_gradient * -1
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def multiply_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient * input_var_1
    outgoing_gradient_1 = incoming_gradient * input_var_0
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def divide_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient / input_var_1
    outgoing_gradient_1 = ((incoming_gradient * -1) * input_var_0) / (input_var_1 * input_var_1)
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def transpose_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    rank = len(forward_instruction.axes)
    axes = [None] * rank
    for index, axis in enumerate(forward_instruction.axes):
        axes[axis] = index
    axes = tuple(axes)
    outgoing_gradient = cnp.transpose(incoming_gradient, axes)
    return (outgoing_gradient,)


def reshape_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var = forward_input_vars[0]
    outgoing_gradient = cnp.reshape(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def sum_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var = forward_input_vars[0]
    outgoing_gradient = cnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def max_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    axis: int = forward_instruction.axis
    keepdims: int = forward_instruction.keepdims

    @cnp.nn.wrap_as_instruction()
    def max_jacobian(incoming_gradient, input_var):
        outgoing_gradient = np.broadcast_to(incoming_gradient, input_var.shape).copy()
        max_values = np.max(input_var, axis, keepdims=keepdims)
        outgoing_gradient[input_var != max_values] = 0
        return outgoing_gradient

    input_var = forward_input_vars[0]
    outgoing_gradient = max_jacobian(incoming_gradient, input_var)
    return (outgoing_gradient,)


def mean_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var = forward_input_vars[0]
    axes = list(range(len(input_var.shape)))
    if isinstance(forward_instruction.axis, int):
        axes = [forward_instruction.axis]
    elif isinstance(forward_instruction.axis, tuple):
        axes = forward_instruction.axis
    num_elements = sum([input_var.shape[axis] for axis in axes])
    outgoing_gradient = cnp.broadcast_to(incoming_gradient, input_var.shape) / num_elements
    return (outgoing_gradient,)


def var_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    raise NotImplementedError


def sqrt_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var = forward_input_vars[0]
    outgoing_gradient = cnp.reciprocal(cnp.sqrt(input_var) * 2) * incoming_gradient
    return (outgoing_gradient,)


def gelu_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    @cnp.nn.wrap_as_instruction()
    def gelu_jacobian(incoming_gradient, input_var):
        return incoming_gradient * (cdf(input_var) + input_var * pdf(input_var))

    input_var = forward_input_vars[0]
    outgoing_gradient = gelu_jacobian(incoming_gradient, input_var)
    return (outgoing_gradient,)


def exp_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    incoming_gradient = incoming_gradients[0]

    input_var = forward_input_vars[0]
    outgoing_gradient = cnp.exp(input_var) * incoming_gradient
    return (outgoing_gradient,)


def split_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    outgoing_gradient = cnp.concatenate(incoming_gradients, axis=forward_instruction.axis)
    return (outgoing_gradient,)


__all__ = [attr for attr in THIS_MODULE.__dict__.keys() if "jacobian" in attr]
