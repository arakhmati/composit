import sys

import persistent_numpy as pnp

THIS_MODULE = sys.modules[__name__]


def matmul_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]

    input_var_1_axes = list(range(input_var_1.rank))
    input_var_1_axes[-2], input_var_1_axes[-1] = input_var_1_axes[-1], input_var_1_axes[-2]

    input_var_0_axes = list(range(input_var_0.rank))
    input_var_0_axes[-2], input_var_0_axes[-1] = input_var_0_axes[-1], input_var_0_axes[-2]

    outgoing_gradient_0 = incoming_gradient @ pnp.transpose(input_var_1, axes=tuple(input_var_1_axes))
    outgoing_gradient_1 = pnp.transpose(input_var_0, axes=tuple(input_var_0_axes)) @ incoming_gradient

    reduction_axes = tuple(range(outgoing_gradient_1.rank))[: -(input_var_1.rank - outgoing_gradient_1.rank)]
    if reduction_axes:
        outgoing_gradient_1 = pnp.sum(outgoing_gradient_1, axis=reduction_axes, keepdims=False)
    return outgoing_gradient_0, outgoing_gradient_1


def _maybe_reduce_jacobian(outgoing_gradient, input_var):
    axes = []
    for axis, _ in enumerate(input_var.shape):
        if outgoing_gradient.shape[axis] > input_var.shape[axis]:
            axes.append(axis)
    if axes:
        axes = tuple(axes)
        outgoing_gradient = pnp.sum(outgoing_gradient, axis=axes, keepdims=True)
    return outgoing_gradient


def add_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient
    outgoing_gradient_1 = incoming_gradient
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def subtract_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient
    outgoing_gradient_1 = incoming_gradient * -1
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def multiply_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient * input_var_1
    outgoing_gradient_1 = incoming_gradient * input_var_0
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def divide_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var_0 = forward_input_vars[0]
    input_var_1 = forward_input_vars[1]
    outgoing_gradient_0 = incoming_gradient / input_var_1
    outgoing_gradient_1 = ((incoming_gradient * -1) * input_var_0) / (input_var_1 * input_var_1)
    outgoing_gradient_1 = _maybe_reduce_jacobian(outgoing_gradient_1, input_var_1)
    return outgoing_gradient_0, outgoing_gradient_1


def transpose_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    rank = len(forward_instruction.axes)
    axes = [None] * rank
    for index, axis in enumerate(forward_instruction.axes):
        axes[axis] = index
    axes = tuple(axes)
    outgoing_gradient = pnp.transpose(incoming_gradient, axes)
    return (outgoing_gradient,)


def reshape_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.reshape(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def sum_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def max_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def mean_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def var_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def sqrt_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    outgoing_gradient = pnp.broadcast_to(incoming_gradient, input_var.shape)
    return (outgoing_gradient,)


def gelu_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    return (input_var,)


def exp_jacobian(forward_instruction, incoming_gradient, forward_input_vars):
    # TODO: implement
    input_var = forward_input_vars[0]
    return (input_var,)


def split_jacobian(forward_instruction, incoming_gradients, forward_input_vars):
    return (pnp.concatenate(incoming_gradients, axis=forward_instruction.axis),)


__all__ = [attr for attr in THIS_MODULE.__dict__.keys() if "gradient" in attr]
