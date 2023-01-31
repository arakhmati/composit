import numpy as np
import pytest
import torch.nn.functional

import persistent_numpy as pnp

from persistent_numpy.nn.module import wrap_module, visualize_modules


@wrap_module
def add_and_apply_gelu(input_var):
    output = input_var
    output = output + output
    output = pnp.nn.gelu(output)
    return output


@wrap_module
def split(input_var):
    return pnp.split(input_var, 5, axis=1)


@wrap_module
def concatenate(*input_vars):
    return pnp.concatenate(input_vars, axis=1)


@wrap_module
def final_0(input_var):
    output = input_var
    output = output * output
    outputs = split(output)
    output = concatenate(*outputs)
    output = pnp.exp(output)
    output = pnp.mean(output)
    return output


@wrap_module
def final_1(input_var):
    output = input_var
    output = output * output
    outputs = split(output)
    output = concatenate(concatenate(*outputs[:2]), concatenate(*outputs[2:]))
    output = pnp.exp(output)
    output = pnp.mean(output)
    return output


@wrap_module
def outer_concatenate(*input_vars):
    return pnp.concatenate((concatenate(*input_vars[:2]), concatenate(*input_vars[2:])), axis=1)


@wrap_module
def final_2(input_var):
    output = input_var
    output = output * output
    outputs = split(output)
    output = outer_concatenate(*outputs)
    output = pnp.exp(output)
    output = pnp.mean(output)
    return output


@wrap_module
def split_concatenate_exp(input):
    output = input
    output = pnp.split(output, 5, axis=1)
    output = pnp.concatenate(output, axis=1)
    return pnp.exp(output)


@wrap_module
def final_3(input_var):
    output = input_var
    output = output * output
    output = split_concatenate_exp(output)
    output = pnp.mean(output)
    return output


@pytest.mark.parametrize(
    "final_module",
    [
        final_0,
        final_1,
        final_2,
        final_3,
    ],
)
def test_modules(final_module):
    array = pnp.random.random((5, 25, 15))
    result = add_and_apply_gelu(array)
    result = result * 2
    result = final_module(result)
    result = result / 3
    visualize_modules(result.graph, render=False)

    torch_array = torch.from_numpy(pnp.evaluate(array))
    torch_result = torch_array
    torch_result = torch_result + torch_result
    torch_result = torch.nn.functional.gelu(torch_result)
    torch_result = torch_result * 2
    torch_result = torch_result * torch_result
    torch_result = torch.exp(torch_result)
    torch_result = torch.mean(torch_result)
    torch_result = torch_result / 3
    torch_result = torch_result.numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)


def test_modules_chain_rule():
    input_shape = (5, 25, 15)

    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.nn.functional.gelu(torch_input + torch_input)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = pnp.nn.variable(name="input_var", shape=input_shape)
    output_var = add_and_apply_gelu(input_var)

    # TODO: implement Module_jacobian
    return

    gradients = pnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)
