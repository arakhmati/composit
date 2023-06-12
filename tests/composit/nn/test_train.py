from __future__ import annotations

import pytest

import numpy as np
import torch

import composit as cnp
from composit.nn.optimize import sgd_optimizer


@pytest.mark.parametrize("input_shape", [(5, 25, 15)])
@pytest.mark.parametrize("parameter_shape", [(15, 30)])
@pytest.mark.parametrize("learning_rate", [0.001])
def test_matmul(
    input_shape: tuple[int, ...],
    parameter_shape: tuple[int, ...],
    learning_rate,
):
    input_var = cnp.random.random(input_shape)
    weight_var = cnp.random.random(parameter_shape)
    output_var = input_var @ weight_var
    loss_var = cnp.mean(output_var)

    gradients = cnp.nn.chain_rule(
        {loss_var: loss_var},
        [weight_var],
    )

    (updated_weight_var,) = cnp.nn.optimize([weight_var], gradients, sgd_optimizer(learning_rate=learning_rate))

    assert not np.allclose(cnp.evaluate(weight_var), cnp.evaluate(updated_weight_var))


def cnp_model(input_var, parameter_0, parameter_1, parameter_2):
    matmul_output_var = input_var @ parameter_0
    add_output_var = matmul_output_var + parameter_1
    output_var = add_output_var + matmul_output_var - cnp.sum(parameter_2, -1, keepdims=True)
    return output_var


def torch_model(torch_input, torch_parameter_0, torch_parameter_1, torch_parameter_2):
    torch_matmul_output = torch_input @ torch_parameter_0
    torch_add_output = torch_matmul_output + torch_parameter_1
    torch_output = torch_add_output + torch_matmul_output - torch_parameter_2.sum(dim=-1, keepdims=True)
    return torch_output


@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("input_shape", [(5, 25, 15)])
@pytest.mark.parametrize("parameter_0_shape", [(15, 30)])
@pytest.mark.parametrize("learning_rate", [0.001])
def test_matmul_add_subtract_sum_autograd_with_multiple_consumers(
    num_iterations,
    input_shape: tuple[int, ...],
    parameter_0_shape: tuple[int, ...],
    learning_rate,
):
    output_shape = input_shape[:-1] + parameter_0_shape[-1:]

    np_inputs = [np.random.random(input_shape) for _ in range(num_iterations)]
    np_parameter_0 = np.random.random(parameter_0_shape)
    np_parameter_1 = np.random.random(output_shape)
    np_parameter_2 = np.random.random(output_shape)
    np_incoming_gradients = [np.random.random(output_shape) for _ in range(num_iterations)]

    parameters = [
        cnp.asarray(np_parameter_0, name="parameter_0"),
        cnp.asarray(np_parameter_1, name="parameter_1"),
        cnp.asarray(np_parameter_2, name="parameter_2"),
    ]

    torch_parameters = [torch.from_numpy(cnp.evaluate(parameter).copy()) for parameter in parameters]
    for torch_parameter in torch_parameters:
        torch_parameter.requires_grad = True

    for np_input, np_incoming_gradient in zip(np_inputs, np_incoming_gradients):
        input_var = cnp.asarray(np_input, name="input_var")
        output_var = cnp_model(input_var, *parameters)

        gradients = cnp.nn.chain_rule(
            {output_var: cnp.asarray(np_incoming_gradient)},
            parameters,
        )

        parameters = cnp.nn.optimize(parameters, gradients, sgd_optimizer(learning_rate=learning_rate))

    torch_optimizer = torch.optim.SGD(torch_parameters, lr=learning_rate)
    for np_input, np_incoming_gradient in zip(np_inputs, np_incoming_gradients):
        torch_optimizer.zero_grad()

        torch_input = torch.from_numpy(np_input.copy()).detach()
        torch_output = torch_model(torch_input, torch_parameters[0], torch_parameters[1], torch_parameters[2])
        torch_incoming_gradient = torch.from_numpy(np_incoming_gradient.copy())
        torch_output.backward(torch_incoming_gradient)

        torch_optimizer.step()

    for parameter, torch_parameter in zip(parameters, torch_parameters):
        assert np.allclose(cnp.evaluate(parameter), torch_parameter.detach().numpy())
