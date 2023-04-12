import pytest

import numpy as np
import torch

import composit as cnp
from composit.nn.optimizer import apply_gradients, sgd_optimizer


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
    num_iterations, input_shape: tuple[int, ...], parameter_0_shape: tuple[int, ...], learning_rate
):

    output_shape = input_shape[:-1] + parameter_0_shape[-1:]

    np_inputs = [np.random.random(input_shape) for _ in range(num_iterations)]
    np_parameter_0 = np.random.random(parameter_0_shape)
    np_parameter_1 = np.random.random(output_shape)
    np_parameter_2 = np.random.random(output_shape)
    np_incoming_gradients = [np.random.random(output_shape) for _ in range(num_iterations)]

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    parameter_0 = cnp.nn.variable(name="parameter_0", shape=np_parameter_0.shape)
    parameter_1 = cnp.nn.variable(name="parameter_1", shape=np_parameter_1.shape)
    parameter_2 = cnp.nn.variable(name="parameter_2", shape=np_parameter_2.shape)
    output_var = cnp_model(input_var, parameter_0, parameter_1, parameter_2)

    parameters = {
        parameter_0: np_parameter_0,
        parameter_1: np_parameter_1,
        parameter_2: np_parameter_2,
    }

    torch_parameters = [torch.from_numpy(np_parameter.copy()) for np_parameter in parameters.values()]
    for torch_parameter in torch_parameters:
        torch_parameter.requires_grad = True

    for np_input, np_incoming_gradient in zip(np_inputs, np_incoming_gradients):

        gradients = cnp.nn.differentiate(
            [output_var],
            [input_var, parameter_0, parameter_1, parameter_2],
            {
                input_var: np_input,
                **parameters,
            },
            {output_var: np_incoming_gradient},
        )

        parameters = apply_gradients(parameters, gradients, sgd_optimizer(learning_rate=learning_rate))

    torch_optimizer = torch.optim.SGD(torch_parameters, lr=learning_rate)
    for np_input, np_incoming_gradient in zip(np_inputs, np_incoming_gradients):
        torch_optimizer.zero_grad()

        torch_input = torch.from_numpy(np_input.copy()).detach()
        torch_output = torch_model(torch_input, torch_parameters[0], torch_parameters[1], torch_parameters[2])
        torch_incoming_gradient = torch.from_numpy(np_incoming_gradient.copy())
        torch_output.backward(torch_incoming_gradient)

        torch_optimizer.step()

    assert np.allclose(parameters[parameter_0], torch_parameters[0].detach().numpy())
    assert np.allclose(parameters[parameter_1], torch_parameters[1].detach().numpy())
    assert np.allclose(parameters[parameter_2], torch_parameters[2].detach().numpy())
