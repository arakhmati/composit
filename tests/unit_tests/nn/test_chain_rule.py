import pytest

import numpy as np
import torch

import persistent_numpy as pnp


def test_matmul_autograd():

    input_0_shape = (5, 25, 15)
    input_1_shape = (15, 30)

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape)
    torch_output = torch_input_0 @ torch_input_1
    torch_output_gradient = torch.rand(tuple([*input_0_shape[:-1], input_1_shape[-1]]))

    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    output_var = input_0_var @ input_1_var

    gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var],
        [(input_0_var, torch_input_0.detach().numpy()), (input_1_var, torch_input_1.detach().numpy())],
        [(output_var, torch_output_gradient.numpy())],
    )

    assert np.allclose(gradient, torch_input_0.grad.detach().numpy())


@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(5, 25, 15), (5, 1, 1)])
def test_add_autograd(input_0_shape, input_1_shape):

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape, requires_grad=True)
    torch_output = torch_input_0 + torch_input_1
    torch_output_gradient = torch.rand(input_0_shape)

    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    output_var = input_0_var + input_1_var

    input_0_gradient, input_1_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var, input_1_var],
        [(input_0_var, torch_input_0.detach().numpy()), (input_1_var, torch_input_1.detach().numpy())],
        [(output_var, torch_output_gradient.numpy())],
    )

    assert np.allclose(input_0_gradient, torch_input_0.grad.detach().numpy())
    assert np.allclose(input_1_gradient, torch_input_1.grad.detach().numpy())
