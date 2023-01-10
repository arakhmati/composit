import pytest

import operator

import numpy as np
import torch

import persistent_numpy as pnp


def test_matmul_autograd():

    input_0_shape = (5, 25, 15)
    input_1_shape = (15, 30)

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape)
    torch_output = torch_input_0 @ torch_input_1

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    output_var = input_0_var @ input_1_var

    gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var],
        {input_0_var: torch_input_0.detach().numpy(), input_1_var: torch_input_1.detach().numpy()},
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(gradient, torch_input_0.grad.detach().numpy())


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(5, 25, 15), (5, 1, 1)])
def test_elementwise_binary_autograd(operation, input_0_shape, input_1_shape):

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape, requires_grad=True)
    torch_output = operation(torch_input_0, torch_input_1)

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    output_var = operation(input_0_var, input_1_var)

    input_0_gradient, input_1_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var, input_1_var],
        {input_0_var: torch_input_0.detach().numpy(), input_1_var: torch_input_1.detach().numpy()},
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(input_0_gradient, torch_input_0.grad.detach().numpy())
    assert np.allclose(input_1_gradient, torch_input_1.grad.detach().numpy())


@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(15, 30)])
def test_matmul_add_subtract_autograd(input_0_shape, input_1_shape):

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape, requires_grad=True)
    torch_output = torch_input_0 @ torch_input_1
    torch_input_2 = torch.rand(torch_output.shape, requires_grad=True)
    torch_output = torch_output + torch_input_2
    torch_input_3 = torch.rand(torch_output.shape, requires_grad=True)
    torch_output = torch_output - torch_input_3

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    input_2_var = pnp.nn.variable(name="input_2_var", shape=torch_input_2.detach().numpy().shape)
    input_3_var = pnp.nn.variable(name="input_3_var", shape=torch_input_3.detach().numpy().shape)
    output_var = (input_0_var @ input_1_var) + input_2_var - input_3_var

    input_0_gradient, input_1_gradient, input_2_gradient, input_3_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var, input_1_var, input_2_var, input_3_var],
        {
            input_0_var: torch_input_0.detach().numpy(),
            input_1_var: torch_input_1.detach().numpy(),
            input_2_var: torch_input_2.detach().numpy(),
            input_3_var: torch_input_3.detach().numpy(),
        },
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(input_0_gradient, torch_input_0.grad.detach().numpy())
    assert np.allclose(input_1_gradient, torch_input_1.grad.detach().numpy())
    assert np.allclose(input_2_gradient, torch_input_2.grad.detach().numpy())
    assert np.allclose(input_3_gradient, torch_input_3.grad.detach().numpy())


@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(15, 30)])
def test_matmul_add_subtract_sum_autograd_with_multiple_consumers(input_0_shape, input_1_shape):

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape, requires_grad=True)
    torch_matmul_output = torch_input_0 @ torch_input_1
    torch_input_2 = torch.rand(torch_matmul_output.shape, requires_grad=True)
    torch_add_output = torch_matmul_output + torch_input_2
    torch_input_3 = torch.rand(torch_add_output.shape, requires_grad=True)
    torch_output = torch_add_output + torch_matmul_output - torch_input_3.sum(dim=-1, keepdims=True)

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_0_var = pnp.nn.variable(name="input_0_var", shape=input_0_shape)
    input_1_var = pnp.nn.variable(name="input_1_var", shape=input_1_shape)
    input_2_var = pnp.nn.variable(name="input_2_var", shape=torch_input_2.detach().numpy().shape)
    input_3_var = pnp.nn.variable(name="input_3_var", shape=torch_input_3.detach().numpy().shape)
    matmul_output_var = input_0_var @ input_1_var
    add_output_var = matmul_output_var + input_2_var
    output_var = add_output_var + matmul_output_var - pnp.sum(input_3_var, -1, keepdims=True)

    input_0_gradient, input_1_gradient, input_2_gradient, input_3_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_0_var, input_1_var, input_2_var, input_3_var],
        {
            input_0_var: torch_input_0.detach().numpy(),
            input_1_var: torch_input_1.detach().numpy(),
            input_2_var: torch_input_2.detach().numpy(),
            input_3_var: torch_input_3.detach().numpy(),
        },
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(input_0_gradient, torch_input_0.grad.detach().numpy())
    assert np.allclose(input_1_gradient, torch_input_1.grad.detach().numpy())
    assert np.allclose(input_2_gradient, torch_input_2.grad.detach().numpy())
    assert np.allclose(input_3_gradient, torch_input_3.grad.detach().numpy())


@pytest.mark.parametrize("input_shape,order", [[(5, 25, 15, 3), (0, 3, 1, 2)], [(19, 1, 15, 3, 8), (1, 3, 0, 4, 2)]])
def test_transpose(input_shape, order):

    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.permute(torch_input, order)

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_var = pnp.nn.variable(name="input_var", shape=input_shape)
    output_var = pnp.transpose(input_var, order)

    input_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(input_gradient, torch_input.grad.detach().numpy())


@pytest.mark.parametrize("input_shape,target_shape", [[(5, 25, 15, 3), (125, 45)], [(18, 1, 15, 3, 8), (6, 90, 12)]])
def test_reshape(input_shape, target_shape):

    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.reshape(torch_input, target_shape)

    torch_output_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_output_gradient)

    input_var = pnp.nn.variable(name="input_var", shape=input_shape)
    output_var = pnp.reshape(input_var, target_shape)

    input_gradient = pnp.nn.compute_gradients(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_output_gradient.numpy()},
    )

    assert np.allclose(input_gradient, torch_input.grad.detach().numpy())
