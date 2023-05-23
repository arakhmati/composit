import pytest

import operator

import numpy as np
import torch

import composit as cnp
import composit.nn


def test_matmul_autograd():
    input_0_shape = (5, 25, 15)
    input_1_shape = (15, 30)

    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape)
    torch_output = torch_input_0 @ torch_input_1

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=input_0_shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=input_1_shape)
    output_var = input_var_0 @ input_var_1

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var_0],
        {
            input_var_0: torch_input_0.detach().numpy(),
            input_var_1: torch_input_1.detach().numpy(),
        },
        {output_var: torch_incoming_gradient.numpy()},
    )

    assert np.allclose(gradients[input_var_0], torch_input_0.grad.numpy())


@pytest.mark.parametrize("operation", [operator.add, operator.sub, operator.mul, operator.truediv])
@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(5, 25, 15), (5, 1, 1)])
def test_elementwise_binary_autograd(operation, input_0_shape, input_1_shape):
    torch_input_0 = torch.rand(input_0_shape, requires_grad=True)
    torch_input_1 = torch.rand(input_1_shape, requires_grad=True)
    torch_output = operation(torch_input_0, torch_input_1)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=input_0_shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=input_1_shape)
    output_var = operation(input_var_0, input_var_1)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var_0, input_var_1],
        {
            input_var_0: torch_input_0.detach().numpy(),
            input_var_1: torch_input_1.detach().numpy(),
        },
        {output_var: torch_incoming_gradient.numpy()},
    )

    assert np.allclose(gradients[input_var_0], torch_input_0.grad.numpy())
    assert np.allclose(gradients[input_var_1], torch_input_1.grad.numpy())


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

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=input_0_shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=input_1_shape)
    input_var_2 = cnp.nn.variable(name="input_var_2", shape=torch_input_2.detach().numpy().shape)
    input_var_3 = cnp.nn.variable(name="input_var_3", shape=torch_input_3.detach().numpy().shape)
    output_var = (input_var_0 @ input_var_1) + input_var_2 - input_var_3

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var_0, input_var_1, input_var_2, input_var_3],
        {
            input_var_0: torch_input_0.detach().numpy(),
            input_var_1: torch_input_1.detach().numpy(),
            input_var_2: torch_input_2.detach().numpy(),
            input_var_3: torch_input_3.detach().numpy(),
        },
        {output_var: torch_incoming_gradient.numpy()},
    )

    assert np.allclose(gradients[input_var_0], torch_input_0.grad.numpy())
    assert np.allclose(gradients[input_var_1], torch_input_1.grad.numpy())
    assert np.allclose(gradients[input_var_2], torch_input_2.grad.numpy())
    assert np.allclose(gradients[input_var_3], torch_input_3.grad.numpy())


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

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=input_0_shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=input_1_shape)
    input_var_2 = cnp.nn.variable(name="input_var_2", shape=torch_input_2.detach().numpy().shape)
    input_var_3 = cnp.nn.variable(name="input_var_3", shape=torch_input_3.detach().numpy().shape)
    matmul_output_var = input_var_0 @ input_var_1
    add_output_var = matmul_output_var + input_var_2
    output_var = add_output_var + matmul_output_var - cnp.sum(input_var_3, -1, keepdims=True)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var_0, input_var_1, input_var_2, input_var_3],
        {
            input_var_0: torch_input_0.detach().numpy(),
            input_var_1: torch_input_1.detach().numpy(),
            input_var_2: torch_input_2.detach().numpy(),
            input_var_3: torch_input_3.detach().numpy(),
        },
        {output_var: torch_incoming_gradient.numpy()},
    )

    assert np.allclose(gradients[input_var_0], torch_input_0.grad.numpy())
    assert np.allclose(gradients[input_var_1], torch_input_1.grad.numpy())
    assert np.allclose(gradients[input_var_2], torch_input_2.grad.numpy())
    assert np.allclose(gradients[input_var_3], torch_input_3.grad.numpy())


@pytest.mark.parametrize(
    "input_shape,order",
    [[(5, 25, 15, 3), (0, 3, 1, 2)], [(19, 1, 15, 3, 8), (1, 3, 0, 4, 2)]],
)
def test_transpose(input_shape, order):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.permute(torch_input, order)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.transpose(input_var, order)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize(
    "input_shape,target_shape",
    [[(5, 25, 15, 3), (125, 45)], [(18, 1, 15, 3, 8), (6, 90, 12)]],
)
def test_reshape(input_shape, target_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.reshape(torch_input, target_shape)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.reshape(input_var, target_shape)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape,slice_size,axis", [[(5, 25, 15, 3), 5, 2]])
def test_split(input_shape, slice_size, axis):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_outputs = torch.split(torch_input, slice_size, dim=axis)

    torch_incoming_gradient = torch.rand(torch_outputs[1].shape)
    torch_outputs[1].backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_vars = cnp.split(input_var, indices_or_sections=input_shape[axis] / slice_size, axis=axis)

    gradients = cnp.nn.differentiate(
        [output_vars[1]],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_vars[1]: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape,slice_size,axis", [[(5, 25, 15, 3), 5, 2]])
def test_split_add(input_shape, slice_size, axis):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_outputs = torch.split(torch_input, slice_size, dim=axis)
    torch_output = torch_outputs[0] + torch_outputs[1] + torch_outputs[2]

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_vars = cnp.split(input_var, indices_or_sections=input_shape[axis] / slice_size, axis=axis)
    output_var = output_vars[0] + output_vars[1] + output_vars[2]

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_exp(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.exp(torch_input)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.exp(input_var)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_sqrt(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.sqrt(torch_input)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.sqrt(input_var)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_square(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.square(torch_input)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.square(input_var)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_gelu(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.nn.functional.gelu(torch_input)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.nn.gelu(input_var)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_max(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output, _ = torch.max(torch_input, dim=2, keepdim=True)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.max(input_var, axis=2, keepdims=True)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)


@pytest.mark.parametrize("input_shape", [(5, 25, 15, 3)])
def test_mean(input_shape):
    torch_input = torch.rand(input_shape, requires_grad=True)
    torch_output = torch.mean(torch_input, dim=2, keepdim=True)

    torch_incoming_gradient = torch.rand(torch_output.shape)
    torch_output.backward(torch_incoming_gradient)

    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_var = cnp.mean(input_var, axis=2, keepdims=True)

    gradients = cnp.nn.differentiate(
        [output_var],
        [input_var],
        {input_var: torch_input.detach().numpy()},
        {output_var: torch_incoming_gradient.numpy()},
    )

    torch_outgoing_gradient = torch_input.grad.numpy()
    assert np.allclose(gradients[input_var], torch_outgoing_gradient)
