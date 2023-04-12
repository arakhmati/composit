import pytest

import numpy as np
import torch.nn.functional

import composit as cnp
import composit.nn


@pytest.mark.parametrize("input_0_shape", [(5, 25, 15)])
@pytest.mark.parametrize("input_1_shape", [(15, 30)])
def test_matmul_add_subtract_sum_autograd_with_multiple_consumers(input_0_shape, input_1_shape):
    matmul_shape = input_0_shape[:-1] + input_1_shape[-1:]

    np_input_0 = np.random.uniform(-0.5, 0.5, input_0_shape)
    np_input_1 = np.random.uniform(-0.5, 0.5, input_1_shape)
    np_input_2 = np.random.uniform(-0.5, 0.5, matmul_shape)
    np_input_3 = np.random.uniform(-0.5, 0.5, matmul_shape)

    torch_input_0 = torch.from_numpy(np_input_0)
    torch_input_1 = torch.from_numpy(np_input_1)
    torch_input_2 = torch.from_numpy(np_input_2)
    torch_input_3 = torch.from_numpy(np_input_3)
    torch_matmul_output = torch_input_0 @ torch_input_1
    torch_add_output = torch_matmul_output + torch_input_2
    torch_output = torch_add_output + torch_matmul_output - torch_input_3.sum(dim=-1, keepdims=True)

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=np_input_0.shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=np_input_1.shape)
    input_var_2 = cnp.nn.variable(name="input_var_2", shape=np_input_2.shape)
    input_var_3 = cnp.nn.variable(name="input_var_3", shape=np_input_3.shape)
    matmul_output_var = input_var_0 @ input_var_1
    add_output_var = matmul_output_var + input_var_2
    output_var = add_output_var + matmul_output_var - cnp.sum(input_var_3, -1, keepdims=True)

    output = cnp.nn.evaluate(
        output_var,
        inputs={
            input_var_0: np_input_0,
            input_var_1: np_input_1,
            input_var_2: np_input_2,
            input_var_3: np_input_3,
        },
    )

    assert np.allclose(output, torch_output.detach().numpy())
