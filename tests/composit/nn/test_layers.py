import pytest

import numpy as np
import torch

import composit as cnp
import composit.nn
from composit.nn.layers import layer_norm, group_norm


@pytest.mark.parametrize("input_shape", [(2, 128, 768)])
def test_layer_norm(input_shape: tuple[int, ...]):
    np_input = np.random.random(input_shape).astype(np.float32)

    torch_layer_norm = torch.nn.LayerNorm(normalized_shape=input_shape[-1], dtype=torch.float32).eval()
    torch_result = torch_layer_norm(torch.from_numpy(np_input)).detach().numpy()

    input_var = cnp.nn.variable(name="input", shape=np_input.shape, dtype=np_input.dtype)
    weight = cnp.asarray(torch_layer_norm.weight.detach().numpy())
    bias = cnp.asarray(torch_layer_norm.bias.detach().numpy())
    result_var = layer_norm(input_var, weight, bias)

    result = cnp.nn.evaluate(result_var, inputs={input_var: np_input})
    assert result.shape == torch_result.shape
    assert np.allclose(result, torch_result, atol=1e-5)


@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_channels", [128])
@pytest.mark.parametrize("height", [64])
@pytest.mark.parametrize("width", [16])
@pytest.mark.parametrize("num_groups", [32])
def test_group_norm(batch_size, num_channels, height, width, num_groups):
    np.random.seed(0)
    torch.manual_seed(0)

    np_input = np.random.uniform(-0.5, 0.5, (batch_size, num_channels, height, width)).astype(np.float32)

    torch_layer_norm = torch.nn.GroupNorm(num_channels=num_channels, num_groups=num_groups, dtype=torch.float32).eval()
    torch_result = torch_layer_norm(torch.from_numpy(np_input)).detach().numpy()

    input_var = cnp.nn.variable(name="input", shape=np_input.shape, dtype=np_input.dtype)
    weight = cnp.asarray(torch_layer_norm.weight.detach().numpy().reshape((num_channels, 1, 1)))
    bias = cnp.asarray(torch_layer_norm.bias.detach().numpy().reshape((num_channels, 1, 1)))
    result_var = group_norm(input_var, weight, bias, channel_axis=1, num_groups=num_groups)

    result = cnp.nn.evaluate(result_var, inputs={input_var: np_input})
    assert result.shape == torch_result.shape
    assert np.allclose(result, torch_result, atol=1e-5)
