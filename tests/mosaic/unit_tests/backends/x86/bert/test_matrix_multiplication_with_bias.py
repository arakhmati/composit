import pytest

import pathlib

import numpy as np
import torch

import composit as cnp
import composit.nn
from composit.hash import deterministic_hash
from mosaic.backends.x86.model import compile_to_mosaic_model, evaluate_mosaic_model
from mosaic.tilelab.layout import TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, ScalarTileLevel

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def cnp_model(input_var, weights, bias):
    return input_var @ weights + bias


def torch_model(input_var, weights, bias):
    input_var = torch.from_numpy(input_var)
    weights = torch.from_numpy(weights)
    bias = torch.from_numpy(bias)
    result = input_var @ weights + bias
    return result.numpy()


def specify_input_var_to_scheme(input_var, weights_var, bias_var):
    return {
        input_var: [
            TileLevel(level_name="tile", tile_shape=(1, 32, 32)),
            ScalarTileLevel(level_name="scalar", rank=len(input_var.shape)),
        ],
        weights_var: [
            TileLevel(level_name="tile", tile_shape=(32, 32), layout=TransposedLayout(order=(1, 0))),
            ScalarTileLevel(level_name="scalar", rank=len(weights_var.shape), layout=TransposedLayout(order=(1, 0))),
        ],
        bias_var: [
            TileLevel(level_name="tile", tile_shape=(32,)),
            ScalarTileLevel(level_name="scalar", rank=len(bias_var.shape)),
        ],
    }


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("hidden_size", [768])
def test_matrix_multiplication_with_bias(
    request,
    batch_size: int,
    sequence_size: int,
    hidden_size: int,
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))
    test_output_path.mkdir(parents=True, exist_ok=True)

    np_input = np.random.random((batch_size, sequence_size, hidden_size)).astype(np.float32)
    np_weights = np.random.random((hidden_size, hidden_size)).astype(np.float32)
    np_bias = np.random.random((hidden_size,)).astype(np.float32)

    input_var = cnp.nn.variable(name="input_var", shape=np_input.shape, dtype=np_input.dtype)
    weights_var = cnp.asarray(np_weights)
    bias_var = cnp.asarray(np_bias)
    output_var = cnp_model(input_var, weights_var, bias_var)

    cnp_output = cnp.nn.evaluate(output_var, inputs={input_var: np_input})

    torch_output = torch_model(np_input, np_weights, np_bias)
    assert np.allclose(cnp_output, torch_output)

    input_var_to_scheme = specify_input_var_to_scheme(input_var, weights_var, bias_var)
    mosaic_model = compile_to_mosaic_model(
        output_var, input_var_to_scheme=input_var_to_scheme, output_path=test_output_path, reuse_buffers=True
    )
    mosaic_output = evaluate_mosaic_model(mosaic_model, output_var, inputs={input_var: np_input})
    assert np.allclose(cnp_output, mosaic_output)
