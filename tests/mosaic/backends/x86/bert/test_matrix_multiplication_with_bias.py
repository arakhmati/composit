from __future__ import annotations

import pytest

import pathlib
import random

import numpy as np

import composit as cnp
from composit.hash import deterministic_hash
from mosaic.backends.x86.passes.evaluate import evaluate as mosaic_evaluate
from mosaic.tilelab.layout import TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, ScalarTileLevel

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def cnp_model(input_var, weights, bias):
    return input_var @ weights + bias


def specify_input_var_to_scheme(input_var, weights_var, bias_var, *, tilize_l1_cache):
    if tilize_l1_cache:
        return {
            input_var: (
                TileLevel(level_name="tile", tile_shape=(1, 32, 32)),
                ScalarTileLevel(level_name="scalar", rank=len(input_var.shape)),
            ),
            weights_var: (
                TileLevel(level_name="tile", tile_shape=(32, 32), layout=TransposedLayout(order=(1, 0))),
                ScalarTileLevel(
                    level_name="scalar", rank=len(weights_var.shape), layout=TransposedLayout(order=(1, 0))
                ),
            ),
            bias_var: (
                TileLevel(level_name="tile", tile_shape=(32,)),
                ScalarTileLevel(level_name="scalar", rank=len(bias_var.shape)),
            ),
        }
    else:
        return {
            input_var: (ScalarTileLevel(level_name="scalar", rank=len(input_var.shape)),),
            weights_var: (
                ScalarTileLevel(
                    level_name="scalar", rank=len(weights_var.shape), layout=TransposedLayout(order=(1, 0))
                ),
            ),
            bias_var: (ScalarTileLevel(level_name="scalar", rank=len(bias_var.shape)),),
        }


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("hidden_size", [768])
@pytest.mark.parametrize("fuse_kernels", [False, True])
@pytest.mark.parametrize("tilize_l1_cache", [False, True])
def test_matrix_multiplication_with_bias(
    request,
    batch_size: int,
    sequence_size: int,
    hidden_size: int,
    fuse_kernels: bool,
    tilize_l1_cache: bool,
):
    random.seed(0)
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))
    test_output_path.mkdir(parents=True, exist_ok=True)

    np_input = np.random.random((batch_size, sequence_size, hidden_size)).astype(np.float32)
    np_weights = np.random.random((hidden_size, hidden_size)).astype(np.float32)
    np_bias = np.random.random((hidden_size,)).astype(np.float32)

    input_var = cnp.asarray(np_input)
    weights_var = cnp.asarray(np_weights)
    bias_var = cnp.asarray(np_bias)
    output_var = cnp_model(input_var, weights_var, bias_var)

    cnp_output = cnp.nn.evaluate(output_var)

    input_var_to_scheme = specify_input_var_to_scheme(input_var, weights_var, bias_var, tilize_l1_cache=tilize_l1_cache)
    mosaic_output = mosaic_evaluate(
        output_var,
        input_var_to_scheme=input_var_to_scheme,
        output_path=test_output_path,
        reuse_buffers=True,
        fuse_kernels=fuse_kernels,
    )
    assert np.allclose(cnp_output, mosaic_output)
