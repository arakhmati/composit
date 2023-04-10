import pytest

import numpy as np

import composit as cnp
from composit.tilelab.tile_view import (
    TilizationLevel,
    create_tile_view,
    retilize_view,
    propagate_tile_views,
)

from composit.tilelab.tile import to_flat_array, from_flat_array, create_tile_metadata


@pytest.mark.parametrize("input_shape", [(4, 32, 32)])
@pytest.mark.parametrize("buffer_tile_shape", [(1, 16, 8), (1, 8, 8)])
@pytest.mark.parametrize("block_tile_shape", [(1, 8, 4), (1, 4, 4)])
@pytest.mark.parametrize("tile_shape", [(1, 4, 4), (1, 2, 4), (1, 2, 2)])
@pytest.mark.parametrize("new_buffer_tile_shape", [(1, 16, 16), (1, 16, 8), (4, 16, 8)])
@pytest.mark.parametrize("new_block_tile_shape", [(1, 8, 8), (1, 8, 4), (1, 8, 4)])
@pytest.mark.parametrize("new_tile_shape", [(1, 4, 4)])
def test_concatenate(
    input_shape,
    buffer_tile_shape,
    block_tile_shape,
    tile_shape,
    new_buffer_tile_shape,
    new_block_tile_shape,
    new_tile_shape,
):

    np_input = np.random.uniform(-0.5, 0.5, input_shape)

    view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=tile_shape),
        ],
    )

    different_view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=new_buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=new_block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=new_tile_shape),
        ],
    )

    retilized_view, steps = retilize_view(view, different_view.hierarchy)

    assert retilized_view == different_view
    assert steps is None


@pytest.mark.parametrize("input_shape", [(1, 32, 32)])
@pytest.mark.parametrize("buffer_tile_shape", [(1, 16, 32), (1, 16, 16)])
@pytest.mark.parametrize("block_tile_shape", [(1, 8, 8)])
@pytest.mark.parametrize("tile_shape", [(1, 4, 4)])
@pytest.mark.parametrize("new_buffer_tile_shape", [(1, 16, 8), (1, 8, 8)])
@pytest.mark.parametrize("new_block_tile_shape", [(1, 4, 8), (1, 4, 4)])
@pytest.mark.parametrize("new_tile_shape", [(1, 4, 4), (1, 2, 2), (1, 2, 1)])
def test_slice(
    input_shape,
    buffer_tile_shape,
    block_tile_shape,
    tile_shape,
    new_buffer_tile_shape,
    new_block_tile_shape,
    new_tile_shape,
):

    np_input = np.random.uniform(-0.5, 0.5, input_shape)

    view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=tile_shape),
        ],
    )

    different_view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=new_buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=new_block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=new_tile_shape),
        ],
    )

    retilized_view, steps = retilize_view(view, different_view.hierarchy)

    assert retilized_view == different_view
    assert steps is None


@pytest.mark.parametrize("input_shape", [(1, 16, 16)])
@pytest.mark.parametrize("buffer_tile_shape", [(1, 16, 16)])
@pytest.mark.parametrize("block_tile_shape", [(1, 8, 8)])
@pytest.mark.parametrize("tile_shape", [(1, 4, 2)])
@pytest.mark.parametrize("new_buffer_tile_shape", [(1, 16, 8)])
@pytest.mark.parametrize("new_block_tile_shape", [(1, 8, 4)])
@pytest.mark.parametrize("new_tile_shape", [(1, 4, 4)])
def test_buffer_slice_block_slice_tile_concatenate(
    input_shape,
    buffer_tile_shape,
    block_tile_shape,
    tile_shape,
    new_buffer_tile_shape,
    new_block_tile_shape,
    new_tile_shape,
):

    np_input = np.random.uniform(-0.5, 0.5, input_shape)

    view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=tile_shape),
        ],
    )

    different_view = create_tile_view(
        np_input.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=new_buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=new_block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=new_tile_shape),
        ],
    )

    retilized_view, steps = retilize_view(view, different_view.hierarchy)

    assert retilized_view == different_view
    assert steps is None


@pytest.mark.parametrize("input_0_shape", [(1, 32, 64)])
@pytest.mark.parametrize("input_1_shape", [(64, 16)])
def test_matmul_add_subtract_sum(input_0_shape, input_1_shape):

    matmul_shape = input_0_shape[:-1] + input_1_shape[-1:]

    input_var_0 = cnp.nn.variable(name="input_var_0", shape=input_0_shape)
    input_var_1 = cnp.nn.variable(name="input_var_1", shape=input_1_shape)
    input_var_2 = cnp.nn.variable(name="input_var_2", shape=matmul_shape)
    input_var_3 = cnp.nn.variable(name="input_var_3", shape=matmul_shape)
    matmul_output_var = input_var_0 @ input_var_1
    add_output_var = matmul_output_var + input_var_2
    output_var = add_output_var + matmul_output_var - cnp.sum(input_var_3, -1, keepdims=True)

    evaluate_inputs = {
        input_var_0: np.random.uniform(-0.5, 0.5, input_var_0.shape),
        input_var_1: np.random.uniform(-0.5, 0.5, input_var_1.shape),
        input_var_2: np.random.uniform(-0.5, 0.5, input_var_2.shape),
        input_var_3: np.random.uniform(-0.5, 0.5, input_var_3.shape),
    }

    output, cache = cnp.nn.evaluate(output_var, inputs=evaluate_inputs, return_cache=True)

    input_var_to_scheme = {
        input_var_0: [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
        ],
        input_var_1: [
            TilizationLevel(level_name="buffer", tile_shape=(16, 8)),
            TilizationLevel(level_name="block", tile_shape=(8, 4)),
            TilizationLevel(level_name="tile", tile_shape=(4, 4)),
        ],
        input_var_2: [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 2)),
        ],
        input_var_3: [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 2)),
        ],
    }

    tile_views = propagate_tile_views(output_var, inputs=input_var_to_scheme)

    assert tile_views[output_var] == create_tile_view(
        output.shape,
        [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 8)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 4)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
        ],
    )

    output_tile_metadata = create_tile_metadata(output.shape, tile_views[output_var].hierarchy)

    output_flat_array = to_flat_array(output, output_tile_metadata)
    assert np.allclose(output, from_flat_array(output_flat_array, output_tile_metadata))
