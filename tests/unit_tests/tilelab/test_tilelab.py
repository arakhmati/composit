import pytest

import numpy as np

import persistent_numpy as pnp
from persistent_numpy.tilelab import TilizationLevel, tilize, retilize


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

    tilized_input = tilize(
        np_input,
        [
            TilizationLevel(level_name="buffer", tile_shape=buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=tile_shape),
        ],
    )

    differently_tilized_input = tilize(
        np_input,
        [
            TilizationLevel(level_name="buffer", tile_shape=new_buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=new_block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=new_tile_shape),
        ],
    )

    retilized_input = retilize(tilized_input, differently_tilized_input)

    assert retilized_input == differently_tilized_input


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

    tilized_input = tilize(
        np_input,
        [
            TilizationLevel(level_name="buffer", tile_shape=buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=tile_shape),
        ],
    )

    differently_tilized_input = tilize(
        np_input,
        [
            TilizationLevel(level_name="buffer", tile_shape=new_buffer_tile_shape),
            TilizationLevel(level_name="block", tile_shape=new_block_tile_shape),
            TilizationLevel(level_name="tile", tile_shape=new_tile_shape),
        ],
    )

    retilized_input = retilize(tilized_input, differently_tilized_input)

    assert retilized_input == differently_tilized_input


@pytest.mark.parametrize("input_0_shape", [(1, 32, 64)])
@pytest.mark.parametrize("input_1_shape", [(64, 16)])
def test_matmul_add_subtract_sum(input_0_shape, input_1_shape):

    matmul_shape = input_0_shape[:-1] + input_1_shape[-1:]

    np_input_0 = np.random.uniform(-0.5, 0.5, input_0_shape)
    np_input_1 = np.random.uniform(-0.5, 0.5, input_1_shape)
    np_input_2 = np.random.uniform(-0.5, 0.5, matmul_shape)
    np_input_3 = np.random.uniform(-0.5, 0.5, matmul_shape)

    input_var_0 = pnp.nn.variable(name="input_var_0", shape=np_input_0.shape)
    input_var_1 = pnp.nn.variable(name="input_var_1", shape=np_input_1.shape)
    input_var_2 = pnp.nn.variable(name="input_var_2", shape=np_input_2.shape)
    input_var_3 = pnp.nn.variable(name="input_var_3", shape=np_input_3.shape)
    matmul_output_var = input_var_0 @ input_var_1
    add_output_var = matmul_output_var + input_var_2
    output_var = add_output_var + matmul_output_var - pnp.sum(input_var_3, -1, keepdims=True)

    matmul_output, add_output, output = pnp.nn.evaluate(
        matmul_output_var,
        add_output_var,
        output_var,
        inputs={
            input_var_0: np_input_0,
            input_var_1: np_input_1,
            input_var_2: np_input_2,
            input_var_3: np_input_3,
        },
    )

    tilized_input_0 = tilize(
        np_input_0,
        [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
        ],
    )

    tilized_input_1 = tilize(
        np_input_1,
        [
            TilizationLevel(level_name="buffer", tile_shape=(16, 8)),
            TilizationLevel(level_name="block", tile_shape=(8, 4)),
            TilizationLevel(level_name="tile", tile_shape=(4, 4)),
        ],
    )

    input_2_tilization_levels = [
        TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
        TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
        TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
    ]

    tilized_input_2 = tilize(
        np_input_2,
        input_2_tilization_levels,
    )

    tilized_input_3 = tilize(
        np_input_3,
        [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 16)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 8)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
        ],
    )

    tilized_matmul = tilized_input_0 @ tilized_input_1
    retilized_matmul = retilize(tilized_matmul, tilized_input_2)
    tilized_add = retilized_matmul + tilized_input_2
    tilized_output = tilized_add + retilized_matmul - tilized_input_3.sum(axis=-1)

    manually_tilized_matmul = tilize(
        matmul_output,
        [
            TilizationLevel(level_name="buffer", tile_shape=(1, 16, 8)),
            TilizationLevel(level_name="block", tile_shape=(1, 8, 4)),
            TilizationLevel(level_name="tile", tile_shape=(1, 4, 4)),
        ],
    )
    assert manually_tilized_matmul == tilized_matmul

    manually_tilized_add = tilize(
        add_output,
        input_2_tilization_levels,
    )
    assert manually_tilized_add == tilized_add

    manually_tilized_output = tilize(
        output,
        input_2_tilization_levels,
    )
    assert manually_tilized_output == tilized_output
