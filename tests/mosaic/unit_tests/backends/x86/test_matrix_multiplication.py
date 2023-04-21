from __future__ import annotations

import pytest

from ctypes import cdll
import math
import pathlib
import time

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

import composit as cnp
from composit.hash import deterministic_hash
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.tilelab.tile_view import TileLevel, create_tile_view
from mosaic.tilelab.tile import create_tile_metadata, to_flat_array, from_flat_array
from mosaic.backends.x86.kernels import matrix_multiplication
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def run_torch(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    def run(np_input_a, np_input_b):
        torch_a = torch.from_numpy(np_input_a)
        torch_b = torch.from_numpy(np_input_b)
        output = torch_a @ torch_b
        return output.numpy()

    np_input_a = np.random.uniform(-0.5, 0.5, input_a_shape).astype(np.float32)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)
    assert np.allclose(run(np_input_a, np_input_b), np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

    execution_times = []
    for i in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.uniform(-0.5, 0.5, input_a_shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)
        run(np_input_a, np_input_b)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_cnp_kernel(
    num_iterations,
    test_output_path,
    input_a_shape,
    input_b_shape,
    l1_cache_a_shape,
    l1_cache_b_shape,
    *,
    transpose_b_levels,
    use_avx_manually,
):
    logger.info("Creating composit graph")
    input_var_a = cnp.nn.variable(name="input_var_a", shape=input_a_shape)
    input_var_b = cnp.nn.variable(name="input_var_b", shape=input_b_shape)
    output_var = input_var_a @ input_var_b

    logger.info("Create tile views")
    input_a_tile_view = create_tile_view(
        input_var_a.shape,
        [TileLevel(level_name="l1_cache", tile_shape=l1_cache_a_shape)],
    )
    input_b_tile_view = create_tile_view(
        input_var_b.shape,
        [TileLevel(level_name="l1_cache", tile_shape=l1_cache_b_shape)],
    )
    output_tile_view = input_a_tile_view @ input_b_tile_view

    logger.info("Create tile metadata")
    input_a_tile_metadata = create_tile_metadata(input_a_tile_view)
    input_b_tile_metadata = create_tile_metadata(input_b_tile_view)
    output_tile_metadata = create_tile_metadata(output_tile_view)

    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generate kernel")
    matrix_multiplication.generate_kernel(
        test_output_path,
        input_a_tile_metadata,
        input_b_tile_metadata,
        transpose_b_levels=transpose_b_levels,
        use_avx_manually=use_avx_manually,
    )

    logger.info("Compile kernel as shared library")
    shared_library = compile_shared_library(test_output_path, matrix_multiplication)

    logger.info("Load kernel")
    kernel = cdll.LoadLibrary(shared_library)

    output_shape = output_var.shape

    transpose_order = list(range(len(input_var_b.shape)))
    transpose_order[-2:] = reversed(transpose_order[-2:])

    def run(np_input_a, np_input_b):
        input_a_flat_array = to_flat_array(np_input_a, input_a_tile_metadata)
        input_b_flat_array = to_flat_array(
            np_input_b,
            input_b_tile_metadata,
            transpose_levels=transpose_b_levels,
            order=transpose_order,
        )
        output_flat_array = np.zeros((math.prod(output_shape),), dtype=input_a_flat_array.dtype)
        kernel.run(
            cast_numpy_array_to_pointer(input_a_flat_array),
            cast_numpy_array_to_pointer(input_b_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_flat_array(output_flat_array, output_tile_metadata)

    logger.info("Run Comparison")
    np_input_a = np.random.uniform(-0.5, 0.5, input_var_a.shape).astype(np.float32)
    np_input_b = np.random.uniform(-0.5, 0.5, input_var_b.shape).astype(np.float32)
    assert np.allclose(run(np_input_a, np_input_b), np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

    logger.info(f"Run Kernel for {num_iterations} iterations")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.uniform(-0.5, 0.5, input_var_a.shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_var_b.shape).astype(np.float32)
        run(np_input_a, np_input_b)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_matrix_multiplication(
    test_output_path,
    num_iterations: int,
    compare_against_torch: bool,
    transpose_b_levels: list[str],
    use_avx_manually: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
):
    fig, ax = plt.subplots()
    if compare_against_torch:
        torch_execution_times = run_torch(num_iterations, input_a_shape, input_b_shape)
        ax.plot(torch_execution_times, color="red")

    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_a_shape,
        input_b_shape,
        l1_cache_a_shape=l1_cache_a_shape,
        l1_cache_b_shape=l1_cache_b_shape,
        transpose_b_levels=transpose_b_levels,
        use_avx_manually=use_avx_manually,
    )

    ax.plot(cnp_execution_times, color="green")

    def center_y_axis(axes):
        y_max = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=0, ymax=y_max)

    center_y_axis(ax)
    fig.savefig(test_output_path / "execution_times.png")
    fig.clf()


@pytest.mark.parametrize("num_iterations", [1000])
@pytest.mark.parametrize("compare_against_torch", [False])
@pytest.mark.parametrize("transpose_b_levels", [[], ["atomic"], ["l1_cache"], ["atomic", "l1_cache"]])
@pytest.mark.parametrize("use_avx_manually", [False, True])
@pytest.mark.parametrize("input_a_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 64, 64)])
@pytest.mark.parametrize("input_b_shape", [(128, 128)])
@pytest.mark.parametrize("l1_cache_b_shape", [(64, 64)])
def test_matrix_multiplication(
    request,
    num_iterations,
    compare_against_torch: bool,
    transpose_b_levels: list[str],
    use_avx_manually: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_matrix_multiplication(
        test_output_path,
        num_iterations,
        compare_against_torch,
        transpose_b_levels,
        use_avx_manually,
        input_a_shape,
        l1_cache_a_shape,
        input_b_shape,
        l1_cache_b_shape,
    )


if __name__ == "__main__":
    batch_size = 3
    sequence_size = 4
    m_size = 128
    k_size = 128
    n_size = 128

    tile_batch_size = 1
    tile_sequence_size = 1
    tile_m_size = 64
    tile_k_size = 64
    tile_n_size = 64

    run_matrix_multiplication(
        FILE_DIR / "test_output" / "custom",
        num_iterations=25,
        compare_against_torch=True,
        transpose_b_levels=["atomic", "l1_cache"],
        use_avx_manually=True,
        input_a_shape=(batch_size, sequence_size, m_size, k_size),
        l1_cache_a_shape=(batch_size, sequence_size, tile_m_size, tile_k_size),
        input_b_shape=(batch_size, sequence_size, k_size, n_size),
        l1_cache_b_shape=(batch_size, sequence_size, tile_k_size, tile_n_size),
    )
