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
import composit.nn
from composit.hash import deterministic_hash
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.tilelab.tile_view import TileLevel, create_tile_view, ScalarTileLevel
from mosaic.tilelab.tile import create_array_tile_config, to_tilized_array, from_tilized_array
from mosaic.backends.x86.kernels import unary_operation
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()

operation_to_np_function = {
    "exp": np.exp,
    "sqrt": np.sqrt,
    "gelu": cnp.nn.vectorized_functions.gelu,
}


def run_torch(num_iterations, input_shape, operation: str):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    operation_to_torch_function = {
        "exp": torch.exp,
        "sqrt": torch.sqrt,
        "gelu": torch.gelu,
    }
    torch_function = operation_to_torch_function[operation]

    np_function = operation_to_np_function[operation]

    def run(np_input):
        torch_input = torch.from_numpy(np_input)
        output = torch_function(torch_input)
        return output.numpy()

    np_input = np.random.uniform(0.0, 0.5, input_shape).astype(np.float32)
    assert np.allclose(run(np_input), np_function(np_input), atol=1e-5, rtol=1e-6)

    execution_times = []
    for i in range(num_iterations):
        start = time.time_ns()

        np_input = np.random.uniform(0.0, 0.5, input_shape).astype(np.float32)
        run(np_input)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


class ScalarTileConfig:
    pass


def run_cnp_kernel(
    num_iterations,
    test_output_path,
    input_shape,
    l1_cache_shape,
    operation,
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_var = cnp.nn.variable(name="input_var", shape=input_shape)
    output_shape = input_var.shape

    logger.info("Create tile views")
    tile_view = create_tile_view(
        input_var.shape,
        [
            TileLevel(level_name="l1_cache", tile_shape=l1_cache_shape),
            ScalarTileLevel(level_name="scalar", rank=len(l1_cache_shape)),
        ],
    )

    logger.info("Create tile metadata")
    input_array_tile_config = create_array_tile_config(tile_view)
    output_array_tile_config = create_array_tile_config(tile_view)

    logger.info("Generate kernel")
    kernel_name = unary_operation.generate_kernel_source_file(
        test_output_path,
        input_array_tile_config,
        operation,
    )

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(test_output_path, kernel_name)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    def run(np_input):
        input_flat_array = to_tilized_array(np_input, input_array_tile_config)
        output_flat_array = np.zeros((math.prod(output_shape),), dtype=input_flat_array.dtype)
        run_kernel(
            cast_numpy_array_to_pointer(input_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_tilized_array(output_flat_array, output_array_tile_config)

    logger.info("Run Comparison")
    np_function = operation_to_np_function[operation]
    np_input = np.random.uniform(0.0, 0.5, input_var.shape).astype(np.float32)
    assert np.allclose(run(np_input), np_function(np_input), atol=1e-5, rtol=1e-6)

    logger.info(f"Run Kernel for {num_iterations} iterations")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input = np.random.uniform(0.0, 0.5, input_var.shape).astype(np.float32)
        run(np_input)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_unary_operation(
    test_output_path,
    num_iterations: int,
    compare_against_torch: bool,
    input_shape: tuple[int, ...],
    l1_cache_shape: tuple[int, ...],
    operation: str,
):
    fig, ax = plt.subplots()
    if compare_against_torch:
        torch_execution_times = run_torch(num_iterations, input_shape, operation)
        ax.plot(torch_execution_times, color="red")

    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_shape,
        l1_cache_shape=l1_cache_shape,
        operation=operation,
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
@pytest.mark.parametrize("input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l1_cache_shape", [(1, 64, 64)])
@pytest.mark.parametrize("operation", ["exp", "sqrt", "gelu"])
def test_unary_operation(
    request,
    num_iterations,
    compare_against_torch: bool,
    input_shape: tuple[int, ...],
    l1_cache_shape: tuple[int, ...],
    operation: str,
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_unary_operation(
        test_output_path,
        num_iterations,
        compare_against_torch,
        input_shape,
        l1_cache_shape,
        operation,
    )


if __name__ == "__main__":
    operation = "exp"

    h = 128
    w = 128

    batch_size = 1
    h_tile = 64
    w_tile = 64

    run_unary_operation(
        FILE_DIR / "test_output" / "custom",
        num_iterations=1000,
        compare_against_torch=True,
        input_shape=(batch_size, h, w),
        l1_cache_shape=(batch_size, h_tile, w_tile),
        operation=operation,
    )
