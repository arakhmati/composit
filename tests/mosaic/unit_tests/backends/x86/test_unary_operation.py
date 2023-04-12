from __future__ import annotations

import pytest

from ctypes import cdll, c_float, POINTER
import math
import pathlib
import time

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

import composit as cnp
from composit.hash import deterministic_hash
from mosaic.tilelab.tile_view import create_tile_view
from mosaic.tilelab.tilization_level import TilizationLevel
from mosaic.tilelab.tile import create_tile_metadata, to_flat_array, from_flat_array
from mosaic.backends.x86.kernels import unary_operation
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()

operation_to_np_function = {
    "exp": np.exp,
}


def run_torch(num_iterations, input_shape, operation: str):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    operation_to_torch_function = {
        "exp": torch.exp,
    }
    torch_function = operation_to_torch_function[operation]

    np_function = operation_to_np_function[operation]

    def run(np_input):
        torch_input = torch.from_numpy(np_input)
        output = torch_function(torch_input)
        return output.numpy()

    np_input = np.random.uniform(-0.5, 0.5, input_shape).astype(np.float32)
    assert np.allclose(run(np_input), np_function(np_input), atol=1e-5, rtol=1e-6)

    execution_times = []
    for i in range(num_iterations):
        start = time.time_ns()

        np_input = np.random.uniform(-0.5, 0.5, input_shape).astype(np.float32)
        run(np_input)

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
    input_shape,
    l1_cache_shape,
    operation,
):

    logger.info("Creating composit graph")
    input_var = cnp.nn.variable(name="input_var_a", shape=input_shape)
    output_shape = input_var.shape

    logger.info("Create tile views")
    input_tile_view = create_tile_view(
        input_var.shape, [TilizationLevel(level_name="l1_cache", tile_shape=l1_cache_shape)]
    )
    output_tile_view = input_tile_view

    logger.info("Create tile metadata")
    input_tile_metadata = create_tile_metadata(input_var.shape, input_tile_view.hierarchy)
    output_tile_metadata = create_tile_metadata(output_shape, output_tile_view.hierarchy)

    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generate kernel")
    unary_operation.generate_kernel(
        test_output_path,
        input_tile_metadata,
    )

    logger.info("Compile kernel as shared library")
    shared_library = compile_shared_library(test_output_path, unary_operation)

    logger.info("Load kernel")
    kernel = cdll.LoadLibrary(shared_library)

    def cast_array(flat_array):
        c_float_p = POINTER(c_float)
        return flat_array.ctypes.data_as(c_float_p)

    def run(np_input):
        input_flat_array = to_flat_array(np_input, input_tile_metadata)
        output_flat_array = np.zeros((math.prod(output_shape),), dtype=input_flat_array.dtype)
        kernel.run(
            cast_array(input_flat_array),
            cast_array(output_flat_array),
        )
        return from_flat_array(output_flat_array, output_tile_metadata)

    logger.info("Run Comparison")
    np_function = operation_to_np_function[operation]
    np_input = np.random.uniform(-0.5, 0.5, input_var.shape).astype(np.float32)
    assert np.allclose(run(np_input), np_function(np_input), atol=1e-5, rtol=1e-6)

    logger.info(f"Run Kernel for {num_iterations} iterations")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input = np.random.uniform(-0.5, 0.5, input_var.shape).astype(np.float32)
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
@pytest.mark.parametrize("operation", ["exp"])
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
