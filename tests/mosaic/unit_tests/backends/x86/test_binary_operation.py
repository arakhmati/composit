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
from mosaic.backends.x86.kernels.binary_operation import operation_to_python_operator
from mosaic.tilelab.tile_view import TileLevel, create_tile_view, propagate_tile_views
from mosaic.tilelab.tile import create_array_tile_config, to_flat_array, from_flat_array
from mosaic.backends.x86.kernels import binary_operation
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()

operation_to_np_function = {
    "add": np.add,
    "subtract": np.subtract,
    "multiply": np.multiply,
    "divide": np.divide,
}


def run_torch(num_iterations, input_a_shape, input_b_shape, operation: str):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    operation_to_torch_function = {
        "add": torch.add,
        "subtract": torch.subtract,
        "multiply": torch.multiply,
        "divide": torch.divide,
    }
    torch_function = operation_to_torch_function[operation]

    np_function = operation_to_np_function[operation]

    def run(np_input_a, np_input_b):
        torch_a = torch.from_numpy(np_input_a)
        torch_b = torch.from_numpy(np_input_b)
        output = torch_function(torch_a, torch_b)
        return output.numpy()

    np_input_a = np.random.uniform(-0.5, 0.5, input_a_shape).astype(np.float32)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)
    assert np.allclose(run(np_input_a, np_input_b), np_function(np_input_a, np_input_b), atol=1e-5, rtol=1e-6)

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
    operation,
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_a_var = cnp.nn.variable(name="input_a_var", shape=input_a_shape)
    input_b_var = cnp.nn.variable(name="input_b_var", shape=input_b_shape)
    output_var = operation_to_python_operator[operation](input_a_var, input_b_var)

    logger.info("Propagate tile views and create tile metadatas")
    l1_cache_b_shape = [
        input_a_tile_dim if input_a_dim == input_b_dim else 1
        for input_a_dim, input_b_dim, input_a_tile_dim in zip(input_a_shape, input_b_shape, l1_cache_a_shape)
    ]
    tile_views = propagate_tile_views(
        output_var.graph,
        inputs={
            input_a_var: [TileLevel(level_name="l1_cache", tile_shape=l1_cache_a_shape)],
            input_b_var: [TileLevel(level_name="l1_cache", tile_shape=l1_cache_b_shape)],
        },
    )
    input_a_array_tile_config = create_array_tile_config(tile_views[input_a_var])
    input_b_array_tile_config = create_array_tile_config(tile_views[input_b_var])
    output_array_tile_config = create_array_tile_config(tile_views[output_var])

    logger.info("Generate kernel")
    kernel_name = binary_operation.generate_kernel(
        test_output_path,
        input_a_array_tile_config,
        input_b_array_tile_config,
        operation,
    )

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(test_output_path, kernel_name)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    def run(np_input_a, np_input_b):
        input_a_flat_array = to_flat_array(np_input_a, input_a_array_tile_config)
        input_b_flat_array = to_flat_array(np_input_b, input_b_array_tile_config)
        output_flat_array = np.zeros((math.prod(output_var.shape),), dtype=input_a_flat_array.dtype)
        run_kernel(
            cast_numpy_array_to_pointer(input_a_flat_array),
            cast_numpy_array_to_pointer(input_b_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_flat_array(output_flat_array, output_array_tile_config)

    logger.info("Run Comparison")
    np_function = operation_to_np_function[operation]
    np_input_a = np.random.uniform(-0.5, 0.5, input_a_var.shape).astype(np.float32)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_var.shape).astype(np.float32)
    assert np.allclose(run(np_input_a, np_input_b), np_function(np_input_a, np_input_b), atol=1e-5, rtol=1e-6)

    logger.info(f"Run Kernel for {num_iterations} iterations")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.uniform(-0.5, 0.5, input_a_var.shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_b_var.shape).astype(np.float32)
        run(np_input_a, np_input_b)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_binary_operation(
    test_output_path,
    num_iterations: int,
    compare_against_torch: bool,
    input_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    operation: str,
):
    fig, ax = plt.subplots()
    if compare_against_torch:
        torch_execution_times = run_torch(num_iterations, input_a_shape, input_b_shape, operation)
        ax.plot(torch_execution_times, color="red")

    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_a_shape,
        input_b_shape,
        l1_cache_a_shape=l1_cache_a_shape,
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
@pytest.mark.parametrize("input_a_shape", [(1, 128, 128)])
@pytest.mark.parametrize("input_b_shape", [(1, 128, 128), (1, 1, 1), (1, 128, 1)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 64, 64)])
@pytest.mark.parametrize("operation", ["add", "subtract", "multiply", "divide"])
def test_binary_operation(
    request,
    num_iterations,
    compare_against_torch: bool,
    input_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    operation: str,
):
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_binary_operation(
        test_output_path,
        num_iterations,
        compare_against_torch,
        input_a_shape,
        input_b_shape,
        l1_cache_a_shape,
        operation,
    )
