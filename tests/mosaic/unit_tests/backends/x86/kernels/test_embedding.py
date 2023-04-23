from __future__ import annotations

import pytest

from ctypes import cdll
import math
import pathlib
import time

import torch
from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

import composit as cnp
import composit.nn
from composit.hash import deterministic_hash
from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.tilelab.tile_view import TileLevel, propagate_tile_views
from mosaic.tilelab.tile import create_array_tile_config, to_flat_array, from_flat_array
from mosaic.backends.x86.kernels import embedding
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def run_torch(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    def run(np_input_a, np_input_b):
        torch_a = torch.from_numpy(np_input_a)
        torch_b = torch.from_numpy(np_input_b)
        output = torch.embedding(torch_b, torch_a)
        return output.numpy()

    np_input_a = np.random.randint(0, len(input_b_shape), input_a_shape, dtype=np.int64)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)
    assert np.allclose(
        run(np_input_a, np_input_b), cnp.nn.vectorized_functions.embedding(np_input_a, np_input_b), atol=1e-5, rtol=1e-6
    )

    execution_times = []
    for i in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.randint(0, len(input_b_shape), input_a_shape, dtype=np.int64)
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
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_a_var = cnp.nn.variable(name="input_a_var", shape=input_a_shape, dtype=np.int64)
    input_b_var = cnp.nn.variable(name="input_b_var", shape=input_b_shape, dtype=np.float32)
    output_var = cnp.nn.embedding(input_a_var, input_b_var)

    logger.info("Propagate tile views and create tile metadatas")
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
    kernel_name = embedding.generate_kernel(test_output_path, output_array_tile_config)

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(test_output_path, kernel_name)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    transpose_order = list(range(len(input_b_var.shape)))
    transpose_order[-2:] = reversed(transpose_order[-2:])

    def run(np_input_a, np_input_b):
        input_a_flat_array = to_flat_array(np_input_a, input_a_array_tile_config)
        input_b_flat_array = to_flat_array(np_input_b, input_b_array_tile_config)
        output_flat_array = np.zeros((math.prod(output_var.shape),), dtype=output_var.dtype)
        run_kernel(
            cast_numpy_array_to_pointer(input_a_flat_array),
            cast_numpy_array_to_pointer(input_b_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_flat_array(output_flat_array, output_array_tile_config)

    logger.info("Run Comparison")
    np_input_a = np.random.randint(0, len(input_b_var.shape), input_a_var.shape, dtype=np.int64)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_var.shape).astype(np.float32)
    assert np.allclose(
        run(np_input_a, np_input_b), cnp.nn.vectorized_functions.embedding(np_input_a, np_input_b), atol=1e-5, rtol=1e-6
    )

    logger.info(f"Run Kernel for {num_iterations} iterations")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.randint(0, len(input_b_var.shape), input_a_var.shape, dtype=np.int64)
        np_input_b = np.random.uniform(-0.5, 0.5, input_b_var.shape).astype(np.float32)
        run(np_input_a, np_input_b)

        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_embedding(
    test_output_path,
    num_iterations: int,
    compare_against_torch: bool,
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
    )

    ax.plot(cnp_execution_times, color="green")

    def center_y_axis(axes):
        y_max = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=0, ymax=y_max)

    center_y_axis(ax)
    fig.savefig(test_output_path / "execution_times.png")
    fig.clf()


@pytest.mark.parametrize("num_iterations", [1000])
@pytest.mark.parametrize("compare_against_torch", [True])
@pytest.mark.parametrize("input_a_shape", [(1, 128)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 128), (1, 32)])
@pytest.mark.parametrize("input_b_shape", [(16, 256)])
@pytest.mark.parametrize("l1_cache_b_shape", [(16, 256), (1, 32)])
def test_embedding(
    request,
    num_iterations,
    compare_against_torch: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
):
    np.random.seed(0)
    torch.manual_seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_embedding(
        test_output_path,
        num_iterations,
        compare_against_torch,
        input_a_shape,
        l1_cache_a_shape,
        input_b_shape,
        l1_cache_b_shape,
    )
