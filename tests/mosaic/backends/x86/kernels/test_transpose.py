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
from mosaic.tilelab.tile_view import TileLevel, propagate_tile_views, ScalarTileLevel
from mosaic.tilelab.tile import create_tile_config, to_tilized_array, from_tilized_array
from mosaic.backends.x86.kernels import transpose
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def run_torch(num_iterations, input_shape, axes):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    def run(np_input):
        torch_input = torch.from_numpy(np_input)
        output = torch.permute(torch_input, axes)
        return output.numpy()

    np_input = np.random.uniform(-0.5, 0.5, input_shape).astype(np.float32)
    assert np.allclose(run(np_input), np.transpose(np_input, axes=axes), atol=1e-5, rtol=1e-6)

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
    axes,
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_var = cnp.ndarray(name="input_var", shape=input_shape)
    output_var = cnp.transpose(input_var, axes=axes)

    logger.info("Propagate tile views and create tile metadatas")
    tile_views = propagate_tile_views(
        output_var.graph,
        inputs={
            input_var: [
                TileLevel(level_name="l1_cache", tile_shape=l1_cache_shape),
                ScalarTileLevel(level_name="scalar", rank=len(l1_cache_shape)),
            ]
        },
    )
    input_tile_config = create_tile_config(tile_views[input_var])
    output_tile_config = create_tile_config(tile_views[output_var])
    logger.info(input_tile_config)
    logger.info(output_tile_config)

    logger.info("Generate kernel")
    kernel_name, kernel_module = transpose.generate_module(
        [input_tile_config],
        output_tile_config,
        [input_var.dtype],
        output_var.dtype,
        axes,
    )
    source_file_name = (test_output_path / kernel_name).with_suffix(".cpp")
    kernel_module.save(source_file_name)

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(source_file_name)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    def run(np_input):
        input_flat_array = to_tilized_array(np_input, input_tile_config)
        output_flat_array = np.zeros((math.prod(output_var.shape),), dtype=input_flat_array.dtype)
        run_kernel(
            cast_numpy_array_to_pointer(input_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_tilized_array(output_flat_array, output_tile_config)

    logger.info("Run Comparison")
    np_input = np.random.uniform(-0.5, 0.5, input_var.shape).astype(np.float32)
    assert np.allclose(run(np_input), np.transpose(np_input, axes=axes), atol=1e-4, rtol=1e-5)

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


def run_transpose(
    test_output_path,
    num_iterations: int,
    compare_against_torch: bool,
    input_shape: tuple[int, ...],
    l1_cache_shape: tuple[int, ...],
    axes,
):
    fig, ax = plt.subplots()
    if compare_against_torch:
        torch_execution_times = run_torch(num_iterations, input_shape, axes)
        ax.plot(torch_execution_times, color="red")

    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_shape,
        l1_cache_shape=l1_cache_shape,
        axes=axes,
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
@pytest.mark.parametrize("input_shape", [(2, 128, 192, 3)])
@pytest.mark.parametrize("l1_cache_shape", [(1, 64, 32, 3)])
@pytest.mark.parametrize("axes", [(0, 1, 2, 3), (0, 2, 1, 3), (0, 1, 3, 2), (3, 2, 1, 0)])
def test_transpose(
    request,
    num_iterations,
    compare_against_torch: bool,
    input_shape: tuple[int, ...],
    l1_cache_shape: tuple[int, ...],
    axes,
):
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_transpose(
        test_output_path,
        num_iterations,
        compare_against_torch,
        input_shape,
        l1_cache_shape,
        axes,
    )
