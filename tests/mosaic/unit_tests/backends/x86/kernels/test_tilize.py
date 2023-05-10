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
from mosaic.tilelab.tile import create_tile_config, from_tilized_array
from mosaic.backends.x86.kernels import tilize
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def run_cnp_kernel(
    num_iterations,
    test_output_path,
    input_shape,
    l1_cache_shape,
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_var = cnp.nn.variable(name="input_var", shape=input_shape, dtype=np.float32)

    logger.info("Create tile views")
    tile_view = create_tile_view(
        input_var.shape,
        [
            TileLevel(level_name="l1_cache", tile_shape=l1_cache_shape),
            ScalarTileLevel(level_name="scalar", rank=len(l1_cache_shape)),
        ],
    )

    logger.info("Create tile metadata")
    tile_config = create_tile_config(tile_view)

    logger.info("Generate kernels")
    kernel_name, kernel_module = tilize.generate_module(
        [tile_config],
        tile_config,
        [input_var.dtype],
        input_var.dtype,
    )
    source_file_name = (test_output_path / kernel_name).with_suffix(".cpp")
    kernel_module.save(source_file_name)

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(source_file_name)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    def run(np_input):
        input_flat_array = np_input
        output_flat_array = np.zeros((math.prod(input_var.shape),), dtype=input_flat_array.dtype)
        run_kernel(cast_numpy_array_to_pointer(input_flat_array), cast_numpy_array_to_pointer(output_flat_array))
        return from_tilized_array(output_flat_array, tile_config)

    logger.info("Run Comparison")
    np_input = np.random.uniform(0.0, 0.5, input_var.shape).astype(np.float32)
    assert np.allclose(run(np_input), np_input, atol=1e-5, rtol=1e-6)

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


def run_to_and_from_flat_array_operation(
    test_output_path, num_iterations: int, input_shape: tuple[int, ...], l1_cache_shape: tuple[int, ...]
):
    cnp_execution_times = run_cnp_kernel(num_iterations, test_output_path, input_shape, l1_cache_shape)

    fig, ax = plt.subplots()
    ax.plot(cnp_execution_times, color="green")

    def center_y_axis(axes):
        y_max = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=0, ymax=y_max)

    center_y_axis(ax)
    fig.savefig(test_output_path / "execution_times.png")
    fig.clf()


@pytest.mark.parametrize("num_iterations", [1000])
@pytest.mark.parametrize("input_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l1_cache_shape", [(1, 64, 64)])
def test_to_and_from_flat_array_operation(
    request,
    num_iterations,
    input_shape: tuple[int, ...],
    l1_cache_shape: tuple[int, ...],
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_to_and_from_flat_array_operation(test_output_path, num_iterations, input_shape, l1_cache_shape)
