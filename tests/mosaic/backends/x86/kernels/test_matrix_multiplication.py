# ruff: noqa: E402
from __future__ import annotations

import os

os.environ["MKL_NUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS "] = "1"

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
from mosaic.tilelab.layout import DefaultLayout, TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, propagate_tile_views, ScalarTileLevel
from mosaic.tilelab.tile import create_tile_config, to_tilized_array, from_tilized_array
from mosaic.backends.x86.kernels import matrix_multiplication
from mosaic.backends.x86.compile import compile_shared_library

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def compute_gflops(input_a_shape, input_b_shape, execution_times):
    return ((2 * math.prod(input_a_shape[-2:]) * input_b_shape[-1]) / (execution_times.mean() / 1e3)) / 1e9


def run_numpy(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run Numpy")

    def run(np_input_a, np_input_b):
        output = np_input_a @ np_input_b
        return output

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


def run_torch(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

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


def run_python(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run Python")

    def run(np_input_a, np_input_b):
        output = np.zeros((input_a_shape[0], input_b_shape[1]), dtype=np_input_a.dtype)
        for m in range(input_a_shape[0]):
            for n in range(input_b_shape[1]):
                for k in range(input_a_shape[1]):
                    output[m, n] += np_input_a[m, k] * np_input_b[k, n]
        return output

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
    l1_cache_b_layout,
    scalar_b_layout,
    use_avx_manually,
    enable_profiling,
):
    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating composit graph")
    input_a_var = cnp.ndarray(name="input_a_var", shape=input_a_shape)
    input_b_var = cnp.ndarray(name="input_b_var", shape=input_b_shape)
    output_var = input_a_var @ input_b_var

    logger.info("Propagate tile views and create tile metadatas")
    tile_views = propagate_tile_views(
        output_var.graph,
        inputs={
            input_a_var: [
                TileLevel(level_name="l1_cache", tile_shape=l1_cache_a_shape),
                ScalarTileLevel(level_name="scalar", rank=len(l1_cache_a_shape)),
            ],
            input_b_var: [
                TileLevel(level_name="l1_cache", tile_shape=l1_cache_b_shape, layout=l1_cache_b_layout),
                ScalarTileLevel(level_name="scalar", rank=len(l1_cache_b_shape), layout=scalar_b_layout),
            ],
        },
    )
    input_a_tile_config = create_tile_config(tile_views[input_a_var])
    input_b_tile_config = create_tile_config(tile_views[input_b_var])
    output_tile_config = create_tile_config(tile_views[output_var])

    logger.info("Generate kernel")

    kernel_name, kernel_module = matrix_multiplication.generate_module(
        [input_a_tile_config, input_b_tile_config],
        output_tile_config,
        [input_a_var.dtype, input_b_var.dtype],
        output_var.dtype,
        use_avx_manually=use_avx_manually,
        enable_tracy=enable_profiling,
    )
    source_file_name = (test_output_path / kernel_name).with_suffix(".cpp")
    kernel_module.save(source_file_name)

    logger.info("Compile kernel as shared library")
    shared_library_file = compile_shared_library(source_file_name, enable_tracy=enable_profiling)

    logger.info("Load kernel")
    shared_library = cdll.LoadLibrary(shared_library_file)
    run_kernel = getattr(shared_library, kernel_name)

    transpose_order = list(range(len(input_b_var.shape)))
    transpose_order[-2:] = reversed(transpose_order[-2:])

    def run(np_input_a, np_input_b):
        input_a_flat_array = to_tilized_array(np_input_a, input_a_tile_config)
        input_b_flat_array = to_tilized_array(np_input_b, input_b_tile_config)
        output_flat_array = np.zeros((math.prod(output_var.shape),), dtype=input_a_flat_array.dtype)
        run_kernel(
            cast_numpy_array_to_pointer(input_a_flat_array),
            cast_numpy_array_to_pointer(input_b_flat_array),
            cast_numpy_array_to_pointer(output_flat_array),
        )
        return from_tilized_array(output_flat_array, output_tile_config)

    logger.info("Run Comparison")
    np_input_a = np.random.uniform(-0.5, 0.5, input_a_var.shape).astype(np.float32)
    np_input_b = np.random.uniform(-0.5, 0.5, input_b_var.shape).astype(np.float32)
    assert np.allclose(run(np_input_a, np_input_b), np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

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


def run_matrix_multiplication(
    test_output_path,
    num_iterations: int,
    compare_against_others: bool,
    use_avx_manually: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
    l1_cache_b_layout,
    scalar_b_layout,
    enable_profiling=False,
):
    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_a_shape,
        input_b_shape,
        l1_cache_a_shape=l1_cache_a_shape,
        l1_cache_b_shape=l1_cache_b_shape,
        l1_cache_b_layout=l1_cache_b_layout,
        scalar_b_layout=scalar_b_layout,
        use_avx_manually=use_avx_manually,
        enable_profiling=enable_profiling,
    )

    fig, ax = plt.subplots()
    ax.plot(cnp_execution_times, color="green")

    if compare_against_others:
        torch_execution_times = run_torch(num_iterations, input_a_shape, input_b_shape)
        numpy_execution_times = run_numpy(num_iterations, input_a_shape, input_b_shape)

        ax.plot(torch_execution_times, color="red")
        ax.plot(numpy_execution_times, color="blue")

        logger.info(f"{compute_gflops(input_a_shape, input_b_shape, torch_execution_times)} GFLOPS (torch)")
        logger.info(f"{compute_gflops(input_a_shape, input_b_shape, numpy_execution_times)} GFLOPS (numpy)")
    logger.info(f"{compute_gflops(input_a_shape, input_b_shape, cnp_execution_times)} GFLOPS (composit)")

    def center_y_axis(axes):
        y_max = np.abs(axes.get_ylim()).max()
        axes.set_ylim(ymin=0, ymax=y_max)

    center_y_axis(ax)
    fig.savefig(test_output_path / "execution_times.png")
    fig.clf()


@pytest.mark.parametrize("num_iterations", [1000])
@pytest.mark.parametrize("compare_against_others", [False])
@pytest.mark.parametrize("use_avx_manually", [False, True])
@pytest.mark.parametrize("input_a_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 64, 64)])
@pytest.mark.parametrize("input_b_shape", [(128, 128)])
@pytest.mark.parametrize("l1_cache_b_shape", [(64, 64)])
@pytest.mark.parametrize("l1_cache_b_layout", [DefaultLayout(), TransposedLayout(order=(1, 0))])
@pytest.mark.parametrize("scalar_b_layout", [DefaultLayout(), TransposedLayout(order=(1, 0))])
def test_matrix_multiplication(
    request,
    num_iterations,
    compare_against_others: bool,
    use_avx_manually: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
    l1_cache_b_layout,
    scalar_b_layout,
):
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_matrix_multiplication(
        test_output_path,
        num_iterations,
        compare_against_others,
        use_avx_manually,
        input_a_shape,
        l1_cache_a_shape,
        input_b_shape,
        l1_cache_b_shape,
        l1_cache_b_layout,
        scalar_b_layout,
    )


@pytest.mark.parametrize("num_iterations", [1000])
@pytest.mark.parametrize("compare_against_others", [False])
@pytest.mark.parametrize("use_avx_manually", [False, True])
@pytest.mark.parametrize("input_a_shape", [(1, 4, 128, 128)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 1, 64, 64)])
@pytest.mark.parametrize("input_b_shape", [(1, 4, 128, 128)])
@pytest.mark.parametrize("l1_cache_b_shape", [(1, 1, 64, 64)])
@pytest.mark.parametrize("l1_cache_b_layout", [DefaultLayout(), TransposedLayout(order=(0, 1, 3, 2))])
@pytest.mark.parametrize("scalar_b_layout", [DefaultLayout(), TransposedLayout(order=(0, 1, 3, 2))])
def test_batched_matrix_multiplication(
    request,
    num_iterations,
    compare_against_others: bool,
    use_avx_manually: bool,
    input_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
    l1_cache_b_layout,
    scalar_b_layout,
):
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_matrix_multiplication(
        test_output_path,
        num_iterations,
        compare_against_others,
        use_avx_manually,
        input_a_shape,
        l1_cache_a_shape,
        input_b_shape,
        l1_cache_b_shape,
        l1_cache_b_layout,
        scalar_b_layout,
    )


def test_modular_benchmark(request):
    np.random.seed(0)

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    m_size = k_size = n_size = 512
    tile_m_size = tile_k_size = tile_n_size = 64
    python_m_size = python_k_size = python_n_size = 128

    num_iterations = 2
    use_avx_manually = True
    input_a_shape = (m_size, k_size)
    l1_cache_a_shape = (tile_m_size, tile_k_size)
    input_b_shape = (k_size, n_size)
    l1_cache_b_shape = (tile_k_size, tile_n_size)
    l1_cache_b_layout = TransposedLayout(order=(1, 0))
    scalar_b_layout = TransposedLayout(order=(1, 0))
    enable_profiling = False

    cnp_execution_times = run_cnp_kernel(
        num_iterations,
        test_output_path,
        input_a_shape,
        input_b_shape,
        l1_cache_a_shape=l1_cache_a_shape,
        l1_cache_b_shape=l1_cache_b_shape,
        l1_cache_b_layout=l1_cache_b_layout,
        scalar_b_layout=scalar_b_layout,
        use_avx_manually=use_avx_manually,
        enable_profiling=enable_profiling,
    )

    numpy_execution_times = run_numpy(num_iterations, input_a_shape, input_b_shape)
    torch_execution_times = run_torch(num_iterations, input_a_shape, input_b_shape)
    python_execution_times = run_python(num_iterations, (python_m_size, python_k_size), (python_k_size, python_n_size))

    python_flops = compute_gflops(
        (python_m_size, python_k_size), (python_k_size, python_n_size), python_execution_times
    )
    numpy_flops = compute_gflops(input_a_shape, input_b_shape, numpy_execution_times)
    torch_flops = compute_gflops(input_a_shape, input_b_shape, torch_execution_times)
    composit_flops = compute_gflops(input_a_shape, input_b_shape, cnp_execution_times)

    logger.info(f"python:   {python_flops:.5f} GFLOP/s")
    logger.info(f"numpy:    {numpy_flops:.5f} GFLOP/s, a {numpy_flops / python_flops:.5f}x speedup over python")
    logger.info(f"torch:    {torch_flops:.5f} GFLOP/s, a {torch_flops / python_flops:.5f}x speedup over python")
    logger.info(f"composit: {composit_flops:.5f} GFLOP/s, a {composit_flops / python_flops:.5f}x speedup over python")


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
        compare_against_others=True,
        use_avx_manually=True,
        input_a_shape=(batch_size, sequence_size, m_size, k_size),
        l1_cache_a_shape=(batch_size, sequence_size, tile_m_size, tile_k_size),
        input_b_shape=(batch_size, sequence_size, k_size, n_size),
        l1_cache_b_shape=(batch_size, sequence_size, tile_k_size, tile_n_size),
        l1_cache_b_layout=TransposedLayout(order=(0, 1, 3, 2)),
        scalar_b_layout=TransposedLayout(order=(0, 1, 3, 2)),
        enable_profiling=False,
    )
