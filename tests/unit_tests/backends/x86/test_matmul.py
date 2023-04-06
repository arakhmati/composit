from __future__ import annotations

import pytest

from ctypes import cdll, c_float, POINTER
import math
import pathlib
import time
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

import composit as cnp
from composit.hash import deterministic_hash
from composit.tilelab.tile_view import create_tile_view
from composit.tilelab.tilization_level import TilizationLevel
from composit.tilelab.tile import create_tile_metadata, to_flat_array, from_flat_array
from composit.backends.x86.kernels.matmul import generate_kernel

FILE_DIR = pathlib.Path(__file__).parent.resolve()

FLAGS = [
    "-std=c++2a",
    "-Ofast",
    "-march=native",
    "-fno-exceptions",
    "-mavx2",
    "-msse4",
    "-mfma",
    "-maes",
    "-shared",
    "-fPIC",
    "-Wall",
    "-Wno-deprecated",
    "-Wno-unused-function",
    "-Wno-multichar",
    "-Wno-subobject-linkage",
    "-Wno-format",
]


def run_torch(num_iterations, input_a_shape, input_b_shape):
    logger.info("Run torch")
    import torch

    torch.set_num_threads(1)

    # Call once to set up torch data structures
    for _ in range(10):

        np_input_a = np.random.uniform(-0.5, 0.5, input_a_shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)

        torch_a = torch.from_numpy(np_input_a)
        torch_b = torch.from_numpy(np_input_b)

        output = torch_a @ torch_b

        assert np.allclose(output.numpy(), np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

    execution_times = []
    for i in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.uniform(-0.5, 0.5, input_a_shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_b_shape).astype(np.float32)

        torch_a = torch.from_numpy(np_input_a)
        torch_b = torch.from_numpy(np_input_b)
        output = torch_a @ torch_b

        output = output.numpy()

        end = time.time_ns()
        execution_times.append(end - start)

        assert np.allclose(output, np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def compile_kernel(test_output_path):
    kernel_name = "matmul"
    source_file = str(FILE_DIR / f"{kernel_name}.cpp")
    assembly = test_output_path / f"{kernel_name}.s"
    assembly.unlink(missing_ok=True)
    assembly = str(assembly)
    command = ["g++", source_file, "-I", str(test_output_path), *FLAGS, "-S", "-fverbose-asm", "-o", assembly]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    shared_library = test_output_path / f"{kernel_name}.so"
    shared_library.unlink(missing_ok=True)
    shared_library = str(shared_library)
    command = ["g++", assembly, "-fPIC", "-shared", "-o", shared_library]
    logger.info(f"Compile Assembly to Binary: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    return shared_library


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
        input_var_a.shape, [TilizationLevel(level_name="l1_cache", tile_shape=l1_cache_a_shape)]
    )
    input_b_tile_view = create_tile_view(
        input_var_b.shape, [TilizationLevel(level_name="l1_cache", tile_shape=l1_cache_b_shape)]
    )
    output_tile_view = input_a_tile_view @ input_b_tile_view

    logger.info("Create tile metadata")
    input_a_tile_metadata = create_tile_metadata(input_var_a.shape, input_a_tile_view.hierarchy)
    input_b_tile_metadata = create_tile_metadata(input_var_b.shape, input_b_tile_view.hierarchy)
    output_tile_metadata = create_tile_metadata(output_var.shape, output_tile_view.hierarchy)

    test_output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Generate kernel")
    generate_kernel(
        test_output_path,
        input_a_tile_metadata,
        input_b_tile_metadata,
        transpose_b_levels=transpose_b_levels,
        use_avx_manually=use_avx_manually,
    )

    logger.info("Compile kernel")
    shared_library = compile_kernel(test_output_path)

    logger.info("Load kernel")
    matmul_kernel = cdll.LoadLibrary(shared_library)

    def cast_array(flat_array):
        c_float_p = POINTER(c_float)
        return flat_array.ctypes.data_as(c_float_p)

    output_shape = output_var.shape

    logger.info("Run Kernel")
    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()

        np_input_a = np.random.uniform(-0.5, 0.5, input_var_a.shape).astype(np.float32)
        np_input_b = np.random.uniform(-0.5, 0.5, input_var_b.shape).astype(np.float32)

        input_a_flat_array = to_flat_array(np_input_a, input_a_tile_metadata)
        input_b_flat_array = to_flat_array(
            np_input_b, input_b_tile_metadata, transpose_levels=transpose_b_levels, order=(1, 0)
        )
        output_flat_array = np.zeros((math.prod(output_shape),), dtype=input_a_flat_array.dtype)

        matmul_kernel.run(
            cast_array(input_a_flat_array),
            cast_array(input_b_flat_array),
            cast_array(output_flat_array),
        )

        output = from_flat_array(output_flat_array, output_tile_metadata)

        end = time.time_ns()
        execution_times.append(end - start)

        assert np.allclose(output, np_input_a @ np_input_b, atol=1e-5, rtol=1e-6)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_matmul(
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
def test_matmul(
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

    run_matmul(
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
    run_matmul(
        FILE_DIR / "test_output" / "custom",
        num_iterations=25,
        compare_against_torch=True,
        transpose_b_levels=["atomic", "l1_cache"],
        use_avx_manually=True,
        input_a_shape=(1, 128, 128),
        l1_cache_a_shape=(1, 64, 64),
        input_b_shape=(128, 128),
        l1_cache_b_shape=(64, 64),
    )
