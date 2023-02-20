from __future__ import annotations

import math

import pytest

import pathlib
import subprocess

from loguru import logger
import matplotlib.pyplot as plt
import numpy as np

import persistent_numpy as pnp
from persistent_numpy.hash import deterministic_hash
from persistent_numpy.tilelab import (
    TilizationLevel,
    tilize,
)
from persistent_numpy.backends.x86.kernels.matmul import generate_kernel, generate_data

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


def run_torch(np_input_a, np_input_b):
    logger.info("Run torch")
    import torch
    import time

    torch_a = torch.from_numpy(np_input_a)
    torch_b = torch.from_numpy(np_input_b)

    # Call once to set up torch data structures
    output = torch_a @ torch_b

    execution_times = []
    for i in range(1000):
        start = time.time_ns()
        output = torch_a @ torch_b
        end = time.time_ns()
        execution_times.append(end - start)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_pnp_kernel(test_output_path):
    kernel_name = "matmul"
    source_file = str(FILE_DIR / f"{kernel_name}.cpp")
    assembly = test_output_path / f"{kernel_name}.s"
    assembly.unlink(missing_ok=True)
    assembly = str(assembly)
    command = ["g++", source_file, "-I", str(test_output_path), *FLAGS, "-S", "-fverbose-asm", "-o", assembly]
    logger.info(f"Compile Source Code to Assembly: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    executable = str(test_output_path / f"{kernel_name}")
    command = ["g++", assembly, "-o", executable]
    logger.info(f"Compile Assembly to Binary: \"{' '.join(command)}\"")
    result = subprocess.run(command)
    assert result.returncode == 0

    logger.info("Run kernel")
    num_iterations = 1000
    execution_times = []
    result = subprocess.run([executable, str(num_iterations)], capture_output=True)
    stdout = result.stdout.decode("utf-8")
    for line in stdout.split("\n"):
        if "execution time" not in line:
            continue
        execution_time = int(line.split()[-2])
        execution_times.append(execution_time)

    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"Maximum Execution Time: {execution_times.max()} milliseconds")
    return execution_times


def run_matmul(
    test_output_path,
    compare_against_torch: bool,
    unroll_levels: list[str],
    transpose_b_levels: list[str],
    input_a_shape: tuple[int, ...],
    l3_cache_a_shape: tuple[int, ...],
    l2_cache_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    avx_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l3_cache_b_shape: tuple[int, ...],
    l2_cache_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
    avx_b_shape: tuple[int, ...],
):
    if math.prod(avx_a_shape) > 64 and "atomic" in unroll_levels:
        pytest.skip("AVX shape is too big for unrolling")

    input_var_a = pnp.nn.variable(name="input_var_a", shape=input_a_shape)
    input_var_b = pnp.nn.variable(name="input_var_b", shape=input_b_shape)
    output_var = input_var_a @ input_var_b

    np.random.seed(0)
    np_input_a = np.random.uniform(-0.5, 0.5, input_var_a.shape)
    np_input_b = np.random.uniform(-0.5, 0.5, input_var_b.shape)

    evaluate_inputs = {
        input_var_a: np_input_a,
        input_var_b: np_input_b,
    }

    input_var_to_scheme = {
        input_var_a: [
            TilizationLevel(level_name="l3_cache", tile_shape=l3_cache_a_shape),
            TilizationLevel(level_name="l2_cache", tile_shape=l2_cache_a_shape),
            TilizationLevel(level_name="l1_cache", tile_shape=l1_cache_a_shape),
            TilizationLevel(level_name="avx", tile_shape=avx_a_shape),
        ],
        input_var_b: [
            TilizationLevel(level_name="l3_cache", tile_shape=l3_cache_b_shape),
            TilizationLevel(level_name="l2_cache", tile_shape=l2_cache_b_shape),
            TilizationLevel(level_name="l1_cache", tile_shape=l1_cache_b_shape),
            TilizationLevel(level_name="avx", tile_shape=avx_b_shape),
        ],
    }

    try:
        tilized_output, cache = tilize(output_var, inputs=input_var_to_scheme, return_cache=True)
    except:
        pytest.skip(f"Tilization Failed")

    test_output_path.mkdir(parents=True, exist_ok=True)

    generate_kernel(
        test_output_path,
        cache[(input_var_a.node, input_var_a.output_index)],
        cache[(input_var_b.node, input_var_b.output_index)],
        unroll_levels=unroll_levels,
        transpose_b_levels=transpose_b_levels,
    )

    generate_data(
        test_output_path,
        cache[(input_var_a.node, input_var_a.output_index)],
        cache[(input_var_b.node, input_var_b.output_index)],
        cache[(output_var.node, output_var.output_index)],
        evaluate_inputs,
        transpose_b_levels=transpose_b_levels,
    )

    if compare_against_torch:
        torch_execution_times = run_torch(np_input_a, np_input_b)
        plt.plot(torch_execution_times, color="red")

    pnp_execution_times = run_pnp_kernel(test_output_path)
    plt.plot(pnp_execution_times, color="green")
    plt.savefig(test_output_path / "execution_times.png")
    plt.clf()


@pytest.mark.parametrize("compare_against_torch", [False])
@pytest.mark.parametrize("unroll_levels", [[], ["atomic"]])
@pytest.mark.parametrize("transpose_b_levels", [[], ["atomic"], ["atomic", "avx", "l1_cache", "l2_cache", "l3_cache"]])
@pytest.mark.parametrize("input_a_shape", [(1, 128, 128)])
@pytest.mark.parametrize("l3_cache_a_shape", [(1, 64, 64)])
@pytest.mark.parametrize("l2_cache_a_shape", [(1, 32, 64)])
@pytest.mark.parametrize("l1_cache_a_shape", [(1, 32, 32)])
@pytest.mark.parametrize("avx_a_shape", [(1, 8, 8), (1, 32, 32)])
@pytest.mark.parametrize("input_b_shape", [(128, 128)])
@pytest.mark.parametrize("l3_cache_b_shape", [(64, 64)])
@pytest.mark.parametrize("l2_cache_b_shape", [(64, 32)])
@pytest.mark.parametrize("l1_cache_b_shape", [(32, 32)])
@pytest.mark.parametrize("avx_b_shape", [(8, 8), (32, 32)])
def test_matmul(
    request,
    compare_against_torch: bool,
    unroll_levels: list[str],
    transpose_b_levels: list[str],
    input_a_shape: tuple[int, ...],
    l3_cache_a_shape: tuple[int, ...],
    l2_cache_a_shape: tuple[int, ...],
    l1_cache_a_shape: tuple[int, ...],
    avx_a_shape: tuple[int, ...],
    input_b_shape: tuple[int, ...],
    l3_cache_b_shape: tuple[int, ...],
    l2_cache_b_shape: tuple[int, ...],
    l1_cache_b_shape: tuple[int, ...],
    avx_b_shape: tuple[int, ...],
):
    if math.prod(avx_a_shape) > 64 and "atomic" in unroll_levels:
        pytest.skip("AVX shape is too big for unrolling")

    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))

    run_matmul(
        test_output_path,
        compare_against_torch,
        unroll_levels,
        transpose_b_levels,
        input_a_shape,
        l3_cache_a_shape,
        l2_cache_a_shape,
        l1_cache_a_shape,
        avx_a_shape,
        input_b_shape,
        l3_cache_b_shape,
        l2_cache_b_shape,
        l1_cache_b_shape,
        avx_b_shape,
    )


if __name__ == "__main__":
    run_matmul(
        FILE_DIR / "test_output" / "custom",
        compare_against_torch=True,
        unroll_levels=[],
        transpose_b_levels=["atomic", "avx", "l1_cache", "l2_cache", "l3_cache"],
        input_a_shape=(1, 128, 128),
        l3_cache_a_shape=(1, 64, 64),
        l2_cache_a_shape=(1, 32, 64),
        l1_cache_a_shape=(1, 32, 32),
        avx_a_shape=(1, 32, 32),
        input_b_shape=(128, 128),
        l3_cache_b_shape=(64, 64),
        l2_cache_b_shape=(64, 32),
        l1_cache_b_shape=(32, 32),
        avx_b_shape=(32, 32),
    )
