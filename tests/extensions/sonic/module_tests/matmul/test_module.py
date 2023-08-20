import pytest

import pathlib
from ctypes import cdll

import jinja2
import numpy as np
import torch

from mosaic.aligned_array import create_aligned_array, align_array
from mosaic.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library

TEST_DIRECTORY = pathlib.Path(__file__).parent


def torch_model(input_tensor, weights):
    input_tensor = torch.from_numpy(input_tensor)
    weights = torch.from_numpy(weights)
    output = input_tensor @ weights
    return output.numpy()


def np_model(input_tensor, weights):
    return input_tensor @ weights


def create_sonic_model(batch_size, m_size, k_size, n_size):
    output_file = TEST_DIRECTORY / "test_output" / f"{batch_size}_{m_size}_{k_size}_{n_size}.cpp"
    output_file.unlink(missing_ok=True)

    template_loader = jinja2.FileSystemLoader(searchpath=TEST_DIRECTORY)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("template.jinja.cpp")
    output_text = template.render(
        batch_size=batch_size,
        m_size=m_size,
        k_size=k_size,
        n_size=n_size,
    )

    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)

    shared_library_file = compile_shared_library(
        output_file,
        include_paths=["src/extensions/sonic/include"],
        flags=["-fconcepts"],
    )

    shared_library = cdll.LoadLibrary(shared_library_file)
    run = getattr(shared_library, "run")

    def run_model(np_input_tensor, np_weights):
        output_shape = np_input_tensor.shape[:-1] + np_weights.shape[-1:]

        np_input_tensor = align_array(np_input_tensor)
        np_output_tensor = create_aligned_array(output_shape, dtype=np_input_tensor.dtype)
        np_output_tensor[:] = 0

        input_buffer = cast_numpy_array_to_pointer(np_input_tensor)
        weights_buffer = cast_numpy_array_to_pointer(np_weights)
        output_buffer = cast_numpy_array_to_pointer(np_output_tensor)

        run(input_buffer, weights_buffer, output_buffer)

        return np_output_tensor

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [128, 3])
def test_torch_vs_sonic(batch_size, m_size, k_size, n_size):
    input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
    weights = np.random.randn(k_size, n_size).astype(np.float32)

    sonic_model = create_sonic_model(batch_size, m_size, k_size, n_size)

    torch_output = torch_model(input_tensor, weights)
    sonic_encoder_output = sonic_model(input_tensor, weights)

    assert np.allclose(
        torch_output, sonic_encoder_output, atol=1e-5, rtol=1e-5
    ), f"{list(torch_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "numpy", "sonic"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [128])
def test_benchmark(benchmark, module, batch_size, m_size, k_size, n_size):
    weights = align_array(np.random.randn(k_size, n_size).astype(np.float32))

    function = None
    if module == "torch":

        def function():
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            torch_model(input_tensor, weights)

    elif module == "numpy":

        def function():
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            np_model(input_tensor, weights)

    elif module == "sonic":
        sonic_model = create_sonic_model(batch_size, m_size, k_size, n_size)

        def function():
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            sonic_model(input_tensor, weights)

    benchmark(function)


@pytest.mark.skip
@pytest.mark.parametrize("batch_size", [1])
def test_compare_frameworks(benchmark, batch_size):
    import time
    import matplotlib.pyplot as plt

    framework_to_average_duration_per_shape = {
        "torch": [],
        "numpy": [],
        "sonic": [],
    }

    for m_size, k_size, n_size in [
        (1, 1, 1),
        (8, 8, 8),
        (16, 16, 16),
        (32, 32, 32),
        (64, 64, 64),
        (128, 128, 128),
        (256, 256, 256),
        (512, 512, 512),
        (768, 768, 768),
        (1024, 1024, 1024),
        (2048, 2048, 2048),
        (4096, 4096, 4096),
    ]:
        weights = align_array(np.random.randn(k_size, n_size).astype(np.float32))
        sonic_model = create_sonic_model(batch_size, m_size, k_size, n_size)

        framework_to_durations = {
            "torch": [],
            "numpy": [],
            "sonic": [],
        }

        def torch_function():
            start = time.time_ns()
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            torch_model(input_tensor, weights)
            end = time.time_ns()
            framework_to_durations["torch"].append(end - start)

        def numpy_function():
            start = time.time_ns()
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            np_model(input_tensor, weights)
            end = time.time_ns()
            framework_to_durations["numpy"].append(end - start)

        def sonic_function():
            start = time.time_ns()
            input_tensor = np.random.randn(batch_size, m_size, k_size).astype(np.float32)
            sonic_model(input_tensor, weights)
            end = time.time_ns()
            framework_to_durations["sonic"].append(end - start)

        for _ in range(5):
            torch_function()
            numpy_function()
            sonic_function()

        for framework, average_duration_per_shape in framework_to_average_duration_per_shape.items():
            durations = framework_to_durations[framework]
            average_duration_per_shape.append(np.mean(durations))

    for framework, average_duration_per_shape in framework_to_average_duration_per_shape.items():
        plt.plot(np.log(average_duration_per_shape), label=framework)
    plt.legend()
    plt.savefig("framework_matmul_comparison.png")
