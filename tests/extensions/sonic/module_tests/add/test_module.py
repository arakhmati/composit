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


def torch_model(input_tensor_a, input_tensor_b):
    input_tensor_a = torch.from_numpy(input_tensor_a)
    input_tensor_b = torch.from_numpy(input_tensor_b)
    output = input_tensor_a + input_tensor_b
    return output.numpy()


def np_model(input_tensor_a, input_tensor_b):
    return input_tensor_a + input_tensor_b


def create_sonic_model(size):
    output_file = TEST_DIRECTORY / "test_output" / f"{size}.cpp"
    output_file.unlink(missing_ok=True)

    template_loader = jinja2.FileSystemLoader(searchpath=TEST_DIRECTORY)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("template.jinja.cpp")
    output_text = template.render(size=size)

    output_file.parent.mkdir(exist_ok=True)
    with open(output_file, "w") as f:
        f.write(output_text)

    shared_library_file = compile_shared_library(
        output_file,
        include_paths=["src/extensions/sonic/include"],
        flags=["-fconcepts"],
        compile_assembly=True,
    )

    shared_library = cdll.LoadLibrary(shared_library_file)
    run = getattr(shared_library, "run")

    def run_model(np_input_tensor_a, np_input_tensor_b):
        np_input_tensor_a = align_array(np_input_tensor_a)
        np_input_tensor_b = align_array(np_input_tensor_b)
        np_output_tensor = create_aligned_array(np_input_tensor_a.shape, dtype=np_input_tensor_a.dtype)
        np_output_tensor[:] = 0

        input_buffer_a = cast_numpy_array_to_pointer(np_input_tensor_a)
        input_buffer_b = cast_numpy_array_to_pointer(np_input_tensor_b)
        output_buffer = cast_numpy_array_to_pointer(np_output_tensor)

        run(input_buffer_a, input_buffer_b, output_buffer)

        return np_output_tensor

    return run_model


@pytest.mark.parametrize("size", [1 << 20])
def test_torch_vs_sonic(size):
    input_tensor_a = np.random.randn(size).astype(np.float32)
    input_tensor_b = np.random.randn(size).astype(np.float32)

    sonic_model = create_sonic_model(size)

    torch_output = torch_model(input_tensor_a, input_tensor_b)
    sonic_encoder_output = sonic_model(input_tensor_a, input_tensor_b)

    assert np.allclose(
        torch_output, sonic_encoder_output, atol=1e-5, rtol=1e-5
    ), f"{list(torch_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "numpy", "sonic"])
@pytest.mark.parametrize("size", [1 << 20])
def test_benchmark(benchmark, module, size):
    function = None
    if module == "torch":

        def function():
            input_tensor_a = np.random.randn(size).astype(np.float32)
            input_tensor_b = np.random.randn(size).astype(np.float32)
            torch_model(input_tensor_a, input_tensor_b)

    elif module == "numpy":

        def function():
            input_tensor_a = np.random.randn(size).astype(np.float32)
            input_tensor_b = np.random.randn(size).astype(np.float32)
            np_model(input_tensor_a, input_tensor_b)

    elif module == "sonic":
        sonic_model = create_sonic_model(size)

        def function():
            input_tensor_a = np.random.randn(size).astype(np.float32)
            input_tensor_b = np.random.randn(size).astype(np.float32)
            sonic_model(input_tensor_a, input_tensor_b)

    benchmark(function)
