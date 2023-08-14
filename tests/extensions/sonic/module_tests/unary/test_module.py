import pytest

import pathlib
from ctypes import cdll

import jinja2
import numpy as np
import torch

from mosaic.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library

from mosaic.aligned_array import create_aligned_array, align_array

TEST_DIRECTORY = pathlib.Path(__file__).parent


def torch_model(input_tensor):
    output = torch.from_numpy(input_tensor)
    output = torch.abs(output)
    output = torch.exp(output)
    output = torch.sqrt(output)
    return output.numpy()


def np_model(input_tensor):
    output = input_tensor
    output = np.abs(output)
    output = np.exp(output)
    output = np.sqrt(output)
    return output


def create_sonic_model(batch_size, height_size, width_size):
    output_file = TEST_DIRECTORY / "test_output" / f"{batch_size}_{height_size}_{width_size}.cpp"
    output_file.unlink(missing_ok=True)

    template_loader = jinja2.FileSystemLoader(searchpath=TEST_DIRECTORY)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("template.jinja.cpp")
    output_text = template.render(
        batch_size=batch_size,
        height_size=height_size,
        width_size=width_size,
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

    def run_model(np_input_tensor):
        np_input_tensor = align_array(np_input_tensor)
        np_output_tensor = create_aligned_array(np_input_tensor.shape, dtype=np_input_tensor.dtype)

        input_buffer = cast_numpy_array_to_pointer(np_input_tensor)
        output_buffer = cast_numpy_array_to_pointer(np_output_tensor)

        run(input_buffer, output_buffer)

        return np_output_tensor

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height_size", [128])
@pytest.mark.parametrize("width_size", [128])
def test_torch_vs_sonic(batch_size, height_size, width_size):
    input_tensor = np.random.randn(batch_size, height_size, width_size).astype(np.float32)

    sonic_model = create_sonic_model(batch_size, height_size, width_size)

    torch_output = torch_model(input_tensor)
    sonic_encoder_output = sonic_model(input_tensor)

    assert np.allclose(
        torch_output, sonic_encoder_output
    ), f"{list(torch_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "numpy", "sonic"])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("height_size", [1024])
@pytest.mark.parametrize("width_size", [1024])
def test_benchmark(benchmark, module, batch_size, height_size, width_size):
    function = None
    if module == "torch":

        def function():
            input_tensor = np.random.randn(batch_size, height_size, width_size).astype(np.float32)
            torch_model(input_tensor)

    elif module == "numpy":

        def function():
            input_tensor = np.random.randn(batch_size, height_size, width_size).astype(np.float32)
            np_model(input_tensor)

    elif module == "sonic":
        sonic_model = create_sonic_model(batch_size, height_size, width_size)

        def function():
            input_tensor = np.random.randn(batch_size, height_size, width_size).astype(np.float32)
            sonic_model(input_tensor)

    benchmark(function)
