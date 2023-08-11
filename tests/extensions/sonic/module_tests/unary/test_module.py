import pytest

import pathlib
import time
from ctypes import cdll

from loguru import logger
import jinja2
import numpy as np
import torch

from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library

TEST_DIRECTORY = pathlib.Path(__file__).parent


def torch_model(input_tensor):
    output = input_tensor
    output = torch.abs(output)
    output = torch.exp(output)
    output = torch.sqrt(output)
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

    def run_model(torch_input_tensor):
        np_input_tensor = torch_input_tensor.numpy()
        np_output_tensor = np.zeros_like(np_input_tensor)

        input_buffer = cast_numpy_array_to_pointer(np_input_tensor)
        output_buffer = cast_numpy_array_to_pointer(np_output_tensor)

        run(input_buffer, output_buffer)

        return np_output_tensor

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height_size", [128])
@pytest.mark.parametrize("width_size", [128])
def test_torch_vs_sonic(tmp_path, batch_size, height_size, width_size):
    input_tensor = torch.randn(batch_size, height_size, width_size)

    sonic_model = create_sonic_model(batch_size, height_size, width_size)

    torch_output = torch_model(input_tensor)
    sonic_encoder_output = sonic_model(input_tensor)

    assert np.allclose(
        torch_output.numpy(), sonic_encoder_output
    ), f"{list(torch_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("num_iterations", [100])
@pytest.mark.parametrize("batch_size", [4])
@pytest.mark.parametrize("height_size", [1024])
@pytest.mark.parametrize("width_size", [1024])
def test_benchmark(tmp_path, num_iterations, batch_size, height_size, width_size):
    sonic_model = create_sonic_model(batch_size, height_size, width_size)

    def torch_function():
        input_tensor = torch.randn(batch_size, height_size, width_size)
        torch_model(input_tensor)

    torch_function()
    start = time.time_ns()
    for _ in range(num_iterations):
        torch_function()
    end = time.time_ns()
    torch_duration = (end - start) / num_iterations / 1e3
    logger.info(f"torch_duration: {torch_duration}")

    def sonic_function():
        input_tensor = torch.randn(batch_size, height_size, width_size)
        sonic_model(input_tensor)

    start = time.time_ns()
    for _ in range(num_iterations):
        sonic_function()
    end = time.time_ns()
    sonic_duration = (end - start) / num_iterations / 1e3
    logger.info(f"sonic_duration: {sonic_duration}")

    logger.info(f"torch is {sonic_duration / torch_duration}x faster")
