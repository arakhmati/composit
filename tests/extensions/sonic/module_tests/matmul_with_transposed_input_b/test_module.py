import pytest

import pathlib
from ctypes import cdll

import jinja2
import numpy as np
import torch

from mosaic.backends.ctypes import cast_numpy_array_to_pointer
from mosaic.backends.x86.compile import compile_shared_library

TEST_DIRECTORY = pathlib.Path(__file__).parent


def torch_model(input_tensor, weights):
    return input_tensor @ weights.T


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

    def run_model(torch_input_tensor, torch_weights):
        output_shape = torch_input_tensor.shape[:-1] + torch_weights.shape[-2:-1]

        np_input_tensor = torch_input_tensor.numpy()
        np_weights = torch_weights.numpy()
        np_output_tensor = np.zeros(output_shape, dtype=np_input_tensor.dtype)

        input_buffer = cast_numpy_array_to_pointer(np_input_tensor)
        weights_buffer = cast_numpy_array_to_pointer(np_weights)
        output_buffer = cast_numpy_array_to_pointer(np_output_tensor)

        run(input_buffer, weights_buffer, output_buffer)

        return np_output_tensor

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [16])
@pytest.mark.parametrize("k_size", [48])
@pytest.mark.parametrize("n_size", [32])
def test_torch_vs_sonic(batch_size, m_size, k_size, n_size):
    input_tensor = torch.randn(batch_size, m_size, k_size)
    weights = torch.randn(k_size, n_size).T.contiguous()

    sonic_model = create_sonic_model(batch_size, m_size, k_size, n_size)

    torch_output = torch_model(input_tensor, weights)
    sonic_encoder_output = sonic_model(input_tensor, weights)

    assert np.allclose(
        torch_output.numpy(), sonic_encoder_output, atol=1e-5, rtol=1e-5
    ), f"{list(torch_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "sonic"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("m_size", [128])
@pytest.mark.parametrize("k_size", [128])
@pytest.mark.parametrize("n_size", [128])
def test_benchmark(benchmark, module, batch_size, m_size, k_size, n_size):
    weights = torch.rand(k_size, n_size).T.contiguous()

    function = None
    if module == "torch":

        def function():
            input_tensor = torch.randn(batch_size, m_size, k_size)
            torch_model(input_tensor, weights)

    elif module == "sonic":
        sonic_model = create_sonic_model(batch_size, m_size, k_size, n_size)

        def function():
            input_tensor = torch.randn(batch_size, m_size, k_size)
            sonic_model(input_tensor, weights)

    benchmark(function)
