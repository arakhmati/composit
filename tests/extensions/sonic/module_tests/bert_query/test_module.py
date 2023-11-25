import pytest

import cProfile
import pathlib
from ctypes import cdll

import jinja2
import numpy as np
import torch

from mosaic.aligned_array import create_aligned_array, align_array
from mosaic.ctypes import cast_numpy_array_to_pointer, cast_numpy_arrays_to_pointer
from mosaic.backends.x86.compile import compile_shared_library

TEST_DIRECTORY = pathlib.Path(__file__).parent


def torch_random(*shape, dtype=torch.float32, low=-0.1, high=0.1):
    return (low - high) * torch.rand(shape).to(dtype) + high


def create_parameters(hidden_size):
    parameters = {"query.weight": torch_random(hidden_size, hidden_size), "query.bias": torch_random(hidden_size)}
    return parameters


def torch_model(hidden_states, parameters, head_size):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    num_attention_heads = hidden_size // head_size

    query = hidden_states @ parameters["query.weight"]
    query = query + parameters["query.bias"]
    query = torch.reshape(query, (batch_size, sequence_size, num_attention_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3))

    return query


def create_sonic_model(batch_size, sequence_size, num_attention_heads, head_size):
    output_file = TEST_DIRECTORY / "test_output" / f"{batch_size}_{sequence_size}_{num_attention_heads}_{head_size}.cpp"
    output_file.unlink(missing_ok=True)

    template_loader = jinja2.FileSystemLoader(searchpath=TEST_DIRECTORY)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("template.jinja.cpp")
    output_text = template.render(
        batch_size=batch_size,
        sequence_size=sequence_size,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
    )

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

    output_shape = (batch_size, num_attention_heads, sequence_size, head_size)

    def run_model(hidden_states, parameter_buffers):
        encoder_input = align_array(hidden_states.numpy())
        encoder_output = create_aligned_array(output_shape, encoder_input.dtype)

        input_buffer = cast_numpy_array_to_pointer(encoder_input)
        output_buffer = cast_numpy_array_to_pointer(encoder_output)

        # Add run_c function so that the profiler picks it up
        def run_c():
            run(input_buffer, output_buffer, parameter_buffers)

        run_c()

        return encoder_output

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
def test_torch_vs_sonic(batch_size, sequence_size, num_attention_heads, head_size):
    hidden_size = num_attention_heads * head_size

    hidden_states = torch_random(batch_size, sequence_size, hidden_size)
    parameters = create_parameters(hidden_size)
    sonic_parameters = {key: align_array(value.numpy()) for key, value in parameters.items()}
    sonic_parameter_buffers = cast_numpy_arrays_to_pointer(sonic_parameters.values())

    sonic_model = create_sonic_model(batch_size, sequence_size, num_attention_heads, head_size)

    encoder_output = torch_model(hidden_states, parameters, head_size)
    sonic_encoder_output = sonic_model(hidden_states, sonic_parameter_buffers)

    assert np.allclose(
        encoder_output.numpy(), sonic_encoder_output, atol=1e-4, rtol=1e-5
    ), f"{list(encoder_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "sonic"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
def test_benchmark(benchmark, module, batch_size, sequence_size, num_attention_heads, head_size):
    hidden_size = num_attention_heads * head_size
    parameters = create_parameters(hidden_size)

    function = None
    if module == "torch":

        def function():
            hidden_states = torch_random(batch_size, sequence_size, hidden_size)
            torch_model(hidden_states, parameters, head_size)

    elif module == "sonic":
        sonic_parameters = {key: align_array(value.numpy()) for key, value in parameters.items()}
        sonic_parameter_buffers = cast_numpy_arrays_to_pointer(sonic_parameters.values())
        sonic_model = create_sonic_model(batch_size, sequence_size, num_attention_heads, head_size)

        def function():
            hidden_states = torch_random(batch_size, sequence_size, hidden_size)
            sonic_model(hidden_states, sonic_parameter_buffers)

    with cProfile.Profile() as pr:
        for _ in range(100):
            function()
    pr.print_stats(sort="cumulative")

    benchmark(function)
