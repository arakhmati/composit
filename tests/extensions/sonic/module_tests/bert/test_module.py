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


def create_parameters(num_encoders, hidden_size):
    intermediate_size = hidden_size * 4

    parameters = {}
    for index in range(num_encoders):
        parameters.update(
            {
                f"encoder.{index}.query.weight": torch.rand(hidden_size, hidden_size),
                f"encoder.{index}.query.bias": torch.rand(hidden_size),
                f"encoder.{index}.key.weight": torch.rand(hidden_size, hidden_size),
                f"encoder.{index}.key.bias": torch.rand(hidden_size),
                f"encoder.{index}.value.weight": torch.rand(hidden_size, hidden_size),
                f"encoder.{index}.value.bias": torch.rand(hidden_size),
                f"encoder.{index}.self_output.weight": torch.rand(hidden_size, hidden_size),
                f"encoder.{index}.self_output.bias": torch.rand(hidden_size),
                f"encoder.{index}.ff1.weight": torch.rand(hidden_size, intermediate_size),
                f"encoder.{index}.ff1.bias": torch.rand(intermediate_size),
                f"encoder.{index}.ff2.weight": torch.rand(intermediate_size, hidden_size),
                f"encoder.{index}.ff2.bias": torch.rand(hidden_size),
            }
        )
    return parameters


def multi_head_attention(hidden_states, parameters, head_size, encoder_index):
    batch_size, sequence_size, hidden_size = hidden_states.shape
    num_attention_heads = hidden_size // head_size

    query = hidden_states @ parameters[f"encoder.{encoder_index}.query.weight"]
    query = query + parameters[f"encoder.{encoder_index}.query.bias"]
    query = torch.reshape(query, (batch_size, sequence_size, num_attention_heads, head_size))
    query = torch.permute(query, (0, 2, 1, 3))

    key = hidden_states @ parameters[f"encoder.{encoder_index}.key.weight"]
    key = key + parameters[f"encoder.{encoder_index}.key.bias"]
    key = torch.reshape(key, (batch_size, sequence_size, num_attention_heads, head_size))
    key = torch.permute(key, (0, 2, 3, 1))

    value = hidden_states @ parameters[f"encoder.{encoder_index}.value.weight"]
    value = value + parameters[f"encoder.{encoder_index}.value.bias"]
    value = torch.reshape(value, (batch_size, sequence_size, num_attention_heads, head_size))
    value = torch.permute(value, (0, 2, 1, 3))

    attention_scores = query @ key
    attention_scores = attention_scores / (head_size**0.5)
    attention_scores = torch.softmax(attention_scores, dim=-1)

    context = attention_scores @ value
    context = torch.permute(context, (0, 2, 1, 3))
    context = torch.reshape(context, (batch_size, sequence_size, hidden_size))

    self_output = context @ parameters[f"encoder.{encoder_index}.self_output.weight"]
    self_output = self_output + parameters[f"encoder.{encoder_index}.self_output.bias"]

    return self_output


def feedforward(hidden_states, parameters, encoder_index):
    hidden_states = hidden_states @ parameters[f"encoder.{encoder_index}.ff1.weight"]
    hidden_states = hidden_states + parameters[f"encoder.{encoder_index}.ff1.bias"]
    hidden_states = hidden_states @ parameters[f"encoder.{encoder_index}.ff2.weight"]
    hidden_states = hidden_states + parameters[f"encoder.{encoder_index}.ff2.bias"]
    return hidden_states


def torch_model(hidden_states, parameters, num_encoders, head_size):
    for encoder_index in range(num_encoders):
        hidden_states = multi_head_attention(hidden_states, parameters, head_size, encoder_index)
        hidden_states = feedforward(hidden_states, parameters, encoder_index)
    return hidden_states


def create_sonic_model(batch_size, num_encoders, sequence_size, num_attention_heads, head_size):
    output_file = (
        TEST_DIRECTORY
        / "test_output"
        / f"{batch_size}_{num_encoders}_{sequence_size}_{num_attention_heads}_{head_size}.cpp"
    )
    output_file.unlink(missing_ok=True)

    template_loader = jinja2.FileSystemLoader(searchpath=TEST_DIRECTORY)
    template_env = jinja2.Environment(loader=template_loader)
    template = template_env.get_template("template.jinja.cpp")
    output_text = template.render(
        batch_size=batch_size,
        num_encoders=num_encoders,
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

    def run_model(hidden_states, parameter_buffers):
        encoder_input = align_array(hidden_states.numpy())
        encoder_output = create_aligned_array(encoder_input.shape, encoder_input.dtype)

        input_buffer = cast_numpy_array_to_pointer(encoder_input)
        output_buffer = cast_numpy_array_to_pointer(encoder_output)

        # Add run_c function so that the profiler picks it up
        def run_c():
            run(input_buffer, output_buffer, parameter_buffers)

        run_c()

        return encoder_output

    return run_model


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
def test_torch_vs_sonic(batch_size, num_encoders, sequence_size, num_attention_heads, head_size):
    hidden_size = num_attention_heads * head_size

    hidden_states = torch.rand(batch_size, sequence_size, hidden_size)
    parameters = create_parameters(num_encoders, hidden_size)
    sonic_parameters = {key: align_array(value.numpy()) for key, value in parameters.items()}
    sonic_parameter_buffers = cast_numpy_arrays_to_pointer(sonic_parameters.values())

    sonic_model = create_sonic_model(batch_size, num_encoders, sequence_size, num_attention_heads, head_size)

    encoder_output = torch_model(hidden_states, parameters, num_encoders, head_size)
    sonic_encoder_output = sonic_model(hidden_states, sonic_parameter_buffers)

    assert np.allclose(
        encoder_output.numpy(), sonic_encoder_output
    ), f"{list(encoder_output.flatten())} != {list(sonic_encoder_output.flatten())}"


@pytest.mark.parametrize("module", ["torch", "sonic"])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [1])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [1])
@pytest.mark.parametrize("head_size", [128])
def test_benchmark(benchmark, module, batch_size, num_encoders, sequence_size, num_attention_heads, head_size):
    hidden_size = num_attention_heads * head_size
    parameters = create_parameters(num_encoders, hidden_size)

    function = None
    if module == "torch":

        def function():
            hidden_states = torch.randn(batch_size, sequence_size, hidden_size)
            torch_model(hidden_states, parameters, num_encoders, head_size)

    elif module == "sonic":
        sonic_parameters = {key: align_array(value.numpy()) for key, value in parameters.items()}
        sonic_parameter_buffers = cast_numpy_arrays_to_pointer(sonic_parameters.values())
        sonic_model = create_sonic_model(batch_size, num_encoders, sequence_size, num_attention_heads, head_size)

        def function():
            hidden_states = torch.randn(batch_size, sequence_size, hidden_size)
            sonic_model(hidden_states, sonic_parameter_buffers)

    with cProfile.Profile() as pr:
        for _ in range(2000):
            function()
    pr.print_stats(sort="cumulative")

    benchmark(function)
