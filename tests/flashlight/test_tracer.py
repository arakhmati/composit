import pytest

import torch

import composit as cnp
import flashlight


def check_output_type(output, run_torch):
    if run_torch:
        assert isinstance(output, flashlight.Tensor)
    else:
        assert isinstance(output, cnp.types.LazyTensor)


@pytest.mark.parametrize("run_torch", [True, False])
def test_add(run_torch):
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = input_a + input_b

    check_output_type(output, run_torch)
    assert len(output.graph) == 3


@pytest.mark.parametrize("run_torch", [True, False])
def test_sub(run_torch):
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = input_a - input_b

    check_output_type(output, run_torch)
    assert len(output.graph) == 3


@pytest.mark.parametrize("run_torch", [True, False])
def test_mul(run_torch):
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = input_a * input_b

    check_output_type(output, run_torch)
    assert len(output.graph) == 3


@pytest.mark.parametrize("run_torch", [True, False])
def test_truediv(run_torch):
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = input_a / input_b

    check_output_type(output, run_torch)
    assert len(output.graph) == 3


@pytest.mark.parametrize("run_torch", [True, False])
def test_matmul(run_torch):
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128, 64)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = input_a @ input_b

    check_output_type(output, run_torch)
    assert len(output.graph) == 3


@pytest.mark.parametrize("run_torch", [True, False])
def test_linear(run_torch):
    model = torch.nn.Linear(128, 64)
    activations = torch.rand(32, 128)

    with flashlight.tracer.trace(run_torch=run_torch):
        output = model(activations)

    check_output_type(output, run_torch)
    assert len(output.graph) == 6


@pytest.mark.parametrize("run_torch", [True, False])
def test_embedding(run_torch):
    model = torch.nn.Embedding(num_embeddings=10, embedding_dim=64)
    input_ids = torch.randint(0, 10, (1, 128))

    with flashlight.tracer.trace(run_torch=run_torch):
        output = model(input_ids)

    check_output_type(output, run_torch)
    assert len(output.graph) == 3
