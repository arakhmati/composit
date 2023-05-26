import torch

import flashlight


def test_add():
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.functional.trace():
        output = input_a + input_b

    assert len(output.graph) == 3


def test_sub():
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.functional.trace():
        output = input_a - input_b

    assert len(output.graph) == 3


def test_mul():
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.functional.trace():
        output = input_a * input_b

    assert len(output.graph) == 3


def test_truediv():
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128)

    with flashlight.functional.trace():
        output = input_a / input_b

    assert len(output.graph) == 3


def test_matmul():
    input_a = torch.rand(32, 128)
    input_b = torch.rand(128, 64)

    with flashlight.functional.trace():
        output = input_a @ input_b

    assert len(output.graph) == 3


def test_linear():
    model = torch.nn.Linear(128, 64)
    activations = torch.rand(32, 128)

    with flashlight.functional.trace():
        output = model(activations)

    assert len(output.graph) == 6


def test_embedding():
    model = torch.nn.Embedding(num_embeddings=10, embedding_dim=64)
    input_ids = torch.randint(0, 10, (1, 128))

    with flashlight.functional.trace():
        output = model(input_ids)

    assert len(output.graph) == 3
