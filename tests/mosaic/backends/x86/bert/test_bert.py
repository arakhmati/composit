from __future__ import annotations

import pathlib
import time

import numpy as np
import pytest
import torch

import transformers
from loguru import logger

import composit as cnp
from composit.hash import deterministic_hash
from mosaic.backends.x86.passes.evaluate import evaluate as mosaic_evaluate
from mosaic.tilelab.layout import TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, ScalarTileLevel

from model_zoo.bert import (
    bert,
    create_random_long,
    convert_parameters_to_numpy,
    create_bert_config,
)

FILE_DIR = pathlib.Path(__file__).parent.resolve()


def get_transformers_model(num_encoders, num_attention_heads, head_size, vocab_size):
    config = create_bert_config(
        num_encoders=num_encoders,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        vocab_size=vocab_size,
    )

    model = transformers.models.bert.modeling_bert.BertModel(config, add_pooling_layer=False)
    return model


def create_composit_parameters(model):
    return {
        name: cnp.asarray(array=value, name=name)
        for name, value in convert_parameters_to_numpy(model).items()
        if "position_embeddings" not in name
    }


def create_tile_shape(var, tile_size):
    shape = var.shape
    if len(shape) == 3:
        return (1, tile_size, tile_size)
    elif len(shape) == 2:
        if "embeddings.weight" in var.name:
            return (1, tile_size)
        return min(shape[0], tile_size), min(shape[1], tile_size)
    elif len(shape) == 1:
        return (min(shape[0], tile_size),)
    else:
        return ()


def create_hierarchy(var, tile_shape):
    kwargs = {}
    if "dense.weight" in var.name:
        kwargs = dict(layout=TransposedLayout(order=(1, 0)))
    return (
        TileLevel(level_name="tile", tile_shape=tile_shape, **kwargs),
        ScalarTileLevel(level_name="scalar", rank=len(tile_shape), **kwargs),
    )


def composit_model(
    input_ids_var,
    token_type_ids_var,
    num_encoders,
    head_size,
    parameters,
):
    output_var = bert(
        input_ids_var,
        token_type_ids_var,
        None,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    # Use a solver to figure out the tilization scheme
    input_var_to_scheme = {
        var: create_hierarchy(var, create_tile_shape(var, head_size))
        for var in [input_ids_var, token_type_ids_var] + list(parameters.values())
    }

    return output_var, input_var_to_scheme


def bert_model(
    input_ids_var,
    token_type_ids_var,
    num_encoders,
    num_attention_heads,
    head_size,
    vocab_size,
):
    np.random.seed(0)

    transformers_model = get_transformers_model(num_encoders, num_attention_heads, head_size, vocab_size)
    composit_parameters = create_composit_parameters(transformers_model)
    output_var, input_var_to_scheme = composit_model(
        input_ids_var, token_type_ids_var, num_encoders, head_size, composit_parameters
    )
    return output_var, input_var_to_scheme, transformers_model


@pytest.mark.xfail(reason="Broken after updating dependencies")
@pytest.mark.parametrize("num_inputs", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [16])
@pytest.mark.parametrize("reuse_buffers", [False, True])
@pytest.mark.parametrize("fuse_kernels", [False])
def test_evaluates_correctly(
    request,
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
    reuse_buffers,
    fuse_kernels,
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))
    test_output_path.mkdir(parents=True, exist_ok=True)

    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)

        input_ids_var = cnp.asarray(input_ids, name="input_ids")
        token_type_ids_var = cnp.asarray(token_type_ids, name="token_type_ids")

        output_var, input_var_to_scheme, _ = bert_model(
            input_ids_var,
            token_type_ids_var,
            num_encoders,
            num_attention_heads,
            head_size,
            vocab_size,
        )

        golden_output = cnp.nn.evaluate(output_var)
        output = mosaic_evaluate(
            output_var,
            input_var_to_scheme=input_var_to_scheme,
            output_path=test_output_path,
            reuse_buffers=reuse_buffers,
            fuse_kernels=fuse_kernels,
        )
        assert np.allclose(output, golden_output, atol=1e-4, rtol=1e-5)


@pytest.mark.xfail(reason="Broken after updating dependencies")
@pytest.mark.parametrize("num_iterations", [10])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [16])
@pytest.mark.parametrize("reuse_buffers", [True])
def test_benchmark(
    request,
    num_iterations,
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
    reuse_buffers,
):
    test_name = request.node.name
    test_output_path = FILE_DIR / "test_output" / str(deterministic_hash(test_name))
    test_output_path.mkdir(parents=True, exist_ok=True)

    torch.set_num_threads(1)

    input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
    token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)

    input_ids_var = cnp.asarray(input_ids, name="input_ids")
    token_type_ids_var = cnp.asarray(token_type_ids, name="token_type_ids")

    output_var, input_var_to_scheme, transformers_model = bert_model(
        input_ids_var,
        token_type_ids_var,
        num_encoders,
        num_attention_heads,
        head_size,
        vocab_size,
    )

    golden_output = cnp.nn.evaluate(output_var)
    output = mosaic_evaluate(
        output_var,
        input_var_to_scheme=input_var_to_scheme,
        output_path=test_output_path,
        reuse_buffers=reuse_buffers,
        fuse_kernels=False,
    )
    assert np.allclose(output, golden_output, atol=1e-4, rtol=1e-5)

    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()
        output = transformers_model(torch.from_numpy(input_ids), token_type_ids=torch.from_numpy(token_type_ids))[
            "last_hidden_state"
        ]
        end = time.time_ns()
        execution_times.append(end - start)
    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"torch Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"torch Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"torch Maximum Execution Time: {execution_times.max()} milliseconds")

    execution_times = []
    for _ in range(num_iterations):
        start = time.time_ns()
        output = mosaic_evaluate(
            output_var,
            input_var_to_scheme=input_var_to_scheme,
            output_path=test_output_path,
            reuse_buffers=reuse_buffers,
            fuse_kernels=False,
        )
        end = time.time_ns()
        execution_times.append(end - start)
    execution_times = np.asarray(execution_times) / 1e6
    logger.info(f"composit Average Execution Time: {execution_times.mean()} milliseconds")
    logger.info(f"composit Minimum Execution Time: {execution_times.min()} milliseconds")
    logger.info(f"composit Maximum Execution Time: {execution_times.max()} milliseconds")
