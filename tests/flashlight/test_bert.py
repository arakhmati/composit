import pytest

import numpy as np
import torch
import transformers

import composit as cnp
import flashlight
from model_zoo.bert import (
    create_bert_config,
)


def trace_bert(batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size, run_torch=True):
    config = create_bert_config(
        num_encoders=num_encoders,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        vocab_size=vocab_size,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_size))
    attention_mask = torch.zeros(batch_size, sequence_size, dtype=torch.float32)
    token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.int64)

    with flashlight.tracer.trace(run_torch=run_torch):
        transformers_model = transformers.models.bert.modeling_bert.BertModel(config)
        flashlight_output = transformers_model(input_ids, attention_mask, token_type_ids=token_type_ids)[
            "last_hidden_state"
        ]

    if isinstance(flashlight_output, flashlight.Tensor):
        return flashlight_output.lazy_tensor, flashlight_output.detach().numpy()
    return flashlight_output, None


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [10])
@pytest.mark.parametrize("run_torch", [True])
def test_trace(batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size, run_torch):
    output_var, torch_output = trace_bert(
        batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size, run_torch
    )

    assert len(output_var.graph) == 246
    composit_output = cnp.evaluate(output_var)
    if run_torch:
        assert np.allclose(composit_output, torch_output, atol=1e-3)


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [10])
def test_graph_cache(batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size):
    output_var, _ = trace_bert(batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size)
    first_hash = hash(output_var)

    output_var, _ = trace_bert(batch_size, num_encoders, sequence_size, num_attention_heads, head_size, vocab_size)
    assert first_hash == hash(output_var)
