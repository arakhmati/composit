import pytest

import numpy as np
import torch
import transformers

import flashlight
from flashlight.tensor import forward
from model_zoo.bert import (
    create_bert_config,
)


def get_extended_attention_mask(
    attention_mask,
    *args,
    **kwargs,
):
    # TODO: lackcontiguous should not be needed
    return attention_mask.contiguous()


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [32])
@pytest.mark.parametrize("vocab_size", [10])
def test_trace(
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
):
    config = create_bert_config(
        num_encoders=num_encoders,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
        vocab_size=vocab_size,
    )

    input_ids = torch.randint(0, vocab_size, (batch_size, sequence_size))
    attention_mask = torch.zeros(batch_size, sequence_size, dtype=torch.float32)
    token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.int64)

    with flashlight.functional.trace():
        transformers_model = transformers.models.bert.modeling_bert.BertModel(config)
        transformers_model.pooler = None
        transformers_model.get_extended_attention_mask = get_extended_attention_mask

        flashlight_output = transformers_model(input_ids, attention_mask, token_type_ids=token_type_ids)[
            "last_hidden_state"
        ]
        assert len(flashlight_output.graph) == 240

        composit_output = forward(flashlight_output, input_tensors=[input_ids, attention_mask, token_type_ids])

        assert np.allclose(composit_output, flashlight_output.detach().numpy(), atol=1e-3)
