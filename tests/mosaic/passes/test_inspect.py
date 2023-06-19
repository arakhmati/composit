import numpy as np
import pytest

import transformers

import composit as cnp

from model_zoo.bert import (
    create_bert_config,
    bert,
    convert_parameters_to_numpy,
)
from mosaic.passes import inspect


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_attention_heads", [4])
@pytest.mark.parametrize("head_size", [48])
@pytest.mark.parametrize("vocab_size", [16])
def test_bert(
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

    transformers_model = transformers.models.bert.modeling_bert.BertModel(config)

    input_ids_var = cnp.ndarray((batch_size, sequence_size), dtype=np.uint64, name="input_ids")
    token_type_ids_var = cnp.ndarray((batch_size, sequence_size), dtype=np.uint64, name="token_type_ids")
    parameters = {
        name: cnp.asarray(value, name=name) for name, value in convert_parameters_to_numpy(transformers_model).items()
    }

    output_var = bert(
        input_ids_var,
        token_type_ids_var,
        None,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    inspect(output_var)
