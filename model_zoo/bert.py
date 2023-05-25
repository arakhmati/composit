import numpy as np
import transformers

import composit as cnp
from composit.nn.module import wrap_module
from composit.nn.layers import layer_norm, multi_head_attention, feedforward


def create_bert_config(num_encoders: int, num_attention_heads: int, head_size: int, vocab_size: int):
    return transformers.models.bert.configuration_bert.BertConfig(
        num_hidden_layers=num_encoders,
        num_attention_heads=num_attention_heads,
        hidden_size=num_attention_heads * head_size,
        intermediate_size=num_attention_heads * head_size * 4,
        vocab_size=vocab_size,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
        position_embedding_type=None,
    )


def create_random_float(shape, minimum=-0.1, maximum=0.1):
    return np.random.uniform(minimum, maximum, shape).astype(np.float64)


def create_random_long(shape, minimum, maximum):
    return np.random.randint(minimum, maximum, shape, dtype=np.int64)


def convert_parameters_to_numpy(model):
    parameters = {}
    for name, value in model.named_parameters():
        new_value = value.detach()
        if "weight" in name and "embedding" not in name:
            new_value = new_value.T
        parameters[name] = new_value.detach().numpy()
    return parameters


def create_parameters(num_encoders, hidden_size, vocab_size, num_question_answering_labels=None):
    intermediate_size = hidden_size * 4

    parameters = {
        "embeddings.word_embeddings.weight": create_random_float((vocab_size, hidden_size)),
        "embeddings.token_type_embeddings.weight": np.zeros((2, hidden_size), dtype=np.float64),
        "embeddings.LayerNorm.weight": create_random_float((hidden_size,)),
        "embeddings.LayerNorm.bias": create_random_float((hidden_size,)),
    }

    for encoder_index in range(num_encoders):
        parameters.update(
            {
                f"encoder.layer.{encoder_index}.attention.self.query.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"encoder.layer.{encoder_index}.attention.self.query.bias": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.attention.self.key.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"encoder.layer.{encoder_index}.attention.self.key.bias": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.attention.self.value.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"encoder.layer.{encoder_index}.attention.self.value.bias": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.attention.output.dense.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"encoder.layer.{encoder_index}.attention.output.dense.bias": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.attention.output.LayerNorm.weight": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.attention.output.LayerNorm.bias": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.intermediate.dense.weight": create_random_float(
                    (hidden_size, intermediate_size)
                ),
                f"encoder.layer.{encoder_index}.intermediate.dense.bias": create_random_float((intermediate_size,)),
                f"encoder.layer.{encoder_index}.output.dense.weight": create_random_float(
                    (intermediate_size, hidden_size)
                ),
                f"encoder.layer.{encoder_index}.output.dense.bias": create_random_float(hidden_size),
                f"encoder.layer.{encoder_index}.output.LayerNorm.weight": create_random_float((hidden_size,)),
                f"encoder.layer.{encoder_index}.output.LayerNorm.bias": create_random_float((hidden_size,)),
            }
        )

    if num_question_answering_labels is not None:
        parameters["qa_outputs.weight"] = create_random_float((hidden_size, num_question_answering_labels))
        parameters["qa_outputs.bias"] = create_random_float((num_question_answering_labels,))

    return {name: array.numpy() for name, array in parameters.items()}


@wrap_module
def bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    head_size,
):
    multi_head_attention_output = multi_head_attention(
        hidden_states,
        attention_mask,
        parameters[f"encoder.layer.{encoder_index}.attention.self.query.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.self.query.bias"],
        parameters[f"encoder.layer.{encoder_index}.attention.self.key.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.self.key.bias"],
        parameters[f"encoder.layer.{encoder_index}.attention.self.value.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.self.value.bias"],
        parameters[f"encoder.layer.{encoder_index}.attention.output.dense.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.output.dense.bias"],
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = layer_norm(
        hidden_states + multi_head_attention_output,
        parameters[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = feedforward(
        multi_head_attention_add_and_layer_norm_output,
        parameters[f"encoder.layer.{encoder_index}.intermediate.dense.weight"],
        parameters[f"encoder.layer.{encoder_index}.intermediate.dense.bias"],
        parameters[f"encoder.layer.{encoder_index}.output.dense.weight"],
        parameters[f"encoder.layer.{encoder_index}.output.dense.bias"],
    )

    feedforward_add_and_layer_norm_output = layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        parameters[f"encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"encoder.layer.{encoder_index}.output.LayerNorm.bias"],
    )

    return feedforward_add_and_layer_norm_output


@wrap_module
def bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    word_embeddings = cnp.nn.embedding(input_ids, parameters["embeddings.word_embeddings.weight"])
    token_type_embeddings = cnp.nn.embedding(token_type_ids, parameters["embeddings.token_type_embeddings.weight"])
    embeddings = word_embeddings + token_type_embeddings

    encoder_input = layer_norm(
        embeddings,
        parameters["embeddings.LayerNorm.weight"],
        parameters["embeddings.LayerNorm.bias"],
    )

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
            head_size=head_size,
        )
        encoder_input = encoder_output
    return encoder_output


@wrap_module
def bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    head_size,
):
    bert_output = bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"]
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]
    return qa_outputs
