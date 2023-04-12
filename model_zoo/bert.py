import numpy as np
import transformers

import composit as cnp
from composit.nn.module import wrap_module


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
def functional_softmax(input_tensor, axis):
    exp_input_tensor = cnp.exp(input_tensor - cnp.max(input_tensor, axis=axis, keepdims=True))
    return exp_input_tensor / cnp.sum(exp_input_tensor, axis=axis, keepdims=True)


@wrap_module
def functional_multi_head_attention(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    sequence_size,
    num_attention_heads,
    head_size,
):
    batch_size = hidden_states.shape[0]

    query = hidden_states @ parameters[f"encoder.layer.{encoder_index}.attention.self.query.weight"]
    query = query + parameters[f"encoder.layer.{encoder_index}.attention.self.query.bias"]
    query = cnp.reshape(query, (batch_size, sequence_size, num_attention_heads, head_size))
    query = cnp.transpose(query, (0, 2, 1, 3))

    key = hidden_states @ parameters[f"encoder.layer.{encoder_index}.attention.self.key.weight"]
    key = key + parameters[f"encoder.layer.{encoder_index}.attention.self.key.bias"]
    key = cnp.reshape(key, (batch_size, sequence_size, num_attention_heads, head_size))
    key = cnp.transpose(key, (0, 2, 3, 1))

    value = hidden_states @ parameters[f"encoder.layer.{encoder_index}.attention.self.value.weight"]
    value = value + parameters[f"encoder.layer.{encoder_index}.attention.self.value.bias"]
    value = cnp.reshape(value, (batch_size, sequence_size, num_attention_heads, head_size))
    value = cnp.transpose(value, (0, 2, 1, 3))

    attention_scores = query @ key

    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = functional_softmax(attention_scores, axis=-1)

    context_layer = attention_probs @ value

    context_layer = cnp.transpose(context_layer, (0, 2, 1, 3))
    context_layer = cnp.reshape(context_layer, (batch_size, sequence_size, num_attention_heads * head_size))

    self_output = context_layer @ parameters[f"encoder.layer.{encoder_index}.attention.output.dense.weight"]
    self_output = self_output + parameters[f"encoder.layer.{encoder_index}.attention.output.dense.bias"]

    return self_output


@wrap_module
def functional_layer_norm(input_tensor, weight, bias, *, epsilon=1e-5):
    mean = cnp.mean(input_tensor, axis=-1, keepdims=True)
    input_tensor_minus_mean = input_tensor - mean
    var = cnp.mean(cnp.square(input_tensor_minus_mean), axis=-1, keepdims=True)
    output = input_tensor_minus_mean / cnp.sqrt(var + epsilon)
    output *= weight
    output += bias
    return output
    """
    mean = cnp.mean(input_tensor, axis=-1, keepdims=True)
    var = cnp.sqrt(cnp.var(input_tensor, axis=-1, keepdims=True) + epsilon)
    output = (input_tensor - mean) / var
    return output * weight + bias
    """


@wrap_module
def functional_feedforward(hidden_states, parameters, encoder_index):
    hidden_states = hidden_states @ parameters[f"encoder.layer.{encoder_index}.intermediate.dense.weight"]
    hidden_states = hidden_states + parameters[f"encoder.layer.{encoder_index}.intermediate.dense.bias"]
    hidden_states = cnp.nn.gelu(hidden_states)
    hidden_states = hidden_states @ parameters[f"encoder.layer.{encoder_index}.output.dense.weight"]
    hidden_states = hidden_states + parameters[f"encoder.layer.{encoder_index}.output.dense.bias"]
    return hidden_states


@wrap_module
def functional_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    sequence_size,
    num_attention_heads,
    head_size,
):
    multi_head_attention_output = functional_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters,
        encoder_index=encoder_index,
        sequence_size=sequence_size,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = functional_layer_norm(
        hidden_states + multi_head_attention_output,
        parameters[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        parameters[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = functional_feedforward(
        multi_head_attention_add_and_layer_norm_output, parameters, encoder_index=encoder_index
    )

    feedforward_add_and_layer_norm_output = functional_layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        parameters[f"encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"encoder.layer.{encoder_index}.output.LayerNorm.bias"],
    )

    return feedforward_add_and_layer_norm_output


@wrap_module
def functional_bert(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
):
    word_embeddings = cnp.nn.embedding(input_ids, parameters["embeddings.word_embeddings.weight"])
    token_type_embeddings = cnp.nn.embedding(token_type_ids, parameters["embeddings.token_type_embeddings.weight"])
    embeddings = word_embeddings + token_type_embeddings

    encoder_input = functional_layer_norm(
        embeddings,
        parameters["embeddings.LayerNorm.weight"],
        parameters["embeddings.LayerNorm.bias"],
    )

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = functional_bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index=encoder_index,
            sequence_size=sequence_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
        )
        encoder_input = encoder_output
    return encoder_output


@wrap_module
def functional_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    *,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
):
    bert_output = functional_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        sequence_size=sequence_size,
        num_attention_heads=num_attention_heads,
        head_size=head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"]
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]
    return qa_outputs
