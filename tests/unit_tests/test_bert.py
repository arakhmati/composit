import pytest

import persistent_numpy as pnp
import torch

import transformers

from persistent_numpy.multidigraph import visualize_graph


def create_random_torch_float_tensor(*shape, minimum=-0.1, maximum=0.1):
    return (maximum - minimum) * torch.rand(shape) + minimum


def create_random_torch_long_tensor(*shape, minimum, maximum):
    return torch.randint(minimum, maximum, shape)


def create_parameters(num_encoders, hidden_size, vocab_size, num_question_answering_labels=None):

    intermediate_size = hidden_size * 4

    parameters = {
        "bert.embeddings.word_embeddings.weight": create_random_torch_float_tensor(vocab_size, hidden_size),
        "bert.embeddings.token_type_embeddings.weight": torch.zeros(2, hidden_size),
        "bert.embeddings.LayerNorm.weight": create_random_torch_float_tensor(hidden_size),
        "bert.embeddings.LayerNorm.bias": create_random_torch_float_tensor(hidden_size),
    }

    for encoder_index in range(num_encoders):
        parameters.update(
            {
                f"bert.encoder.layer.{encoder_index}.attention.self.query.weight": create_random_torch_float_tensor(
                    hidden_size, hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.query.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.key.weight": create_random_torch_float_tensor(
                    hidden_size, hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.key.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.value.weight": create_random_torch_float_tensor(
                    hidden_size, hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.value.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight": create_random_torch_float_tensor(
                    hidden_size, hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight": create_random_torch_float_tensor(
                    hidden_size, intermediate_size
                ),
                f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias": create_random_torch_float_tensor(
                    intermediate_size
                ),
                f"bert.encoder.layer.{encoder_index}.output.dense.weight": create_random_torch_float_tensor(
                    intermediate_size, hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.output.dense.bias": create_random_torch_float_tensor(hidden_size),
                f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight": create_random_torch_float_tensor(
                    hidden_size
                ),
                f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias": create_random_torch_float_tensor(
                    hidden_size
                ),
            }
        )

    if num_question_answering_labels is not None:
        parameters["qa_outputs.weight"] = create_random_torch_float_tensor(hidden_size, num_question_answering_labels)
        parameters["qa_outputs.bias"] = create_random_torch_float_tensor(num_question_answering_labels)

    return parameters


def visualize_node(graph, node):
    instruction = graph.get_node_attribute(node, "instruction")
    return f"{type(instruction)}: {node.name}"


def functional_embedding(input_tensor, weights, name):
    batch_size, sequence_size = input_tensor.shape
    result = pnp.named_ndarray((batch_size, sequence_size, weights.shape[1]), name=name)
    for batch_index in range(batch_size):
        for sequence_index in range(sequence_size):
            result = result.set_at_indices(
                (batch_index, sequence_index), weights[input_tensor[batch_index, sequence_index]]
            )
    return result


def functional_multi_head_attention(
    hidden_states, attention_mask, parameters, encoder_index, sequence_size, num_heads, head_size
):

    batch_size = hidden_states.shape[0]

    query = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"]
    query = query + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"]
    query = query.view(batch_size, sequence_size, num_heads, head_size)
    query = query.transpose(2, 1)

    key = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"]
    key = key + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"]
    key = key.view(batch_size, sequence_size, num_heads, head_size)
    key = key.transpose(2, 1)
    key = key.transpose(3, 2)

    value = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"]
    value = value + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"]
    value = value.view(batch_size, sequence_size, num_heads, head_size)
    value = value.transpose(2, 1)

    attention_scores = query @ key

    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = torch.nn.functional.softmax(attention_scores, dim=-1)

    context_layer = attention_probs @ value

    context_layer = context_layer.transpose(2, 1).contiguous()
    context_layer = context_layer.view(batch_size, sequence_size, num_heads * head_size)

    self_output = context_layer @ parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"]
    self_output = self_output + parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"]

    return self_output


def functional_layer_norm(input_tensor, weight, bias, epsilon=0):
    """
    mean = torch.mean(input_tensor, dim=-1, keepdim=True)
    var = torch.square(input_tensor - mean).mean(dim=-1, keepdim=True)
    output = (input_tensor - mean) / torch.sqrt(var + epsilon)
    output *= weight
    output += bias
    """

    output = torch.nn.functional.layer_norm(
        input_tensor,
        (input_tensor.shape[-1],),
        weight,
        bias,
        eps=epsilon,
    )
    return output


def functional_feedforward(hidden_states, parameters, encoder_index):
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"]
    hidden_states = torch.nn.functional.gelu(hidden_states)
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"]
    return hidden_states


def functional_bert_encoder(
    hidden_states, attention_mask, parameters, encoder_index, sequence_size, num_heads, head_size
):

    multi_head_attention_output = functional_multi_head_attention(
        hidden_states, attention_mask, parameters, encoder_index, sequence_size, num_heads, head_size
    )
    multi_head_attention_add_and_layer_norm_output = functional_layer_norm(
        hidden_states + multi_head_attention_output,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = functional_feedforward(
        multi_head_attention_add_and_layer_norm_output, parameters, encoder_index
    )

    feedforward_add_and_layer_norm_output = functional_layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
    )

    return feedforward_add_and_layer_norm_output


def functional_bert(
    input_ids, token_type_ids, attention_mask, parameters, num_encoders, sequence_size, num_heads, head_size
):

    word_embeddings = functional_embedding(
        input_ids, parameters["bert.embeddings.word_embeddings.weight"], "word_embeddings"
    )
    return word_embeddings
    # token_type_embeddings = torch.nn.functional.embedding(token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"])
    # embeddings = word_embeddings + token_type_embeddings
    #
    # encoder_input = functional_layer_norm(
    #     embeddings,
    #     parameters["bert.embeddings.LayerNorm.weight"],
    #     parameters["bert.embeddings.LayerNorm.bias"],
    # )
    #
    # encoder_output = None
    # for encoder_index in range(num_encoders):
    #     encoder_output = functional_bert_encoder(encoder_input, attention_mask, parameters, encoder_index, sequence_size, num_heads, head_size)
    #     encoder_input = encoder_output
    # return encoder_output


def functional_bert_for_question_answering(
    input_ids, token_type_ids, attention_mask, parameters, num_encoders, sequence_size, num_heads, head_size
):
    bert_output = functional_bert(
        input_ids, token_type_ids, attention_mask, parameters, num_encoders, sequence_size, num_heads, head_size
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"]
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]
    return qa_outputs


@pytest.mark.parametrize("functional_bert_function", [functional_bert])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [30522])
def test_functional_bert_vs_transformers_bert(
    functional_bert_function, batch_size, num_encoders, sequence_size, num_heads, head_size, vocab_size
):

    config = transformers.models.bert.configuration_bert.BertConfig()
    config.hidden_dropout_prob = 0.0  # Disable dropout after the embeddings
    config.attention_probs_dropout_prob = 0.0  # Disable dropout in the self-attention
    config.position_embedding_type = None  # Disable position embedding

    input_ids = create_random_torch_long_tensor(batch_size, sequence_size, minimum=0, maximum=vocab_size)
    token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.long)

    if functional_bert_function == functional_bert:
        model = transformers.models.bert.modeling_bert.BertModel(config)
        transformers_output = model(input_ids)["last_hidden_state"]

    elif functional_bert_function == functional_bert_for_question_answering:
        model = transformers.models.bert.modeling_bert.BertForQuestionAnswering(config)
        qa_outputs = model(input_ids)
        start_logits = qa_outputs["start_logits"].reshape((1, 128, 1))
        end_logits = qa_outputs["end_logits"].reshape((1, 128, 1))
        transformers_output = torch.cat((start_logits, end_logits), dim=-1)

    transformers_parameters = model.state_dict()
    parameters = {}
    for name, value in transformers_parameters.items():
        new_value = value
        if "weight" in name and "embedding" not in name:
            new_value = value.T
        parameters[name] = new_value

    if functional_bert_function == functional_bert:
        # Update parameter names to include "bert." prefix to match the names of parameters in the models with heads
        parameters = {f"bert.{name}": value for name, value in parameters.items()}

    output = functional_bert_function(
        pnp.asarray(input_ids.numpy(), name="input_ids"),
        pnp.asarray(token_type_ids.numpy(), name="token_type_ids"),
        None,
        {
            key: pnp.asarray(value.numpy(), name=key)
            for key, value in parameters.items()
            if key == "bert.embeddings.word_embeddings.weight"
        },
        num_encoders,
        sequence_size,
        num_heads,
        head_size,
    )
    # output = torch.from_numpy(output.numpy())

    # assert torch.allclose(output, transformers_output)
