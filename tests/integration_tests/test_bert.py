import numpy as np
import pytest

import torch
import transformers

import persistent_numpy as pnp


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

    return {name: array.numpy() for name, array in parameters.items()}


def functional_softmax(input_tensor, axis):
    exp_input_tensor = pnp.exp(input_tensor - pnp.max(input_tensor, axis=axis, keepdims=True))
    return exp_input_tensor / pnp.sum(exp_input_tensor, axis=axis, keepdims=True)


def functional_multi_head_attention(
    hidden_states,
    attention_mask,
    parameters,
    encoder_index,
    sequence_size,
    num_heads,
    head_size,
):
    batch_size = hidden_states.shape[0]

    query = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"]
    query = query + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"]
    query = pnp.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = pnp.transpose(query, (0, 2, 1, 3))

    key = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"]
    key = key + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"]
    key = pnp.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = pnp.transpose(key, (0, 2, 3, 1))

    value = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"]
    value = value + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"]
    value = pnp.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = pnp.transpose(value, (0, 2, 1, 3))

    attention_scores = query @ key

    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = functional_softmax(attention_scores, axis=-1)

    context_layer = attention_probs @ value

    context_layer = pnp.transpose(context_layer, (0, 2, 1, 3))
    context_layer = pnp.reshape(context_layer, (batch_size, sequence_size, num_heads * head_size))

    self_output = context_layer @ parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"]
    self_output = self_output + parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"]

    return self_output


def functional_layer_norm(input_tensor, weight, bias, epsilon=0):
    mean = pnp.mean(input_tensor, axis=-1, keepdims=True)
    input_tensor_minus_mean = input_tensor - mean
    var = pnp.mean(pnp.square(input_tensor_minus_mean), axis=-1, keepdims=True)
    output = input_tensor_minus_mean / pnp.sqrt(var + epsilon)
    output *= weight
    output += bias
    return output
    """
    mean = pnp.mean(input_tensor, axis=-1, keepdims=True)
    var = pnp.sqrt(pnp.var(input_tensor, axis=-1, keepdims=True) + epsilon)
    output = (input_tensor - mean) / var
    return output * weight + bias
    """


def functional_feedforward(hidden_states, parameters, encoder_index):
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"]
    hidden_states = pnp.nn.gelu(hidden_states)
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"]
    return hidden_states


def functional_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    encoder_index,
    sequence_size,
    num_heads,
    head_size,
):

    multi_head_attention_output = functional_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters,
        encoder_index,
        sequence_size,
        num_heads,
        head_size,
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
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    num_encoders,
    sequence_size,
    num_heads,
    head_size,
):

    word_embeddings = pnp.nn.embedding(input_ids, parameters["bert.embeddings.word_embeddings.weight"])
    token_type_embeddings = pnp.nn.embedding(token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"])
    embeddings = word_embeddings + token_type_embeddings

    encoder_input = functional_layer_norm(
        embeddings,
        parameters["bert.embeddings.LayerNorm.weight"],
        parameters["bert.embeddings.LayerNorm.bias"],
    )

    encoder_output = None
    for encoder_index in range(num_encoders):
        encoder_output = functional_bert_encoder(
            encoder_input,
            attention_mask,
            parameters,
            encoder_index,
            sequence_size,
            num_heads,
            head_size,
        )
        encoder_input = encoder_output
    return encoder_output


def functional_bert_for_question_answering(
    input_ids,
    token_type_ids,
    attention_mask,
    parameters,
    num_encoders,
    sequence_size,
    num_heads,
    head_size,
):
    bert_output = functional_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders,
        sequence_size,
        num_heads,
        head_size,
    )

    qa_outputs = bert_output
    qa_outputs = qa_outputs @ parameters["qa_outputs.weight"]
    qa_outputs = qa_outputs + parameters["qa_outputs.bias"]
    return qa_outputs


@pytest.mark.parametrize(
    "functional_bert_function",
    [functional_bert, functional_bert_for_question_answering],
)
@pytest.mark.parametrize("num_inputs", [4])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [30522])
def test_functional_bert_vs_transformers_bert(
    functional_bert_function,
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_heads,
    head_size,
    vocab_size,
):

    config = transformers.models.bert.configuration_bert.BertConfig()
    config.hidden_dropout_prob = 0.0  # Disable dropout after the embeddings
    config.attention_probs_dropout_prob = 0.0  # Disable dropout in the self-attention
    config.position_embedding_type = None  # Disable position embedding

    if functional_bert_function == functional_bert:
        transformers_model = transformers.models.bert.modeling_bert.BertModel(config)

        def compute_torch(*args):
            return transformers_model(*args)["last_hidden_state"]

    elif functional_bert_function == functional_bert_for_question_answering:
        transformers_model = transformers.models.bert.modeling_bert.BertForQuestionAnswering(config)

        def compute_torch(*args):
            qa_outputs = transformers_model(*args)
            start_logits = qa_outputs["start_logits"].reshape((1, 128, 1))
            end_logits = qa_outputs["end_logits"].reshape((1, 128, 1))
            return torch.cat((start_logits, end_logits), dim=-1)

    else:
        raise ValueError("Unknown")

    transformers_parameters = transformers_model.state_dict()
    parameters = {}
    for name, value in transformers_parameters.items():
        new_value = value
        if "weight" in name and "embedding" not in name:
            new_value = value.T
        parameters[name] = new_value.numpy()

    if functional_bert_function == functional_bert:
        # Update parameter names to include "bert." prefix to match the names of parameters in the models with heads
        parameters = {f"bert.{name}": value for name, value in parameters.items()}

    model = functional_bert_function(
        pnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size)),
        pnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size)),
        None,
        {name: pnp.nn.variable(name=name, shape=value.shape) for name, value in parameters.items()},
        num_encoders,
        sequence_size,
        num_heads,
        head_size,
    )

    model_inputs = []
    for _ in range(num_inputs):
        input_ids = create_random_torch_long_tensor(batch_size, sequence_size, minimum=0, maximum=vocab_size)
        token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.long)
        model_inputs.append((input_ids, token_type_ids))

    transformers_outputs = []
    for model_input in model_inputs:
        transformers_outputs.append(compute_torch(model_input[0]))

    pnp_outputs = []
    for model_input in model_inputs:
        input_ids, token_type_ids = model_input
        output = pnp.nn.evaluate(
            model,
            inputs=dict(
                input_ids=input_ids.numpy(),
                token_type_ids=token_type_ids.numpy(),
                **parameters,
            ),
        )
        pnp_outputs.append(output)

    for output, transformers_output in zip(pnp_outputs, transformers_outputs):
        output = torch.from_numpy(output)
        assert torch.allclose(output, transformers_output.double(), atol=1e-5)


@pytest.mark.parametrize("num_inputs", [1])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [30522])
def test_functional_bert_autograd(
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_heads,
    head_size,
    vocab_size,
):

    config = transformers.models.bert.configuration_bert.BertConfig()
    config.hidden_dropout_prob = 0.0  # Disable dropout after the embeddings
    config.attention_probs_dropout_prob = 0.0  # Disable dropout in the self-attention
    config.position_embedding_type = None  # Disable position embedding

    parameters = create_parameters(config.num_hidden_layers, config.hidden_size, config.vocab_size)

    input_ids_variable = pnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size))
    token_type_ids_variable = pnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size))
    parameter_variables = {name: pnp.nn.variable(name=name, shape=value.shape) for name, value in parameters.items()}
    model: pnp.PersistentArray = functional_bert(
        input_ids_variable,
        token_type_ids_variable,
        None,
        parameter_variables,
        num_encoders,
        sequence_size,
        num_heads,
        head_size,
    )
    loss = pnp.sum(model, keepdims=True)

    input_vars_to_differentiate = [
        parameter_variables["bert.embeddings.LayerNorm.weight"],
        parameter_variables["bert.embeddings.LayerNorm.bias"],
    ]
    for encoder_index in range(num_encoders):
        input_vars_to_differentiate.extend(
            [
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.output.dense.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.output.dense.bias"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
                parameter_variables[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
            ]
        )

    for _ in range(num_inputs):
        input_ids = create_random_torch_long_tensor(batch_size, sequence_size, minimum=0, maximum=vocab_size)
        token_type_ids = torch.zeros(batch_size, sequence_size, dtype=torch.long)
        gradients = pnp.nn.differentiate(
            [loss],
            input_vars_to_differentiate,
            {
                input_ids_variable: input_ids.numpy(),
                token_type_ids_variable: token_type_ids.numpy(),
                **{parameter_variables[key]: parameters[key] for key in parameters},
            },
            {loss: np.asarray(1.0)},
        )
        assert len(gradients) == len(input_vars_to_differentiate)

    assert loss.shape == (1, 1, 1)
