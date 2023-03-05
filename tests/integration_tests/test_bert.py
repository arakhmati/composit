import numpy as np
import pytest

import torch
import transformers

import composit as cnp
from composit.nn.module import wrap_module


def create_random_float(shape, minimum=-0.1, maximum=0.1):
    return np.random.uniform(minimum, maximum, shape).astype(np.float64)


def create_random_long(shape, minimum, maximum):
    return np.random.randint(minimum, maximum, shape, dtype=np.int64)


def create_parameters(num_encoders, hidden_size, vocab_size, num_question_answering_labels=None):

    intermediate_size = hidden_size * 4

    parameters = {
        "bert.embeddings.word_embeddings.weight": create_random_float((vocab_size, hidden_size)),
        "bert.embeddings.token_type_embeddings.weight": np.zeros((2, hidden_size), dtype=np.float64),
        "bert.embeddings.LayerNorm.weight": create_random_float((hidden_size,)),
        "bert.embeddings.LayerNorm.bias": create_random_float((hidden_size,)),
    }

    for encoder_index in range(num_encoders):
        parameters.update(
            {
                f"bert.encoder.layer.{encoder_index}.attention.self.query.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.query.bias": create_random_float((hidden_size,)),
                f"bert.encoder.layer.{encoder_index}.attention.self.key.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.key.bias": create_random_float((hidden_size,)),
                f"bert.encoder.layer.{encoder_index}.attention.self.value.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"bert.encoder.layer.{encoder_index}.attention.self.value.bias": create_random_float((hidden_size,)),
                f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight": create_random_float(
                    (hidden_size, hidden_size)
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias": create_random_float((hidden_size,)),
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight": create_random_float(
                    (hidden_size,)
                ),
                f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias": create_random_float(
                    (hidden_size,)
                ),
                f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight": create_random_float(
                    (hidden_size, intermediate_size)
                ),
                f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias": create_random_float(
                    (intermediate_size,)
                ),
                f"bert.encoder.layer.{encoder_index}.output.dense.weight": create_random_float(
                    (intermediate_size, hidden_size)
                ),
                f"bert.encoder.layer.{encoder_index}.output.dense.bias": create_random_float(hidden_size),
                f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight": create_random_float((hidden_size,)),
                f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias": create_random_float((hidden_size,)),
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
    num_heads,
    head_size,
):
    batch_size = hidden_states.shape[0]

    query = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.weight"]
    query = query + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.query.bias"]
    query = cnp.reshape(query, (batch_size, sequence_size, num_heads, head_size))
    query = cnp.transpose(query, (0, 2, 1, 3))

    key = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.weight"]
    key = key + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.key.bias"]
    key = cnp.reshape(key, (batch_size, sequence_size, num_heads, head_size))
    key = cnp.transpose(key, (0, 2, 3, 1))

    value = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.weight"]
    value = value + parameters[f"bert.encoder.layer.{encoder_index}.attention.self.value.bias"]
    value = cnp.reshape(value, (batch_size, sequence_size, num_heads, head_size))
    value = cnp.transpose(value, (0, 2, 1, 3))

    attention_scores = query @ key

    attention_scores = attention_scores / (head_size**0.5)
    if attention_mask is not None:
        attention_scores = attention_scores + attention_mask

    attention_probs = functional_softmax(attention_scores, axis=-1)

    context_layer = attention_probs @ value

    context_layer = cnp.transpose(context_layer, (0, 2, 1, 3))
    context_layer = cnp.reshape(context_layer, (batch_size, sequence_size, num_heads * head_size))

    self_output = context_layer @ parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.weight"]
    self_output = self_output + parameters[f"bert.encoder.layer.{encoder_index}.attention.output.dense.bias"]

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
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.intermediate.dense.bias"]
    hidden_states = cnp.nn.gelu(hidden_states)
    hidden_states = hidden_states @ parameters[f"bert.encoder.layer.{encoder_index}.output.dense.weight"]
    hidden_states = hidden_states + parameters[f"bert.encoder.layer.{encoder_index}.output.dense.bias"]
    return hidden_states


@wrap_module
def functional_bert_encoder(
    hidden_states,
    attention_mask,
    parameters,
    *,
    encoder_index,
    sequence_size,
    num_heads,
    head_size,
):

    multi_head_attention_output = functional_multi_head_attention(
        hidden_states,
        attention_mask,
        parameters,
        encoder_index=encoder_index,
        sequence_size=sequence_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    multi_head_attention_add_and_layer_norm_output = functional_layer_norm(
        hidden_states + multi_head_attention_output,
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
    )

    feedforward_output = functional_feedforward(
        multi_head_attention_add_and_layer_norm_output, parameters, encoder_index=encoder_index
    )

    feedforward_add_and_layer_norm_output = functional_layer_norm(
        multi_head_attention_add_and_layer_norm_output + feedforward_output,
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.weight"],
        parameters[f"bert.encoder.layer.{encoder_index}.output.LayerNorm.bias"],
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
    num_heads,
    head_size,
):

    word_embeddings = cnp.nn.embedding(input_ids, parameters["bert.embeddings.word_embeddings.weight"])
    token_type_embeddings = cnp.nn.embedding(token_type_ids, parameters["bert.embeddings.token_type_embeddings.weight"])
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
            encoder_index=encoder_index,
            sequence_size=sequence_size,
            num_heads=num_heads,
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
    num_heads,
    head_size,
):
    bert_output = functional_bert(
        input_ids,
        token_type_ids,
        attention_mask,
        parameters,
        num_encoders=num_encoders,
        sequence_size=sequence_size,
        num_heads=num_heads,
        head_size=head_size,
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
    config.num_attention_heads = num_heads
    config.num_hidden_layers = num_encoders
    config.hidden_size = num_heads * head_size
    config.vocab_size = vocab_size

    if functional_bert_function == functional_bert:
        transformers_model = transformers.models.bert.modeling_bert.BertModel(config)

        def torch_forward(*args):
            return transformers_model(*args)["last_hidden_state"]

    elif functional_bert_function == functional_bert_for_question_answering:
        transformers_model = transformers.models.bert.modeling_bert.BertForQuestionAnswering(config)

        def torch_forward(*args):
            qa_outputs = transformers_model(*args)
            start_logits = qa_outputs["start_logits"].reshape((batch_size, sequence_size, 1))
            end_logits = qa_outputs["end_logits"].reshape((batch_size, sequence_size, 1))
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

    input_ids_var = cnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size))
    token_type_ids_var = cnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size))
    parameters = {cnp.nn.variable(name=name, shape=value.shape): parameters[name] for name, value in parameters.items()}

    model = functional_bert_function(
        input_ids_var,
        token_type_ids_var,
        None,
        {var.node.name: var for var in parameters.keys()},
        num_encoders=num_encoders,
        sequence_size=sequence_size,
        num_heads=num_heads,
        head_size=head_size,
    )

    model_inputs = []
    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)
        model_inputs.append((input_ids, token_type_ids))

    transformers_outputs = []
    for model_input in model_inputs:
        model_input = [torch.from_numpy(x) for x in model_input]
        transformers_outputs.append(torch_forward(model_input[0]))

    cnp_outputs = []
    for model_input in model_inputs:
        input_ids, token_type_ids = model_input
        output = cnp.nn.evaluate(
            model,
            inputs={
                input_ids_var: input_ids,
                token_type_ids_var: token_type_ids,
                **parameters,
            },
        )
        cnp_outputs.append(output)

    for output, transformers_output in zip(cnp_outputs, transformers_outputs):
        transformers_output = transformers_output.detach().numpy()
        assert np.allclose(output, transformers_output, atol=1e-3)


@pytest.mark.parametrize("num_inputs", [2])
@pytest.mark.parametrize("batch_size", [2])
@pytest.mark.parametrize("num_encoders", [3])
@pytest.mark.parametrize("sequence_size", [32])
@pytest.mark.parametrize("num_heads", [3])
@pytest.mark.parametrize("head_size", [16])
@pytest.mark.parametrize("vocab_size", [16])
def test_functional_bert_autograd(
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_heads,
    head_size,
    vocab_size,
):
    cnp.nn.module.DISABLE = True

    config = transformers.models.bert.configuration_bert.BertConfig()
    config.hidden_dropout_prob = 0.0  # Disable dropout after the embeddings
    config.attention_probs_dropout_prob = 0.0  # Disable dropout in the self-attention
    config.position_embedding_type = None  # Disable position embedding
    config.num_attention_heads = num_heads
    config.num_hidden_layers = num_encoders
    config.hidden_size = num_heads * head_size
    config.vocab_size = vocab_size

    transformers_model = transformers.models.bert.modeling_bert.BertModel(config)
    for parameter in transformers_model.parameters():
        parameter.requires_grad = True
    transformers_parameters = {name: parameter for name, parameter in transformers_model.named_parameters()}

    def torch_forward(*args):
        hidden_states = transformers_model(*args)["last_hidden_state"]
        return hidden_states

    parameters = {}
    for name, value in transformers_parameters.items():
        new_value = value.detach()
        if "weight" in name and "embedding" not in name:
            new_value = new_value.T
        parameters[name] = new_value.numpy()
    parameters = {f"bert.{name}": value for name, value in parameters.items()}

    input_ids_variable = cnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size))
    token_type_ids_variable = cnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size))
    parameter_variables = {name: cnp.nn.variable(name=name, shape=value.shape) for name, value in parameters.items()}
    model: cnp.PersistentArray = functional_bert(
        input_ids_variable,
        token_type_ids_variable,
        None,
        parameter_variables,
        num_encoders=num_encoders,
        sequence_size=sequence_size,
        num_heads=num_heads,
        head_size=head_size,
    )
    loss = model

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

    accumulated_gradients = {}
    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)
        torch_loss = torch_forward(torch.from_numpy(input_ids), torch.from_numpy(token_type_ids))

        incoming_gradient = create_random_float((batch_size, sequence_size, num_heads * head_size), -0.001, 0.001)
        torch_loss.backward(torch.from_numpy(incoming_gradient))

        cnp_gradients = cnp.nn.differentiate(
            [loss],
            input_vars_to_differentiate,
            {
                input_ids_variable: input_ids,
                token_type_ids_variable: token_type_ids,
                **{parameter_variables[key]: parameters[key] for key in parameters},
            },
            {loss: incoming_gradient},
        )
        assert len(cnp_gradients) == len(input_vars_to_differentiate)
        cnp_gradients = {var.node.name: value for var, value in cnp_gradients.items()}

        # Accumulate gradients
        for key, value in cnp_gradients.items():
            if key in accumulated_gradients:
                accumulated_gradients[key] += value
            else:
                accumulated_gradients[key] = value

        for name, cnp_gradient in reversed(accumulated_gradients.items()):
            torch_parameter = transformers_parameters[name.replace("bert.", "")]
            torch_gradient = torch_parameter.grad.numpy()
            if "weight" in name and "embedding" not in name:
                torch_gradient = torch_gradient.T

            all_close = np.allclose(cnp_gradient, torch_gradient, atol=1e-3)
            assert all_close

    cnp.nn.module.DISABLE = False
