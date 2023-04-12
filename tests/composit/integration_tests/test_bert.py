import numpy as np
import pytest

import torch
import transformers

import composit as cnp
from composit.nn.module import wrap_module

from model_zoo.bert import (
    create_random_float,
    create_random_long,
    create_bert_config,
    functional_bert,
    convert_parameters_to_numpy,
)


@pytest.mark.parametrize("num_inputs", [4])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [30522])
def test_functional_bert_vs_transformers_bert(
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
):
    config = create_bert_config(
        num_encoders=num_encoders, num_attention_heads=num_attention_heads, head_size=head_size, vocab_size=vocab_size
    )

    transformers_model = transformers.models.bert.modeling_bert.BertModel(config)

    input_ids_var = cnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    token_type_ids_var = cnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    parameters = {
        cnp.nn.variable(name=name, shape=value.shape): value
        for name, value in convert_parameters_to_numpy(transformers_model).items()
    }

    model = functional_bert(
        input_ids_var,
        token_type_ids_var,
        None,
        {var.node.name: var for var in parameters.keys()},
        num_encoders=num_encoders,
        sequence_size=sequence_size,
        num_attention_heads=num_attention_heads,
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
        transformers_outputs.append(transformers_model(model_input[0])["last_hidden_state"])

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
@pytest.mark.parametrize("num_attention_heads", [3])
@pytest.mark.parametrize("head_size", [16])
@pytest.mark.parametrize("vocab_size", [16])
def test_functional_bert_autograd(
    num_inputs,
    batch_size,
    num_encoders,
    sequence_size,
    num_attention_heads,
    head_size,
    vocab_size,
):

    config = create_bert_config(
        num_encoders=num_encoders, num_attention_heads=num_attention_heads, head_size=head_size, vocab_size=vocab_size
    )

    transformers_model = transformers.models.bert.modeling_bert.BertModel(config)
    for parameter in transformers_model.parameters():
        parameter.requires_grad = True

    parameters = convert_parameters_to_numpy(transformers_model)

    input_ids_variable = cnp.nn.variable(name="input_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    token_type_ids_variable = cnp.nn.variable(name="token_type_ids", shape=(batch_size, sequence_size), dtype=np.int64)
    parameter_variables = {name: cnp.nn.variable(name=name, shape=value.shape) for name, value in parameters.items()}

    with cnp.nn.module.disable_modules():
        model: cnp.PersistentArray = functional_bert(
            input_ids_variable,
            token_type_ids_variable,
            None,
            parameter_variables,
            num_encoders=num_encoders,
            sequence_size=sequence_size,
            num_attention_heads=num_attention_heads,
            head_size=head_size,
        )

    loss = model

    input_vars_to_differentiate = [
        parameter_variables["embeddings.LayerNorm.weight"],
        parameter_variables["embeddings.LayerNorm.bias"],
    ]
    for encoder_index in range(num_encoders):
        input_vars_to_differentiate.extend(
            [
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.query.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.query.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.key.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.key.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.value.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.self.value.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.output.dense.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.output.dense.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.attention.output.LayerNorm.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.intermediate.dense.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.intermediate.dense.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.output.dense.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.output.dense.bias"],
                parameter_variables[f"encoder.layer.{encoder_index}.output.LayerNorm.weight"],
                parameter_variables[f"encoder.layer.{encoder_index}.output.LayerNorm.bias"],
            ]
        )

    accumulated_gradients = {}
    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)

        torch_loss = transformers_model(torch.from_numpy(input_ids), torch.from_numpy(token_type_ids))[
            "last_hidden_state"
        ]

        incoming_gradient = create_random_float(
            (batch_size, sequence_size, num_attention_heads * head_size), -0.001, 0.001
        )
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
            torch_parameter = transformers_model.get_parameter(name)
            torch_gradient = torch_parameter.grad.numpy()
            if "weight" in name and "embedding" not in name:
                torch_gradient = torch_gradient.T

            all_close = np.allclose(cnp_gradient, torch_gradient, atol=1e-3)
            assert all_close
