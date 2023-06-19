import numpy as np
import pytest

import torch
import transformers

import composit as cnp

from model_zoo.bert import (
    create_random_float,
    create_random_long,
    create_bert_config,
    bert,
    convert_parameters_to_numpy,
)


@pytest.mark.parametrize("num_inputs", [4])
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_encoders", [12])
@pytest.mark.parametrize("sequence_size", [128])
@pytest.mark.parametrize("num_attention_heads", [12])
@pytest.mark.parametrize("head_size", [64])
@pytest.mark.parametrize("vocab_size", [30522])
def test_torch_vs_composit(
    num_inputs,
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

    parameters = {
        name: cnp.asarray(value, name=name) for name, value in convert_parameters_to_numpy(transformers_model).items()
    }

    model_inputs = []
    for _ in range(num_inputs):
        input_ids = create_random_long((batch_size, sequence_size), minimum=0, maximum=vocab_size)
        token_type_ids = np.zeros((batch_size, sequence_size), dtype=np.int64)
        model_inputs.append((input_ids, token_type_ids))

    transformers_outputs = []
    for model_input in model_inputs:
        input_ids, token_type_ids = [torch.from_numpy(x) for x in model_input]
        transformers_outputs.append(transformers_model(input_ids, token_type_ids=token_type_ids)["last_hidden_state"])

    cnp_outputs = []
    for model_input in model_inputs:
        input_ids, token_type_ids = model_input

        input_ids_var = cnp.asarray(input_ids, name="input_ids")
        token_type_ids_var = cnp.asarray(token_type_ids, name="token_type_ids")

        output_var = bert(
            input_ids_var,
            token_type_ids_var,
            None,
            parameters,
            num_encoders=num_encoders,
            head_size=head_size,
        )

        output = cnp.nn.evaluate(output_var)
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
def test_autograd(
    num_inputs,
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
    for parameter in transformers_model.parameters():
        parameter.requires_grad = True

    parameters = convert_parameters_to_numpy(transformers_model)
    parameter_variables = {name: cnp.asarray(value, name=name) for name, value in parameters.items()}

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

        torch_output = transformers_model(torch.from_numpy(input_ids), token_type_ids=torch.from_numpy(token_type_ids))[
            "last_hidden_state"
        ]

        incoming_gradient = create_random_float(
            (batch_size, sequence_size, num_attention_heads * head_size), -0.001, 0.001
        )
        torch_output.backward(torch.from_numpy(incoming_gradient))

        input_ids_var = cnp.asarray(input_ids, name="input_ids")
        token_type_ids_var = cnp.asarray(token_type_ids, name="token_type_ids")

        output_var: cnp.LazyTensor = bert(
            input_ids_var,
            token_type_ids_var,
            None,
            parameter_variables,
            num_encoders=num_encoders,
            head_size=head_size,
        )

        cnp_gradients = cnp.nn.chain_rule({output_var: cnp.asarray(incoming_gradient)}, input_vars_to_differentiate)
        cnp_gradients = cnp.nn.evaluate(*cnp_gradients)
        cnp_gradients = {
            input_var.name: gradient for gradient, input_var in zip(cnp_gradients, input_vars_to_differentiate)
        }

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
