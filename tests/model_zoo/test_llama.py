# TODO(arakhmati): remove the line below once all of the lines are uncommented
# ruff: noqa: F401
import pytest

import numpy as np
import torch
import transformers

import composit as cnp
import flashlight


@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("num_hidden_layers", [1])
@pytest.mark.parametrize("sequence_size", [64])
def test_torch_vs_composit(batch_size, num_hidden_layers, sequence_size):
    llama_config = transformers.models.llama.configuration_llama.LlamaConfig(num_hidden_layers=num_hidden_layers)

    hugging_face_reference_model = transformers.models.llama.LlamaForCausalLM(llama_config)
    hugging_face_reference_model.eval()

    llama_input = torch.randint(0, 100, (batch_size, sequence_size))

    with flashlight.tracer.trace(run_torch=True), torch.no_grad():
        flashlight_output = hugging_face_reference_model(llama_input).logits

    torch_output = flashlight_output.detach().numpy()

    assert len(flashlight_output.graph) == 119
    composit_output = cnp.evaluate(flashlight_output.lazy_tensor)

    assert torch_output.shape == composit_output.shape
    # assert np.allclose(composit_output, torch_output, atol=1e-1, rtol=1e-1)
