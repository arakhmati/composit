import numpy as np
import torch
import transformers

import flashlight.tracer
from flashlight.tensor import forward, from_torch


def build_causal_attention_mask(*args, **kwargs):
    return None


def test_torch_vs_composit(prompt="a photograph of an astronaut riding a horse"):
    tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder.text_model._build_causal_attention_mask = build_causal_attention_mask

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )

    input_ids = from_torch(text_input.input_ids)
    position_ids = from_torch(torch.arange(text_encoder.config.max_position_embeddings).expand((1, -1)))

    with flashlight.tracer.trace():
        torch_text_embeddings = text_encoder(input_ids, position_ids=position_ids).last_hidden_state
        assert len(torch_text_embeddings.graph) == 958

        composit_text_embeddings = forward(torch_text_embeddings, input_tensors=[input_ids, position_ids])
        assert np.allclose(composit_text_embeddings, torch_text_embeddings.detach().numpy(), atol=1e-5)
