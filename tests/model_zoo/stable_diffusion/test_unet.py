import diffusers
import numpy as np
import torch

import flashlight.tracer
from flashlight.tensor import forward, from_torch


def test_torch_vs_composit(
    batch_size=1, tokenizer_model_max_length=77, clip_hidden_size=768, height=16, width=16, num_inference_steps=1
):
    unet = diffusers.UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    scheduler = diffusers.LMSDiscreteScheduler(
        beta_start=0.00085, beta_end=0.012, beta_schedule="scaled_linear", num_train_timesteps=1000
    )
    scheduler.set_timesteps(num_inference_steps)

    latents = (
        torch.randn(
            (batch_size, unet.in_channels, height // 8, width // 8),
            generator=torch.manual_seed(0),
        )
        * scheduler.init_noise_sigma
    )

    # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
    latent_model_input = torch.cat([latents] * 2)

    timestep = from_torch(torch.tensor(0))
    latent_model_input = from_torch(scheduler.scale_model_input(latent_model_input, timestep=timestep))
    text_embeddings = from_torch(torch.randn(2, tokenizer_model_max_length, clip_hidden_size))

    with flashlight.tracer.trace(), torch.no_grad():
        torch_noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        assert len(torch_noise_pred.graph) == 3776

        composit_noise_pred = forward(torch_noise_pred, input_tensors=[latent_model_input, timestep, text_embeddings])
        assert np.allclose(composit_noise_pred, torch_noise_pred.detach().numpy(), atol=1e-5)
