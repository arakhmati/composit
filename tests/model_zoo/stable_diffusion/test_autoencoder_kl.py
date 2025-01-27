import pytest

import diffusers
import numpy as np
import torch


import composit as cnp
from model_zoo.stable_diffusion.autoencoder_kl import decoder, convert_parameters_to_numpy


@pytest.mark.xfail(reason="Broken after updating dependencies")
@pytest.mark.parametrize("batch_size", [1])
@pytest.mark.parametrize("height", [16])
@pytest.mark.parametrize("width", [16])
@pytest.mark.parametrize("channels_last", [True])
def test_torch_vs_composit(batch_size, height, width, channels_last, latent_channels=4, reduction_factor=8):
    np.random.seed(0)

    torch_model = diffusers.AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae").eval()
    assert not torch_model.use_slicing
    assert not torch_model.use_tiling

    latents = np.random.random(
        (batch_size, latent_channels, height // reduction_factor, width // reduction_factor)
    ).astype(np.float32)

    torch_decoder = torch_model.decoder
    torch_output = torch_decoder(torch.from_numpy(latents)).detach().numpy()

    parameters = {
        name: cnp.asarray(value, name)
        for name, value in convert_parameters_to_numpy(torch_decoder, channels_last).items()
    }

    latents_var = cnp.asarray(latents, name="latents")
    model = decoder(latents_var, parameters, channels_last=channels_last)

    output = cnp.nn.evaluate(model)

    assert np.allclose(output, torch_output, atol=1e-4)
