"""
https://huggingface.co/blog/stable_diffusion
"""

from dataclasses import dataclass

import diffusers
from PIL import Image
import torch
import transformers
from tqdm.auto import tqdm


@dataclass(kw_only=True)
class TorchStableDiffusion:
    vae: torch.nn.Module
    tokenizer: transformers.PreTrainedTokenizer
    text_encoder: torch.nn.Module
    unet: torch.nn.Module


def load_torch_stable_diffusion():
    # 1. Load the autoencoder model which will be used to decode the latents into image space.
    vae = diffusers.AutoencoderKL.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="vae")

    # 2. Load the tokenizer and text encoder to tokenize and encode the text.
    tokenizer = transformers.CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    text_encoder = transformers.CLIPTextModel.from_pretrained("openai/clip-vit-large-patch14")

    # 3. The UNet model for generating the latents.
    unet = diffusers.UNet2DConditionModel.from_pretrained("CompVis/stable-diffusion-v1-4", subfolder="unet")

    return TorchStableDiffusion(vae=vae, tokenizer=tokenizer, text_encoder=text_encoder, unet=unet)


def evaluate(stable_diffusion: TorchStableDiffusion, prompt, height, width, num_inference_steps, guidance_scale):
    vae = stable_diffusion.vae
    tokenizer = stable_diffusion.tokenizer
    text_encoder = stable_diffusion.text_encoder
    unet = stable_diffusion.unet

    text_input = tokenizer(
        prompt, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt"
    )
    text_embeddings = text_encoder(text_input.input_ids)[0]

    batch_size = len(prompt)
    max_length = text_input.input_ids.shape[-1]
    uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
    uncond_embeddings = text_encoder(uncond_input.input_ids)[0]

    text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

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

    for timestep in tqdm(scheduler.timesteps):
        # expand the latents if we are doing classifier-free guidance to avoid doing two forward passes.
        latent_model_input = torch.cat([latents] * 2)

        latent_model_input = scheduler.scale_model_input(latent_model_input, timestep=timestep)

        # predict the noise residual
        with torch.no_grad():
            noise_pred = unet(latent_model_input, timestep, encoder_hidden_states=text_embeddings).sample

        # perform guidance
        noise_pred_uncond, noise_pred_text = noise_pred.chunk(2)
        noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_text - noise_pred_uncond)

        # compute the previous noisy sample x_t -> x_t-1
        latents = scheduler.step(noise_pred, timestep, latents).prev_sample

    # scale and decode the image latents with vae
    latents = 1 / 0.18215 * latents
    with torch.no_grad():
        images = vae.decode(latents).sample

    images = (images / 2 + 0.5).clamp(0, 1)
    images = images.detach().cpu().permute(0, 2, 3, 1).numpy()
    images = (images * 255).round().astype("uint8")
    pil_image, *_ = [Image.fromarray(image) for image in images]
    return pil_image


def test_torch(
    prompt="a photograph of an astronaut riding a horse", height=16, width=16, num_inference_steps=1, guidance_scale=7.5
):
    torch_stable_diffusion = load_torch_stable_diffusion()
    evaluate(
        torch_stable_diffusion,
        prompt=[prompt],
        height=height,
        width=width,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
    )
