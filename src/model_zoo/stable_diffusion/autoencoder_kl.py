import torch

import composit as cnp
import composit.nn
from composit.nn.layers import group_norm, multi_head_attention
from composit.nn.core import wrap_as_instruction


@wrap_as_instruction()
def interpolate(input_tensor, channels_last):
    if channels_last:
        input_tensor = input_tensor.transpose((0, 3, 1, 2))
    input_tensor = torch.from_numpy(input_tensor)
    output_tensor = torch.nn.functional.interpolate(input_tensor, scale_factor=2.0, mode="nearest")
    output_tensor = output_tensor.detach().numpy()
    if channels_last:
        output_tensor = output_tensor.transpose((0, 2, 3, 1))
    return output_tensor


def convert_parameters_to_numpy(model, channels_last):
    parameters = {}
    for name, value in model.state_dict().items():
        new_value = value.detach().numpy()
        if ("conv" in name and "weight" in name) and channels_last and len(value.shape) == 4:
            new_value = new_value.transpose((0, 2, 3, 1))
        elif "query.weight" in name or "key.weight" in name or "value.weight" in name or "proj_attn.weight" in name:
            new_value = new_value.transpose((1, 0))
        parameters[name] = new_value
    return parameters


def attention_block(
    hidden_states,
    group_norm_weight,
    group_norm_bias,
    query_weight,
    query_bias,
    key_weight,
    key_bias,
    value_weight,
    value_bias,
    output_weight,
    output_bias,
    channels_last,
):
    residual = hidden_states

    hidden_states = group_norm(
        hidden_states,
        group_norm_weight,
        group_norm_bias,
        channel_axis=-1 if channels_last else 1,
        num_groups=32,
        epsilon=1e-6,
    )

    batch_size, height, width, num_channels = hidden_states.shape
    hidden_states = cnp.reshape(hidden_states, (batch_size, height * width, num_channels))

    hidden_states = multi_head_attention(
        hidden_states,
        None,
        query_weight,
        query_bias,
        key_weight,
        key_bias,
        value_weight,
        value_bias,
        output_weight,
        output_bias,
        head_size=num_channels,
    )

    hidden_states = cnp.reshape(hidden_states, (batch_size, height, width, num_channels))

    return hidden_states + residual


def resnet_block(
    hidden_states,
    group_norm_0_weight,
    group_norm_0_bias,
    conv_0_weight,
    conv_0_bias,
    group_norm_1_weight,
    group_norm_1_bias,
    conv_1_weight,
    conv_1_bias,
    conv_shortcut_weight,
    conv_shortcut_bias,
    channels_last,
):
    residual = hidden_states

    hidden_states = group_norm(
        hidden_states,
        group_norm_0_weight,
        group_norm_0_bias,
        channel_axis=-1 if channels_last else 1,
        num_groups=32,
        epsilon=1e-6,
    )

    hidden_states = cnp.nn.silu(hidden_states)

    hidden_states = cnp.nn.convolution(hidden_states, conv_0_weight, padding=(1, 1), channels_last=channels_last)
    hidden_states += conv_0_bias

    hidden_states = group_norm(
        hidden_states,
        group_norm_1_weight,
        group_norm_1_bias,
        channel_axis=-1 if channels_last else 1,
        num_groups=32,
        epsilon=1e-6,
    )

    hidden_states = cnp.nn.silu(hidden_states)

    hidden_states = cnp.nn.convolution(hidden_states, conv_1_weight, padding=(1, 1), channels_last=channels_last)
    hidden_states += conv_1_bias

    if conv_shortcut_weight is not None:
        residual = cnp.nn.convolution(residual, conv_shortcut_weight, channels_last=channels_last)
        residual += conv_shortcut_bias

    return hidden_states + residual


def decoder(latents, parameters, *, channels_last):
    if channels_last:
        image = cnp.transpose(latents, (0, 2, 3, 1))

    output = cnp.nn.convolution(image, parameters["conv_in.weight"], padding=(1, 1), channels_last=channels_last)
    output += parameters["conv_in.bias"]

    output = resnet_block(
        output,
        parameters["mid_block.resnets.0.norm1.weight"],
        parameters["mid_block.resnets.0.norm1.bias"],
        parameters["mid_block.resnets.0.conv1.weight"],
        parameters["mid_block.resnets.0.conv1.bias"],
        parameters["mid_block.resnets.0.norm2.weight"],
        parameters["mid_block.resnets.0.norm2.bias"],
        parameters["mid_block.resnets.0.conv2.weight"],
        parameters["mid_block.resnets.0.conv2.bias"],
        None,
        None,
        channels_last=channels_last,
    )

    output = attention_block(
        output,
        parameters["mid_block.attentions.0.group_norm.weight"],
        parameters["mid_block.attentions.0.group_norm.bias"],
        parameters["mid_block.attentions.0.query.weight"],
        parameters["mid_block.attentions.0.query.bias"],
        parameters["mid_block.attentions.0.key.weight"],
        parameters["mid_block.attentions.0.key.bias"],
        parameters["mid_block.attentions.0.value.weight"],
        parameters["mid_block.attentions.0.value.bias"],
        parameters["mid_block.attentions.0.proj_attn.weight"],
        parameters["mid_block.attentions.0.proj_attn.bias"],
        channels_last=channels_last,
    )

    output = resnet_block(
        output,
        parameters["mid_block.resnets.1.norm1.weight"],
        parameters["mid_block.resnets.1.norm1.bias"],
        parameters["mid_block.resnets.1.conv1.weight"],
        parameters["mid_block.resnets.1.conv1.bias"],
        parameters["mid_block.resnets.1.norm2.weight"],
        parameters["mid_block.resnets.1.norm2.bias"],
        parameters["mid_block.resnets.1.conv2.weight"],
        parameters["mid_block.resnets.1.conv2.bias"],
        None,
        None,
        channels_last=channels_last,
    )

    for up_block_index in range(4):
        up_block_prefix = f"up_blocks.{up_block_index}"
        for resnet_index in range(3):
            output = resnet_block(
                output,
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.norm1.weight"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.norm1.bias"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.conv1.weight"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.conv1.bias"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.norm2.weight"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.norm2.bias"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.conv2.weight"],
                parameters[f"{up_block_prefix}.resnets.{resnet_index}.conv2.bias"],
                parameters.get(f"{up_block_prefix}.resnets.{resnet_index}.conv_shortcut.weight"),
                parameters.get(f"{up_block_prefix}.resnets.{resnet_index}.conv_shortcut.bias"),
                channels_last=channels_last,
            )

        if up_block_index <= 2:
            output = interpolate(output, channels_last=channels_last)
            output = cnp.nn.convolution(
                output,
                parameters[f"{up_block_prefix}.upsamplers.0.conv.weight"],
                padding=(1, 1),
                channels_last=channels_last,
            )
            output += parameters[f"{up_block_prefix}.upsamplers.0.conv.bias"]

    output = group_norm(
        output,
        parameters["conv_norm_out.weight"],
        parameters["conv_norm_out.bias"],
        channel_axis=-1 if channels_last else 1,
        num_groups=32,
        epsilon=1e-6,
    )

    output = cnp.nn.silu(output)

    output = cnp.nn.convolution(output, parameters["conv_out.weight"], padding=(1, 1), channels_last=channels_last)
    output += parameters["conv_out.bias"]

    if channels_last:
        output = cnp.transpose(output, (0, 3, 1, 2))

    return output
