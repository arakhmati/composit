import composit as cnp
import composit.nn


def convert_parameters_to_numpy(model, data_format):
    parameters = {}
    for name, value in model.state_dict().items():
        new_value = value.detach().numpy()
        if "fc.weight" in name:
            new_value = new_value.transpose((1, 0))
        elif (("conv" in name and "weight" in name) or "downsample.0" in name) and data_format == "NHWC":
            new_value = new_value.transpose((0, 2, 3, 1))
        elif ("bn" in name or "downsample.1" in name) and data_format == "NCHW":
            new_value = new_value.reshape((-1, 1, 1))
        parameters[name] = new_value
    return parameters


def functional_batch_norm(input_tensor, mean, var, weight, bias, *, epsilon=1e-5):
    output = input_tensor - mean
    output = output / cnp.sqrt(var + epsilon)
    output *= weight
    output += bias
    return output


def functional_resnet_module(
    input_tensor,
    right_branch_conv_0_weight,
    right_branch_bn_0_running_mean,
    right_branch_bn_0_running_var,
    right_branch_bn_0_weight,
    right_branch_bn_0_bias,
    right_branch_conv_1_weight,
    right_branch_bn_1_running_mean,
    right_branch_bn_1_running_var,
    right_branch_bn_1_weight,
    right_branch_bn_1_bias,
    right_branch_conv_2_weight,
    right_branch_bn_2_running_mean,
    right_branch_bn_2_running_var,
    right_branch_bn_2_weight,
    right_branch_bn_2_bias,
    left_branch_conv_weight,
    left_branch_bn_running_mean,
    left_branch_bn_running_var,
    left_branch_bn_weight,
    left_branch_bn_bias,
    data_format,
    module_strides=(1, 1),
):
    left_branch = input_tensor
    right_branch = input_tensor

    if left_branch_conv_weight is not None:
        left_branch = cnp.nn.convolution(
            left_branch, left_branch_conv_weight, strides=module_strides, data_format=data_format
        )
        left_branch = functional_batch_norm(
            left_branch,
            left_branch_bn_running_mean,
            left_branch_bn_running_var,
            left_branch_bn_weight,
            left_branch_bn_bias,
        )

    right_branch = cnp.nn.convolution(right_branch, right_branch_conv_0_weight, strides=(1, 1), data_format=data_format)
    right_branch = functional_batch_norm(
        right_branch,
        right_branch_bn_0_running_mean,
        right_branch_bn_0_running_var,
        right_branch_bn_0_weight,
        right_branch_bn_0_bias,
    )
    right_branch = cnp.nn.relu(right_branch)

    right_branch = cnp.nn.convolution(
        right_branch, right_branch_conv_1_weight, strides=module_strides, padding=(1, 1), data_format=data_format
    )
    right_branch = functional_batch_norm(
        right_branch,
        right_branch_bn_1_running_mean,
        right_branch_bn_1_running_var,
        right_branch_bn_1_weight,
        right_branch_bn_1_bias,
    )
    right_branch = cnp.nn.relu(right_branch)

    right_branch = cnp.nn.convolution(right_branch, right_branch_conv_2_weight, strides=(1, 1), data_format=data_format)
    right_branch = functional_batch_norm(
        right_branch,
        right_branch_bn_2_running_mean,
        right_branch_bn_2_running_var,
        right_branch_bn_2_weight,
        right_branch_bn_2_bias,
    )

    output = cnp.nn.relu(left_branch + right_branch)
    return output


def functional_resnet(
    image,
    parameters,
    *,
    data_format,
):
    if data_format == "NHWC":
        image = cnp.transpose(image, (0, 2, 3, 1))

    output = cnp.nn.convolution(
        image, parameters["conv1.weight"], strides=(2, 2), padding=(3, 3), data_format=data_format
    )
    output = functional_batch_norm(
        output,
        parameters["bn1.running_mean"],
        parameters["bn1.running_var"],
        parameters["bn1.weight"],
        parameters["bn1.bias"],
    )
    output = cnp.nn.relu(output)
    output = cnp.nn.max_pool(output, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), data_format=data_format)

    for layer_index, num_layers in enumerate((3, 4, 6, 3)):
        for index in range(num_layers):
            layer_name = f"layer{layer_index + 1}.{index}"
            output = functional_resnet_module(
                output,
                parameters[f"{layer_name}.conv1.weight"],
                parameters[f"{layer_name}.bn1.running_mean"],
                parameters[f"{layer_name}.bn1.running_var"],
                parameters[f"{layer_name}.bn1.weight"],
                parameters[f"{layer_name}.bn1.bias"],
                parameters[f"{layer_name}.conv2.weight"],
                parameters[f"{layer_name}.bn2.running_mean"],
                parameters[f"{layer_name}.bn2.running_var"],
                parameters[f"{layer_name}.bn2.weight"],
                parameters[f"{layer_name}.bn2.bias"],
                parameters[f"{layer_name}.conv3.weight"],
                parameters[f"{layer_name}.bn3.running_mean"],
                parameters[f"{layer_name}.bn3.running_var"],
                parameters[f"{layer_name}.bn3.weight"],
                parameters[f"{layer_name}.bn3.bias"],
                parameters.get(f"{layer_name}.downsample.0.weight"),
                parameters.get(f"{layer_name}.downsample.1.running_mean"),
                parameters.get(f"{layer_name}.downsample.1.running_var"),
                parameters.get(f"{layer_name}.downsample.1.weight"),
                parameters.get(f"{layer_name}.downsample.1.bias"),
                data_format=data_format,
                module_strides=(2, 2) if layer_index > 0 and index == 0 else (1, 1),
            )

    if data_format == "NHWC":
        output = cnp.transpose(output, (0, 3, 1, 2))

    output = cnp.mean(output, axis=(2, 3))

    output = output @ parameters["fc.weight"]
    output = output + parameters["fc.bias"]

    return output
