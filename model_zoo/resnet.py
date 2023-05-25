import composit as cnp
import composit.nn
from composit.nn.layers import batch_norm, resnet_module


def convert_parameters_to_numpy(model, data_layout):
    parameters = {}
    for name, value in model.state_dict().items():
        new_value = value.detach().numpy()
        if "fc.weight" in name:
            new_value = new_value.transpose((1, 0))
        elif (("conv" in name and "weight" in name) or "downsample.0" in name) and data_layout == "NHWC":
            new_value = new_value.transpose((0, 2, 3, 1))
        parameters[name] = new_value
    return parameters


def resnet(
    image,
    parameters,
    *,
    data_layout,
):
    if data_layout == "NHWC":
        image = cnp.transpose(image, (0, 2, 3, 1))

    output = cnp.nn.convolution(
        image, parameters["conv1.weight"], strides=(2, 2), padding=(3, 3), data_layout=data_layout
    )
    output = batch_norm(
        output,
        parameters["bn1.running_mean"],
        parameters["bn1.running_var"],
        parameters["bn1.weight"],
        parameters["bn1.bias"],
        data_layout=data_layout,
    )
    output = cnp.nn.relu(output)
    output = cnp.nn.max_pool(output, kernel_size=(3, 3), strides=(2, 2), padding=(1, 1), data_layout=data_layout)

    for layer_index, num_layers in enumerate((3, 4, 6, 3)):
        for index in range(num_layers):
            layer_name = f"layer{layer_index + 1}.{index}"
            output = resnet_module(
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
                data_layout=data_layout,
                module_strides=(2, 2) if layer_index > 0 and index == 0 else (1, 1),
            )

    if data_layout == "NHWC":
        output = cnp.transpose(output, (0, 3, 1, 2))

    output = cnp.mean(output, axis=(2, 3))

    output = output @ parameters["fc.weight"]
    output = output + parameters["fc.bias"]

    return output
