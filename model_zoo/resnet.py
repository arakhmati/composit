import composit as cnp
import composit.nn


def convert_parameters_to_numpy(model):
    parameters = {}
    for name, value in model.named_parameters():
        new_value = value.detach().numpy()
        parameters[name] = new_value
    return parameters


def functional_resnet(
    image,
    parameters,
    *,
    data_format,
):
    image = cnp.transpose(image, (0, 2, 3, 1))
    conv1_weight = cnp.transpose(parameters["conv1.weight"], (0, 2, 3, 1))

    output = cnp.nn.convolution(image, conv1_weight, strides=(2, 2), padding=(3, 3), data_format=data_format)
    output = cnp.nn.relu(output)

    return cnp.transpose(output, (0, 3, 1, 2))
