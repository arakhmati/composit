from io import BytesIO
import requests

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


import composit as cnp
from model_zoo.resnet import (
    functional_resnet,
    convert_parameters_to_numpy,
)


def load_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0).numpy()


def evaluate_torch_model(model, image):
    output = model.conv1(image)
    output = model.relu(output)
    return output.detach().numpy()


def test_functional_resnet_vs_torch_resnet(data_format="NHWC"):
    torch_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True)

    image = load_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")

    torch_image = torch.from_numpy(image)
    torch_output = evaluate_torch_model(torch_model, torch_image)

    parameters = {
        cnp.nn.variable(name=name, shape=value.shape): value
        for name, value in convert_parameters_to_numpy(torch_model).items()
    }

    image_var = cnp.nn.variable(name="image_channels_last", shape=image.shape, dtype=image.dtype)

    model = functional_resnet(image_var, {var.node.name: var for var in parameters.keys()}, data_format=data_format)

    output = cnp.nn.evaluate(
        model,
        inputs={
            image_var: image,
            **parameters,
        },
    )

    assert np.allclose(output, torch_output, atol=1e-5)
