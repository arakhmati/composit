import pytest

from io import BytesIO
import requests

import numpy as np
from PIL import Image
import torch
from torchvision import transforms


import composit as cnp
from model_zoo.resnet import (
    resnet,
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


@pytest.mark.parametrize("channels_last", [True])
def test_torch_vs_composit(channels_last):
    torch_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet50", pretrained=True).eval()

    image = load_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")

    torch_image = torch.from_numpy(image)
    torch_output = torch_model(torch_image).detach().numpy()

    parameters = {
        name: cnp.asarray(value, name)
        for name, value in convert_parameters_to_numpy(torch_model, channels_last).items()
    }

    image_var = cnp.asarray(image, name="image")
    model = resnet(image_var, parameters, channels_last=channels_last)

    output = cnp.nn.evaluate(model)

    assert np.allclose(output, torch_output, atol=1e-4)
