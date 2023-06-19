import pytest

from io import BytesIO
import requests

import numpy as np
from PIL import Image
import torch
from torchvision import transforms

import composit as cnp
import flashlight


def load_image(url):
    response = requests.get(url)
    image = Image.open(BytesIO(response.content))
    preprocess = transforms.Compose(
        [
            transforms.Resize(64),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    input_tensor = preprocess(image)
    return input_tensor.unsqueeze(0).numpy()


def test_resnet(run_torch=True):
    torch_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

    image = load_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
    image = torch.from_numpy(image)

    torch_output = torch_model(image)

    with flashlight.tracer.trace(run_torch=run_torch):
        flashlight_output = torch_model(image)

    if isinstance(flashlight_output, flashlight.Tensor):
        return flashlight_output.lazy_tensor, torch_output.detach().numpy()
    return flashlight_output, None


@pytest.mark.parametrize("run_torch", [True])
def test_trace(run_torch):
    output_var, torch_output = test_resnet(run_torch)
    assert len(output_var.graph) == 414
    composit_output = cnp.evaluate(output_var)
    if run_torch:
        assert np.allclose(composit_output, torch_output, atol=1e-3)


def test_graph_cache():
    output_var, _ = test_resnet()
    first_hash = hash(output_var)

    output_var, _ = test_resnet()
    assert first_hash == hash(output_var)
