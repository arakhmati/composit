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


def test_trace():
    torch_model = torch.hub.load("pytorch/vision:v0.10.0", "resnet18", pretrained=True).eval()

    image = load_image("https://github.com/pytorch/hub/raw/master/images/dog.jpg")
    image = torch.from_numpy(image)

    with flashlight.tracer.trace():
        flashlight_output = torch_model(image)

    assert len(flashlight_output.graph) == 414
    composit_output = cnp.evaluate(flashlight_output.lazy_tensor)
    assert np.allclose(composit_output, flashlight_output.detach().numpy(), atol=1e-3)
