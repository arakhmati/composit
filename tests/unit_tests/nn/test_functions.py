import numpy as np
import torch.nn.functional

import persistent_numpy as pnp
import persistent_numpy.nn


def test_embedding():
    input_tensor = pnp.zeros((5, 25), dtype=np.int32)
    weights = pnp.random.random((25, 10))
    result = pnp.nn.embedding(input_tensor, weights)

    torch_input_tensor = torch.from_numpy(pnp.evaluate(input_tensor))
    torch_weights = torch.from_numpy(pnp.evaluate(weights))
    torch_result = torch.nn.functional.embedding(torch_input_tensor, torch_weights).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)


def test_gelu():
    array = pnp.random.random((5, 25, 15))
    result = pnp.nn.gelu(array)

    torch_array = torch.from_numpy(pnp.evaluate(array))
    torch_result = torch.nn.functional.gelu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)


def test_convolution():
    image = pnp.random.random((1, 3, 28, 28))
    filters = pnp.random.random((32, 3, 3, 3))
    result = pnp.nn.convolution(image, filters)

    torch_image = torch.from_numpy(pnp.evaluate(image))
    torch_filters = torch.from_numpy(pnp.evaluate(filters))
    torch_result = torch.nn.functional.conv2d(torch_image, torch_filters).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)


def test_average_pooling():
    image = pnp.random.random((1, 3, 28, 28))
    result = pnp.nn.average_pooling(image, kernel_size=(2, 2))

    torch_image = torch.from_numpy(pnp.evaluate(image))
    torch_result = torch.nn.functional.avg_pool2d(torch_image, kernel_size=[2, 2], stride=[1, 1]).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)


def test_max_pooling():
    image = pnp.random.random((1, 3, 28, 28))
    result = pnp.nn.max_pooling(image, kernel_size=(2, 2))

    torch_image = torch.from_numpy(pnp.evaluate(image))
    torch_result = torch.nn.functional.max_pool2d(torch_image, kernel_size=[2, 2], stride=[1, 1]).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(pnp.evaluate(result), torch_result)
