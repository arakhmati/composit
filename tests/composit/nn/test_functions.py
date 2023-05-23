import numpy as np
import torch.nn.functional

import composit as cnp
import composit.nn


def test_embedding():
    input_tensor = cnp.zeros((5, 25), dtype=np.int32)
    weights = cnp.random.random((25, 10))
    result = cnp.nn.embedding(input_tensor, weights)

    torch_input_tensor = torch.from_numpy(cnp.evaluate(input_tensor))
    torch_weights = torch.from_numpy(cnp.evaluate(weights))
    torch_result = torch.nn.functional.embedding(torch_input_tensor, torch_weights).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_gelu():
    array = cnp.random.random((5, 25, 15))
    result = cnp.nn.gelu(array)

    torch_array = torch.from_numpy(cnp.evaluate(array))
    torch_result = torch.nn.functional.gelu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_convolution_channels_first():
    image = cnp.random.random((1, 3, 28, 28))
    filters = cnp.random.random((32, 3, 5, 5))
    result = cnp.nn.convolution(image, filters, data_format="NCHW")

    torch_image = torch.from_numpy(cnp.evaluate(image))
    torch_filters = torch.from_numpy(cnp.evaluate(filters))
    torch_result = torch.nn.functional.conv2d(torch_image, torch_filters).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_convolution_channels_last():
    image = cnp.random.random((1, 28, 28, 3))
    filters = cnp.random.random((32, 5, 5, 3))
    result = cnp.nn.convolution(image, filters, data_format="NHWC")

    torch_image = torch.from_numpy(cnp.evaluate(image).transpose((0, 3, 1, 2)))
    torch_filters = torch.from_numpy(cnp.evaluate(filters).transpose((0, 3, 1, 2)))
    torch_result = torch.nn.functional.conv2d(torch_image, torch_filters).numpy().transpose((0, 2, 3, 1))

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_average_pooling():
    image = cnp.random.random((1, 3, 28, 28))
    result = cnp.nn.average_pooling(image, kernel_size=(2, 2))

    torch_image = torch.from_numpy(cnp.evaluate(image))
    torch_result = torch.nn.functional.avg_pool2d(torch_image, kernel_size=[2, 2], stride=[1, 1]).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_max_pooling():
    image = cnp.random.random((1, 3, 28, 28))
    result = cnp.nn.max_pooling(image, kernel_size=(2, 2))

    torch_image = torch.from_numpy(cnp.evaluate(image))
    torch_result = torch.nn.functional.max_pool2d(torch_image, kernel_size=[2, 2], stride=[1, 1]).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)
