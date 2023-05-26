import pytest

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


def test_relu():
    array = cnp.random.random((5, 25, 15))
    result = cnp.nn.relu(array)

    torch_array = torch.from_numpy(cnp.evaluate(array))
    torch_result = torch.nn.functional.relu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_sigmoid():
    array = cnp.random.random((5, 25, 15))
    result = cnp.nn.sigmoid(array)

    torch_array = torch.from_numpy(cnp.evaluate(array))
    torch_result = torch.nn.functional.sigmoid(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


def test_silu():
    array = cnp.random.random((5, 25, 15))
    result = cnp.nn.silu(array)

    torch_array = torch.from_numpy(cnp.evaluate(array))
    torch_result = torch.nn.functional.silu(torch_array).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(cnp.evaluate(result), torch_result)


@pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
@pytest.mark.parametrize("padding", [(0, 0), (3, 3)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_convolution(strides, padding, channels_last):
    np_image = np.random.random((1, 3, 28, 38))
    np_filters = np.random.random((32, 3, 5, 5))

    image_var = image_arg = cnp.nn.variable(name="image", shape=np_image.shape, dtype=np_image.dtype)
    filters_var = filters_arg = cnp.nn.variable(name="filters", shape=np_filters.shape, dtype=np_filters.dtype)

    if channels_last:
        image_arg = cnp.transpose(image_var, (0, 2, 3, 1))
        filters_arg = cnp.transpose(filters_var, (0, 2, 3, 1))

    result_var = cnp.nn.convolution(
        image_arg, filters_arg, strides=strides, padding=padding, channels_last=channels_last
    )

    if channels_last:
        result_var = cnp.transpose(result_var, (0, 3, 1, 2))

    result = cnp.nn.evaluate(result_var, inputs={image_var: np_image, filters_var: np_filters})

    torch_result = torch.nn.functional.conv2d(
        torch.from_numpy(np_image), torch.from_numpy(np_filters), stride=strides, padding=padding
    ).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(result, torch_result)


@pytest.mark.parametrize(
    "composit_and_torch_functions",
    [(cnp.nn.average_pool, torch.nn.functional.avg_pool2d), (cnp.nn.max_pool, torch.nn.functional.max_pool2d)],
)
@pytest.mark.parametrize("kernel_size", [(2, 2), (3, 3)])
@pytest.mark.parametrize("strides", [(1, 1), (2, 2)])
@pytest.mark.parametrize("padding", [(0, 0), (1, 1)])
@pytest.mark.parametrize("channels_last", [False, True])
def test_pool(composit_and_torch_functions, kernel_size, strides, padding, channels_last):
    composit_function, torch_function = composit_and_torch_functions

    np_image = np.random.random((1, 3, 28, 38))

    image_var = image_arg = cnp.nn.variable(name="image", shape=np_image.shape, dtype=np_image.dtype)

    if channels_last:
        image_arg = cnp.transpose(image_var, (0, 2, 3, 1))

    result_var = composit_function(
        image_arg, kernel_size=kernel_size, strides=strides, padding=padding, channels_last=channels_last
    )

    if channels_last:
        result_var = cnp.transpose(result_var, (0, 3, 1, 2))

    result = cnp.nn.evaluate(result_var, inputs={image_var: np_image})

    torch_result = torch_function(
        torch.from_numpy(np_image), kernel_size=kernel_size, stride=strides, padding=padding
    ).numpy()

    assert result.shape == torch_result.shape
    assert np.allclose(result, torch_result)
