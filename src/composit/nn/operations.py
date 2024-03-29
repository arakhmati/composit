import math

import numpy as np

import composit as cnp
from composit.nn import numpy_functions


@cnp.wrap_as_operation()
def embedding(input_tensor, weights):
    return numpy_functions.embedding(input_tensor, weights)


@cnp.wrap_as_operation()
def gelu(input_tensor):
    return numpy_functions.gelu(input_tensor)


@cnp.wrap_as_operation()
def relu(input_tensor):
    return np.maximum(input_tensor, 0)


@cnp.wrap_as_operation()
def sigmoid(input_tensor):
    return numpy_functions.sigmoid(input_tensor)


@cnp.wrap_as_operation()
def silu(input_tensor):
    return input_tensor * numpy_functions.sigmoid(input_tensor)


def convolution_output_dim(input_dim, kernel_dim, stride):
    return math.floor((input_dim - (kernel_dim - 1) - 1) / stride + 1)


def convolution_1d_channels_first(input_tensor, filters, strides, padding):
    if isinstance(strides, tuple):
        (strides,) = strides

    if isinstance(padding, tuple):
        (padding,) = padding

    input_tensor = np.pad(
        input_tensor,
        ((0, 0), (0, 0), (padding, padding)),
        mode="constant",
        constant_values=0,
    )

    batch_size, _, size = input_tensor.shape
    num_output_channels, num_input_channels, kernel_size = filters.shape

    output_size = convolution_output_dim(size, kernel_size, strides)

    output = np.zeros((batch_size, num_output_channels, output_size), dtype=input_tensor.dtype)
    for batch_index in range(batch_size):
        for output_channel_index in range(num_output_channels):
            for input_channel_index in range(num_input_channels):
                for output_size_index in range(output_size):
                    output[batch_index, output_channel_index, output_size_index] += (
                        input_tensor[
                            batch_index,
                            input_channel_index,
                            output_size_index * strides : output_size_index * strides + kernel_size,
                        ].flatten()
                        @ filters[output_channel_index, input_channel_index].flatten()
                    )
    return output


def convolution_1d_channels_last(input_tensor, filters, strides, padding):
    if isinstance(strides, tuple):
        (strides,) = strides

    if isinstance(padding, tuple):
        (padding,) = padding

    input_tensor = np.pad(
        input_tensor,
        ((0, 0), (padding, padding), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    batch_size, size, _ = input_tensor.shape
    num_output_channels, kernel_size, _ = filters.shape

    output_size = convolution_output_dim(size, kernel_size, strides)

    output = np.zeros((batch_size, output_size, num_output_channels), dtype=input_tensor.dtype)
    for batch_index in range(batch_size):
        for output_size_index in range(output_size):
            for output_channel_index in range(num_output_channels):
                output[batch_index, output_size_index, output_channel_index] += (
                    input_tensor[
                        batch_index,
                        output_size_index * strides : output_size_index * strides + kernel_size,
                    ].flatten()
                    @ filters[output_channel_index].flatten()
                )
    return output


def convolution_2d_channels_first(image, filters, strides, padding):
    padding_height, padding_width = padding
    image = np.pad(
        image,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
        mode="constant",
        constant_values=0,
    )

    batch_size, _, height, width = image.shape
    num_output_channels, num_input_channels, kernel_height, kernel_width = filters.shape

    strides_height, strides_width = strides

    output_height = convolution_output_dim(height, kernel_height, strides_height)
    output_width = convolution_output_dim(width, kernel_width, strides_width)

    output = np.zeros((batch_size, num_output_channels, output_height, output_width), dtype=image.dtype)
    for batch_index in range(batch_size):
        for output_channel_index in range(num_output_channels):
            for input_channel_index in range(num_input_channels):
                for output_height_index in range(output_height):
                    for output_width_index in range(output_width):
                        output[batch_index, output_channel_index, output_height_index, output_width_index] += (
                            image[
                                batch_index,
                                input_channel_index,
                                output_height_index * strides_height : output_height_index * strides_height
                                + kernel_height,
                                output_width_index * strides_width : output_width_index * strides_width + kernel_width,
                            ].flatten()
                            @ filters[output_channel_index, input_channel_index].flatten()
                        )
    return output


def convolution_2d_channels_last(image, filters, strides, padding):
    padding_height, padding_width = padding
    image = np.pad(
        image,
        ((0, 0), (padding_height, padding_height), (padding_width, padding_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    batch_size, height, width, _ = image.shape
    num_output_channels, kernel_height, kernel_width, _ = filters.shape

    strides_height, strides_width = strides

    output_height = convolution_output_dim(height, kernel_height, strides_height)
    output_width = convolution_output_dim(width, kernel_width, strides_width)

    filters = filters.reshape((num_output_channels, -1))

    output = np.zeros((batch_size, output_height, output_width, num_output_channels), dtype=image.dtype)
    for batch_index in range(batch_size):
        for output_height_index in range(output_height):
            for output_width_index in range(output_width):
                image_patch = image[
                    batch_index,
                    output_height_index * strides_height : output_height_index * strides_height + kernel_height,
                    output_width_index * strides_width : output_width_index * strides_width + kernel_width,
                ].flatten()
                for output_channel_index in range(num_output_channels):
                    output[
                        batch_index,
                        output_height_index,
                        output_width_index,
                        output_channel_index,
                    ] += (
                        image_patch @ filters[output_channel_index]
                    )

    return output


@cnp.wrap_as_operation()
def convolution(input_tensor, filters, *, channels_last, strides=None, padding=None):
    if len(filters.shape) == 3:
        if strides is None:
            strides = (1,)
        if padding is None:
            padding = (0,)

        data_format_to_function = {
            False: convolution_1d_channels_first,
            True: convolution_1d_channels_last,
        }
        function = data_format_to_function[channels_last]

    elif len(filters.shape) == 4:
        if strides is None:
            strides = (1, 1)
        if padding is None:
            padding = (0, 0)

        data_format_to_function = {
            False: convolution_2d_channels_first,
            True: convolution_2d_channels_last,
        }
        function = data_format_to_function[channels_last]

    else:
        raise RuntimeError("Invalid rank for filters of the convolution")
    return function(input_tensor, filters, strides, padding)


def pool_output_dim(input_dim, kernel_dim, stride, pool_function):
    if pool_function == np.max:
        return math.floor((input_dim - (kernel_dim - 1) - 1) / stride + 1)
    else:
        return math.floor((input_dim - kernel_dim) / stride + 1)


def pool_channels_first(image, *, kernel_size, strides, padding, pool_function):
    padding_height, padding_width = padding
    image = np.pad(
        image,
        ((0, 0), (0, 0), (padding_height, padding_height), (padding_width, padding_width)),
        mode="constant",
        constant_values=-np.inf if pool_function == np.max else 0,
    )

    batch_size, num_channels, height, width = image.shape
    kernel_height, kernel_width = kernel_size

    strides_height, strides_width = strides

    output_height = pool_output_dim(height, kernel_height, strides_height, pool_function)
    output_width = pool_output_dim(width, kernel_width, strides_width, pool_function)

    output = np.zeros((batch_size, num_channels, output_height, output_width), dtype=image.dtype)
    for batch_index in range(batch_size):
        for channel_index in range(num_channels):
            for output_height_index in range(output_height):
                for output_width_index in range(output_width):
                    output[batch_index, channel_index, output_height_index, output_width_index] = pool_function(
                        image[
                            batch_index,
                            channel_index,
                            output_height_index * strides_height : output_height_index * strides_height + kernel_height,
                            output_width_index * strides_width : output_width_index * strides_width + kernel_width,
                        ]
                    )
    return output


def pool_channels_last(image, *, kernel_size, strides, padding, pool_function):
    padding_height, padding_width = padding
    image = np.pad(
        image,
        ((0, 0), (padding_height, padding_height), (padding_width, padding_width), (0, 0)),
        mode="constant",
        constant_values=-np.inf if pool_function == np.max else 0,
    )

    batch_size, height, width, num_channels = image.shape
    kernel_height, kernel_width = kernel_size

    strides_height, strides_width = strides

    output_height = pool_output_dim(height, kernel_height, strides_height, pool_function)
    output_width = pool_output_dim(width, kernel_width, strides_width, pool_function)

    output = np.zeros((batch_size, output_height, output_width, num_channels), dtype=image.dtype)
    for batch_index in range(batch_size):
        for output_height_index in range(output_height):
            for output_width_index in range(output_width):
                for channel_index in range(num_channels):
                    output[batch_index, output_height_index, output_width_index, channel_index] = pool_function(
                        image[
                            batch_index,
                            output_height_index * strides_height : output_height_index * strides_height + kernel_height,
                            output_width_index * strides_width : output_width_index * strides_width + kernel_width,
                            channel_index,
                        ]
                    )
    return output


@cnp.wrap_as_operation()
def pool(image, *, pool_function, kernel_size, channels_last, strides=(1, 1), padding=(0, 0)):
    data_format_to_function = {
        False: pool_channels_first,
        True: pool_channels_last,
    }
    function = data_format_to_function[channels_last]
    return function(image, kernel_size=kernel_size, strides=strides, padding=padding, pool_function=pool_function)


def average_pool(*args, **kwargs):
    return pool(*args, **kwargs, pool_function=np.mean)


def max_pool(*args, **kwargs):
    return pool(*args, **kwargs, pool_function=np.max)


__all__ = [
    "embedding",
    "gelu",
    "relu",
    "sigmoid",
    "silu",
    "convolution",
    "average_pool",
    "max_pool",
]
