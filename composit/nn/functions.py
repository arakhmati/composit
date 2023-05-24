import numpy as np


from composit.multidigraph import MultiDiGraph
from composit.nn.core import Variable, wrap_as_instruction
from composit.nn import vectorized_functions
from composit.persistent_array import PersistentArray, Node


def variable(*, name: str, shape: tuple, dtype=None) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=Variable(), shapes=(shape,), dtypes=(np.dtype(dtype),))
    return PersistentArray(graph=graph, node=node)


@wrap_as_instruction()
def embedding(input_tensor, weights):
    return vectorized_functions.embedding(input_tensor, weights)


@wrap_as_instruction()
def gelu(input_tensor):
    return vectorized_functions.gelu(input_tensor)


def convolution_channels_first(image, filters, strides, padding):
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

    output_height = (height - kernel_height) // strides_height + 1
    output_width = (width - kernel_width) // strides_width + 1

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


def convolution_channels_last(image, filters, strides, padding):
    padding_height, padding_width = padding
    image = np.pad(
        image,
        ((0, 0), (padding_height, padding_height), (padding_width, padding_width), (0, 0)),
        mode="constant",
        constant_values=0,
    )

    batch_size, height, width, _ = image.shape
    num_output_channels, kernel_height, kernel_width, num_input_channels = filters.shape

    strides_height, strides_width = strides

    output_height = (height - kernel_height) // strides_height + 1
    output_width = (width - kernel_width) // strides_width + 1

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


@wrap_as_instruction()
def convolution(image, filters, *, data_format, strides=(1, 1), padding=(0, 0)):
    data_format_to_function = {
        "NCHW": convolution_channels_first,
        "NHWC": convolution_channels_last,
    }
    function = data_format_to_function[data_format]
    return function(image, filters, strides, padding)


@wrap_as_instruction()
def average_pooling(image, *, kernel_size):
    # TODO: Make average_pooling generic for all dimensions?

    batch_size, num_channels, height, width = image.shape
    kernel_height, kernel_width = kernel_size

    output_height = (height - kernel_size[0]) + 1
    output_width = (width - kernel_size[1]) + 1

    output = np.zeros((batch_size, num_channels, output_height, output_width), dtype=image.dtype)
    for batch_index in range(batch_size):
        for channel_index in range(num_channels):
            for output_height_index in range(output_height):
                for output_width_index in range(output_width):
                    output[batch_index, channel_index, output_height_index, output_width_index] = np.mean(
                        image[
                            batch_index,
                            channel_index,
                            output_height_index : output_height_index + kernel_height,
                            output_width_index : output_width_index + kernel_width,
                        ]
                    )
    return output


@wrap_as_instruction()
def max_pooling(image, *, kernel_size):
    # TODO: Make max_pooling generic for all dimensions?

    batch_size, num_channels, height, width = image.shape
    kernel_height, kernel_width = kernel_size

    output_height = (height - kernel_size[0]) + 1
    output_width = (width - kernel_size[1]) + 1

    output = np.zeros((batch_size, num_channels, output_height, output_width), dtype=image.dtype)
    for batch_index in range(batch_size):
        for channel_index in range(num_channels):
            for output_height_index in range(output_height):
                for output_width_index in range(output_width):
                    output[batch_index, channel_index, output_height_index, output_width_index] = np.max(
                        image[
                            batch_index,
                            channel_index,
                            output_height_index : output_height_index + kernel_height,
                            output_width_index : output_width_index + kernel_width,
                        ]
                    )
    return output


__all__ = [
    "variable",
    "embedding",
    "gelu",
    "convolution",
    "average_pooling",
    "max_pooling",
]
