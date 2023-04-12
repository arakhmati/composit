import numpy as np


from composit.multidigraph import MultiDiGraph
from composit.nn.core import Variable, wrap_as_instruction
from composit.nn.vectorized_functions import cdf
from composit.persistent_array import PersistentArray, Node


def variable(*, name: str, shape: tuple, dtype=None) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=Variable(), shapes=(shape,), dtypes=(np.dtype(dtype),))
    return PersistentArray(graph=graph, node=node)


@wrap_as_instruction()
def embedding(input_tensor, weights):
    batch_size, sequence_size = input_tensor.shape
    result = np.zeros((batch_size, sequence_size, weights.shape[1]), dtype=weights.dtype)
    for batch_index in range(batch_size):
        for sequence_index in range(sequence_size):
            result[batch_index, sequence_index] = weights[input_tensor[batch_index, sequence_index]]
    return result


@wrap_as_instruction()
def gelu(input_tensor):
    return input_tensor * cdf(input_tensor)


@wrap_as_instruction()
def convolution(image, filters):
    # TODO: Make convolution generic for all dimensions?

    batch_size, _, height, width = image.shape
    num_output_channels, num_input_channels, kernel_height, kernel_width = filters.shape

    output_height = (height - kernel_height) + 1
    output_width = (width - kernel_width) + 1

    output = np.zeros((batch_size, num_output_channels, output_height, output_width))
    for batch_index in range(batch_size):
        for output_channel_index in range(num_output_channels):
            for input_channel_index in range(num_input_channels):
                for output_height_index in range(output_height):
                    for output_width_index in range(output_width):
                        output[batch_index, output_channel_index, output_height_index, output_width_index] += (
                            image[
                                batch_index,
                                input_channel_index,
                                output_height_index : output_height_index + kernel_height,
                                output_width_index : output_width_index + kernel_width,
                            ].flatten()
                            @ filters[output_channel_index, input_channel_index].flatten()
                        )
    return output


@wrap_as_instruction()
def average_pooling(image, *, kernel_size):
    # TODO: Make average_pooling generic for all dimensions?

    batch_size, num_channels, height, width = image.shape
    kernel_height, kernel_width = kernel_size

    output_height = (height - kernel_size[0]) + 1
    output_width = (width - kernel_size[1]) + 1

    output = np.zeros((batch_size, num_channels, output_height, output_width))
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

    output = np.zeros((batch_size, num_channels, output_height, output_width))
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
