import math

import numpy as np

from toolz import first

from composit.multidigraph import topological_traversal
from composit.numpy.core import get_operands

from mosaic.backends.x86.types import ModelWithoutKernelFusion, ModelWithKernelFusion


def initialize_variable_buffers(buffer_graph, inputs, buffer_descriptor_to_buffer):
    for input_var, array in inputs.items():
        input_node = input_var.node
        buffer = buffer_descriptor_to_buffer[first(buffer_graph.nodes[input_node]["buffer_descriptors"])]
        buffer.array[:] = array.flatten()


def evaluate_mosaic_model_without_kernel_fusion(model: ModelWithoutKernelFusion):
    nodes_to_evaluate = filter(
        lambda node: model.buffer_graph.in_degree(node) > 0, topological_traversal(model.buffer_graph)
    )

    for node in nodes_to_evaluate:
        input_buffers = [
            model.buffer_descriptor_to_buffer[first(model.buffer_graph.nodes[input_node]["buffer_descriptors"])]
            for input_node, _ in get_operands(model.buffer_graph, node)
        ]
        output_buffer = model.buffer_descriptor_to_buffer[first(model.buffer_graph.nodes[node]["buffer_descriptors"])]

        input_pointers = [input_buffer.data() for input_buffer in input_buffers]
        output_pointer = output_buffer.data()

        run_kernel = model.node_to_run_kernel[node]
        run_kernel(*input_pointers, output_pointer)


def evaluate_mosaic_model_with_kernel_fusion(model: ModelWithKernelFusion):
    buffers = [model.buffer_descriptor_to_buffer[key] for key in sorted(model.buffer_descriptor_to_buffer)]
    pointers = [buffer.data() for buffer in buffers]
    model.run_model(*pointers)


def evaluate_mosaic_model(model: ModelWithoutKernelFusion | ModelWithKernelFusion, output_var, inputs):
    initialize_variable_buffers(model.buffer_graph, inputs, model.buffer_descriptor_to_buffer)

    if isinstance(model, ModelWithKernelFusion):
        evaluate_mosaic_model_with_kernel_fusion(model)
    else:
        evaluate_mosaic_model_without_kernel_fusion(model)

    output_node = first(model.buffer_graph.successors(output_var.node))
    buffer_descriptor = first(model.buffer_graph.nodes[output_node]["buffer_descriptors"])
    buffer = model.buffer_descriptor_to_buffer[buffer_descriptor]
    shape = first(model.buffer_graph.nodes[output_node]["shapes"])
    dtype = first(model.buffer_graph.nodes[output_node]["dtypes"])
    return np.frombuffer(memoryview(buffer.array), count=math.prod(shape), dtype=dtype).reshape(shape)
