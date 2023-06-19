import math

import numpy as np
from pyrsistent import pmap
from toolz import first

from composit.numpy.core import Input
from composit.multidigraph import topological_traversal
from composit.numpy.core import get_operands

from mosaic.backends.x86.types import ModelWithoutKernelFusion, ModelWithKernelFusion
from mosaic.backends.x86.passes.compile_to_mosaic_model import compile_to_mosaic_model


def initialize_input_buffers(graph, buffer_descriptor_to_buffer):
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        if not isinstance(instruction, Input):
            continue

        array = instruction()
        buffer = buffer_descriptor_to_buffer[first(graph.nodes[node]["buffer_descriptors"])]
        buffer.array[:] = array.flatten()


def evaluate_without_kernel_fusion(model: ModelWithoutKernelFusion):
    nodes_to_evaluate = filter(lambda node: model.graph.in_degree(node) > 0, topological_traversal(model.graph))

    for node in nodes_to_evaluate:
        input_buffers = [
            model.buffer_descriptor_to_buffer[first(model.graph.nodes[input_node]["buffer_descriptors"])]
            for input_node, _ in get_operands(model.graph, node)
        ]
        output_buffer = model.buffer_descriptor_to_buffer[first(model.graph.nodes[node]["buffer_descriptors"])]

        input_pointers = [input_buffer.data() for input_buffer in input_buffers]
        output_pointer = output_buffer.data()

        run_kernel = model.node_to_run_kernel[node]
        run_kernel(*input_pointers, output_pointer)


def evaluate_with_kernel_fusion(model: ModelWithKernelFusion):
    buffers = [model.buffer_descriptor_to_buffer[key] for key in sorted(model.buffer_descriptor_to_buffer)]
    pointers = [buffer.data() for buffer in buffers]
    model.run_model(*pointers)


def evaluate(*output_vars, input_var_to_scheme, **kwargs):
    model: ModelWithoutKernelFusion | ModelWithKernelFusion
    input_var_to_scheme = pmap(input_var_to_scheme)
    model = compile_to_mosaic_model(*output_vars, input_var_to_scheme=input_var_to_scheme, **kwargs)

    initialize_input_buffers(model.graph, model.buffer_descriptor_to_buffer)

    if isinstance(model, ModelWithKernelFusion):
        evaluate_with_kernel_fusion(model)
    else:
        evaluate_without_kernel_fusion(model)

    result = []
    for output_var in output_vars:
        output_node = first(model.graph.successors(output_var.node))
        buffer_descriptor = first(model.graph.nodes[output_node]["buffer_descriptors"])
        buffer = model.buffer_descriptor_to_buffer[buffer_descriptor]
        shape = first(model.graph.nodes[output_node]["shapes"])
        dtype = first(model.graph.nodes[output_node]["dtypes"])
        output_array = np.frombuffer(memoryview(buffer.array), count=math.prod(shape), dtype=dtype).reshape(shape)
        result.append(output_array)

    if len(result) == 1:
        (result,) = result
    return result
