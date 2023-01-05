import inspect
import math

import numpy as np
from pyrsistent import immutable, PClass, field

from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal, compose_all
from persistent_numpy.persistent_array import PersistentArray, Node

from persistent_numpy.numpy_functions import create_from_numpy_compute_instruction, node_operands


class _variable(PClass):
    requires_gradient = field(type=bool)


def variable(*, name: str, shape: tuple, requires_gradient: bool = False):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=_variable(requires_gradient=requires_gradient), shapes=[shape])
    return PersistentArray(graph=graph, node=node)


def embedding(*operands):
    def compute(self, input_tensor, weights):
        batch_size, sequence_size = input_tensor.shape
        result = np.zeros((batch_size, sequence_size, weights.shape[1]))
        for batch_index in range(batch_size):
            for sequence_index in range(sequence_size):
                result[batch_index, sequence_index] = weights[input_tensor[batch_index, sequence_index]]
        return result

    function_name = inspect.currentframe().f_code.co_name
    klass = immutable(name=function_name)
    klass.__call__ = compute
    instruction = klass()
    return create_from_numpy_compute_instruction(*operands, instruction=instruction)


def gelu(operand):
    def compute(self, input_tensor):
        return 0.5 * input_tensor * (1 + np.vectorize(math.erf)(input_tensor / np.sqrt(2)))

    function_name = inspect.currentframe().f_code.co_name
    klass = immutable(name=function_name)
    klass.__call__ = compute
    instruction = klass()
    return create_from_numpy_compute_instruction(operand, instruction=instruction)


def initial_cache(graph, **variable_to_array):
    cache = {}
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, _variable):
            cache[(node, 0)] = variable_to_array[node.name]
    return cache


def evaluate(*models, **variable_to_array):

    graph = compose_all(*tuple(model.graph for model in models))

    cache = initial_cache(graph, **variable_to_array)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in node_operands(graph, node)]
        instruction_output = instruction(*input_arrays)
        if isinstance(instruction_output, np.ndarray):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, list):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError("Unsupported type")

    result = [cache[(model.node, model.output_index)] for model in models]
    if len(result) == 1:
        return result[0]
    return result
