import inspect
import math

import numpy as np
from pyrsistent import immutable, PClass, pmap_field, pmap

from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal, compose_all
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.numpy import create_from_numpy_compute_instruction, node_operands


class Variable(PClass):
    ...


def variable(*, name: str, shape: tuple) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=Variable(), shapes=[shape])
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


class Cache(PClass):
    node_output_to_array = pmap_field(key_type=tuple, value_type=np.ndarray)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_array=pmap(dictionary))

    def __getitem__(self, persistent_array: PersistentArray):
        node = persistent_array.node
        output_index = persistent_array.output_index
        return self.node_output_to_array[(node, output_index)]


def initial_cache(graph, inputs):
    cache = {}
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, Variable):
            cache[(node, 0)] = inputs[node.name]
    return cache


def evaluate(*outputs, inputs: dict[Variable, np.ndarray], return_cache: bool = False):

    graph = compose_all(*tuple(output.graph for output in outputs))

    cache = initial_cache(graph, inputs)
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

    cache = Cache.from_dict(cache)

    result = [cache[output] for output in outputs]
    if len(result) == 1:
        (result,) = result

    if return_cache:
        return result, cache
    else:
        return result


__all__ = [
    "Variable",
    "variable",
    "embedding",
    "gelu",
    "Cache",
    "evaluate",
]
