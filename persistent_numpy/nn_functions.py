import inspect
import math

import numpy as np
from pyrsistent import immutable, PClass, field

from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal
from persistent_numpy.ndarray import PersistentArray, Node

from persistent_numpy.numpy_functions import _create_from_numpy_compute_instruction


class _variable(PClass):
    requires_gradient = field(type=bool)


def variable(*, name: str, shape: tuple, requires_gradient: bool = False):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=_variable(requires_gradient=requires_gradient), shape=shape)
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
    return _create_from_numpy_compute_instruction(*operands, instruction=instruction)


def gelu(operand):
    def compute(self, input_tensor):
        return 0.5 * input_tensor * (1 + np.vectorize(math.erf)(input_tensor / np.sqrt(2)))

    function_name = inspect.currentframe().f_code.co_name
    klass = immutable(name=function_name)
    klass.__call__ = compute
    instruction = klass()
    return _create_from_numpy_compute_instruction(operand, instruction=instruction)


def _operands(graph, node):
    result = ((data["sink_input_port"], predecessor) for predecessor, _, data in graph.in_edges(node, data=True))
    return [element[1] for element in sorted(result, key=lambda element: element[0])]


def model_variables(model):
    graph = model.graph
    variables = set()
    for node in graph:
        instruction = graph.get_node_attribute(node, "instruction")
        if isinstance(instruction, _variable):
            variables.add(node.name)
    return variables


def forward(model, **variable_to_array):

    variables = model_variables(model)
    for variable in variables:
        assert variable in variable_to_array

    graph = model.graph
    cache = variable_to_array.copy()
    for node in topological_traversal(graph):
        if node.name in cache:
            continue
        instruction = graph.get_node_attribute(node, "instruction")
        input_arrays = [cache[operand.name] for operand in _operands(graph, node)]
        cache[node.name] = instruction(*input_arrays)
    return cache[model.node.name]
