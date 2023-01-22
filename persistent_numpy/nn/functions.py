import numba
import numpy as np
from pyrsistent import immutable, PClass, pmap_field, pmap

from persistent_numpy.nn.vectorized_functions import cdf
from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal, compose_all
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.numpy import create_from_numpy_compute_instruction, get_operands


class Variable(PClass):
    ...


def variable(*, name: str, shape: tuple) -> PersistentArray:
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=Variable(), shapes=(shape,))
    return PersistentArray(graph=graph, node=node)


def wrap_as_instruction(compute_function, *, use_njit=True):
    if use_njit:
        compute_function = numba.jit(
            compute_function, nopython=True, parallel=True, cache=True, error_model="numpy", fastmath=True
        )
    compute_function = staticmethod(compute_function)

    def wrapper(*operands):
        function_name = compute_function.__name__
        klass = immutable(name=function_name)
        klass.__call__ = staticmethod(compute_function)
        instruction = klass()
        return create_from_numpy_compute_instruction(*operands, instruction=instruction)

    return wrapper


@wrap_as_instruction
def embedding(input_tensor, weights):
    batch_size, sequence_size = input_tensor.shape
    result = np.zeros((batch_size, sequence_size, weights.shape[1]))
    for batch_index in range(batch_size):
        for sequence_index in range(sequence_size):
            result[batch_index, sequence_index] = weights[input_tensor[batch_index, sequence_index]]
    return result


@wrap_as_instruction
def gelu(input_tensor):
    return input_tensor * cdf(input_tensor)


class Cache(PClass):
    node_output_to_array = pmap_field(key_type=tuple, value_type=np.ndarray)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_array=pmap(dictionary))

    def __getitem__(self, persistent_array: PersistentArray):
        node = persistent_array.node
        output_index = persistent_array.output_index
        return self.node_output_to_array[(node, output_index)]

    def as_dict(self):
        return {node.name: array for (node, _), array in self.node_output_to_array.items()}


def initialize_cache(graph, inputs):
    cache = {}
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, Variable):
            cache[(node, 0)] = inputs[node.name]
    return cache


def evaluate(
    *output_vars,
    inputs: dict[Variable, np.ndarray],
    initialize_cache_function=initialize_cache,
    return_cache: bool = False,
    always_return_tuple: bool = False,
):

    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    cache = initialize_cache_function(graph, inputs)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]
        instruction_output = instruction(*input_arrays)
        if isinstance(instruction_output, np.ndarray):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, list):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError("Unsupported type")

    cache = Cache.from_dict(cache)

    result = [cache[output_var] for output_var in output_vars]
    if len(result) == 1 and not always_return_tuple:
        (result,) = result

    if return_cache:
        return result, cache
    else:
        return result


__all__ = [
    "Variable",
    "variable",
    # Compute functions
    "wrap_as_instruction",
    "embedding",
    "gelu",
    # Numpy evaluation
    "Cache",
    "evaluate",
]
