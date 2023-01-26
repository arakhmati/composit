import numba
import numpy as np
from pyrsistent import immutable, PClass

from persistent_numpy.nn.vectorized_functions import cdf
from persistent_numpy.multidigraph import MultiDiGraph
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.numpy import create_from_numpy_compute_instruction


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


__all__ = [
    "Variable",
    "variable",
    # Compute functions
    "wrap_as_instruction",
    "embedding",
    "gelu",
]
