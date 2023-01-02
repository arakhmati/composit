import random

import numpy as np

from persistent_numpy.multidigraph import MultiDiGraph, Node
from persistent_numpy.ndarray import PersistentArray
from persistent_numpy import instructions


def _create_from_array(name: str, array: np.ndarray):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=instructions.Constant(array=array))
    return PersistentArray(graph, node)


def _create_from_instruction(*operands, instruction) -> "PersistentArray":

    operands = list(operands)
    for index, operand in enumerate(operands):
        if isinstance(operands[index], (int, float)):
            operands[index] = _create_from_array(f"Scalar({operands[index]})", np.asarray(operands[index]))

    args_string = ", ".join(operand.name for operand in operands)
    name = f"{type(instruction)}({args_string})"
    new_node = Node(name=name)

    graph = MultiDiGraph()
    for operand in operands:
        graph = graph.merge(operand.graph)

    graph = graph.add_node(new_node, instruction=instruction)

    for index, operand in enumerate(operands):
        graph = graph.add_edge(operand.node, new_node, source_output_port=0, sink_input_port=index)

    return PersistentArray(graph, new_node)


def get_from_indices(self, indices) -> "PersistentArray":
    if not isinstance(indices, PersistentArray):
        name = f"{indices}"
        indices = _create_from_array(name, np.asarray(indices))
    return _create_from_instruction(self, indices, instruction=instructions.GetFromIndices())


PersistentArray.get_from_indices = get_from_indices
PersistentArray.__getitem__ = get_from_indices


def set_at_indices(self, indices, values) -> "PersistentArray":
    return _create_from_instruction(self, values, instruction=instructions.SetAtIndices(indices=indices))


PersistentArray.set_at_indices = set_at_indices


def __add__(self, other: "PersistentArray") -> "PersistentArray":
    return _create_from_instruction(self, other, instruction=instructions.Add())


PersistentArray.__add__ = __add__


def __sub__(self, other: "PersistentArray") -> "PersistentArray":
    return _create_from_instruction(self, other, instruction=instructions.Subtract())


PersistentArray.__sub__ = __sub__


def __mul__(self, other: "PersistentArray") -> "PersistentArray":
    return _create_from_instruction(self, other, instruction=instructions.Multiply())


PersistentArray.__mul__ = __mul__


def __truediv__(self, other: "PersistentArray") -> "PersistentArray":
    return _create_from_instruction(self, other, instruction=instructions.Divide())


PersistentArray.__truediv__ = __truediv__


def __matmul__(self, other: "PersistentArray") -> "PersistentArray":
    return _create_from_instruction(self, other, instruction=instructions.MatrixMultiply())


PersistentArray.__matmul__ = __matmul__


def asarray(array, name=None):
    if name is None:
        name = "".join(random.choice("abcdef0123456789") for i in range(10))

    return _create_from_array(name, array)


def named_ndarray(*args, name, **kwargs):
    array = np.ndarray(*args, **kwargs)
    return _create_from_array(name, array)


def ndarray(*args, **kwargs):
    return named_ndarray(*args, name=f"ndarray({args}, {kwargs})", **kwargs)


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    return _create_from_array(f"zeros({args}, {kwargs})", array)


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    return _create_from_array(f"ones({args}, {kwargs})", array)


def matmul(operand_a, operand_b):
    result = operand_a @ operand_b
    return result


def transpose(operand, axes):
    return _create_from_instruction(operand, instruction=instructions.Transpose(order=axes))


def reshape(operand, newshape):
    return _create_from_instruction(operand, instruction=instructions.Reshape(new_shape=newshape))


def exp(operand):
    return _create_from_instruction(operand, instruction=instructions.Exponent())


def sum(operand, axis):
    return _create_from_instruction(operand, instruction=instructions.Reduce(operation="sum", axis=axis))