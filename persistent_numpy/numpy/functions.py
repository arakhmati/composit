import inspect
import sys
from typing import Union

import numpy as np
from pyrsistent import immutable, PClass, field
from toolz import functoolz

from persistent_numpy.introspection import get_name_from_args_and_kwargs
from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal, merge_graphs, compose_all
from persistent_numpy.persistent_array import PersistentArray, Node
from persistent_numpy.string import random_string

THIS_MODULE = sys.modules[__name__]


def get_operands(graph, node):
    def sort_key(in_edge):
        _, _, data = in_edge
        return data["sink_input_index"]

    def node_operand(in_edge):
        predecessor, _, data = in_edge
        return predecessor, data["source_output_index"]

    return functoolz.pipe(
        graph.in_edges(node, data=True),
        list,
        functoolz.partial(sorted, key=sort_key),
        functoolz.partial(map, node_operand),
    )


def to_numpy(*outputs: tuple[PersistentArray]):
    graph = compose_all(*tuple(output.graph for output in outputs))

    cache = {}
    for node in topological_traversal(graph):
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

    result = [cache[(output.node, output.output_index)] for output in outputs]
    if len(result) == 1:
        return result[0]
    return result


class _ndarray(PClass):
    array = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        return self.array

    def __hash__(self):
        return int(self.array.sum())


class _get_item(PClass):
    def __call__(self, *input_arrays: list[np.ndarray]):
        array, indices = input_arrays
        if isinstance(indices, np.ndarray):
            if indices.shape == ():
                indices = indices.reshape((1,))
            indices = tuple(indices)
        return array[indices]


class _set_item(PClass):
    indices = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        old_array, new_slice = input_arrays
        new_array = old_array.copy()
        new_array[self.indices] = new_slice
        return new_array


def create_ndarray(name: str, array: np.ndarray):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=_ndarray(array=array), shapes=(array.shape,))
    return PersistentArray(graph=graph, node=node)


@functoolz.memoize
def instruction_shape(instruction, input_shapes):
    dummy_input_arrays = [np.zeros(input_shape, dtype=np.int32) for input_shape in input_shapes]
    result = instruction(*dummy_input_arrays)
    if isinstance(result, np.ndarray):
        return (result.shape,)
    elif isinstance(result, list):
        return tuple(array.shape for array in result)
    elif isinstance(result, np.int32):
        return (np.asarray(result).shape,)
    else:
        raise RuntimeError(f"Unsupported type: {type(result)}")


def create_from_numpy_compute_instruction(*operands, instruction) -> Union[PersistentArray, tuple[PersistentArray]]:

    operands = list(operands)
    for index, operand in enumerate(operands):
        if isinstance(operands[index], (int, float)):
            operands[index] = create_ndarray(f"Scalar({operands[index]})", np.asarray(operands[index]))

    graph = merge_graphs(*tuple((operand.graph, operand.node) for operand in operands))

    shapes = instruction_shape(
        instruction,
        tuple(operand.shape for operand in operands),
    )

    name = f"{type(instruction).__name__}-{random_string()}"
    new_node = Node(name=name)
    graph = graph.add_node(new_node, instruction=instruction, shapes=shapes)
    for index, operand in enumerate(operands):
        graph = graph.add_edge(operand.node, new_node, source_output_index=operand.output_index, sink_input_index=index)

    result = tuple(
        PersistentArray(graph=graph, node=new_node, output_index=output_index) for output_index, _ in enumerate(shapes)
    )
    if len(result) == 1:
        return result[0]
    return result


def get_item(self, indices) -> "PersistentArray":
    if not isinstance(indices, PersistentArray):
        name = f"{indices}"
        indices = create_ndarray(name, np.asarray(indices, dtype=int))
    return create_from_numpy_compute_instruction(self, indices, instruction=_get_item())


PersistentArray.__getitem__ = get_item


def set_item(self, indices, values) -> "PersistentArray":
    return create_from_numpy_compute_instruction(self, values, instruction=_set_item(indices=indices))


PersistentArray.set_item = set_item


def asarray(array, name=None):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if name is None:
        name = random_string()

    return create_ndarray(name, array)


def named_ndarray(*args, name, **kwargs):
    array = np.ndarray(*args, **kwargs)
    array[:] = 0
    return create_ndarray(name, array)


def ndarray(*args, **kwargs):
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs) + "-" + random_string()
    return named_ndarray(*args, name=name, **kwargs)


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_ndarray(name, array)


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_ndarray(name, array)


def _create_numpy_compute_instruction(function_name, *args, **kwargs):
    numpy_function = getattr(np, function_name)
    if isinstance(numpy_function, np.ufunc):
        klass_kwargs = {}
    else:
        numpy_function_parameters = inspect.signature(numpy_function).parameters

        klass_args = [key for key, value in numpy_function_parameters.items() if value.default == inspect._empty]
        klass_kwargs = {
            key: value.default for key, value in numpy_function_parameters.items() if value.default != inspect._empty
        }

        # Use passed in kwargs
        for key, value in kwargs.items():
            klass_kwargs[key] = value

        # Distribute args across klass_args and klass_kwargs
        for index, arg in enumerate(args):
            if isinstance(arg, PersistentArray):
                continue
            if index < len(klass_args):
                key = klass_args[index]
                klass_kwargs[key] = arg
            else:
                kwargs_index = len(klass_args) - index
                key = list(klass_kwargs.keys())[kwargs_index]
                klass_kwargs[key] = arg

    def compute(self, *args, **kwargs):
        return numpy_function(*args, **kwargs, **self._asdict())

    klass_attributes = list(klass_kwargs.keys())
    klass = immutable(klass_attributes, name=function_name)
    klass.__call__ = compute

    return klass(**klass_kwargs)


def _create_numpy_compute_function(function_name):
    def function(*args, **kwargs):
        operands = [arg for arg in args if isinstance(arg, PersistentArray)]
        return create_from_numpy_compute_instruction(
            *operands,
            instruction=_create_numpy_compute_instruction(function_name, *args, **kwargs),
        )

    return function


COMPUTE_FUNCTIONS = [
    # Unary
    "abs",
    "exp",
    "sqrt",
    "square",
    # Data Movement
    "transpose",
    "reshape",
    "split",
    # Reduce
    "sum",
    "max",
    "mean",
    "var",
    # Broadcast,
    "broadcast_to",
]

__all__ = COMPUTE_FUNCTIONS.copy()

for function_name in COMPUTE_FUNCTIONS:
    setattr(THIS_MODULE, function_name, _create_numpy_compute_function(function_name))


def _create_numpy_binary_compute_function(function_name):
    def function(*args, **kwargs):
        operand_a, operand_b, *args = args
        if not isinstance(operand_b, PersistentArray):
            operand_b = create_ndarray(f"Scalar({operand_b})", np.asarray(operand_b))
        return create_from_numpy_compute_instruction(
            operand_a,
            operand_b,
            instruction=_create_numpy_compute_instruction(function_name, *args, **kwargs),
        )

    return function


BINARY_COMPUTE_FUNCTIONS = [
    "add",
    ("subtract", "__sub__"),
    ("multiply", "__mul__"),
    ("divide", "__truediv__"),
    "matmul",
]

for function_name in BINARY_COMPUTE_FUNCTIONS:
    if isinstance(function_name, tuple):
        function_name, dunder_method_name = function_name
    else:
        dunder_method_name = f"__{function_name}__"

    __all__.append(function_name)
    setattr(THIS_MODULE, function_name, _create_numpy_binary_compute_function(function_name))
    function = getattr(THIS_MODULE, function_name)
    setattr(PersistentArray, dunder_method_name, function)


def _create_numpy_concatenate_function():
    def _create_concatenate_instruction(axis=0, out=None, dtype=None, casting="same_kind"):
        klass_kwargs = {
            "axis": axis,
            "out": out,
            "dtype": dtype,
            "casting": casting,
        }

        def compute(self, *inputs):
            return np.concatenate(inputs, **self._asdict())

        klass_attributes = list(klass_kwargs.keys())
        klass = immutable(klass_attributes, name="concatenate")
        klass.__call__ = compute

        return klass(**klass_kwargs)

    def function(inputs, *args, **kwargs):
        return create_from_numpy_compute_instruction(
            *inputs,
            instruction=_create_concatenate_instruction(*args, **kwargs),
        )

    return function


setattr(THIS_MODULE, "concatenate", _create_numpy_concatenate_function())
__all__.append("concatenate")


__all__.extend(
    [
        "create_ndarray",
        "create_from_numpy_compute_instruction",
        "get_operands",
        "ndarray",
        "zeros",
        "ones",
        "set_item",
        "to_numpy",
    ]
)
