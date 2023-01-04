import inspect
import random
import sys

import numpy as np
from pyrsistent import immutable, PClass, field
from toolz import functoolz

from persistent_numpy.multidigraph import MultiDiGraph
from persistent_numpy.ndarray import PersistentArray, Node

THIS_MODULE = sys.modules[__name__]


def _random_string(num_characters=10):
    result = "".join(random.choice("abcdef0123456789") for i in range(num_characters))
    return result


class _ndarray(PClass):
    array = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        return self.array


class get_item(PClass):
    def __call__(self, *input_arrays: list[np.ndarray]):
        array, indices = input_arrays
        if isinstance(indices, np.ndarray):
            if indices.shape == ():
                indices = indices.reshape((1,))
            indices = tuple(indices)
        return array[indices]


class set_item(PClass):
    indices = field(type=tuple)

    def __call__(self, *input_arrays: list[np.ndarray]):
        old_array, new_slice = input_arrays
        new_array = old_array.copy()
        new_array[self.indices] = new_slice
        return new_array


def _create_ndarray(name: str, array: np.ndarray):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=_ndarray(array=array), shape=array.shape)
    return PersistentArray(graph=graph, node=node)


@functoolz.memoize
def instruction_shape(instruction, input_shapes):
    dummy_input_arrays = [np.zeros(input_shape, dtype=np.int32) for input_shape in input_shapes]
    return instruction(*dummy_input_arrays).shape


def _create_from_numpy_compute_instruction(*operands, instruction) -> "PersistentArray":

    operands = list(operands)
    for index, operand in enumerate(operands):
        if isinstance(operands[index], (int, float)):
            operands[index] = _create_ndarray(f"Scalar({operands[index]})", np.asarray(operands[index]))

    graph = operands[0].graph
    for operand in operands[1:]:
        graph = graph.merge(operand.graph, operand.node)

    shape = instruction_shape(
        instruction, tuple(graph.get_node_attribute(operand.node, "shape") for operand in operands)
    )

    name = f"{type(instruction).__name__}-{_random_string()}"
    new_node = Node(name=name)
    graph = graph.add_node(new_node, instruction=instruction, shape=shape)
    for index, operand in enumerate(operands):
        graph = graph.add_edge(operand.node, new_node, source_output_port=0, sink_input_port=index)

    return PersistentArray(graph=graph, node=new_node)


def get_item(self, indices) -> "PersistentArray":
    if not isinstance(indices, PersistentArray):
        name = f"{indices}"
        indices = _create_ndarray(name, np.asarray(indices, dtype=int))
    return _create_from_numpy_compute_instruction(self, indices, instruction=get_item())


PersistentArray.__getitem__ = get_item


def set_item(self, indices, values) -> "PersistentArray":
    return _create_from_numpy_compute_instruction(self, values, instruction=set_item(indices=indices))


PersistentArray.set_item = set_item


def asarray(array, name=None):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if name is None:
        name = _random_string()

    return _create_ndarray(name, array)


def _get_name_from_args_and_kwargs(function_name, *args, **kwargs):
    args_string = ", ".join(f"{arg}" for arg in args)
    kwargs_string = ", ".join(f"{key}={value}" for key, value in kwargs.items())
    result = f"{function_name}({args_string}"
    if kwargs_string:
        result = f"{result}, {kwargs_string})"
    else:
        result = f"{result})"
    return result


def named_ndarray(*args, name, **kwargs):
    array = np.ndarray(*args, **kwargs)
    array[:] = 0
    return _create_ndarray(name, array)


def ndarray(*args, **kwargs):
    name = (
        _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs) + "-" + _random_string()
    )
    return named_ndarray(*args, name=name, **kwargs)


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    name = _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return _create_ndarray(name, array)


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    name = _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return _create_ndarray(name, array)


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
        return _create_from_numpy_compute_instruction(
            *operands, instruction=_create_numpy_compute_instruction(function_name, *args, **kwargs)
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
    # Reduce
    "sum",
    "max",
    "mean",
    "var",
]

for function_name in COMPUTE_FUNCTIONS:
    setattr(THIS_MODULE, function_name, _create_numpy_compute_function(function_name))


def _create_numpy_binary_compute_function(function_name):
    def function(*args, **kwargs):
        operand_a, operand_b, *args = args
        if not isinstance(operand_b, PersistentArray):
            operand_b = _create_ndarray(f"Scalar({operand_b})", np.asarray(operand_b))
        return _create_from_numpy_compute_instruction(
            operand_a, operand_b, instruction=_create_numpy_compute_instruction(function_name, *args, **kwargs)
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
    setattr(THIS_MODULE, function_name, _create_numpy_binary_compute_function(function_name))
    function = getattr(THIS_MODULE, function_name)
    setattr(PersistentArray, dunder_method_name, function)
