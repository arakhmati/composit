import inspect
import math
import random
import sys

import numpy as np
from pyrsistent import immutable
from toolz import functoolz

from persistent_numpy.multidigraph import MultiDiGraph, Node
from persistent_numpy.ndarray import PersistentArray
from persistent_numpy import instructions

THIS_MODULE = sys.modules[__name__]


def _random_string(num_characters=10):
    result = "".join(random.choice("abcdef0123456789") for i in range(num_characters))
    return result


def _create_from_array(name: str, array: np.ndarray):
    node = Node(name=name)
    graph = MultiDiGraph().add_node(node, instruction=instructions.ndarray(array=array), shape=array.shape)
    return PersistentArray(graph, node)


@functoolz.memoize
def instruction_shape(instruction, input_shapes):
    dummy_input_arrays = [np.zeros(input_shape, dtype=np.int32) for input_shape in input_shapes]
    return instruction(*dummy_input_arrays).shape


def _create_from_numpy_compute_instruction(*operands, instruction) -> "PersistentArray":

    operands = list(operands)
    for index, operand in enumerate(operands):
        if isinstance(operands[index], (int, float)):
            operands[index] = _create_from_array(f"Scalar({operands[index]})", np.asarray(operands[index]))

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

    return PersistentArray(graph, new_node)


def get_item(self, indices) -> "PersistentArray":
    if not isinstance(indices, PersistentArray):
        name = f"{indices}"
        indices = _create_from_array(name, np.asarray(indices, dtype=int))
    return _create_from_numpy_compute_instruction(self, indices, instruction=instructions.get_item())


PersistentArray.__getitem__ = get_item


def set_item(self, indices, values) -> "PersistentArray":
    return _create_from_numpy_compute_instruction(self, values, instruction=instructions.set_item(indices=indices))


PersistentArray.set_item = set_item


def asarray(array, name=None):
    if not isinstance(array, np.ndarray):
        array = np.asarray(array)
    if name is None:
        name = _random_string()

    return _create_from_array(name, array)


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
    return _create_from_array(name, array)


def ndarray(*args, **kwargs):
    name = (
        _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs) + "-" + _random_string()
    )
    return named_ndarray(*args, name=name, **kwargs)


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    name = _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return _create_from_array(name, array)


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    name = _get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return _create_from_array(name, array)


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
            operand_b = _create_from_array(f"Scalar({operand_b})", np.asarray(operand_b))
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
