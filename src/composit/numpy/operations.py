from __future__ import annotations

import inspect
import sys

import numpy as np
from pyrsistent import PClass, field

from composit.numpy.core import (
    create_input,
    create_from_numpy_compute_operation,
    create_numpy_compute_function,
    create_numpy_binary_compute_function,
    create_numpy_concatenate_function,
    wrap_as_operation,
)
from composit.introspection import get_name_from_args_and_kwargs
from composit.types import LazyTensor
from composit.string import random_string

THIS_MODULE = sys.modules[__name__]
__all__ = []


class IntegerIndex(PClass):
    value = field()


class SliceIndex(PClass):
    start = field()
    stop = field()
    step = field()

    @property
    def value(self):
        return slice(self.start, self.stop, self.step)


class EllipsisIndex(PClass):
    @property
    def value(self):
        return Ellipsis


def process_indices(indices):
    result = []
    for index in indices:
        if isinstance(index, int):
            index = IntegerIndex(value=index)
        elif isinstance(index, slice):
            index = SliceIndex(start=index.start, stop=index.stop, step=index.step)
        elif index == Ellipsis:
            index = EllipsisIndex()
        else:
            raise TypeError(f"Unsupported type: {type(index)}")
        result.append(index)
    return tuple(result)


class GetItem(PClass):
    indices = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        (array,) = input_arrays
        indices = tuple(index.value for index in self.indices)
        return array[indices]


class DynamicGetItem(PClass):
    def __call__(self, *input_arrays: list[np.ndarray]):
        array, indices = input_arrays
        if isinstance(indices, np.ndarray):
            if indices.shape == ():
                indices = indices.reshape((1,))
            indices = tuple(indices)
        return array[indices]


def get_item(self, indices) -> "LazyTensor":
    if not all(isinstance(index, int) for index in indices):
        indices = process_indices(indices)
        return create_from_numpy_compute_operation(self, operation=GetItem(indices=indices))

    if not isinstance(indices, LazyTensor):
        name = f"Indices({indices})"
        indices_as_array = np.asarray(indices, dtype=int)

        def initialize_callback():
            return indices_as_array

        indices = create_input(name, initialize_callback, indices_as_array.shape, indices_as_array.dtype)

    return create_from_numpy_compute_operation(self, indices, operation=DynamicGetItem())


__all__.append("get_item")
LazyTensor.__getitem__ = get_item


class SetItem(PClass):
    indices = field()

    def __call__(self, *input_arrays: list[np.ndarray]):
        old_array, new_slice = input_arrays
        new_array = old_array.copy()
        indices = tuple(index.value for index in self.indices)
        new_array[indices] = new_slice
        return new_array


def set_item(self, indices, values) -> "LazyTensor":
    indices = process_indices(indices)
    return create_from_numpy_compute_operation(self, values, operation=SetItem(indices=indices))


__all__.append("set_item")
LazyTensor.set_item = set_item


def asarray(array, name=None):
    if isinstance(array, LazyTensor):
        return array

    if not isinstance(array, np.ndarray):
        array = np.asarray(array)

    if name is None:
        name = f"array_{random_string()}"

    return create_input(name, lambda: array, array.shape, array.dtype)


__all__.append("asarray")


def ndarray(*args, name=None, **kwargs):
    if name is None:
        function_name = inspect.currentframe().f_code.co_name
        name = f"{get_name_from_args_and_kwargs(function_name, *args,**kwargs)}-{random_string()}"

    array = np.ndarray(*args, **kwargs)
    array[:] = 0
    return create_input(name, lambda: array, array.shape, array.dtype)


__all__.append("ndarray")


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_input(name, lambda: array, array.shape, array.dtype)


__all__.append("zeros")


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_input(name, lambda: array, array.shape, array.dtype)


__all__.append("ones")


def arange(*args, **kwargs):
    array = np.arange(*args, **kwargs)
    name = get_name_from_args_and_kwargs(inspect.currentframe().f_code.co_name, *args, **kwargs)
    return create_input(name, lambda: array, array.shape, array.dtype)


__all__.append("arange")


def square(input_tensor):
    return input_tensor * input_tensor


__all__.append("square")


COMPUTE_FUNCTIONS = [
    # Unary
    "abs",
    "exp",
    "sqrt",
    "reciprocal",
    "sin",
    "cos",
    "tanh",
    "log",
    # Data Manipulation
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

for function_name in COMPUTE_FUNCTIONS:
    __all__.append(function_name)
    setattr(THIS_MODULE, function_name, create_numpy_compute_function(function_name))


BINARY_COMPUTE_FUNCTIONS = [
    ("add", "__add__"),
    ("subtract", "__sub__"),
    ("multiply", "__mul__"),
    ("divide", "__truediv__"),
    ("matmul", "__matmul__"),
    "power",
]

for function_name in BINARY_COMPUTE_FUNCTIONS:
    if isinstance(function_name, tuple):
        function_name, dunder_method_name = function_name
    else:
        dunder_method_name = None

    __all__.append(function_name)
    setattr(THIS_MODULE, function_name, create_numpy_binary_compute_function(function_name))
    function = getattr(THIS_MODULE, function_name)
    if dunder_method_name is not None:
        setattr(LazyTensor, dunder_method_name, function)


setattr(THIS_MODULE, "concatenate", create_numpy_concatenate_function())
__all__.append("concatenate")


@wrap_as_operation()
def astype(input_tensor, *, dtype: np.dtype):
    return input_tensor.astype(dtype)


__all__.append("astype")
