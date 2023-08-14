from ctypes import POINTER, c_float, c_double, c_int64
from typing import Sequence

import numpy as np


def get_ctype_from_numpy_dtype(dtype):
    return {
        np.dtype(np.float32): c_float,
        np.dtype(np.float64): c_double,
        np.dtype(np.int64): c_int64,
    }[dtype]


def get_ctype_string_from_numpy_dtype(dtype):
    return {
        np.dtype(np.float32): "float",
        np.dtype(np.float64): "double",
        np.dtype(np.int64): "int64_t",
    }[dtype]


def cast_numpy_array_to_pointer(array: np.array):
    c_type = get_ctype_from_numpy_dtype(array.dtype)
    pointer_type = POINTER(c_type)
    return array.ctypes.data_as(pointer_type)


def cast_numpy_arrays_to_pointer(arrays: Sequence[np.array]):
    dtypes = set(array.dtype for array in arrays)
    assert len(dtypes) == 1

    pointers = [cast_numpy_array_to_pointer(array) for array in arrays]
    pointer_to_array = type(pointers[0])
    result = (pointer_to_array * len(pointers))()
    for index, pointer in enumerate(pointers):
        result[index] = pointer
    return result
