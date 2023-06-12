from ctypes import POINTER, c_float, c_double, c_int64

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
