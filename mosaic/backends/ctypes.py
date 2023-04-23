from ctypes import POINTER, c_float, c_int64

import numpy as np


def cast_numpy_array_to_pointer(flat_array):
    numpy_dtype_to_c_type = {
        np.dtype(np.float32): c_float,
        np.dtype(np.int64): c_int64,
    }
    c_type = numpy_dtype_to_c_type[flat_array.dtype]
    pointer_type = POINTER(c_type)
    return flat_array.ctypes.data_as(pointer_type)
