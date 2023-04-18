from ctypes import POINTER, c_float


def cast_numpy_array_to_pointer(flat_array):
    c_float_p = POINTER(c_float)
    return flat_array.ctypes.data_as(c_float_p)
