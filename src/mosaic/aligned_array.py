import numpy as np


def create_aligned_array(shape, dtype, alignment=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + alignment, dtype=np.uint8)
    offset = buffer.ctypes.data % alignment
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def create_aligned_array_like(array, alignment=32):
    size = array.itemsize * array.size
    buffer = np.empty(size + alignment, dtype=np.uint8)
    offset = buffer.ctypes.data % alignment
    array = np.ndarray(array.shape, dtype=array.dtype, buffer=buffer, offset=offset)
    return array


def align_array(array, alignment=1024):
    if array.__array_interface__["data"][0] % alignment == 0:
        return array
    aligned_array = create_aligned_array_like(array, alignment=alignment)
    aligned_array[:] = array
    return aligned_array
