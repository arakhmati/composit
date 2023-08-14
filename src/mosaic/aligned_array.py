import numpy as np


def create_aligned_array(shape, dtype, alignment=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + alignment, dtype=np.uint8)
    offset = buffer.ctypes.data % alignment
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def align_array(array, alignment=32):
    aligned_array = create_aligned_array(array.shape, array.dtype, alignment=alignment)
    aligned_array[:] = array
    return aligned_array
