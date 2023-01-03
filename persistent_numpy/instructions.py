import numpy as np
from pyrsistent import PClass, field


class ndarray(PClass):
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
