from typing import Union

import numpy as np
from pyrsistent import PClass, field


class Constant(PClass):
    array = field()

    def compute_shape(self, input_shapes: list[tuple]):
        return self.array.shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return self.array


class Transpose(PClass):
    order = field(type=tuple)

    def compute_shape(self, input_shapes: list[tuple]):
        (input_shape,) = input_shapes
        shape = [input_shape[axis] for axis in self.order]
        return tuple(shape)

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.transpose(input_arrays[0], self.order)


class Reshape(PClass):
    new_shape = field(type=tuple)

    def compute_shape(self, input_shapes: list[tuple]):
        return self.new_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.reshape(input_arrays[0], self.new_shape)


UnaryInstruction = Union[Transpose, Reshape]


class GetFromIndices(PClass):
    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        array, indices = input_arrays
        if isinstance(indices, np.ndarray):
            indices = tuple(indices)
        return array[indices]


class SetAtIndices(PClass):
    indices = field(type=tuple)

    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        old_array, new_slice = input_arrays
        new_array = old_array.copy()
        new_array[self.indices] = new_slice
        return new_array


class Add(PClass):
    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.add(*input_arrays)


class Subtract(PClass):
    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.subtract(*input_arrays)


class Multiply(PClass):
    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.multiply(*input_arrays)


class Divide(PClass):
    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, _ = input_shapes
        return input_a_shape

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.divide(*input_arrays)


class MatrixMultiply(PClass):

    def compute_shape(self, input_shapes: list[tuple]):
        input_a_shape, input_b_shape = input_shapes
        shape = list(input_a_shape)
        shape[-1] = input_b_shape[-1]
        return tuple(shape)

    def compute_data(self, input_arrays: list[np.ndarray]):
        input_a, input_b = input_arrays
        return np.matmul(input_a, input_b)


class Exponent(PClass):

    def compute_shape(self, input_shapes: list[tuple]):
        return tuple(input_shapes[0])

    def compute_data(self, input_arrays: list[np.ndarray]):
        return np.exp(input_arrays[0])


class Reduce(PClass):
    operation = field(type=str)
    axis = field(type=int)

    def compute_shape(self, input_shapes: list[tuple]):
        return tuple(input_shapes[0][:-1])

    def compute_data(self, input_arrays: list[np.ndarray]):
        function = getattr(np, self.operation)
        return function(input_arrays[0], axis=self.axis)
