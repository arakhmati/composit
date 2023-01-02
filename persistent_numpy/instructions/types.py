from typing import Union

import numpy as np
from pyrsistent import PClass, field


class Constant(PClass):
    array = field()


class GetFromIndices(PClass):
    def function(self, array, indices):
        return array[indices]


class Transpose(PClass):
    order = field(type=tuple)

    def function(self, array):
        return np.transpose(array, self.order)


UnaryInstruction = Union[Transpose]


class SetAtIndices(PClass):
    indices = field(type=tuple)

    def function(self, old_array, new_slice):
        new_array = old_array.copy()
        new_array[self.indices] = new_slice
        return new_array


class Add(PClass):
    function = np.add


class Subtract(PClass):
    function = np.subtract


class Multiply(PClass):
    function = np.multiply


class Divide(PClass):
    function = np.divide


class MatrixMultiply(PClass):
    function = np.matmul


BinaryInstruction = Union[GetFromIndices, SetAtIndices, Add, Subtract, Multiply, Divide, MatrixMultiply]
