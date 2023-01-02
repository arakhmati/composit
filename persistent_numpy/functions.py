import random

import numpy as np

from persistent_numpy.ndarray import PersistentArray, create_unary_instruction
from persistent_numpy.instructions import Transpose


def asarray(array, name=None):
    if name is None:
        name = "".join(random.choice("abcdef0123456789") for i in range(10))

    return PersistentArray.create(name, array)


def named_ndarray(*args, name, **kwargs):
    array = np.ndarray(*args, **kwargs)
    return PersistentArray.create(name, array)


def ndarray(*args, **kwargs):
    return named_ndarray(*args, name=f"ndarray({args}, {kwargs})", **kwargs)


def zeros(*args, **kwargs):
    array = np.zeros(*args, **kwargs)
    return PersistentArray.create(f"zeros({args}, {kwargs})", array)


def ones(*args, **kwargs):
    array = np.ones(*args, **kwargs)
    return PersistentArray.create(f"ones({args}, {kwargs})", array)


def matmul(operand_a, operand_b):
    result = operand_a @ operand_b
    return result


def transpose(operand, axes):
    return create_unary_instruction(operand, "transpose", Transpose(order=axes))
