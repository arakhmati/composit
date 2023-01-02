from multimethod import multimethod

import numpy

from .types import (
    Constant,
    UnaryInstruction,
    BinaryInstruction,
)


@multimethod
def compute_data(instruction: Constant, input_arrays: list[numpy.ndarray]) -> tuple:
    return instruction.array


@multimethod
def compute_data(instruction: UnaryInstruction, input_arrays: list[numpy.ndarray]) -> tuple:
    return instruction.function(input_arrays[0])


@multimethod
def compute_data(instruction: BinaryInstruction, input_arrays: list[numpy.ndarray]) -> tuple:
    return instruction.function(*input_arrays)
