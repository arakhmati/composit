from multimethod import multimethod

from persistent_numpy.multidigraph import MultiDiGraph, Node

from .types import (
    Constant,
    UnaryInstruction,
    BinaryInstruction,
    Transpose,
    MatrixMultiply,
)


@multimethod
def compute_shape(instruction: Constant, input_shapes: list[tuple]) -> tuple:
    return instruction.array.shape


@multimethod
def compute_shape(instruction: BinaryInstruction, input_shapes: list[tuple]) -> tuple:
    input_a_shape, _ = input_shapes
    return input_a_shape


@multimethod
def compute_shape(instruction: UnaryInstruction, input_shapes: list[tuple]) -> tuple:
    (input_shape,) = input_shapes
    return tuple(input_shape)


@multimethod
def compute_shape(instruction: Transpose, input_shapes: list[tuple]) -> tuple:
    (input_shape,) = input_shapes
    shape = [input_shape[axis] for axis in instruction.order]
    return tuple(shape)


@multimethod
def compute_shape(instruction: MatrixMultiply, input_shapes: list[tuple]) -> tuple:
    input_a_shape, input_b_shape = input_shapes
    shape = list(input_a_shape)
    shape[-1] = input_b_shape[-1]
    return tuple(shape)
