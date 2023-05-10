import enum

import numpy as np

from pyrsistent import PClass, field

from mosaic.backends.ctypes import cast_numpy_array_to_pointer


class BufferType(enum.Enum):
    ConstantInput = enum.auto()
    VariableInput = enum.auto()
    Intermediate = enum.auto()


class BufferDescriptor(PClass):
    name: str = field()
    buffer_type: BufferType = field()

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"BufferDescriptor(name={self.name}, buffer_type={self.buffer_type})"

    def __lt__(self, other):
        return self.name < other.name


class ConstantBufferDescriptor(PClass):
    name: str = field()
    buffer_type: BufferType = field()
    array = field()

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other: "ConstantBufferDescriptor"):
        return (
            self.name == other.name and self.buffer_type == other.buffer_type and np.allclose(self.array, other.array)
        )

    def __repr__(self):
        return f"ConstantBufferDescriptor(name={self.name}, buffer_type={self.buffer_type}, array={self.array})"

    def __lt__(self, other):
        return self.name < other.name


class Buffer(PClass):
    array = field()

    def data(self):
        return cast_numpy_array_to_pointer(self.array)


class ModelWithoutKernelFusion(PClass):
    buffer_graph = field()
    node_to_run_kernel = field()
    buffer_descriptor_to_buffer = field()


class ModelWithKernelFusion(PClass):
    buffer_graph = field()
    run_model = field()
    buffer_descriptor_to_buffer = field()
