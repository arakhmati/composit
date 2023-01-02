import numpy as np

from persistent_numpy.multidigraph import MultiDiGraph, Node, topological_traversal
from persistent_numpy.instructions import compute_data, compute_shape
from persistent_numpy.instructions import (
    Constant,
    GetFromIndices,
    SetAtIndices,
    Add,
    Subtract,
    Multiply,
    Divide,
    MatrixMultiply,
)


class PersistentArray:
    def __init__(self, graph, node):
        self.graph = graph
        self.node = node

    @classmethod
    def create(cls, name: str, array: np.ndarray):
        node = Node(name=name)
        graph = MultiDiGraph().add_node(node, instruction=Constant(array=array))
        return cls(graph, node)

    @property
    def name(self) -> str:
        return self.node.name

    def numpy(self) -> np.ndarray:
        sorted_nodes = list(topological_traversal(self.graph))
        cache = {}
        for node in sorted_nodes:
            instruction = self.graph.get_node_attribute(node, "instruction")
            input_arrays = [cache[operand] for operand in operands(self.graph, node)]
            cache[node] = compute_data(instruction, input_arrays)
        return cache[self.node]

    @property
    def shape(self) -> tuple:
        sorted_nodes = topological_traversal(self.graph)
        cache = {}
        for node in sorted_nodes:
            instruction = self.graph.get_node_attribute(node, "instruction")
            input_shapes = [cache[operand] for operand in operands(self.graph, node)]
            cache[node] = compute_shape(instruction, input_shapes)
        return cache[self.node]

    def get_from_indices(self, indices) -> "PersistentArray":
        if not isinstance(indices, PersistentArray):
            name = f"{indices}"
            indices = PersistentArray.create(name, indices)
        return create_binary_instruction(self, indices, "get_from_indices", GetFromIndices())

    __getitem__ = get_from_indices

    def set_at_indices(self, indices, values) -> "PersistentArray":
        return create_binary_instruction(self, values, "set_at_indices", SetAtIndices(indices=indices))

    def __add__(self, other: "PersistentArray") -> "PersistentArray":
        return create_binary_instruction(self, other, "add", Add())

    def __sub__(self, other: "PersistentArray") -> "PersistentArray":
        return create_binary_instruction(self, other, "subtract", Subtract())

    def __mul__(self, other: "PersistentArray") -> "PersistentArray":
        return create_binary_instruction(self, other, "multiply", Multiply())

    def __truediv__(self, other: "PersistentArray") -> "PersistentArray":
        return create_binary_instruction(self, other, "divide", Divide())

    def __matmul__(self, other: "PersistentArray") -> "PersistentArray":
        return create_binary_instruction(self, other, "matmul", MatrixMultiply())


def create_unary_instruction(operand, op_name, instruction) -> "PersistentArray":

    name = f"{op_name}({operand.name})"
    new_node = Node(name=name)

    graph = operand.graph.add_node(new_node, instruction=instruction).add_edge(
        operand.node, new_node, source_output_port=0, sink_input_port=0
    )

    return PersistentArray(graph, new_node)


def create_binary_instruction(operand_a, operand_b, op_name, instruction) -> "PersistentArray":

    if isinstance(operand_b, (int, float)):
        operand_b = PersistentArray.create(f"Scalar({operand_b})", np.asarray(operand_b))

    name = f"{op_name}({operand_a.name}, {operand_b.name})"
    new_node = Node(name=name)

    graph = (
        operand_a.graph.merge(operand_b.graph)
        .add_node(new_node, instruction=instruction)
        .add_edge(operand_a.node, new_node, source_output_port=0, sink_input_port=0)
        .add_edge(operand_b.node, new_node, source_output_port=0, sink_input_port=1)
    )

    return PersistentArray(graph, new_node)


def operands(graph, node):
    operands = []
    for predecessor in graph.predecessors(node):
        for edge in graph._pred[node][predecessor]:
            operands.append((edge["sink_input_port"], predecessor))
    return [element[1] for element in sorted(operands, key=lambda element: element[0])]
