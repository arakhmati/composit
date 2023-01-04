import numpy as np
from pyrsistent import PClass, field

from persistent_numpy.multidigraph import MultiDiGraph, topological_traversal
from persistent_numpy.hash import deterministic_hash


class Node(PClass):
    name = field(type=str, mandatory=True)

    def __hash__(self):
        return deterministic_hash(self)


class PersistentArray(PClass):
    graph = field(type=MultiDiGraph)
    node = field(type=Node)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> tuple:
        shape = self.graph.get_node_attribute(self.node, "shape")
        return shape

    def to_numpy(self) -> np.ndarray:
        return _compute(self.graph, self.node)


def _operands(graph, node):
    result = ((data["sink_input_port"], predecessor) for predecessor, _, data in graph.in_edges(node, data=True))
    return [element[1] for element in sorted(result, key=lambda element: element[0])]


def _compute(graph, result_node):

    sorted_nodes = list(topological_traversal(graph))
    cache = {}
    for node in sorted_nodes:
        instruction = graph.get_node_attribute(node, "instruction")
        input_arrays = [cache[operand] for operand in _operands(graph, node)]
        cache[node] = instruction(*input_arrays)
    return cache[result_node]
