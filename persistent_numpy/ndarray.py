import numpy as np

from persistent_numpy.multidigraph import topological_traversal


class PersistentArray:
    def __init__(self, graph, node):
        self.graph = graph
        self.node = node

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> tuple:
        shape = self.graph.get_node_attribute(self.node, "shape")
        return shape

    def to_numpy(self) -> np.ndarray:
        sorted_nodes = list(topological_traversal(self.graph))
        cache = {}
        for node in sorted_nodes:
            instruction = self.graph.get_node_attribute(node, "instruction")
            input_arrays = [cache[operand] for operand in _operands(self.graph, node)]
            cache[node] = instruction(*input_arrays)
        return cache[self.node]


def _operands(graph, node):
    result = ((data["sink_input_port"], predecessor) for predecessor, _, data in graph.in_edges(node, data=True))
    return [element[1] for element in sorted(result, key=lambda element: element[0])]
