from pyrsistent import PClass, field

from persistent_numpy.multidigraph import MultiDiGraph, visualize_graph
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

    def visualize(self):
        def visualize_node(graph, node):
            shape = graph.get_node_attribute(node, "shape")
            return f"{node}:{shape}"

        visualize_graph(self.graph, visualize_node=visualize_node)
