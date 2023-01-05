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
    output_index = field(type=int, initial=0)

    @property
    def name(self) -> str:
        return self.node.name

    @property
    def shape(self) -> tuple:
        return self.graph.nodes[self.node]["shapes"][self.output_index]

    def visualize(self):
        def visualize_node(graph, node):
            shapes = graph.nodes[node]["shapes"]
            return f"{node}:{shapes}"

        visualize_graph(self.graph, visualize_node=visualize_node)
