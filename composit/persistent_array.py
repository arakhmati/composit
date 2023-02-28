from pyrsistent import PClass, field

from composit.multidigraph import MultiDiGraph, visualize_graph, compose_all
from composit.hash import deterministic_hash


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

    @property
    def rank(self) -> int:
        return len(self.shape)


def visualize(*arrays):
    def visualize_node(graphviz_graph, graph, node):
        shapes = graph.nodes[node]["shapes"]
        graphviz_graph.node(node.name, label=f"{node}:{shapes}")

    graph = compose_all(*(array.graph for array in arrays))

    visualize_graph(graph, visualize_node=visualize_node)
