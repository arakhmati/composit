from pyrsistent import PClass, field

from composit.introspection import class_name
from composit.multidigraph import MultiDiGraph, visualize_graph, compose_all
from composit.hash import deterministic_hash


class Node(PClass):
    name = field(type=str, mandatory=True)

    def __hash__(self):
        return deterministic_hash(self)


class LazyTensor(PClass):
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
    def dtype(self) -> tuple:
        return self.graph.nodes[self.node]["dtypes"][self.output_index]

    @property
    def rank(self) -> int:
        return len(self.shape)


def visualize(*lazy_tensors, **kwargs):
    def visualize_node(graphviz_graph, graph, node):
        instruction = graph.nodes[node]["instruction"]
        shapes = graph.nodes[node]["shapes"]
        if len(shapes) == 1:
            (shapes,) = shapes
        graphviz_graph.node(node.name, label=f"{node}\n{class_name(instruction)}\n{shapes}")

    graph = compose_all(*(lazy_tensor.graph for lazy_tensor in lazy_tensors))

    return visualize_graph(graph, visualize_node=visualize_node, **kwargs)
