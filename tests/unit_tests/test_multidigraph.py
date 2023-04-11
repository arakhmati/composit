import networkx
from pyrsistent import PClass, field

from composit.multidigraph import MultiDiGraph
from composit.multidigraph import topological_traversal
from composit.hash import deterministic_hash


class Node(PClass):
    name = field(type=str, mandatory=True)

    def __hash__(self):
        return deterministic_hash(self)


def test_init():
    graph = MultiDiGraph()
    assert graph == MultiDiGraph()


def test_add_node_method():
    graph = MultiDiGraph()
    node = Node(name="0")
    new_graph = graph.add_node(node)
    another_new_graph = graph.add_node(node)
    assert new_graph != graph
    assert new_graph == another_new_graph


def test_adj_method():
    graph = MultiDiGraph()
    nodes = graph.adj


def test_nodes_method():
    graph = MultiDiGraph()
    networkx_graph = networkx.MultiDiGraph()
    assert graph.nodes() == networkx_graph.nodes()

    node = Node(name="0")
    graph = graph.add_node(node)
    networkx_graph.add_node(node)

    assert graph.nodes() == networkx_graph.nodes()
    print(networkx_graph.nodes(data=True, default=0))
    # assert graph.nodes(data=True) == networkx_graph.nodes(data=True, default=pmap)


def test_topological_sort():
    graph = MultiDiGraph()

    node_0 = Node(name="0")
    node_1 = Node(name="1")
    node_2 = Node(name="2")

    graph = graph.add_node(node_0).add_node(node_1).add_node(node_2).add_edge(node_0, node_2).add_edge(node_1, node_2)

    sorted_nodes = topological_traversal(graph)
    assert list(sorted_nodes) == [node_0, node_1, node_2]
