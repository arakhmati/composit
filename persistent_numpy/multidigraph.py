import zlib
import pickle


import graphviz
import networkx
from pyrsistent import PClass, field, pmap_field, PMap, pmap, PVector, pvector

from networkx.classes.coreviews import AdjacencyView, MultiAdjacencyView
from networkx.classes.reportviews import InDegreeView, EdgeView, NodeView, NodeDataView


[
    "_adj",
    "_node",
    "_pred",
    "_succ",
    "add_edge",
    "add_edges_from",
    "add_node",
    "add_nodes_from",
    "add_weighted_edges_from",
    "adj",
    "adjacency",
    "adjlist_inner_dict_factory",
    "adjlist_outer_dict_factory",
    "clear",
    "clear_edges",
    "copy",
    "degree",
    "edge_attr_dict_factory",
    "edge_key_dict_factory",
    "edge_subgraph",
    "edges",
    "get_edge_data",
    "graph",
    "graph_attr_dict_factory",
    "has_edge",
    "has_node",
    "has_predecessor",
    "has_successor",
    "in_degree",
    "in_edges",
    "is_directed",
    "is_multigraph",
    "name",
    "nbunch_iter",
    "neighbors",
    "new_edge_key",
    "node_attr_dict_factory",
    "node_dict_factory",
    "nodes",
    "number_of_edges",
    "number_of_nodes",
    "order",
    "out_degree",
    "out_edges",
    "pred",
    "predecessors",
    "remove_edge",
    "remove_edges_from",
    "remove_node",
    "remove_nodes_from",
    "reverse",
    "size",
    "subgraph",
    "succ",
    "successors",
    "to_directed",
    "to_directed_class",
    "to_undirected",
    "to_undirected_class",
    "update",
]

Hash = int


def deterministic_hash(obj):
    bytes = pickle.dumps(obj)
    return zlib.adler32(bytes)


class Node(PClass):
    name = field(type=str, mandatory=True)

    def __hash__(self):
        return deterministic_hash(self)


class MultiDiGraph(PClass):
    _node = pmap_field(Node, PMap)
    _succ = pmap_field(Node, PMap)
    _pred = pmap_field(Node, PMap)

    @property
    def _adj(self):
        return self._succ

    def __len__(self):
        return len(self._node)

    def __iter__(self):
        return iter(self._node)

    def __getitem__(self, node):
        return self.adj()[node]

    def __contains__(self, node):
        return node in self._node

    def is_directed(self) -> bool:
        return True

    def is_multigraph(self) -> bool:
        return True

    def add_node(self, node, **kwargs):
        _node = self._node.set(node, pmap(kwargs))
        _pred = self._pred.set(node, pmap())
        _succ = self._succ.set(node, pmap())
        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def add_edge(self, source: Node, sink: Node, **kwargs):

        predecessors = self._pred.get(sink, pmap())
        edges = predecessors.get(source, pvector())
        edges = edges.append(kwargs)
        predecessors = predecessors.set(source, edges)
        _pred = self._pred.set(sink, predecessors)

        successors = self._succ.get(source, pmap())
        edges = successors.get(sink, pvector())
        edges = edges.append(kwargs)
        successors = successors.set(sink, edges)
        _succ = self._succ.set(source, successors)

        new_graph = self.set(_pred=_pred, _succ=_succ)
        return new_graph

    def merge(self, other: "MultiDiGraph"):
        _node = self._node.update(other._node)

        def merge_edges(node_to_neighbors, other_node_to_neighbors):
            for other_node, other_neighbors in other_node_to_neighbors.items():
                if other_node not in node_to_neighbors:
                    node_to_neighbors = node_to_neighbors.set(other_node, other_neighbors)
                    continue

                neighbors = node_to_neighbors[other_node]
                for other_neighbor, other_edges in other_neighbors.items():
                    if other_neighbor not in neighbors:
                        neighbors = neighbors.set(other_neighbor, other_edges)
                        continue

                    edges = neighbors[other_neighbor].extend(other_neighbors[other_neighbor])
                    neighbors = neighbors.set(other_neighbor, edges)

                node_to_neighbors = node_to_neighbors.set(other_node, neighbors)
            return node_to_neighbors

        _pred = merge_edges(self._pred, other._pred)
        _succ = merge_edges(self._succ, other._succ)

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def nodes(self, data=False):
        if data:
            return NodeView(self)
        else:
            return NodeDataView(self, default=pmap)

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:
            return iter(self)
        return (node for node in nbunch if node in self)

    def adj(self):
        return AdjacencyView(self._adj)

    def neighbors(self, node):
        return [neighbor for neighbor in self.adj()[node]]

    def in_degree(self):
        result = {}
        for node in self.nodes():
            result[node] = len(self._pred[node])
        return InDegreeView(self, result)

    def predecessors(self, node):
        return iter(self._pred[node].keys())

    def get_node_attribute(self, node, name):
        return self._node[node][name]


def topological_traversal(graph):
    return networkx.topological_sort(graph)


def default_visualize_node(graph, node):
    return f"{node}"


def default_visualize_edge(graph, source, sink, edge):
    return ""


def visualize_graph(
    graph: MultiDiGraph, visualize_node=default_visualize_node, visualize_edge=default_visualize_edge
) -> None:
    dot = graphviz.Digraph()

    for node in graph.nodes():
        dot.node(node.name, visualize_node(graph, node))

        for successor in graph[node]:
            for edge in graph[node][successor]:
                dot.edge(node.name, successor.name, label=visualize_edge(graph, successor, node, edge), color="red")

        for predecessor in graph.predecessors(node):
            for edge in graph._pred[node][predecessor]:
                dot.edge(predecessor.name, node.name, label=visualize_edge(graph, node, predecessor, edge))

    dot.render("graph.gv", view=True, format="svg")