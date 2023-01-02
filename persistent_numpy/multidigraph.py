from collections.abc import Iterable
import pickle
import zlib


import graphviz
import networkx
from pyrsistent import PClass, field, pmap_field, PMap, pmap

from networkx.classes.coreviews import MultiAdjacencyView
from networkx.classes.reportviews import (
    DiMultiDegreeView,
    InMultiDegreeView,
    OutMultiDegreeView,
    NodeView,
    InMultiEdgeView,
    OutMultiEdgeView,
)

# Fields in networkx.MultiDiGraph
# A commented out line means that the function is implemented
[
    # "_adj",
    # "_node",
    # "_pred",
    # "_succ",
    # "add_edge",
    "add_edges_from",
    # "add_node",
    "add_nodes_from",
    "add_weighted_edges_from",
    # "adj",
    "adjacency",
    "adjlist_inner_dict_factory",
    "adjlist_outer_dict_factory",
    "clear",
    "clear_edges",
    "copy",
    # "degree",
    "edge_attr_dict_factory",
    "edge_key_dict_factory",
    "edge_subgraph",
    # "edges",
    "get_edge_data",
    # "graph",
    "graph_attr_dict_factory",
    "has_edge",
    # "has_node",
    # "has_predecessor",
    # "has_successor",
    # "in_degree",
    # "in_edges",
    # "is_directed",
    # "is_multigraph",
    "name",
    # "nbunch_iter",
    # "neighbors",
    "new_edge_key",
    "node_attr_dict_factory",
    "node_dict_factory",
    # "nodes",
    # "number_of_edges",
    "number_of_nodes",
    "order",
    # "out_degree",
    # "out_edges",
    # "pred",
    # "predecessors",
    # "remove_edge",
    "remove_edges_from",
    "remove_node",
    "remove_nodes_from",
    # "reverse",
    # "size",
    "subgraph",
    # "succ",
    # "successors",
    "to_directed",
    "to_directed_class",
    # "to_undirected",
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
    _node = pmap_field(object, PMap)
    _succ = pmap_field(object, PMap)
    _pred = pmap_field(object, PMap)
    _attributes = field(PMap, initial=pmap)

    @property
    def _adj(self):
        return self._succ

    @property
    def adj(self):
        return MultiAdjacencyView(self._adj)

    @property
    def succ(self):
        return self._succ

    @property
    def pred(self):
        return self._pred

    @property
    def graph(self):
        return self._attributes

    def __len__(self):
        return len(self._node)

    def __iter__(self):
        return iter(self._node)

    def __getitem__(self, node):
        return self.adj[node]

    def __contains__(self, node):
        return node in self._node

    def is_directed(self) -> bool:
        return True

    def is_multigraph(self) -> bool:
        return True

    def add_attributes(self, **kwargs) -> "MultiDiGraph":
        _attributes = self._attributes.update(kwargs)
        return self.set(_attributes=_attributes)

    def has_node(self, node):
        return node in self

    def add_node(self, node, **kwargs):
        _node = self._node.set(node, pmap(kwargs))
        _pred = self._pred.set(node, pmap())
        _succ = self._succ.set(node, pmap())
        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def add_edge(self, source, sink, key=None, **kwargs):

        _node = self._node
        if source not in self:
            _node = _node.set(source, pmap(kwargs))
        if sink not in self:
            _node = _node.set(sink, pmap(kwargs))

        def _add_edge(node_to_neighbors, from_node, to_node, edge_key):
            neighbors = node_to_neighbors.get(from_node, pmap())
            edges = neighbors.get(to_node, pmap())
            if edge_key is None:
                edge_key = max(edges.keys()) + 1 if edges.keys() else 0
            edges = edges.set(edge_key, pmap(kwargs))
            neighbors = neighbors.set(to_node, edges)
            node_to_neighbors = node_to_neighbors.set(from_node, neighbors)
            return node_to_neighbors

        _pred = _add_edge(self._pred, sink, source, key)
        _succ = _add_edge(self._succ, source, sink, key)

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph

    def remove_edge(self, source, sink, key=None):
        def _remove_edge(node_to_neighbors, from_node, to_node, edge_key):
            neighbors = node_to_neighbors.get(from_node, pmap())
            edges = neighbors.get(to_node, pmap())
            if not edges:
                raise networkx.NetworkXError("There is no edge to remove!")
            elif edge_key is None:
                edge_key = max(edges.keys())
            elif edge_key not in edges:
                raise networkx.NetworkXError("There is no edge to remove!")
            edges = edges.remove(edge_key)
            neighbors = neighbors.set(to_node, edges)
            node_to_neighbors = node_to_neighbors.set(from_node, neighbors)
            return node_to_neighbors

        _pred = _remove_edge(self._pred, sink, source, key)
        _succ = _remove_edge(self._succ, source, sink, key)

        new_graph = self.set(_pred=_pred, _succ=_succ)
        return new_graph

    @property
    def nodes(self, data=False, default=None):
        return NodeView(self)(data=data, default=default)

    def edges(self, nbunch=None, data=False, keys=False, default=None):
        if nbunch is not None and not isinstance(nbunch, Iterable):
            if nbunch not in self:
                raise KeyError(f"{nbunch} is not a node in the graph")
        return OutMultiEdgeView(self)(data=data, nbunch=nbunch, keys=keys, default=default)

    out_edges = edges

    def in_edges(self, nbunch=None, data=False, keys=False, default=None):
        if nbunch is not None and not isinstance(nbunch, Iterable):
            if nbunch not in self:
                raise KeyError(f"{nbunch} is not a node in the graph")
        return InMultiEdgeView(self)(data=data, nbunch=nbunch, keys=keys, default=default)

    def nbunch_iter(self, nbunch=None):
        if nbunch is None:
            return iter(self)
        if not isinstance(nbunch, Iterable):
            nbunch = [nbunch]
        return (node for node in nbunch if node in self)

    def neighbors(self, node):
        return [neighbor for neighbor in self.adj[node]]

    def degree(self, nbunch=None, weight=None):
        return DiMultiDegreeView(self)(nbunch, weight)

    def in_degree(self, nbunch=None, weight=None):
        return InMultiDegreeView(self)(nbunch, weight)

    def out_degree(self, nbunch=None, weight=None):
        return OutMultiDegreeView(self)(nbunch, weight)

    size = networkx.MultiDiGraph.size
    number_of_edges = networkx.MultiDiGraph.number_of_edges

    def has_successor(self, node, successor):
        return successor in self._succ[node]

    def successors(self, node):
        return iter(self._succ[node].keys())

    def has_predecessor(self, node, predecessor):
        return predecessor in self._pred[node]

    def predecessors(self, node):
        return iter(self.pred[node].keys())

    def to_undirected(self, **kwargs) -> "networkx.MultiGraph":
        return self.to_networkx().to_undirected(**kwargs)

    def reverse(self, **kwargs) -> "networkx.MultiDiGraph":
        graph = self.to_networkx()
        graph = graph.reverse(**kwargs)
        graph = from_networkx(graph)
        return graph

    # Non-networkx methods

    def to_networkx(self) -> "networkx.MultiDiGraph":
        graph = networkx.MultiDiGraph()

        for node, data in self.nodes(data=True):
            graph.add_node(node, **data)

        for source, sink, key, data in self.edges(keys=True, data=True):
            graph.add_edge(source, sink, key, **data)

        graph.graph.update(**self.graph)
        return graph

    def get_node_attribute(self, node, name):
        return self._node[node][name]

    def merge(self, other: "MultiDiGraph") -> "MultiDiGraph":
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

                    edges = neighbors[other_neighbor].update(other_neighbors[other_neighbor])
                    neighbors = neighbors.set(other_neighbor, edges)

                node_to_neighbors = node_to_neighbors.set(other_node, neighbors)
            return node_to_neighbors

        _pred = merge_edges(self._pred, other._pred)
        _succ = merge_edges(self._succ, other._succ)

        new_graph = self.set(_node=_node, _pred=_pred, _succ=_succ)
        return new_graph


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


def from_networkx(nx_graph: networkx.MultiDiGraph) -> MultiDiGraph:
    graph = MultiDiGraph()

    for node, data in nx_graph.nodes(data=True):
        graph = graph.add_node(node, **data)

    for source, sink, key, data in nx_graph.edges(keys=True, data=True):
        graph = graph.add_edge(source, sink, key, **data)

    graph = graph.add_attributes(**nx_graph.graph)

    return graph
