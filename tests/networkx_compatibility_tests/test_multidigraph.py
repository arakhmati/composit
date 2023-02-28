from collections import UserDict

import networkx
import pytest

import networkx as nx
from networkx.utils import edges_equal

from .test_multigraph import BaseMultiGraphTester
from .test_multigraph import TestEdgeSubgraph as _TestMultiGraphEdgeSubgraph
from .test_multigraph import TestMultiGraph as _TestMultiGraph

from composit import multidigraph


def networkx_graph(graph):
    return graph


def to_nx_edges(edges):
    return sorted([(source, sink, {key: value for key, value in iterator.items()}) for source, sink, iterator in edges])


class BaseMultiDiGraphTester(BaseMultiGraphTester):
    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_edges(self, graph_function):
        G = graph_function(self.K3)
        edges = [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges()) == edges
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        pytest.raises((KeyError, nx.NetworkXError), G.edges, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_edges_data(self, graph_function):
        G = graph_function(self.K3)
        edges = [(0, 1, {}), (0, 2, {}), (1, 0, {}), (1, 2, {}), (2, 0, {}), (2, 1, {})]
        assert to_nx_edges(G.edges(data=True)) == edges
        assert to_nx_edges(G.edges(0, data=True)) == [(0, 1, {}), (0, 2, {})]
        pytest.raises((KeyError, nx.NetworkXError), G.neighbors, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_edges_multi(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.edges(0)) == [(0, 1), (0, 2)]
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1)
        else:
            G = G.add_edge(0, 1)
        assert sorted(G.edges()) == [
            (0, 1),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_out_edges(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        pytest.raises((KeyError, nx.NetworkXError), G.out_edges, -1)
        assert sorted(G.out_edges(0, keys=True)) == [(0, 1, 0), (0, 2, 0)]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_out_edges_multi(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.out_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.out_edges(0)) == [(0, 1), (0, 2)]
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1, 2)
        else:
            G = G.add_edge(0, 1, 2)
        assert sorted(G.out_edges()) == [
            (0, 1),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_out_edges_data(self, graph_function):
        G = graph_function(self.K3)
        assert to_nx_edges(G.edges(0, data=True)) == [(0, 1, {}), (0, 2, {})]
        if isinstance(G, nx.MultiDiGraph):
            G.remove_edge(0, 1)
            G.add_edge(0, 1, data=1)
        else:
            G = G.remove_edge(0, 1)
            G = G.add_edge(0, 1, data=1)
        assert to_nx_edges(G.edges(0, data=True)) == [(0, 1, {"data": 1}), (0, 2, {})]
        assert sorted(G.edges(0, data="data")) == [(0, 1, 1), (0, 2, None)]
        assert sorted(G.edges(0, data="data", default=-1)) == [(0, 1, 1), (0, 2, -1)]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_in_edges(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.in_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(0)) == [(1, 0), (2, 0)]
        pytest.raises((KeyError, nx.NetworkXError), G.in_edges, -1)
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1, 2)
        else:
            G = G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == [
            (0, 1),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]
        assert sorted(G.in_edges(0, keys=True)) == [(1, 0, 0), (2, 0, 0)]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_in_edges_no_keys(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.in_edges()) == [(0, 1), (0, 2), (1, 0), (1, 2), (2, 0), (2, 1)]
        assert sorted(G.in_edges(0)) == [(1, 0), (2, 0)]
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1, 2)
        else:
            G = G.add_edge(0, 1, 2)
        assert sorted(G.in_edges()) == [
            (0, 1),
            (0, 1),
            (0, 2),
            (1, 0),
            (1, 2),
            (2, 0),
            (2, 1),
        ]

        assert to_nx_edges(G.in_edges(data=True, keys=False)) == [
            (0, 1, {}),
            (0, 1, {}),
            (0, 2, {}),
            (1, 0, {}),
            (1, 2, {}),
            (2, 0, {}),
            (2, 1, {}),
        ]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_in_edges_data(self, graph_function):
        G = graph_function(self.K3)
        assert to_nx_edges(G.in_edges(0, data=True)) == [(1, 0, {}), (2, 0, {})]
        if isinstance(G, nx.MultiDiGraph):
            G.remove_edge(1, 0)
            G.add_edge(1, 0, data=1)
        else:
            G = G.remove_edge(1, 0)
            G = G.add_edge(1, 0, data=1)
        assert to_nx_edges(G.in_edges(0, data=True)) == [(1, 0, {"data": 1}), (2, 0, {})]
        assert sorted(G.in_edges(0, data="data")) == [(1, 0, 1), (2, 0, None)]
        assert sorted(G.in_edges(0, data="data", default=-1)) == [(1, 0, 1), (2, 0, -1)]

    def is_shallow(self, H, G):
        # graph
        assert G.graph["foo"] == H.graph["foo"]
        G.graph["foo"].append(1)
        assert G.graph["foo"] == H.graph["foo"]
        # node
        assert G.nodes[0]["foo"] == H.nodes[0]["foo"]
        G.nodes[0]["foo"].append(1)
        assert G.nodes[0]["foo"] == H.nodes[0]["foo"]
        # edge
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]
        G[1][2][0]["foo"].append(1)
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]

    def is_deep(self, H, G):
        # graph
        assert G.graph["foo"] == H.graph["foo"]
        G.graph["foo"].append(1)
        assert G.graph["foo"] != H.graph["foo"]
        # node
        assert G.nodes[0]["foo"] == H.nodes[0]["foo"]
        G.nodes[0]["foo"].append(1)
        assert G.nodes[0]["foo"] != H.nodes[0]["foo"]
        # edge
        assert G[1][2][0]["foo"] == H[1][2][0]["foo"]
        G[1][2][0]["foo"].append(1)
        assert G[1][2][0]["foo"] != H[1][2][0]["foo"]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_to_undirected(self, graph_function):
        # MultiDiGraph -> MultiGraph changes number of edges so it is
        # not a copy operation... use is_shallow, not is_shallow_copy
        G = graph_function(self.K3)
        G = self.add_attributes(G)
        H = nx.MultiGraph(G)
        # self.is_shallow(H,G)
        # the result is traversal order dependent so we
        # can't use the is_shallow() test here.
        try:
            assert edges_equal(H.edges(), [(0, 1), (1, 2), (2, 0)])
        except AssertionError:
            assert edges_equal(H.edges(), [(0, 1), (1, 2), (1, 2), (2, 0)])
        H = G.to_undirected()
        self.is_deep(H, G)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_has_successor(self, graph_function):
        G = graph_function(self.K3)
        assert G.has_successor(0, 1)
        assert not G.has_successor(0, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_successors(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.successors(0)) == [1, 2]
        pytest.raises((KeyError, nx.NetworkXError), G.successors, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_has_predecessor(self, graph_function):
        G = graph_function(self.K3)
        assert G.has_predecessor(0, 1)
        assert not G.has_predecessor(0, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_predecessors(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.predecessors(0)) == [1, 2]
        pytest.raises((KeyError, nx.NetworkXError), G.predecessors, -1)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_degree(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.degree()) == [(0, 4), (1, 4), (2, 4)]
        assert dict(G.degree()) == {0: 4, 1: 4, 2: 4}
        assert G.degree(0) == 4
        assert list(G.degree(iter([0]))) == [(0, 4)]
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1, weight=0.3, other=1.2)
        else:
            G = G.add_edge(0, 1, weight=0.3, other=1.2)
        assert sorted(G.degree(weight="weight")) == [(0, 4.3), (1, 4.3), (2, 4)]
        assert sorted(G.degree(weight="other")) == [(0, 5.2), (1, 5.2), (2, 4)]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_in_degree(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.in_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.in_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.in_degree(0) == 2
        assert list(G.in_degree(iter([0]))) == [(0, 2)]
        assert G.in_degree(0, weight="weight") == 2

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_out_degree(self, graph_function):
        G = graph_function(self.K3)
        assert sorted(G.out_degree()) == [(0, 2), (1, 2), (2, 2)]
        assert dict(G.out_degree()) == {0: 2, 1: 2, 2: 2}
        assert G.out_degree(0) == 2
        assert list(G.out_degree(iter([0]))) == [(0, 2)]
        assert G.out_degree(0, weight="weight") == 2

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_size(self, graph_function):
        G = graph_function(self.K3)
        assert G.size() == 6
        assert G.number_of_edges() == 6
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(0, 1, weight=0.3, other=1.2)
        else:
            G = G.add_edge(0, 1, weight=0.3, other=1.2)
        assert round(G.size(weight="weight"), 2) == 6.3
        assert round(G.size(weight="other"), 2) == 7.2

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_to_undirected_reciprocal(self, graph_function):
        G = graph_function(self.Graph())
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(1, 2)
        else:
            G = G.add_edge(1, 2)
        assert G.to_undirected().has_edge(1, 2)
        assert not G.to_undirected(reciprocal=True).has_edge(1, 2)
        if isinstance(G, nx.MultiDiGraph):
            G.add_edge(2, 1)
        else:
            G = G.add_edge(2, 1)
        assert G.to_undirected(reciprocal=True).has_edge(1, 2)

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_reverse_copy(self, graph_function):
        G = graph_function(nx.MultiDiGraph([(0, 1), (0, 1)]))
        R = G.reverse()
        assert sorted(R.edges()) == [(1, 0), (1, 0)]
        if isinstance(G, nx.MultiDiGraph):
            R.remove_edge(1, 0)
        else:
            R = R.remove_edge(1, 0)
        assert sorted(R.edges()) == [(1, 0)]
        assert sorted(G.edges()) == [(0, 1), (0, 1)]

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_reverse_nocopy(self, graph_function):
        G = graph_function(nx.MultiDiGraph([(0, 1), (0, 1)]))
        R = G.reverse(copy=False)
        assert sorted(R.edges()) == [(1, 0), (1, 0)]
        if isinstance(G, nx.MultiDiGraph):
            pytest.raises(nx.NetworkXError, R.remove_edge, 1, 0)
        else:
            pytest.skip("Persistent graph does not support copy=False")

    @pytest.mark.parametrize("graph_function", [networkx_graph, multidigraph.from_networkx])
    def test_di_attributes_cached(self, graph_function):
        G = graph_function(self.K3.copy())
        if not isinstance(G, networkx.MultiDiGraph):
            pytest.skip("Persistent graph does not caches attributes")
        assert id(G.in_edges) == id(G.in_edges)
        assert id(G.out_edges) == id(G.out_edges)
        assert id(G.in_degree) == id(G.in_degree)
        assert id(G.out_degree) == id(G.out_degree)
        assert id(G.succ) == id(G.succ)
        assert id(G.pred) == id(G.pred)


# TODO: modify tests below to check persistent graph
class TestMultiDiGraph(BaseMultiDiGraphTester, _TestMultiGraph):
    def setup_method(self):
        self.Graph = nx.MultiDiGraph
        # build K3
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = {0: {}, 1: {}, 2: {}}
        # K3._adj is synced with K3._succ
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = {}
        self.K3._node[0] = {}
        self.K3._node[1] = {}
        self.K3._node[2] = {}

    def test_add_edge(self):
        G = self.Graph()
        G.add_edge(0, 1)
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        G = self.Graph()
        G.add_edge(*(0, 1))
        assert G._adj == {0: {1: {0: {}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}}}}
        with pytest.raises(ValueError, match="None cannot be a node"):
            G.add_edge(None, 3)

    def test_add_edges_from(self):
        G = self.Graph()
        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})])
        assert G._adj == {0: {1: {0: {}, 1: {"weight": 3}}}, 1: {}}
        assert G._succ == {0: {1: {0: {}, 1: {"weight": 3}}}, 1: {}}
        assert G._pred == {0: {}, 1: {0: {0: {}, 1: {"weight": 3}}}}

        G.add_edges_from([(0, 1), (0, 1, {"weight": 3})], weight=2)
        assert G._succ == {
            0: {1: {0: {}, 1: {"weight": 3}, 2: {"weight": 2}, 3: {"weight": 3}}},
            1: {},
        }
        assert G._pred == {
            0: {},
            1: {0: {0: {}, 1: {"weight": 3}, 2: {"weight": 2}, 3: {"weight": 3}}},
        }

        G = self.Graph()
        edges = [
            (0, 1, {"weight": 3}),
            (0, 1, (("weight", 2),)),
            (0, 1, 5),
            (0, 1, "s"),
        ]
        G.add_edges_from(edges)
        keydict = {0: {"weight": 3}, 1: {"weight": 2}, 5: {}, "s": {}}
        assert G._succ == {0: {1: keydict}, 1: {}}
        assert G._pred == {1: {0: keydict}, 0: {}}

        # too few in tuple
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0,)])
        # too many in tuple
        pytest.raises(nx.NetworkXError, G.add_edges_from, [(0, 1, 2, 3, 4)])
        # not a tuple
        pytest.raises(TypeError, G.add_edges_from, [0])
        with pytest.raises(ValueError, match="None cannot be a node"):
            G.add_edges_from([(None, 3), (3, 2)])

    def test_remove_edge(self):
        G = self.K3
        G.remove_edge(0, 1)
        assert G._succ == {
            0: {2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        assert G._pred == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, 0, 2, key=1)

    def test_remove_multiedge(self):
        G = self.K3
        G.add_edge(0, 1, key="parallel edge")
        G.remove_edge(0, 1, key="parallel edge")
        assert G._adj == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }

        assert G._succ == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }

        assert G._pred == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        G.remove_edge(0, 1)
        assert G._succ == {
            0: {2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        assert G._pred == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        pytest.raises((KeyError, nx.NetworkXError), G.remove_edge, -1, 0)

    def test_remove_edges_from(self):
        G = self.K3
        G.remove_edges_from([(0, 1)])
        assert G._succ == {
            0: {2: {0: {}}},
            1: {0: {0: {}}, 2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        assert G._pred == {
            0: {1: {0: {}}, 2: {0: {}}},
            1: {2: {0: {}}},
            2: {0: {0: {}}, 1: {0: {}}},
        }
        G.remove_edges_from([(0, 0)])  # silent fail


class TestEdgeSubgraph(_TestMultiGraphEdgeSubgraph):
    """Unit networkx_tests for the :meth:`MultiDiGraph.edge_subgraph` method."""

    def setup_method(self):
        # Create a quadruply-linked path graph on five nodes.
        G = nx.MultiDiGraph()
        nx.add_path(G, range(5))
        nx.add_path(G, range(5))
        nx.add_path(G, reversed(range(5)))
        nx.add_path(G, reversed(range(5)))
        # Add some node, edge, and graph attributes.
        for i in range(5):
            G.nodes[i]["name"] = f"node{i}"
        G.adj[0][1][0]["name"] = "edge010"
        G.adj[0][1][1]["name"] = "edge011"
        G.adj[3][4][0]["name"] = "edge340"
        G.adj[3][4][1]["name"] = "edge341"
        G.graph["name"] = "graph"
        # Get the subgraph induced by one of the first edges and one of
        # the last edges.
        self.G = G
        self.H = G.edge_subgraph([(0, 1, 0), (3, 4, 1)])


class CustomDictClass(UserDict):
    pass


class MultiDiGraphSubClass(nx.MultiDiGraph):
    node_dict_factory = CustomDictClass  # type: ignore
    node_attr_dict_factory = CustomDictClass  # type: ignore
    adjlist_outer_dict_factory = CustomDictClass  # type: ignore
    adjlist_inner_dict_factory = CustomDictClass  # type: ignore
    edge_key_dict_factory = CustomDictClass  # type: ignore
    edge_attr_dict_factory = CustomDictClass  # type: ignore
    graph_attr_dict_factory = CustomDictClass  # type: ignore


class TestMultiDiGraphSubclass(TestMultiDiGraph):
    def setup_method(self):
        self.Graph = MultiDiGraphSubClass
        # build K3
        self.k3edges = [(0, 1), (0, 2), (1, 2)]
        self.k3nodes = [0, 1, 2]
        self.K3 = self.Graph()
        self.K3._succ = self.K3.adjlist_outer_dict_factory(
            {
                0: self.K3.adjlist_inner_dict_factory(),
                1: self.K3.adjlist_inner_dict_factory(),
                2: self.K3.adjlist_inner_dict_factory(),
            }
        )
        # K3._adj is synced with K3._succ
        self.K3._pred = {0: {}, 1: {}, 2: {}}
        for u in self.k3nodes:
            for v in self.k3nodes:
                if u == v:
                    continue
                d = {0: {}}
                self.K3._succ[u][v] = d
                self.K3._pred[v][u] = d
        self.K3._node = self.K3.node_dict_factory()
        self.K3._node[0] = self.K3.node_attr_dict_factory()
        self.K3._node[1] = self.K3.node_attr_dict_factory()
        self.K3._node[2] = self.K3.node_attr_dict_factory()
