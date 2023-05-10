from pyrsistent import PClass

from composit.introspection import class_name
from composit.multidigraph import topological_traversal, MultiDiGraph
from composit.persistent_array import Node


class Tilize(PClass):
    def __call__(self, input_tensor):
        return input_tensor


class Untilize(PClass):
    def __call__(self, input_tensor):
        return input_tensor


def insert_tilize_and_untilize_instructions(graph):
    new_graph = MultiDiGraph()
    operand_to_new_operand = {}
    for node in topological_traversal(graph):
        attributes = graph.nodes[node]

        new_graph = new_graph.add_node(node, **attributes)
        operand_to_new_operand[(node, 0)] = (node, 0)

        if class_name(graph.nodes[node]["instruction"]) == "Variable":
            tilize_node = Node(name=f"tilize_{node.name}")
            new_graph = new_graph.add_node(
                tilize_node,
                instruction=Tilize(),
                shapes=attributes["shapes"],
                dtypes=attributes["dtypes"],
                tile_configs=attributes["tile_configs"],
            )
            new_graph = new_graph.add_edge(node, tilize_node, source_output_index=0, sink_input_index=0)
            operand_to_new_operand[(node, 0)] = (tilize_node, 0)
        elif graph.out_degree(node) == 0:
            untilize_node = Node(name=f"untilize_{node.name}")
            new_graph = new_graph.add_node(
                untilize_node,
                instruction=Untilize(),
                shapes=attributes["shapes"],
                dtypes=attributes["dtypes"],
                tile_configs=attributes["tile_configs"],
            )
            new_graph = new_graph.add_edge(node, untilize_node, source_output_index=0, sink_input_index=0)

        for source, sink, edge_attributes in graph.in_edges(node, data=True):
            operand = (source, edge_attributes["source_output_index"])
            (new_source, new_source_output_index) = operand_to_new_operand[operand]
            new_graph = new_graph.add_edge(
                new_source,
                node,
                source_output_index=new_source_output_index,
                sink_input_index=edge_attributes["sink_input_index"],
            )

    return new_graph
