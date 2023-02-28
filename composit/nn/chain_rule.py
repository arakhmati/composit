from __future__ import annotations

import networkx as nx
from pyrsistent import pset
from toolz import functoolz

from composit.numpy.core import get_operands
from composit.multidigraph import topological_traversal, compose_all
import composit.nn as nn
import composit.nn.jacobians as jacobians


def get_incoming_gradient_name(node_name, output_index):
    return f"{node_name}_gradient_{output_index}"


def get_incoming_gradients(node, backward_graph, node_to_incoming_gradients):
    if node in node_to_incoming_gradients:
        incoming_gradients = node_to_incoming_gradients[node]
    else:
        incoming_gradients = [
            nn.variable(name=get_incoming_gradient_name(node.name, output_index), shape=shape)
            for output_index, shape in enumerate(backward_graph.nodes[node]["shapes"])
        ]

    assert isinstance(incoming_gradients, list)
    if len(incoming_gradients) == 1:
        (incoming_gradients,) = incoming_gradients
    return incoming_gradients


@functoolz.memoize
def chain_rule(*output_vars, input_vars: list[nn.Variable]):

    forward_graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    input_nodes = [input_var.node for input_var in input_vars]
    nodes_from_inputs = pset(input_nodes)
    for input_node in input_nodes:
        nodes_from_inputs = nodes_from_inputs.union(nx.descendants(forward_graph, input_node))

    output_nodes = [output_var.node for output_var in output_vars]
    nodes_from_output_vars = pset(output_nodes)
    for output_node in output_nodes:
        nodes_from_output_vars = nodes_from_output_vars.union(nx.ancestors(forward_graph, output_node))

    chain_rule_nodes = nodes_from_inputs.intersection(nodes_from_output_vars)
    reversed_forward_subgraph = forward_graph.subgraph(chain_rule_nodes).reverse()

    sorted_nodes = topological_traversal(reversed_forward_subgraph)

    node_to_incoming_gradients = {}
    for node in sorted_nodes:
        if reversed_forward_subgraph.out_degree(node) == 0:
            continue

        forward_instruction = forward_graph.nodes[node]["instruction"]
        forward_operands = tuple(in_edge for in_edge in get_operands(forward_graph, node))
        forward_input_vars = tuple(
            nn.variable(name=node.name, shape=forward_graph.nodes[node]["shapes"][output_index])
            for (node, output_index) in forward_operands
        )

        create_outgoing_gradients = getattr(jacobians, f"{forward_instruction.__class__.__name__}_jacobian")
        incoming_gradients = get_incoming_gradients(node, reversed_forward_subgraph, node_to_incoming_gradients)
        outgoing_gradients = create_outgoing_gradients(forward_instruction, incoming_gradients, forward_input_vars)

        for (operand_node, output_index), outgoing_gradient in zip(forward_operands, outgoing_gradients):
            if operand_node in node_to_incoming_gradients:
                outgoing_gradient_accumulator = node_to_incoming_gradients[operand_node][output_index]
                if outgoing_gradient_accumulator is not None:
                    outgoing_gradient += outgoing_gradient_accumulator
            else:
                node_to_incoming_gradients[operand_node] = [None] * len(forward_graph.nodes[operand_node]["shapes"])
            node_to_incoming_gradients[operand_node][output_index] = outgoing_gradient

    # Input vars always have only 1 output
    result = [node_to_incoming_gradients[input_var.node][0] for input_var in input_vars]
    return result
