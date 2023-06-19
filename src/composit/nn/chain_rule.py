from __future__ import annotations

import networkx as nx
from pyrsistent import pset

import composit as cnp
from composit.introspection import class_name
from composit.numpy.core import get_operands
from composit.multidigraph import topological_traversal, compose_all
from composit.types import LazyTensor
import composit.nn.jacobians as jacobians


def get_incoming_gradients(graph, node, node_to_incoming_gradients):
    incoming_gradients = node_to_incoming_gradients[node]
    for index, incoming_gradient in enumerate(incoming_gradients):
        if incoming_gradient is not None:
            continue
        shape = graph.nodes[node]["shapes"][index]
        incoming_gradients[index] = cnp.zeros(shape)
    return incoming_gradients


def get_subgraph(graph, input_nodes, output_nodes):
    input_nodes = pset(input_nodes)
    for input_node in input_nodes:
        input_nodes = input_nodes.union(nx.descendants(graph, input_node))

    output_nodes = pset(output_nodes)
    for output_node in output_nodes:
        output_nodes = output_nodes.union(nx.ancestors(graph, output_node))

    subgraph_nodes = input_nodes.intersection(output_nodes)
    return graph.subgraph(subgraph_nodes)


def chain_rule(output_vars_with_incoming_gradients, input_vars_to_differentiate):
    output_vars = tuple(output_var for output_var in output_vars_with_incoming_gradients)
    forward_graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    input_nodes = pset([input_var.node for input_var in input_vars_to_differentiate])
    output_nodes = pset([output_var.node for output_var in output_vars])
    forward_subgraph = get_subgraph(forward_graph, input_nodes, output_nodes)
    reversed_forward_subgraph = forward_subgraph.reverse()

    sorted_nodes = topological_traversal(reversed_forward_subgraph)

    node_to_incoming_gradients = {}

    for output_node in output_nodes:
        node_to_incoming_gradients[output_node] = [None for _ in range(len(forward_graph.nodes[output_node]["shapes"]))]

    for output_var, incoming_gradient in output_vars_with_incoming_gradients.items():
        node_to_incoming_gradients[output_var.node][output_var.output_index] = incoming_gradient

    for node in sorted_nodes:
        if reversed_forward_subgraph.out_degree(node) == 0:
            continue

        forward_operation = forward_graph.nodes[node]["operation"]
        forward_operands = tuple(operand for operand in get_operands(forward_graph, node))
        forward_input_vars = tuple(
            LazyTensor(
                graph=forward_graph,
                node=operand_node,
                output_index=operand_output_index,
            )
            for (operand_node, operand_output_index) in forward_operands
        )

        create_outgoing_gradients = getattr(jacobians, f"{class_name(forward_operation)}_jacobian")
        incoming_gradients = get_incoming_gradients(forward_subgraph, node, node_to_incoming_gradients)
        outgoing_gradients = create_outgoing_gradients(forward_operation, incoming_gradients, forward_input_vars)

        for (operand_node, output_index), outgoing_gradient in zip(forward_operands, outgoing_gradients):
            if operand_node in node_to_incoming_gradients:
                outgoing_gradient_accumulator = node_to_incoming_gradients[operand_node][output_index]
                if outgoing_gradient_accumulator is not None:
                    outgoing_gradient += outgoing_gradient_accumulator
            else:
                node_to_incoming_gradients[operand_node] = [None] * len(forward_graph.nodes[operand_node]["shapes"])
            node_to_incoming_gradients[operand_node][output_index] = outgoing_gradient

    # Input vars always have only 1 output
    result = [node_to_incoming_gradients[input_var.node][0] for input_var in input_vars_to_differentiate]
    return result
