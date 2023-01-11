import sys

import networkx as nx
import numpy as np
from pyrsistent import pset

import persistent_numpy as pnp
from persistent_numpy.multidigraph import topological_traversal, compose_all
import persistent_numpy.nn as nn
import persistent_numpy.nn.jacobians as jacobians

THIS_MODULE = sys.modules[__name__]


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
    backward_graph = forward_graph.subgraph(chain_rule_nodes).reverse()

    sorted_nodes = topological_traversal(backward_graph)

    node_to_gradients = {}
    for node in sorted_nodes:
        if backward_graph.out_degree(node) == 0:
            continue

        forward_instruction = forward_graph.nodes[node]["instruction"]
        forward_operands = (in_edge for in_edge in pnp.get_operands(forward_graph, node))
        forward_input_vars = tuple(
            nn.variable(name=node.name, shape=forward_graph.nodes[node]["shapes"][output_index])
            for (node, output_index) in forward_operands
        )
        if node in node_to_gradients:
            incoming_gradients = node_to_gradients[node]
        else:
            incoming_gradients = [
                nn.variable(name=f"{node.name}_{output_index}_gradient", shape=shape)
                for output_index, shape in enumerate(backward_graph.nodes[node]["shapes"])
            ]
            if len(incoming_gradients) == 1:
                (incoming_gradients,) = incoming_gradients

        create_outgoing_gradients = getattr(jacobians, f"{forward_instruction.__class__.__name__}_jacobian")
        outgoing_gradients = create_outgoing_gradients(forward_instruction, incoming_gradients, forward_input_vars)

        for forward_input_var, outgoing_gradient in zip(forward_input_vars, outgoing_gradients):
            if forward_input_var.node in node_to_gradients:
                outgoing_gradient_accumulator = node_to_gradients[forward_input_var.node]
                outgoing_gradient = outgoing_gradient_accumulator + outgoing_gradient
            node_to_gradients[forward_input_var.node] = outgoing_gradient

    result = [node_to_gradients[input_var.node] for input_var in input_vars]
    return result


def initialize_backward_cache(graph, inputs):
    cache = {}
    for node in graph:
        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, nn.Variable):
            if node.name in inputs:
                cache[(node, 0)] = inputs[node.name]
            else:
                # If gradient array does not exist, then it's initialized to 0
                shapes = graph.nodes[node]["shapes"]
                assert len(shapes) == 1
                shape = shapes[0]
                cache[(node, 0)] = np.zeros(shape)
    return cache


def compute_gradients(output_vars, input_vars_to_differentiate, inputs, incoming_gradients):
    inputs = {input.node.name: array for input, array in inputs.items()}
    gradient_vars = pnp.nn.chain_rule(*output_vars, input_vars=input_vars_to_differentiate)
    # TODO: evaluate calls below can be combined into a single function once the graphs can be merged together
    _, forward_cache = pnp.nn.evaluate(
        *output_vars,
        inputs=inputs,
        return_cache=True,
    )

    incoming_gradients = {
        f"{incoming_gradient.name}_gradient": array for incoming_gradient, array in incoming_gradients.items()
    }
    outgoing_gradients = pnp.nn.evaluate(
        *gradient_vars,
        inputs=dict(**incoming_gradients, **forward_cache.as_dict()),
        initialize_cache_function=initialize_backward_cache,
    )
    return outgoing_gradients
