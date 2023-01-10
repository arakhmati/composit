import sys

import networkx as nx
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

    gradients = {}
    for node in sorted_nodes:
        if backward_graph.out_degree(node) == 0:
            continue

        forward_instruction = forward_graph.nodes[node]["instruction"]
        forward_operands = (in_edge for in_edge in pnp.get_operands(forward_graph, node))
        forward_input_vars = tuple(
            nn.variable(name=node.name, shape=forward_graph.nodes[node]["shapes"][output_index])
            for (node, output_index) in forward_operands
        )
        if node in gradients:
            output_gradient = gradients[node]
        else:
            output_gradient = nn.variable(name=f"{node.name}_gradient", shape=backward_graph.nodes[node]["shapes"][0])

        create_input_gradients = getattr(jacobians, f"{forward_instruction.__class__.__name__}_jacobian")
        forward_input_gradients = create_input_gradients(forward_instruction, output_gradient, forward_input_vars)
        for forward_input_var, forward_input_gradient in zip(forward_input_vars, forward_input_gradients):
            if forward_input_var.node in gradients:
                gradient_accumulator = gradients[forward_input_var.node]
                forward_input_gradient = gradient_accumulator + forward_input_gradient
            gradients[forward_input_var.node] = forward_input_gradient

    result = [gradients[input_var.node] for input_var in input_vars]
    return result


def compute_gradients(output_vars, input_vars_to_differentiate, inputs, output_gradients):
    inputs = {input.node.name: array for input, array in inputs.items()}
    output_gradients = {
        f"{output_gradient.node.name}_gradient": array for output_gradient, array in output_gradients.items()
    }
    gradient_vars = pnp.nn.chain_rule(*output_vars, input_vars=input_vars_to_differentiate)
    # TODO: evaluate calls below can be combined into a single function once the graphs can be merged together
    _, forward_cache = pnp.nn.evaluate(
        *output_vars,
        inputs=inputs,
        return_cache=True,
    )
    gradients = pnp.nn.evaluate(*gradient_vars, inputs=dict(**output_gradients, **forward_cache.as_dict()))
    return gradients
