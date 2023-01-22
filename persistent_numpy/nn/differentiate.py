import numpy as np

import persistent_numpy.nn as nn


def initialize_cache(graph, inputs):
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


def differentiate(output_vars, input_vars_to_differentiate, inputs, incoming_gradients):
    gradient_vars = nn.chain_rule(*output_vars, input_vars=input_vars_to_differentiate)

    # TODO: evaluate calls below can be combined into a single function once the graphs can be merged together
    inputs = {input.node.name: array for input, array in inputs.items()}
    _, forward_cache = nn.evaluate(
        *output_vars,
        inputs=inputs,
        return_cache=True,
    )

    incoming_gradients = {
        f"{incoming_gradient.name}_gradient": array for incoming_gradient, array in incoming_gradients.items()
    }
    outgoing_gradients = nn.evaluate(
        *gradient_vars,
        inputs=dict(**incoming_gradients, **forward_cache.as_dict()),
        initialize_cache_function=initialize_cache,
        always_return_tuple=True,
    )
    return outgoing_gradients
