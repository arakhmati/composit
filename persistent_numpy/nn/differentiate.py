import numpy as np

import persistent_numpy.nn as nn
from persistent_numpy.nn.chain_rule import get_incoming_gradient_name


def initialize_cache(graph, inputs):
    cache = {}
    for parray, array in inputs.items():
        cache[(parray.node, 0)] = array

    for node in graph:
        if (node, 0) in cache:
            continue

        instruction = graph.nodes[node]["instruction"]
        if isinstance(instruction, nn.Variable):
            # If gradient array does not exist, then it's initialized to 0
            shapes = graph.nodes[node]["shapes"]
            assert len(shapes) == 1
            shape = shapes[0]
            cache[(node, 0)] = np.zeros(shape)

    return cache


def differentiate(output_vars, input_vars_to_differentiate, inputs, incoming_gradients):
    gradient_vars = nn.chain_rule(*output_vars, input_vars=tuple(input_vars_to_differentiate))

    # TODO: evaluate calls below can be combined into a single function once the graphs can be merged together
    _, forward_cache = nn.evaluate(
        *output_vars,
        inputs=inputs,
        return_cache=True,
    )

    incoming_gradients = {
        nn.variable(
            name=get_incoming_gradient_name(incoming_gradient.node.name, incoming_gradient.output_index),
            shape=array.shape,
        ): array
        for incoming_gradient, array in incoming_gradients.items()
    }

    outgoing_gradients = nn.evaluate(
        *gradient_vars,
        inputs={**incoming_gradients, **forward_cache.as_dict()},
        initialize_cache_function=initialize_cache,
        always_return_tuple=True,
    )

    outgoing_gradients = {
        input_var: gradient for input_var, gradient in zip(input_vars_to_differentiate, outgoing_gradients)
    }
    return outgoing_gradients
