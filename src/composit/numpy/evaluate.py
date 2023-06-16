from __future__ import annotations

import numpy as np

from composit.numpy.core import get_operands
from composit.multidigraph import topological_traversal, compose_all
from composit.types import LazyTensor


def update_cache(cache, node, node_output):
    if np.isscalar(node_output):
        node_output = np.asarray(node_output)

    if isinstance(node_output, np.ndarray):
        cache[(node, 0)] = node_output
    elif isinstance(node_output, list):
        for output_index, node_output in enumerate(node_output):
            cache[(node, output_index)] = node_output
    else:
        raise RuntimeError("Unsupported type")
    return cache


def evaluate(*outputs: tuple[LazyTensor]):
    graph = compose_all(*tuple(output.graph for output in outputs))

    cache = {}
    for node in topological_traversal(graph):
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]
        node_output = instruction(*input_arrays)
        cache = update_cache(cache, node, node_output)

    result = [cache[(output.node, output.output_index)] for output in outputs]
    if len(result) == 1:
        return result[0]
    return result


__all__ = ["evaluate"]
