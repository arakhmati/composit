from __future__ import annotations

import numpy as np
from pyrsistent import PClass, pmap_field, pmap

from composit.multidigraph import topological_traversal, compose_all
from composit.numpy.core import get_operands
from composit.types import LazyTensor


class Cache(PClass):
    node_output_to_array = pmap_field(key_type=tuple, value_type=np.ndarray)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_array=pmap(dictionary))

    def __getitem__(self, lazy_tensor: LazyTensor):
        node = lazy_tensor.node
        output_index = lazy_tensor.output_index
        return self.node_output_to_array[(node, output_index)]


def update_cache(cache, node, node_output):
    if np.isscalar(node_output):
        node_output = np.asarray(node_output)

    if isinstance(node_output, np.ndarray):
        cache[(node, 0)] = node_output
    elif isinstance(node_output, (list, tuple)):
        for output_index, node_output in enumerate(node_output):
            cache[(node, output_index)] = node_output
    else:
        raise RuntimeError(f"Unsupported type {type(node_output)}")
    return cache


def evaluate(
    *output_vars,
    return_cache: bool = False,
    always_return_tuple: bool = False,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    cache = {}
    for node in topological_traversal(graph):
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]
        node_output = instruction(*input_arrays)
        cache = update_cache(cache, node, node_output)

    cache = Cache.from_dict(cache)

    result = [cache[output_var] for output_var in output_vars]
    if len(result) == 1 and not always_return_tuple:
        (result,) = result

    if return_cache:
        return result, cache
    else:
        return result


__all__ = [
    "Cache",
    "evaluate",
]
