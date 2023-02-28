from __future__ import annotations

import collections

import numpy as np
from pyrsistent import PClass, pmap_field, pmap

from composit.multidigraph import topological_traversal, compose_all
from composit import nn
from composit.numpy.core import get_operands
from composit.persistent_array import PersistentArray


class Cache(PClass):
    node_output_to_array = pmap_field(key_type=tuple, value_type=np.ndarray)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_array=pmap(dictionary))

    def __getitem__(self, persistent_array: PersistentArray):
        node = persistent_array.node
        output_index = persistent_array.output_index
        return self.node_output_to_array[(node, output_index)]

    def as_dict_from_variable_to_array(self):
        return {
            nn.variable(name=node.name, shape=array.shape): array
            for (node, _), array in self.node_output_to_array.items()
        }


def initialize_cache(graph, inputs):
    cache = {}
    for parray, array in inputs.items():
        cache[(parray.node, parray.output_index)] = array
    return cache


def evaluate(
    *output_vars,
    inputs: dict[PersistentArray, np.ndarray],
    initialize_cache_function=initialize_cache,
    return_cache: bool = False,
    always_return_tuple: bool = False,
):

    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    cache = initialize_cache_function(graph, inputs)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]
        instruction_output = instruction(*input_arrays)

        if np.isscalar(instruction_output):
            instruction_output = np.asarray(instruction_output)

        if isinstance(instruction_output, np.ndarray):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, collections.abc.Iterable):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError(f"Unsupported type: {type(instruction_output)}")

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
