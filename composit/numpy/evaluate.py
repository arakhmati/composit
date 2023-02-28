from __future__ import annotations

import numpy as np

from composit.numpy.core import get_operands
from composit.multidigraph import topological_traversal, compose_all
from composit.persistent_array import PersistentArray


def evaluate(*outputs: tuple[PersistentArray]):
    graph = compose_all(*tuple(output.graph for output in outputs))

    cache = {}
    for node in topological_traversal(graph):
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]
        instruction_output = instruction(*input_arrays)

        if np.isscalar(instruction_output):
            instruction_output = np.asarray(instruction_output)

        if isinstance(instruction_output, np.ndarray):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, list):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError("Unsupported type")

    result = [cache[(output.node, output.output_index)] for output in outputs]
    if len(result) == 1:
        return result[0]
    return result


__all__ = ["evaluate"]
