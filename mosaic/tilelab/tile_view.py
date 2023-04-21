from __future__ import annotations

import collections
import math

from pyrsistent import PClass, field, pmap, pmap_field, PVector

import composit as cnp
from composit.persistent_array import PersistentArray
from composit.multidigraph import compose_all, topological_traversal
from composit.numpy.core import get_operands


class TileLevel(PClass):
    level_name = field()
    tile_shape = field()


class TileView(PClass):
    shape = field()
    hierarchy: PVector[TileLevel] = field()

    @property
    def num_levels(self):
        return len(self.hierarchy)

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, hierarchy={self.hierarchy})"

    def __add__(self, other):
        return binary_operation(self, other)

    def __sub__(self, other):
        return binary_operation(self, other)

    def __matmul__(self, other):
        shape = self.shape[:-1] + other.shape[-1:]
        hierarchy = []
        for a_level, b_level in zip(self.hierarchy, other.hierarchy):
            hierarchy.append(
                TileLevel(
                    level_name=a_level.level_name,
                    tile_shape=a_level.tile_shape[:-1] + b_level.tile_shape[-1:],
                )
            )

        return TileView(
            shape=shape,
            hierarchy=hierarchy,
        )


def binary_operation(view_a: TileView, _: TileView) -> TileView:
    return TileView(shape=view_a.shape, hierarchy=view_a.hierarchy)


def _sum(view: TileView, axis) -> TileView:
    shape = list(view.shape)
    shape[axis] = 1

    hierarchy = []
    for level in view.hierarchy:
        tile_shape = list(level.tile_shape)
        tile_shape[axis] = 1
        hierarchy.append(TileLevel(level_name=level.level_name, tile_shape=tile_shape))

    return TileView(shape=shape, hierarchy=hierarchy)


def create_tile_view(
    shape: tuple[int, ...],
    hierarchy: list[TileLevel],
):
    return TileView(
        shape=shape,
        hierarchy=hierarchy,
    )


def retilize_view(old_view: TileView, hierarchy: list[TileLevel]):
    new_view = TileView(
        shape=old_view.shape,
        hierarchy=hierarchy,
    )

    steps = None

    return new_view, steps


class Cache(PClass):
    node_output_to_array = pmap_field(key_type=tuple, value_type=TileView)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_array=pmap(dictionary))

    def __getitem__(self, persistent_array: PersistentArray):
        node = persistent_array.node
        output_index = persistent_array.output_index
        return self.node_output_to_array[(node, output_index)]

    def as_dict_from_variable_to_array(self):
        return {
            cnp.nn.variable(name=node.name, shape=array.shape): array
            for (node, _), array in self.node_output_to_array.items()
        }


def initialize_cache(graph, inputs):
    cache = {}
    for parray, tile_levels in inputs.items():
        cache[(parray.node, parray.output_index)] = create_tile_view(parray.shape, tile_levels)
    return cache


def propagate_tile_views(
    *output_vars,
    inputs,
    initialize_cache_function=initialize_cache,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    cache = initialize_cache_function(graph, inputs)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]

        if instruction.__class__.__name__ == "matmul":
            instruction_output = input_arrays[0] @ input_arrays[1]
        elif instruction.__class__.__name__ == "sum":
            instruction_output = _sum(input_arrays[0], axis=instruction.axis)
        elif instruction.__class__.__name__ == "add":
            input_0, input_1 = input_arrays
            if input_0.hierarchy != input_1.hierarchy and math.prod(input_0.shape) == math.prod(input_1.shape):
                input_1, _ = retilize_view(input_1, input_0.hierarchy)
            instruction_output = input_0 + input_1
        elif instruction.__class__.__name__ == "subtract":
            input_0, input_1 = input_arrays
            if input_0.hierarchy != input_1.hierarchy and math.prod(input_0.shape) == math.prod(input_1.shape):
                input_1, _ = retilize_view(input_1, input_0.hierarchy)
            instruction_output = input_0 - input_1
        else:
            raise RuntimeError(f"Unrecognized instruction: {instruction}")

        if isinstance(instruction_output, TileView):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, collections.abc.Iterable):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError(f"Unsupported type: {type(instruction_output)}")

    return Cache.from_dict(cache)
