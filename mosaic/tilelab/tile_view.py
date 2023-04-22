from __future__ import annotations

import collections
import math

from pyrsistent import PClass, field, pmap, pmap_field, PVector

from composit.introspection import class_name
from composit.persistent_array import PersistentArray
from composit.multidigraph import topological_traversal
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
        return f"{class_name(self)}(shape={self.shape}, hierarchy={self.hierarchy})"


def _matmul(self, other):
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


def _binary_operation(view_a: TileView, _: TileView) -> TileView:
    return TileView(shape=view_a.shape, hierarchy=view_a.hierarchy)


def _reduce(view: TileView, axis) -> TileView:
    if isinstance(axis, tuple):
        axes = axis
    else:
        axes = [axis]

    def new_shape(shape):
        shape = list(shape)
        for axis in axes:
            shape[axis] = 1
        return tuple(shape)

    hierarchy = []
    for level in view.hierarchy:
        hierarchy.append(TileLevel(level_name=level.level_name, tile_shape=new_shape(level.tile_shape)))

    return TileView(shape=new_shape(view.shape), hierarchy=hierarchy)


def _reshape(view: TileView, newshape) -> TileView:
    def new_shape(shape):
        if len(shape) < len(newshape):
            return (1, 32, 4, 32)
        else:
            return (1, 32, 32)

    hierarchy = []
    for level in view.hierarchy:
        hierarchy.append(TileLevel(level_name=level.level_name, tile_shape=new_shape(level.tile_shape)))

    return TileView(shape=newshape, hierarchy=hierarchy)


def _transpose(view: TileView, order) -> TileView:
    def new_shape(shape):
        shape = [shape[axis] for axis in order]
        return tuple(shape)

    hierarchy = []
    for level in view.hierarchy:
        hierarchy.append(TileLevel(level_name=level.level_name, tile_shape=new_shape(level.tile_shape)))

    return TileView(shape=new_shape(view.shape), hierarchy=hierarchy)


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
    node_output_to_tile_view = pmap_field(key_type=tuple, value_type=TileView)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_tile_view=pmap(dictionary))

    def __getitem__(self, persistent_array: PersistentArray):
        node = persistent_array.node
        output_index = persistent_array.output_index
        return self.node_output_to_tile_view[(node, output_index)]

    def __iter__(self):
        return iter(self.node_output_to_tile_view.items())


def initialize_cache(inputs):
    cache = {}
    for parray, tile_levels in inputs.items():
        cache[(parray.node, parray.output_index)] = create_tile_view(parray.shape, tile_levels)
    return cache


def propagate_tile_views(
    graph,
    inputs,
    initialize_cache_function=initialize_cache,
):
    cache = initialize_cache_function(inputs)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        instruction_class_name = class_name(instruction)
        input_tile_views = [cache[operand] for operand in get_operands(graph, node)]

        if instruction_class_name == "Constant":
            tile_view = create_tile_view((), [TileLevel(level_name="tile", tile_shape=())])
        elif instruction_class_name == "embedding":
            tile_view = create_tile_view(
                graph.nodes[node]["shapes"][0], [TileLevel(level_name="tile", tile_shape=(1, 32, 32))]
            )
        elif instruction_class_name == "matmul":
            tile_view = _matmul(input_tile_views[0], input_tile_views[1])
        elif instruction_class_name in {"sum", "mean", "max"}:
            tile_view = _reduce(input_tile_views[0], axis=instruction.axis)
        elif instruction_class_name in {"add", "subtract", "multiply", "divide"}:
            if input_tile_views[0].hierarchy != input_tile_views[1].hierarchy and math.prod(
                input_tile_views[0].shape
            ) == math.prod(input_tile_views[1].shape):
                input_tile_views[1], _ = retilize_view(input_tile_views[1], input_tile_views[0].hierarchy)
            tile_view = _binary_operation(input_tile_views[0], input_tile_views[1])
        elif instruction_class_name in {"sqrt", "exp", "gelu"}:
            (tile_view,) = input_tile_views
        elif instruction_class_name == "reshape":
            tile_view = _reshape(input_tile_views[0], instruction.newshape)
        elif instruction_class_name == "transpose":
            tile_view = _transpose(input_tile_views[0], instruction.axes)
        else:
            raise RuntimeError(f"Unrecognized instruction: {instruction}")

        if isinstance(tile_view, TileView):
            cache[(node, 0)] = tile_view
        elif isinstance(tile_view, collections.abc.Iterable):
            for output_index, tile_view_of_output in enumerate(tile_view):
                cache[(node, output_index)] = tile_view_of_output
        else:
            raise RuntimeError(f"Unsupported type: {type(tile_view)}")

    return Cache.from_dict(cache)
