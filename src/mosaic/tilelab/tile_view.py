from __future__ import annotations

import collections
import math

from pyrsistent import PClass, field, pmap_field, PVector
from pyimmer import pmap

from composit.introspection import class_name
from composit.types import LazyTensor
from composit.multidigraph import topological_traversal
from composit.numpy.core import get_operands

from mosaic.tilelab.layout import DefaultLayout


class TileLevel(PClass):
    level_name = field()
    tile_shape = field()
    layout = field(initial=DefaultLayout())


class ScalarTileLevel(PClass):
    level_name = field()
    rank: int = field()
    layout = field(initial=DefaultLayout())

    @property
    def tile_shape(self):
        return tuple(1 for _ in range(self.rank))


class TileView(PClass):
    shape = field()
    hierarchy: PVector[TileLevel] = field()

    @property
    def num_levels(self):
        return len(self.hierarchy)

    def __repr__(self):
        return f"{class_name(self)}(shape={self.shape}, hierarchy={self.hierarchy})"


def _embedding(view_a: TileView, view_b: TileView) -> TileView:
    batch_size, sequence_size = view_a.shape
    _, hidden_size = view_b.shape

    hierarchy = []
    for view_a_level, view_b_level in zip(view_a.hierarchy, view_b.hierarchy):
        tile_batch_size, tile_sequence_size = view_a_level.tile_shape
        _, tile_hidden_size = view_b_level.tile_shape
        hierarchy.append(
            TileLevel(
                level_name=view_a_level.level_name, tile_shape=(tile_batch_size, tile_sequence_size, tile_hidden_size)
            )
        )

    return TileView(shape=(batch_size, sequence_size, hidden_size), hierarchy=hierarchy)


def _matmul(view_a: TileView, view_b: TileView) -> TileView:
    shape = view_a.shape[:-1] + view_b.shape[-1:]
    hierarchy = []
    for view_a_level, view_b_level in zip(view_a.hierarchy, view_b.hierarchy):
        hierarchy.append(
            TileLevel(
                level_name=view_a_level.level_name,
                tile_shape=view_a_level.tile_shape[:-1] + view_b_level.tile_shape[-1:],
            )
        )

    return TileView(shape=shape, hierarchy=hierarchy)


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
        tile_size = shape[-1]
        if len(shape) < len(newshape):
            return (1, tile_size, 1, tile_size)
        else:
            return (1, tile_size, tile_size)

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


def create_tile_view(shape: tuple[int, ...], hierarchy: list[TileLevel]):
    return TileView(shape=shape, hierarchy=hierarchy)


def retilize_view(old_view: TileView, hierarchy: list[TileLevel]):
    new_view = TileView(shape=old_view.shape, hierarchy=hierarchy)
    steps = None
    return new_view, steps


class Cache(PClass):
    node_output_to_tile_view = pmap_field(key_type=tuple, value_type=TileView)

    @classmethod
    def from_dict(cls, dictionary):
        return cls(node_output_to_tile_view=pmap(dictionary))

    def __getitem__(self, lazy_tensor: LazyTensor):
        node = lazy_tensor.node
        output_index = lazy_tensor.output_index
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
        operation = graph.nodes[node]["operation"]
        operation_class_name = class_name(operation)
        input_tile_views = [cache[operand] for operand in get_operands(graph, node)]

        if operation_class_name == "Input":
            shape = graph.nodes[node]["shapes"][0]
            tile_view = create_tile_view(
                shape,
                [
                    TileLevel(level_name="tile", tile_shape=shape),
                    ScalarTileLevel(level_name="scalar", rank=len(shape)),
                ],
            )
        elif operation_class_name == "embedding":
            tile_view = _embedding(input_tile_views[0], input_tile_views[1])
        elif operation_class_name == "matmul":
            tile_view = _matmul(input_tile_views[0], input_tile_views[1])
        elif operation_class_name in {"sum", "mean", "max"}:
            tile_view = _reduce(input_tile_views[0], axis=operation.axis)
        elif operation_class_name in {"add", "subtract", "multiply", "divide"}:
            if input_tile_views[0].hierarchy != input_tile_views[1].hierarchy and math.prod(
                input_tile_views[0].shape
            ) == math.prod(input_tile_views[1].shape):
                input_tile_views[1], _ = retilize_view(input_tile_views[1], input_tile_views[0].hierarchy)
            tile_view = _binary_operation(input_tile_views[0], input_tile_views[1])
        elif operation_class_name in {"sqrt", "exp", "gelu"}:
            (tile_view,) = input_tile_views
        elif operation_class_name == "reshape":
            tile_view = _reshape(input_tile_views[0], operation.newshape)
        elif operation_class_name == "transpose":
            tile_view = _transpose(input_tile_views[0], operation.axes)
        else:
            raise RuntimeError(f"Unrecognized operation: {operation}")

        if isinstance(tile_view, TileView):
            cache[(node, 0)] = tile_view
        elif isinstance(tile_view, collections.abc.Iterable):
            for output_index, tile_view_of_output in enumerate(tile_view):
                cache[(node, output_index)] = tile_view_of_output
        else:
            raise RuntimeError(f"Unsupported type: {type(tile_view)}")

    return Cache.from_dict(cache)
