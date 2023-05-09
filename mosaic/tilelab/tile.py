from __future__ import annotations

import math
import itertools

import numpy as np
from pyrsistent import PClass, field, pmap
from toolz import first

from composit.introspection import class_name
from mosaic.tilelab.layout import TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, TileView, ScalarTileLevel


class ArrayTileConfig(PClass):
    level_name = field()
    shape = field()
    layout = field()
    index_to_tile = field()

    @property
    def hierarchy(self):
        current = [TileLevel(level_name=self.level_name, tile_shape=self.tile_shape, layout=self.layout)]
        downstream = first(self.index_to_tile.values()).hierarchy
        return current + downstream

    @property
    def tile_shape(self):
        return first(self.index_to_tile.values()).shape

    @property
    def rank(self):
        return len(self.shape)

    def next_level(self):
        return self[tuple(0 for _ in range(self.rank))]

    def num_tiles_per_axis(self):
        return tuple(dim // tile_dim for dim, tile_dim in zip(self.shape, self.tile_shape))

    def __getitem__(self, index):
        return self.index_to_tile[index]

    def __repr__(self):
        result = (
            f"{class_name(self)}(level_name={self.level_name}, shape={self.shape}, "
            f"tile_shape={self.tile_shape}, num_tiles={len(self.index_to_tile)})"
        )
        first_child = self.next_level()
        if isinstance(first_child, ArrayTileConfig):
            first_child_repr = f"\n{first_child}"
        else:
            first_child_repr = ""

        return f"{result}{first_child_repr}"


class AtomicTileConfig(PClass):
    level_name = field()
    shape = field()
    layout = field()

    @property
    def tile_shape(self):
        return tuple(1 for _ in self.shape)

    @property
    def hierarchy(self):
        return [ScalarTileLevel(level_name=self.level_name, rank=len(self.shape), layout=self.layout)]

    def __repr__(self):
        return f"{class_name(self)}(level_name={self.level_name}, shape={self.shape}, layout={self.layout})"

    def num_tiles_per_axis(self):
        # Each scalar value is a tile
        return self.shape


def create_array_tile_config(
    tile_view: TileView,
    offsets=None,
) -> ArrayTileConfig:
    shape = tile_view.shape
    hierarchy = tile_view.hierarchy

    if offsets is None:
        offsets = tuple(0 for _ in shape)

    tile_level, *remaining_hierarchy = hierarchy
    tile_shape = tile_level.tile_shape

    if len(shape) != len(tile_shape):
        raise RuntimeError(f"Shapes must have the same rank: {shape} != {tile_shape}")

    ranges = (range(0, tensor_dim, tile_dim) for tensor_dim, tile_dim in zip(shape, tile_shape))

    index_to_tile = {}
    for indices in itertools.product(*ranges):
        tile_indices = tuple(tensor_index // tile_dim for tensor_index, tile_dim in zip(indices, tile_shape))

        new_offsets = [offset + index for offset, index in zip(offsets, indices)]

        if len(remaining_hierarchy) > 1:
            index_to_tile[tile_indices] = create_array_tile_config(
                TileView(shape=tile_shape, hierarchy=remaining_hierarchy), offsets=new_offsets
            )
        else:
            leaf_level = remaining_hierarchy[-1]
            index_to_tile[tile_indices] = AtomicTileConfig(
                level_name=leaf_level.level_name,
                shape=tile_shape,
                layout=leaf_level.layout,
            )

    array_tile_config = ArrayTileConfig(
        level_name=tile_level.level_name,
        shape=shape,
        layout=tile_level.layout,
        index_to_tile=pmap(index_to_tile),
    )

    return array_tile_config


def create_aligned_array(shape, dtype, align=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + align, dtype=np.uint8)
    offset = buffer.ctypes.data % align
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def get_all_num_tiles_per_axis(array_tile_config):
    if isinstance(array_tile_config, ArrayTileConfig):
        return array_tile_config.num_tiles_per_axis() + get_all_num_tiles_per_axis(array_tile_config.next_level())
    else:
        return array_tile_config.num_tiles_per_axis()


def transpose_sequence(sequence, axes):
    new_sequence = list(sequence)
    for axis, new_axis in enumerate(axes):
        new_sequence[new_axis] = sequence[axis]
    return tuple(new_sequence)


def compute_shape_before_tilization(array_tile_config: ArrayTileConfig):
    axes = list(range(len(array_tile_config.shape)))
    all_num_tiles_per_axis = get_all_num_tiles_per_axis(array_tile_config)
    tilized_shape = []
    for axis in axes:
        tilized_shape += all_num_tiles_per_axis[axis :: len(axes)]
    return tilized_shape


def compute_tilize_transpose_order(array_tile_config):
    axes = list(range(len(array_tile_config.shape)))
    hierarchy = list(array_tile_config.hierarchy)
    result = []
    for level_index, level in enumerate(hierarchy):
        order = [level_index + axis * len(hierarchy) for axis in axes]
        if isinstance(level.layout, TransposedLayout):
            order = transpose_sequence(order, level.layout.order)
        result += order
    return result


def compute_untilize_transpose_order(array_tile_config):
    axes = list(range(len(array_tile_config.shape)))
    num_levels = len(array_tile_config.hierarchy)
    order = [axis + level_index * len(axes) for axis in axes for level_index in range(num_levels)]
    return order


def to_tilized_array(array: np.array, array_tile_config: ArrayTileConfig) -> np.ndarray:
    tilized_array = create_aligned_array((math.prod(array.shape),), array.dtype)
    shape_before = compute_shape_before_tilization(array_tile_config)
    transpose_order = compute_tilize_transpose_order(array_tile_config)
    array = array.reshape(shape_before)
    array = array.transpose(transpose_order)
    tilized_array[:] = array.flatten()
    return tilized_array


def from_tilized_array(array: np.array, array_tile_config: ArrayTileConfig) -> np.ndarray:
    transpose_order = compute_untilize_transpose_order(array_tile_config)
    array = array.reshape(get_all_num_tiles_per_axis(array_tile_config))
    array = array.transpose(transpose_order)
    return array.reshape(array_tile_config.shape)
