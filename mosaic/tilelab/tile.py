from __future__ import annotations

import math

import numpy as np
from pyrsistent import PClass, field

from composit.introspection import class_name
from mosaic.tilelab.layout import TransposedLayout
from mosaic.tilelab.tile_view import TileLevel, TileView, ScalarTileLevel


class TileConfig(PClass):
    level_name = field()
    shape = field()
    layout = field()
    next_level_tile_config = field()

    @property
    def hierarchy(self):
        current = [TileLevel(level_name=self.level_name, tile_shape=self.tile_shape, layout=self.layout)]
        downstream = self.next_level_tile_config.hierarchy
        return current + downstream

    @property
    def tile_shape(self):
        return self.next_level_tile_config.shape

    @property
    def rank(self):
        return len(self.shape)

    def num_tiles_per_axis(self):
        return tuple(dim // tile_dim for dim, tile_dim in zip(self.shape, self.tile_shape))

    def __repr__(self):
        result = (
            f"{class_name(self)}(level_name={self.level_name}, shape={self.shape}, "
            f"tile_shape={self.tile_shape}, num_tiles={math.prod(self.num_tiles_per_axis())})"
        )
        first_child = self.next_level_tile_config
        if isinstance(first_child, TileConfig):
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


def create_tile_config(tile_view: TileView) -> TileConfig | AtomicTileConfig:
    shape = tile_view.shape
    hierarchy = tile_view.hierarchy

    tile_level, *remaining_hierarchy = hierarchy

    if len(remaining_hierarchy) > 1:
        tile_shape = tile_level.tile_shape
        if len(shape) != len(tile_shape):
            raise RuntimeError(f"Shapes must have the same rank: {shape} != {tile_shape}")

        next_level_tile_config = create_tile_config(TileView(shape=tile_shape, hierarchy=remaining_hierarchy))

        return TileConfig(
            level_name=tile_level.level_name,
            shape=shape,
            layout=tile_level.layout,
            next_level_tile_config=next_level_tile_config,
        )
    else:
        return AtomicTileConfig(
            level_name=tile_level.level_name,
            shape=shape,
            layout=tile_level.layout,
        )


def create_aligned_array(shape, dtype, align=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + align, dtype=np.uint8)
    offset = buffer.ctypes.data % align
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def get_all_num_tiles_per_axis(tile_config):
    if isinstance(tile_config, TileConfig):
        return tile_config.num_tiles_per_axis() + get_all_num_tiles_per_axis(tile_config.next_level_tile_config)
    else:
        return tile_config.num_tiles_per_axis()


def transpose_sequence(sequence, axes):
    new_sequence = list(sequence)
    for axis, new_axis in enumerate(axes):
        new_sequence[new_axis] = sequence[axis]
    return tuple(new_sequence)


def compute_shape_before_tilization(tile_config: TileConfig):
    axes = list(range(len(tile_config.shape)))
    all_num_tiles_per_axis = get_all_num_tiles_per_axis(tile_config)
    tilized_shape = []
    for axis in axes:
        tilized_shape += all_num_tiles_per_axis[axis :: len(axes)]
    return tilized_shape


def compute_tilize_transpose_order(tile_config):
    axes = list(range(len(tile_config.shape)))
    hierarchy = list(tile_config.hierarchy)
    result = []
    for level_index, level in enumerate(hierarchy):
        order = [level_index + axis * len(hierarchy) for axis in axes]
        if isinstance(level.layout, TransposedLayout):
            order = transpose_sequence(order, level.layout.order)
        result += order
    return result


def compute_untilize_transpose_order(tile_config):
    axes = list(range(len(tile_config.shape)))
    num_levels = len(tile_config.hierarchy)
    order = [axis + level_index * len(axes) for axis in axes for level_index in range(num_levels)]
    return order


def to_tilized_array(array: np.array, tile_config: TileConfig) -> np.ndarray:
    tilized_array = create_aligned_array((math.prod(array.shape),), array.dtype)
    shape_before = compute_shape_before_tilization(tile_config)
    transpose_order = compute_tilize_transpose_order(tile_config)
    array = array.reshape(shape_before)
    array = array.transpose(transpose_order)
    tilized_array[:] = array.flatten()
    return tilized_array


def from_tilized_array(array: np.array, tile_config: TileConfig) -> np.ndarray:
    transpose_order = compute_untilize_transpose_order(tile_config)
    array = array.reshape(get_all_num_tiles_per_axis(tile_config))
    array = array.transpose(transpose_order)
    return array.reshape(tile_config.shape)
