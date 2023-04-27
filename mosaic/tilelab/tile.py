from __future__ import annotations

import itertools
import math

import numpy as np
from pyrsistent import PClass, field, pmap
from toolz import first

from composit.introspection import class_name
from mosaic.tilelab.tile_view import TileLevel, TileView


class ArrayTileConfig(PClass):
    level_name = field()
    shape = field()
    index_to_tile = field()

    @property
    def hierarchy(self):
        current = [TileLevel(level_name=self.level_name, tile_shape=self.tile_shape)]
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
        first_child = first(self.index_to_tile.values())
        if isinstance(first_child, ArrayTileConfig):
            first_child_repr = f"\n{first_child}"
        else:
            first_child_repr = ""

        return f"{result}{first_child_repr}"


class AtomicTileConfig(PClass):
    level_name = field(initial="atomic")
    shape = field()
    slices = field()

    @property
    def hierarchy(self):
        return []

    def __repr__(self):
        return f"{class_name(self)}(shape={self.shape}, slices={self.slices})"

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

        if remaining_hierarchy:
            index_to_tile[tile_indices] = create_array_tile_config(
                TileView(shape=tile_shape, hierarchy=remaining_hierarchy), offsets=new_offsets
            )
        else:
            tile_slices = tuple(slice(index, index + tile_dim) for index, tile_dim in zip(new_offsets, tile_shape))
            index_to_tile[tile_indices] = AtomicTileConfig(shape=tile_shape, slices=tile_slices)

    array_tile_config = ArrayTileConfig(
        level_name=tile_level.level_name,
        shape=shape,
        index_to_tile=pmap(index_to_tile),
    )

    return array_tile_config


def to_tiles(tensor, arg, transpose_levels, order):
    if isinstance(arg, ArrayTileConfig):
        array_tile_config = arg
        ranges = tuple(range(num_tiles) for num_tiles in array_tile_config.num_tiles_per_axis())
        if array_tile_config.level_name in transpose_levels:
            ranges = [ranges[axis] for axis in order]
        for tile_indices in itertools.product(*ranges):
            if array_tile_config.level_name in transpose_levels:
                tile_indices = tuple([tile_indices[axis] for axis in order])
            yield from to_tiles(tensor, array_tile_config[tile_indices], transpose_levels, order)
    else:
        slice_metadata = arg
        tensor = tensor[slice_metadata.slices]
        if slice_metadata.level_name in transpose_levels:
            tensor = np.transpose(tensor, order)
        yield tensor


def from_tiles(tiles, arg):
    if isinstance(arg, ArrayTileConfig):
        array_tile_config = arg
        output = np.zeros(array_tile_config.shape)
        ranges = (range(num_tiles) for num_tiles in array_tile_config.num_tiles_per_axis())
        for tile_indices in itertools.product(*ranges):
            tile = from_tiles(tiles, array_tile_config[tile_indices])
            tile_slices = tuple(
                slice(tile_index * tile_dim, (tile_index + 1) * tile_dim)
                for tile_index, tile_dim in zip(tile_indices, array_tile_config.tile_shape)
            )
            output[tile_slices] = tile
        return output
    else:
        return next(tiles)


def create_aligned_array(shape, dtype, align=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + align, dtype=np.uint8)
    offset = buffer.ctypes.data % align
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def to_tilized_array(
    array: np.array, array_tile_config: ArrayTileConfig, *, transpose_levels=None, order=None
) -> np.ndarray:
    if transpose_levels is None:
        transpose_levels = set()

    tiles = to_tiles(array, array_tile_config, transpose_levels, order)

    tile_size = math.prod(array_tile_config.hierarchy[-1].tile_shape)
    start = 0
    end = tile_size

    flat_array = create_aligned_array((math.prod(array.shape),), array.dtype)
    for tile in tiles:
        flat_array[start:end] = tile.flatten()
        start = end
        end += tile_size
    return flat_array


def from_tilized_array(flat_array: np.array, array_tile_config: ArrayTileConfig) -> np.ndarray:
    tile_level = array_tile_config.hierarchy[-1]
    num_tiles = len(flat_array) / math.prod(tile_level.tile_shape)
    tiles = (tile.reshape(tile_level.tile_shape) for tile in np.array_split(flat_array, num_tiles))
    return from_tiles(tiles, array_tile_config)
