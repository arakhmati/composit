from __future__ import annotations

import itertools
import math

import numpy as np
from pyrsistent import PClass, field, pmap
from toolz import first

from mosaic.tilelab.tilization_level import TilizationLevel


class TileMetadata(PClass):
    level_name = field()
    shape = field()
    index_to_tile = field()

    @property
    def hierarchy(self):
        return [TilizationLevel(level_name=self.level_name, tile_shape=self.tile_shape)] + first(
            self.index_to_tile.values()
        ).hierarchy

    @property
    def tile_shape(self):
        return first(self.index_to_tile.values()).shape

    def num_tiles_per_axis(self):
        return tuple(dim // tile_dim for dim, tile_dim in zip(self.shape, self.tile_shape))

    def __getitem__(self, index):
        return self.index_to_tile[index]

    def __repr__(self):
        result = f"{self.__class__.__name__}(level_name={self.level_name}, shape={self.shape}, tile_shape={self.tile_shape}, num_tiles={len(self.index_to_tile)})"
        return f"{result}\n{first(self.index_to_tile.values())}"


class SliceMetadata(PClass):
    level_name = field(initial="atomic")
    shape = field()
    slices = field()

    @property
    def hierarchy(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape}, slices={self.slices})"


def create_tile_metadata(
    shape: tuple[int, ...],
    hierarchy: list[TilizationLevel],
    offsets=None,
) -> TileMetadata:
    if offsets is None:
        offsets = tuple(0 for _ in shape)

    tilization_level, *remaining_hierarchy = hierarchy
    tile_shape = tilization_level.tile_shape

    if len(shape) != len(tile_shape):
        raise RuntimeError("Shapes must have the same rank")

    ranges = (range(0, tensor_dim, tile_dim) for tensor_dim, tile_dim in zip(shape, tile_shape))

    index_to_tile = {}
    for indices in itertools.product(*ranges):
        tile_indices = tuple(tensor_index // tile_dim for tensor_index, tile_dim in zip(indices, tile_shape))

        new_offsets = [offset + index for offset, index in zip(offsets, indices)]

        if remaining_hierarchy:
            index_to_tile[tile_indices] = create_tile_metadata(tile_shape, remaining_hierarchy, offsets=new_offsets)
        else:
            tile_slices = tuple(slice(index, index + tile_dim) for index, tile_dim in zip(new_offsets, tile_shape))
            index_to_tile[tile_indices] = SliceMetadata(shape=tile_shape, slices=tile_slices)

    tile_metadata = TileMetadata(
        level_name=tilization_level.level_name,
        shape=shape,
        index_to_tile=pmap(index_to_tile),
    )

    return tile_metadata


def to_tiles(tensor, arg, transpose_levels, order):
    if isinstance(arg, TileMetadata):
        tile_metadata = arg
        ranges = tuple(range(num_tiles) for num_tiles in tile_metadata.num_tiles_per_axis())
        if tile_metadata.level_name in transpose_levels:
            ranges = [ranges[axis] for axis in order]
        for tile_indices in itertools.product(*ranges):
            if tile_metadata.level_name in transpose_levels:
                tile_indices = tuple([tile_indices[axis] for axis in order])
            yield from to_tiles(tensor, tile_metadata[tile_indices], transpose_levels, order)
    else:
        slice_metadata = arg
        tensor = tensor[slice_metadata.slices]
        if slice_metadata.level_name in transpose_levels:
            tensor = np.transpose(tensor, order)
        yield tensor


def from_tiles(tiles, arg):
    if isinstance(arg, TileMetadata):
        tile_metadata = arg
        output = np.zeros(tile_metadata.shape)
        ranges = (range(num_tiles) for num_tiles in tile_metadata.num_tiles_per_axis())
        for tile_indices in itertools.product(*ranges):
            tile = from_tiles(tiles, tile_metadata[tile_indices])
            tile_slices = tuple(
                slice(tile_index * tile_dim, (tile_index + 1) * tile_dim)
                for tile_index, tile_dim in zip(tile_indices, tile_metadata.tile_shape)
            )
            output[tile_slices] = tile
        return output
    else:
        return next(tiles)


def aligned_array(shape, dtype, align=32):
    size = np.prod(shape) * np.dtype(dtype).itemsize
    buffer = np.empty(size + align, dtype=np.uint8)
    offset = buffer.ctypes.data % align
    array = np.ndarray(shape, dtype=dtype, buffer=buffer, offset=offset)
    return array


def to_flat_array(array: np.array, tile_metadata: TileMetadata, *, transpose_levels=None, order=None) -> np.ndarray:
    if transpose_levels is None:
        transpose_levels = set()

    tiles = to_tiles(array, tile_metadata, transpose_levels, order)

    tile_size = math.prod(tile_metadata.hierarchy[-1].tile_shape)
    start = 0
    end = tile_size

    flat_array = aligned_array((math.prod(array.shape),), array.dtype)
    for tile in tiles:
        flat_array[start:end] = tile.flatten()
        start = end
        end += tile_size
    return flat_array


def from_flat_array(flat_array: np.array, tile_metadata: TileMetadata) -> np.ndarray:
    tile_level = tile_metadata.hierarchy[-1]
    num_tiles = len(flat_array) / math.prod(tile_level.tile_shape)
    tiles = (tile.reshape(tile_level.tile_shape) for tile in np.array_split(flat_array, num_tiles))
    return from_tiles(tiles, tile_metadata)
