from __future__ import annotations

import collections
import itertools
import math
import operator

import numpy as np
from pyrsistent import PClass, field, pmap
from toolz import first

from composit.multidigraph import compose_all, topological_traversal
from composit.numpy.core import get_operands
from composit.tilelab.tilization_level import TilizationLevel


def binary_operation(self, other, operation):
    ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())

    index_to_tile = {}
    for index in itertools.product(*ranges):
        b_index = tuple(i % num_tiles for i, num_tiles in zip(index, other.num_tiles_per_axis()))
        a_tile = self[index]
        b_tile = other[b_index]
        tile = operation(a_tile, b_tile)
        index_to_tile[index] = tile

    return TilizedTensor(
        level_name=self.level_name,
        shape=self.shape,
        tile_shape=self.tile_shape,
        index_to_tile=index_to_tile,
    )


class TilizedTensor(PClass):
    level_name = field()
    shape = field()
    tile_shape = field()
    index_to_tile = field()

    @property
    def hierarchy(self):
        return [TilizationLevel(level_name=self.level_name, tile_shape=self.tile_shape)] + first(
            self.index_to_tile.values()
        ).hierarchy

    def num_tiles_per_axis(self):
        return tuple(dim // tile_dim for dim, tile_dim in zip(self.shape, self.tile_shape))

    def __getitem__(self, index):
        return self.index_to_tile[index]

    def __repr__(self):
        result = f"{self.__class__.__name__}(level_name={self.level_name}, shape={self.shape}, tile_shape={self.tile_shape}, num_tiles={len(self.index_to_tile)})"
        return f"{result}\n{first(self.index_to_tile.values())}"

    def __eq__(self, other):
        if self.num_tiles_per_axis() != other.num_tiles_per_axis():
            return False
        if len(self.index_to_tile) != len(other.index_to_tile):
            return False
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())

        equals = True
        for index in itertools.product(*ranges):
            a_tile = self[index]
            b_tile = other[index]
            equals &= a_tile == b_tile
        return equals

    def __add__(*args):
        return binary_operation(*args, operator.add)

    def __sub__(*args):
        return binary_operation(*args, operator.sub)

    def __matmul__(self, other):
        a_num_tiles_per_axis = self.num_tiles_per_axis()
        b_num_tiles_per_axis = other.num_tiles_per_axis()
        a_ranges = (range(num_tiles) for num_tiles in a_num_tiles_per_axis)
        _, n_range = (range(num_tiles) for num_tiles in b_num_tiles_per_axis)

        index_to_tile = {}
        for *a_indices, n in itertools.product(*a_ranges, n_range):
            a_indices = tuple(a_indices)
            *batch_indices, m, k = a_indices
            batch_indices = tuple(batch_indices)
            b_indices = (k, n)
            output_indices = batch_indices + (m, n)
            a_tile = self[a_indices]
            b_tile = other[b_indices]
            tile = a_tile @ b_tile
            if output_indices not in index_to_tile:
                index_to_tile[output_indices] = tile
            else:
                index_to_tile[output_indices] += tile

        return TilizedTensor(
            level_name=self.level_name,
            shape=self.shape[:-1] + other.shape[-1:],
            tile_shape=self.tile_shape[:-1] + other.tile_shape[-1:],
            index_to_tile=index_to_tile,
        )

    def sum(self, axis, keepdims=True):
        num_tiles_per_axis = self.num_tiles_per_axis()
        ranges = (range(num_tiles) for num_tiles in num_tiles_per_axis)

        index_to_tile = {}
        for index in itertools.product(*ranges):
            output_index = list(index)
            output_index[axis] = 0
            output_index = tuple(output_index)
            input_tile = self[index].sum(axis=axis, keepdims=keepdims)
            if output_index not in index_to_tile:
                output_tile = input_tile
            else:
                output_tile = index_to_tile[output_index] + input_tile
            index_to_tile[output_index] = output_tile

        shape = list(self.shape)
        shape[axis] = output_tile.shape[axis]
        shape = tuple(shape)

        tile_shape = list(self.tile_shape)
        tile_shape[axis] = shape[axis]
        tile_shape = tuple(tile_shape)

        return TilizedTensor(
            level_name=self.level_name,
            shape=shape,
            tile_shape=tile_shape,
            index_to_tile=index_to_tile,
        )

    def tiles(self):
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())
        for tile_indices in itertools.product(*ranges):
            yield from self[tile_indices].tiles()


class Tile(PClass):
    tile = field()
    level_name = field(initial="atomic")

    @property
    def shape(self):
        return self.tile.shape

    @property
    def hierarchy(self):
        return []

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __eq__(self, other) -> bool:
        return np.allclose(self.tile, other.tile)

    def __add__(self, other):
        return Tile(tile=self.tile + other.tile)

    def __sub__(self, other):
        return Tile(tile=self.tile - other.tile)

    def __matmul__(self, other):
        return Tile(tile=self.tile @ other.tile)

    def sum(self, axis, keepdims):
        result = np.sum(self.tile, axis=axis, keepdims=keepdims)
        """
        To pad, or not to pad, that is the question
        pad_width = [[0, 0] for _ in self.tile.shape]
        pad_width[axis][1] = self.tile.shape[axis] - result.shape[axis]
        result = np.pad(result, pad_width)
        """
        return Tile(tile=result)

    def tiles(self):
        yield self.tile


def tilize_tensor(
    tensor: np.ndarray,
    hierarchy: list[TilizationLevel],
):
    tilization_level, *remaining_hierarchy = hierarchy
    tile_shape = tilization_level.tile_shape

    if len(tensor.shape) != len(tile_shape):
        raise RuntimeError("Shapes must have the same rank")

    ranges = (range(0, tensor_dim, tile_dim) for tensor_dim, tile_dim in zip(tensor.shape, tile_shape))

    index_to_tile = {}
    for indices in itertools.product(*ranges):
        tile_slices = tuple(
            slice(tensor_index, tensor_index + tile_dim) for tensor_index, tile_dim in zip(indices, tile_shape)
        )
        tile_indices = tuple(tensor_index // tile_dim for tensor_index, tile_dim in zip(indices, tile_shape))
        tile = tensor[tile_slices]

        if remaining_hierarchy:
            index_to_tile[tile_indices] = tilize_tensor(tile, remaining_hierarchy)
        else:
            index_to_tile[tile_indices] = Tile(tile=tile)

    tilized_tensor = TilizedTensor(
        level_name=tilization_level.level_name,
        shape=tensor.shape,
        tile_shape=tile_shape,
        index_to_tile=pmap(index_to_tile),
    )

    return tilized_tensor


def slice_tensors(tensor, axis, start, end, slice_size):
    shape = list(tensor.shape)
    shape[axis] = (end - start) * slice_size

    if isinstance(tensor, TilizedTensor):
        index_to_tile = {}
        ranges = tuple(range(num_tiles) for num_tiles in tensor.num_tiles_per_axis())
        for indices in itertools.product(*ranges):
            if not (start <= indices[axis] < end):
                continue
            new_indices = list(indices)
            new_indices[axis] -= start
            new_indices = tuple(new_indices)
            index_to_tile[new_indices] = tensor.index_to_tile[indices]

        tilized_tensor = TilizedTensor(
            level_name=tensor.level_name,
            shape=tuple(shape),
            tile_shape=tensor.tile_shape,
            index_to_tile=pmap(index_to_tile),
        )
        return tilized_tensor
    else:
        slices = tuple(
            slice(start * slice_size, end * slice_size) if i == axis else slice(0, None) for i, dim in enumerate(shape)
        )
        tile = tensor.tile[slices]
        return Tile(tile=tile)


def concatenate_tensors(tensors, axis):
    first_tensor, *_ = tensors

    if isinstance(first_tensor, TilizedTensor):
        shape = list(first_tensor.shape)
        index_to_tile = dict(first_tensor.index_to_tile)
        offset = first_tensor.num_tiles_per_axis()[axis]

        for other_tensor in tensors[1:]:
            shape[axis] += other_tensor.shape[axis]

            ranges = tuple(range(num_tiles) for num_tiles in other_tensor.num_tiles_per_axis())
            for indices in itertools.product(*ranges):
                new_indices = list(indices)
                new_indices[axis] += offset
                new_indices = tuple(new_indices)
                index_to_tile[new_indices] = other_tensor.index_to_tile[indices]

            offset += other_tensor.num_tiles_per_axis()[axis]

        tilized_tensor = TilizedTensor(
            level_name=first_tensor.level_name,
            shape=tuple(shape),
            tile_shape=first_tensor.tile_shape,
            index_to_tile=pmap(index_to_tile),
        )
        return tilized_tensor
    else:
        tile = np.concatenate([tensor.tile for tensor in tensors], axis)
        return Tile(tile=tile)


def retilize_tensor(tensor_to_retilize: TilizedTensor, tilization_hierarchy: list[TilizationLevel]):
    if isinstance(tensor_to_retilize, Tile):
        return tensor_to_retilize

    tilization_level, *remaining_tilization_hierarchy = tilization_hierarchy

    shape = list(tensor_to_retilize.shape)
    tile_shape = list(tensor_to_retilize.tile_shape)

    tilized_tensor = tensor_to_retilize
    for axis_to_retilize, (dim, target_dim) in enumerate(zip(tilized_tensor.tile_shape, tilization_level.tile_shape)):
        index_to_tile = {}
        if dim < target_dim:
            assert target_dim % dim == 0
            factor = target_dim // dim

            ranges = tuple(
                range(num_tiles // factor) if axis_to_retilize == axis else range(num_tiles)
                for axis, num_tiles in enumerate(tilized_tensor.num_tiles_per_axis())
            )
            for indices in itertools.product(*ranges):
                concatenate_inputs = []
                input_indices = list(indices)
                input_indices[axis_to_retilize] *= factor
                for _ in range(factor):
                    concatenate_inputs.append(tilized_tensor[tuple(input_indices)])
                    input_indices[axis_to_retilize] += 1
                index_to_tile[indices] = concatenate_tensors(concatenate_inputs, axis=axis_to_retilize)

        elif dim > target_dim:
            assert dim % target_dim == 0
            factor = dim // target_dim

            ranges = tuple(
                range(num_tiles * factor) if axis_to_retilize == axis else range(num_tiles)
                for axis, num_tiles in enumerate(tilized_tensor.num_tiles_per_axis())
            )
            for new_indices in itertools.product(*ranges):
                indices = list(new_indices)
                indices[axis_to_retilize] //= factor
                indices = tuple(indices)
                if isinstance(tensor_to_retilize[(0, 0, 0)], TilizedTensor):
                    num_tiles = target_dim // tensor_to_retilize[(0, 0, 0)].tile_shape[axis_to_retilize]
                    slice_size = tensor_to_retilize[(0, 0, 0)].tile_shape[axis_to_retilize]
                else:
                    num_tiles = 1
                    slice_size = target_dim

                start = new_indices[axis_to_retilize] % factor
                start *= num_tiles
                end = start + num_tiles

                index_to_tile[new_indices] = slice_tensors(
                    tilized_tensor[indices],
                    axis=axis_to_retilize,
                    start=start,
                    end=end,
                    slice_size=slice_size,
                )
        else:
            continue

        tile_shape[axis_to_retilize] = tilization_level.tile_shape[axis_to_retilize]
        tilized_tensor = TilizedTensor(
            level_name=tilized_tensor.level_name,
            shape=tuple(shape),
            tile_shape=tuple(tile_shape),
            index_to_tile=pmap(index_to_tile),
        )
        assert math.prod(tilized_tensor.num_tiles_per_axis()) == len(tilized_tensor.index_to_tile)

    # Retilize sub-tiles
    ranges = tuple(range(num_tiles) for axis, num_tiles in enumerate(tilized_tensor.num_tiles_per_axis()))
    index_to_tile = {}
    for indices in itertools.product(*ranges):
        index_to_tile[indices] = retilize_tensor(tilized_tensor.index_to_tile[indices], remaining_tilization_hierarchy)
    tilized_tensor = tilized_tensor.set(index_to_tile=pmap(index_to_tile))

    return tilized_tensor


def initialize_cache(inputs, tile_views):
    cache = {}
    for parray, array in inputs.items():
        tile_view = tile_views[parray]
        cache[(parray.node, parray.output_index)] = tilize_tensor(array, tile_view.hierarchy)
    return cache


def tilize(
    *output_vars,
    tile_views,
    inputs=pmap(),
    initialize_cache_function=initialize_cache,
    return_cache=False,
):
    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    cache = initialize_cache_function(inputs, tile_views)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_tensors = [cache[operand] for operand in get_operands(graph, node)]

        if instruction.__class__.__name__ == "matmul":
            output_tensor = input_tensors[0] @ input_tensors[1]
        elif instruction.__class__.__name__ == "sum":
            output_tensor = input_tensors[0].sum(axis=instruction.axis, keepdims=instruction.keepdims)
        elif instruction.__class__.__name__ == "add":
            input_0, input_1 = input_tensors
            if input_0.hierarchy != input_1.hierarchy and math.prod(input_0.shape) == math.prod(input_1.shape):
                input_1 = retilize_tensor(input_1, input_0.hierarchy)
            output_tensor = input_0 + input_1
        elif instruction.__class__.__name__ == "subtract":
            input_0, input_1 = input_tensors
            if input_0.hierarchy != input_1.hierarchy and math.prod(input_0.shape) == math.prod(input_1.shape):
                input_1 = retilize_tensor(input_1, input_0.hierarchy)
            output_tensor = input_0 - input_1
        else:
            raise RuntimeError(f"Unrecognized instruction: {instruction}")

        if isinstance(output_tensor, TilizedTensor):
            cache[(node, 0)] = output_tensor
        elif isinstance(output_tensor, collections.abc.Iterable):
            for output_index, output_tensor in enumerate(output_tensor):
                cache[(node, output_index)] = output_tensor
        else:
            raise RuntimeError(f"Unsupported type: {type(output_tensor)}")

    result = [cache[(output_var.node, output_var.output_index)] for output_var in output_vars]
    if len(result) == 1:
        (result,) = result

    if return_cache:
        return result, cache
    return result


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

        new_offsets = [offset * tile_dim + index for offset, index, tile_dim in zip(offsets, indices, tile_shape)]

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


def to_flat_array(array: np.array, tile_metadata: TileMetadata, *, transpose_levels=None, order=None) -> np.ndarray:
    if transpose_levels is None:
        transpose_levels = set()

    tiles = to_tiles(array, tile_metadata, transpose_levels, order)
    flat_array = np.concatenate([tile.flatten() for tile in tiles]).astype(np.float32)
    return flat_array


def from_flat_array(flat_array: np.array, tile_metadata: TileMetadata) -> np.ndarray:
    tile_level = tile_metadata.hierarchy[-1]
    num_tiles = len(flat_array) / math.prod(tile_level.tile_shape)
    tiles = (tile.reshape(tile_level.tile_shape) for tile in np.array_split(flat_array, num_tiles))
    return from_tiles(tiles, tile_metadata)
