import collections
import itertools
import operator

import numpy as np
from pyrsistent import PClass, field, pmap, pvector
from toolz.itertoolz import first

import persistent_numpy as pnp
from persistent_numpy.multidigraph import compose_all, topological_traversal
from persistent_numpy.nn import evaluate
from persistent_numpy.numpy.core import get_operands


class TilizationLevel(PClass):
    level_name = field()
    tile_shape = field()


class TilizedTensor(PClass):
    levels = field()
    shape = field()
    tile_shape = field()
    tiles = field()

    @property
    def level_name(self):
        return self.levels[0].level_name

    @property
    def num_levels(self):
        return len(self.levels)

    def num_tiles_per_axis(self):
        return tuple(dim // tile_dim for dim, tile_dim in zip(self.shape, self.tile_shape))

    def __getitem__(self, index):
        return self.tiles[index]

    def __repr__(self):
        result = f"{self.__class__.__name__}(level_name={self.level_name}, num_levels={self.num_levels}, shape={self.shape}, tile_shape={self.tile_shape}, num_tiles={len(self.tiles)})"
        return f"{result}\n{first(self.tiles.values())}"

    def __eq__(self, other):
        if self.num_tiles_per_axis() != other.num_tiles_per_axis():
            return False
        if len(self.tiles) != len(other.tiles):
            return False
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())

        equals = True
        for index in itertools.product(*ranges):
            a_tile = self[index]
            b_tile = other[index]
            equals &= a_tile == b_tile
        return equals

    def binary_operation(self, other, operation):
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())

        tiles = {}
        for index in itertools.product(*ranges):
            b_index = tuple(i % num_tiles for i, num_tiles in zip(index, other.num_tiles_per_axis()))
            a_tile = self[index]
            b_tile = other[b_index]
            tile = operation(a_tile, b_tile)
            tiles[index] = tile

        return TilizedTensor(
            levels=self.levels,
            shape=self.shape,
            tile_shape=self.tile_shape,
            tiles=tiles,
        )

    def __add__(self, other):
        return self.binary_operation(other, operator.add)

    def __sub__(self, other):
        return self.binary_operation(other, operator.sub)

    def __matmul__(self, other):
        a_num_tiles_per_axis = self.num_tiles_per_axis()
        b_num_tiles_per_axis = other.num_tiles_per_axis()
        a_ranges = (range(num_tiles) for num_tiles in a_num_tiles_per_axis)
        _, n_range = (range(num_tiles) for num_tiles in b_num_tiles_per_axis)

        tiles = {}
        for *a_indices, n in itertools.product(*a_ranges, n_range):
            a_indices = tuple(a_indices)
            *batch_indices, m, k = a_indices
            batch_indices = tuple(batch_indices)
            b_indices = (k, n)
            output_indices = batch_indices + (m, n)
            a_tile = self[a_indices]
            b_tile = other[b_indices]
            tile = a_tile @ b_tile
            if output_indices not in tiles:
                tiles[output_indices] = tile
            else:
                tiles[output_indices] += tile

        return TilizedTensor(
            levels=self.levels,
            shape=self.shape[:-1] + other.shape[-1:],
            tile_shape=self.tile_shape[:-1] + other.tile_shape[-1:],
            tiles=tiles,
        )

    def sum(self, axis, keepdims=True):
        num_tiles_per_axis = self.num_tiles_per_axis()
        ranges = (range(num_tiles) for num_tiles in num_tiles_per_axis)

        tiles = {}
        for index in itertools.product(*ranges):
            output_index = list(index)
            output_index[axis] = 0
            output_index = tuple(output_index)
            input_tile = self[index].sum(axis=axis, keepdims=keepdims)
            if output_index not in tiles:
                output_tile = input_tile
            else:
                output_tile = tiles[output_index] + input_tile
            tiles[output_index] = output_tile

        shape = list(self.shape)
        shape[axis] = output_tile.shape[axis]
        shape = tuple(shape)

        tile_shape = list(self.tile_shape)
        tile_shape[axis] = shape[axis]
        tile_shape = tuple(tile_shape)

        return TilizedTensor(
            levels=self.levels,
            shape=shape,
            tile_shape=tile_shape,
            tiles=tiles,
        )

    def evaluate(self, inputs=None):
        output = np.zeros(self.shape)
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())
        for tile_indices in itertools.product(*ranges):
            tile = self[tile_indices].evaluate(inputs=inputs)
            tile_slices = tuple(
                slice(tile_index * tile_dim, (tile_index + 1) * tile_dim)
                for tile_index, tile_dim in zip(tile_indices, self.tile_shape)
            )
            output[tile_slices] = tile
        return output

    def __iter__(self):
        ranges = (range(num_tiles) for num_tiles in self.num_tiles_per_axis())
        for tile_indices in itertools.product(*ranges):
            yield from self[tile_indices]


class Tile(PClass):
    tile = field()

    @property
    def shape(self):
        return self.tile.shape

    def __repr__(self):
        return f"{self.__class__.__name__}(shape={self.shape})"

    def __eq__(self, other) -> bool:
        return self.tile.shape == other.tile.shape

    def __add__(self, other):
        return Tile(tile=self.tile + other.tile)

    def __sub__(self, other):
        return Tile(tile=self.tile - other.tile)

    def __matmul__(self, other):
        return Tile(tile=self.tile @ other.tile)

    def sum(self, axis, keepdims):
        result = pnp.sum(self.tile, axis=axis, keepdims=keepdims)
        """
        To pad, or not to pad, that is the question
        pad_width = [[0, 0] for _ in self.tile.shape]
        pad_width[axis][1] = self.tile.shape[axis] - result.shape[axis]
        result = np.pad(result, pad_width)
        """
        return Tile(tile=result)

    def evaluate(self, inputs=None):
        if inputs is None:
            inputs = {}
        return evaluate(self.tile, inputs=inputs)

    def __iter__(self):
        yield self.tile


def tilize_tensor(
    tensor,
    tilization_hierarchy: list[TilizationLevel],
):

    tilization_level, *remaining_tilization_hierarchy = tilization_hierarchy
    tile_shape = tilization_level.tile_shape

    if len(tensor.shape) != len(tile_shape):
        raise RuntimeError("Shapes must have the same rank")

    ranges = (range(0, tensor_dim, tile_dim) for tensor_dim, tile_dim in zip(tensor.shape, tile_shape))

    tiles = {}
    for indices in itertools.product(*ranges):
        tile_slices = tuple(
            slice(tensor_index, tensor_index + tile_dim) for tensor_index, tile_dim in zip(indices, tile_shape)
        )
        tile_indices = tuple(tensor_index // tile_dim for tensor_index, tile_dim in zip(indices, tile_shape))
        tile = tensor[tile_slices]

        if remaining_tilization_hierarchy:
            tiles[tile_indices] = tilize_tensor(tile, remaining_tilization_hierarchy)
        else:
            tiles[tile_indices] = Tile(tile=pnp.asarray(tile))

    tilized_tensor = TilizedTensor(
        levels=pvector(tilization_hierarchy),
        shape=tensor.shape,
        tile_shape=tile_shape,
        tiles=pmap(tiles),
    )

    return tilized_tensor


def slice_tensors(tensor, axis, start, end, slice_size):
    shape = list(tensor.shape)
    shape[axis] = (end - start) * slice_size

    if isinstance(tensor, TilizedTensor):

        tiles = {}
        ranges = tuple(range(num_tiles) for num_tiles in tensor.num_tiles_per_axis())
        for indices in itertools.product(*ranges):
            if not (start <= indices[axis] < end):
                continue
            new_indices = list(indices)
            new_indices[axis] -= start
            new_indices = tuple(new_indices)
            tiles[new_indices] = tensor.tiles[indices]

        tilized_tensor = TilizedTensor(
            levels=tensor.levels,
            shape=tuple(shape),
            tile_shape=tensor.tile_shape,
            tiles=pmap(tiles),
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
        tiles = dict(first_tensor.tiles)
        offset = first_tensor.num_tiles_per_axis()[axis]

        for other_tensor in tensors[1:]:
            shape[axis] += other_tensor.shape[axis]

            ranges = tuple(range(num_tiles) for num_tiles in other_tensor.num_tiles_per_axis())
            for indices in itertools.product(*ranges):
                new_indices = list(indices)
                new_indices[axis] += offset
                new_indices = tuple(new_indices)
                tiles[new_indices] = other_tensor.tiles[indices]

            offset += other_tensor.num_tiles_per_axis()[axis]

        tilized_tensor = TilizedTensor(
            levels=first_tensor.levels,
            shape=tuple(shape),
            tile_shape=first_tensor.tile_shape,
            tiles=pmap(tiles),
        )
        return tilized_tensor
    else:
        tile = pnp.concatenate([tensor.tile for tensor in tensors], axis)
        return Tile(tile=tile)


def retilize_tensor(tensor_to_retilize: TilizedTensor, target_tensor: TilizedTensor):
    if isinstance(tensor_to_retilize, Tile):
        return tensor_to_retilize

    if tensor_to_retilize.level_name != target_tensor.level_name:
        raise RuntimeError("Level names must match")

    if tensor_to_retilize.num_levels != target_tensor.num_levels:
        raise RuntimeError("Number of levels must match")

    shape = list(tensor_to_retilize.shape)
    tile_shape = list(tensor_to_retilize.tile_shape)

    tilized_tensor = tensor_to_retilize
    for axis_to_retilize, (dim, target_dim) in enumerate(zip(tilized_tensor.tile_shape, target_tensor.tile_shape)):
        tiles = {}
        if dim < target_dim:
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
                tiles[indices] = concatenate_tensors(concatenate_inputs, axis=axis_to_retilize)

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
                start = new_indices[axis_to_retilize] % factor
                end = start + 1
                tiles[new_indices] = slice_tensors(
                    tilized_tensor[tuple(indices)], axis=axis_to_retilize, start=start, end=end, slice_size=target_dim
                )
        else:
            continue

        shape[axis_to_retilize] = target_tensor.shape[axis_to_retilize]
        tile_shape[axis_to_retilize] = target_tensor.tile_shape[axis_to_retilize]
        tilized_tensor = TilizedTensor(
            levels=target_tensor.levels,
            shape=tuple(shape),
            tile_shape=tuple(tile_shape),
            tiles=pmap(tiles),
        )

    # Retilize sub-tiles
    ranges = tuple(range(num_tiles) for axis, num_tiles in enumerate(tilized_tensor.num_tiles_per_axis()))
    tiles = {}
    for indices in itertools.product(*ranges):
        tiles[indices] = retilize_tensor(tilized_tensor.tiles[indices], target_tensor.tiles[indices])
    tilized_tensor = tilized_tensor.set(tiles=pmap(tiles))

    return tilized_tensor


def initialize_cache(graph, inputs):
    cache = {}
    for parray, tilization_levels in inputs.items():
        cache[(parray.node, parray.output_index)] = tilize_tensor(parray, tilization_levels)
    return cache


def tilize(
    *output_vars,
    inputs,
    initialize_cache_function=initialize_cache,
):

    graph = compose_all(*tuple(output_var.graph for output_var in output_vars))

    add_count = 0

    cache = initialize_cache_function(graph, inputs)
    for node in topological_traversal(graph):
        if (node, 0) in cache:
            continue
        instruction = graph.nodes[node]["instruction"]
        input_arrays = [cache[operand] for operand in get_operands(graph, node)]

        if instruction.__class__.__name__ == "matmul":
            instruction_output = input_arrays[0] @ input_arrays[1]
        elif instruction.__class__.__name__ == "sum":
            instruction_output = input_arrays[0].sum(axis=instruction.axis, keepdims=instruction.keepdims)
        elif instruction.__class__.__name__ == "add":
            if input_arrays[0].levels != input_arrays[1].levels:
                if add_count == 0:
                    input_arrays[0] = retilize_tensor(input_arrays[0], input_arrays[1])
                else:
                    input_arrays[1] = retilize_tensor(input_arrays[1], input_arrays[0])
            instruction_output = input_arrays[0] + input_arrays[1]
            add_count += 1
        elif instruction.__class__.__name__ == "subtract":
            if input_arrays[0].levels != input_arrays[1].levels:
                input_arrays[1] = retilize_tensor(input_arrays[1], input_arrays[0])
            instruction_output = input_arrays[0] - input_arrays[1]
        else:
            raise RuntimeError(f"Unrecognized instruction: {instruction}")

        if isinstance(instruction_output, TilizedTensor):
            cache[(node, 0)] = instruction_output
        elif isinstance(instruction_output, collections.abc.Iterable):
            for output_index, instruction_output in enumerate(instruction_output):
                cache[(node, output_index)] = instruction_output
        else:
            raise RuntimeError(f"Unsupported type: {type(instruction_output)}")

    result = [cache[(output_var.node, output_var.output_index)] for output_var in output_vars]
    if len(result) == 1:
        (result,) = result

    return result
