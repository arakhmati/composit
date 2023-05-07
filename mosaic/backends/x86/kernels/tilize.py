import math
import pathlib

import codegen as c
from mosaic.tilelab.layout import TransposedLayout

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name
from mosaic.backends.ctypes import get_ctype_string_from_numpy_dtype


def generate_kernel_source_file(path, array_tile_config: ArrayTileConfig, dtype):
    kernel_name = create_kernel_name(pathlib.Path(__file__).stem, array_tile_config, dtype)

    ctype_string = get_ctype_string_from_numpy_dtype(dtype)
    InputType = c.Type(ctype_string).const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
    OutputType = c.Type(ctype_string).pointer().restrict().aligned(MEMORY_ALIGNMENT)

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    body = generate_body(
        array_tile_config,
        arguments=[input_var, output_var],
        offset=c.literal(0),
        original_shape=array_tile_config.shape,
    )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".cpp"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(kernel_name),
                arguments=[input_var, output_var],
                body=body,
            ).extern_c(),
        ],
    )
    file.save()
    return kernel_name


def compute_offset(offset, indices, num_tiles_per_axis, next_level_volume):
    for axis, index in enumerate(indices):
        offset = offset + index * c.literal(math.prod(num_tiles_per_axis[axis + 1 :]))
    offset = offset * c.literal(next_level_volume)
    return offset


def compute_contiguous_indices(shape, all_indices):
    axes = [axis for axis, _ in enumerate(shape)]

    original_indices = []
    for axis in axes:
        index = None
        for indices in all_indices:
            if index is None:
                index = indices[axis]
            else:
                index = index + indices[axis]
        original_indices.append(index)

    result = c.literal(0)
    for axis, index in enumerate(original_indices):
        offset = c.literal(math.prod(shape[axis + 1 :]))
        result = result + index * offset
    return result


def compute_original_indices(indices, tile_shape):
    return [index * c.literal(tile_shape[axis]) for axis, index in enumerate(indices)]


def transpose_sequence(sequence, axes):
    new_sequence = list(sequence)
    for axis, new_axis in enumerate(axes):
        new_sequence[new_axis] = sequence[axis]
    return tuple(new_sequence)


def generate_body(array_tile_config, arguments, offset, original_shape, all_indices=None):
    if all_indices is None:
        all_indices = []

    input_var, output_var = arguments

    level_name = array_tile_config.level_name

    num_tiles_per_axis = array_tile_config.num_tiles_per_axis()
    ranges = tuple(num_tiles for num_tiles in num_tiles_per_axis)
    indices = [c.variable(c.Type("uint32_t"), f"{level_name}_index_{axis}") for axis, _ in enumerate(ranges)]
    original_indices = list(indices)

    if isinstance(array_tile_config.layout, TransposedLayout):
        order = array_tile_config.layout.order
        num_tiles_per_axis = transpose_sequence(num_tiles_per_axis, order)
        ranges = transpose_sequence(ranges, order)
        indices = transpose_sequence(indices, order)

    if isinstance(array_tile_config, ArrayTileConfig):
        all_indices += [compute_original_indices(original_indices, array_tile_config.tile_shape)]
        next_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_offset")

        declare_next_offset = next_offset << (
            compute_offset(offset, indices, num_tiles_per_axis, math.prod(array_tile_config.tile_shape))
        )

        inner_loop_body = c.block(declare_next_offset)
        inner_loop_body += generate_body(
            array_tile_config[tuple(0 for _ in range(len(array_tile_config.shape)))],
            arguments=[input_var, output_var],
            offset=next_offset,
            original_shape=original_shape,
            all_indices=all_indices,
        )
    else:
        all_indices += [original_indices]
        tilized_index = c.variable(c.Type("uint32_t"), "tilized_index")
        declare_tilized_index = tilized_index << (compute_offset(offset, indices, num_tiles_per_axis, 1))

        original_index = c.variable(c.Type("uint32_t"), "original_index")
        declare_original_index = original_index << (compute_contiguous_indices(original_shape, all_indices))

        inner_loop_body = c.block(
            declare_tilized_index,
            declare_original_index,
            c.assign(output_var[tilized_index], input_var[original_index]),
        )

    loop = inner_loop_body
    for index, num_input_iterations in zip(reversed(indices), reversed(ranges)):
        loop = c.ForLoop(
            c.Declare(index, c.literal(0)),
            index < c.literal(num_input_iterations),
            c.add_in_place(index, c.literal(1)),
            loop,
        )
        loop = c.block(loop)

    return loop
