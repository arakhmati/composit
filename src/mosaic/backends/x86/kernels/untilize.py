import math
import pathlib

import codegen as c

from mosaic.tilelab.tile import TileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name
from mosaic.backends.ctypes import get_ctype_string_from_numpy_dtype


def generate_module(input_tile_configs, output_tile_config, input_dtypes, output_dtype):
    kernel_name = create_kernel_name(pathlib.Path(__file__).stem, output_tile_config, output_dtype)

    ctype_string = get_ctype_string_from_numpy_dtype(output_dtype)
    InputType = c.Type(ctype_string).const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
    OutputType = c.Type(ctype_string).pointer().restrict().aligned(MEMORY_ALIGNMENT)

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        members=[
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=[input_var, output_var],
                body_function=generate_body,
                tile_config=output_tile_config,
                offset=c.literal(0),
                original_shape=output_tile_config.shape,
            ).extern_c()
        ],
    )
    return kernel_name, module


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


def generate_body(arguments, tile_config, offset, original_shape, all_indices=None):
    if all_indices is None:
        all_indices = []

    input_var, output_var = arguments

    level_name = tile_config.level_name

    num_tiles_per_axis = tile_config.num_tiles_per_axis()
    ranges = tuple(num_tiles for num_tiles in num_tiles_per_axis)

    indices = [c.variable(c.Type("uint32_t"), f"{level_name}_index_{axis}") for axis, _ in enumerate(ranges)]

    if isinstance(tile_config, TileConfig):
        all_indices += [compute_original_indices(indices, tile_config.tile_shape)]
        next_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_offset")

        declare_next_offset = next_offset << (
            compute_offset(offset, indices, num_tiles_per_axis, math.prod(tile_config.tile_shape))
        )

        inner_loop_body = c.block(declare_next_offset)
        inner_loop_body += generate_body(
            arguments=arguments,
            tile_config=tile_config.next_level_tile_config,
            offset=next_offset,
            original_shape=original_shape,
            all_indices=all_indices,
        )
    else:
        all_indices += [indices]
        tilized_index = c.variable(c.Type("uint32_t"), "tilized_index")
        declare_tilized_index = tilized_index << (compute_offset(offset, indices, num_tiles_per_axis, 1))

        original_index = c.variable(c.Type("uint32_t"), "original_index")
        declare_original_index = original_index << (compute_contiguous_indices(original_shape, all_indices))

        inner_loop_body = c.block(
            declare_tilized_index,
            declare_original_index,
            c.assign(output_var[original_index], input_var[tilized_index]),
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
