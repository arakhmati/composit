import math
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


def generate_kernel(path, input_array_tile_config, output_array_tile_config: ArrayTileConfig, axes):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_array_tile_config,
        output_array_tile_config,
        axes,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    body = generate_body(
        input_array_tile_config,
        output_array_tile_config,
        axes=axes,
        c_variables=dict(input_var=input_var, output_var=output_var),
        offsets=dict(input=c.literal(0), output=c.literal(0)),
    )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".c"),
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
            ),
        ],
    )
    file.save()
    return kernel_name


def transpose_sequence(sequence, axes):
    new_sequence = list(sequence)
    for axis, new_axis in enumerate(axes):
        new_sequence[new_axis] = sequence[axis]
    return tuple(new_sequence)


def compute_offset(offset, indices, num_tiles_per_axis, next_level_volume):
    for axis, index in enumerate(indices):
        offset = offset + index * c.literal(math.prod(num_tiles_per_axis[axis + 1 :]))
    offset = offset * c.literal(next_level_volume)
    return offset


def generate_body(input_array_tile_config, output_array_tile_config, axes, c_variables, offsets):
    level_name = input_array_tile_config.level_name

    input_num_tiles_per_axis = input_array_tile_config.num_tiles_per_axis()
    output_num_tiles_per_axis = output_array_tile_config.num_tiles_per_axis()

    input_ranges = tuple(num_tiles for num_tiles in input_num_tiles_per_axis)
    output_ranges = tuple(num_tiles for num_tiles in output_num_tiles_per_axis)

    input_indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_index_input_{axis}") for axis, _ in enumerate(input_ranges)
    ]
    output_indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_index_output_{axis}") for axis, _ in enumerate(output_ranges)
    ]

    if isinstance(input_array_tile_config, ArrayTileConfig):
        next_input_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_input_offset")
        next_output_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_output_offset")

        declare_next_input_offset = next_input_offset << (
            compute_offset(
                offsets["input"], input_indices, input_num_tiles_per_axis, math.prod(input_array_tile_config.tile_shape)
            )
        )

        declare_next_output_offset = next_output_offset << (
            compute_offset(
                offsets["output"],
                output_indices,
                output_num_tiles_per_axis,
                math.prod(output_array_tile_config.tile_shape),
            )
        )

        inner_loop_body = c.block(declare_next_input_offset, declare_next_output_offset)
        inner_loop_body += generate_body(
            input_array_tile_config[tuple(0 for _ in range(len(input_array_tile_config.shape)))],
            output_array_tile_config[tuple(0 for _ in range(len(output_array_tile_config.shape)))],
            axes=axes,
            c_variables=c_variables,
            offsets=dict(input=next_input_offset, output=next_output_offset),
        )
    else:
        input_index = c.variable(c.Type("uint32_t"), "input_index")
        output_index = c.variable(c.Type("uint32_t"), "output_index")

        declare_index = input_index << (compute_offset(offsets["input"], input_indices, input_num_tiles_per_axis, 1))
        declare_output_index = output_index << (
            compute_offset(offsets["output"], output_indices, output_num_tiles_per_axis, 1)
        )

        input_var = c_variables["input_var"][input_index]
        output_var = c_variables["output_var"][output_index]
        inner_loop_body = c.block(declare_index, declare_output_index, c.assign(output_var, input_var))

    loop = inner_loop_body
    for input_index, output_index, num_input_iterations in zip(
        reversed(input_indices), reversed(transpose_sequence(output_indices, axes)), reversed(input_ranges)
    ):
        loop = c.ForLoop(
            c.Declare(input_index, c.literal(0)),
            input_index < c.literal(num_input_iterations),
            c.add_in_place(input_index, c.literal(1)),
            loop + c.block(c.Statement(c.add_in_place(output_index, c.literal(1)))),
        )
        declare_output_index = c.block(output_index << (c.literal(0)))
        loop = declare_output_index + c.block(loop)

    return loop
