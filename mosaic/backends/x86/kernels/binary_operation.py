import math
import operator
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)

operation_to_python_operator = {
    "add": operator.add,
    "subtract": operator.sub,
    "multiply": operator.mul,
    "divide": operator.truediv,
}


def generate_kernel_source_file(path, input_a_array_tile_config, input_b_array_tile_config: ArrayTileConfig, operation):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_a_array_tile_config,
        input_b_array_tile_config,
        operation,
    )

    input_a_var = c.variable(InputType, "input_a_var")
    input_b_var = c.variable(InputType, "input_b_var")
    output_var = c.variable(OutputType, "output_var")
    arguments = (input_a_var, input_b_var, output_var)

    if input_a_array_tile_config == input_b_array_tile_config:
        body_function = generate_optimized_body
        body_function_kwargs = dict(
            input_a_array_tile_config=input_a_array_tile_config,
            operation=operation,
        )
    else:
        body_function = generate_body
        body_function_kwargs = dict(
            input_a_array_tile_config=input_a_array_tile_config,
            input_b_array_tile_config=input_b_array_tile_config,
            operation=operation,
            offsets=dict(a=c.literal(0), b=c.literal(0)),
        )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".cpp"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.NewLine(),
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=arguments,
                body_function=body_function,
                **body_function_kwargs,
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


def generate_optimized_body(arguments, *, input_a_array_tile_config, operation):
    input_a_var, input_b_var, output_var = arguments

    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(input_a_array_tile_config.shape)

    loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(
            c.assign(output_var[index], operation_to_python_operator[operation](input_a_var[index], input_b_var[index]))
        ),
    )

    return c.block(loop)


def generate_body(arguments, *, input_a_array_tile_config, input_b_array_tile_config, operation, offsets):
    input_a_var, input_b_var, output_var = arguments

    level_name = input_a_array_tile_config.level_name

    a_num_tiles_per_axis = input_a_array_tile_config.num_tiles_per_axis()
    b_num_tiles_per_axis = input_b_array_tile_config.num_tiles_per_axis()

    a_ranges = tuple(num_tiles for num_tiles in a_num_tiles_per_axis)
    b_ranges = tuple(num_tiles for num_tiles in b_num_tiles_per_axis)

    a_indices = [c.variable(c.Type("uint32_t"), f"{level_name}_index_a_{axis}") for axis, _ in enumerate(a_ranges)]
    b_indices = [c.variable(c.Type("uint32_t"), f"{level_name}_index_b_{axis}") for axis, _ in enumerate(b_ranges)]

    if isinstance(input_a_array_tile_config, ArrayTileConfig):
        next_a_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_a_offset")
        next_b_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_b_offset")

        declare_next_a_offset = next_a_offset << (
            compute_offset(
                offsets["a"], a_indices, a_num_tiles_per_axis, math.prod(input_a_array_tile_config.tile_shape)
            )
        )

        declare_next_b_offset = next_b_offset << (
            compute_offset(
                offsets["b"], b_indices, b_num_tiles_per_axis, math.prod(input_b_array_tile_config.tile_shape)
            )
        )

        inner_loop_body = c.block(declare_next_a_offset, declare_next_b_offset)
        inner_loop_body += generate_body(
            arguments,
            input_a_array_tile_config=input_a_array_tile_config.next_level(),
            input_b_array_tile_config=input_b_array_tile_config.next_level(),
            operation=operation,
            offsets=dict(a=next_a_offset, b=next_b_offset),
        )
    else:
        a_index = c.variable(c.Type("uint32_t"), "a_index")
        b_index = c.variable(c.Type("uint32_t"), "b_index")

        declare_index = a_index << (compute_offset(offsets["a"], a_indices, a_num_tiles_per_axis, 1))
        declare_b_index = b_index << (compute_offset(offsets["b"], b_indices, b_num_tiles_per_axis, 1))

        inner_loop_body = c.block(
            declare_index,
            declare_b_index,
            c.assign(
                output_var[a_index], operation_to_python_operator[operation](input_a_var[a_index], input_b_var[b_index])
            ),
        )

    loop = inner_loop_body
    for a_axis, (a_index, num_a_iterations) in enumerate(zip(reversed(a_indices), reversed(a_ranges))):
        b_axis = len(b_ranges) - 1 - a_axis
        if b_axis >= 0:
            b_index = b_indices[b_axis]
            num_b_iterations = b_ranges[b_axis]
            declare_b_index = c.block(b_index << (a_index if num_a_iterations == num_b_iterations else c.literal(0)))
            body = declare_b_index + loop
        else:
            body = loop

        loop = c.ForLoop(
            c.Declare(a_index, c.literal(0)),
            a_index < c.literal(num_a_iterations),
            c.add_in_place(a_index, c.literal(1)),
            body,
        )
        loop = c.block(loop)

    return loop
