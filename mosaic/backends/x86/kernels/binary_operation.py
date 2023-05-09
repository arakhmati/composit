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


def generate_module(input_array_tile_configs, output_array_tile_config, input_dtypes, output_dtype, operation):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_array_tile_configs[0],
        input_array_tile_configs[1],
        operation,
    )

    input_var = c.variable(InputType, "input_var")
    broadcasted_input_var = c.variable(InputType, "broadcasted_input_var")
    output_var = c.variable(OutputType, "output_var")

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        members=[
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=[input_var, broadcasted_input_var, output_var],
                body_function=generate_body,
                input_tile_config=input_array_tile_configs[0],
                broadcasted_input_tile_config=input_array_tile_configs[1],
                operation=operation,
                offsets=dict(input=c.literal(0), broadcasted_input=c.literal(0)),
            ).extern_c()
        ],
    )
    return kernel_name, module


def compute_index(offset, indices):
    return sum(indices, offset)


def compute_offset(sequence, axis, offset):
    return math.prod(sequence[axis:]) * offset


def generate_body(arguments, *, input_tile_config, broadcasted_input_tile_config, operation, offsets):
    input_var, broadcasted_input_var, output_var = arguments

    level_name = input_tile_config.level_name

    ranges = tuple(num_tiles for num_tiles in input_tile_config.num_tiles_per_axis())
    broadcasted_ranges = tuple(num_tiles for num_tiles in broadcasted_input_tile_config.num_tiles_per_axis())

    indices = [c.variable(c.Type("uint32_t"), f"{level_name}_index_{axis}") for axis, _ in enumerate(ranges)]
    broadcasted_indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_broadcasted_index_{axis}")
        for axis, _ in enumerate(broadcasted_ranges)
    ]

    tile_volume = math.prod(input_tile_config.tile_shape)
    broadcasted_tile_volume = math.prod(broadcasted_input_tile_config.tile_shape)

    if isinstance(input_tile_config, ArrayTileConfig):
        next_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_offset")
        next_broadcasted_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_broadcasted_offset")

        declare_next_offset = next_offset << (compute_index(offsets["input"], indices))

        declare_next_broadcasted_offset = next_broadcasted_offset << (
            compute_index(offsets["broadcasted_input"], broadcasted_indices)
        )

        inner_loop_body = c.block(declare_next_offset, declare_next_broadcasted_offset)
        inner_loop_body += generate_body(
            arguments,
            input_tile_config=input_tile_config.next_level(),
            broadcasted_input_tile_config=broadcasted_input_tile_config.next_level(),
            operation=operation,
            offsets=dict(input=next_offset, broadcasted_input=next_broadcasted_offset),
        )
    else:
        index = c.variable(c.Type("uint32_t"), "index")
        broadcasted_index = c.variable(c.Type("uint32_t"), "broadcasted_index")

        declare_index = index << (compute_index(offsets["input"], indices))
        declare_broadcasted_index = broadcasted_index << (
            compute_index(offsets["broadcasted_input"], broadcasted_indices)
        )

        if operation == "divide":
            binary_operation = c.assign(
                output_var[index], input_var[index] * (c.literal(1.0) / broadcasted_input_var[broadcasted_index])
            )
        else:
            binary_operation = c.assign(
                output_var[index],
                operation_to_python_operator[operation](input_var[index], broadcasted_input_var[broadcasted_index]),
            )

        inner_loop_body = c.block(declare_index, declare_broadcasted_index, binary_operation)

    loop = inner_loop_body
    for reversed_axis, (index, num_iterations) in enumerate(zip(reversed(indices), reversed(ranges))):
        axis = len(indices) - reversed_axis - 1
        increment = compute_offset(ranges, axis + 1, tile_volume)
        limit = compute_offset(ranges, axis, tile_volume)

        broadcasted_axis = len(broadcasted_ranges) - reversed_axis - 1
        if broadcasted_axis >= 0:
            broadcast_increment = compute_offset(broadcasted_ranges, broadcasted_axis + 1, broadcasted_tile_volume)
            broadcast_factor = increment // broadcast_increment
            broadcasted_index = broadcasted_indices[broadcasted_axis]
            num_broadcasted_iterations = broadcasted_ranges[broadcasted_axis]
            if num_broadcasted_iterations == 1:
                broadcasted_index_value = c.literal(0)
            elif num_broadcasted_iterations == num_iterations:
                broadcasted_index_value = index / c.literal(broadcast_factor)
            else:
                raise NotImplementedError
            declare_broadcasted_index = c.block(broadcasted_index << broadcasted_index_value)
            body = declare_broadcasted_index + loop
        else:
            body = loop

        loop = c.ForLoop(
            c.Declare(index, c.literal(0)),
            index < c.literal(limit),
            c.add_in_place(index, c.literal(increment)),
            body,
        )
        loop = c.block(loop)

    return loop
