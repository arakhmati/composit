import math
import operator
import pathlib

import codegen as c

from mosaic.tilelab.tile import TileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


def generate_module(input_tile_configs, output_tile_config, input_dtypes, output_dtype, operation):
    input_tile_config, *_ = input_tile_configs
    kwargs = dict(mean_coefficient=1.0 / (math.prod(input_tile_config.shape) / math.prod(output_tile_config.shape)))

    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_tile_config,
        output_tile_config,
        operation,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    body = initialize_output(output_tile_config, operation=operation, arguments=[input_var, output_var])

    body += generate_body(
        arguments=[input_var, output_var],
        input_tile_config=input_tile_config,
        output_tile_config=output_tile_config,
        operation=operation,
        offsets=dict(input=c.literal(0), output=c.literal(0)),
        **kwargs,
    )

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        members=[
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(kernel_name),
                arguments=[input_var, output_var],
                body=body,
            ).extern_c()
        ],
    )
    return kernel_name, module


def initialize_output(output_tile_config, operation, arguments):
    input_var, output_var = arguments

    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(output_tile_config.shape)

    value = c.literal(0)
    if operation == "max":
        value = c.literal("-INFINITY")

    loop = c.block(
        c.ForLoop(
            c.Declare(index, c.literal(0)),
            index < c.literal(num_iterations),
            c.add_in_place(index, c.literal(1)),
            c.block(c.assign(output_var[index], value)),
        )
    )

    body = c.If(c.not_equals(input_var, output_var), loop)

    return c.block(body)


def compute_offset(offset, indices, num_tiles_per_axis, next_level_volume):
    for axis, index in enumerate(indices):
        offset = offset + index * c.literal(math.prod(num_tiles_per_axis[axis + 1 :]))
    offset = offset * c.literal(next_level_volume)
    return offset


def generate_body(arguments, input_tile_config, output_tile_config, operation, offsets, **kwargs):
    input_var, output_var = arguments

    level_name = input_tile_config.level_name

    input_num_tiles_per_axis = input_tile_config.num_tiles_per_axis()
    output_num_tiles_per_axis = output_tile_config.num_tiles_per_axis()

    input_ranges = tuple(num_tiles for num_tiles in input_num_tiles_per_axis)
    output_ranges = tuple(num_tiles for num_tiles in output_num_tiles_per_axis)

    input_indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_index_input_{axis}") for axis, _ in enumerate(input_ranges)
    ]
    output_indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_index_output_{axis}") for axis, _ in enumerate(output_ranges)
    ]

    if isinstance(input_tile_config, TileConfig):
        next_input_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_input_offset")
        next_output_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_output_offset")

        declare_next_input_offset = next_input_offset << (
            compute_offset(
                offsets["input"], input_indices, input_num_tiles_per_axis, math.prod(input_tile_config.tile_shape)
            )
        )

        declare_next_output_offset = next_output_offset << (
            compute_offset(
                offsets["output"],
                output_indices,
                output_num_tiles_per_axis,
                math.prod(output_tile_config.tile_shape),
            )
        )

        inner_loop_body = c.block(declare_next_input_offset, declare_next_output_offset)
        inner_loop_body += generate_body(
            arguments=arguments,
            input_tile_config=input_tile_config.next_level_tile_config,
            output_tile_config=output_tile_config.next_level_tile_config,
            operation=operation,
            offsets=dict(input=next_input_offset, output=next_output_offset),
            **kwargs,
        )
    else:
        input_index = c.variable(c.Type("uint32_t"), "input_index")
        output_index = c.variable(c.Type("uint32_t"), "output_index")

        declare_index = input_index << (compute_offset(offsets["input"], input_indices, input_num_tiles_per_axis, 1))
        declare_output_index = output_index << (
            compute_offset(offsets["output"], output_indices, output_num_tiles_per_axis, 1)
        )

        lambdas = []
        if operation == "sum":
            reduction_step = operator.add(output_var[output_index], input_var[input_index])
        elif operation == "mean":
            reduction_step = operator.add(
                output_var[output_index], input_var[input_index] * c.literal(kwargs["mean_coefficient"])
            )
        elif operation == "max":
            lambdas = [
                c.Lambda(
                    "auto _maxf = [](auto input_a, auto input_b) { return input_a > input_b ? input_a : input_b; };"
                )
            ]
            reduction_step = c.invoke("_maxf", output_var[output_index], input_var[input_index])
        else:
            raise NotImplementedError

        inner_loop_body = c.block(
            declare_index, declare_output_index, *lambdas, c.assign(output_var[output_index], reduction_step)
        )

    loop = inner_loop_body
    for input_index, output_index, num_input_iterations, num_output_iterations in zip(
        reversed(input_indices), reversed(output_indices), reversed(input_ranges), reversed(output_ranges)
    ):
        declare_output_index = output_index << (
            input_index if num_input_iterations == num_output_iterations else c.literal(0)
        )
        loop = c.ForLoop(
            c.Declare(input_index, c.literal(0)),
            input_index < c.literal(num_input_iterations),
            c.add_in_place(input_index, c.literal(1)),
            c.block(declare_output_index, loop),
        )

    return c.block(loop)
