import math
import pathlib

import codegen as c

from mosaic.tilelab.tile import TileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name

InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)

operation_to_c_function = {
    "exp": "expf",
    "sqrt": "sqrtf",
    "gelu": "geluf",
}


def generate_module(input_tile_configs, output_tile_config, input_dtypes, output_dtype, operation):
    input_tile_config, *_ = input_tile_configs
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_tile_config,
        operation,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        members=[
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=[input_var, output_var],
                body_function=generate_body,
                input_tile_config=input_tile_config,
                operation=operation,
            ).extern_c()
        ],
    )
    return kernel_name, module


def generate_body(arguments, *, input_tile_config, operation):
    lambdas = []
    if operation == "gelu":
        lambdas = [
            c.Lambda("auto cdf = [](float input) { return 0.5 * (1 + erff(input / sqrtf(2))); };"),
            c.Lambda("auto geluf = [&cdf](float input) { return input * cdf(input); };"),
        ]

    body = c.block(*lambdas)
    body += generate_loops(
        arguments, input_tile_config=input_tile_config, operation=operation, offsets=dict(input=c.literal(0))
    )
    return body


def compute_index(offset, indices):
    return sum(indices, offset)


def compute_offset(sequence, axis, offset):
    return math.prod(sequence[axis:]) * offset


def generate_loops(arguments, *, input_tile_config, operation, offsets):
    level_name = input_tile_config.level_name

    num_tiles_per_axis = tuple(num_tiles for num_tiles in input_tile_config.num_tiles_per_axis())
    indices = [
        c.variable(c.Type("uint32_t"), f"{level_name}_index_{axis}") for axis, _ in enumerate(num_tiles_per_axis)
    ]
    tile_volume = math.prod(input_tile_config.tile_shape)

    index = c.variable(c.Type("uint32_t"), f"{level_name}_index")
    declare_index = index << (compute_index(offsets["input"], indices))

    loop_body = c.block(declare_index)
    if isinstance(input_tile_config, TileConfig):
        loop_body += generate_loops(
            arguments,
            input_tile_config=input_tile_config.next_level_tile_config,
            operation=operation,
            offsets=dict(input=index),
        )
    else:
        input_var, output_var = arguments
        computation = c.assign(output_var[index], c.invoke(operation_to_c_function[operation], input_var[index]))
        loop_body += c.block(computation)

    for reversed_axis, (index, num_iterations) in enumerate(zip(reversed(indices), reversed(num_tiles_per_axis))):
        axis = len(indices) - reversed_axis - 1
        increment = compute_offset(num_tiles_per_axis, axis + 1, tile_volume)
        limit = compute_offset(num_tiles_per_axis, axis, tile_volume)

        loop = c.ForLoop(
            c.Declare(index, c.literal(0)),
            index < c.literal(limit),
            c.add_in_place(index, c.literal(increment)),
            loop_body,
        )
        loop_body = c.block(loop)

    return loop_body
