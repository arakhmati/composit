import math
import pathlib

import codegen as c

from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("uint64_t").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
WeightsType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


def generate_module(input_tile_configs, output_tile_config, input_dtypes, output_dtype):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        output_tile_config,
    )

    input_var = c.variable(InputType, "input_var")
    weights = c.variable(WeightsType, "weights")
    output_var = c.variable(OutputType, "output_var")

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        members=[
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=[input_var, weights, output_var],
                body_function=generate_body,
                output_tile_config=output_tile_config,
            ).extern_c()
        ],
    )
    return kernel_name, module


def generate_body(
    arguments,
    output_tile_config,
):
    input_var, weights, output_var = arguments
    batch_size, sequence_size, hidden_size = output_tile_config.shape

    batch_size_index = c.variable(c.Type("uint32_t"), "batch_size_index")
    sequence_size_index = c.variable(c.Type("uint32_t"), "sequence_size_index")
    hidden_size_index = c.variable(c.Type("uint32_t"), "hidden_size_index")

    word_index = input_var[batch_size_index * c.literal(sequence_size) + sequence_size_index]
    weights_index = word_index * c.literal(hidden_size) + hidden_size_index

    tile_shape = output_tile_config.tile_shape
    _, tile_sequence_size, tile_hidden_size = tile_shape

    output_index = (
        batch_size_index * c.literal(sequence_size) * c.literal(hidden_size)
        + (
            (
                (sequence_size_index / c.literal(tile_sequence_size))
                * (c.literal(hidden_size) / c.literal(tile_hidden_size))
                + (hidden_size_index / c.literal(tile_hidden_size))
            )
            * c.literal(math.prod(tile_shape))
        )
        + (sequence_size_index % c.literal(tile_sequence_size)) * c.literal(tile_hidden_size)
        + (hidden_size_index % c.literal(tile_hidden_size))
    )

    hidden_size_loop = c.ForLoop(
        c.Declare(hidden_size_index, c.literal(0)),
        hidden_size_index < c.literal(hidden_size),
        c.add_in_place(hidden_size_index, c.literal(1)),
        c.block(c.assign(output_var[output_index], weights[weights_index])),
    )

    sequence_size_loop = c.ForLoop(
        c.Declare(sequence_size_index, c.literal(0)),
        sequence_size_index < c.literal(sequence_size),
        c.add_in_place(sequence_size_index, c.literal(1)),
        c.block(hidden_size_loop),
    )

    batch_size_loop = c.ForLoop(
        c.Declare(batch_size_index, c.literal(0)),
        batch_size_index < c.literal(batch_size),
        c.add_in_place(batch_size_index, c.literal(1)),
        c.block(sequence_size_loop),
    )

    return c.block(batch_size_loop)
