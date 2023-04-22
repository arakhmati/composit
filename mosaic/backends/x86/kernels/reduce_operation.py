import math
import operator
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


max_function = """     
float _maxf(float input_a, float input_b) {
    return input_a > input_b ? input_a : input_b;
}
"""


def generate_kernel(path, input_array_tile_config, output_array_tile_config: ArrayTileConfig, operation, axis):
    if isinstance(axis, tuple):
        axes = axis
    else:
        axes = [axis]

    kwargs = dict(
        num_reduced_elements=math.prod(input_array_tile_config.shape) / math.prod(output_array_tile_config.shape)
    )

    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_array_tile_config,
        output_array_tile_config,
        operation,
        axis,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    body = c.block()
    if operation == "max":
        body = initialize_output(
            output_array_tile_config,
            c_variables=dict(input_var=input_var, output_var=output_var),
        )

    body += generate_body(
        input_array_tile_config,
        output_array_tile_config,
        operation=operation,
        axes=axes,
        c_variables=dict(input_var=input_var, output_var=output_var),
        offsets=dict(input=c.literal(0), output=c.literal(0)),
        **kwargs,
    )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".c"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.Text(max_function),
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


def initialize_output(output_array_tile_config, c_variables):
    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(output_array_tile_config.shape)

    input_var = c_variables["input_var"]
    output_var = c_variables["output_var"]

    loop = c.block(
        c.ForLoop(
            c.Declare(index, c.literal(0)),
            index < c.literal(num_iterations),
            c.add_in_place(index, c.literal(1)),
            c.block(
                c.assign(
                    c_variables["output_var"][index],
                    c.literal("-INFINITY"),
                )
            ),
        )
    )

    body = c.If(c.not_equals(input_var, output_var), loop)

    return c.block(body)


def compute_offset(offset, indices, num_tiles_per_axis, next_level_volume):
    for axis, index in enumerate(indices):
        offset = offset + index * c.literal(math.prod(num_tiles_per_axis[axis + 1 :]))
    offset = offset * c.literal(next_level_volume)
    return offset


def generate_body(input_array_tile_config, output_array_tile_config, operation, axes, c_variables, offsets, **kwargs):
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
            operation=operation,
            axes=axes,
            c_variables=c_variables,
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

        input_var = c_variables["input_var"][input_index]
        output_var = c_variables["output_var"][output_index]
        if operation == "sum":
            reduction_step = operator.add(output_var, input_var)
        elif operation == "mean":
            reduction_step = operator.add(output_var, input_var / c.literal(kwargs["num_reduced_elements"]))
        elif operation == "max":
            reduction_step = c.invoke("_maxf", output_var, input_var)
        else:
            raise NotImplementedError

        inner_loop_body = c.block(declare_index, declare_output_index, c.assign(output_var, reduction_step))

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
