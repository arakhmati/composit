import math
import operator
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernels.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


def generate_kernel(path, input_a_array_tile_config, input_b_array_tile_config: ArrayTileConfig, operation):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_a_array_tile_config,
        input_b_array_tile_config,
        operation,
    )

    input_a_var = c.variable(InputType, "input_a_var")
    input_b_var = c.variable(InputType, "input_b_var")
    output_var = c.variable(OutputType, "output_var")

    body = generate_body(
        input_a_array_tile_config,
        input_b_array_tile_config,
        c_variables=dict(input_a_var=input_a_var, input_b_var=input_b_var, output_var=output_var),
        operation=operation,
    )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".c"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.NewLine(),
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(kernel_name),
                arguments=[input_a_var, input_b_var, output_var],
                body=body,
            ),
        ],
    )
    file.save()
    return kernel_name


def generate_body(
    input_a_array_tile_config,
    input_b_array_tile_config,
    c_variables,
    operation,
):
    operation_to_python_operator = {
        "add": operator.add,
        "subtract": operator.sub,
        "multiply": operator.mul,
        "divide": operator.truediv,
    }
    python_operator = operation_to_python_operator[operation]

    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(input_a_array_tile_config.shape)

    b_loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(
            c.assign(
                c_variables["output_var"][index],
                python_operator(c_variables["input_a_var"][index], c_variables["input_b_var"][index]),
            )
        ),
    )

    return c.block(b_loop)
