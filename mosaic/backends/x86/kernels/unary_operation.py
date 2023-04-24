import math
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)

operation_to_c_function = {
    "exp": "expf",
    "sqrt": "sqrtf",
    "gelu": "geluf",
}

gelu_function = """     
static inline float cdf(float input) {
    return 0.5 * (1 + erff(input / sqrtf(2)));
}
static inline float geluf(float input) {
    return input * cdf(input);
}
"""


def generate_kernel(path, input_array_tile_config: ArrayTileConfig, operation):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_array_tile_config,
        operation,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    body = generate_body(
        input_array_tile_config,
        operation=operation,
        arguments=[input_var, output_var],
    )

    file = c.File(
        (path / pathlib.Path(kernel_name)).with_suffix(".c"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.Text(gelu_function),
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


def generate_body(
    input_array_tile_config,
    operation,
    arguments,
):
    input_var, output_var = arguments

    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(input_array_tile_config.shape)

    loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(c.assign(output_var[index], c.invoke(operation_to_c_function[operation], input_var[index]))),
    )

    return c.block(loop)
