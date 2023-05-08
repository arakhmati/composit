import math
import pathlib

import codegen as c

from mosaic.backends.x86.constants import MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)

operation_to_c_function = {
    "exp": "expf",
    "sqrt": "sqrtf",
    "gelu": "geluf",
}


def generate_module(input_array_tile_configs, output_array_tile_config, input_dtypes, output_dtype, operation):
    input_array_tile_config, *_ = input_array_tile_configs
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_array_tile_config,
        operation,
    )

    input_var = c.variable(InputType, "input_var")
    output_var = c.variable(OutputType, "output_var")

    module = c.Module(
        includes=[c.Include("math.h"), c.Include("stdint.h")],
        functions=[
            c.void_function(
                name=c.Identifier(kernel_name),
                arguments=[input_var, output_var],
                body_function=generate_body,
                input_array_tile_config=input_array_tile_config,
                operation=operation,
            ).extern_c()
        ],
    )
    return kernel_name, module


def generate_body(
    arguments,
    input_array_tile_config,
    operation,
):
    input_var, output_var = arguments

    lambdas = []
    if operation == "gelu":
        lambdas = [
            c.Lambda("auto cdf = [](float input) { return 0.5 * (1 + erff(input / sqrtf(2))); };"),
            c.Lambda("auto geluf = [&cdf](float input) { return input * cdf(input); };"),
        ]

    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(input_array_tile_config.shape)

    loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(c.assign(output_var[index], c.invoke(operation_to_c_function[operation], input_var[index]))),
    )

    return c.block(*lambdas, loop)
