import math
import pathlib

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.constants import MEMORY_ALIGNMENT


InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)


def generate_kernel(path, input_array_tile_config: ArrayTileConfig):
    input_var = c.variable(InputType, "input")
    output_var = c.variable(OutputType, "output")

    body = generate_body(
        input_array_tile_config,
        c_variables=dict(input_var=input_var, output_var=output_var),
    )

    file = c.File(
        (path / pathlib.Path(__file__).stem).with_suffix(".c"),
        [
            c.Include("math.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            c.NewLine(),
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier("run"),
                arguments=[input_var, output_var],
                body=body,
            ),
        ],
    )
    file.save()


def generate_body(
    input_array_tile_config,
    c_variables,
):
    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(input_array_tile_config.shape)

    b_loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(c.assign(c_variables["output_var"][index], c.invoke("expf", c_variables["input_var"][index]))),
    )

    return c.block(b_loop)
