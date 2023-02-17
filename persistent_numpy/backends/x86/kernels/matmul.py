import itertools
import math

import numpy as np
from loguru import logger

import persistent_numpy as pnp
import persistent_numpy.nn
from persistent_numpy.tilelab import TilizedTensor


def generate_kernel(path, input_a, input_b, *, unroll_levels=False):
    with open(path / "matmul.hpp", "w") as f:
        f.write(
            "void MatmulKernel(const float* __restrict__ input_a, const float* __restrict__ input_b, float* output) {\n"
        )
        generate_indices(f, input_a, input_b, unroll_levels=unroll_levels)
        f.write("}\n")


def generate_indices(f, input_a, input_b, a_offset="0", b_offset="0", output_offset="0", *, unroll_levels):
    level_name = input_a.level_name
    if isinstance(input_a, TilizedTensor):
        if level_name in unroll_levels:

            a_num_tiles_per_axis = input_a.num_tiles_per_axis()
            b_num_tiles_per_axis = input_b.num_tiles_per_axis()
            a_ranges = tuple(range(num_tiles) for num_tiles in a_num_tiles_per_axis)
            _, n_range = (range(num_tiles) for num_tiles in b_num_tiles_per_axis)

            for *a_indices, n in itertools.product(*a_ranges, n_range):
                a_indices = tuple(a_indices)
                *_, m, k = a_indices
                b_indices = (k, n)
                a_tile = input_a[a_indices]
                b_tile = input_b[b_indices]

                next_a_offset = (
                    f"{a_offset} + {(m * input_a.num_tiles_per_axis()[-1] + k) * math.prod(input_a.tile_shape)}"
                )
                next_b_offset = (
                    f"{b_offset} + {(k * input_b.num_tiles_per_axis()[-1] + n) * math.prod(input_b.tile_shape)}"
                )
                next_output_offset = f"{output_offset} + {(m * input_b.num_tiles_per_axis()[-1] + n) * math.prod([*input_a.tile_shape[:-1], input_b.tile_shape[-1]])}"

                generate_indices(
                    f,
                    a_tile,
                    b_tile,
                    a_offset=next_a_offset,
                    b_offset=next_b_offset,
                    output_offset=next_output_offset,
                    unroll_levels=unroll_levels,
                )
        else:

            a_num_tiles_per_axis = input_a.num_tiles_per_axis()
            b_num_tiles_per_axis = input_b.num_tiles_per_axis()
            a_ranges = tuple(num_tiles for num_tiles in a_num_tiles_per_axis)
            _, n_range = tuple(num_tiles for num_tiles in b_num_tiles_per_axis)

            b_size, m_size, k_size, n_size = a_ranges + (n_range,)

            b = f"{level_name}_b"
            m = f"{level_name}_m"
            k = f"{level_name}_k"
            n = f"{level_name}_n"

            f.write(f"for (auto {b} = 0; {b} < {b_size}; {b}++) {{\n")
            f.write(f"for (auto {m} = 0; {m} < {m_size}; {m}++) {{\n")
            f.write(f"for (auto {k} = 0; {k} < {k_size}; {k}++) {{\n")
            f.write(f"for (auto {n} = 0; {n} < {n_size}; {n}++) {{\n")

            next_a_offset = f"{level_name}_a_offset"
            next_b_offset = f"{level_name}_b_offset"
            next_output_offset = f"{level_name}_output_offset"

            f.write(f"auto {next_a_offset} = {a_offset} + ({m} * {k_size} + {k}) * {math.prod(input_a.tile_shape)};\n")
            f.write(f"auto {next_b_offset} = {b_offset} + ({k} * {n_size} + {n}) * {math.prod(input_b.tile_shape)};\n")
            f.write(
                f"auto {next_output_offset} = {output_offset} + ({m} * {n_size} + {n}) * {math.prod([*input_a.tile_shape[:-1], input_b.tile_shape[-1]])};\n"
            )

            a_tile = input_a[(0, 0, 0)]
            b_tile = input_b[(0, 0)]

            generate_indices(
                f,
                a_tile,
                b_tile,
                a_offset=next_a_offset,
                b_offset=next_b_offset,
                output_offset=next_output_offset,
                unroll_levels=unroll_levels,
            )

            f.write("}\n")
            f.write("}\n")
            f.write("}\n")
            f.write("}\n")
    else:
        if level_name in unroll_levels:
            a_ranges = tuple(range(num_tiles) for num_tiles in input_a.shape)
            _, n_range = (range(num_tiles) for num_tiles in input_b.shape)

            for *a_indices, n in itertools.product(*a_ranges, n_range):
                *batch_indices, m, k = a_indices
                assert all(batch_index == 0 for batch_index in batch_indices)

                a_index = f"{a_offset} + {m * input_a.shape[-1] + k}"
                b_index = f"{b_offset} + {k * input_b.shape[-1] + n}"
                output_index = f"{output_offset} + {m * input_b.shape[-1] + n}"

                string = f"output[{output_index}] += input_a[{a_index}] * input_b[{b_index}];\n"
                f.write(string)
        else:
            a_ranges = tuple(num_tiles for num_tiles in input_a.shape)
            _, n_range = (num_tiles for num_tiles in input_b.shape)

            b_size, m_size, k_size, n_size = a_ranges + (n_range,)

            b = f"_b"
            m = f"_m"
            k = f"_k"
            n = f"_n"

            f.write(f"for (auto {b} = 0; {b} < {b_size}; {b}++) {{\n")
            f.write(f"for (auto {m} = 0; {m} < {m_size}; {m}++) {{\n")
            f.write(f"for (auto {k} = 0; {k} < {k_size}; {k}++) {{\n")
            f.write("#pragma GCC ivdep\n")
            f.write(f"for (auto {n} = 0; {n} < {n_size}; {n}++) {{\n")

            a_index = f"_a_offset"
            b_index = f"_b_offset"
            output_index = f"_output_offset"

            f.write(f"auto {a_index} = {a_offset} + ({m} * {k_size} + {k});\n")
            f.write(f"auto {b_index} = {b_offset} + ({k} * {n_size} + {n});\n")
            f.write(f"auto {output_index} = {output_offset} + ({m} * {n_size} + {n});\n")

            f.write(f"output[{output_index}] += input_a[{a_index}] * input_b[{b_index}];\n")

            f.write("}\n")
            f.write("}\n")
            f.write("}\n")
            f.write("}\n")


def generate_data(path, input_a, input_b, output, evaluate_inputs):
    with open(path / "matmul_data.hpp", "w") as f:
        f.write(f"#include<array>\n\n")

        input_var_0_tiles = list(input_a.tiles())
        input_var_1_tiles = list(input_b.tiles())
        output_var_tiles = list(output.tiles())

        tiles = pnp.nn.evaluate(*input_var_0_tiles, *input_var_1_tiles, *output_var_tiles, inputs=evaluate_inputs)

        offset = 0
        flat_data = [
            float(x) for index in range(len(input_var_0_tiles)) for x in tiles[offset + index].flatten().tolist()
        ]
        lhs = f"std::array<float, {len(flat_data)}> input_0"
        rhs = ",\n".join(f"{x}" for x in flat_data)
        f.write(f"{lhs} = {{ \n{rhs} }};\n")

        offset += len(input_var_0_tiles)
        flat_data = [
            float(x) for index in range(len(input_var_1_tiles)) for x in tiles[offset + index].flatten().tolist()
        ]
        lhs = f"std::array<float, {len(flat_data)}> input_1"
        rhs = ",\n".join(f"{x}" for x in flat_data)
        f.write(f"{lhs} = {{ \n{rhs} }};\n")

        offset += len(input_var_1_tiles)
        flat_data = [
            float(x) for index in range(len(output_var_tiles)) for x in tiles[offset + index].flatten().tolist()
        ]
        lhs = f"std::array<float, {len(flat_data)}> golden_matmul_output"
        rhs = ",\n".join(f"{x}" for x in flat_data)
        f.write(f"{lhs} = {{ \n{rhs} }};\n")
