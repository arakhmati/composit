import itertools
import math

import composit as cnp
import composit.nn
from composit.tilelab import TilizedTensor


def generate_kernel(path, input_a, input_b, *, transpose_b_levels, use_avx_manually: bool):
    if transpose_b_levels is None:
        transpose_b_levels = set()

    with open(path / "matmul.hpp", "w") as f:
        f.write(
            """  
#include <immintrin.h>

constexpr auto AVX_SIZE = 8;

static inline float _mm256_reduce_add_ps(__m256 x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
}

void MatmulKernel(const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_a, const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_b, float* __restrict__ __attribute__((aligned(ALIGNMENT))) output) {\n
"""
        )
        generate_indices(f, input_a, input_b, transpose_b_levels=transpose_b_levels, use_avx_manually=use_avx_manually)
        f.write("}\n")


def generate_indices(
    f, input_a, input_b, a_offset="0", b_offset="0", output_offset="0", *, transpose_b_levels, use_avx_manually: bool
):
    level_name = input_a.level_name
    if isinstance(input_a, TilizedTensor):
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

        if level_name in transpose_b_levels:
            f.write(f"for (auto {n} = 0; {n} < {n_size}; {n}++) {{\n")

            next_output_offset = f"{level_name}_output_offset"
            f.write(
                f"auto {next_output_offset} = {output_offset} + ({m} * {n_size} + {n}) * {math.prod([*input_a.tile_shape[:-1], input_b.tile_shape[-1]])};\n"
            )

            f.write(f"for (auto {k} = 0; {k} < {k_size}; {k}++) {{\n")

            next_a_offset = f"{level_name}_a_offset"
            next_b_offset = f"{level_name}_b_offset"

            f.write(f"auto {next_a_offset} = {a_offset} + ({m} * {k_size} + {k}) * {math.prod(input_a.tile_shape)};\n")
            f.write(f"auto {next_b_offset} = {b_offset} + ({n} * {k_size} + {k}) * {math.prod(input_b.tile_shape)};\n")

            a_tile = input_a[(0, 0, 0)]
            b_tile = input_b[(0, 0)]

            generate_indices(
                f,
                a_tile,
                b_tile,
                a_offset=next_a_offset,
                b_offset=next_b_offset,
                output_offset=next_output_offset,
                transpose_b_levels=transpose_b_levels,
                use_avx_manually=use_avx_manually,
            )

            f.write("}\n")

        else:
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
                transpose_b_levels=transpose_b_levels,
                use_avx_manually=use_avx_manually,
            )

            f.write("}\n")
        f.write("}\n")
        f.write("}\n")
        f.write("}\n")
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

        if level_name in transpose_b_levels:
            f.write(f"for (auto {n} = 0; {n} < {n_size}; {n}++) {{\n")

            if use_avx_manually:
                f.write(f"__m256 output_vector = _mm256_setzero_ps();\n")

                f.write(f"for (auto {k} = 0; {k} < {k_size}; {k} += AVX_SIZE) {{\n")

                f.write(f"__m256 input_a_vector = _mm256_load_ps(input_a + {a_offset} + {m} * {k_size} + {k});\n")
                f.write(f"__m256 input_b_vector = _mm256_load_ps(input_b + {b_offset} + {n} * {k_size} + {k});\n")
                f.write(f"output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);\n")

                f.write("}\n")

                f.write(f"output[{output_offset} + {m} * {n_size} + {n}] += _mm256_reduce_add_ps(output_vector);\n")
            else:
                f.write(f"for (auto {k} = 0; {k} < {n_size}; {k}++) {{\n")

                output_index = f"_output_offset"
                f.write(f"auto {output_index} = {output_offset} + ({m} * {n_size} + {n});\n")
                a_index = f"_a_offset"
                b_index = f"_b_offset"

                f.write(f"auto {a_index} = {a_offset} + ({m} * {k_size} + {k});\n")
                f.write(f"auto {b_index} = {b_offset} + ({n} * {k_size} + {k});\n")

                f.write(f"output[{output_index}] += input_a[{a_index}] * input_b[{b_index}];\n")

                f.write("}\n")

        else:

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


def transpose_tiles(tensor, transpose_levels, order):
    if isinstance(tensor, TilizedTensor):
        ranges = tuple(range(num_tiles) for num_tiles in tensor.num_tiles_per_axis())
        if tensor.level_name in transpose_levels:
            ranges = [ranges[axis] for axis in order]
        for tile_indices in itertools.product(*ranges):
            if tensor.level_name in transpose_levels:
                tile_indices = tuple([tile_indices[axis] for axis in order])
            yield from transpose_tiles(tensor[tile_indices], transpose_levels, order)
    else:
        if tensor.level_name in transpose_levels:
            yield cnp.transpose(tensor.tile, order)
        else:
            yield tensor.tile


def generate_data(path, input_a, input_b, output, evaluate_inputs, *, transpose_b_levels=None):
    if transpose_b_levels is None:
        transpose_b_levels = set()

    with open(path / "matmul_data.hpp", "w") as f:
        f.write(f"#include<array>\n\n")

        input_var_0_tiles = list(input_a.tiles())
        input_var_1_tiles = list(transpose_tiles(input_b, transpose_b_levels, (1, 0)))
        output_var_tiles = list(output.tiles())

        tiles = cnp.nn.evaluate(*input_var_0_tiles, *input_var_1_tiles, *output_var_tiles, inputs=evaluate_inputs)

        offset = 0
        flat_data = [
            float(x) for index in range(len(input_var_0_tiles)) for x in tiles[offset + index].flatten().tolist()
        ]
        lhs = f"std::array<float, {len(flat_data)}> input_0"
        rhs = ",\n".join(f"{x}" for x in flat_data)
        f.write(f"{lhs} = {{ \n{rhs} }};\n")

        offset += len(input_var_0_tiles)
        flat_data = [
            float(x) for index in range(len(input_var_1_tiles)) for x in (tiles[offset + index]).flatten().tolist()
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
