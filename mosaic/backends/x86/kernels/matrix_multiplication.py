import math
import pathlib
from typing import Union

import codegen as c

from mosaic.tilelab.tile import ArrayTileConfig
from mosaic.backends.x86.avx import _mm256_load_ps, _mm256_fmadd_ps
from mosaic.backends.x86.constants import AVX_SIZE, MEMORY_ALIGNMENT

OffsetType = Union[c.Variable, c.Expression]

InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)
Vector256Type = c.Type("__m256")


mm256_reduce_add_ps = c.Text(
    """ \
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
"""
)


def generate_kernel(
    path, input_a_array_tile_config, input_b_array_tile_config, *, transpose_b_levels, use_avx_manually: bool
):
    if transpose_b_levels is None:
        transpose_b_levels = set()

    input_a_var = c.variable(InputType, "input_a_var")
    input_b_var = c.variable(InputType, "input_b_var")
    output_var = c.variable(OutputType, "output_var")

    body = generate_body(
        input_a_array_tile_config,
        input_b_array_tile_config,
        c_variables=dict(input_a_var=input_a_var, input_b_var=input_b_var, output_var=output_var),
        offsets=dict(input_a_var=c.literal(0), input_b_var=c.literal(0), output_var=c.literal(0)),
        transpose_b_levels=transpose_b_levels,
        use_avx_manually=use_avx_manually,
    )

    file = c.File(
        (path / pathlib.Path(__file__).stem).with_suffix(".c"),
        [
            c.Include("immintrin.h"),
            c.Include("stdint.h"),
            c.NewLine(),
            c.NewLine(),
            mm256_reduce_add_ps,
            c.NewLine(),
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier("run"),
                arguments=[input_a_var, input_b_var, output_var],
                body=body,
            ),
        ],
    )
    file.save()


def generate_body(
    input_a_array_tile_config,
    input_b_array_tile_config,
    c_variables,
    offsets,
    *,
    transpose_b_levels,
    use_avx_manually: bool,
):
    level_name = input_a_array_tile_config.level_name

    inner_loop_increment = c.literal(1)
    outer_loop_body_after = c.block()

    if isinstance(input_a_array_tile_config, ArrayTileConfig):
        a_num_tiles_per_axis = input_a_array_tile_config.num_tiles_per_axis()
        b_num_tiles_per_axis = input_b_array_tile_config.num_tiles_per_axis()
        a_ranges = tuple(num_tiles for num_tiles in a_num_tiles_per_axis)
        *_, n_range = tuple(num_tiles for num_tiles in b_num_tiles_per_axis)

        *b_sizes, m_size, k_size, n_size = a_ranges + (n_range,)
        b_size = math.prod(b_sizes)

        a_tile = input_a_array_tile_config[tuple(0 for _ in range(len(a_num_tiles_per_axis)))]
        b_tile = input_b_array_tile_config[tuple(0 for _ in range(len(b_num_tiles_per_axis)))]

        b = c.variable(c.Type("uint32_t"), f"{level_name}_b")
        m = c.variable(c.Type("uint32_t"), f"{level_name}_m")
        n = c.variable(c.Type("uint32_t"), f"{level_name}_n")
        k = c.variable(c.Type("uint32_t"), f"{level_name}_k")

        output_tile_volume = math.prod(
            [*input_a_array_tile_config.tile_shape[:-1], input_b_array_tile_config.tile_shape[-1]]
        )

        next_a_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_a_offset")
        next_b_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_b_offset")
        next_output_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_output_offset")

        declare_next_a_offset = next_a_offset << (
            offsets["input_a_var"]
            + ((m * c.literal(k_size) + k) * c.literal(math.prod(input_a_array_tile_config.tile_shape)))
        )
        declare_next_output_offset = next_output_offset << (
            offsets["output_var"] + ((m * c.literal(n_size) + n) * c.literal(output_tile_volume))
        )

        if level_name in transpose_b_levels:
            declare_next_b_offset = next_b_offset << (
                offsets["input_b_var"]
                + ((n * c.literal(k_size) + k) * c.literal(math.prod(input_b_array_tile_config.tile_shape)))
            )
            inner_loop_index = k
            inner_loop_size = c.literal(k_size)
            outer_loop_index = n
            outer_loop_size = c.literal(n_size)

            inner_loop_body = c.block(declare_next_a_offset, declare_next_b_offset)
            outer_loop_body_before = c.block(declare_next_output_offset)
        else:
            declare_next_b_offset = next_b_offset << (
                offsets["input_b_var"]
                + ((k * c.literal(n_size) + n) * c.literal(math.prod(input_b_array_tile_config.tile_shape)))
            )
            inner_loop_index = n
            inner_loop_size = c.literal(n_size)
            outer_loop_index = k
            outer_loop_size = c.literal(k_size)

            inner_loop_body = c.block(declare_next_b_offset, declare_next_output_offset)
            outer_loop_body_before = c.block(declare_next_a_offset)

        inner_loop_body += generate_body(
            a_tile,
            b_tile,
            c_variables=c_variables,
            offsets=dict(input_a_var=next_a_offset, input_b_var=next_b_offset, output_var=next_output_offset),
            transpose_b_levels=transpose_b_levels,
            use_avx_manually=use_avx_manually,
        )

    else:
        a_ranges = tuple(num_tiles for num_tiles in input_a_array_tile_config.shape)
        *_, n_range = (num_tiles for num_tiles in input_b_array_tile_config.shape)

        *b_sizes, m_size, k_size, n_size = a_ranges + (n_range,)
        b_size = math.prod(b_sizes)

        b = c.variable(c.Type("uint32_t"), "b")
        m = c.variable(c.Type("uint32_t"), "m")
        n = c.variable(c.Type("uint32_t"), "n")
        k = c.variable(c.Type("uint32_t"), "k")

        if level_name in transpose_b_levels:
            inner_loop_index = k
            inner_loop_size = c.literal(k_size)
            outer_loop_index = n
            outer_loop_size = c.literal(n_size)

            if use_avx_manually:
                inner_loop_increment = c.literal(AVX_SIZE)

                input_a_vector = c.variable(Vector256Type, "input_a_vector")
                input_b_vector = c.variable(Vector256Type, "input_b_vector")
                output_vector = c.variable(Vector256Type, "output_vector")

                inner_loop_body = c.block(
                    input_a_vector
                    << _mm256_load_ps(
                        c_variables["input_a_var"]
                        + offsets["input_a_var"]
                        + b * c.literal(m_size) * c.literal(k_size)
                        + m * c.literal(k_size)
                        + k
                    ),
                    input_b_vector
                    << _mm256_load_ps(
                        c_variables["input_b_var"]
                        + offsets["input_b_var"]
                        + b * c.literal(n_size) * c.literal(k_size)
                        + n * c.literal(k_size)
                        + k
                    ),
                    c.assign(
                        output_vector,
                        _mm256_fmadd_ps(
                            input_a_vector,
                            input_b_vector,
                            output_vector,
                        ),
                    ),
                )

                outer_loop_body_before = c.block(
                    output_vector << c.invoke(c.Identifier("_mm256_setzero_ps")),
                )

                outer_loop_body_after = c.block(
                    c.Statement(
                        c.add_in_place(
                            c_variables["output_var"][
                                offsets["output_var"]
                                + b * c.literal(m_size) * c.literal(n_size)
                                + m * c.literal(n_size)
                                + n
                            ],
                            c.invoke(c.Identifier("_mm256_reduce_add_ps"), output_vector),
                        )
                    ),
                )

            else:
                a_index = c.variable(c.Type("uint32_t"), "a_index")
                b_index = c.variable(c.Type("uint32_t"), "b_index")
                output_index = c.variable(c.Type("uint32_t"), "output_index")

                declare_a_index = a_index << (
                    offsets["input_a_var"] + (b * c.literal(m_size) * c.literal(k_size)) + (m * c.literal(k_size) + k)
                )
                declare_b_index = b_index << (
                    offsets["input_b_var"] + (b * c.literal(n_size) * c.literal(k_size)) + (n * c.literal(k_size) + k)
                )
                declare_output_index = output_index << (
                    offsets["output_var"] + (b * c.literal(m_size) * c.literal(n_size)) + (m * c.literal(n_size) + n)
                )
                inner_loop_body = c.block(
                    declare_a_index,
                    declare_b_index,
                    c.Statement(
                        c.add_in_place(
                            c_variables["output_var"][output_index],
                            c_variables["input_a_var"][a_index] * c_variables["input_b_var"][b_index],
                        )
                    ),
                )
                outer_loop_body_before = c.block(declare_output_index)

        else:
            inner_loop_index = n
            inner_loop_size = c.literal(n_size)
            outer_loop_index = k
            outer_loop_size = c.literal(k_size)

            a_index = c.variable(c.Type("uint32_t"), "a_index")
            b_index = c.variable(c.Type("uint32_t"), "b_index")
            output_index = c.variable(c.Type("uint32_t"), "output_index")

            declare_a_index = a_index << (
                offsets["input_a_var"] + (b * c.literal(m_size) * c.literal(k_size)) + (m * c.literal(k_size) + k)
            )
            declare_b_index = b_index << (
                offsets["input_b_var"] + (b * c.literal(k_size) * c.literal(n_size)) + (k * c.literal(n_size) + n)
            )
            declare_output_index = output_index << (
                offsets["output_var"] + (b * c.literal(m_size) * c.literal(n_size)) + (m * c.literal(n_size) + n)
            )

            inner_loop_body = c.block(
                declare_b_index,
                declare_output_index,
                c.Statement(
                    c.add_in_place(
                        c_variables["output_var"][output_index],
                        c_variables["input_a_var"][a_index] * c_variables["input_b_var"][b_index],
                    )
                ),
            )
            outer_loop_body_before = c.block(declare_a_index)

    inner_loop = c.ForLoop(
        c.Declare(inner_loop_index, c.literal(0)),
        inner_loop_index < c.literal(inner_loop_size),
        c.add_in_place(inner_loop_index, inner_loop_increment),
        inner_loop_body,
    )

    outer_loop = c.ForLoop(
        c.Declare(outer_loop_index, c.literal(0)),
        outer_loop_index < c.literal(outer_loop_size),
        c.add_in_place(outer_loop_index, c.literal(1)),
        outer_loop_body_before + c.block(inner_loop) + outer_loop_body_after,
    )

    m_loop = c.ForLoop(
        c.Declare(m, c.literal(0)),
        m < c.literal(m_size),
        c.add_in_place(m, c.literal(1)),
        c.block(outer_loop),
    )

    b_loop = c.ForLoop(
        c.Declare(b, c.literal(0)),
        b < c.literal(b_size),
        c.add_in_place(b, c.literal(1)),
        c.block(m_loop),
    )

    return c.block(b_loop)
