import math
import pathlib

import codegen as c
from mosaic.tilelab.layout import TransposedLayout

from mosaic.tilelab.tile import TileConfig
from mosaic.backends.x86.avx import _mm256_load_ps, _mm256_fmadd_ps
from mosaic.backends.x86.constants import AVX_SIZE, MEMORY_ALIGNMENT
from mosaic.backends.x86.kernel_name import create_kernel_name

OffsetType = c.Variable | c.Expression

InputType = c.Type("float").const().pointer().restrict().aligned(MEMORY_ALIGNMENT)
OutputType = c.Type("float").pointer().restrict().aligned(MEMORY_ALIGNMENT)
Vector256Type = c.Type("__m256")


mm256_reduce_add_ps = c.Lambda(
    """ \
auto _mm256_reduce_add_ps = [](const auto& x) {
    /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    /* Conversion to float is a no-op on x86-64 */
    return _mm_cvtss_f32(x32);
};
"""
)


def generate_module(
    input_tile_configs,
    output_tile_config,
    input_dtypes,
    output_dtype,
    *,
    use_avx_manually: bool,
    enable_tracy: bool = False,
):
    kernel_name = create_kernel_name(
        pathlib.Path(__file__).stem,
        input_tile_configs[0],
        input_tile_configs[1],
        use_avx_manually,
    )

    input_a_var = c.variable(InputType, "input_a_var")
    input_b_var = c.variable(InputType, "input_b_var")
    output_var = c.variable(OutputType, "output_var")

    body = initialize_output(output_tile_config, output_var)
    body += generate_body(
        arguments=[input_a_var, input_b_var, output_var],
        input_a_tile_config=input_tile_configs[0],
        input_b_tile_config=input_tile_configs[1],
        offsets=dict(input_a_var=c.literal(0), input_b_var=c.literal(0), output_var=c.literal(0)),
        use_avx_manually=use_avx_manually,
        enable_tracy=enable_tracy,
    )

    includes = [
        c.Include("immintrin.h"),
        c.Include("stdint.h"),
    ]
    if enable_tracy:
        includes.append(c.Include("tracy/Tracy.hpp"))

    module = c.Module(
        includes=includes,
        members=[
            c.Function(
                return_type=c.Type("void"),
                name=c.Identifier(kernel_name),
                arguments=[input_a_var, input_b_var, output_var],
                body=body,
            ).extern_c()
        ],
    )
    return kernel_name, module


def initialize_output(output_tile_config, output_var):
    index = c.variable(c.Type("uint32_t"), "index")
    num_iterations = math.prod(output_tile_config.shape)

    value = c.literal(0)
    loop = c.ForLoop(
        c.Declare(index, c.literal(0)),
        index < c.literal(num_iterations),
        c.add_in_place(index, c.literal(1)),
        c.block(c.assign(output_var[index], value)),
    )

    return c.block(loop)


def input_b_is_transposed(tile_config):
    if not isinstance(tile_config.layout, TransposedLayout):
        return False

    order = tuple(tile_config.layout.order)
    contiguous_order = tuple(range(len(order)))
    if order[:-2] == contiguous_order[:-2] and order[-2:] != contiguous_order[-2:]:
        return True
    return False


def generate_body(
    arguments,
    *,
    input_a_tile_config,
    input_b_tile_config,
    offsets,
    use_avx_manually: bool,
    enable_tracy: bool = False,
):
    input_a_var, input_b_var, output_var = arguments

    level_name = input_a_tile_config.level_name

    inner_loop_increment = c.literal(1)
    outer_loop_body_after = c.block()

    b = c.variable(c.Type("uint32_t"), f"{level_name}_b")
    m = c.variable(c.Type("uint32_t"), f"{level_name}_m")
    n = c.variable(c.Type("uint32_t"), f"{level_name}_n")
    k = c.variable(c.Type("uint32_t"), f"{level_name}_k")

    a_ranges = tuple(num_tiles for num_tiles in input_a_tile_config.num_tiles_per_axis())
    *_, n_range = (num_tiles for num_tiles in input_b_tile_config.num_tiles_per_axis())

    *b_sizes, m_size, k_size, n_size = a_ranges + (n_range,)
    b_size = math.prod(b_sizes)

    if isinstance(input_a_tile_config, TileConfig):
        output_tile_volume = math.prod([*input_a_tile_config.tile_shape[:-1], input_b_tile_config.tile_shape[-1]])

        next_a_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_a_offset")
        next_b_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_b_offset")
        next_output_offset = c.variable(c.Type("uint32_t"), f"{level_name}_next_output_offset")

        declare_next_a_offset = next_a_offset << (
            offsets["input_a_var"]
            + (b * c.literal(m_size) * c.literal(k_size) * c.literal(math.prod(input_a_tile_config.tile_shape)))
            + ((m * c.literal(k_size) + k) * c.literal(math.prod(input_a_tile_config.tile_shape)))
        )
        declare_next_output_offset = next_output_offset << (
            offsets["output_var"]
            + (b * c.literal(m_size) * c.literal(n_size) * c.literal(output_tile_volume))
            + ((m * c.literal(n_size) + n) * c.literal(output_tile_volume))
        )

        if input_b_is_transposed(input_b_tile_config):
            declare_next_b_offset = next_b_offset << (
                offsets["input_b_var"]
                + (b * c.literal(n_size) * c.literal(k_size) * c.literal(math.prod(input_b_tile_config.tile_shape)))
                + ((n * c.literal(k_size) + k) * c.literal(math.prod(input_b_tile_config.tile_shape)))
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
                + (b * c.literal(k_size) * c.literal(n_size) * c.literal(math.prod(input_b_tile_config.tile_shape)))
                + ((k * c.literal(n_size) + n) * c.literal(math.prod(input_b_tile_config.tile_shape)))
            )
            inner_loop_index = n
            inner_loop_size = c.literal(n_size)
            outer_loop_index = k
            outer_loop_size = c.literal(k_size)

            inner_loop_body = c.block(declare_next_b_offset, declare_next_output_offset)
            outer_loop_body_before = c.block(declare_next_a_offset)

        inner_loop_body += generate_body(
            arguments=arguments,
            input_a_tile_config=input_a_tile_config.next_level(),
            input_b_tile_config=input_b_tile_config.next_level(),
            offsets=dict(input_a_var=next_a_offset, input_b_var=next_b_offset, output_var=next_output_offset),
            use_avx_manually=use_avx_manually,
        )

    else:
        if input_b_is_transposed(input_b_tile_config):
            inner_loop_index = k
            inner_loop_size = c.literal(k_size)
            outer_loop_index = n
            outer_loop_size = c.literal(n_size)

            if use_avx_manually:
                inner_loop_increment = c.literal(AVX_SIZE)

                input_a_vector = c.variable(Vector256Type, "input_a_vector")
                input_b_vector = c.variable(Vector256Type, "input_b_vector")
                output_vector = c.variable(Vector256Type, "output_vector")

                outer_loop_body_before = c.block(
                    output_vector << c.invoke(c.Identifier("_mm256_setzero_ps")),
                )

                inner_loop_body = c.block(
                    input_a_vector
                    << _mm256_load_ps(
                        input_a_var
                        + offsets["input_a_var"]
                        + b * c.literal(m_size) * c.literal(k_size)
                        + m * c.literal(k_size)
                        + k
                    ),
                    input_b_vector
                    << _mm256_load_ps(
                        input_b_var
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

                outer_loop_body_after = c.block(
                    mm256_reduce_add_ps,
                    c.Statement(
                        c.add_in_place(
                            output_var[
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
                            output_var[output_index],
                            input_a_var[a_index] * input_b_var[b_index],
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
                        output_var[output_index],
                        input_a_var[a_index] * input_b_var[b_index],
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

    m_loop_body = [outer_loop]
    if level_name == "l1_cache" and enable_tracy:
        mark_m_loop_zone = c.Statement(c.Text('ZoneScopedNS("l1_cache_m_loop", 4)'))
        m_loop_body.insert(0, mark_m_loop_zone)

    m_loop = c.ForLoop(
        c.Declare(m, c.literal(0)),
        m < c.literal(m_size),
        c.add_in_place(m, c.literal(1)),
        c.block(*m_loop_body),
    )

    b_loop_body = [m_loop]
    if level_name == "l1_cache" and enable_tracy:
        mark_b_loop_zone = c.Statement(c.Text('ZoneScopedNS("l1_cache_b_loop", 4)'))
        b_loop_body.insert(0, mark_b_loop_zone)

    b_loop = c.ForLoop(
        c.Declare(b, c.literal(0)),
        b < c.literal(b_size),
        c.add_in_place(b, c.literal(1)),
        c.block(*b_loop_body),
    )

    main_block = [b_loop]

    if enable_tracy:
        mark_frame = c.Statement(c.Text("FrameMark"))
        main_block.append(mark_frame)

    return c.block(*main_block)
