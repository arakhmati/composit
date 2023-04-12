import codegen as c


def test_declare_variable():
    index = c.variable(c.AUTO, "index") << c.literal(0)
    expected = "auto index = 0;"
    assert str(index) == expected


def test_function_that_gets_array_element_at_index():

    array = c.variable(c.Type("float").pointer(), "array")
    index = c.variable(c.Type("int"), "index")
    result = c.variable(c.AUTO, "result")

    function = c.Function(
        return_type=c.Type("float"),
        name=c.Identifier("get"),
        arguments=[array, index],
        body=c.Block(
            [
                c.Statement(c.Declare(result, array[index])),
                c.Return(c.Statement(result.value)),
            ]
        ),
    )

    expected = """\
float get(float* array, int index)
{
    auto result = array[index];
    return result;
}"""

    assert str(function) == expected


def test_function_that_reduces_array_using_sum():

    array = c.variable(c.Type("float").pointer().const().restrict(), "array")
    index = c.variable(c.Type("int"), "index")
    result = c.variable(c.AUTO, "result")

    function = c.Function(
        return_type=c.Type("float"),
        name=c.Identifier("get"),
        arguments=[array, index],
        body=c.Block(
            [
                c.Statement(c.Declare(result, c.literal(0))),
                c.ForLoop(
                    c.Declare(index, c.literal(0)),
                    index < c.literal(10),
                    c.add_in_place(index, c.literal(1)),
                    c.Block(
                        [
                            c.Statement(c.add_in_place(result, array[index])),
                        ]
                    ),
                ),
                c.Return(c.Statement(result.value)),
            ]
        ),
    )

    expected = """\
float get(const float* __restrict__ array, int index)
{
    auto result = 0;
    for (int index = 0; (index < 10); index++)
    {
        result += array[index];
    }
    return result;
}"""

    assert str(function) == expected


def _mm256_reduce_add_ps():
    Vector256Type = c.Type("__m256")
    Vector128Type = c.Type("__m128")
    x = c.variable(Vector256Type, "x")
    x128 = c.variable(Vector128Type.const(), "x128")
    x64 = c.variable(Vector128Type.const(), "x64")
    x32 = c.variable(Vector128Type.const(), "x32")

    return (
        c.Function(
            return_type=c.Type("float"),
            name=c.Identifier("_mm256_reduce_add_ps"),
            arguments=[x],
            body=c.Block(
                [
                    c.Statement(
                        c.Declare(
                            x128,
                            c.invoke(
                                c.Identifier("_mm_add_ps"),
                                c.invoke(c.Identifier("_mm256_extractf128_ps"), x, c.literal(1)),
                                c.invoke(c.Identifier("_mm256_castps256_ps128"), x),
                            ),
                        )
                    ),
                    c.Statement(
                        c.Declare(
                            x64,
                            c.invoke(
                                c.Identifier("_mm_add_ps"),
                                x128,
                                c.invoke(c.Identifier("_mm_movehl_ps"), x128, x128),
                            ),
                        )
                    ),
                    c.Statement(
                        c.Declare(
                            x32,
                            c.invoke(
                                c.Identifier("_mm_add_ss"),
                                x64,
                                c.invoke(c.Identifier("_mm_shuffle_ps"), x64, x64, c.literal("0x55")),
                            ),
                        )
                    ),
                    c.Return(c.Statement(c.invoke(c.Identifier("_mm_cvtss_f32"), x32))),
                ]
            ),
        )
        .inline()
        .static()
    )


def matmul_kernel(avx_size):

    Vector256Type = c.Type("__m256")
    InputType = c.Type("float").const().pointer().restrict().aligned("ALIGNMENT")
    OutputType = c.Type("float").pointer().restrict().aligned("ALIGNMENT")
    input_a = c.variable(InputType, "input_a")
    input_b = c.variable(InputType, "input_b")
    output = c.variable(OutputType, "output")

    input_a_vector = c.variable(Vector256Type, "input_a_vector")
    input_b_vector = c.variable(Vector256Type, "input_b_vector")
    output_vector = c.variable(Vector256Type, "output_vector")

    b = c.variable(c.AUTO, "b")
    m = c.variable(c.AUTO, "m")
    n = c.variable(c.AUTO, "n")
    k = c.variable(c.AUTO, "k")

    kernel_body = c.Block(
        [
            c.Statement(
                c.Declare(
                    input_a_vector,
                    c.invoke(
                        c.Identifier("_mm256_load_ps"),
                        input_a + m * c.literal(64) + k,
                    ),
                )
            ),
            c.Statement(
                c.Declare(
                    input_b_vector,
                    c.invoke(
                        c.Identifier("_mm256_load_ps"),
                        input_b + n * c.literal(64) + k,
                    ),
                )
            ),
            c.Statement(
                c.Assign(
                    output_vector,
                    c.invoke(
                        c.Identifier("_mm256_fmadd_ps"),
                        input_a_vector,
                        input_b_vector,
                        output_vector,
                    ),
                )
            ),
        ]
    )
    k_for_loop = c.ForLoop(
        c.Declare(k, c.literal(0)),
        k < c.literal(64),
        c.add_in_place(k, avx_size),
        kernel_body,
    )

    n_for_loop = c.ForLoop(
        c.Declare(n, c.literal(0)),
        n < c.literal(64),
        c.add_in_place(n, c.literal(1)),
        c.Block(
            [
                c.Statement(
                    c.Declare(
                        output_vector,
                        c.invoke(c.Identifier("_mm256_setzero_ps")),
                    )
                ),
                k_for_loop,
                c.Statement(
                    c.add_in_place(
                        output[m * c.literal(64) + n],
                        c.invoke(c.Identifier("_mm256_reduce_add_ps"), output_vector),
                    )
                ),
            ]
        ),
    )

    m_for_loop = c.ForLoop(
        c.Declare(m, c.literal(0)),
        m < c.literal(64),
        c.add_in_place(m, c.literal(1)),
        c.Block([n_for_loop]),
    )
    b_for_loop = c.ForLoop(
        c.Declare(b, c.literal(0)),
        b < c.literal(1),
        c.add_in_place(b, c.literal(1)),
        c.Block([m_for_loop]),
    )

    return c.Function(
        return_type=c.Type("void"),
        name=c.Identifier("MatmulKernel"),
        arguments=[input_a, input_b, output],
        body=c.Block(
            [
                b_for_loop,
            ]
        ),
    )


def test_matmul_kernel_file():
    avx_size = c.variable(c.AUTO.constexpr(), "AVX_SIZE")
    file = c.File(
        "matmul.hpp",
        [
            c.Include("immintrin.h"),
            c.NewLine(),
            avx_size << c.literal(8),
            c.NewLine(),
            _mm256_reduce_add_ps(),
            c.NewLine(),
            matmul_kernel(avx_size),
        ],
    )

    
    # ruff: noqa: E501
    expected = """\
#include <immintrin.h>

constexpr auto AVX_SIZE = 8;

static inline float _mm256_reduce_add_ps(__m256 x)
{
    const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
    const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
    const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
    return _mm_cvtss_f32(x32);
}

void MatmulKernel(const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_a, const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_b, float* __restrict__ __attribute__((aligned(ALIGNMENT))) output)
{
    for (auto b = 0; (b < 1); b++)
    {
        for (auto m = 0; (m < 64); m++)
        {
            for (auto n = 0; (n < 64); n++)
            {
                __m256 output_vector = _mm256_setzero_ps();
                for (auto k = 0; (k < 64); k += AVX_SIZE)
                {
                    __m256 input_a_vector = _mm256_load_ps(((input_a + (m * 64)) + k));
                    __m256 input_b_vector = _mm256_load_ps(((input_b + (n * 64)) + k));
                    output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
                }
                output[((m * 64) + n)] += _mm256_reduce_add_ps(output_vector);
            }
        }
    }
}
"""

    assert str(file) == expected
