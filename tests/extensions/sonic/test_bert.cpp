#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/tensor.hpp"
#include "sonic/vector.hpp"

template<
    auto QueryWeights,
    auto QueryBias,
    auto KeyWeights,
    auto KeyBias,
    auto ValueWeights,
    auto ValueBias
>
struct EncoderParameters {
    static constexpr auto query_weights = QueryWeights;
    static constexpr auto query_bias = QueryBias;
    static constexpr auto key_weights = KeyWeights;
    static constexpr auto key_bias = KeyBias;
    static constexpr auto value_weights = ValueWeights;
    static constexpr auto value_bias = ValueBias;
};

template<
    auto Encoder0
>
struct Parameters {
    static constexpr auto encoder_0 = Encoder0;
};

template<typename DataType, auto HiddenSize>
constexpr auto create_parameters() {
    using namespace sonic::tensor;
    constexpr auto query_weights = arange<DataType, Shape<HiddenSize, HiddenSize>>();
    constexpr auto query_bias = arange<DataType, Shape<HiddenSize>>();
    constexpr auto key_weights = arange<DataType, Shape<HiddenSize, HiddenSize>>();
    constexpr auto key_bias = arange<DataType, Shape<HiddenSize>>();
    constexpr auto value_weights = arange<DataType, Shape<HiddenSize, HiddenSize>>();
    constexpr auto value_bias = arange<DataType, Shape<HiddenSize>>();
    return Parameters<
        EncoderParameters<
            query_weights,
            query_bias,
            key_weights,
            key_bias,
            value_weights,
            value_bias
        >{}
    >{};
}

template<auto BatchSize, auto SequenceSize, auto NumHeads, auto HeadSize>
constexpr auto encoder(const auto& hidden_states, const auto& parameters) {
    using namespace sonic::tensor;
    auto query_matmul = matmul(hidden_states, parameters.query_weights);
    auto query_matmul_plus_bias = query_matmul + parameters.query_bias;
    auto reshaped_query_matmul_plus_bias = reshape<Shape<BatchSize, SequenceSize, NumHeads, HeadSize>>(query_matmul_plus_bias);
    auto query = transpose<Order<0, 2, 1, 3>>(reshaped_query_matmul_plus_bias);

    auto key_matmul = matmul(hidden_states, parameters.key_weights);
    auto key_matmul_plus_bias = key_matmul + parameters.key_bias;

    auto value_matmul = matmul(hidden_states, parameters.value_weights);
    auto value_matmul_plus_bias = value_matmul + parameters.value_bias;

    return sqrt(abs(query_matmul_plus_bias));
}

TEST_CASE("test bert encoder") {
    using namespace sonic::tensor;

    using data_type_t = float;
    constexpr auto batch_size = 1;
    constexpr auto sequence_size = 4;
    constexpr auto num_heads = 2;
    constexpr auto head_size = 4;
    constexpr auto hidden_size = num_heads * head_size;
    using shape_t = Shape<batch_size, sequence_size, hidden_size>;

    constexpr auto input = arange<data_type_t, shape_t>();
    constexpr auto parameters = create_parameters<data_type_t, hidden_size>();
    constexpr auto output = encoder<batch_size, sequence_size, num_heads, head_size>(input, parameters.encoder_0);

    using input_shape = decltype(input)::shape_t;
    static_assert(std::is_same_v<input_shape, Shape<batch_size, sequence_size, hidden_size>>);
    static_assert(input_shape::volume == sequence_size * hidden_size);

    using output_shape = decltype(output)::shape_t;
    static_assert(std::is_same_v<output_shape, Shape<batch_size, sequence_size, hidden_size>>);
    static_assert(output_shape::volume == sequence_size * hidden_size);

    const auto golden_output_data = std::array<data_type_t, shape_t::volume>{33.4664,33.8969,34.322,34.7419,35.1568,35.5668,35.9722,36.3731,53.963,54.8179,55.6597,56.4889,57.3062,58.112,58.9067,59.6909,68.5857,69.7209,70.8378,71.9375,73.0205,74.0878,75.1399,76.1774,80.5978,81.9573,83.2947,84.6109,85.9069,87.1837,88.4421,89.6828};
    constexpr auto golden_output = as_tensor<data_type_t, shape_t>(golden_output_data);

    CHECK(allclose(output, golden_output));
}