#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto num_encoders = {{num_encoders}};
constexpr auto batch_size = {{batch_size}};
constexpr auto sequence_size = {{sequence_size}};
constexpr auto num_attention_heads = {{num_attention_heads}};
constexpr auto head_size = {{head_size}};
constexpr auto hidden_size = num_attention_heads * head_size;
using shape_t = sonic::shape::shape_t<batch_size, sequence_size, hidden_size>;

template <typename query_weights_t,
          typename query_bias_t,
          typename key_weights_t,
          typename key_bias_t,
          typename value_weights_t,
          typename value_bias_t,
          typename self_output_weights_t,
          typename self_output_bias_t>
struct multi_head_attention_parameters_t {
  const query_weights_t query_weights;
  const query_bias_t query_bias;
  const key_weights_t key_weights;
  const key_bias_t key_bias;
  const value_weights_t value_weights;
  const value_bias_t value_bias;
  const self_output_weights_t self_output_weights;
  const self_output_bias_t self_output_bias;

  explicit multi_head_attention_parameters_t(query_weights_t&& query_weights,
                                             query_bias_t&& query_bias,
                                             key_weights_t&& key_weights,
                                             key_bias_t&& key_bias,
                                             value_weights_t&& value_weights,
                                             value_bias_t&& value_bias,
                                             self_output_weights_t&& self_output_weights,
                                             self_output_bias_t&& self_output_bias)
      : query_weights(query_weights),
        query_bias(query_bias),
        key_weights(key_weights),
        key_bias(key_bias),
        value_weights(value_weights),
        value_bias(value_bias),
        self_output_weights(self_output_weights),
        self_output_bias(self_output_bias) {}
};

template <typename ff1_weights_t, typename ff1_bias_t, typename ff2_weights_t, typename ff2_bias_t>
struct feedforward_parameters_t {
  const ff1_weights_t ff1_weights;
  const ff1_bias_t ff1_bias;
  const ff2_weights_t ff2_weights;
  const ff2_bias_t ff2_bias;

  explicit feedforward_parameters_t(ff1_weights_t&& ff1_weights,
                                    ff1_bias_t&& ff1_bias,
                                    ff2_weights_t&& ff2_weights,
                                    ff2_bias_t&& ff2_bias)
      : ff1_weights(ff1_weights), ff1_bias(ff1_bias), ff2_weights(ff2_weights), ff2_bias(ff2_bias) {}
};

template <typename multi_head_attention_parameters_t, typename feedforward_parameters_t>
struct encoder_parameters_t {
  const multi_head_attention_parameters_t multi_head_attention;
  const feedforward_parameters_t feedforward;

  explicit encoder_parameters_t(multi_head_attention_parameters_t&& multi_head_attention,
                                feedforward_parameters_t&& feedforward)
      : multi_head_attention(multi_head_attention), feedforward(feedforward) {}
};

template <typename encoder_parameters_t>
struct parameters_t {
  const std::array<encoder_parameters_t, num_encoders> encoders;

  explicit parameters_t(std::array<encoder_parameters_t, num_encoders>&& encoders) : encoders(encoders) {}
};

template <auto hidden_size>
auto create_encoder_parameters(const data_type_t** parameter_buffers) {
  using namespace sonic::lazy_computation;

  constexpr auto intermediate_size = hidden_size * 4;

  auto query_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[0]));
  auto query_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[1]));
  auto key_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[2]));
  auto key_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[3]));
  auto value_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[4]));
  auto value_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[5]));
  auto self_output_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[6]));
  auto self_output_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[7]));
  auto ff1_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, intermediate_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[8]));
  auto ff1_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<intermediate_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[9]));
  auto ff2_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<intermediate_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[10]));
  auto ff2_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[11]));
  return encoder_parameters_t{
      multi_head_attention_parameters_t{std::move(query_weights), std::move(query_bias), std::move(key_weights),
                                        std::move(key_bias), std::move(value_weights), std::move(value_bias),
                                        std::move(self_output_weights), std::move(self_output_bias)},
      feedforward_parameters_t{
          std::move(ff1_weights),
          std::move(ff1_bias),
          std::move(ff2_weights),
          std::move(ff2_bias),
      }};
}

template <auto hidden_size>
auto create_parameters(const data_type_t** parameter_buffers) {
  using namespace sonic::lazy_computation;
  return parameters_t{std::array{
      create_encoder_parameters<hidden_size>(parameter_buffers),
      //    create_encoder_parameters<hidden_size>(parameter_buffers + 12),
  }};
}

template <auto head_size,
          template <typename...>
          typename hidden_states_t,
          typename data_type_t,
          auto batch_size,
          auto sequence_size,
          auto hidden_size,
          typename stride_t,
          typename... rest_t>
auto multi_head_attention(const hidden_states_t<data_type_t,
                                                const sonic::shape::shape_t<batch_size, sequence_size, hidden_size>,
                                                const stride_t,
                                                rest_t...>& hidden_states,
                          const auto& parameters) {
  using namespace sonic::lazy_computation;

  constexpr auto num_attention_heads = hidden_size / head_size;

  auto query_matmul_plus_bias = linear(hidden_states, parameters.query_weights, parameters.query_bias);
  auto reshaped_query_matmul_plus_bias =
      reshape<sonic::shape::shape_t<batch_size, sequence_size, num_attention_heads, head_size>>(query_matmul_plus_bias);
  auto query = transpose<order_t<0, 2, 1, 3>>(reshaped_query_matmul_plus_bias);

  auto key_matmul_plus_bias = linear(hidden_states, parameters.key_weights, parameters.key_bias);
  auto reshaped_key_matmul_plus_bias =
      reshape<sonic::shape::shape_t<batch_size, sequence_size, num_attention_heads, head_size>>(key_matmul_plus_bias);
  auto key = transpose<order_t<0, 2, 3, 1>>(reshaped_key_matmul_plus_bias);

  auto value_matmul_plus_bias = linear(hidden_states, parameters.value_weights, parameters.value_bias);
  auto reshaped_value_matmul_plus_bias =
      reshape<sonic::shape::shape_t<batch_size, sequence_size, num_attention_heads, head_size>>(value_matmul_plus_bias);
  auto value = transpose<order_t<0, 2, 1, 3>>(reshaped_value_matmul_plus_bias);

  auto query_times_key = matmul(query, key);
  auto normalized_query_times_key = divide_in_place(query_times_key, static_cast<data_type_t>(std::sqrt(head_size)));
  auto attention_scores = softmax<>(normalized_query_times_key);

  auto transposed_and_reshaped_context = matmul(attention_scores, value);
  auto reshaped_context = transpose<order_t<0, 2, 1, 3>>(transposed_and_reshaped_context);
  auto context =
      reshape<sonic::shape::shape_t<batch_size, sequence_size, num_attention_heads * head_size>>(reshaped_context);

  return linear(context, parameters.self_output_weights, parameters.self_output_bias);
}

auto feedforward(const auto& hidden_states, const auto& parameters) {
  using namespace sonic::lazy_computation;
  auto ff1 = linear(hidden_states, parameters.ff1_weights, parameters.ff1_bias);
  return linear(ff1, parameters.ff2_weights, parameters.ff2_bias);
}

template <std::size_t head_size>
auto encoder(const auto& hidden_states, const auto& parameters) {
  auto mha_output = multi_head_attention<head_size>(hidden_states, parameters.multi_head_attention);
  auto ff_output = feedforward(mha_output, parameters.feedforward);
  return ff_output;
}

template <std::size_t head_size, std::size_t index>
auto encoder_loop(const auto& encoder_input, const auto& parameters) {
  if constexpr (index == num_encoders) {
    return encoder_input;
  } else {
    auto encoder_output = encoder<head_size>(encoder_input, parameters.encoders[index]);
    return encoder_loop<head_size, index + 1>(encoder_output, parameters);
  }
}

extern "C" void run(const data_type_t* input_buffer,
                    data_type_t* output_buffer,
                    const data_type_t** parameter_buffers) {
  using namespace sonic::lazy_computation;

  auto input = as_lazy_computation<data_type_t, shape_t>(std::assume_aligned<sizeof(data_type_t)>(input_buffer));
  auto parameters = create_parameters<hidden_size>(parameter_buffers);
  auto output = encoder_loop<head_size, 0>(input, parameters);

  evaluate_to(output, std::assume_aligned<sizeof(data_type_t)>(output_buffer));
}