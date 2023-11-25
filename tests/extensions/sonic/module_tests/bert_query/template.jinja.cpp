#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto batch_size = {{batch_size}};
constexpr auto sequence_size = {{sequence_size}};
constexpr auto num_attention_heads = {{num_attention_heads}};
constexpr auto head_size = {{head_size}};
constexpr auto hidden_size = num_attention_heads * head_size;

template <auto head_size,
          template <typename...>
          typename hidden_states_t,
          typename data_type_t,
          auto batch_size,
          auto sequence_size,
          auto hidden_size,
          typename stride_t,
          typename... rest_t>
auto query(const hidden_states_t<data_type_t,
                                 const sonic::shape::shape_t<batch_size, sequence_size, hidden_size>,
                                 const stride_t,
                                 rest_t...>& hidden_states,
           const auto& query_weights,
           const auto& query_bias) {
  using namespace sonic::lazy_computation;

  constexpr auto num_attention_heads = hidden_size / head_size;

  auto query_matmul_plus_bias = linear(hidden_states, query_weights, query_bias);
  auto reshaped_query_matmul_plus_bias =
      reshape<sonic::shape::shape_t<batch_size, sequence_size, num_attention_heads, head_size>>(query_matmul_plus_bias);
  auto query = transpose<order_t<0, 2, 1, 3>>(reshaped_query_matmul_plus_bias);
  return query;
}

extern "C" void run(const data_type_t* input_buffer,
                    data_type_t* output_buffer,
                    const data_type_t** parameter_buffers) {
  using namespace sonic::lazy_computation;

  auto input = as_lazy_computation<data_type_t, sonic::shape::shape_t<batch_size, sequence_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(input_buffer));
  auto query_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size, hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[0]));
  auto query_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<hidden_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[1]));
  auto output = query<head_size>(input, query_weights, query_bias);

  evaluate_to(output, std::assume_aligned<sizeof(data_type_t)>(output_buffer));
}