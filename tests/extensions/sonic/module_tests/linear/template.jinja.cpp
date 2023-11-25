#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto batch_size = {{batch_size}};
constexpr auto m_size = {{m_size}};
constexpr auto k_size = {{k_size}};
constexpr auto n_size = {{n_size}};

extern "C" void run(const data_type_t* input_buffer,
                    data_type_t* output_buffer,
                    const data_type_t** parameter_buffers) {
  using namespace sonic::lazy_computation;
  using namespace sonic::tensor;

  auto input = as_lazy_computation<data_type_t, sonic::shape::shape_t<batch_size, m_size, k_size>>(
      std::assume_aligned<sizeof(data_type_t)>(input_buffer));
  auto query_weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<k_size, n_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[0]));
  auto query_bias = as_lazy_computation<data_type_t, sonic::shape::shape_t<n_size>>(
      std::assume_aligned<sizeof(data_type_t)>(parameter_buffers[1]));
  auto output = linear(input, query_weights, query_bias);

  evaluate_to<vector8_float32>(output, std::assume_aligned<sizeof(data_type_t)>(output_buffer));
}