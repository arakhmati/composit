#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto batch_size = {{batch_size}};
constexpr auto m_size = {{m_size}};
constexpr auto k_size = {{k_size}};
constexpr auto n_size = {{n_size}};

extern "C" void run(const data_type_t* input_buffer, const data_type_t* weights_buffer, data_type_t* output_buffer) {
  using namespace sonic::lazy_computation;

  const auto input = as_lazy_computation<data_type_t, sonic::shape::shape_t<batch_size, m_size, k_size>>(
      std::assume_aligned<sizeof(data_type_t)>(input_buffer));
  const auto weights = as_lazy_computation<data_type_t, sonic::shape::shape_t<k_size, n_size>>(
      std::assume_aligned<sizeof(data_type_t)>(weights_buffer));
  matmul(input, weights, std::assume_aligned<sizeof(data_type_t)>(output_buffer))();
}