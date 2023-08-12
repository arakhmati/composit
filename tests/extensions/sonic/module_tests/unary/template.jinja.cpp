#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto batch_size = {{batch_size}};
constexpr auto height_size = {{height_size}};
constexpr auto width_size = {{width_size}};

extern "C" void run(const data_type_t* input_buffer, data_type_t* output_buffer) {
  using namespace sonic::lazy_computation;
  using sonic::tensor::vector8_float32;

  const auto input = as_lazy_computation<data_type_t, sonic::shape::shape_t<batch_size, height_size, width_size>>(
      std::assume_aligned<sizeof(data_type_t)>(input_buffer));

  const auto abs_output = abs(input);
  const auto exp_output = exp(abs_output);
  const auto sqrt_output = sqrt(exp_output);

  evaluate_to<vector8_float32>(sqrt_output, output_buffer);
}