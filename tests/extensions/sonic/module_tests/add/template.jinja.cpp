#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

#include <memory>

using data_type_t = float;

constexpr auto size = {{size}};

extern "C" void run(const data_type_t* input_buffer_a, const data_type_t* input_buffer_b, data_type_t* output_buffer) {
  using namespace sonic::lazy_computation;
  using namespace sonic::tensor;

  const auto input_a =
      as_lazy_computation<data_type_t, sonic::shape::shape_t<size>>(std::assume_aligned<sizeof(32)>(input_buffer_a));
  const auto input_b =
      as_lazy_computation<data_type_t, sonic::shape::shape_t<size>>(std::assume_aligned<sizeof(32)>(input_buffer_b));
  const auto output = input_a + input_b;
  evaluate_to<vector8_float32>(output, std::assume_aligned<sizeof(32)>(output_buffer));
}