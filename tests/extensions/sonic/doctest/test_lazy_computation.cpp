#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/lazy_computation.hpp"
#include "sonic/shape.hpp"

TEST_CASE("test lazy tensor indexing") {
  using namespace sonic::lazy_computation;

  using data_type_t = float;
  using shape_t = sonic::shape::shape_t<3, 4, 8>;

  constexpr auto input = arange<data_type_t, shape_t>();

  CHECK(input()[std::make_tuple(0, 0, 0)] == 0.0f);
  CHECK(input()[std::make_tuple(0, 0, 1)] == 1.0f);
  CHECK(input()[std::make_tuple(0, 1, 0)] == 8.0f);
  CHECK(input()[std::make_tuple(2, 3, 1)] == 89.0f);
}

TEST_CASE("test lazy exp") {
  using namespace sonic::lazy_computation;

  using data_type_t = float;
  using shape_t = sonic::shape::shape_t<1, 4, 8>;

  constexpr auto input = random<data_type_t, shape_t>();
  constexpr auto output = exp(input);
  auto output_tensor = evaluate(output);

  const auto golden_output_data = sonic::tensor::aligned_array<data_type_t, shape_t::volume>{
      1.20405,  1.99079,  2.04601, 2.00271,  1.28034, 0.793552, 0.667023, 0.412065, 0.634646, 0.956313, 1.86701,
      0.960746, 0.807001, 1.95846, 0.722377, 1.34493, 0.768345, 2.49505,  0.487094, 2.0963,   0.948585, 1.82544,
      1.04181,  1.43012,  1.55467, 1.17826,  1.07761, 1.67738,  0.454669, 0.94857,  0.534013, 1.60614};
  auto golden_output_tensor = decltype(output_tensor)(std::move(golden_output_data));

  CHECK(sonic::tensor::allclose(output_tensor, golden_output_tensor));
}

TEST_CASE("test lazy transpose") {
  using namespace sonic::lazy_computation;

  using data_type_t = float;
  using input_shape_t = sonic::shape::shape_t<1, 3, 2, 4>;

  constexpr auto input = arange<data_type_t, input_shape_t>();
  constexpr auto output = transpose<order_t<0, 2, 1, 3>>(input);
  auto output_tensor = evaluate(output);

  using input_stride_t = decltype(input)::stride_t;
  static_assert(std::is_same_v<input_stride_t, const sonic::stride::stride_t<24, 8, 4, 1>>);

  using output_shape_t = decltype(output)::shape_t;
  using output_stride_t = decltype(output)::stride_t;
  static_assert(std::is_same_v<output_shape_t, const sonic::shape::shape_t<1, 2, 3, 4>>);
  static_assert(std::is_same_v<output_stride_t, const sonic::stride::stride_t<24, 4, 8, 1>>);
  static_assert(output_shape_t::volume == input_shape_t::volume);

  const auto golden_output_data = sonic::tensor::aligned_array<data_type_t, output_shape_t::volume>{
      0, 1, 2, 3, 8, 9, 10, 11, 16, 17, 18, 19, 4, 5, 6, 7, 12, 13, 14, 15, 20, 21, 22, 23};
  auto golden_output_tensor = decltype(output_tensor)(std::move(golden_output_data));

  CHECK(sonic::tensor::allclose(output_tensor, golden_output_tensor));
}