#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/shape.hpp"
#include "sonic/tensor.hpp"

TEST_CASE("test tensor add") {
  using namespace sonic::tensor;

  using data_type_t = float;
  using shape_t = sonic::shape::shape_t<4>;

  auto tensor_a = tensor_t<data_type_t, shape_t>{{0, 1, 2, 3}};
  auto tensor_b = tensor_t<data_type_t, shape_t>{{4, 5, 6, 7}};
  auto sum = add(tensor_a, tensor_b);
  auto tensor_sum = as_tensor(sum);

  CHECK(tensor_sum == tensor_t<data_type_t, shape_t>{{4, 6, 8, 10}});
}