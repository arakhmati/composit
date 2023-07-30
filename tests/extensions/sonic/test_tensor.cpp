#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/tensor.hpp"
#include "sonic/shape.hpp"

TEST_CASE("test tensor add") {
    using namespace sonic::tensor;

    using data_type_t = float;
    constexpr auto size = 4;

    auto tensor_a = Tensor<data_type_t, size>{{0, 1, 2, 3}};
    auto tensor_b = Tensor<data_type_t, size>{{4, 5, 6, 7}};
    auto sum = tensor_a + tensor_b;
    auto tensor_sum = Tensor<data_type_t, size>(sum);

    CHECK(tensor_sum == Tensor<data_type_t, size>{{4, 6, 8, 10}});
}