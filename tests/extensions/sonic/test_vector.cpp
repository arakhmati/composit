#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/vector.hpp"

TEST_CASE("test vector add") {
    using namespace sonic::vector;

    using data_type_t = float;
    constexpr auto size = 4;

    auto vector_a = Vector<data_type_t, size>{{0, 1, 2, 3}};
    auto vector_b = Vector<data_type_t, size>{{4, 5, 6, 7}};
    auto sum = vector_a + vector_b;
    auto vector_sum = Vector<data_type_t, size>(sum);

    CHECK(vector_sum == Vector<data_type_t, size>{{4, 6, 8, 10}});
}