#define DOCTEST_CONFIG_IMPLEMENT_WITH_MAIN
#include "doctest.hpp"

#include "sonic/shape.hpp"
#include "sonic/thread_pool.hpp"

#include <array>

TEST_CASE("test empty queue") {
  using namespace sonic::thread_pool;

  constexpr auto num_threads = 12;
  constexpr auto num_elements = 1000;

  struct element_t {
    int value;
    int thread_index;
  };
  std::array<element_t, num_elements> data;
  std::fill(std::begin(data), std::end(data), element_t{0, -1});

  {
    auto thread_pool = thread_pool_t<num_threads>{};
    for (auto i = 0; i < num_elements; i++)
      thread_pool.push_back_computation([&data, i](std::size_t thread_index) {
        data[i].value += i;
        data[i].thread_index = thread_index;
      });
  }

  for (auto i = 0; i < num_elements; i++) {
    CHECK(data[i].value == i);
    CHECK(data[i].thread_index != -1);
  }
}
