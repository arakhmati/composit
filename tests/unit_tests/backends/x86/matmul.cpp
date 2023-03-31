#include <cassert>
#include <cmath>
#include <cstring>
#include <stdlib.h>

#include <sched.h>

constexpr auto ALIGNMENT = 32;

#include "matmul.hpp"

constexpr auto EPSILON = 1e-5f;

bool AreSame(double a, double b) { return std::abs(a - b) < EPSILON; }

auto CompareArrays(const float *array_a, const float *array_b,
                   const std::size_t size) {
  for (std::size_t index = 0; index < size; index++) {
    auto value_a = array_a[index];
    auto value_b = array_b[index];
    if (not AreSame(value_a, value_b)) {
      return false;
    }
  }
  return true;
}

extern "C" {
void run(float *input_0, std::size_t input_0_size, float *input_1,
         std::size_t input_1_size, float *output, std::size_t output_size) {

  constexpr auto core = 0;
  constexpr auto pid = 0;
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(core, &cpu_set);
  sched_setaffinity(pid, sizeof(cpu_set_t), &cpu_set);

  auto memory_pool = static_cast<float *>(std::aligned_alloc(
      ALIGNMENT, (input_0_size + input_1_size + output_size) * sizeof(float)));

  float *pool_input_0 __attribute__((aligned(ALIGNMENT))) = memory_pool;
  float *pool_input_1 __attribute__((aligned(ALIGNMENT))) =
      memory_pool + (input_0_size * 1);
  float *pool_output __attribute__((aligned(ALIGNMENT))) =
      memory_pool + (input_0_size + input_1_size);

  memcpy(pool_input_0, input_0, input_0_size * sizeof(float));
  memcpy(pool_input_1, input_1, input_1_size * sizeof(float));
  memset(pool_output, 0.0f, output_size * sizeof(float));

  MatmulKernel(pool_input_0, pool_input_1, pool_output);

  memcpy(output, pool_output, output_size * sizeof(float));

  std::free(memory_pool);
}
}