#include <cassert>
#include <cmath>
#include <cstring>
#include <stdlib.h>

#include <sched.h>

constexpr auto ALIGNMENT = 32;

#include "matmul.hpp"

extern "C" {
void run(const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_a, const float* __restrict__ __attribute__((aligned(ALIGNMENT))) input_b, float* __restrict__ __attribute__((aligned(ALIGNMENT))) output) {

  constexpr auto core = 0;
  constexpr auto pid = 0;
  cpu_set_t cpu_set;
  CPU_ZERO(&cpu_set);
  CPU_SET(core, &cpu_set);
  sched_setaffinity(pid, sizeof(cpu_set_t), &cpu_set);

  MatmulKernel(input_a, input_b, output);
}
}