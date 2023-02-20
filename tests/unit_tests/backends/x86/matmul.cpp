#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>
#include <stdlib.h>

constexpr auto ALIGNMENT = 32;

#include "matmul.hpp"
#include "matmul_data.hpp"

constexpr auto EPSILON = 1e-5f;

bool AreSame(double a, double b)
{
    return std::abs(a - b) < EPSILON;
}

auto CompareArrays(const float* array_a, const float* array_b, const std::size_t size) {
    for (std::size_t index = 0; index < size; index++) {
        auto value_a = array_a[index];
        auto value_b = array_b[index];
        if (not AreSame(value_a, value_b)) {
            return false;
        }
    }
    return true;
}

int main(int argc, char **argv) {
    auto count = std::stoi(std::string(argv[1]));

    auto memory_pool = static_cast<float*>(std::aligned_alloc(ALIGNMENT, (input_0.size() + input_1.size() +  golden_matmul_output.size() * 2) * sizeof(float)));

    float* pool_input_0 __attribute__((aligned(ALIGNMENT))) = memory_pool;
    float* pool_input_1 __attribute__((aligned(ALIGNMENT))) = memory_pool + (input_0.size() * 1);
    float* pool_output __attribute__((aligned(ALIGNMENT))) = memory_pool + (input_0.size() + input_1.size());
    float* pool_golden_matmul_output __attribute__((aligned(ALIGNMENT))) = memory_pool + (input_0.size() + input_1.size() + golden_matmul_output.size());

    for (auto i = 0lu; i < input_0.size(); i++) {
        pool_input_0[i] = input_0[i];
    }

    for (auto i = 0lu; i < input_1.size(); i++) {
        pool_input_1[i] = input_1[i];
    }

    for (auto i = 0lu; i < golden_matmul_output.size(); i++) {
        pool_golden_matmul_output[i] = golden_matmul_output[i];
    }

    for (auto i = 0; i < count; i++) {

        for (auto i = 0lu; i < golden_matmul_output.size(); i++) {
            pool_output[i] = 0;
        }

        auto start = std::chrono::steady_clock::now();
        MatmulKernel(pool_input_0, pool_input_1, pool_output);
        auto end = std::chrono::steady_clock::now();

        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "execution time: " << nanoseconds << " nanoseconds\n";

        assert(CompareArrays(pool_golden_matmul_output, pool_output, golden_matmul_output.size()));
    }

    std::free(memory_pool);

    return 0;
}