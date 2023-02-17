#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <iostream>

#include "matmul.hpp"
#include "matmul_data.hpp"

constexpr auto EPSILON = 1e-5f;

bool AreSame(double a, double b)
{
    return std::abs(a - b) < EPSILON;
}

template<typename T, std::size_t N>
auto CompareArrays(std::array<T, N> array_a, std::array<T, N> array_b) {
    for (std::size_t index = 0; index < array_a.size(); index++) {
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
    for (auto i = 0; i < count; i++) {
        std::array<float, golden_matmul_output.size()> matmul_output{};

        auto start = std::chrono::steady_clock::now();
        MatmulKernel(input_0.data(), input_1.data(), matmul_output.data());
        auto end = std::chrono::steady_clock::now();

        auto nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
        std::cout << "execution time: " << nanoseconds << " nanoseconds\n";

        assert(CompareArrays(golden_matmul_output, matmul_output));
    }
    return 0;
}