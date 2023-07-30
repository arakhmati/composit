#pragma once

#include "vector.hpp"

#include <array>
#include <functional>
#include <random>

namespace sonic {

namespace tensor {

template<auto ... Dims>
struct Shape {
    static constexpr auto volume = (1 * ... * Dims);
};

template<auto ... Axes>
struct Order {};

template<typename DataType, typename Shape, typename Function>
struct Tensor {
    using data_type_t = DataType;
    using shape_t = Shape;
    const Function function;

    constexpr Tensor(const Function& function) : function(function) {}
    constexpr auto operator()() const {;
        return this->function();
    }
};

template<typename DataType, typename Shape>
constexpr auto as_tensor(auto&& data) {
    const auto function = [data = std::move(data)] {
        return data;
    };
    return Tensor<DataType, Shape, decltype(function)>{function};
}

template<typename DataType, typename Shape>
constexpr auto arange() {
    constexpr auto function = [] {
        auto output = std::array<DataType, Shape::volume>{};
        auto index = 0.0f;
        for (auto& value : output) {
            value = index++;
        }
        return vector::Vector{std::move(output)};
    };
    return Tensor<DataType, Shape, decltype(function)>{function};
}


template<typename DataType, typename Shape>
constexpr auto random() {
    constexpr auto function = [] {
        std::mt19937 generator(0);
        std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
        auto output = std::array<DataType, Shape::volume>{};
        for (auto& value : output) {
            value = distribution(generator);
        }
        return vector::Vector{std::move(output)};
    };
    return Tensor<DataType, Shape, decltype(function)>{function};
}

template<typename OutputShape, typename DataType, typename InputShape, typename Function>
constexpr auto reshape(
    const Tensor<DataType, InputShape, Function>& input_tensor
) {
    static_assert(InputShape::volume == OutputShape::volume);
    return Tensor<DataType, OutputShape, Function>{input_tensor.function};
}

template<typename Order, typename DataType, auto ... Dims, typename Function>
constexpr auto transpose(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
//    static_assert(InputShape::volume == OutputShape::volume);
    return Tensor<DataType, Shape<Dims...>, Function>{input_tensor.function};
}

template<typename DataType, auto M, auto K, auto N, typename FunctionA, typename FunctionB>
constexpr auto matmul(
    const Tensor<DataType, Shape<1, M, K>, FunctionA>& input_tensor_a,
    const Tensor<DataType, Shape<K, N>, FunctionB>& input_tensor_b
) {
    using output_shape_t =  Shape<1, M, N>;
    const auto function = [input_tensor_a, input_tensor_b] {
        const auto input_a = input_tensor_a();
        const auto input_b = input_tensor_b();
        auto output = std::array<DataType, output_shape_t::volume>{};

        for (auto m = 0; m < M; m++) {
            for (auto k = 0; k < K; k++) {
                for (auto n = 0; n < N; n++) {
                    output[m * N + n] += input_a[m * K + k] * input_b[k * N + n];
                }
            }
        }
        return vector::Vector{std::move(output)};
    };
    return Tensor<DataType, output_shape_t, decltype(function)>{function};
}

template<typename DataType, auto ... Dims, typename FunctionA, typename FunctionB>
constexpr auto operator+(
    const Tensor<DataType, Shape<Dims...>, FunctionA>& input_tensor_a,
    const Tensor<DataType, Shape<Dims...>, FunctionB>& input_tensor_b
) {
    const auto function = [input_tensor_a, input_tensor_b] {
        const auto input_a = input_tensor_a();
        const auto input_b = input_tensor_b();
        return input_a + input_b;
    };
    return Tensor<DataType, Shape<Dims...>, decltype(function)>{function};
}

template<typename DataType, auto Dim_0, auto Dim_1, auto Dim_2, typename FunctionA, typename FunctionB>
constexpr auto operator+(
    const Tensor<DataType, Shape<Dim_0, Dim_1, Dim_2>, FunctionA>& input_tensor_a,
    const Tensor<DataType, Shape<Dim_2>, FunctionB>& input_tensor_b
) {
    const auto function = [input_tensor_a, input_tensor_b] {
        auto input_a = input_tensor_a();
        const auto input_b = input_tensor_b();
        auto index = 0;
        for (auto dim_0 = 0; dim_0 < Dim_0; dim_0++) {
            for (auto dim_1 = 0; dim_1 < Dim_1; dim_1++) {
                for (auto dim_2 = 0; dim_2 < Dim_2; dim_2++) {
                    input_a[index++] += input_b[dim_2];
                }
            }
        }
        return input_a;
    };
    return Tensor<DataType, Shape<Dim_0, Dim_1, Dim_2>, decltype(function)>{function};
}

template<typename DataType, auto ... Dims, typename Function>
constexpr auto exp(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
    const auto function = [input_tensor] {
        const auto input = input_tensor();
        return vector::exp(input);
    };
    return Tensor<DataType, Shape<Dims...>, decltype(function)>{function};
}

template<typename DataType, auto ... Dims, typename Function>
constexpr auto sqrt(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
    const auto function = [input_tensor] {
        const auto input = input_tensor();
        return vector::sqrt(input);
    };
    return Tensor<DataType, Shape<Dims...>, decltype(function)>{function};
}

template<typename DataType, auto ... Dims, typename Function>
constexpr auto abs(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
    const auto function = [input_tensor] {
        const auto input = input_tensor();
        return vector::abs(input);
    };
    return Tensor<DataType, Shape<Dims...>, decltype(function)>{function};
}

template<typename DataType, auto ... Dims, typename FunctionA, typename FunctionB>
bool allclose(
    const Tensor<DataType, Shape<Dims...>, FunctionA>& input_tensor_a,
    const Tensor<DataType, Shape<Dims...>, FunctionB>& input_tensor_b,
    float rtol = 1e-6,
    float atol = 1e-5
) {
    auto scalar_allclose = [atol, rtol] (const auto a, const auto b) {
        return std::abs(a - b) <= (atol + rtol * std::abs(b));
    };

    const auto input_a = input_tensor_a();
    const auto input_b = input_tensor_b();
    constexpr auto volume = Shape<Dims...>::volume;
    for (auto index = 0; index < volume; index++) {
        if (not scalar_allclose(input_a[index], input_b[index])) {
            return false;
        }
    }
    return true;
}


template<typename DataType, auto ... Dims, typename Function>
const auto evaluate(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
    const auto input = input_tensor();
    constexpr auto volume = Shape<Dims...>::volume;
    return vector::Vector<DataType, volume>(input);
}


template<typename DataType, auto ... Dims, typename Function>
void print(
    const Tensor<DataType, Shape<Dims...>, Function>& input_tensor
) {
    constexpr auto volume = Shape<Dims...>::volume;
    const auto input = input_tensor();
    std::cout << "{";
    for (auto index = 0; index < volume; index++) {
        std::cout << input[index];
        if (index != volume - 1) {
            std::cout << ",";
        }
    }
    std::cout << "}";
    std::cout << std::endl;
}

}  // namespace tensor

}  // namespace sonic