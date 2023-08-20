#pragma once

#include "shape.hpp"
#include "stride.hpp"
#include "tensor.hpp"

#include <array>
#include <random>
#include <thread>
#include <tuple>

#include <experimental/array>

namespace sonic {

namespace lazy_computation {

template <std::size_t... axes>
struct order_t {};

template <typename data_type_template, typename shape_template, typename stride_template, typename function_type_t>
struct lazy_computation_t {
  using data_type_t = const data_type_template;
  using shape_t = const shape_template;
  using stride_t = const stride_template;

  const function_type_t function;

  constexpr lazy_computation_t(const function_type_t& function) : function(function) {}
  constexpr auto operator()() const { return this->function(); }
};

template <typename data_type_t, typename shape_t>
constexpr auto as_lazy_computation(auto&& storage) {
  using stride_t = decltype(sonic::stride::compute_stride(shape_t{}));
  const auto function = [storage = std::move(storage)] {
    return tensor::tensor_t<data_type_t, shape_t, stride_t, decltype(storage)>{std::move(storage)};
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t>
constexpr auto as_lazy_computation(tensor::tensor_t<data_type_t, shape_t>&& tensor) {
  using stride_t = tensor::tensor_t<data_type_t, shape_t>::stride_t;
  const auto function = [tensor = std::move(tensor)] { return tensor; };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t>
constexpr auto arange() {
  using stride_t = decltype(sonic::stride::compute_stride(shape_t{}));
  constexpr auto function = [] {
    auto storage = sonic::tensor::aligned_array<data_type_t, shape_t::volume>{};
    auto index = 0.0f;
    for (auto& value : storage) {
      value = index++;
    }
    return tensor::tensor_t<data_type_t, shape_t, stride_t>{std::move(storage)};
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t>
constexpr auto random() {
  using stride_t = decltype(sonic::stride::compute_stride(shape_t{}));
  constexpr auto function = [] {
    std::mt19937 generator(0);
    std::uniform_real_distribution<> distribution(-1.0f, 1.0f);
    auto storage = sonic::tensor::aligned_array<data_type_t, shape_t::volume>{};
    for (auto& value : storage) {
      value = distribution(generator);
    }
    return tensor::tensor_t<data_type_t, shape_t, stride_t>{std::move(storage)};
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename output_shape_t,
          typename data_type_t,
          typename input_shape_t,
          typename input_stride_t,
          typename function_type_t>
constexpr auto reshape(
    const lazy_computation_t<data_type_t, const input_shape_t, const input_stride_t, function_type_t>&
        input_computation) {
  using output_stride_t = decltype(sonic::stride::compute_stride(output_shape_t{}));

  static_assert(input_shape_t::volume == output_shape_t::volume);

  const auto function = [input_computation] {
    const auto input_tensor = input_computation();
    if constexpr (std::is_same_v<input_stride_t, decltype(sonic::stride::compute_stride(input_shape_t{}))>) {
      return tensor::view<data_type_t, output_shape_t, output_stride_t>(input_tensor);
    } else {
      auto reshaped_input_tensor = tensor::as_tensor(input_tensor);
      return tensor::view<data_type_t, output_shape_t, output_stride_t>(reshaped_input_tensor);
    }
  };
  return lazy_computation_t<data_type_t, const output_shape_t, const output_stride_t, decltype(function)>{function};
}

namespace detail {
template <auto... dimensions, template <std::size_t...> typename order_t, std::size_t axis>
constexpr auto reorder_shape(const order_t<axis>&) {
  constexpr std::size_t dimension = std::get<axis>(std::make_tuple(dimensions...));
  return sonic::shape::shape_t<dimension>{};
}

template <auto... dimensions, template <std::size_t...> typename order_t, std::size_t axis, std::size_t... axes>
constexpr auto reorder_shape(const order_t<axis, axes...>&) {
  constexpr std::size_t dimension = std::get<axis>(std::make_tuple(dimensions...));
  auto outer_shape = sonic::shape::shape_t<dimension>{};
  auto inner_shape = reorder_shape<dimensions...>(order_t<axes...>{});
  return sonic::metaprogramming::merge(outer_shape, inner_shape);
}

template <auto... strides, template <std::size_t...> typename order_t, std::size_t axis>
constexpr auto reorder_stride(const order_t<axis>&) {
  constexpr std::size_t stride = std::get<axis>(std::make_tuple(strides...));
  return sonic::stride::stride_t<stride>{};
}

template <auto... strides, template <std::size_t...> typename order_t, std::size_t axis, std::size_t... axes>
constexpr auto reorder_stride(const order_t<axis, axes...>&) {
  constexpr std::size_t stride = std::get<axis>(std::make_tuple(strides...));
  auto outer_stride = sonic::stride::stride_t<stride>{};
  auto inner_stride = reorder_stride<strides...>(order_t<axes...>{});
  return sonic::metaprogramming::merge(outer_stride, inner_stride);
}
}  // namespace detail

template <typename order_t,
          typename data_type_t,
          std::size_t... dimensions,
          std::size_t... strides,
          typename function_type_t>
constexpr auto transpose(const lazy_computation_t<data_type_t,
                                                  const sonic::shape::shape_t<dimensions...>,
                                                  const sonic::stride::stride_t<strides...>,
                                                  function_type_t>& input_computation) {
  using output_shape_t = decltype(detail::reorder_shape<dimensions...>(order_t{}));
  using output_stride_t = decltype(detail::reorder_stride<strides...>(order_t{}));
  const auto function = [input_computation] {
    const auto input = input_computation();
    return tensor::view<data_type_t, output_shape_t, output_stride_t>(input);
  };
  return lazy_computation_t<data_type_t, const output_shape_t, const output_stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto batch_size,
          auto m_size,
          auto k_size,
          auto n_size,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto matmul(
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<batch_size, m_size, k_size>, const rest_a_t...>&
        input_computation_a,
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<k_size, n_size>, const rest_b_t...>&
        input_computation_b,
    auto output_storage) {
  using output_shape_t = sonic::shape::shape_t<batch_size, m_size, n_size>;
  using output_stride_t = decltype(sonic::stride::compute_stride(output_shape_t{}));
  const auto function = [input_computation_a, input_computation_b, output_storage] {
    const auto input_a = input_computation_a();
    const auto input_b = input_computation_b();
    auto output =
        tensor::tensor_t<data_type_t, output_shape_t, output_stride_t, decltype(output_storage)>{output_storage};

    auto run_thread = [&input_a, &input_b, &output](const auto m_start, const auto m_end, const auto n_start,
                                                    const auto n_end) {
      constexpr std::size_t m_tile_size = 32;
      constexpr std::size_t k_tile_size = 32;
      constexpr std::size_t n_tile_size = 32;
      if constexpr (m_size % m_tile_size == 0 and k_size % k_tile_size == 0 and n_size % n_tile_size == 0) {
        for (std::size_t batch_index = 0; batch_index < batch_size; batch_index++) {
          for (std::size_t m_tile = m_start; m_tile < m_end; m_tile += m_tile_size) {
            for (std::size_t k_tile = 0; k_tile < k_size; k_tile += k_tile_size) {
              for (std::size_t n_tile = n_start; n_tile < n_end; n_tile += n_tile_size) {
                for (std::size_t m = m_tile; m < m_tile + m_tile_size; m++) {
                  for (std::size_t k = k_tile; k < k_tile + k_tile_size; k++) {
                    auto input_a_vector = _mm256_set1_ps(input_a[std::make_tuple(batch_index, m, k)]);
                    for (std::size_t n = n_tile; n < n_tile + n_tile_size; n += sonic::tensor::vector8_float32::size) {
                      auto input_b_vector =
                          input_b.template load<sonic::tensor::vector8_float32>(std::make_tuple(k, n));
                      auto output_vector =
                          output.template load<sonic::tensor::vector8_float32>(std::make_tuple(batch_index, m, n));
                      output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
                      output.template store<sonic::tensor::vector8_float32>(std::make_tuple(batch_index, m, n),
                                                                            output_vector);
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (std::size_t batch_index = 0; batch_index < batch_size; batch_index++) {
          for (std::size_t m = m_start; m < m_end; m++) {
            for (std::size_t k = 0; k < k_size; k++) {
              if constexpr (n_size % sonic::tensor::vector8_float32::size == 0) {
                auto input_a_vector = _mm256_set1_ps(input_a[std::make_tuple(batch_index, m, k)]);
                for (std::size_t n = n_start; n < n_end; n += sonic::tensor::vector8_float32::size) {
                  auto input_b_vector = input_b.template load<sonic::tensor::vector8_float32>(std::make_tuple(k, n));
                  auto output_vector =
                      output.template load<sonic::tensor::vector8_float32>(std::make_tuple(batch_index, m, n));
                  output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
                  output.template store<sonic::tensor::vector8_float32>(std::make_tuple(batch_index, m, n),
                                                                        output_vector);
                }
              } else {
                for (std::size_t n = n_start; n < n_end; n++) {
                  output[std::make_tuple(batch_index, m, n)] +=
                      input_a[std::make_tuple(batch_index, m, k)] * input_b[std::make_tuple(k, n)];
                }
              }
            }
          }
        }
      }
    };

    /*
    auto threads = std::experimental::make_array(
        std::thread(run_thread, m_size / 2 * 0, m_size / 2 * 1, n_size / 4 * 0, n_size / 4 * 1),
        std::thread(run_thread, m_size / 2 * 1, m_size / 2 * 2, n_size / 4 * 0, n_size / 4 * 1),
        std::thread(run_thread, m_size / 2 * 0, m_size / 2 * 1, n_size / 4 * 1, n_size / 4 * 2),
        std::thread(run_thread, m_size / 2 * 1, m_size / 2 * 2, n_size / 4 * 1, n_size / 4 * 2),
        std::thread(run_thread, m_size / 2 * 0, m_size / 2 * 1, n_size / 4 * 2, n_size / 4 * 3),
        std::thread(run_thread, m_size / 2 * 1, m_size / 2 * 2, n_size / 4 * 2, n_size / 4 * 3),
        std::thread(run_thread, m_size / 2 * 0, m_size / 2 * 1, n_size / 4 * 3, n_size / 4 * 4),
        std::thread(run_thread, m_size / 2 * 1, m_size / 2 * 2, n_size / 4 * 3, n_size / 4 * 4)
    );

    for (auto& thread : threads) {
      thread.join();
    }*/

    run_thread(0, m_size, 0, n_size);
    return output;
  };
  return lazy_computation_t<data_type_t, const output_shape_t, const output_stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto batch_size,
          auto m_size,
          auto k_size,
          auto n_size,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto matmul(
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<batch_size, m_size, k_size>, const rest_a_t...>&
        input_computation_a,
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<k_size, n_size>, const rest_b_t...>&
        input_computation_b) {
  using output_shape_t = sonic::shape::shape_t<batch_size, m_size, n_size>;
  return matmul(input_computation_a, input_computation_b,
                sonic::tensor::aligned_array<data_type_t, output_shape_t::volume>{});
}

namespace detail {
constexpr auto _mm256_reduce_add_ps(const auto& x) {
  /* ( x3+x7, x2+x6, x1+x5, x0+x4 ) */
  const __m128 x128 = _mm_add_ps(_mm256_extractf128_ps(x, 1), _mm256_castps256_ps128(x));
  /* ( -, -, x1+x3+x5+x7, x0+x2+x4+x6 ) */
  const __m128 x64 = _mm_add_ps(x128, _mm_movehl_ps(x128, x128));
  /* ( -, -, -, x0+x1+x2+x3+x4+x5+x6+x7 ) */
  const __m128 x32 = _mm_add_ss(x64, _mm_shuffle_ps(x64, x64, 0x55));
  /* Conversion to float is a no-op on x86-64 */
  return _mm_cvtss_f32(x32);
}
}  // namespace detail

template <typename data_type_t,
          auto batch_size,
          auto m_size,
          auto k_size,
          auto n_size,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto matmul_with_transposed_input_b(
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<batch_size, m_size, k_size>, const rest_a_t...>&
        input_computation_a,
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<n_size, k_size>, const rest_b_t...>&
        input_computation_b,
    auto output_storage) {
  using output_shape_t = sonic::shape::shape_t<batch_size, m_size, n_size>;
  using output_stride_t = decltype(sonic::stride::compute_stride(output_shape_t{}));
  const auto function = [input_computation_a, input_computation_b, output_storage] {
    const auto input_a = input_computation_a();
    const auto input_b = input_computation_b();
    auto output =
        tensor::tensor_t<data_type_t, output_shape_t, output_stride_t, decltype(output_storage)>{output_storage};

    for (std::size_t batch_index = 0; batch_index < batch_size; batch_index++) {
      for (std::size_t m = 0; m < m_size; m++) {
        for (std::size_t n = 0; n < n_size; n++) {
          if constexpr (k_size % sonic::tensor::vector8_float32::size == 0) {
            __m256 output_vector = _mm256_setzero_ps();
            for (std::size_t k = 0; k < k_size; k += sonic::tensor::vector8_float32::size) {
              auto input_a_vector =
                  input_a.template load<sonic::tensor::vector8_float32>(std::make_tuple(batch_index, m, k));
              auto input_b_vector = input_b.template load<sonic::tensor::vector8_float32>(std::make_tuple(n, k));
              output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
            }
            output[std::make_tuple(batch_index, m, n)] += detail::_mm256_reduce_add_ps(output_vector);
          } else {
            for (std::size_t k = 0; k < k_size; k++) {
              output[std::make_tuple(batch_index, m, n)] +=
                  input_a[std::make_tuple(batch_index, m, k)] * input_b[std::make_tuple(n, k)];
            }
          }
        }
      }
    }
    return output;
  };
  return lazy_computation_t<data_type_t, const output_shape_t, const output_stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto batch_size_0,
          auto batch_size_1,
          auto m_size,
          auto k_size,
          auto n_size,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto matmul(const lazy_computation_t<data_type_t,
                                               const sonic::shape::shape_t<batch_size_0, batch_size_1, m_size, k_size>,
                                               const rest_a_t...>& input_computation_a,
                      const lazy_computation_t<data_type_t,
                                               const sonic::shape::shape_t<batch_size_0, batch_size_1, k_size, n_size>,
                                               const rest_b_t...>& input_computation_b,
                      auto&& output_storage) {
  using output_shape_t = sonic::shape::shape_t<batch_size_0, batch_size_1, m_size, n_size>;
  using output_stride_t = const decltype(sonic::stride::compute_stride(output_shape_t{}));
  const auto function = [input_computation_a, input_computation_b, output_storage = std::move(output_storage)] {
    const auto input_a = input_computation_a();
    const auto input_b = tensor::as_tensor(input_computation_b());
    auto output = tensor::tensor_t<data_type_t, output_shape_t, output_stride_t>{std::move(output_storage)};

    auto run_thread = [&input_a, &input_b, &output](const std::size_t batch_1_start, const std::size_t batch_1_end,
                                                    const std::size_t m_start, const std::size_t m_end,
                                                    const std::size_t n_start, const std::size_t n_end) {
      constexpr std::size_t m_tile_size = 32;
      constexpr std::size_t k_tile_size = 32;
      constexpr std::size_t n_tile_size = 32;
      if constexpr (m_size % m_tile_size == 0 and k_size % k_tile_size == 0 and n_size % n_tile_size == 0) {
        for (std::size_t batch_index_0 = 0; batch_index_0 < batch_size_0; batch_index_0++) {
          for (std::size_t batch_index_1 = batch_1_start; batch_index_1 < batch_1_end; batch_index_1++) {
            for (std::size_t m_tile = m_start; m_tile < m_end; m_tile += m_tile_size) {
              for (std::size_t k_tile = 0; k_tile < k_size; k_tile += k_tile_size) {
                for (std::size_t n_tile = n_start; n_tile < n_end; n_tile += n_tile_size) {
                  for (std::size_t m = m_tile; m < m_tile + m_tile_size; m++) {
                    for (std::size_t k = k_tile; k < k_tile + k_tile_size; k++) {
                      auto input_a_vector =
                          _mm256_set1_ps(input_a[std::make_tuple(batch_index_0, batch_index_1, m, k)]);
                      for (std::size_t n = n_tile; n < n_tile + n_tile_size;
                           n += sonic::tensor::vector8_float32::size) {
                        auto input_b_vector = input_b.template load<sonic::tensor::vector8_float32>(
                            std::make_tuple(batch_index_0, batch_index_1, k, n));
                        auto output_vector = output.template load<sonic::tensor::vector8_float32>(
                            std::make_tuple(batch_index_0, batch_index_1, m, n));
                        output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
                        output.template store<sonic::tensor::vector8_float32>(
                            std::make_tuple(batch_index_0, batch_index_1, m, n), output_vector);
                      }
                    }
                  }
                }
              }
            }
          }
        }
      } else {
        for (std::size_t batch_index_0 = 0; batch_index_0 < batch_size_0; batch_index_0++) {
          for (std::size_t batch_index_1 = batch_1_start; batch_index_1 < batch_1_end; batch_index_1++) {
            for (std::size_t m = m_start; m < m_end; m++) {
              for (std::size_t k = 0; k < k_size; k++) {
                if constexpr (n_size % sonic::tensor::vector8_float32::size == 0) {
                  auto input_a_vector = _mm256_set1_ps(input_a[std::make_tuple(batch_index_0, batch_index_1, m, k)]);
                  for (std::size_t n = n_start; n < n_end; n += sonic::tensor::vector8_float32::size) {
                    auto input_b_vector = input_b.template load<sonic::tensor::vector8_float32>(
                        std::make_tuple(batch_index_0, batch_index_1, k, n));
                    auto output_vector = output.template load<sonic::tensor::vector8_float32>(
                        std::make_tuple(batch_index_0, batch_index_1, m, n));
                    output_vector = _mm256_fmadd_ps(input_a_vector, input_b_vector, output_vector);
                    output.template store<sonic::tensor::vector8_float32>(
                        std::make_tuple(batch_index_0, batch_index_1, m, n), output_vector);
                  }
                } else {
                  for (std::size_t n = n_start; n < n_end; n++) {
                    output[std::make_tuple(batch_index_0, batch_index_1, m, n)] +=
                        input_a[std::make_tuple(batch_index_0, batch_index_1, m, k)] *
                        input_b[std::make_tuple(batch_index_0, batch_index_1, k, n)];
                  }
                }
              }
            }
          }
        }
      }
    };

    /*
    auto threads = std::experimental::make_array(
        std::thread(run_thread, 0, 1, 0, m_size, 0, n_size),
        std::thread(run_thread, 1, 2, 0, m_size, 0, n_size),
        std::thread(run_thread, 2, 3, 0, m_size, 0, n_size),
        std::thread(run_thread, 3, 4, 0, m_size, 0, n_size),
        std::thread(run_thread, 4, 5, 0, m_size, 0, n_size),
        std::thread(run_thread, 5, 6, 0, m_size, 0, n_size),
        std::thread(run_thread, 6, 7, 0, m_size, 0, n_size),
        std::thread(run_thread, 7, 8, 0, m_size, 0, n_size),
        std::thread(run_thread, 8, 9, 0, m_size, 0, n_size),
        std::thread(run_thread, 9, 10, 0, m_size, 0, n_size),
        std::thread(run_thread, 10, 11, 0, m_size, 0, n_size),
        std::thread(run_thread, 11, 12, 0, m_size, 0, n_size)
    );

    for (auto& thread : threads) {
      thread.join();
    }
    */

    run_thread(0, batch_size_1, 0, m_size, 0, n_size);
    return output;
  };
  return lazy_computation_t<data_type_t, const output_shape_t, const output_stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto batch_size_0,
          auto batch_size_1,
          auto m_size,
          auto k_size,
          auto n_size,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto matmul(const lazy_computation_t<data_type_t,
                                               const sonic::shape::shape_t<batch_size_0, batch_size_1, m_size, k_size>,
                                               const rest_a_t...>& input_computation_a,
                      const lazy_computation_t<data_type_t,
                                               const sonic::shape::shape_t<batch_size_0, batch_size_1, k_size, n_size>,
                                               const rest_b_t...>& input_computation_b) {
  using output_shape_t = sonic::shape::shape_t<batch_size_0, batch_size_1, m_size, n_size>;
  return matmul(input_computation_a, input_computation_b,
                sonic::tensor::aligned_array<data_type_t, output_shape_t::volume>{});
}

template <typename data_type_t, typename shape_t, typename stride_t, typename... rest_a_t, typename... rest_b_t>
constexpr auto operator+(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const rest_a_t...>& input_computation_a,
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const rest_b_t...>& input_computation_b) {
  const auto function = [input_computation_a, input_computation_b] {
    const auto input_a = input_computation_a();
    const auto input_b = input_computation_b();
    return add<data_type_t, shape_t, stride_t>(input_a, input_b);
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto Dimension_0,
          auto Dimension_1,
          auto Dimension_2,
          typename stride_t,
          typename... rest_a_t,
          typename... rest_b_t>
constexpr auto add_in_place(
    const lazy_computation_t<data_type_t,
                             const sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2>,
                             const stride_t,
                             const rest_a_t...>& input_computation_a,
    const lazy_computation_t<data_type_t, const sonic::shape::shape_t<Dimension_2>, const rest_b_t...>&
        input_computation_b) {
  using shape_t = sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2>;
  const auto function = [input_computation_a, input_computation_b] {
    auto input_a = input_computation_a();
    const auto input_b = input_computation_b();
    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          input_a[std::make_tuple(dim_0, dim_1, dim_2)] += input_b[std::make_tuple(dim_2)];
        }
      }
    }
    return input_a;
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto Dimension_0,
          auto Dimension_1,
          auto Dimension_2,
          auto Dimension_3,
          typename stride_t,
          typename... rest_a_t>
constexpr auto divide_in_place(
    const lazy_computation_t<data_type_t,
                             const sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2, Dimension_3>,
                             const stride_t,
                             const rest_a_t...>& input_computation_a,
    const data_type_t input_b) {
  using shape_t = sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2, Dimension_3>;
  const auto function = [input_computation_a, input_b] {
    auto input_a = input_computation_a();
    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          for (std::size_t dim_3 = 0; dim_3 < Dimension_3; dim_3++) {
            input_a[std::make_tuple(dim_0, dim_1, dim_2, dim_3)] /= input_b;
          }
        }
      }
    }
    return input_a;
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t, typename stride_t, typename... rest_t>
constexpr auto exp(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const rest_t...>& input_computation) {
  const auto function = [input_computation] {
    const auto input = input_computation();
    return tensor::exp<data_type_t, shape_t, stride_t>(input);
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t, typename stride_t, typename... rest_t>
constexpr auto sqrt(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const rest_t...>& input_computation) {
  const auto function = [input_computation] {
    const auto input = input_computation();
    return tensor::sqrt<data_type_t, shape_t, stride_t>(input);
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t, typename shape_t, typename stride_t, typename... rest_t>
constexpr auto abs(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const rest_t...>& input_computation) {
  const auto function = [input_computation] {
    const auto input = input_computation();
    return tensor::abs<data_type_t, shape_t, stride_t>(input);
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

template <typename data_type_t,
          auto Dimension_0,
          auto Dimension_1,
          auto Dimension_2,
          auto Dimension_3,
          typename stride_t,
          typename... rest_t>
constexpr auto softmax(
    const lazy_computation_t<data_type_t,
                             const sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2, Dimension_3>,
                             const stride_t,
                             const rest_t...>& input_computation) {
  using shape_t = sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2, Dimension_3>;
  using reduced_shape_t = sonic::shape::shape_t<Dimension_0, Dimension_1, Dimension_2>;
  const auto function = [input_computation] {
    auto input = input_computation();

    auto temporary = tensor::tensor_t<data_type_t, reduced_shape_t>{
        sonic::tensor::aligned_array<data_type_t, reduced_shape_t::volume>{}};
    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          for (std::size_t dim_3 = 0; dim_3 < Dimension_3; dim_3++) {
            auto indices = std::make_tuple(dim_0, dim_1, dim_2, dim_3);
            auto reduced_indices = std::make_tuple(dim_0, dim_1, dim_2);
            temporary[reduced_indices] = std::max(temporary[reduced_indices], input[indices]);
          }
        }
      }
    }

    auto output = tensor::copy(input);
    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          for (std::size_t dim_3 = 0; dim_3 < Dimension_3; dim_3++) {
            auto indices = std::make_tuple(dim_0, dim_1, dim_2, dim_3);
            auto reduced_indices = std::make_tuple(dim_0, dim_1, dim_2);
            output[indices] -= temporary[reduced_indices];
            output[indices] = std::exp(output[indices]);
          }
        }
      }
    }

    temporary = tensor::tensor_t<data_type_t, reduced_shape_t>{
        sonic::tensor::aligned_array<data_type_t, reduced_shape_t::volume>{}};
    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          for (std::size_t dim_3 = 0; dim_3 < Dimension_3; dim_3++) {
            auto indices = std::make_tuple(dim_0, dim_1, dim_2, dim_3);
            auto reduced_indices = std::make_tuple(dim_0, dim_1, dim_2);
            temporary[reduced_indices] += output[indices];
          }
        }
      }
    }

    for (std::size_t dim_0 = 0; dim_0 < Dimension_0; dim_0++) {
      for (std::size_t dim_1 = 0; dim_1 < Dimension_1; dim_1++) {
        for (std::size_t dim_2 = 0; dim_2 < Dimension_2; dim_2++) {
          for (std::size_t dim_3 = 0; dim_3 < Dimension_3; dim_3++) {
            auto indices = std::make_tuple(dim_0, dim_1, dim_2, dim_3);
            auto reduced_indices = std::make_tuple(dim_0, dim_1, dim_2);
            output[indices] /= temporary[reduced_indices];
          }
        }
      }
    }
    return output;
  };
  return lazy_computation_t<data_type_t, const shape_t, const stride_t, decltype(function)>{function};
}

auto linear = [](const auto& activations, const auto& weights, const auto& bias) {
  return add_in_place(matmul(activations, weights), bias);
};

template <typename compute_data_type_t = float,
          typename data_type_t,
          typename shape_t,
          typename stride_t,
          typename function_type_t>
const auto evaluate(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const function_type_t>& input_computation) {
  const auto input = input_computation();
  return tensor::as_tensor<compute_data_type_t>(input);
}

template <typename compute_data_type_t = float,
          typename data_type_t,
          typename shape_t,
          typename stride_t,
          typename function_type_t>
void evaluate_to(
    const lazy_computation_t<data_type_t, const shape_t, const stride_t, const function_type_t>& input_computation,
    data_type_t* output_buffer) {
  const auto input = input_computation();
  tensor::write<compute_data_type_t>(input, output_buffer);
}

}  // namespace lazy_computation

}  // namespace sonic