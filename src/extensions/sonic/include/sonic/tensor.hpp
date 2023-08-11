#pragma once

#include "shape.hpp"
#include "stride.hpp"

#include <array>
#include <cmath>
#include <cstdint>
#include <tuple>

namespace sonic {

namespace tensor {

template <typename T, std::size_t N>
struct __attribute__((aligned(32))) aligned_array : public std::array<T, N> {};

template <typename data_type_template,
          typename shape_template,
          typename stride_template = decltype(sonic::stride::compute_stride(shape_template{})),
          typename storage_t = aligned_array<data_type_template, shape_template::volume>>
struct tensor_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  explicit tensor_t(data_type_t* storage) : storage{storage} {}

  explicit tensor_t(const data_type_t* storage) : storage{storage} {}

  explicit tensor_t(aligned_array<data_type_t, shape_t::volume>&& storage) : storage{storage} {}

  explicit tensor_t(const aligned_array<data_type_t, shape_t::volume>& storage) : storage{storage} {}

  inline auto& operator[](std::int64_t index) { return this->storage[index]; }
  inline auto operator[](std::int64_t index) const { return this->storage[index]; }

  template <typename... Indices>
  inline auto& operator[](const std::tuple<Indices...>& indices) {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }

  inline const data_type_t* data() const {
    if constexpr (std::is_same_v<const data_type_t*, const storage_t>) {
      return storage;
    } else {
      return storage.data();
    }
  }

 private:
  storage_t storage;
};

template <typename data_type_template,
          typename shape_template,
          typename stride_template,
          typename expression_a_t,
          typename expression_b_t>
struct add_expression_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  const expression_a_t expression_a;
  const expression_b_t expression_b;

  explicit add_expression_t(const expression_a_t& expression_a, const expression_b_t& expression_b)
      : expression_a(expression_a), expression_b(expression_b) {}

  inline auto operator[](std::int64_t index) const { return this->expression_a[index] + this->expression_b[index]; }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename data_type_t,
          typename shape_t,
          typename stride_t,
          typename... rest_a_t,
          typename... rest_b_t,
          template <typename...>
          typename expression_a_t,
          template <typename...>
          typename expression_b_t>
inline auto add(const expression_a_t<data_type_t, shape_t, stride_t, rest_a_t...>& expression_a,
                const expression_b_t<data_type_t, shape_t, stride_t, rest_b_t...>& expression_b) {
  return add_expression_t<data_type_t, shape_t, stride_t, expression_a_t<data_type_t, shape_t, stride_t, rest_a_t...>,
                          expression_b_t<data_type_t, shape_t, stride_t, rest_b_t...>>(expression_a, expression_b);
}

template <typename data_type_template, typename shape_template, typename stride_template, typename expression_t>
struct sqrt_expression_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  const expression_t expression;

  explicit sqrt_expression_t(const expression_t& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const { return std::sqrt(this->expression[index]); }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename data_type_t, typename shape_t, typename stride_t, typename expression_t>
inline auto sqrt(const expression_t& expression) {
  return sqrt_expression_t<data_type_t, shape_t, stride_t, expression_t>(expression);
}

template <typename data_type_template, typename shape_template, typename stride_template, typename expression_t>
struct exp_expression_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  const expression_t expression;

  explicit exp_expression_t(const expression_t& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const { return std::exp(this->expression[index]); }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename data_type_t, typename shape_t, typename stride_t, typename expression_t>
inline auto exp(const expression_t& expression) {
  return exp_expression_t<data_type_t, shape_t, stride_t, expression_t>(expression);
}

template <typename data_type_template, typename shape_template, typename stride_template, typename expression_t>
struct abs_expression_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  const expression_t expression;

  explicit abs_expression_t(const expression_t& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const { return std::abs(this->expression[index]); }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_t{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename data_type_t, typename shape_t, typename stride_t, typename expression_t>
inline auto abs(const expression_t& expression) {
  return abs_expression_t<data_type_t, shape_t, stride_t, expression_t>(expression);
}

template <typename data_type_template, typename shape_template, typename stride_template, typename expression_t>
struct view_expression_t {
  using data_type_t = data_type_template;
  using shape_t = shape_template;
  using stride_t = stride_template;

  const expression_t expression;
  explicit view_expression_t(const expression_t& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const { return this->expression[index]; }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    using sonic::stride::compute_flat_index;
    auto flat_index = compute_flat_index(stride_template{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename data_type_t, typename shape_t, typename stride_t, typename expression_t>
inline auto view(const expression_t& expression) {
  return view_expression_t<data_type_t, shape_t, stride_t, expression_t>(expression);
}

namespace detail {
template <typename Function, auto Limit, auto... Limits, typename... Indices>
auto comparison_loop(const Function& function, const sonic::shape::shape_t<Limit, Limits...>&, Indices... indices) {
  bool matches = true;
  for (std::size_t index = 0; index < Limit; index++) {
    if constexpr (sizeof...(Limits) == 0) {
      matches &= function(std::make_tuple(indices..., index));
    } else {
      matches &= comparison_loop(function, sonic::shape::shape_t<Limits...>{}, indices..., index);
    }
    if (not matches) {
      return false;
    }
  }
  return true;
}
}  // namespace detail

template <typename data_type_t, typename shape_t, typename stride_type_t>
bool operator==(const tensor_t<data_type_t, shape_t, stride_type_t>& tensor_a,
                const tensor_t<data_type_t, shape_t, stride_type_t>& tensor_b) {
  auto function = [&tensor_a, &tensor_b](auto&& index) { return tensor_a[index] == tensor_b[index]; };
  return detail::comparison_loop(function, shape_t{});
}

template <typename data_type_t, typename shape_t, typename... rest_a_t, typename... rest_b_t>
bool allclose(const tensor_t<data_type_t, shape_t, rest_a_t...>& tensor_a,
              const tensor_t<data_type_t, shape_t, rest_b_t...>& tensor_b,
              float rtol = 1e-6,
              float atol = 1e-5) {
  if constexpr (std::is_same_v<data_type_t, float>) {
    auto scalar_allclose = [atol, rtol](const auto a, const auto b) {
      return std::abs(a - b) <= (atol + rtol * std::abs(b));
    };
    auto function = [&scalar_allclose, &tensor_a, &tensor_b](auto&& index) {
      return scalar_allclose(tensor_a[index], tensor_b[index]);
    };
    return detail::comparison_loop(function, shape_t{});
  } else if constexpr (std::is_same_v<data_type_t, std::size_t>) {
    auto function = [&tensor_a, &tensor_b](auto&& index) { return tensor_a[index] == tensor_b[index]; };
    return detail::comparison_loop(function, shape_t{});
  }
  return false;
}

namespace detail {
template <typename Function, auto Limit, auto... Limits, typename... Indices>
void void_loop(const Function& function, const sonic::shape::shape_t<Limit, Limits...>&, Indices... indices) {
  for (std::size_t index = 0; index < Limit; index++) {
    if constexpr (sizeof...(Limits) == 0) {
      function(std::make_tuple(indices..., index));
    } else {
      void_loop(function, sonic::shape::shape_t<Limits...>{}, indices..., index);
    }
  }
}
}  // namespace detail

template <typename data_type_t, typename shape_t, typename stride_t>
void print(const tensor_t<data_type_t, shape_t, stride_t>& tensor, auto& stream) {
  auto function = [&tensor, &stream](auto&& index) -> void { stream << tensor[index] << ","; };
  detail::void_loop(function, shape_t{});
}

template <typename data_type_t, typename shape_t, typename... rest_t, template <typename...> typename expression_t>
auto copy(const expression_t<data_type_t, shape_t, rest_t...>& expression, data_type_t* output_buffer) {
  std::size_t flat_index = 0;
  auto function = [&expression, &flat_index, &output_buffer](auto&& index) -> void {
    output_buffer[flat_index] = expression[index];
    flat_index++;
  };
  detail::void_loop(function, shape_t{});
}

template <typename data_type_t, typename shape_t, typename... rest_t, template <typename...> typename expression_t>
auto to_tensor(const expression_t<data_type_t, shape_t, rest_t...>& expression) {
  if constexpr (std::is_same_v<expression_t<data_type_t, shape_t, rest_t...>, tensor_t<data_type_t, shape_t>>) {
    return expression;
  } else {
    aligned_array<data_type_t, shape_t::volume> storage;
    copy(expression, storage.data());
    return tensor_t<data_type_t, shape_t>(std::move(storage));
  }
}

template <typename data_type_t, typename shape_t, typename... rest_t, template <typename...> typename expression_t>
auto copy_tensor(const expression_t<data_type_t, shape_t, rest_t...>& expression) {
  aligned_array<data_type_t, shape_t::volume> storage;
  copy(expression, storage.data());
  return tensor_t<data_type_t, shape_t>(std::move(storage));
}

}  // namespace tensor

}  // namespace sonic
