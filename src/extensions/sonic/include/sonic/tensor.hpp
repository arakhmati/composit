#pragma once

#include <array>

#include "shape.hpp"

namespace sonic {

namespace tensor {

using sonic::shape::Shape;

namespace detail {

template <typename Head, typename... Tail>
std::tuple<Tail...> tuple_tail(const std::tuple<Head, Tail...>& t) {
  return std::apply(
      [](auto head, auto... tail) { return std::make_tuple(tail...); }, t);
}

template <template <auto...> typename Shape>
constexpr auto compute_flat_index(const Shape<>&, const std::tuple<>& indices) {
  return 0;
}

template <template <auto...> typename Shape, auto Dim, auto... Dims,
          typename Index, typename... Indices>
constexpr auto compute_flat_index(
    const Shape<Dim, Dims...>&, const std::tuple<Index, Indices...>& indices) {
  static_assert(sizeof...(Dims) == sizeof...(Indices));
  auto flat_index = std::get<0>(indices) * Shape<Dims...>::volume +
                    compute_flat_index(Shape<Dims...>{}, tuple_tail(indices));
  return flat_index;
}

}  // namespace detail

template <typename DataType, typename Shape>
struct Tensor {
  using data_type_t = DataType;
  using shape_t = Shape;

  explicit Tensor(std::array<DataType, Shape::volume>&& data) : data(data) {}

  template <typename Expression>
  explicit Tensor(const Expression& expression) {
    for (auto index = 0; index < Shape::volume; index++) {
      this->data[index] = expression[index];
    }
  }

  inline auto& operator[](std::int64_t index) { return this->data[index]; }
  inline auto operator[](std::int64_t index) const { return this->data[index]; }

  template <typename... Indices>
  inline auto& operator[](const std::tuple<Indices...>& indices) {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }

  std::size_t size() const { return this->data.size(); }

 private:
  std::array<DataType, Shape::volume> data;
};

template <typename DataType, typename Shape>
bool operator==(const Tensor<DataType, Shape>& tensor_a,
                const Tensor<DataType, Shape>& tensor_b) {
  for (auto index = 0; index < Shape::volume; index++) {
    if (tensor_a[index] != tensor_b[index]) {
      return false;
    }
  }
  return true;
}

template <typename ExpressionA, typename ExpressionB>
struct Add {
  explicit Add(const ExpressionA& expression_a, const ExpressionB& expression_b)
      : expression_a(expression_a), expression_b(expression_b) {}

  inline auto operator[](std::int64_t index) const {
    return this->expression_a[index] + this->expression_b[index];
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }

 private:
  const ExpressionA expression_a;
  const ExpressionB expression_b;
};

template <typename ExpressionA, typename ExpressionB>
Add<ExpressionA, ExpressionB> operator+(const ExpressionA& expression_a,
                                        const ExpressionB& expression_b) {
  return Add(expression_a, expression_b);
}

template <typename Shape, typename Expression>
struct Sqrt {
  const Expression expression;
  explicit Sqrt(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::sqrt(this->expression[index]);
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename Shape, typename Expression>
auto sqrt(const Expression& expression) {
  return Sqrt<Shape, Expression>(expression);
}

template <typename Shape, typename Expression>
struct Exp {
  const Expression expression;
  explicit Exp(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::exp(this->expression[index]);
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename Shape, typename Expression>
auto exp(const Expression& expression) {
  return Exp<Shape, Expression>(expression);
}

template <typename Shape, typename Expression>
struct Abs {
  const Expression expression;
  explicit Abs(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::abs(this->expression[index]);
  }

  template <typename... Indices>
  inline auto operator[](const std::tuple<Indices...>& indices) const {
    auto flat_index = detail::compute_flat_index(Shape{}, indices);
    return this->operator[](flat_index);
  }
};

template <typename Shape, typename Expression>
auto abs(const Expression& expression) {
  return Abs<Shape, Expression>(expression);
}

}  // namespace tensor

}  // namespace sonic
