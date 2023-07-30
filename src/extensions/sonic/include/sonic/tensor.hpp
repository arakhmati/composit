#pragma once

#include <array>

#include "shape.hpp"

namespace sonic {

namespace tensor {

using sonic::shape::Shape;

template <typename DataType, auto Size>
struct Tensor {
  using data_type_t = DataType;
//  using shape_t = Shape;

  explicit Tensor(std::array<DataType, Size>&& data) : data(data) {}

  template <typename Expression>
  explicit Tensor(const Expression& expression) {
    for (auto index = 0; index < Size; index++) {
      this->data[index] = expression[index];
    }
  }

  auto& operator[](std::int64_t index) { return this->data[index]; }

  auto operator[](std::int64_t index) const { return this->data[index]; }
  std::size_t size() const { return this->data.size(); }

 private:
  std::array<DataType, Size> data;
};

template <typename DataType, auto Size>
bool operator==(const Tensor<DataType, Size>& tensor_a,
                const Tensor<DataType, Size>& tensor_b) {
  for (auto index = 0; index < Size; index++) {
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

 private:
  const ExpressionA expression_a;
  const ExpressionB expression_b;
};

template <typename ExpressionA, typename ExpressionB>
Add<ExpressionA, ExpressionB> operator+(const ExpressionA& expression_a,
                                        const ExpressionB& expression_b) {
  return Add(expression_a, expression_b);
}

template <typename Expression>
struct Sqrt {
  const Expression expression;
  explicit Sqrt(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::sqrt(this->expression[index]);
  }
};

template <typename Expression>
auto sqrt(const Expression& expression) {
  return Sqrt<Expression>(expression);
}

template <typename Expression>
struct Exp {
  const Expression expression;
  explicit Exp(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::exp(this->expression[index]);
  }
};

template <typename Expression>
auto exp(const Expression& expression) {
  return Exp<Expression>(expression);
}

template <typename Expression>
struct Abs {
  const Expression expression;
  explicit Abs(const Expression& expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::abs(this->expression[index]);
  }
};

template <typename Expression>
auto abs(const Expression& expression) {
  return Abs<Expression>(expression);
}

}  // namespace tensor

}  // namespace sonic
