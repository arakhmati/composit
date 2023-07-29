#pragma once

#include <array>
#include <functional>
#include <iostream>
#include <random>

namespace sonic {

namespace vector {

template <typename DataType, auto Size> struct Vector {

  explicit Vector(std::array<DataType, Size> &&data) : data(data) {}

  template <typename Expression> explicit Vector(const Expression &expression) {
    for (auto index = 0; index < Size; index++) {
      this->data[index] = expression[index];
    }
  }

  auto &operator[](std::int64_t index) { return this->data[index]; }

  auto operator[](std::int64_t index) const { return this->data[index]; }
  std::size_t size() const { return this->data.size(); }

private:
  std::array<DataType, Size> data;
};

template <typename DataType, auto Size>
bool operator==(const Vector<DataType, Size> &vector_a,
                const Vector<DataType, Size> &vector_b) {
  for (auto index = 0; index < Size; index++) {
    if (vector_a[index] != vector_b[index]) {
      return false;
    }
  }
  return true;
}

template <typename T, auto N>
std::ostream &operator<<(std::ostream &os, const Vector<T, N> &vector) {
  os << "{";
  for (auto index = 0; index < vector.size(); index++) {
    auto element = vector[index];
    os << element << ", ";
  }
  os << "}";
  return os;
}

template <typename ExpressionA, typename ExpressionB> struct Add {
  explicit Add(const ExpressionA &expression_a, const ExpressionB &expression_b)
      : expression_a(expression_a), expression_b(expression_b) {}

  inline auto operator[](std::int64_t index) const {
    return this->expression_a[index] + this->expression_b[index];
  }

private:
  const ExpressionA expression_a;
  const ExpressionB expression_b;
};

template <typename ExpressionA, typename ExpressionB>
Add<ExpressionA, ExpressionB> operator+(const ExpressionA &expression_a,
                                        const ExpressionB &expression_b) {
  return Add(expression_a, expression_b);
}

template <typename Expression> struct Sqrt {
  const Expression expression;
  explicit Sqrt(const Expression &expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::sqrt(this->expression[index]);
  }
};

template <typename Expression> auto sqrt(const Expression &expression) {
  return Sqrt<Expression>(expression);
}

template <typename Expression> struct Exp {
  const Expression expression;
  explicit Exp(const Expression &expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::exp(this->expression[index]);
  }
};

template <typename Expression> auto exp(const Expression &expression) {
  return Exp<Expression>(expression);
}

template <typename Expression> struct Abs {
  const Expression expression;
  explicit Abs(const Expression &expression) : expression(expression) {}

  inline auto operator[](std::int64_t index) const {
    return std::abs(this->expression[index]);
  }
};

template <typename Expression> auto abs(const Expression &expression) {
  return Abs<Expression>(expression);
}

} // namespace vector

} // namespace sonic
