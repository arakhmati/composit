#pragma once

#include <tuple>

#include "metaprogramming.hpp"
#include "shape.hpp"

namespace sonic {

namespace stride {

template <std::size_t... Dimensions>
struct stride_t {};

template <std::size_t Dimension>
constexpr const stride_t<1> compute_stride(const sonic::shape::shape_t<Dimension>&) {
  return {};
}

template <std::size_t Dimension, std::size_t... Dimensions>
constexpr auto compute_stride(const sonic::shape::shape_t<Dimension, Dimensions...>&) {
  auto outer_stride = stride_t<sonic::shape::shape_t<Dimensions...>::volume>{};
  auto inner_stride = compute_stride(sonic::shape::shape_t<Dimensions...>{});
  return sonic::metaprogramming::merge(outer_stride, inner_stride);
}

constexpr auto compute_flat_index(const stride_t<>&, const std::tuple<>&) {
  return 0;
}

template <std::size_t Dimension, std::size_t... Dimensions, typename Index, typename... Indices>
constexpr auto compute_flat_index(const stride_t<Dimension, Dimensions...>&,
                                  const std::tuple<Index, Indices...>& indices) {
  static_assert(sizeof...(Dimensions) == sizeof...(Indices));
  auto flat_index = std::get<0>(indices) * Dimension +
                    compute_flat_index(stride_t<Dimensions...>{}, sonic::metaprogramming::rest(indices));
  return flat_index;
}

}  // namespace stride

}  // namespace sonic